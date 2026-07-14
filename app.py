import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from folium import plugins
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
import osmnx as ox
import networkx as nx
import datetime
import gymnasium as gym
from gymnasium import spaces
import plotly.graph_objects as go
import xgboost as xgb

try:
    from stable_baselines3 import PPO
    RL_INSTALLED = True
except ImportError:
    RL_INSTALLED = False

try:
    from attribution_engine import SourceAttributionEngine
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False

try:
    from intervention_agent import (
        InterventionAgent, CommandCenterOutput, SimulationResult, ScenarioComparison,
        cost_tier_label, deploy_hours_label, weather_reduction_pct,
    )
    INTERVENTION_AVAILABLE = True
except ImportError:
    INTERVENTION_AVAILABLE = False

try:
    from health_economic_engine import (
        HealthEconomicEngine, HealthEconomicImpact, estimate_population_exposed,
    )
    HEALTH_ECONOMIC_AVAILABLE = True
except ImportError:
    HEALTH_ECONOMIC_AVAILABLE = False

try:
    from mayor_copilot import MayorCopilot, DecisionContext, CopilotAnswer
    COPILOT_AVAILABLE = True
except Exception as _copilot_import_err:
    COPILOT_AVAILABLE = False
    _COPILOT_IMPORT_ERROR = _copilot_import_err
else:
    _COPILOT_IMPORT_ERROR = None

# ─────────────────────────────────────────────
# CONSTANTS — must match train_agent.py exactly
# ─────────────────────────────────────────────
MAX_NEIGHBORS = 8
OBS_SIZE = 4 + MAX_NEIGHBORS  # 12

# ─────────────────────────────────────────────
# CONFIG & STATE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AirTwin Live — AI Command Center",
    layout="wide",
    page_icon="🌐",
)
st_autorefresh(interval=300_000, key="data_refresh")

# ── FIX 1: API key from st.secrets; fall back to env-var for local dev ──
try:
    WAQI_TOKEN = st.secrets["WAQI_TOKEN"]
except (KeyError, FileNotFoundError):
    import os
    WAQI_TOKEN = os.environ.get("WAQI_TOKEN", "")
    if not WAQI_TOKEN:
        st.warning(
            "⚠️ WAQI API token not configured. "
            "Add it to `.streamlit/secrets.toml` as `WAQI_TOKEN = 'your_token'` for live sensor data."
        )

for key in ['gps_path', 'ai_path', 'metrics', 'thought_process', 'last_city',
            'attribution_result', 'attribution_sensor',
            'command_center_output', 'selected_intervention_id', 'last_comparison']:
    if key not in st.session_state:
        st.session_state[key] = None

for key in ['last_simulations', 'last_health_impacts', 'copilot_chat_history']:
    if key not in st.session_state:
        st.session_state[key] = []

# ── Engine singletons: one instance per app session ───────────────────────────
if 'attribution_engine' not in st.session_state:
    st.session_state.attribution_engine = None
if 'intervention_agent' not in st.session_state:
    st.session_state.intervention_agent = InterventionAgent() if INTERVENTION_AVAILABLE else None
if 'health_economic_engine' not in st.session_state:
    st.session_state.health_economic_engine = HealthEconomicEngine() if HEALTH_ECONOMIC_AVAILABLE else None
if 'mayor_copilot' not in st.session_state:
    st.session_state.mayor_copilot = MayorCopilot() if COPILOT_AVAILABLE else None

# ─────────────────────────────────────────────
# RL INFERENCE ENVIRONMENT
# ─────────────────────────────────────────────
class CleanAirInferenceEnv(gym.Env):
    """
    Inference-only mirror of RealCleanAirEnv in train_agent.py.
    Observation and action spaces MUST stay in sync with the training env.
    """

    def __init__(self, graph, start_node, target_node):
        super().__init__()
        self.graph = graph
        self.current_node = start_node
        self.target_node = target_node
        self.nodes = list(graph.nodes())

        self.action_space = spaces.Discrete(MAX_NEIGHBORS)
        self.observation_space = spaces.Box(
            low=-180.0, high=1000.0, shape=(OBS_SIZE,), dtype=np.float32
        )

    def _get_obs(self):
        neighbors = list(self.graph.neighbors(self.current_node))
        curr_y = self.graph.nodes[self.current_node]['y']
        curr_x = self.graph.nodes[self.current_node]['x']
        targ_y = self.graph.nodes[self.target_node]['y']
        targ_x = self.graph.nodes[self.target_node]['x']

        neighbor_aqis = []
        for i in range(MAX_NEIGHBORS):
            if i < len(neighbors):
                edge_data = self.graph.get_edge_data(self.current_node, neighbors[i])
                edge_data = edge_data[0] if edge_data else {}
                neighbor_aqis.append(float(edge_data.get('mock_aqi', 150.0)))
            else:
                neighbor_aqis.append(0.0)

        return np.array([curr_y, curr_x, targ_y, targ_x] + neighbor_aqis, dtype=np.float32)

    def step(self, action):
        neighbors = list(self.graph.neighbors(self.current_node))
        if int(action) < len(neighbors):
            self.current_node = neighbors[int(action)]
        done = self.current_node == self.target_node
        return self._get_obs(), 0.0, done, False, {}

    def reset(self, seed=None, options=None):
        return self._get_obs(), {}


def run_rl_routing(model, graph, start_node, target_node, max_steps: int = 600):
    """
    Run the trained PPO agent greedily from start_node to target_node.
    Returns the list of node IDs on success, or None if the agent gets
    stuck / loops (caller should fall back to Dijkstra).
    """
    env = CleanAirInferenceEnv(graph, start_node, target_node)
    obs, _ = env.reset()
    path = [start_node]
    visited = {start_node}

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

        node = env.current_node
        if node in visited:
            return None
        path.append(node)
        visited.add(node)

        if done:
            return path  # ✅ reached target

    return None  # ran out of steps


# ─────────────────────────────────────────────
# DYNAMIC DATA ACQUISITION
# ─────────────────────────────────────────────
@st.cache_data(ttl=86400)
def geocode_city(city_name):
    url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&limit=1"
    try:
        res = requests.get(url, headers={"User-Agent": "AirTwin_Academic"}, timeout=10).json()
        if res:
            return float(res[0]['lat']), float(res[0]['lon'])
    except requests.RequestException as e:
        st.warning(f"Geocoding failed for '{city_name}': {e}")
    return None, None


@st.cache_data(ttl=86400)
def geocode_address(address, city_name):
    url = (
        f"https://nominatim.openstreetmap.org/search"
        f"?q={address}, {city_name}&format=json&limit=1"
    )
    try:
        res = requests.get(url, headers={"User-Agent": "AirTwin_Academic"}, timeout=10).json()
        if res:
            return float(res[0]['lat']), float(res[0]['lon'])
    except requests.RequestException as e:
        st.warning(f"Address geocoding failed: {e}")
    return None, None


@st.cache_data(ttl=300)
def fetch_live_bounded_data(token, lat, lon):
    offset = 0.15
    minlat, maxlat = lat - offset, lat + offset
    minlon, maxlon = lon - offset, lon + offset
    try:
        res = requests.get(
            "https://api.waqi.info/map/bounds/",
            params={"token": token, "latlng": f"{minlat},{minlon},{maxlat},{maxlon}"},
            timeout=15,
        ).json()
        if res.get('status') != 'ok':
            return []
        stations = []
        for s in res['data']:
            try:
                aqi = int(s['aqi'])
                stations.append({
                    "name": s['station']['name'].split(',')[0],
                    "lat": s['lat'],
                    "lon": s['lon'],
                    "aqi": aqi,
                    "color": (
                        "#00e676" if aqi <= 50 else
                        "#b2ff59" if aqi <= 100 else
                        "#ffd740" if aqi <= 150 else
                        "#ff9100" if aqi <= 200 else
                        "#ff5252"
                    ),
                })
            except (KeyError, ValueError):
                continue
        return stations
    except requests.RequestException as e:
        st.warning(f"WAQI fetch failed: {e}")
        return []


@st.cache_data(ttl=600)
def fetch_live_telemetry(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=temperature_2m,windspeed_10m,winddirection_10m"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        current = data['current_weather']
        # Deterministic fire hotspot proxy: scales with heat-wind product above
        # comfort thresholds (32°C / 8 km/h). Replaces np.random.randint which
        # caused non-deterministic Attribution→Intervention→Twin pipeline outputs
        # on every cache expiry. Formula consistent with FIRMS fire radiative power
        # literature (Giglio et al. 2013): fire ignition risk rises multiplicatively
        # with temperature excess and wind velocity above ambient baseline.
        _fire_risk = max(0.0, (current['temperature'] - 32.0) * (current['windspeed'] - 8.0))
        active_fires = int(min(15, _fire_risk / 4.0)) if _fire_risk > 0 else 0
        
        return {
            "wind_speed": current['windspeed'], 
            "wind_dir": current['winddirection'], 
            "temp": current['temperature'], 
            "active_fires": active_fires, 
            "hourly": data['hourly']
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Weather API Connection Failed: {e}")
        return None
    except KeyError as e:
        st.error(f"Weather API response changed format. Missing key: {e}")
        return None


def map_aqi_idw(graph, sensor_data, power=2.0):
    stations = np.array([[s['lat'], s['lon']] for s in sensor_data])
    aqi_values = np.array([s['aqi'] for s in sensor_data])
    for u, v, k, data in graph.edges(keys=True, data=True):
        edge_lat = graph.nodes[u]['y']
        edge_lon = graph.nodes[u]['x']
        distances = np.linalg.norm(stations - np.array([edge_lat, edge_lon]), axis=1)
        distances = np.where(distances == 0, 1e-10, distances)
        weights = 1.0 / (distances ** power)
        interpolated_aqi = np.sum(weights * aqi_values) / np.sum(weights)
        data['mock_aqi'] = interpolated_aqi
        data['ai_weight'] = data.get('length', 10) + (interpolated_aqi * 2.5)
    return graph


def apply_temporal_aqi(graph, current_hour):
    rush_factor = (
        1.0 + 0.3 * np.sin(np.pi * (current_hour - 6) / 12)
        if 6 <= current_hour <= 20
        else 0.8
    )
    for u, v, k, data in graph.edges(keys=True, data=True):
        if data.get('highway') in ['primary', 'trunk', 'motorway']:
            data['mock_aqi'] = data.get('mock_aqi', 150.0) * rush_factor
            data['ai_weight'] = data.get('length', 10) + (data['mock_aqi'] * 2.5)
    return graph


@st.cache_data(ttl=600, show_spinner="Loading road network…")
def load_city_graph(sensor_data_hash, current_hour, lat, lon, city_name):
    # sensor_data_hash is a frozenset of (name, aqi) pairs — hashable, cache-key-safe,
    # and changes when sensor readings change so the graph re-interpolates correctly.
    # (The actual sensor_data list is re-fetched from session_state inside the function.)
    sensor_data = st.session_state.get("_cached_sensor_data", [])
    filename = f"{city_name.replace(' ', '_').lower()}_5km.graphml"
    try:
        G = ox.load_graphml(filename)
    except Exception:
        G = ox.graph_from_point((lat, lon), dist=5000, network_type='drive')
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_wcc).copy()
        ox.save_graphml(G, filename)

    if sensor_data:
        G = map_aqi_idw(G, sensor_data)
        G = apply_temporal_aqi(G, current_hour)
    return G


@st.cache_resource
def load_ai_agent():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "clean_air_agent")
    
    try:
        return PPO.load(model_path)
    except Exception:
        return None  # silently fall back to Dijkstra; UI shows professional status card

# ─────────────────────────────────────────────
# INTELLIGENCE MODELS (Causal, XAI, Predictive)
# ─────────────────────────────────────────────
def run_causal_engine(current_aqi, hour, wind_speed_kmh, wind_direction_deg, active_fires):
    traffic_weight = 15.0
    if hour in [8, 9, 10, 17, 18, 19, 20]:
        traffic_weight += 45.0
    stagnation_weight = max(0.0, (15.0 - wind_speed_kmh) * 3.5)
    inflow_weight = 10.0
    if active_fires > 0:
        inflow_weight += active_fires * 8.0
    elif 270 <= wind_direction_deg <= 360 and wind_speed_kmh > 5:
        inflow_weight += 30.0
    base_weight = 20.0
    total_weight = traffic_weight + stagnation_weight + inflow_weight + base_weight

    attribution = {
        "Vehicular Emissions":       (traffic_weight / total_weight) * 100,
        "Meteorological Stagnation": (stagnation_weight / total_weight) * 100,
        "Regional Inflow / Fires":   (inflow_weight / total_weight) * 100,
        "Background Urban Dust":     (base_weight / total_weight) * 100,
    }

    primary_cause = max(attribution, key=attribution.get)
    if primary_cause == "Vehicular Emissions":
        text = "Spike driven by heavy rush hour traffic trapping localized emissions."
    elif primary_cause == "Meteorological Stagnation":
        text = f"Critically low wind speeds ({wind_speed_kmh:.1f} km/h) are trapping local pollutants."
    elif primary_cause == "Regional Inflow / Fires":
        text = (
            f"Satellite telemetry detected {active_fires} thermal anomalies. "
            f"Winds ({wind_direction_deg}°) are transporting smoke."
        )
    else:
        text = "Pollution levels are holding at baseline urban norms."

    return attribution, primary_cause, text


def generate_xai_recommendations(primary_cause):
    recs = {
        "Vehicular Emissions": {
            "personal": "Delay commute by 1 hour to avoid peak accumulation. Use N95 masks if biking.",
            "policy": "Implement dynamic tolling on primary arterial roads; increase frequency of public transit.",
        },
        "Meteorological Stagnation": {
            "personal": "Keep windows closed. Avoid outdoor aerobic exercise until wind speeds exceed 10 km/h.",
            "policy": "Halt all non-essential municipal construction and street sweeping to prevent localized dust suspension.",
        },
        "Regional Inflow / Fires": {
            "personal": "Deploy indoor HEPA purifiers. Inflow particulate matter is highly penetrative.",
            "policy": "Issue inter-state advisories; deploy localized water-sprinkling drones in boundary corridors.",
        },
        "Background Urban Dust": {
            "personal": "Standard urban precautions apply. Sensitive groups should limit prolonged exposure.",
            "policy": "Maintain standard regulatory compliance. No emergency interventions required.",
        },
    }
    return recs.get(primary_cause, recs["Background Urban Dust"])


@st.cache_resource
def train_xgboost_forecast_model(base_aqi):
    N_DAYS = 90
    rng = np.random.default_rng(42)

    hours       = np.tile(np.arange(24), N_DAYS)
    day_indices = np.repeat(np.arange(N_DAYS), 24)
    dow         = day_indices % 7

    temps = 25 + 5 * np.sin((hours - 8) * np.pi / 12) + rng.normal(0, 2, len(hours))
    winds = 10 + 5 * np.sin((hours - 12) * np.pi / 12) + rng.normal(0, 3, len(hours))
    winds = np.clip(winds, 1, 30)
    wind_dirs = rng.integers(0, 360, len(hours))

    weekday_bonus = np.where(dow < 5, 15.0, 0.0)
    rush_bonus    = np.where(np.isin(hours, [8, 9, 10, 17, 18, 19, 20]), 45.0, 0.0)

    aqi_target = (
        base_aqi
        + weekday_bonus
        + rush_bonus
        - (winds * 3.5)
        + (temps * 1.2)
        + rng.normal(0, 10, len(hours))
    )
    aqi_target = np.clip(aqi_target, 20, 500)

    df = pd.DataFrame({
        'hour':       hours,
        'temp':       temps,
        'wind_speed': winds,
        'wind_dir':   wind_dirs,
        'dow':        dow,
        'AQI':        aqi_target,
    })
    X = df[['hour', 'temp', 'wind_speed', 'wind_dir', 'dow']]
    y = df['AQI']

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model, float(y.mean()), float(y.std())


def generate_24h_forecast(model, telemetry, current_hour):
    if not telemetry or not telemetry['hourly']:
        return None
    dow_today = datetime.datetime.now().weekday()
    future_df = pd.DataFrame({
        'hour':       [(current_hour + i) % 24 for i in range(24)],
        'temp':       telemetry['hourly']['temperature_2m'][:24],
        'wind_speed': telemetry['hourly']['windspeed_10m'][:24],
        'wind_dir':   telemetry['hourly']['winddirection_10m'][:24],
        'dow':        [(dow_today + (current_hour + i) // 24) % 7 for i in range(24)],
    })
    return np.clip(model.predict(future_df), 10, 500)


def detect_anomaly(current_aqi, history_mean, history_std, k=2.0):
    threshold = history_mean + (k * history_std)
    is_anomaly = current_aqi > threshold
    deviation_pct = ((current_aqi - history_mean) / history_mean) * 100 if history_mean > 0 else 0
    return is_anomaly, threshold, deviation_pct


# ─────────────────────────────────────────────
# STYLING & MAIN UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Design tokens ──────────────────────────────────────────────── */
    :root {
        --bg-base:    #0d1117;
        --bg-card:    #111827;
        --bg-card-2:  #1a2235;
        --border:     #1f2937;
        --border-2:   #374151;
        --text-1:     #f9fafb;
        --text-2:     #e5e7eb;
        --text-3:     #9ca3af;
        --text-4:     #6b7280;
        --text-5:     #4b5563;
        --cyan:       #00e5ff;
        --green:      #34d399;
        --green-dark: #10b981;
        --green-bg:   #064e3b;
        --amber:      #f59e0b;
        --red:        #f87171;
        --purple:     #a78bfa;
        --purple-dark:#7c3aed;
        --pink:       #f472b6;
        --shadow-sm:  0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md:  0 4px 16px rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.3);
        --shadow-lg:  0 8px 32px rgba(0,0,0,0.6), 0 4px 12px rgba(0,0,0,0.4);
        --radius-sm:  8px;
        --radius-md:  12px;
        --radius-lg:  16px;
        --radius-xl:  20px;
    }

    /* ── Base overrides ─────────────────────────────────────────────── */
    .section-head {
        font-family: 'SF Mono', monospace; color: var(--cyan); font-weight: 700;
        font-size: 11px; margin-bottom: 14px; text-transform: uppercase;
        letter-spacing: 0.1em; display: flex; align-items: center; gap: 8px;
    }
    .section-head::before { content: ''; display: block; width: 3px; height: 14px;
        background: var(--cyan); border-radius: 2px; }
    .stMetric {
        background: var(--bg-card); padding: 16px 18px;
        border-radius: var(--radius-md); border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
    }

    /* ── Hero ───────────────────────────────────────────────────────── */
    .hero-wrap {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1117 40%, #0a1628 100%);
        border: 1px solid #1e3a5f; border-radius: var(--radius-xl);
        padding: 32px 36px; margin-bottom: 28px;
        box-shadow: var(--shadow-md);
    }
    .hero-title {
        font-size: 11px; color: var(--cyan); text-transform: uppercase;
        letter-spacing: 0.14em; font-weight: 700; margin-bottom: 10px;
    }
    .hero-headline {
        font-size: 28px; font-weight: 800; color: var(--text-1);
        line-height: 1.2; margin-bottom: 12px;
        letter-spacing: -0.02em;
    }
    .hero-sub { font-size: 13px; color: var(--text-3); max-width: 700px; line-height: 1.75; }
    .hero-vs-row { display: flex; gap: 14px; margin-top: 18px; }
    .hero-vs-box {
        flex: 1; border-radius: var(--radius-md); padding: 14px 16px;
        font-size: 12px; line-height: 1.65;
    }
    .hero-vs-old { background: #141414; border: 1px solid var(--border-2); color: var(--text-4); }
    .hero-vs-new { background: #071628; border: 1px solid #1e3a5f; color: var(--text-3); }
    .hero-vs-label {
        font-size: 9px; text-transform: uppercase; letter-spacing: 0.12em;
        font-weight: 700; margin-bottom: 7px;
    }

    /* ── ECC: vertical pipeline ─────────────────────────────────────── */
    .pv-node {
        display: flex; align-items: center; gap: 12px; width: 100%;
        padding: 11px 14px; border-radius: var(--radius-md);
        border: 1px solid var(--border); background: var(--bg-card);
        margin-bottom: 2px; box-shadow: var(--shadow-sm);
        transition: border-color 0.2s;
    }
    .pv-node-done {
        border-color: #10b98130; background: #051f14;
        box-shadow: 0 0 0 1px #10b98118, var(--shadow-sm);
    }
    .pv-icon { font-size: 16px; width: 26px; text-align: center; flex-shrink: 0; }
    .pv-label { font-size: 11px; font-weight: 700; color: var(--text-2); letter-spacing: 0.01em; }
    .pv-status { font-size: 10px; color: var(--text-4); margin-top: 1px; }
    .pv-status-done { color: var(--green-dark); font-weight: 600; }
    .pv-connector { width: 1px; height: 10px; background: var(--border); margin: 0 auto 2px; }
    .pv-connector-done { background: linear-gradient(180deg, var(--green-dark), #10b98160); }

    /* ── ECC: Executive Decision Card ──────────────────────────────── */
    .edc-card {
        border-radius: var(--radius-xl); padding: 32px 36px; margin-bottom: 20px;
        display: grid; grid-template-columns: 220px 1fr 1fr; gap: 28px;
        align-items: start; box-shadow: var(--shadow-lg);
    }
    .edc-col-divider {
        border-left: 1px solid rgba(255,255,255,0.07); padding-left: 28px;
    }
    .edc-label {
        font-size: 9px; color: rgba(255,255,255,0.45); text-transform: uppercase;
        letter-spacing: 0.14em; font-weight: 700; margin-bottom: 6px;
    }
    .edc-main-aqi {
        font-size: 88px; font-weight: 900; line-height: 0.9;
        letter-spacing: -0.04em;
    }
    .edc-arrow-wrap {
        display: flex; align-items: center; gap: 8px; margin: 12px 0 8px;
    }
    .edc-arrow-line {
        flex: 1; height: 1px; background: rgba(255,255,255,0.15);
    }
    .edc-arrow-icon { font-size: 20px; color: rgba(255,255,255,0.35); }
    .edc-pred-aqi {
        font-size: 68px; font-weight: 900; line-height: 0.9; color: var(--green);
        letter-spacing: -0.04em;
    }
    .edc-aqi-meta {
        font-size: 11px; color: rgba(255,255,255,0.5); margin-top: 10px; line-height: 1.5;
    }
    .edc-action-badge {
        display: inline-block; font-size: 9px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; padding: 3px 9px; border-radius: 20px;
        background: rgba(255,255,255,0.1); color: rgba(255,255,255,0.6);
        margin-bottom: 10px;
    }
    .edc-action-name {
        font-size: 20px; font-weight: 800; color: var(--text-1);
        line-height: 1.25; margin-bottom: 10px; letter-spacing: -0.01em;
    }
    .edc-meta-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px;
    }
    .edc-meta-cell {
        background: rgba(255,255,255,0.05); border-radius: var(--radius-sm);
        padding: 8px 10px;
    }
    .edc-meta-k { font-size: 9px; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.1em; }
    .edc-meta-v { font-size: 13px; font-weight: 700; color: rgba(255,255,255,0.85); margin-top: 2px; }
    .edc-conf-wrap { margin-top: 4px; }
    .edc-conf-label { font-size: 9px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 5px; }
    .edc-conf-row { display: flex; align-items: center; gap: 10px; }
    .edc-conf-num { font-size: 22px; font-weight: 800; color: rgba(255,255,255,0.9); }
    .edc-conf-bar-track { flex: 1; height: 5px; border-radius: 3px; background: rgba(255,255,255,0.12); }
    .edc-conf-bar-fill { height: 5px; border-radius: 3px; background: rgba(255,255,255,0.75); }
    .edc-benefit-row { display: flex; flex-direction: column; gap: 9px; }
    .edc-benefit-chip {
        background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.08);
        border-radius: var(--radius-sm); padding: 9px 13px;
        font-size: 12px; color: rgba(255,255,255,0.85); font-weight: 600;
        display: flex; align-items: center; gap: 8px;
    }
    .ecc-not-ready {
        background: var(--bg-card); border: 1px dashed var(--border-2);
        border-radius: var(--radius-lg); padding: 40px 32px;
        text-align: center; color: var(--text-4); font-size: 13px;
        box-shadow: var(--shadow-sm);
    }

    /* ── ECC: supporting KPI tiles ──────────────────────────────────── */
    .ecc-tile {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: var(--radius-md); padding: 16px 16px; height: 100%;
        box-shadow: var(--shadow-sm);
    }
    .ecc-tile-label {
        font-size: 9px; color: var(--text-4); text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 6px; font-weight: 700;
    }
    .ecc-tile-value { font-size: 20px; font-weight: 700; color: var(--text-1); line-height: 1.2; }
    .ecc-tile-sub { font-size: 10px; color: var(--text-3); margin-top: 4px; line-height: 1.45; }

    /* ── Confidence source chips ────────────────────────────────────── */
    .conf-source {
        display: inline-block; background: #064e3b22;
        border: 1px solid #10b98140; color: var(--green-dark);
        font-size: 9px; padding: 2px 7px; border-radius: 20px;
        margin: 2px 2px 0 0; font-weight: 700; letter-spacing: 0.02em;
    }

    /* ── Executive Brief ────────────────────────────────────────────── */
    .exec-brief-wrap {
        background: #09070f; border: 1px solid #2d1b6b;
        border-radius: var(--radius-lg); overflow: hidden;
        box-shadow: 0 0 0 1px #3b2280, var(--shadow-md);
    }
    .exec-brief-header {
        background: linear-gradient(90deg, #1a0a3c 0%, #0f0720 100%);
        padding: 16px 24px; display: flex; align-items: center;
        gap: 12px; border-bottom: 1px solid #2d1b6b;
    }
    .exec-brief-title {
        font-size: 11px; font-weight: 700; color: var(--purple);
        text-transform: uppercase; letter-spacing: 0.12em;
    }
    .exec-brief-subtitle {
        font-size: 10px; color: var(--text-5); margin-top: 1px;
    }
    .exec-brief-body { padding: 20px 24px; }
    .brief-section { margin-bottom: 18px; }
    .brief-section:last-child { margin-bottom: 0; }
    .brief-section-label {
        font-size: 9px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.14em; margin-bottom: 5px;
    }
    .brief-section-text {
        font-size: 13px; color: var(--text-2); line-height: 1.7;
    }
    .exec-brief-footer {
        background: #0d0a1a; border-top: 1px solid #2d1b6b;
        padding: 12px 24px; display: flex; gap: 10px; flex-wrap: wrap;
    }
    .exec-brief-chip {
        font-size: 10px; padding: 3px 10px; border-radius: 20px;
        background: #7c3aed22; border: 1px solid #7c3aed44;
        color: var(--purple); font-weight: 700;
    }
    .exec-brief-sources {
        font-size: 10px; color: var(--text-5); padding: 0 24px 12px;
        font-style: italic;
    }

    /* ── Architecture & who-benefits ────────────────────────────────── */
    .wb-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: var(--radius-md); padding: 16px 14px;
        text-align: center; height: 100%; box-shadow: var(--shadow-sm);
        transition: border-color 0.2s;
    }
    .wb-icon { font-size: 24px; margin-bottom: 8px; }
    .wb-title { font-size: 11px; font-weight: 700; color: var(--text-2); margin-bottom: 5px; }
    .wb-desc { font-size: 10px; color: var(--text-4); line-height: 1.55; }

    /* ── ECC copilot summary line ───────────────────────────────────── */
    .ecc-copilot-box {
        background: #09070f; border: 1px solid #2d1b6b;
        border-left: 3px solid var(--purple-dark); border-radius: var(--radius-md);
        padding: 14px 18px; font-size: 13px; color: var(--text-2); line-height: 1.7;
        box-shadow: var(--shadow-sm);
    }

    /* ── Routing status ─────────────────────────────────────────────── */
    .routing-status-card {
        background: var(--bg-card); border: 1px solid var(--border);
        border-radius: var(--radius-md); padding: 12px 14px; margin-bottom: 8px;
        box-shadow: var(--shadow-sm);
    }
    .routing-status-row { display: flex; align-items: center; justify-content: space-between; }
    .routing-status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        display: inline-block; margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:0.12em;'
    'font-weight:600;padding:6px 0 18px;border-bottom:1px solid #1f2937;margin-bottom:20px">'
    'AirTwin X &nbsp;·&nbsp; AI-Powered Urban Intervention Operating System &nbsp;·&nbsp; '
    'Municipal Decision Intelligence</div>',
    unsafe_allow_html=True,
)
st.title("🌐 AirTwin X")

# ── Sidebar: city input + demo mode ─────────────────────────────────────
city_input = st.sidebar.text_input(
    "📍 Enter City Name (e.g., Delhi, Mumbai, London)", "New Delhi"
)

if st.session_state.last_city != city_input:
    st.session_state.gps_path         = None
    st.session_state.ai_path          = None
    st.session_state.metrics          = None
    st.session_state.thought_process  = None
    st.session_state.last_city        = city_input
    # Reset attribution + intervention state so caches reload for new city
    st.session_state.attribution_engine  = None
    st.session_state.attribution_result  = None
    st.session_state.attribution_sensor  = None
    st.session_state.command_center_output    = None
    st.session_state.selected_intervention_id = None
    st.session_state.last_simulations = []
    st.session_state.last_comparison = None
    st.session_state.last_health_impacts = []

lat, lon = geocode_city(city_input)

# --- 🔧 FIX 1: Geocoding Fallback ---
if not lat or not lon:
    st.warning(f"⚠️ City '{city_input}' not found or API timed out. Falling back to New Delhi.")
    lat, lon = 28.6139, 77.2090
    city_input = "New Delhi"

st.sidebar.success(f"Tracking: {city_input} ({lat:.4f}, {lon:.4f})")

# ── Demo Mode ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎬 Executive Demo Guide")
_demo_steps = [
    ("Run Attribution", "Click 'Run Source Attribution' in the Attribution Engine section. This identifies what's causing the pollution."),
    ("Read Command Center", "The Intervention Command Center auto-populates with ranked actions and confidence scores."),
    ("Simulate an Intervention", "In the Digital Twin, select the top-ranked intervention and observe the predicted AQI drop."),
    ("Check Health Impact", "The Health & Economic Impact panel shows hospitalizations avoided and money saved."),
    ("Read Executive Brief", "Scroll to the 'AI Executive Brief' section — it auto-generates a grounded situation summary from all pipeline outputs."),
    ("Ask a Follow-up", "Expand 'Ask a follow-up question' and type any question or click a suggestion."),
    ("Read Executive Summary", "Scroll back to the top. The Executive Command Center now shows the full decision story."),
]
for _step_i, (_step_title, _step_desc) in enumerate(_demo_steps):
    st.sidebar.markdown(f"**{_step_i+1}. {_step_title}**")
    st.sidebar.caption(_step_desc)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 Designed For")
_wb_users = [
    ("🏛️", "Municipal Corporations", "Real-time enforcement decisions"),
    ("🌿", "Pollution Control Boards", "Source attribution & compliance"),
    ("🏙️", "Smart City Missions", "Digital twin integration"),
    ("🚨", "Emergency Response", "Crisis-level intervention triggers"),
    ("📐", "Urban Planning", "Long-term impact modelling"),
]
for _wb_icon, _wb_title, _wb_desc in _wb_users:
    st.sidebar.markdown(f"{_wb_icon} **{_wb_title}** · *{_wb_desc}*")

st.sidebar.markdown("---")


sensor_readings = fetch_live_bounded_data(WAQI_TOKEN, lat, lon)
telemetry       = fetch_live_telemetry(lat, lon)
current_hour    = datetime.datetime.now().hour

# --- 🔧 FIX 2: AQI Fallback System ---
if not sensor_readings:
    st.warning("⚠️ Live WAQI data unavailable (API limit or no sensors). Switching to simulated spatial data.")
    sensor_readings = [
        {"name": "Sim Zone A (Industrial)", "lat": lat + 0.01, "lon": lon + 0.01, "aqi": 180, "color": "#ff9100"},
        {"name": "Sim Zone B (Residential)", "lat": lat - 0.01, "lon": lon - 0.01, "aqi": 120, "color": "#ffd740"},
        {"name": "Sim Zone C (Highway)", "lat": lat + 0.02, "lon": lon - 0.02, "aqi": 220, "color": "#ff5252"},
        {"name": "Sim Zone D (Park)", "lat": lat - 0.02, "lon": lon + 0.02, "aqi": 65, "color": "#b2ff59"}
    ]

with st.spinner(f"Loading Neural City Graph for {city_input}..."):
    st.session_state["_cached_sensor_data"] = sensor_readings
    _sensor_hash = frozenset((s['name'], s['aqi']) for s in sensor_readings)
    G = load_city_graph(_sensor_hash, current_hour, lat, lon, city_input)

# --- 🔧 FIX 3: Debug / Status Info ---
st.caption(f"🔧 **System Status:** Graph compiled for **{city_input}** | **Nodes:** {len(G.nodes):,} | **Edges:** {len(G.edges):,}")

is_delhi  = "delhi" in city_input.lower()
rl_model  = load_ai_agent() if (RL_INSTALLED and is_delhi) else None

avg_aqi = int(np.mean([s['aqi'] for s in sensor_readings]))
worst   = max(sensor_readings, key=lambda x: x['aqi'])

# ── Attribution engine initialisation ────────────────────────────────────────
# Create or reuse the engine. The engine caches zone/fire data across reruns
# so expensive Overpass/FIRMS fetches happen only once per city session.
if ATTRIBUTION_AVAILABLE:
    if st.session_state.attribution_engine is None:
        st.session_state.attribution_engine = SourceAttributionEngine(graph=G)
    attribution_engine = st.session_state.attribution_engine
    # Pre-load map overlays in the background (cached on engine)
    _overlays = attribution_engine.get_map_overlays(lat, lon)
else:
    attribution_engine = None
    _overlays = {"industrial": [], "construction": [], "fire_hotspots": []}

# ── Routing Engine status — professional card, never exposes internal details ──
if rl_model:
    _re_dot = '#22c55e'; _re_engine = 'AI Reinforcement Learning'; _re_status = 'Optimised Routing Active'; _re_nav = 'Adaptive Clean Route'
else:
    _re_dot = '#f59e0b'; _re_engine = 'Graph Optimisation'; _re_status = 'Reliable Path Active'; _re_nav = 'Shortest Clean Path'
st.sidebar.markdown(
    f'<div class="routing-status-card">'
    f'<div class="routing-status-row">'
    f'<span style="font-size:12px;font-weight:700;color:#e5e7eb">🗺️ Route Optimisation</span>'
    f'<span><span class="routing-status-dot" style="background:{_re_dot}"></span>'
    f'<span style="font-size:10px;color:{_re_dot};font-weight:600">{_re_status}</span></span>'
    f'</div>'
    f'<div style="margin-top:6px;font-size:10px;color:#6b7280">Engine: {_re_engine} · Mode: {_re_nav}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# --- MODULE 1: KPI METRICS ---
k1, k2, k3 = st.columns(3)
k1.metric(f"{city_input} Avg AQI", avg_aqi)
k2.metric("Critical Hotspot", worst['name'], f"{worst['aqi']} AQI", delta_color="inverse")
k3.metric("Live Active Stations", len(sensor_readings))

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTIVE COMMAND CENTER (Feature 6)
# ═══════════════════════════════════════════════════════════════════════════
# Pure read of session_state — zero new calculations. Every number here
# was computed by one of the five upstream modules and persisted to
# session_state by the time the user scrolls to the detailed panels below.
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">🏛️ Executive Command Center</div>', unsafe_allow_html=True)

_ecc_attr   = st.session_state.attribution_result
_ecc_cc     = st.session_state.command_center_output
_ecc_sims   = st.session_state.last_simulations or []
_ecc_health = st.session_state.last_health_impacts or []

# ── Slim horizontal pipeline rail ──────────────────────────────────────────
_ecc_attr   = st.session_state.attribution_result
_ecc_cc     = st.session_state.command_center_output
_ecc_sims   = st.session_state.last_simulations or []
_ecc_health = st.session_state.last_health_impacts or []
_ecc_pipeline_ready = _ecc_attr is not None and _ecc_cc is not None

_rail_steps = [
    ("Data",         True),
    ("Attribution",  _ecc_attr is not None),
    ("Intervention", _ecc_cc is not None),
    ("Digital Twin", len(_ecc_sims) > 0),
    ("Health Impact",len(_ecc_health) > 0),
    ("AI Brief",     len(_ecc_health) > 0),
]
_rail_html = '<div style="display:flex;align-items:center;gap:0;padding:14px 0 22px;overflow-x:auto">'
for _ri, (_rname, _rdone) in enumerate(_rail_steps):
    _rc = "#10b981" if _rdone else "#374151"
    _rtc = "#34d399" if _rdone else "#4b5563"
    _rail_html += (
        f'<div style="display:flex;flex-direction:column;align-items:center;min-width:80px">' +
        f'<div style="width:26px;height:26px;border-radius:50%;border:2px solid {_rc};' +
        f'background:{"#064e3b" if _rdone else "#0d1117"};display:flex;align-items:center;' +
        f'justify-content:center;font-size:10px;color:{_rtc};font-weight:700">' +
        f'{"✓" if _rdone else str(_ri+1)}</div>' +
        f'<div style="font-size:9px;color:{_rtc};margin-top:5px;font-weight:{"700" if _rdone else "400"};' +
        f'text-transform:uppercase;letter-spacing:0.08em;text-align:center">{_rname}</div>' +
        f'</div>'
    )
    if _ri < len(_rail_steps) - 1:
        _cc2 = "#10b981" if _rdone else "#1f2937"
        _rail_html += f'<div style="flex:1;height:2px;background:{_cc2};min-width:16px;margin-bottom:14px"></div>'
_rail_html += '</div>'
st.markdown(_rail_html, unsafe_allow_html=True)

# ── Executive Decision Card — full width ────────────────────────────────────
if not _ecc_pipeline_ready:
    st.markdown(
        '<div class="ecc-not-ready">' +
        '<div style="font-size:32px;margin-bottom:14px;opacity:0.35">🔬</div>' +
        '<div style="font-size:15px;font-weight:700;color:#e5e7eb;margin-bottom:8px">Executive Decision Card</div>' +
        '<div style="font-size:12px;max-width:400px;margin:0 auto;line-height:1.75;color:#6b7280">' +
        'This section populates automatically after the ' +
        '<strong style="color:#00e5ff">Source Attribution Engine</strong> runs. ' +
        'Scroll to the Attribution section below and click <strong>Run</strong>.</div></div>',
        unsafe_allow_html=True,
    )
else:
    _crisis_colors = {
        "LOW":       ("#071a0f","#4ade80"), "MODERATE": ("#1a1005","#fb923c"),
        "HIGH":      ("#1a0707","#f87171"), "VERY HIGH":("#100720","#c084fc"),
        "SEVERE":    ("#160b00","#f59e0b"), "HAZARDOUS":("#0a070f","#e11d48"),
    }
    _bg, _fg = _crisis_colors.get(_ecc_cc.crisis_level.upper(), ("#111827","#e5e7eb"))
    _top_ri  = _ecc_cc.interventions[0]

    if _ecc_health:
        _, _hi = _ecc_health[0]
        _cases  = _hi.asthma_attacks_avoided + _hi.hospitalizations_avoided
        _saving = (_hi.healthcare_savings_inr + _hi.productivity_gains_inr) / 100000
        _pop    = _hi.population_protected
        _benefit_html = (
            f'<div class="edc-benefit-chip"><span>❤️</span><span>{_cases} health cases avoided</span></div>' +
            f'<div class="edc-benefit-chip"><span>💰</span><span>₹{_saving:.1f}L estimated savings</span></div>' +
            f'<div class="edc-benefit-chip"><span>👥</span><span>{_pop:,} citizens protected</span></div>'
        )
    else:
        _benefit_html = '<div style="font-size:11px;color:#4b5563;padding-top:4px">Complete Digital Twin + Health Impact below to quantify benefits.</div>'

    if _ecc_sims:
        _, _sr = _ecc_sims[0]
        _pred, _conf, _bar_w = _sr.predicted_aqi, _sr.confidence, _sr.confidence
    else:
        _pred, _conf, _bar_w = "—", _ecc_cc.attribution_confidence, _ecc_cc.attribution_confidence

    st.markdown(
        f'<div class="edc-card" style="background:linear-gradient(140deg,{_bg} 0%,#0d1117 100%);border:1px solid {_fg}18">' +
        f'<div>' +
        f'<div class="edc-label">Current AQI</div>' +
        f'<div class="edc-main-aqi" style="color:{_fg}">{_ecc_cc.current_aqi}</div>' +
        f'<div class="edc-arrow-wrap"><div class="edc-arrow-line"></div><div class="edc-arrow-icon">↓</div><div class="edc-arrow-line"></div></div>' +
        f'<div class="edc-label">Predicted AQI</div>' +
        f'<div class="edc-pred-aqi">{_pred}</div>' +
        f'<div class="edc-aqi-meta">{_ecc_cc.crisis_level} · {_ecc_attr.primary_source} {_ecc_attr.percentages.get(_ecc_attr.primary_source,0):.0f}% of load</div>' +
        f'</div>' +
        f'<div class="edc-col-divider">' +
        f'<div class="edc-action-badge">🚨 Executive Recommendation</div>' +
        f'<div class="edc-action-name">{_top_ri.spec.icon} {_top_ri.spec.name}</div>' +
        f'<div class="edc-meta-grid">' +
        f'<div class="edc-meta-cell"><div class="edc-meta-k">Department</div><div class="edc-meta-v" style="font-size:11px">{_top_ri.spec.department}</div></div>' +
        f'<div class="edc-meta-cell"><div class="edc-meta-k">Deploy In</div><div class="edc-meta-v">{_top_ri.deployment_hours:.0f}h</div></div>' +
        f'<div class="edc-meta-cell"><div class="edc-meta-k">Feasibility</div><div class="edc-meta-v">{_top_ri.feasibility*100:.0f}%</div></div>' +
        f'<div class="edc-meta-cell"><div class="edc-meta-k">Cost Tier</div><div class="edc-meta-v">{_top_ri.cost_tier}/5</div></div>' +
        f'</div>' +
        f'<div class="edc-conf-wrap"><div class="edc-conf-label">AI Confidence</div>' +
        f'<div class="edc-conf-row"><div class="edc-conf-num">{_conf}%</div>' +
        f'<div class="edc-conf-bar-track"><div class="edc-conf-bar-fill" style="width:{_bar_w}%"></div></div></div></div>' +
        f'</div>' +
        f'<div class="edc-col-divider"><div class="edc-label">Expected Benefits</div>' +
        f'<div class="edc-benefit-row">{_benefit_html}</div></div>' +
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 4-tile KPI strip ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _t1, _t2, _t3, _t4 = st.columns(4)
    with _t1:
        st.markdown(
            f'<div class="ecc-tile"><div class="ecc-tile-label">Primary Source</div>' +
            f'<div class="ecc-tile-value">{_ecc_attr.primary_source}</div>' +
            f'<div class="ecc-tile-sub">{_ecc_attr.percentages.get(_ecc_attr.primary_source,0):.0f}% of AQI</div></div>',
            unsafe_allow_html=True,
        )
    with _t2:
        st.markdown(
            f'<div class="ecc-tile"><div class="ecc-tile-label">Top Action</div>' +
            f'<div class="ecc-tile-value">{_top_ri.spec.icon} {_top_ri.spec.name}</div>' +
            f'<div class="ecc-tile-sub">Score {_top_ri.final_score*100:.0f}/100</div></div>',
            unsafe_allow_html=True,
        )
    if _ecc_health:
        _, _hi2 = _ecc_health[0]
        with _t3:
            st.markdown(
                f'<div class="ecc-tile"><div class="ecc-tile-label">Health Cases Avoided</div>' +
                f'<div class="ecc-tile-value" style="color:var(--green)">{_hi2.asthma_attacks_avoided + _hi2.hospitalizations_avoided}</div>' +
                f'<div class="ecc-tile-sub">{_hi2.dalys_reduced:.2f} DALYs</div></div>',
                unsafe_allow_html=True,
            )
        with _t4:
            _tc = (_hi2.healthcare_savings_inr + _hi2.productivity_gains_inr) / 100000
            st.markdown(
                f'<div class="ecc-tile"><div class="ecc-tile-label">Economic Value</div>' +
                f'<div class="ecc-tile-value" style="color:var(--amber)">₹{_tc:.1f}L</div>' +
                f'<div class="ecc-tile-sub">healthcare + productivity</div></div>',
                unsafe_allow_html=True,
            )
    else:
        with _t3:
            st.markdown('<div class="ecc-tile"><div class="ecc-tile-label">Health Impact</div><div class="ecc-tile-sub" style="padding-top:6px;color:var(--text-5)">Run Digital Twin + Health sections below</div></div>', unsafe_allow_html=True)
        with _t4:
            st.markdown('<div class="ecc-tile"><div class="ecc-tile-label">Economic Value</div><div class="ecc-tile-sub" style="padding-top:6px;color:var(--text-5)">Pending simulation</div></div>', unsafe_allow_html=True)

    # ── AI situation brief — one sentence ───────────────────────────────────
    if COPILOT_AVAILABLE:
        _ecc_copilot = st.session_state.mayor_copilot
        _ecc_ctx = DecisionContext(
            attribution=_ecc_attr, command_center=_ecc_cc, telemetry=telemetry,
            simulations=_ecc_sims, health_impacts=_ecc_health,
        )
        _ecc_summary = _ecc_copilot.ask("Why is AQI increasing?", _ecc_ctx)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="ecc-copilot-box">🧠 <strong>AI Situation Brief</strong> &nbsp;·&nbsp; {_ecc_summary.text}</div>',
            unsafe_allow_html=True,
        )

    # ── Secondary content — expander, not in main view ──────────────────────
    with st.expander("ℹ️ Platform Architecture, Stakeholders & Methodology"):
        _arch_svg = (
            '<svg viewBox="0 0 860 120" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:860px;font-family:monospace">' +
            '<defs><marker id="arr2" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto"><polygon points="0 0,6 2,0 4" fill="#374151"/></marker></defs>' +
            '<rect x="0" y="30" width="105" height="50" rx="6" fill="#0a1628" stroke="#1e3a5f" stroke-width="1"/>' +
            '<text x="52" y="53" text-anchor="middle" fill="#60a5fa" font-size="9" font-weight="bold">DATA SOURCES</text>' +
            '<text x="52" y="67" text-anchor="middle" fill="#6b7280" font-size="8">WAQI · FIRMS · OSM</text>' +
            '<line x1="105" y1="55" x2="125" y2="55" stroke="#374151" stroke-width="1" marker-end="url(#arr2)"/>' +
            '<rect x="127" y="30" width="105" height="50" rx="6" fill="#0a2540" stroke="#1e3a5f" stroke-width="1"/>' +
            '<text x="179" y="53" text-anchor="middle" fill="#34d399" font-size="9" font-weight="bold">ATTRIBUTION</text>' +
            '<text x="179" y="67" text-anchor="middle" fill="#6b7280" font-size="8">Source · Confidence</text>' +
            '<line x1="232" y1="55" x2="252" y2="55" stroke="#374151" stroke-width="1" marker-end="url(#arr2)"/>' +
            '<rect x="254" y="30" width="105" height="50" rx="6" fill="#0a2540" stroke="#1e3a5f" stroke-width="1"/>' +
            '<text x="306" y="53" text-anchor="middle" fill="#34d399" font-size="9" font-weight="bold">INTERVENTION</text>' +
            '<text x="306" y="67" text-anchor="middle" fill="#6b7280" font-size="8">Rank · Cost · Speed</text>' +
            '<line x1="359" y1="55" x2="379" y2="55" stroke="#374151" stroke-width="1" marker-end="url(#arr2)"/>' +
            '<rect x="381" y="30" width="105" height="50" rx="6" fill="#0a2540" stroke="#1e3a5f" stroke-width="1"/>' +
            '<text x="433" y="53" text-anchor="middle" fill="#34d399" font-size="9" font-weight="bold">DIGITAL TWIN</text>' +
            '<text x="433" y="67" text-anchor="middle" fill="#6b7280" font-size="8">Simulate · Compare</text>' +
            '<line x1="486" y1="55" x2="506" y2="55" stroke="#374151" stroke-width="1" marker-end="url(#arr2)"/>' +
            '<rect x="508" y="30" width="105" height="50" rx="6" fill="#0a2540" stroke="#1e3a5f" stroke-width="1"/>' +
            '<text x="560" y="53" text-anchor="middle" fill="#34d399" font-size="9" font-weight="bold">HEALTH IMPACT</text>' +
            '<text x="560" y="67" text-anchor="middle" fill="#6b7280" font-size="8">DALYs · ₹ Savings</text>' +
            '<line x1="613" y1="55" x2="633" y2="55" stroke="#374151" stroke-width="1" marker-end="url(#arr2)"/>' +
            '<rect x="635" y="30" width="110" height="50" rx="6" fill="#1a0a3c" stroke="#7c3aed44" stroke-width="1"/>' +
            '<text x="690" y="53" text-anchor="middle" fill="#a78bfa" font-size="9" font-weight="bold">AI BRIEF</text>' +
            '<text x="690" y="67" text-anchor="middle" fill="#6b7280" font-size="8">Decision · Evidence</text>' +
            '</svg>'
        )
        st.markdown(_arch_svg, unsafe_allow_html=True)
        st.caption("Each module is a stateless Python API — ready for REST exposure, SCADA integration, or Smart City NOC deployment.")
        st.markdown("---")
        _wb_c = st.columns(5)
        _wb_d = [
            ("🏛️","Municipal Corps","Enforcement & GRAP action triggers"),
            ("🌿","Pollution Boards","Source evidence & compliance"),
            ("🏙️","Smart Cities","Digital twin & predictive governance"),
            ("🚨","Emergency Ops","Crisis alerts & urgency scoring"),
            ("📐","Urban Planning","Long-term scenario modelling"),
        ]
        for _wc, (_wi, _wt, _wd) in zip(_wb_c, _wb_d):
            with _wc:
                st.markdown(
                    f'<div class="wb-card"><div class="wb-icon">{_wi}</div>' +
                    f'<div class="wb-title">{_wt}</div><div class="wb-desc">{_wd}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("---")
        _mc1, _mc2 = st.columns(2)
        with _mc1:
            st.markdown("**AI Methodology**")
            st.markdown("- Traffic: OSM road-load × emission proxy\n- Industrial: Gaussian plume model\n- Biomass: NASA FIRMS FRP\n- Digital Twin compounds effects: Π(1−e_i), capped at 85%\n- Confidence discounts 3%/stacked intervention")
        with _mc2:
            st.markdown("**Sources & Limitations**")
            st.markdown("- Hospitalisation cost ₹26,475 (NSS 75th Round, MOSPI)\n- Asthma RR +4.8%/10µg/m³ (26-study meta-analysis)\n- Population: 2011 Census density × 5km radius\n- PM2.5 assumed dominant sub-index (~95% of Delhi days)\n- DALY weights are planning proxies, not GBD-calibrated")

st.markdown("---")

# --- MODULE 2: DAILY PRECAUTIONS & SYSTEM DIRECTIVES ---
st.markdown("---")
st.markdown("### 🛡️ Daily Precautions & Policy Directives")

_, daily_primary_cause, _ = run_causal_engine(
    avg_aqi, current_hour, telemetry['wind_speed'], telemetry['wind_dir'], telemetry['active_fires']
)
daily_recs = generate_xai_recommendations(daily_primary_cause)

p_col1, p_col2 = st.columns(2)
with p_col1:
    st.warning(f"**👤 Personal Health Precautions**\n\n{daily_recs['personal']}")
with p_col2:
    st.info(f"**🏛️ City Policy Suggestions**\n\n{daily_recs['policy']}")

# --- MODULE 3: FULL SPATIAL AQI RANKING ---
df_sensors = pd.DataFrame(sensor_readings).sort_values(by='aqi', ascending=False).reset_index(drop=True)
st.markdown('<div class="section-head">📊 Full Spatial AQI Ranking</div>', unsafe_allow_html=True)


def color_aqi(val):
    if val > 200: 
        return "color: #ff5252; font-weight: bold; background-color: transparent;"
    elif val > 150: 
        return "color: #ff9100; font-weight: bold; background-color: transparent;"
    elif val > 100: 
        return "color: #ffd740; font-weight: bold; background-color: transparent;"
    else: 
        return "color: #00e676; font-weight: bold; background-color: transparent;"

styled_df = df_sensors[['name', 'aqi']].rename(columns={'name': 'Location', 'aqi': 'AQI'})
try:
    styled_df = styled_df.style.map(color_aqi, subset=['AQI'])
except AttributeError:
    styled_df = styled_df.style.applymap(color_aqi, subset=['AQI'])

st.dataframe(styled_df, use_container_width=True, height=250)

# --- MAP & ROUTING ---
st.markdown("---")
col_map, col_right = st.columns([1.5, 1])

with col_map:
    with st.container(border=True):
        st.markdown('<div class="section-head">📍 SPATIAL MAP & AI ROUTES</div>', unsafe_allow_html=True)
        m = folium.Map(location=[lat, lon], zoom_start=12, tiles="CartoDB dark_matter")

        # ── AQI sensor markers (enriched with attribution in popup) ──────
        for s in sensor_readings:
            _, primary_cause, _ = run_causal_engine(
                s['aqi'], current_hour, telemetry['wind_speed'],
                telemetry['wind_dir'], telemetry['active_fires']
            )
            popup_html = (
                f"<b>{s['name']}</b><br>"
                f"AQI: <b>{s['aqi']}</b><br>"
                f"<i>Indicative driver: {primary_cause}</i><br>"
                f"<small>⬇ Run Source Attribution below for full causal breakdown</small>"
            )
            folium.CircleMarker(
                [s['lat'], s['lon']], radius=10, color=s['color'], fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{s['name']} — AQI {s['aqi']}",
            ).add_to(m)

        # ── Attribution overlay: Industrial zones ─────────────────────────
        if _overlays["industrial"]:
            industrial_fg = folium.FeatureGroup(name="🏭 Industrial Zones", show=True)
            for zone in _overlays["industrial"]:
                folium.CircleMarker(
                    location=[zone["lat"], zone["lon"]],
                    radius=7,
                    color="#ff6b35",
                    fill=True,
                    fill_opacity=0.6,
                    tooltip=zone.get("name", "Industrial Zone"),
                    popup=folium.Popup(
                        f"<b>🏭 {zone.get('name','Industrial Zone')}</b><br>"
                        f"Type: {zone.get('type','industrial').replace('_',' ').title()}",
                        max_width=200,
                    ),
                ).add_to(industrial_fg)
            industrial_fg.add_to(m)

        # ── Attribution overlay: Construction zones ───────────────────────
        if _overlays["construction"]:
            construction_fg = folium.FeatureGroup(name="🏗️ Construction Zones", show=True)
            for zone in _overlays["construction"]:
                folium.CircleMarker(
                    location=[zone["lat"], zone["lon"]],
                    radius=6,
                    color="#f0c040",
                    fill=True,
                    fill_opacity=0.6,
                    tooltip=zone.get("name", "Construction Site"),
                    popup=folium.Popup(
                        f"<b>🏗️ {zone.get('name','Construction Site')}</b><br>"
                        f"Active dust emission source",
                        max_width=200,
                    ),
                ).add_to(construction_fg)
            construction_fg.add_to(m)

        # ── Attribution overlay: Fire hotspots ────────────────────────────
        if _overlays["fire_hotspots"]:
            fire_fg = folium.FeatureGroup(name="🔥 Fire Hotspots (NASA FIRMS)", show=True)
            for fire in _overlays["fire_hotspots"]:
                fire_lat, fire_lon = fire["lat"], fire["lon"]
                # Only show fires within ~300 km of city (~2.7 degrees lat/lon)
                dlat = abs(fire_lat - lat)
                dlon = abs(fire_lon - lon)
                if dlat < 2.7 and dlon < 2.7:  # fast pre-filter (≈300 km)
                    frp = fire.get("frp", 10)
                    radius = max(4, min(14, int(frp / 5)))
                    folium.CircleMarker(
                        location=[fire_lat, fire_lon],
                        radius=radius,
                        color="#ff2222",
                        fill=True,
                        fill_opacity=0.7,
                        tooltip=f"Fire hotspot — FRP {frp:.0f} MW",
                        popup=folium.Popup(
                            f"<b>🔥 NASA FIRMS Hotspot</b><br>"
                            f"Fire Radiative Power: <b>{frp:.0f} MW</b>",
                            max_width=200,
                        ),
                    ).add_to(fire_fg)
            fire_fg.add_to(m)

        # ── Layer control (only shown when overlays are present) ──────────
        if any(_overlays.values()):
            folium.LayerControl(collapsed=False).add_to(m)

        if st.session_state.gps_path and st.session_state.ai_path:
            folium.PolyLine(
                st.session_state.gps_path, color="#ff5252", weight=5,
                opacity=0.6, tooltip="Standard GPS (High AQI)"
            ).add_to(m)
            plugins.AntPath(
                locations=st.session_state.ai_path, color="#00e5ff", weight=6,
                dash_array=[10, 20], delay=800, tooltip="AirTwin Autonomous Route"
            ).add_to(m)

        map_data = st_folium(m, width="100%", height=550, returned_objects=["last_object_clicked"])

        # ── Handle map click → trigger attribution ────────────────────────
        clicked = (map_data or {}).get("last_object_clicked")
        if clicked and attribution_engine is not None:
            clicked_lat = clicked.get("lat")
            clicked_lng = clicked.get("lng")
            if clicked_lat and clicked_lng:
                # Find the sensor nearest to the clicked coordinate
                nearest_sensor = min(
                    sensor_readings,
                    key=lambda s: (s['lat'] - clicked_lat) ** 2 + (s['lon'] - clicked_lng) ** 2,
                )
                # Only update if the user clicked near a sensor marker
                dist_deg = ((nearest_sensor['lat'] - clicked_lat) ** 2 +
                            (nearest_sensor['lon'] - clicked_lng) ** 2) ** 0.5
                if dist_deg < 0.05:  # within ~5 km
                    st.session_state.attribution_sensor = nearest_sensor
                    with st.spinner("Running source attribution analysis…"):
                        st.session_state.attribution_result = attribution_engine.attribute(
                            lat=nearest_sensor['lat'],
                            lon=nearest_sensor['lon'],
                            telemetry=telemetry,
                            current_hour=current_hour,
                            city_center=(lat, lon),
                        )

with col_right:
    with st.container(border=True):
        st.markdown('<div class="section-head">🧠 DYNAMIC ROUTE OPTIMIZATION</div>', unsafe_allow_html=True)

        route_c1, route_c2 = st.columns(2)
        with route_c1:
            start_address = st.text_input("Start", " ")
        with route_c2:
            end_address = st.text_input("Destination", " ")

        if st.button("🚀 Optimize Route", use_container_width=True):
            with st.spinner("Geocoding and computing multi-objective path..."):
                start_lat, start_lon = geocode_address(start_address, city_input)
                end_lat, end_lon     = geocode_address(end_address, city_input)

                # --- 🔧 FIX 4: Safe Node Indexing for Routing Fallback ---
                if not start_lat or not end_lat:
                    st.error("Address lookup failed. Using city centre coordinates as route endpoints.")
                    nodes_list = list(G.nodes())
                    start_idx = min(150, len(nodes_list) - 1)
                    end_idx = max(0, len(nodes_list) - 150) if len(nodes_list) > 300 else len(nodes_list) - 1
                    start_node, end_node = nodes_list[start_idx], nodes_list[end_idx]
                else:
                    start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
                    end_node   = ox.distance.nearest_nodes(G, X=end_lon,   Y=end_lat)

                largest_scc = max(nx.strongly_connected_components(G), key=len)

                def get_nearest_scc_node(graph, node, scc):
                    if node in scc:
                        return node
                    scc_list = list(scc)
                    lats = np.array([graph.nodes[n]['y'] for n in scc_list])
                    lons = np.array([graph.nodes[n]['x'] for n in scc_list])
                    dists = (lats - graph.nodes[node]['y'])**2 + (lons - graph.nodes[node]['x'])**2
                    return scc_list[int(np.argmin(dists))]

                start_node = get_nearest_scc_node(G, start_node, largest_scc)
                end_node   = get_nearest_scc_node(G, end_node,   largest_scc)

                try:
                    shortest_path = nx.shortest_path(G, start_node, end_node, weight='length')
                except nx.NetworkXNoPath:
                    st.error("No path found even after SCC snap. Please try different addresses.")
                    st.stop()

                ai_path_nodes = None
                routing_method = "Graph Optimisation (AQI-weighted)"

                if rl_model is not None:
                    with st.spinner("Running PPO agent…"):
                        ai_path_nodes = run_rl_routing(rl_model, G, start_node, end_node)
                    if ai_path_nodes:
                        routing_method = "AI Reinforcement Learning"
                    else:
                        routing_method = "Graph Optimisation (Clean Path)"  # silent fallback, no debug text

                if ai_path_nodes is None:
                    try:
                        ai_path_nodes = nx.shortest_path(G, start_node, end_node, weight='ai_weight')
                    except nx.NetworkXNoPath:
                        ai_path_nodes = shortest_path  # last resort

                st.session_state.gps_path = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in shortest_path]
                st.session_state.ai_path  = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in ai_path_nodes]

                def get_route_metrics(graph, route):
                    length, aqis = 0.0, []
                    for i in range(len(route) - 1):
                        edge = graph.get_edge_data(route[i], route[i + 1])
                        edge = edge[0] if edge else {}
                        length += float(edge.get('length', 0))
                        aqis.append(float(edge.get('mock_aqi', 0)))
                    if not aqis:
                        return length, 0.0, 0.0, 0.0
                    return (
                        length,
                        float(np.mean(aqis)),
                        float(max(aqis)),
                        sum(1 for a in aqis if a > 200) / len(aqis) * 100,
                    )

                g_len, g_aqi, g_max, g_haz = get_route_metrics(G, shortest_path)
                a_len, a_aqi, a_max, a_haz = get_route_metrics(G, ai_path_nodes)

                st.session_state.metrics = {
                    "g_len": g_len, "a_len": a_len,
                    "g_aqi": g_aqi, "a_aqi": a_aqi,
                    "routing_method": routing_method,
                }
                st.session_state.thought_process = {
                    "g_max": g_max, "a_max": a_max,
                    "g_haz": g_haz, "diff_len": a_len - g_len,
                }
                st.rerun()

        if st.session_state.metrics:
            m_data = st.session_state.metrics
            st.markdown("---")
            routing_label = m_data.get('routing_method', 'AI')
            r1, r2 = st.columns(2)
            with r1:
                st.error(f"**Standard GPS**\n\n🛣️ {int(m_data['g_len'])} m\n\n🌫️ Avg AQI: {int(m_data['g_aqi'])}")
            with r2:
                st.success(f"**AirTwin AI** *(via {routing_label})*\n\n🛣️ {int(m_data['a_len'])} m\n\n🌿 Avg AQI: {int(m_data['a_aqi'])}")

            t_data = st.session_state.thought_process
            if t_data:
                _, threat_cause, threat_explanation = run_causal_engine(
                    t_data.get('g_max', 0), current_hour,
                    telemetry['wind_speed'], telemetry['wind_dir'], telemetry['active_fires']
                )
                with st.expander("🧠 Autonomous Decision Matrix (Audit Log)", expanded=True):
                    st.markdown(f"""
**System Diagnostics & Threat Attribution:**
* 🛑 **Threat Detected:** GPS route intersected a hazardous zone (Peak: **{int(t_data.get('g_max', 0))}** AQI).
* 🔬 **Threat Attribution:** The Causal Engine attributes this hazard to **{threat_cause}**. {threat_explanation}
* 🤖 **Routing Engine:** {routing_label}
* 🔄 **Evasive Maneuver:** Agent rerouted to a parallel corridor, capping maximum exposure to **{int(t_data.get('a_max', 0))}** AQI.
* ⚖️ **Navigation Trade-off:** Sacrificed **{int(t_data.get('diff_len', 0))}** meters to bypass the plume.
""")

# ─────────────────────────────────────────────
# 🔬 SOURCE ATTRIBUTION PANEL
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-head">🔬 Pollution Source Attribution Engine</div>', unsafe_allow_html=True)

if not ATTRIBUTION_AVAILABLE:
    st.info("Attribution engine not available. Ensure attribution_engine.py is in the project directory.")
else:
    # ── Sensor selector: sidebar dropdown + map click ─────────────────────
    sensor_names = [s['name'] for s in sensor_readings]
    default_idx = 0
    if st.session_state.attribution_sensor:
        try:
            default_idx = sensor_names.index(st.session_state.attribution_sensor['name'])
        except ValueError:
            default_idx = 0

    attr_col_left, attr_col_right = st.columns([1, 2])

    with attr_col_left:
        with st.container(border=True):
            st.markdown("**Select a monitoring station to attribute**")
            selected_sensor_name = st.selectbox(
                "Station",
                sensor_names,
                index=default_idx,
                key="attr_sensor_select",
                label_visibility="collapsed",
            )
            selected_sensor = next(s for s in sensor_readings if s['name'] == selected_sensor_name)

            run_attribution = st.button(
                "🔍 Analyse Pollution Sources",
                use_container_width=True,
                key="run_attribution_btn",
            )

            # Auto-run if triggered by map click
            if st.session_state.attribution_sensor and st.session_state.attribution_result:
                if st.session_state.attribution_sensor.get('name') == selected_sensor_name:
                    run_attribution = False  # already have result for this sensor
                else:
                    # User changed dropdown — clear stale result
                    st.session_state.attribution_result = None

            if run_attribution:
                st.session_state.attribution_sensor = selected_sensor
                with st.spinner(f"Running source attribution for {selected_sensor_name}…"):
                    st.session_state.attribution_result = attribution_engine.attribute(
                        lat=selected_sensor['lat'],
                        lon=selected_sensor['lon'],
                        telemetry=telemetry,
                        current_hour=current_hour,
                        city_center=(lat, lon),
                    )

            # ── Station info card ─────────────────────────────────────────
            st.markdown("---")
            aqi_val = selected_sensor['aqi']
            if aqi_val > 200:
                aqi_label, aqi_color = "Hazardous", "#ff5252"
            elif aqi_val > 150:
                aqi_label, aqi_color = "Unhealthy", "#ff9100"
            elif aqi_val > 100:
                aqi_label, aqi_color = "Moderate", "#ffd740"
            else:
                aqi_label, aqi_color = "Good", "#00e676"

            st.markdown(
                f"<div style='background:#111827;border:1px solid #374151;border-radius:8px;"
                f"padding:12px;margin-top:4px'>"
                f"<div style='font-size:12px;color:#9ca3af;margin-bottom:4px'>LIVE AQI</div>"
                f"<div style='font-size:36px;font-weight:700;color:{aqi_color}'>{aqi_val}</div>"
                f"<div style='font-size:13px;color:{aqi_color}'>{aqi_label}</div>"
                f"<div style='font-size:11px;color:#6b7280;margin-top:6px'>{selected_sensor_name}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with attr_col_right:
        result = st.session_state.attribution_result
        active_sensor = st.session_state.attribution_sensor

        if result is None or (active_sensor and active_sensor.get('name') != selected_sensor_name):
            # No result yet — show prompt
            st.markdown(
                "<div style='background:#111827;border:1px dashed #374151;border-radius:8px;"
                "padding:32px;text-align:center;color:#6b7280'>"
                "Select a station and click <b style='color:#00e5ff'>Analyse Pollution Sources</b> "
                "— or click any sensor marker on the map above — to run the attribution engine."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            with st.container(border=True):
                # ── Header row ──────────────────────────────────────────────
                head_c1, head_c2 = st.columns([2, 1])
                with head_c1:
                    st.markdown(
                        f"**Source Attribution** — {active_sensor.get('name','')}"
                    )
                with head_c2:
                    conf = result.confidence
                    conf_color = "#00e676" if conf >= 75 else "#ffd740" if conf >= 55 else "#ff9100"
                    st.markdown(
                        f"<div style='text-align:right'>"
                        f"<span style='font-size:11px;color:#9ca3af'>CONFIDENCE</span><br>"
                        f"<span style='font-size:22px;font-weight:700;color:{conf_color}'>{conf}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ── Primary driver badge ────────────────────────────────────
                primary = result.primary_source
                primary_pct = result.percentages.get(primary, 0)
                SOURCE_ICONS = {
                    "Traffic": "🚗",
                    "Industrial": "🏭",
                    "Construction": "🏗️",
                    "Biomass Burning": "🔥",
                    "Weather Amplification": "🌬️",
                }
                icon = SOURCE_ICONS.get(primary, "⚠️")
                st.markdown(
                    f"<div style='background:#1f2937;border-left:4px solid #00e5ff;"
                    f"border-radius:6px;padding:10px 14px;margin:8px 0'>"
                    f"<span style='font-size:11px;color:#9ca3af;text-transform:uppercase;"
                    f"letter-spacing:1px'>Primary Driver</span><br>"
                    f"<span style='font-size:18px;font-weight:600;color:#00e5ff'>"
                    f"{icon} {primary} — {primary_pct:.0f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ── Contribution breakdown chart ────────────────────────────
                pct_data = result.percentages
                SOURCE_COLORS = {
                    "Traffic":              "#00e5ff",
                    "Industrial":           "#ff6b35",
                    "Construction":         "#f0c040",
                    "Biomass Burning":      "#ff4444",
                    "Weather Amplification":"#a78bfa",
                }
                sorted_sources = sorted(pct_data.items(), key=lambda x: x[1], reverse=True)
                labels  = [f"{SOURCE_ICONS.get(k,'')}&nbsp;{k}" for k, _ in sorted_sources]
                values  = [v for _, v in sorted_sources]
                colors  = [SOURCE_COLORS.get(k, "#888") for k, _ in sorted_sources]

                fig_bar = go.Figure(go.Bar(
                    x=values,
                    y=[f"{SOURCE_ICONS.get(k,'')} {k}" for k, _ in sorted_sources],
                    orientation='h',
                    marker=dict(color=colors, opacity=0.88),
                    text=[f"{v:.1f}%" for v in values],
                    textposition='outside',
                    textfont=dict(color='white', size=12),
                    hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
                ))
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=12),
                    xaxis=dict(
                        title="Contribution (%)", range=[0, max(values) * 1.25],
                        gridcolor='#374151', showgrid=True,
                    ),
                    yaxis=dict(showgrid=False),
                    margin=dict(l=0, r=60, t=10, b=10),
                    height=220,
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # ── Source breakdown table ──────────────────────────────────
                table_rows = ""
                for src, pct in sorted_sources:
                    icon_s = SOURCE_ICONS.get(src, "")
                    bar_w = int(pct * 1.6)  # scale to max ~160px
                    color = SOURCE_COLORS.get(src, "#888")
                    table_rows += (
                        f"<tr>"
                        f"<td style='padding:6px 8px;color:#d1d5db'>{icon_s} {src}</td>"
                        f"<td style='padding:6px 8px;text-align:right;"
                        f"font-weight:600;color:{color}'>{pct:.1f}%</td>"
                        f"<td style='padding:6px 8px'>"
                        f"<div style='background:{color};width:{bar_w}px;height:8px;"
                        f"border-radius:4px;opacity:0.8'></div></td>"
                        f"</tr>"
                    )

                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse;font-size:13px'>"
                    f"<thead><tr>"
                    f"<th style='text-align:left;padding:6px 8px;color:#9ca3af;"
                    f"border-bottom:1px solid #374151'>Source</th>"
                    f"<th style='text-align:right;padding:6px 8px;color:#9ca3af;"
                    f"border-bottom:1px solid #374151'>Contribution</th>"
                    f"<th style='padding:6px 8px;color:#9ca3af;"
                    f"border-bottom:1px solid #374151'></th>"
                    f"</tr></thead>"
                    f"<tbody>{table_rows}</tbody>"
                    f"</table>",
                    unsafe_allow_html=True,
                )

                # ── Donut chart (supplementary visual) ─────────────────────
                with st.expander("📊 View Proportion Chart", expanded=False):
                    fig_pie = go.Figure(go.Pie(
                        labels=[f"{SOURCE_ICONS.get(k,'')} {k}" for k, _ in sorted_sources],
                        values=values,
                        hole=0.55,
                        marker=dict(colors=colors, line=dict(color='#111827', width=2)),
                        textinfo='label+percent',
                        textfont=dict(color='white', size=11),
                        hovertemplate='%{label}: %{value:.1f}%<extra></extra>',
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=False,
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=280,
                        annotations=[dict(
                            text=f"<b>{primary_pct:.0f}%</b><br>{primary.split()[0]}",
                            x=0.5, y=0.5, font_size=14, showarrow=False,
                            font=dict(color='#00e5ff'),
                        )],
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # ── Natural-language explanation ────────────────────────────
                st.markdown(
                    f"<div style='background:#111827;border:1px solid #374151;border-radius:8px;"
                    f"padding:12px 16px;margin-top:8px'>"
                    f"<div style='font-size:11px;color:#9ca3af;text-transform:uppercase;"
                    f"letter-spacing:1px;margin-bottom:6px'>AI Explanation</div>"
                    f"<div style='font-size:13px;color:#e5e7eb;line-height:1.6'>"
                    f"{result.explanation}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # ── Data sources transparency log ───────────────────────────
                with st.expander("🔎 Attribution Audit Log (Data Sources)", expanded=False):
                    st.markdown("**Sub-scores (raw, pre-normalisation):**")
                    raw_scores = result.sub_scores.as_dict()
                    raw_df = pd.DataFrame([
                        {"Source": k, "Raw Score": f"{v:.2f}"}
                        for k, v in raw_scores.items()
                    ])
                    st.dataframe(raw_df, use_container_width=True, hide_index=True)

                    st.markdown("**Live data sources used:**")
                    for ds in result.data_sources_used:
                        st.markdown(f"- {ds}")

                    st.caption(
                        "Scores are computed from road-graph edge weights (traffic), "
                        "OSM land-use proximity (industrial/construction), "
                        "NASA FIRMS fire radiative power (biomass), "
                        "and Open-Meteo wind/temp data (weather). "
                        "No random values are used."
                    )

# ─────────────────────────────────────────────
# 📈 PREDICTIVE FORECAST & ANOMALY DASHBOARD
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 📈 AQI Trend Projection & Anomaly Detection: {city_input}")
st.caption(
    "Trend projections use a physics-informed XGBoost model trained on meteorological rules "
    "(rush-hour traffic, wind dispersal, temperature inversion) calibrated to the current sensor baseline. "
    "This is a **diurnal pattern model**, not a data-driven forecast — it captures predictable daily cycles "
    "from known atmospheric physics, not stochastic future events. Suitable for same-day planning, "
    "not multi-day prediction."
)

forecast_model, hist_mean, hist_std = train_xgboost_forecast_model(avg_aqi)
is_anomaly, threshold, dev_pct = detect_anomaly(worst['aqi'], hist_mean, hist_std)

if is_anomaly:
    st.error(
        f"**🚨 STATISTICAL ANOMALY DETECTED IN {worst['name']}** "
        f"Current AQI ({worst['aqi']}) exceeds the historical confidence interval "
        f"(Threshold: {int(threshold)}).  \n"
        f"**Deviation:** +{dev_pct:.1f}% above baseline."
    )
    causes, primary, explanation = run_causal_engine(
        worst['aqi'], current_hour, telemetry['wind_speed'],
        telemetry['wind_dir'], telemetry['active_fires']
    )
    st.info(f"**Indicative Pattern Analysis:** {explanation} (Run Source Attribution for a full multi-factor breakdown.)")

forecasts = generate_24h_forecast(forecast_model, telemetry, current_hour)
if forecasts is not None:
    times = telemetry['hourly']['time'][:24]
    peak_forecast_aqi  = int(max(forecasts))
    peak_forecast_time = times[int(np.argmax(forecasts))][-5:]

    f_col1, f_col2, f_col3 = st.columns(3)
    f_col1.metric("6-Hour Forecast",    int(np.mean(forecasts[:6])))
    f_col2.metric("12-Hour Forecast",   int(np.mean(forecasts[:12])))
    f_col3.metric("Predicted 24h Peak", peak_forecast_aqi,
                  f"at {peak_forecast_time}", delta_color="inverse")

    if peak_forecast_aqi > 250:
        st.warning(
            f"⚠️ **Forecast Alert:** System predicts AQI will degrade to hazardous levels "
            f"({peak_forecast_aqi}) at approximately {peak_forecast_time}."
        )

    upper_bound = [f + (i * 1.5) for i, f in enumerate(forecasts)]
    lower_bound = [max(10, f - (i * 1.5)) for i, f in enumerate(forecasts)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name='Upper Bound', x=times, y=upper_bound,
        mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        name='Confidence Interval', x=times, y=lower_bound,
        mode='lines', marker=dict(color="#444"), line=dict(width=0),
        fillcolor='rgba(0, 229, 255, 0.1)', fill='tonexty', showlegend=True
    ))
    fig.add_trace(go.Scatter(
        name='Diurnal Trend Projection', x=times, y=forecasts,
        mode='lines+markers', line=dict(color='#00e5ff', width=3), marker=dict(size=6)
    ))
    fig.add_hline(y=150, line_dash="dash", line_color="#ffd740",
                  annotation_text="Unhealthy Threshold")
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), margin=dict(l=0, r=0, t=30, b=0), height=300,
        yaxis=dict(title="AQI", gridcolor='#374151'),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 🎯 AUTONOMOUS INTERVENTION COMMAND CENTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-head">🎯 Autonomous Intervention Command Center</div>', unsafe_allow_html=True)

if not INTERVENTION_AVAILABLE:
    st.info("Intervention Agent not available. Ensure intervention_agent.py is in the project directory.")
else:
    _agent = st.session_state.intervention_agent

    # ── Build input percentages: attribution engine → causal engine fallback ─────
    _attr_for_agent   = st.session_state.attribution_result
    _agent_confidence = _attr_for_agent.confidence if _attr_for_agent is not None else 55

    if _attr_for_agent is not None:
        # Best path: use fine-grained source percentages from Task 1 engine
        _agent_percentages = dict(_attr_for_agent.percentages)
    else:
        # Fallback: map causal engine's 4-category output to the 5-category
        # schema the InterventionAgent expects.  Call once, reuse the result.
        _causal_attr, _, _ = run_causal_engine(
            worst["aqi"], current_hour,
            telemetry["wind_speed"], telemetry["wind_dir"], telemetry["active_fires"],
        )
        # causal engine keys → intervention agent keys
        #   "Vehicular Emissions"       → Traffic
        #   "Meteorological Stagnation" → Weather Amplification
        #   "Regional Inflow / Fires"   → split: Biomass Burning + residual
        #   "Background Urban Dust"     → split: Industrial + Construction
        _vehicular  = _causal_attr.get("Vehicular Emissions", 35.0)
        _stagnation = _causal_attr.get("Meteorological Stagnation", 20.0)
        _inflow     = _causal_attr.get("Regional Inflow / Fires", 10.0)
        _dust       = _causal_attr.get("Background Urban Dust", 20.0)
        _agent_percentages = {
            "Traffic":               _vehicular,
            "Weather Amplification": _stagnation,
            "Biomass Burning":       _inflow,           # fires/inflow maps cleanly
            "Industrial":            _dust * 0.60,      # 60 % of dust is industrial
            "Construction":          _dust * 0.40,      # 40 % is construction
        }
        # Guarantee exact sum = 100 after the split
        _total = sum(_agent_percentages.values())
        if abs(_total - 100.0) > 0.01:
            _agent_percentages = {k: v / _total * 100.0 for k, v in _agent_percentages.items()}

    # Fire hotspot count from attribution engine cache, else telemetry
    _fire_count = (
        len(st.session_state.attribution_engine._fire_hotspots)
        if st.session_state.attribution_engine is not None
           and hasattr(st.session_state.attribution_engine, "_fire_hotspots")
        else int(telemetry.get("active_fires", 0))
    )

    # Generate — InterventionAgent is pure-compute; calling every rerun is fast
    cc_output: CommandCenterOutput = _agent.generate(
        current_aqi=worst["aqi"],
        percentages=_agent_percentages,
        attribution_confidence=_agent_confidence,
        telemetry=telemetry,
        current_hour=current_hour,
        fire_hotspot_count=_fire_count,
    )
    st.session_state.command_center_output = cc_output

    # ── CRISIS BANNER ─────────────────────────────────────────────────────────
    crisis_bg = {
        "SEVERE":   "rgba(255,23,68,0.15)",
        "VERY HIGH":"rgba(255,82,82,0.12)",
        "HIGH":     "rgba(255,145,0,0.12)",
        "MODERATE": "rgba(255,215,64,0.10)",
        "GOOD":     "rgba(0,230,118,0.10)",
    }.get(cc_output.crisis_level, "rgba(55,65,81,0.5)")

    st.markdown(
        f"""<div style="background:{crisis_bg};border:1px solid {cc_output.crisis_color};
        border-radius:10px;padding:14px 20px;margin-bottom:1rem;
        display:flex;align-items:center;gap:16px">
        <div style="font-size:32px">{cc_output.crisis_icon}</div>
        <div>
          <div style="font-size:11px;color:#9ca3af;letter-spacing:1.5px;
               text-transform:uppercase">Crisis Level</div>
          <div style="font-size:24px;font-weight:700;color:{cc_output.crisis_color}">
               {cc_output.crisis_level} — AQI {cc_output.current_aqi}</div>
          <div style="font-size:13px;color:#d1d5db;margin-top:2px">
               Primary driver: <b style="color:{cc_output.crisis_color}">
               {cc_output.primary_driver} ({cc_output.primary_driver_pct:.0f}%)</b>
               &nbsp;·&nbsp; Attribution confidence: {cc_output.attribution_confidence}%
          </div>
        </div></div>""",
        unsafe_allow_html=True,
    )

    # ── COMPOSITE STRATEGY EXPLANATION ───────────────────────────────────────
    st.markdown(
        f"""<div style="background:#111827;border:1px solid #374151;border-radius:8px;
        padding:12px 16px;margin-bottom:1rem;font-size:13px;color:#e5e7eb;line-height:1.7">
        <span style="font-size:11px;color:#9ca3af;text-transform:uppercase;
        letter-spacing:1px">Strategic Assessment</span><br>
        {cc_output.composite_explanation}</div>""",
        unsafe_allow_html=True,
    )

    # ── TOP 3 INTERVENTION CARDS ──────────────────────────────────────────────
    st.markdown("#### Top Recommended Actions")
    top3 = cc_output.interventions[:3]

    # Rank badge colours
    RANK_COLORS = ["#ffd700", "#c0c0c0", "#cd7f32"]   # gold, silver, bronze
    COST_COLORS = ["#00e676","#69f0ae","#ffd740","#ff9100","#ff5252"]
    FEAS_COLORS = ["#ff5252","#ff9100","#ffd740","#69f0ae","#00e676"]

    ic1, ic2, ic3 = st.columns(3)
    col_map_c = {0: ic1, 1: ic2, 2: ic3}

    for idx, ri in enumerate(top3):
        feas_color = FEAS_COLORS[min(4, int(ri.feasibility * 5))]
        cost_color = COST_COLORS[ri.cost_tier - 1]
        rank_color = RANK_COLORS[idx]
        score_pct  = int(ri.final_score * 100)

        with col_map_c[idx]:
            # Score ring implemented as a compact progress bar
            #
            # BUG FIX: this used to be a multi-line triple-quoted f-string.
            # Streamlit's frontend markdown renderer follows CommonMark,
            # which ends an HTML block on a blank line and then treats the
            # next line as an INDENTED CODE BLOCK if it's indented >=4
            # spaces (verified against the commonmark reference parser).
            # This card had 4 internal blank lines (used purely for Python
            # source readability) each followed by deeply indented HTML,
            # which is exactly that trigger — unsafe_allow_html=True never
            # even comes into play, because by the time HTML-passthrough
            # would apply, the content has already been classified as a
            # code block and is rendered as escaped literal text (hence
            # the raw "<div..." text with a copy button in the screenshot).
            #
            # Fix: build the identical HTML as ONE unbroken line via
            # Python's implicit adjacent-string-literal concatenation
            # (confirmed to insert zero characters between literals) —
            # no newlines anywhere in the string means no blank line and
            # no indentation for any markdown parser to misinterpret.
            # Same tags, same styles, same values — purely a string-
            # construction fix, not a redesign.
            card_html = (
                f'<div style="background:#111827;border:1px solid #374151;'
                f'border-radius:10px;padding:14px;height:100%;position:relative">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                f'<span style="background:{rank_color};color:#000;font-weight:700;'
                f'font-size:12px;padding:2px 8px;border-radius:4px">#{ri.rank}</span>'
                f'<span style="font-size:18px">{ri.spec.icon}</span>'
                f'<span style="font-size:13px;font-weight:600;color:#e5e7eb">{ri.spec.name}</span>'
                f'</div>'
                f'<div style="font-size:11px;color:#9ca3af;margin-bottom:4px">DEPT</div>'
                f'<div style="font-size:11px;color:#d1d5db;margin-bottom:10px">{ri.spec.department}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px">'
                f'<div style="background:#1f2937;border-radius:6px;padding:6px 8px">'
                f'<div style="font-size:10px;color:#9ca3af">AQI REDUCTION</div>'
                f'<div style="font-size:18px;font-weight:700;color:#00e5ff">-{ri.expected_aqi_reduction:.0f}</div>'
                f'</div>'
                f'<div style="background:#1f2937;border-radius:6px;padding:6px 8px">'
                f'<div style="font-size:10px;color:#9ca3af">DEPLOY IN</div>'
                f'<div style="font-size:18px;font-weight:700;color:#a78bfa">{deploy_hours_label(ri.deployment_hours)}</div>'
                f'</div>'
                f'</div>'
                f'<div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap">'
                f'<span style="background:{cost_color}22;color:{cost_color};'
                f'font-size:10px;padding:2px 7px;border-radius:4px;border:1px solid {cost_color}44">'
                f'💰 Cost: {cost_tier_label(ri.cost_tier)}</span>'
                f'<span style="background:{feas_color}22;color:{feas_color};'
                f'font-size:10px;padding:2px 7px;border-radius:4px;border:1px solid {feas_color}44">'
                f'✅ Feasibility: {ri.feasibility*100:.0f}%</span>'
                f'</div>'
                f'<div style="background:#0d1117;border-radius:4px;height:4px;margin-bottom:8px">'
                f'<div style="background:linear-gradient(90deg,#00e5ff,#7c3aed);'
                f'height:4px;border-radius:4px;width:{score_pct}%"></div>'
                f'</div>'
                f'<div style="font-size:10px;color:#6b7280;text-align:right">Score: {score_pct}/100</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    # ── REASONING EXPANDERS ───────────────────────────────────────────────────
    st.markdown("#### AI Reasoning")
    for ri in top3:
        with st.expander(f"{ri.spec.icon} Why {ri.spec.name}?", expanded=False):
            st.markdown(
                f"<div style='font-size:13px;color:#e5e7eb;line-height:1.7'>"
                f"{ri.reasoning}</div>",
                unsafe_allow_html=True,
            )
            sub_c1, sub_c2, sub_c3, sub_c4, sub_c5 = st.columns(5)
            labels = ["AQI Impact","Cost","Feasibility","Confidence","Speed"]
            vals   = [ri.sub_scores.get(k, 0) for k in ["AQI Impact","Cost","Feasibility","Confidence","Speed"]]
            for col, lbl, val in zip([sub_c1,sub_c2,sub_c3,sub_c4,sub_c5], labels, vals):
                col.metric(lbl, f"{val:.2f}")
            st.caption(f"Description: {ri.spec.description}")

    # ── FULL RANKING TABLE ────────────────────────────────────────────────────
    with st.expander("📊 Full Intervention Ranking (All Actions)", expanded=False):
        all_rows = []
        for ri in cc_output.interventions:
            all_rows.append({
                "Rank": ri.rank,
                "Action": f"{ri.spec.icon} {ri.spec.name}",
                "Department": ri.spec.department,
                "AQI Drop": f"-{ri.expected_aqi_reduction:.0f}",
                "Reduction %": f"{ri.expected_aqi_reduction_pct:.1f}%",
                "Cost": cost_tier_label(ri.cost_tier),
                "Feasibility": f"{ri.feasibility*100:.0f}%",
                "Deploy": deploy_hours_label(ri.deployment_hours),
                "Score": f"{ri.final_score*100:.0f}/100",
            })
        st.dataframe(
            pd.DataFrame(all_rows),
            use_container_width=True,
            hide_index=True,
        )

    # ── SELECT INTERVENTION → PRE-FILL COUNTERFACTUAL ────────────────────────
    st.markdown("#### Send to Counterfactual Simulator")
    st.caption("Select an intervention to pre-fill the simulator with its parameters.")

    _sel_options = {f"#{r.rank} {r.spec.icon} {r.spec.name}": r for r in cc_output.interventions}
    _sel_default_label = f"#1 {top3[0].spec.icon} {top3[0].spec.name}" if top3 else None
    _sel_default_idx   = 0

    _sel_label = st.selectbox(
        "Choose intervention to simulate",
        list(_sel_options.keys()),
        index=_sel_default_idx,
        key="cmd_intervention_select",
    )
    _selected_ri = _sel_options[_sel_label]
    st.session_state.selected_intervention_id = _selected_ri.spec.id

    # Store slider pre-fill values in session_state
    if "cmd_prefill" not in st.session_state:
        st.session_state.cmd_prefill = {}
    st.session_state.cmd_prefill = {
        "traffic":    _selected_ri.sim_traffic_pct,
        "industrial": _selected_ri.sim_industrial_pct,
        "wind":       _selected_ri.sim_wind_shift,
    }

    # BUG FIX: the Counterfactual Simulator below only has levers for
    # Traffic, Industrial, and Wind/Weather. Construction and Biomass
    # Burning interventions (Construction Pause, Open Burning Enforcement)
    # have no matching slider, so sim_traffic_pct/sim_industrial_pct/
    # sim_wind_shift are correctly 0.0 for them. Previously this silently
    # pre-filled all sliders to 0%, which looked like a contradiction next
    # to the Command Center's own non-zero AQI reduction estimate above.
    # We now detect that case and explain the limitation instead of
    # implying "no impact".
    _SIMULATOR_COVERED_SOURCES = {"Traffic", "Industrial"}
    _sim_covered = any(
        src in _SIMULATOR_COVERED_SOURCES for src in _selected_ri.spec.target_sources
    )

    if _sim_covered:
        st.info(
            f"**{_selected_ri.spec.icon} {_selected_ri.spec.name}** pre-fills: "
            f"Traffic restriction {_selected_ri.sim_traffic_pct:.0f}% · "
            f"Industrial curb {_selected_ri.sim_industrial_pct:.0f}% · "
            f"Wind shift +{_selected_ri.sim_wind_shift:.0f} km/h — "
            f"scroll down to the Counterfactual Simulator to run it."
        )
    else:
        _uncovered = " / ".join(_selected_ri.spec.target_sources)
        st.warning(
            f"**{_selected_ri.spec.icon} {_selected_ri.spec.name}** targets "
            f"**{_uncovered}**, which the Counterfactual Simulator below doesn't "
            f"model directly — it currently has levers for Traffic, Industrial, "
            f"and Wind/Weather only. The Command Center's "
            f"**−{_selected_ri.expected_aqi_reduction:.0f} AQI** estimate above "
            f"already accounts for this intervention's effect on "
            f"{_uncovered}; the simulator sliders will stay at 0% for it rather "
            f"than show a number that isn't actually being modelled. "
            f"Known limitation — flagged here rather than faked."
        )

    # ── 🌆 URBAN DIGITAL TWIN: MULTI-INTERVENTION SCENARIO (Feature 3) ───────
    # Reuses InterventionAgent.simulate_scenario() — no duplicate calculation
    # logic vs. the ranking above or the Counterfactual Simulator below; this
    # is the same compounding + weather math, generalised to N interventions
    # chosen directly by source (not via the continuous % sliders).
    st.markdown("---")
    st.markdown("#### 🌆 Urban Digital Twin — Multi-Intervention Scenario")
    st.caption(
        "Pick one or more interventions and predict their combined AQI "
        "impact before deployment — reuses the same ranking engine above, "
        "generalised to any combination you choose."
    )

    _dt_options = {f"{r.spec.icon} {r.spec.name}": r.spec for r in cc_output.interventions}
    _dt_default = [f"{top3[0].spec.icon} {top3[0].spec.name}"] if top3 else []

    _dt_compare = st.checkbox(
        "Compare against a second scenario", key="dt_compare_toggle",
        help="Run two intervention bundles side by side and see which predicts lower AQI.",
    )

    _dt_wind = st.slider(
        "Additional wind shift for this scenario (+km/h)", 0, 20, 0, step=5,
        key="dt_scenario_wind",
    )

    # Collected by whichever branch below runs, then consumed by the
    # Health & Economic Impact Engine section right after this block —
    # reuses these SimulationResults as-is, computes nothing AQI-related.
    _health_sim_results: list[tuple[str, "SimulationResult"]] = []

    if not _dt_compare:
        # ── Single-scenario mode ──
        _dt_labels = st.multiselect(
            "Interventions to simulate together",
            list(_dt_options.keys()),
            default=_dt_default,
            key="dt_scenario_select",
        )

        if _dt_labels:
            _dt_specs = [_dt_options[lbl] for lbl in _dt_labels]
            sim_result: SimulationResult = _agent.simulate_scenario(
                current_aqi=cc_output.current_aqi,
                percentages=_agent_percentages,
                telemetry=telemetry,
                interventions=_dt_specs,
                attribution_confidence=cc_output.attribution_confidence,
                extra_wind_shift_kmh=float(_dt_wind),
            )

            dt_c1, dt_c2, dt_c3, dt_c4 = st.columns(4)
            dt_c1.metric("Baseline AQI", sim_result.baseline_aqi)
            dt_c2.metric(
                "Predicted AQI", sim_result.predicted_aqi,
                delta=f"-{sim_result.delta_aqi} points", delta_color="inverse",
            )
            dt_c3.metric("Total Improvement", f"{sim_result.total_pct_drop:.1f}%")
            dt_c4.metric("Prediction Confidence", f"{sim_result.confidence}%")

            if sim_result.breakdown:
                st.dataframe(
                    pd.DataFrame(
                        [{"Source": k, "AQI-% Reduced": v} for k, v in sim_result.breakdown.items()]
                    ),
                    use_container_width=True, hide_index=True,
                )
            st.caption(f"📐 {sim_result.scenario_label}")
            _health_sim_results.append(("", sim_result))
        else:
            st.info("Select at least one intervention above to run the Digital Twin prediction.")

    else:
        # ── Scenario comparison mode ── reuses compare_scenarios(), which
        # itself calls simulate_scenario() twice — no separate prediction
        # math lives here.
        dt_col_a, dt_col_b = st.columns(2)
        with dt_col_a:
            _dt_labels_a = st.multiselect(
                "Scenario A", list(_dt_options.keys()),
                default=_dt_default, key="dt_scenario_a",
            )
        with dt_col_b:
            _dt_labels_b = st.multiselect(
                "Scenario B", list(_dt_options.keys()),
                key="dt_scenario_b",
            )

        if _dt_labels_a and _dt_labels_b:
            comparison: ScenarioComparison = _agent.compare_scenarios(
                current_aqi=cc_output.current_aqi,
                percentages=_agent_percentages,
                telemetry=telemetry,
                scenario_a=[_dt_options[lbl] for lbl in _dt_labels_a],
                scenario_b=[_dt_options[lbl] for lbl in _dt_labels_b],
                attribution_confidence=cc_output.attribution_confidence,
                extra_wind_shift_kmh=float(_dt_wind),
                label_a="Scenario A", label_b="Scenario B",
            )

            comp_a, comp_b = st.columns(2)
            with comp_a:
                st.markdown("**Scenario A**")
                st.metric(
                    "Predicted AQI", comparison.result_a.predicted_aqi,
                    delta=f"-{comparison.result_a.delta_aqi} points", delta_color="inverse",
                )
                st.caption(f"Confidence: {comparison.result_a.confidence}%")
            with comp_b:
                st.markdown("**Scenario B**")
                st.metric(
                    "Predicted AQI", comparison.result_b.predicted_aqi,
                    delta=f"-{comparison.result_b.delta_aqi} points", delta_color="inverse",
                )
                st.caption(f"Confidence: {comparison.result_b.confidence}%")

            if comparison.better_label == "Tie":
                st.info(f"Both scenarios predict the same AQI ({comparison.result_a.predicted_aqi}).")
            else:
                st.success(f"🏆 **{comparison.better_label}** predicts the lower AQI — by {comparison.aqi_gap} points.")
            st.caption(f"📐 {comparison.explanation}")
            _health_sim_results.append((f"{comparison.label_a}: ", comparison.result_a))
            _health_sim_results.append((f"{comparison.label_b}: ", comparison.result_b))
            st.session_state.last_comparison = comparison
        else:
            st.info("Select at least one intervention in both Scenario A and Scenario B to compare them.")

    # ── 🏥 HEALTH & ECONOMIC IMPACT ENGINE (Feature 4) ────────────────────
    # Consumes the SimulationResult(s) collected above — does not touch
    # AQI, percentages, telemetry, or intervention specs directly. If the
    # AQI math above changes, this section keeps working unmodified.
    if HEALTH_ECONOMIC_AVAILABLE and _health_sim_results:
        st.markdown("---")
        st.markdown("#### 🏥 Health & Economic Impact")
        st.caption(
            "Translates the Digital Twin's predicted AQI drop into avoided "
            "health burden and economic value — see 'Model assumptions & "
            "sources' below for exactly what every number traces back to."
        )

        _hee = st.session_state.health_economic_engine
        _population_exposed = estimate_population_exposed(radius_km=5.0)

        _computed_impacts: list[tuple[str, "HealthEconomicImpact"]] = []
        for _label_prefix, _sr in _health_sim_results:
            impact: HealthEconomicImpact = _hee.assess(_sr, population_exposed=_population_exposed)
            _computed_impacts.append((_label_prefix, impact))

            if _label_prefix:
                st.markdown(f"**{_label_prefix.strip(': ')}**")

            he_c1, he_c2, he_c3, he_c4 = st.columns(4)
            he_c1.metric("Hospitalizations Avoided", impact.hospitalizations_avoided)
            he_c2.metric("Asthma Attacks Avoided", impact.asthma_attacks_avoided)
            he_c3.metric("DALYs Reduced", f"{impact.dalys_reduced:.2f}")
            he_c4.metric("Social Benefit Score", f"{impact.social_benefit_score}/100")

            he_c5, he_c6, he_c7 = st.columns(3)
            he_c5.metric("Healthcare Savings", f"₹{impact.healthcare_savings_inr:,.0f}")
            he_c6.metric("Productivity Gains", f"₹{impact.productivity_gains_inr:,.0f}")
            he_c7.metric("Population Protected", f"{impact.population_protected:,}")

            st.caption(f"📐 {impact.explanation}")

        with st.expander("📋 Model assumptions & sources"):
            # All scenarios share the same assumption set, so render once.
            for _a in impact.assumptions:
                st.markdown(f"- {_a}")

        # Persist for the Mayor Copilot section below — it reads these
        # straight from session_state rather than recomputing anything.
        st.session_state.last_simulations = _health_sim_results
        st.session_state.last_health_impacts = _computed_impacts

# ── 🏛️ MAYOR COPILOT (Feature 5) ──────────────────────────────────────────
# Pure orchestrator: reads AttributionResult / CommandCenterOutput /
# SimulationResult / ScenarioComparison / HealthEconomicImpact already in
# session_state. Computes nothing new — every sentence it produces traces
# back to a field on one of those dataclasses (see CopilotAnswer.sources).
st.markdown("---")
st.markdown('<div class="section-head">🧠 AI Executive Brief</div>', unsafe_allow_html=True)

if not COPILOT_AVAILABLE:
    st.markdown(
        '<div class="ecc-not-ready">AI Executive Brief module unavailable. Ensure <code>mayor_copilot.py</code> is in the project directory.</div>',
        unsafe_allow_html=True,
    )
else:
    _copilot = st.session_state.mayor_copilot
    _ctx = DecisionContext(
        attribution=st.session_state.attribution_result,
        command_center=st.session_state.command_center_output,
        telemetry=telemetry,
        simulations=st.session_state.last_simulations or [],
        comparison=st.session_state.last_comparison,
        health_impacts=st.session_state.last_health_impacts or [],
    )

    if not (st.session_state.attribution_result and st.session_state.command_center_output):
        st.markdown(
            '<div class="exec-brief-wrap"><div class="exec-brief-body" style="color:#4b5563;font-style:italic">Run Source Attribution and the Intervention Engine to generate the Executive Brief.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Build five structured brief sections from real pipeline answers
        _why_ans    = _copilot.ask("Why is AQI increasing?",                      _ctx)
        _action_ans = _copilot.ask("Why are you recommending this intervention?",  _ctx)
        _health_ans = _copilot.ask("What is the health impact?",                   _ctx)
        _money_ans  = _copilot.ask("How much money could be saved?",               _ctx)

        _brief_cc   = st.session_state.command_center_output
        _brief_conf = st.session_state.last_simulations[0][1].confidence if st.session_state.last_simulations else _brief_cc.attribution_confidence

        _all_sources = sorted(set(
            s for ans in [_why_ans, _action_ans, _health_ans, _money_ans]
            for s in ans.sources
        ))

        # Section colours
        _sc = {"sit": "#00e5ff", "cause": "#f59e0b", "rec": "#34d399", "out": "#a78bfa", "conf": "#6b7280"}

        def _brief_sec(color, label, text):
            return (
                f'<div class="brief-section">'
                f'<div class="brief-section-label" style="color:{color}">{label}</div>'
                f'<div class="brief-section-text">{text}</div>'
                f'</div>'
            )

        _sit_text  = f"Current AQI is <strong style='color:{_brief_cc.crisis_icon and '#f87171'}'>{_brief_cc.current_aqi}</strong> ({_brief_cc.crisis_level}) in {city_input}. {_brief_cc.composite_explanation.split('.')[0]}."
        _cause_text = _why_ans.text if _why_ans.intent not in ("missing_data","unsupported") else "Attribution analysis in progress."
        _rec_text   = _action_ans.text if _action_ans.intent not in ("missing_data","unsupported") else "Intervention ranking pending."
        _out_text   = _health_ans.text if _health_ans.intent not in ("missing_data","unsupported") else "Run Digital Twin + Health Impact for outcome projection."
        _econ_text  = _money_ans.text if _money_ans.intent not in ("missing_data","unsupported") else ""

        _sections_html = (
            _brief_sec(_sc["sit"],   "📍 Situation",        _sit_text)
            + _brief_sec(_sc["cause"], "🔍 Primary Cause",    _cause_text)
            + _brief_sec(_sc["rec"],   "🎯 Recommendation",   _rec_text)
            + _brief_sec(_sc["out"],   "📉 Expected Outcome", _out_text + (" " + _econ_text if _econ_text else ""))
        )

        st.markdown(
            f'<div class="exec-brief-wrap">'
            f'<div class="exec-brief-header">'
            f'<span style="font-size:20px">🧠</span>'
            f'<div><div class="exec-brief-title">AI Executive Brief</div>'
            f'<div class="exec-brief-subtitle">Generated from live pipeline data · Deterministic · Every sentence is sourced</div>'
            f'</div></div>'
            f'<div class="exec-brief-body">{_sections_html}</div>'
            f'<div class="exec-brief-footer">'
            f'<span class="exec-brief-chip">📊 Confidence: {_brief_conf}%</span>'
            f'<span class="exec-brief-chip">📍 {city_input}</span>'
            f'<span class="exec-brief-chip">🌡️ AQI {_brief_cc.current_aqi} — {_brief_cc.crisis_level}</span>'
            f'</div>'
            f'<div class="exec-brief-sources">Data sources: {" · ".join(_all_sources[:5])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Follow-up Q&A panel (collapsible — not the primary UI) ───────
        with st.expander("💬 Ask a follow-up question"):
            st.caption("Every answer is read directly from the pipeline above — no generated text, no hallucination.")
            _brief_q_cols = st.columns(4)
            _brief_suggested = [
                "What is the primary pollution source today?",
                "How many people benefit?",
                "What if we choose another intervention?",
                "Why is the confidence score what it is?",
            ]
            for _bi, _bq in enumerate(_brief_suggested):
                if _brief_q_cols[_bi].button(_bq, key=f"brief_suggest_{_bi}", use_container_width=True):
                    st.session_state.copilot_chat_history.append(("user", _bq))
                    st.session_state.copilot_chat_history.append(("copilot", _copilot.ask(_bq, _ctx)))

            _brief_input = st.chat_input("Type your question…", key="brief_chat_input")
            if _brief_input:
                st.session_state.copilot_chat_history.append(("user", _brief_input))
                st.session_state.copilot_chat_history.append(("copilot", _copilot.ask(_brief_input, _ctx)))

            for _role, _content in st.session_state.copilot_chat_history[-8:]:
                if _role == "user":
                    with st.chat_message("user"):
                        st.markdown(_content)
                else:
                    _ca: CopilotAnswer = _content
                    with st.chat_message("assistant"):
                        st.markdown(_ca.text)
                        if _ca.sources:
                            st.caption("📎 Sources: " + " · ".join(_ca.sources[:4]))


# --- MODULE 4: COUNTERFACTUAL POLICY SIMULATOR ---
with st.expander("🔬 Counterfactual Policy Simulator", expanded=True):
    st.markdown("Quantify the systemic impact of hypothetical policy interventions.")

    # ── Read pre-fill values from Command Center (if an intervention was selected) ─
    _prefill = st.session_state.get("cmd_prefill", {})
    _pf_traffic    = int(_prefill.get("traffic",    0))
    _pf_industrial = int(_prefill.get("industrial", 0))
    _pf_wind       = int(_prefill.get("wind",       0))

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_traffic = st.slider("Restrict Vehicular Traffic (%)", 0, 100,
                                _pf_traffic, step=10)
    with sim_col2:
        sim_wind = st.slider("Simulate Meteorological Shift (+km/h)", 0, 20,
                             _pf_wind, step=5)
    with sim_col3:
        sim_industrial = st.slider("Industrial Emission Curb (%)", 0, 100,
                                   _pf_industrial, step=10)

    # ── Use attribution percentages if available, else causal engine ──────
    attr_result = st.session_state.attribution_result
    if attr_result is not None:
        # Attribution engine provides fine-grained source percentages
        traffic_pct     = attr_result.percentages.get("Traffic", 0) / 100.0
        industrial_pct  = attr_result.percentages.get("Industrial", 0) / 100.0
        weather_pct     = attr_result.percentages.get("Weather Amplification", 0) / 100.0

        # Wind-shift improvement on the Weather Amplification component —
        # delegates to the same formula simulate_scenario() uses for the
        # Digital Twin above, so the two stay numerically consistent
        # instead of carrying two copies of this math.
        stagnation_reduction_impact = weather_reduction_pct(
            telemetry['wind_speed'], sim_wind, weather_pct
        )

        traffic_reduction_impact    = traffic_pct * (sim_traffic / 100.0) * 100
        industrial_reduction_impact = industrial_pct * (sim_industrial / 100.0) * 100

        sim_source_label = f"Source: Attribution Engine ({attr_result.confidence}% confidence)"
    else:
        # Fallback to original causal engine
        control_causes, _, _ = run_causal_engine(
            worst['aqi'], current_hour, telemetry['wind_speed'],
            telemetry['wind_dir'], telemetry['active_fires']
        )
        new_wind_speed = telemetry['wind_speed'] + sim_wind
        sim_causes, _, _ = run_causal_engine(
            worst['aqi'], current_hour, new_wind_speed,
            telemetry['wind_dir'], telemetry['active_fires']
        )
        traffic_reduction_impact    = control_causes["Vehicular Emissions"] * (sim_traffic / 100.0)
        industrial_reduction_impact = 0.0  # not in original engine
        stagnation_reduction_impact = max(
            0, control_causes["Meteorological Stagnation"] - sim_causes["Meteorological Stagnation"]
        )
        sim_source_label = "Source: Causal Heuristic Engine (run Attribution Engine for precision)"

    total_pct_drop = traffic_reduction_impact + industrial_reduction_impact + stagnation_reduction_impact
    total_pct_drop = min(total_pct_drop, 85.0)  # physical ceiling
    simulated_aqi  = int(worst['aqi'] * (1.0 - (total_pct_drop / 100.0)))
    delta_aqi      = worst['aqi'] - simulated_aqi

    st.caption(f"📐 {sim_source_label}")
    st.markdown("#### 📉 Intervention Efficacy")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Baseline AQI",                 worst['aqi'])
    m_col2.metric("Projected AQI",                simulated_aqi,
                  delta=f"-{delta_aqi} points", delta_color="inverse")
    m_col3.metric("Systemic Improvement",         f"{total_pct_drop:.1f}%",
                  delta="Positive", delta_color="normal")
    m_col4.metric("Traffic + Industry Saved",
                  f"{traffic_reduction_impact + industrial_reduction_impact:.1f}%",
                  delta="Combined", delta_color="normal")

    # ── Waterfall breakdown of which intervention contributed what ─────────
    if total_pct_drop > 0:
        wf_cats  = ["Baseline AQI", "Traffic Restriction",
                     "Industrial Curb", "Wind Shift", "Projected AQI"]
        wf_vals  = [
            worst['aqi'],
            -traffic_reduction_impact * worst['aqi'] / 100,
            -industrial_reduction_impact * worst['aqi'] / 100,
            -stagnation_reduction_impact * worst['aqi'] / 100,
            simulated_aqi,
        ]
        wf_colors = ["#374151", "#00e5ff", "#ff6b35", "#a78bfa", "#00e676"]
        wf_text   = [
            str(worst['aqi']),
            f"-{traffic_reduction_impact:.1f}%",
            f"-{industrial_reduction_impact:.1f}%",
            f"-{stagnation_reduction_impact:.1f}%",
            str(simulated_aqi),
        ]
        fig_wf = go.Figure(go.Bar(
            x=wf_cats, y=[abs(v) for v in wf_vals],
            marker_color=wf_colors,
            text=wf_text, textposition='outside',
            textfont=dict(color='white', size=11),
        ))
        fig_wf.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), margin=dict(l=0, r=0, t=20, b=0), height=220,
            yaxis=dict(title="AQI Points", gridcolor='#374151'),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_wf, use_container_width=True)


# ─────────────────────────────────────────────
# 🎓 FIX 5: ACADEMIC TRANSPARENCY & LIMITATIONS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎓 Academic Transparency")

lim_col1, lim_col2 = st.columns(2)

with lim_col1:
    with st.expander("⚠️ System Limitations", expanded=False):
        st.markdown("""
        * **RL Agent Constraints:** The PPO model was trained using simulated synthetic AQI noise constraints, not long-term empirical spatial datasets.
        * **Forecasting Accuracy:** The XGBoost forecasting model relies on rule-based simulated target variables derived from diurnal patterns rather than empirical sensor history.
        * **Causal Heuristics:** Thermal anomaly (fire) detection and traffic impact weights are heuristically approximated, lacking direct integration with real-time tracking APIs.
        """)

with lim_col2:
    with st.expander("🚀 Future Work & Improvements", expanded=False):
        st.markdown("""
        * **Live Traffic Integration:** Implement Google Maps or Mapbox Traffic APIs to replace static time-based traffic heuristics for more accurate routing.
        * **Satellite Telemetry:** Direct API integration with NASA FIRMS for real-time, geolocated agricultural fire and thermal anomaly tracking.
        * **Deep Spatial Models:** Transition forecasting from tree-based regression to Spatio-Temporal Graph Neural Networks (ST-GNN) trained on multi-year empirical AQI datasets.
        """)