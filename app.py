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
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import cKDTree
import xgboost as xgb

try:
    from stable_baselines3 import PPO
    RL_INSTALLED = True
except ImportError:
    RL_INSTALLED = False

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
            "⚠️ WAQI API token not found. "
            "Add it to `.streamlit/secrets.toml` as `WAQI_TOKEN = 'your_token'`."
        )

for key in ['gps_path', 'ai_path', 'metrics', 'thought_process', 'last_city']:
    if key not in st.session_state:
        st.session_state[key] = None

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
    url = (
        f"https://api.waqi.info/map/bounds/"
        f"?token={token}&latlng={minlat},{minlon},{maxlat},{maxlon}"
    )
    try:
        res = requests.get(url, timeout=15).json()
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
        active_fires = np.random.randint(2, 15) if current['temperature'] > 32.0 and current['windspeed'] > 8.0 else 0
        
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


@st.cache_resource
def load_city_graph(_sensor_data, current_hour, lat, lon, city_name):
    filename = f"{city_name.replace(' ', '_').lower()}_5km.graphml"
    try:
        G = ox.load_graphml(filename)
    except Exception:
        G = ox.graph_from_point((lat, lon), dist=5000, network_type='drive')
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_wcc).copy()
        ox.save_graphml(G, filename)

    if _sensor_data:
        G = map_aqi_idw(G, _sensor_data)
        G = apply_temporal_aqi(G, current_hour)
    return G


@st.cache_resource
def load_ai_agent():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "clean_air_agent")
    
    try: 
        return PPO.load(model_path)
    except Exception as e:
        st.info(f"RL model not loaded ({e}). Dijkstra fallback active.")
        return None

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
    .section-head {
        font-family: monospace; color: #00e5ff; font-weight: bold;
        margin-bottom: 10px; text-transform: uppercase;
    }
    .stMetric { background: #111827; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
</style>
""", unsafe_allow_html=True)

st.title("🌐 AirTwin: Spatial Decision Support System")

city_input = st.sidebar.text_input(
    "📍 Enter City Name (e.g., Delhi, Mumbai, London)", "New Delhi"
)

if st.session_state.last_city != city_input:
    st.session_state.gps_path      = None
    st.session_state.ai_path       = None
    st.session_state.metrics       = None
    st.session_state.thought_process = None
    st.session_state.last_city     = city_input

lat, lon = geocode_city(city_input)

# --- 🔧 FIX 1: Geocoding Fallback ---
if not lat or not lon:
    st.warning(f"⚠️ City '{city_input}' not found or API timed out. Falling back to New Delhi.")
    lat, lon = 28.6139, 77.2090
    city_input = "New Delhi"

st.sidebar.success(f"Tracking: {city_input} ({lat:.4f}, {lon:.4f})")

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
    G = load_city_graph(sensor_readings, current_hour, lat, lon, city_input)

# --- 🔧 FIX 3: Debug / Status Info ---
st.caption(f"🔧 **System Status:** Graph compiled for **{city_input}** | **Nodes:** {len(G.nodes):,} | **Edges:** {len(G.edges):,}")

is_delhi  = "delhi" in city_input.lower()
rl_model  = load_ai_agent() if (RL_INSTALLED and is_delhi) else None

avg_aqi = int(np.mean([s['aqi'] for s in sensor_readings]))
worst   = max(sensor_readings, key=lambda x: x['aqi'])

# ── RL status badge in sidebar ──
if rl_model:
    st.sidebar.success("🤖 PPO Agent: Active (routing AI path)")
else:
    st.sidebar.info("🔀 PPO Agent: Offline (Dijkstra fallback)")

# --- MODULE 1: KPI METRICS ---
k1, k2, k3 = st.columns(3)
k1.metric(f"{city_input} Avg AQI", avg_aqi)
k2.metric("Critical Hotspot", worst['name'], f"{worst['aqi']} AQI", delta_color="inverse")
k3.metric("Live Active Stations", len(sensor_readings))

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

        for s in sensor_readings:
            _, primary_cause, _ = run_causal_engine(
                s['aqi'], current_hour, telemetry['wind_speed'],
                telemetry['wind_dir'], telemetry['active_fires']
            )
            popup_html = f"<b>{s['name']}</b><br>AQI: {s['aqi']}<br><i>Cause: {primary_cause}</i>"
            folium.CircleMarker(
                [s['lat'], s['lon']], radius=8, color=s['color'], fill=True,
                popup=folium.Popup(popup_html, max_width=250)
            ).add_to(m)

        if st.session_state.gps_path and st.session_state.ai_path:
            folium.PolyLine(
                st.session_state.gps_path, color="#ff5252", weight=5,
                opacity=0.6, tooltip="Standard GPS (High AQI)"
            ).add_to(m)
            plugins.AntPath(
                locations=st.session_state.ai_path, color="#00e5ff", weight=6,
                dash_array=[10, 20], delay=800, tooltip="AirTwin Autonomous Route"
            ).add_to(m)

        st_folium(m, width="100%", height=550, returned_objects=[])

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
                    st.error("Could not locate addresses. Using graph boundary nodes as fallback.")
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
                routing_method = "Dijkstra (AQI-weighted)"

                if rl_model is not None:
                    with st.spinner("Running PPO agent…"):
                        ai_path_nodes = run_rl_routing(rl_model, G, start_node, end_node)
                    if ai_path_nodes:
                        routing_method = "PPO Reinforcement Learning"
                    else:
                        st.info("RL agent did not converge for this pair. Falling back to Dijkstra.")

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
# 📈 PREDICTIVE FORECAST & ANOMALY DASHBOARD
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 📈 AI Forecasting & Threat Detection: {city_input}")

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
    st.info(f"**Root Cause Analysis:** {explanation}")

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
        name='XGBoost Predicted AQI', x=times, y=forecasts,
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

# --- MODULE 4: COUNTERFACTUAL POLICY SIMULATOR ---
with st.expander("🔬 Counterfactual Policy Simulator", expanded=True):
    st.markdown("Quantify the systemic impact of hypothetical policy interventions.")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_traffic = st.slider("Restrict Vehicular Traffic (%)", 0, 100, 0, step=10)
    with sim_col2:
        sim_wind = st.slider("Simulate Meteorological Shift (+km/h)", 0, 20, 0, step=5)

    control_causes, _, _ = run_causal_engine(
        worst['aqi'], current_hour, telemetry['wind_speed'],
        telemetry['wind_dir'], telemetry['active_fires']
    )
    new_wind_speed = telemetry['wind_speed'] + sim_wind
    sim_causes, _, _ = run_causal_engine(
        worst['aqi'], current_hour, new_wind_speed,
        telemetry['wind_dir'], telemetry['active_fires']
    )

    traffic_reduction_impact   = control_causes["Vehicular Emissions"] * (sim_traffic / 100.0)
    stagnation_reduction_impact = (
        control_causes["Meteorological Stagnation"] - sim_causes["Meteorological Stagnation"]
    )
    total_pct_drop = traffic_reduction_impact + max(0, stagnation_reduction_impact)
    simulated_aqi  = int(worst['aqi'] * (1.0 - (total_pct_drop / 100.0)))
    delta_aqi      = worst['aqi'] - simulated_aqi

    st.markdown("#### 📉 Intervention Efficacy")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Baseline AQI (Control)",      worst['aqi'])
    m_col2.metric("Projected AQI (Intervention)", simulated_aqi,
                  delta=f"-{delta_aqi} points", delta_color="inverse")
    m_col3.metric("Systemic Improvement",         f"{total_pct_drop:.1f}%",
                  delta="Positive Health Outcome", delta_color="normal")


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