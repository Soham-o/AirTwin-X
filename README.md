# AirTwin X — AI-Powered Urban Intervention Operating System

**ET AI Hackathon 2.0 Submission**

> Traditional AQI dashboards tell you *what* the pollution level is.  
> AirTwin X tells you *why* it is happening, *what to do about it*, and *what will happen if you act* — all in one integrated AI decision pipeline.

---

## Problem Statement

Urban air quality crises in Indian cities — particularly Delhi NCT — cause thousands of preventable hospitalizations and billions in economic losses annually. City administrators currently lack tools to:

- Attribute pollution to specific, actionable sources in real time
- Predict the outcome of interventions *before* deploying them
- Quantify the health and economic benefit of each option
- Explain AI recommendations in plain language to non-technical decision-makers

## Solution

AirTwin X is a six-layer AI decision intelligence platform that transforms raw sensor data into a ranked, explainable intervention plan — complete with health impact projections, economic value estimates, and a natural-language AI advisor.

---

## Architecture

```
Weather API · WAQI · NASA FIRMS · OpenStreetMap
                    ↓
     Feature 1 — Geospatial Source Attribution Engine
     (Traffic / Construction / Industrial / Biomass / Weather)
                    ↓
     Feature 2 — Autonomous Intervention Command Engine
     (8 interventions, ranked by AQI impact · cost · feasibility · speed)
                    ↓
     Feature 3 — Urban Digital Twin Simulator
     (Single & multi-intervention scenario comparison)
                    ↓
     Feature 4 — Health & Economic Impact Engine
     (Hospitalisations avoided · DALYs · ₹ healthcare savings · ₹ productivity)
                    ↓
     Feature 5 — Mayor Copilot (AI Decision Support)
     (Deterministic, grounded answers from pipeline outputs)
                    ↓
     Feature 6 — Executive Command Center
     (30-second situational briefing for city administrators)
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit |
| Road network | OSMnx · NetworkX |
| RL routing agent | Stable-Baselines3 (PPO) · Gymnasium |
| Intervention ranking | XGBoost · custom scoring engine |
| Geospatial | Folium · GeoPy |
| Visualisation | Plotly · Folium |
| Data sources | WAQI API · NASA FIRMS · OpenWeatherMap |
| Backend | Python 3.11 · pure dataclasses (no database) |

---

## AI Pipeline Detail

### Feature 1 — Source Attribution Engine (`attribution_engine.py`)
- Gaussian plume dispersion model for industrial sources
- OSM road-load proxy for traffic attribution
- NASA FIRMS FRP for biomass burning
- Wind-stagnation index for weather amplification
- Confidence score from data-source coverage × recency

### Feature 2 — Intervention Command Engine (`intervention_agent.py`)
- Library of 8 India-specific interventions (GRAP-aligned)
- 5-factor weighted ranking: AQI impact (40%) · cost (20%) · feasibility (20%) · confidence (10%) · speed (10%)
- Context-aware feasibility adjustment (rush hour, fire count, wind, weekday)
- Zero randomness — every score is a deterministic function of named inputs

### Feature 3 — Urban Digital Twin (`intervention_agent.py` — `simulate_scenario()`)
- Overlap-compounding formula: Π(1 − effectiveness_i) per source
- Prevents double-counting when multiple interventions target the same source
- Supports N-intervention scenarios and A/B scenario comparison
- Confidence discounts 3%/intervention stacked beyond the first

### Feature 4 — Health & Economic Impact Engine (`health_economic_engine.py`)
- AQI→PM2.5 via official CPCB breakpoints (2014)
- Concentration-response coefficients from peer-reviewed meta-analyses
- NSS 75th Round hospitalization rates and costs (MOSPI, 2017-18)
- Human-capital productivity: Delhi per-capita income ₹493,024/year (NCT Delhi, 2024-25)
- Every assumption documented and surfaced in the UI

### Feature 5 — Mayor Copilot (`mayor_copilot.py`)
- Deterministic intent-matching (not an LLM) — no hallucination risk
- 8 supported question types; every answer cites upstream module fields
- Fails gracefully: explicitly states when upstream data is not yet available

### Feature 6 — Executive Command Center (`app.py`)
- Reads exclusively from session_state — zero new calculations
- Pipeline completion tracker (which modules have run)
- Threat-level banner, current vs predicted AQI comparison
- 6 KPI tiles, Mayor Copilot auto-summary
- Architecture diagram, Who Benefits panel

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/airtwin-x.git
cd airtwin-x

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API token (optional — falls back to mock data without it)
mkdir -p .streamlit
echo 'WAQI_TOKEN = "your_token_here"' > .streamlit/secrets.toml

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501` and runs immediately — all data falls back to physics-inspired mock values if the WAQI token is absent.

---

## Configuration

| Config | Method | Required? |
|---|---|---|
| WAQI API token | `.streamlit/secrets.toml` → `WAQI_TOKEN` | Optional — mock fallback |
| Trained PPO agent | Place `clean_air_agent.zip` in project root | Optional — Dijkstra fallback |
| City name | Sidebar text input | Default: New Delhi |

**Get a free WAQI token:** https://aqicn.org/data-platform/token/

---

## Demo Workflow (5 minutes)

1. **Open the app** — hero section and Executive Command Center greet the judge
2. **Scroll to Attribution Engine** → click "Run Source Attribution" — watch the pipeline tracker light up
3. **Read the Intervention Command Center** — top 3 ranked actions with reasoning
4. **Select the top intervention in the Digital Twin** → observe predicted AQI drop
5. **Check Health & Economic Impact** — hospitalisations, DALYs, rupees saved
6. **Ask Mayor Copilot** — type "Why are you recommending this?" or "How many people benefit?"
7. **Scroll back to the top** — Executive Command Center now shows the complete decision story

See `DEMO_GUIDE.md` for the full 5-minute script with talking points.

---

## Files

```
app.py                     # Main Streamlit dashboard (all 6 features integrated)
attribution_engine.py      # Feature 1: Source Attribution
intervention_agent.py      # Features 2 & 3: Intervention Engine + Digital Twin
health_economic_engine.py  # Feature 4: Health & Economic Impact
mayor_copilot.py           # Feature 5: AI Decision Support Copilot
train_agent.py             # RL routing agent training script (PPO)
new_delhi_5km.graphml      # Cached OSM road network (New Delhi, 5 km radius)
requirements.txt           # Python dependencies
README.md                  # This file
ARCHITECTURE.md            # Technical deep-dive
DEMO_GUIDE.md              # 5-minute live demo script
```

---

## Future Improvements

- Real-time Kafka stream for continuous AQI updates (replace 5-min polling)
- CPCB API integration for authoritative ground-truth sensor data
- MobileNetV3 CV model for camera-based haze estimation
- GBD-calibrated DALY weights (replace planning proxies)
- Multi-city support with city-specific intervention libraries
- Bayesian fusion for dynamic confidence weight learning
- REST API wrapper for each engine (SCADA/Smart City NOC integration)

---

## Known Limitations

- PM2.5 assumed to be the dominant AQI sub-index (correct for Delhi ~95% of days)
- Respiratory/cardiovascular hospitalisation shares are planning estimates, not local registry data
- DALY proxies are order-of-magnitude; not GBD disability-weight calibrated
- PPO routing agent trained on synthetic AQI — requires local fine-tuning for production
- Economic values use 2017-18 NSS costs; inflation-adjusted figures recommended for production

---

## License

MIT — see `LICENSE`