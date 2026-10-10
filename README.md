<div align="center">

# AirTwin X

**A Streamlit decision-support prototype for urban air-quality interventions**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status: prototype](https://img.shields.io/badge/status-prototype-orange?style=flat-square)](#project-status)

</div>

AirTwin X takes a city's current air-quality reading and walks it through a
pipeline: estimate which sources are driving the pollution, rank a fixed set
of municipal interventions, simulate what combining them would do, and
translate the predicted improvement into health and economic estimates. A
rule-based "Mayor Copilot" answers questions using only the outputs of those
earlier steps.

> **Screenshots / demo GIF:** not yet captured. Run it locally with the
> [Quick Start](#quick-start) to see the interface.

---

## Contents

- [The idea](#the-idea)
- [What it does](#what-it-does)
- [System architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Data sources and configuration](#data-sources-and-configuration)
- [Testing](#testing)
- [Project status](#project-status)
- [Known limitations](#known-limitations)
- [Documentation](#documentation)

---

## The idea

Most city AQI dashboards report a number and a generic warning, then stop.
They describe the problem but don't help anyone decide what to do about it.
Simply wrapping a large language model around a sensor feed doesn't fix that
either: a generative model can produce confident but untraceable impact
estimates, and will often add intervention effects together as if they were
independent.

AirTwin X treats the problem as a **decision pipeline** instead:

- When two interventions target the same pollution source, their effects
  **compound** rather than add: the remaining share of a source is
  `Π (1 − effectiveness_i)`, and the combined reduction is capped.
- Every number the Copilot reports comes from an upstream module's output.
  It does not generate free-form text from a language model.

## What it does

| Module | File | What is implemented |
|---|---|---|
| **Source attribution** | `attribution_engine.py` | Splits current AQI into Traffic, Construction, Industrial, Biomass Burning and Weather Amplification percentages using OpenStreetMap road data, weather telemetry and fire-hotspot data, with a confidence score and the list of data sources actually used. |
| **Intervention ranking** | `intervention_agent.py` | Scores 8 predefined interventions with a weighted multi-criteria formula (AQI impact 0.40, cost 0.20, feasibility 0.20, confidence 0.10, speed 0.10), a simplified TOPSIS-style weighting. |
| **Scenario simulation ("digital twin")** | `intervention_agent.py` | `simulate_scenario()` / `compare_scenarios()` apply the overlap-compounding rule above to one or more selected interventions. |
| **Health & economic impact** | `health_economic_engine.py` | Converts a predicted PM2.5 reduction into estimated admissions avoided, DALYs and cost savings. Deterministic, with its assumptions listed in the output. |
| **Mayor Copilot** | `mayor_copilot.py` | Deterministic keyword-based intent matching over the outputs above. No LLM. |
| **Dashboard** | `app.py` | Streamlit UI with Folium maps and Plotly charts, an hourly AQI projection from an XGBoost model trained on synthetic data (see limitations), and route planning that uses a PPO agent if a trained model is present, falling back to Dijkstra otherwise. |

## System architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                           │
│  [Open-Meteo]   [WAQI API]   [NASA FIRMS]   [OpenStreetMap]     │
└────────┬─────────────┬─────────────┬──────────────┬─────────────┘
         ▼             ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 GEOSPATIAL SOURCE ATTRIBUTION                   │
│         (Traffic, Industrial, Construction, Biomass)            │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  INTERVENTION RANKING (MCDA)                    │
│            (Impact, Cost, Feasibility, Confidence, Speed)       │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SCENARIO SIMULATION                          │
│              (Overlap-compounding, capped reduction)            │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                HEALTH & ECONOMIC IMPACT ENGINE                  │
│        (Admissions avoided, DALYs, estimated savings)           │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│             MAYOR COPILOT & COMMAND CENTER (UI)                 │
│           (Rule-based answers grounded in the above)            │
└─────────────────────────────────────────────────────────────────┘
```

See [Architecture.md](Architecture.md) for each module's inputs, outputs and
design decisions.

## Quick Start

```bash
git clone https://github.com/Soham-o/AirTwin-X.git
cd AirTwin-X
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`. New Delhi is selected by default.

`requirements.txt` includes `torch` and `stable-baselines3`, so the first
install is large. A `.devcontainer/` configuration is also included.

## Data sources and configuration

| Source | Used for | Configuration |
|---|---|---|
| [WAQI](https://aqicn.org/api/) | Live sensor readings | Optional `WAQI_TOKEN` in `.streamlit/secrets.toml` or as an environment variable. Without it, the app shows a warning and uses **simulated** sensor data. |
| [Open-Meteo](https://open-meteo.com/) | Wind and temperature | No key needed. |
| OpenStreetMap (via `osmnx`) | Road network for traffic attribution | No key needed. `new_delhi_5km.graphml` is included as a local graph. |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/) | Fire hotspots for biomass attribution | Optional `NASA_FIRMS_MAP_KEY` environment variable. Without a key, the app skips the request and uses the Open-Meteo-based heuristic. |

Never commit `.streamlit/secrets.toml` or real API tokens.

## Testing

```bash
pip install pytest numpy requests
python -m pytest test_airtwin.py -q
```

`test_airtwin.py` contains 29 tests covering determinism, intervention
ranking weights, overlap-compounding bounds, CPCB AQI→PM2.5 breakpoints,
health-impact monotonicity, Copilot intent routing, and missing/configured
NASA FIRMS key behavior. They need only
`numpy`, `requests` and `pytest`, not the full Streamlit stack. There is no
CI workflow yet.

## Project status

**Prototype.** The pipeline modules and their tests run locally. Outputs are
model-based estimates for exploring policy options, not validated forecasts,
and should not be used for real public-health or budget decisions.

## Known limitations

- **NASA FIRMS data is optional.** Set `NASA_FIRMS_MAP_KEY` in the environment to enable fire-hotspot requests. Without it, the app skips the request and uses the documented Open-Meteo-based heuristic. Do not commit the key.
- **No trained RL model is committed.** `app.py` looks for a
  `clean_air_agent` PPO model; without it, routing uses Dijkstra. Train one
  with `train_agent.py`.
- **The XGBoost AQI projection is trained on synthetic data** generated in
  `app.py` (daily cycles plus noise around the current reading), not on
  historical sensor data.
- **Attribution weights and intervention effectiveness values are fixed
  assumptions** in the source code, not values fitted to measured data.
- **Fallbacks are silent in places.** If WAQI is unavailable the dashboard
  switches to simulated data with a warning; check the data-source labels in
  the UI before trusting a number.
- No license file has been added yet.

## Documentation

- [Architecture.md](Architecture.md) — module responsibilities and design decisions
- [Demo guide.md](Demo%20guide.md) — a five-minute walkthrough of the interface
