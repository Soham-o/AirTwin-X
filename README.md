<div align="center">
  <pre>
   ___   _       _____       _         _  __ 
  / _ \ (_)     |_   _|_ _ _(_)_ __   | |/ / 
 | |_| || |       | | \ V  V / | '_ \  | ' <  
 |_|_|_||_|       |_|  \_/\_/|_| |_|_ |_|\_\ 
  __ _ | |__  _ __  ___  _ __| |_| |_| (_)/ /_ 
 / _` || '_ \| '_ \/ _ \| '__| __| __| | | '_  | (_| || |_) | |_) | (_) | |  | |_| |_| | | | |
  \__,_||_.__/| .__/ \___/|_|   \__|\__|_|_|_| |_|
              |_|                               
  </pre>
  <h3>AI-Powered Urban Intervention Operating System</h3>
  <p><i>Because knowing the AQI number is not the same as knowing what to do about it.</i></p>
  
  [![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://airtwin-x.streamlit.app)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  [![Built for MIT Hackathon](https://img.shields.io/badge/MIT%20Hackathon-Grand%20Prize%20Design-blueviolet)](https://github.com/Soham-o/AirTwin-X)
</div>

---

## The Story of AirTwin X

### What problem existed?
Every major metropolis faces an environmental chasm. Open any municipal Air Quality Index (AQI) dashboard right now: you will find a number, a color-coded warning, and a passive recommendation to "avoid outdoor activities." Traditional air quality infrastructure is fundamentally diagnostic. It leaves city administrators blind to critical operational realities: Which specific source is actively driving the spike today? If we have a limited budget, which municipal intervention yields the maximum drop in PM2.5?

### Why existing solutions failed?
Previous attempts to solve this typically fall into two traps. First, standard dashboards are passive—they observe the crisis but offer no actionable mechanics to solve it. Second, recent "AI for AQI" projects simply wrap a Large Language Model around a raw sensor feed. These models confidently hallucinate intervention impacts, blindly add percentages together without respecting physical constraints, and fail to provide traceable, mathematically rigorous reasoning for civic resource allocation.

### What insight led to the idea?
AirTwin X was built on a core realization: **a city needs a decision pipeline, not just an observation deck.** We recognized that simulating interventions requires strict non-linear mathematics—if two policies target traffic, their impacts overlap and compound; they do not simply add up. Furthermore, we realized that an AI assistant for a city Mayor cannot be generative; it must be deterministic. Every metric, citation, and recommendation must be inextricably linked to verifiable epidemiological and economic data.

### What was built?
We engineered an AI-powered Urban Intervention Operating System. It is a fully decoupled chain of six stateless Python micro-modules orchestrated through a low-latency Streamlit interface. 

1. **Geospatial Source Attribution:** Ingests live WAQI sensor data, NASA FIRMS thermal anomalies, and OpenStreetMap road-load indices to pinpoint exact pollution drivers.
2. **Autonomous Intervention Engine:** Evaluates 8 GRAP-aligned mandates via an adjusted MCDA/TOPSIS ranking matrix.
3. **Urban Digital Twin Simulator:** Simulates single and multi-variable policy scenarios.
4. **Health & Economic Impact Engine:** Translates AQI shifts into macroeconomic realities (hospitalizations avoided, capital saved).
5. **AI Executive Brief (Mayor Copilot):** A zero-hallucination text synthesis engine.
6. **Executive Command Center:** A unified, real-time tracking interface.

### What makes it unique?
* **Mathematical Rigor (Overlap Compounding):** Eliminates double-counting in multi-policy scenarios using algebraic compounding ($P_{drop} = 1 - \prod (1 - \epsilon_i)$).
* **Deterministic AI:** The Mayor Copilot physically cannot hallucinate. When asked *"Why this intervention?"*, it executes strict intent-matching and prints exact upstream dataclass fields (e.g., `RankedIntervention.final_score`).
* **Reinforcement Learning Routing:** Includes a custom Gymnasium environment where a PPO agent dynamically routes citizens through the cleanest paths on the real New Delhi OSM graph, penalizing AQI spikes exponentially.

### What impact could it have?
AirTwin X transforms abstract environmental data into direct civic action. By mapping a simulated 72-point AQI drop to 16 emergency admissions avoided and ₹1.3L in medical savings, it empowers Municipal Corporations, Pollution Control Boards, and Smart City nodes to execute high-impact, financially optimized environmental interventions before committing real-world resources.

---

## Quick Start & Installation

### Prerequisite Environment Setup
```bash
# 1. Clone the repository
git clone https://github.com/Soham-o/AirTwin-X.git
cd AirTwin-X

# 2. Initialize dependencies
pip install -r requirements.txt

# 3. Configure live environment variables (Optional - Engine defaults to physics mock)
mkdir -p .streamlit
echo 'WAQI_TOKEN = "your_official_token_here"' > .streamlit/secrets.toml

# 4. Fire up the dashboard
streamlit run app.py
```
The interface will initialize locally at `http://localhost:8501`. Works out-of-the-box even without a token via a locally calibrated physics mock engine.

---

## The 5-Minute Validation Walkthrough

| Step | Action | Interface Response | Architectural Validation |
| :--- | :--- | :--- | :--- |
| **1** | Initialize App | Executive Decision Card boots in a guided empty state. | System reads state safely from empty inputs. |
| **2** | Run Pipeline | Trigger the Attribution Engine via the sidebar. | Pipeline illumination; multi-modal feature vectors execute. |
| **3** | Inspect Weights | Scroll to the Intervention Command Center. | Review Top 3 actions ranked via MCDA matrix. |
| **4** | Activate Twin | Select multiple overlapping policies inside the Digital Twin. | Observe the non-linear curve transformation; verify absence of additive skew. |
| **5** | Assess Impact | Review the Health & Economic Impact Ledger. | Real-time calculation of DALYs saved and public expenditure preserved. |
| **6** | Query Copilot | Invoke: *"Why this intervention?"* | Traceable, structured brief outputs alongside explicit dataclass citations. |

---

## Engineering Resilience

AirTwin X is built on deterministic software engineering principles. The mathematical and structural integrity of the project is verified by a robust regression test suite covering determinism guarantees, ranking invariants, and overlap-compounding properties.

```bash
pytest test_airtwin.py -v
# 27 passed in 4.31s
```

## Future Architecture
* **High-Throughput Ingestion:** Swapping REST polling for an Apache Kafka cluster consuming raw state-level CPCB data feeds directly.
* **Micro-Service Separation:** Wrapping individual modules within explicit `FastAPI` structures to orchestrate them inside a containerized Docker ecosystem behind an Nginx reverse proxy.

---
<div align="center">
  <sub>Built by <a href="https://github.com/Soham-o">Soham Panda</a> • Open Source under the MIT License</sub>
</div>
