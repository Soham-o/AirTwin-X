<div align="center">

  <!-- Animated Typing Text for "Look" -->
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=26&pause=1000&color=2EA043&center=true&vCenter=true&width=800&lines=AI-Powered+Urban+Intervention+Operating+System;Simulate+Before+You+Act;Deterministic+AI+for+Smart+Cities" alt="Typing Animation" />

  <pre>
   ___   _       _____       _         _  __ 
  / _ \ (_)     |_   _|_ _ _(_)_ __   | |/ / 
 | |_| || |       | | \ V  V / | '_ \  | ' <  
 |_|_|_||_|       |_|  \_/\_/|_| |_|_ |_|\_\ 
  __ _ | |__  _ __  ___  _ __| |_| |_| (_)/ /_ 
 / _` || '_ \| '_ \/ _ \| '__| __| __| | | '_ \
 | (_| || |_) | |_) | (_) | |  | |_| |_| | | | |
  \__,_||_.__/| .__/ \___/|_|   \__|\__|_|_|_| |_|
              |_|                               
  </pre>
  
  <p><i>Because knowing the AQI number is not the same as knowing what to do about it.</i></p>
  <!-- Fixed Streamlit Badge & Hackathon Badge -->
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  [![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://airtwin-x.streamlit.app)
  <br><br>

  <!-- App Screenshot / GIF Animation Placeholder -->
  <!-- ⚠️ INSTRUCTION: Drop your screen recording GIF into an 'assets' folder in your repo and name it 'demo.gif' -->
  <img src="https://raw.githubusercontent.com/Soham-o/AirTwin-X/main/assets/demo.gif" onerror="this.onerror=null; this.src='https://placehold.co/800x400/1e1e1e/4caf50?text=AirTwin+X+Dashboard\\n(Upload+demo.gif+to+/assets/)';" alt="AirTwin X App Interface" width="85%" style="border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.5);">

</div>

---

## 📖 The Story of AirTwin X

### 🚨 What problem existed?
Every major metropolis faces an environmental chasm. Open any municipal Air Quality Index (AQI) dashboard right now: you will find a number, a color-coded warning, and a passive recommendation to "avoid outdoor activities." Traditional air quality infrastructure is fundamentally diagnostic. It leaves city administrators blind to critical operational realities: Which specific source is actively driving the spike today? If we have a limited budget, which municipal intervention yields the maximum drop in PM2.5?

### ❌ Why existing solutions failed?
Previous attempts to solve this typically fall into two traps:
1. **Passive Observation:** Standard dashboards merely observe the crisis but offer no actionable mechanics to solve it. 
2. **Hallucinated Interventions:** Recent "AI for AQI" projects simply wrap a Large Language Model around a raw sensor feed. These models confidently hallucinate intervention impacts, blindly add percentages together without respecting physical constraints, and fail to provide traceable, mathematically rigorous reasoning for civic resource allocation.

### 💡 What insight led to the idea?
AirTwin X was built on a core realization: **a city needs a decision pipeline, not just an observation deck.** 
We recognized that simulating interventions requires strict non-linear mathematics—if two policies target traffic, their impacts overlap and compound; they do not simply add up. Furthermore, we realized that an AI assistant for a city Mayor cannot be generative; it must be deterministic. Every metric, citation, and recommendation must be inextricably linked to verifiable epidemiological and economic data.

### ⚙️ What was built?
We engineered an AI-powered Urban Intervention Operating System. It is a fully decoupled chain of six stateless Python micro-modules orchestrated through a low-latency Streamlit interface.

<details>
<summary><b>Click to expand the 6-Layer Architecture</b></summary>

1. **Geospatial Source Attribution:** Ingests live WAQI sensor data, NASA FIRMS thermal anomalies, and OpenStreetMap road-load indices to pinpoint exact pollution drivers.
2. **Autonomous Intervention Engine:** Evaluates 8 GRAP-aligned mandates via an adjusted MCDA/TOPSIS ranking matrix.
3. **Urban Digital Twin Simulator:** Simulates single and multi-variable policy scenarios.
4. **Health & Economic Impact Engine:** Translates AQI shifts into macroeconomic realities (hospitalizations avoided, capital saved).
5. **AI Executive Brief (Mayor Copilot):** A zero-hallucination text synthesis engine.
6. **Executive Command Center:** A unified, real-time tracking interface.
</details>

### ✨ What makes it unique?
* 🧮 **Mathematical Rigor (Overlap Compounding):** Eliminates double-counting in multi-policy scenarios using algebraic compounding ($P_{drop} = 1 - \prod (1 - \epsilon_i)$).
* 🤖 **Deterministic AI:** The Mayor Copilot physically cannot hallucinate. When asked *"Why this intervention?"*, it executes strict intent-matching and prints exact upstream dataclass fields (e.g., `RankedIntervention.final_score`).
* 🗺️ **Reinforcement Learning Routing:** Includes a custom Gymnasium environment where a PPO agent dynamically routes citizens through the cleanest paths on the real New Delhi OSM graph, penalizing AQI spikes exponentially.

### 🌍 What impact could it have?
AirTwin X transforms abstract environmental data into direct civic action. By mapping a simulated 72-point AQI drop to 16 emergency admissions avoided and ₹1.3L in medical savings, it empowers Municipal Corporations, Pollution Control Boards, and Smart City nodes to execute high-impact, financially optimized environmental interventions before committing real-world resources.

---

## 🚀 Quick Start & Installation

### Prerequisite Environment Setup
```bash
# 1. Clone the repository
git clone [https://github.com/Soham-o/AirTwin-X.git](https://github.com/Soham-o/AirTwin-X.git)
cd AirTwin-X

# 2. Initialize dependencies
pip install -r requirements.txt

# 3. Configure live environment variables (Optional - Engine defaults to physics mock)
mkdir -p .streamlit
echo 'WAQI_TOKEN = "your_official_token_here"' > .streamlit/secrets.toml

# 4. Fire up the dashboard
streamlit run app.py
```
