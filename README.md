<div align="center">

# AirTwin X

**AI-Powered Urban Intervention Operating System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://airtwin-x.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/Soham-o/AirTwin-X?style=for-the-badge&color=yellow)](https://github.com/Soham-o/AirTwin-X/stargazers)
[![Deployment](https://img.shields.io/badge/Deployment-Live-success?style=for-the-badge)](#)

<br>

<!-- ⚠️ INSTRUCTION: Drop your banner image into an 'assets' folder and name it 'banner.png' -->
<img src="https://raw.githubusercontent.com/Soham-o/AirTwin-X/main/assets/banner.png" onerror="this.onerror=null; this.src='https://placehold.co/1000x300/1e1e1e/4caf50?text=AirTwin+X+Banner\\n(Upload+banner.png+to+/assets/)';" alt="AirTwin X Banner" width="100%" style="border-radius: 12px;">

<br>

<!-- ⚠️ INSTRUCTION: Drop your screen recording GIF into an 'assets' folder and name it 'demo.gif' -->
<img src="https://raw.githubusercontent.com/Soham-o/AirTwin-X/main/assets/demo.gif" onerror="this.onerror=null; this.src='https://placehold.co/800x450/1e1e1e/4caf50?text=AirTwin+X+Demo\\n(Upload+demo.gif+to+/assets/)';" alt="AirTwin X App Interface" width="85%" style="border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.5);">

<br>

</div>

---

## Table of Contents

- [The Story of AirTwin X](#the-story-of-airtwin-x)
- [Why AirTwin X?](#why-airtwin-x)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [AI Pipeline & Algorithms](#ai-pipeline--algorithms)
- [Technology Stack](#technology-stack)
- [Installation & Configuration](#installation--configuration)
- [Interface Previews](#interface-previews)
- [Documentation](#documentation)
- [Testing](#testing)
- [Roadmap & Future Work](#roadmap--future-work)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

---

## The Story of AirTwin X

### What problem existed?
Every major metropolis in the developing world faces a perpetual environmental crisis. Open any municipal dashboard today, and it will passively tell you the current Air Quality Index (AQI) number alongside a generic warning to "avoid outdoor activities." Standard monitoring infrastructure is fundamentally diagnostic, leaving city administrators blind to critical operational realities.

### Why existing solutions failed?
Previous attempts to solve this typically fall into two traps. First, standard dashboards are passive—they observe the crisis but offer no actionable mechanics to solve it. Second, recent "AI for AQI" projects simply wrap a Large Language Model around a raw sensor feed. These models confidently hallucinate intervention impacts, blindly add percentages together without respecting physical constraints, and fail to provide traceable, mathematically rigorous reasoning for civic resource allocation.

### What insight led to the idea?
AirTwin X was built on a core realization: **a city needs a decision pipeline, not just an observation deck.** Simulating interventions requires strict non-linear mathematics—if two policies target traffic, their impacts overlap and compound; they do not simply add up. Furthermore, an AI assistant for a city Mayor cannot be generative; it must be deterministic. Every metric, citation, and recommendation must be inextricably linked to verifiable epidemiological and economic data.

### What was built?
We engineered an AI-powered Urban Intervention Operating System. It is a fully decoupled chain of six stateless Python micro-modules orchestrated through a low-latency Streamlit interface. It bridges the gap between environmental data and civic execution.

### What makes it unique?
It answers the *Why* and the *How*. Instead of asking "What is the AQI?", AirTwin X determines which geospatial source is responsible, which specific GRAP-aligned intervention provides the highest ROI, and exactly how many citizens will benefit. Every executive brief is backed by immutable dataclass citations using algebraic overlap-compounding logic.

### What impact could it have?
AirTwin X transforms abstract environmental data into direct civic action. By allowing administrators to simulate policies before deploying capital, it translates predicted AQI drops into empirical outcomes: estimating emergency admissions avoided, Disability-Adjusted Life Years (DALYs) reduced, and direct public healthcare budget savings.

---

## Why AirTwin X?

| Capability | Traditional AQI Dashboard | AirTwin X |
| :--- | :--- | :--- |
| **Primary Output** | "AQI is 340 (Severe)" | "Traffic is driving 46% of pollution. Restrict heavy vehicles." |
| **AI Integration** | None (or generative LLM wrapper) | Deterministic Mayor Copilot (Zero Hallucination) |
| **Intervention Planning** | Manual guesswork | Autonomous MCDA/TOPSIS ranking & Digital Twin Simulation |
| **Metric Translation** | PM2.5 / PM10 levels | Hospitalizations avoided, DALYs reduced, ₹ Saved |
| **Policy Compounding** | Naive addition (A+B = % drop) | Algebraic overlap-compounding logic |

---

## Core Features

| 🌍 Geospatial Attribution Engine | ⚡ Intervention Command Engine |
| :--- | :--- |
| **Pinpoints the source.** Granular attribution for traffic, industrial emissions, construction dust, and biomass burning, adjusted by weather stagnation factors. Includes explicit confidence scoring. | **Recommends the action.** Autonomously ranks municipal interventions based on a multi-criteria matrix evaluating AQI reduction potential, financial cost, physical feasibility, deployment speed, and confidence. |

| 🏙️ Urban Digital Twin | 🏥 Health & Economic Impact Engine |
| :--- | :--- |
| **Simulates the future.** Test single or multi-intervention scenarios before implementation. Provides precise scenario comparisons and confidence bounds to prevent double-counting. | **Calculates the human ROI.** Translates predicted AQI drops into empirical outcomes: hospital admissions avoided, DALYs reduced, productivity gains, and direct budget savings. |

| 🤖 Mayor Copilot | 🎛️ Executive Command Center |
| :--- | :--- |
| **Grounded decision support.** Strictly deterministic—no hallucinations. Every answer is synthesized purely from the outputs of preceding upstream modules. | **The 30-second briefing.** A unified dashboard delivering a snapshot of the crisis, recommended actions, and simulated economic impact instantly. |

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                           │
│  [Open-Meteo]   [WAQI API]   [NASA FIRMS]   [OpenStreetMap]     │
└────────┬─────────────┬─────────────┬──────────────┬─────────────┘
         │             │             │              │
         ▼             ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 GEOSPATIAL SOURCE ATTRIBUTION                   │
│         (Traffic, Industrial, Construction, Biomass)            │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│             AUTONOMOUS INTERVENTION COMMAND ENGINE              │
│        (MCDA Ranking: Impact, Cost, Speed, Feasibility)         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      URBAN DIGITAL TWIN                         │
│     (Multi-scenario Simulation & Overlap-Compounding Math)      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                HEALTH & ECONOMIC IMPACT ENGINE                  │
│       (DALYs, Hospitalizations Avoided, Financial ROI)          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│            MAYOR COPILOT & EXECUTIVE COMMAND CENTER             │
│        (Deterministic AI Briefing & Policy Execution)           │
└─────────────────────────────────────────────────────────────────┘
```
