# AirTwin X — Architecture Reference

## Module Responsibilities

### `attribution_engine.py` — Source Attribution Engine

**Responsibility:** Given live sensor data, telemetry, and geospatial context, decompose the current AQI into a percentage breakdown across five source categories.

**Inputs:** City coordinates, WAQI sensor readings, OpenWeatherMap telemetry, NASA FIRMS fire data, OSM road network

**Output:** `AttributionResult`
```python
@dataclass
class AttributionResult:
    percentages: dict[str, float]    # {'Traffic': 46.0, 'Construction': 28.0, ...}
    primary_source: str
    confidence: int                  # 0-100
    explanation: str                 # plain-language summary
    sub_scores: SourceScores
    data_sources_used: list[str]
```

**Design decision:** Pure function — same inputs always produce the same outputs. No session state, no caching.

---

### `intervention_agent.py` — Intervention Command Engine + Digital Twin

**Responsibility:** Given `AttributionResult.percentages`, rank 8 pre-defined interventions and expose a simulation API.

**Key classes:**
- `InterventionSpec` — static description of one intervention (name, effectiveness, cost tier, target sources, deployment time, GRAP reference)
- `RankedIntervention` — one scored, ranked result
- `CommandCenterOutput` — full ranked list + crisis level + composite explanation
- `SimulationResult` — output of `simulate_scenario()`
- `ScenarioComparison` — output of `compare_scenarios()`

**Ranking formula (5-factor weighted):**
```
score = (aqi_impact × 0.40) + (cost × 0.20) + (feasibility × 0.20)
      + (confidence × 0.10) + (speed × 0.10)
```

**Overlap-compounding rule (shared helper `_compound_remaining_fraction`):**
```
remaining(source) = Π (1 − effectiveness_i)   for each intervention i targeting source
```
Used by both `_composite_explanation()` and `simulate_scenario()` — defined once, no duplication.

**Design decision:** The Digital Twin (`simulate_scenario`) lives in this file rather than a separate one because it needs direct access to `InterventionSpec.source_effectiveness` and `InterventionSpec.target_sources`. A separate file would require either re-importing or duplicating these fields.

---

### `health_economic_engine.py` — Health & Economic Impact Engine

**Responsibility:** Translate `SimulationResult` (a predicted PM2.5 reduction) into health outcomes and economic value.

**Inputs:** `SimulationResult`, `population_exposed: int`

**Output:** `HealthEconomicImpact`
```python
@dataclass
class HealthEconomicImpact:
    population_exposed: int
    population_protected: int
    hospitalizations_avoided: int
    asthma_attacks_avoided: int
    dalys_reduced: float
    healthcare_savings_inr: float
    productivity_gains_inr: float
    social_benefit_score: int        # 0-100 composite
    delta_pm25: float
    explanation: str
    assumptions: list[str]           # rendered in UI expander
```

**Calculation chain:**
1. `aqi_to_pm25()` (CPCB official breakpoints) → `delta_pm25`
2. `delta_pm25 / 10` × concentration-response coefficient × baseline rate × population → cases avoided
3. Cases avoided × unit costs → `healthcare_savings_inr`
4. Cases avoided × lost-workdays × per-capita-income/365 → `productivity_gains_inr`
5. DALYs + savings + productivity → `social_benefit_score` (weighted composite, 0-100)

**Design decision:** No AQI arithmetic lives here — the engine only consumes `SimulationResult.baseline_aqi` and `SimulationResult.predicted_aqi`, then calls `aqi_to_pm25()` on both. This means if the Digital Twin's simulation formula changes, the health engine automatically gets the updated numbers.

---

### `mayor_copilot.py` — AI Decision Support Copilot

**Responsibility:** Answer natural-language questions about the pipeline's outputs, grounded exclusively in `DecisionContext`.

**Design decision: deterministic intent-matching, not an LLM**

A free-form LLM bolted on top of this pipeline would risk hallucinating numbers the upstream modules actually computed. Instead:
- Keyword groups map each question type to a named intent
- Each intent handler is a pure function of `DecisionContext` fields
- `CopilotAnswer.sources` lists the exact dataclass fields every sentence derives from
- Unsupported questions get an honest "I can only answer..." with the supported list

**Supported intents:**
| Intent | Triggered by |
|---|---|
| `why_aqi_increasing` | "why", "aqi", "increasing/high/rising" |
| `primary_source` | "primary/main/biggest", "source" |
| `why_intervention` | "why", "recommend/suggest" |
| `what_if_other` | "compare", "alternative", "another intervention" |
| `how_many_benefit` | "how many", "benefit", "population protected" |
| `how_much_saved` | "how much money", "savings", "economic" |
| `why_confidence` | "confidence score", "how sure/certain" |
| `health_impact` | "hospitalization", "asthma", "daly", "health impact" |

**Graceful degradation:** every handler checks whether its required upstream objects are present in `DecisionContext` before accessing fields, returning a clear "not yet computed" message rather than an AttributeError.

---

### `app.py` — Streamlit Dashboard

**Responsibility:** Orchestrate all five modules, manage session state, and render the UI.

**Session state keys:**
| Key | Type | Set by |
|---|---|---|
| `attribution_result` | `AttributionResult \| None` | Attribution Engine section |
| `command_center_output` | `CommandCenterOutput \| None` | Intervention section |
| `last_simulations` | `list[tuple[str, SimulationResult]]` | Digital Twin section |
| `last_comparison` | `ScenarioComparison \| None` | Digital Twin comparison mode |
| `last_health_impacts` | `list[tuple[str, HealthEconomicImpact]]` | Health Impact section |
| `copilot_chat_history` | `list[tuple[str, CopilotAnswer]]` | Mayor Copilot section |
| `intervention_agent` | `InterventionAgent` | App startup (singleton) |
| `health_economic_engine` | `HealthEconomicEngine` | App startup (singleton) |
| `mayor_copilot` | `MayorCopilot` | App startup (singleton) |

**Page section order:**
1. Hero section (static — renders on every load)
2. Executive Command Center (reads session_state — populates as modules run)
3. Module 2: Daily Precautions
4. Module 3: Spatial AQI Ranking + Map + Routing
5. Pollution Source Attribution Engine (populates `attribution_result`)
6. Autonomous Intervention Command Center (reads attribution → populates `command_center_output`)
7. Urban Digital Twin (reads CC output → populates `last_simulations`)
8. Health & Economic Impact (reads simulations → populates `last_health_impacts`)
9. Mayor Copilot chat interface
10. Module 4: Counterfactual Policy Simulator (legacy parallel tool)

---

## Data Flow

```
fetch_live_bounded_data()    → sensor_readings
fetch_live_telemetry()       → telemetry
          ↓
SourceAttributionEngine.attribute() → AttributionResult
          ↓                            → session_state.attribution_result
InterventionAgent.generate()         → CommandCenterOutput
          ↓                            → session_state.command_center_output
InterventionAgent.simulate_scenario() → SimulationResult
  or compare_scenarios()               → session_state.last_simulations
          ↓                            → session_state.last_comparison
HealthEconomicEngine.assess()         → HealthEconomicImpact
          ↓                            → session_state.last_health_impacts
MayorCopilot.ask()                    → CopilotAnswer  (on demand)
          ↓
Executive Command Center              (reads all session_state keys above)
```

---

## Design Decisions

### Why are modules stateless?
Each engine (`InterventionAgent`, `HealthEconomicEngine`, `MayorCopilot`) is instantiated once at app startup and stored as a session_state singleton. The singleton holds no mutable state — all state lives in `session_state` as output dataclasses. This means the same engine instance can safely serve multiple Streamlit re-runs per session without stale data contaminating results.

### Why is MayorCopilot deterministic rather than LLM-based?
The requirement "every answer must be grounded in pipeline outputs, no hallucination" is structurally incompatible with a generative LLM: even a well-prompted LLM can paraphrase past the point of numerical accuracy. Keyword-based intent matching with dataclass-field handlers gives a 100% auditable answer — every sentence can be traced to a specific field on a specific dataclass.

### Why does the Digital Twin live in `intervention_agent.py`?
`simulate_scenario()` reuses `_compound_remaining_fraction()`, the same overlap-compounding helper used by `_composite_explanation()`. Splitting the Digital Twin into a separate file would require importing this helper across modules, creating a shared-utility import that's harder to maintain than a single cohesive module.

### Why is health_economic_engine.py zero-dependency at runtime?
All cross-module imports are under `TYPE_CHECKING`. This means `health_economic_engine.py` can be imported and used even if `intervention_agent.py` fails to load (e.g. if `xgboost` or `stable_baselines3` are not installed). The engine receives `SimulationResult` as an already-constructed object from `app.py`.

### Why does the HTML card use adjacent string literals instead of triple-quoted f-strings?
Streamlit's frontend markdown renderer follows CommonMark. In CommonMark, a blank line followed by 4+ spaces of indentation is classified as an **indented code block** — even inside `unsafe_allow_html=True`. Triple-quoted multi-line HTML with blank lines for readability triggers this rule and causes raw HTML to appear as literal text. Single-line adjacent string literal concatenation produces an HTML string with no newlines, making code-block misclassification impossible.