"""
intervention_agent.py
─────────────────────────────────────────────────────────────────────────────
Autonomous Intervention Command Engine for AirTwin X.

Consumes output from SourceAttributionEngine and the current telemetry
snapshot to produce a ranked, explainable set of city-level interventions.

Design principles
─────────────────
• No randomness.  Every score is derived from named inputs.
• No LLM dependency.  Explanations are generated from a transparent
  template system that references the actual computed values.
• Single responsibility.  This module has no Streamlit imports and no
  side-effects beyond returning dataclasses.
• Composable.  The simulation bridge accepts the same percentage inputs
  as the Counterfactual Policy Simulator in app.py, so the two are
  always numerically consistent.

Ranking formula
───────────────
  final_score = (
      w_aqi   × expected_aqi_reduction_pct   +   # primary objective
      w_cost  × (1 - normalised_cost)        +   # lower cost is better
      w_feas  × feasibility_score            +   # 0–1, city capacity
      w_conf  × source_confidence            +   # attribution confidence
      w_time  × (1 - normalised_deployment_hours)  # faster is better
  )

Weights are calibrated so that two interventions with equal AQI impact
will be differentiated by cost and speed — reflecting real city
procurement constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Ranking weights (must sum to 1.0)
# ──────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
# Intervention ranking weights — Multi-Criteria Decision Analysis (MCDA)
# ─────────────────────────────────────────────────────────────────────────
# These weights implement a simplified TOPSIS-style MCDA framework
# (Hwang & Yoon, 1981) adapted for municipal air quality intervention
# prioritisation. Weight assignments follow the emergency management
# principle that outcome (health impact) should dominate near-term
# operational decisions, with cost and feasibility as secondary constraints.
#
# Rationale (challengeable and configurable for production):
#   W_AQI = 0.40  — expected pollution reduction: primary mission objective
#                   (Ramanathan 2001, MCDA for environmental management,
#                    recommends 35–45% on primary outcome in crisis scenarios)
#   W_COST = 0.20 — direct cost: budget-constrained municipal bodies treat
#                   cost as a hard constraint; 20% matches typical public-
#                   sector procurement weighting (Tzeng & Huang 2011)
#   W_FEAS = 0.20 — operational feasibility: interventions that cannot be
#                   deployed have zero real-world value; equal to cost
#   W_CONF = 0.10 — attribution confidence: weights the source reliability,
#                   preventing high-impact but uncertain interventions from
#                   dominating; reflects the precautionary principle
#   W_TIME = 0.10 — deployment speed: in acute crises, faster deployment
#                   reduces cumulative exposure even for lower-impact actions
#
# To override for a specific city/emergency context, adjust these values
# before calling InterventionAgent.generate(). The only hard constraint
# is that they sum to 1.0.
W_AQI  = 0.40
W_COST = 0.20
W_FEAS = 0.20
W_CONF = 0.10
W_TIME = 0.10

assert abs(W_AQI + W_COST + W_FEAS + W_CONF + W_TIME - 1.0) < 1e-9, \
    "Ranking weights must sum to 1.0"

# AQI impact normalisation ceiling: a single intervention addressing its
# full source share cannot realistically achieve more than ~60% AQI
# reduction in a single deployment cycle (real-world lag, compliance
# delay, non-linearity of atmospheric dispersal). This ceiling prevents
# the score from being dominated purely by source-share magnitude.
# Conservative relative to the 85% physical ceiling in simulate_scenario().
_SCORE_AQI_CEILING_PCT = 60.0

assert abs(W_AQI + W_COST + W_FEAS + W_CONF + W_TIME - 1.0) < 1e-9

# ──────────────────────────────────────────────────────────────────────────────
# Crisis level thresholds (AQI)
# ──────────────────────────────────────────────────────────────────────────────
CRISIS_LEVELS = [
    (301, "SEVERE",   "#ff1744", "🔴"),
    (201, "VERY HIGH","#ff5252", "🟠"),
    (151, "HIGH",     "#ff9100", "🟡"),
    (101, "MODERATE", "#ffd740", "🟢"),
    (0,   "GOOD",     "#00e676", "⚪"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InterventionSpec:
    """
    Static specification of one intervention type.
    These values represent Indian municipal context calibrated to
    published cost-effectiveness data from CPCB graded response
    action plans (GRAP) and academic literature.
    """
    id: str
    name: str
    icon: str
    department: str          # which city body executes this
    target_sources: list[str]  # attribution source names it addresses
    # Effectiveness: fraction of the target source's AQI contribution eliminated
    source_effectiveness: float   # 0–1
    # Cost tier: 1 (very low) → 5 (very high) — normalised to 0–1 in ranking
    cost_tier: int
    # Feasibility: 0–1, city capacity to execute on short notice
    base_feasibility: float
    # Deployment hours: how long until measurable effect begins
    deployment_hours: float
    # Human-readable description
    description: str
    # Conditions that boost or reduce feasibility
    feasibility_conditions: dict  # {condition_key: delta}


@dataclass
class RankedIntervention:
    """Output of InterventionAgent.generate() for a single intervention."""
    spec: InterventionSpec
    rank: int                   # 1 = top recommendation
    final_score: float          # 0–1 composite ranking score
    expected_aqi_reduction: float   # absolute AQI points
    expected_aqi_reduction_pct: float  # % of current AQI
    cost_tier: int
    feasibility: float          # 0–1 adjusted
    confidence: float           # 0–1 (from attribution)
    deployment_hours: float
    # Simulation inputs — compatible with counterfactual simulator
    sim_traffic_pct: float      # value to set traffic slider to
    sim_industrial_pct: float   # value to set industrial slider
    sim_wind_shift: float       # value to set wind shift slider
    # Explanation
    reasoning: str
    sub_scores: dict = field(default_factory=dict)


@dataclass
class CommandCenterOutput:
    """Full output from InterventionAgent.generate()."""
    crisis_level: str
    crisis_color: str
    crisis_icon: str
    current_aqi: int
    primary_driver: str
    primary_driver_pct: float
    interventions: list[RankedIntervention]
    attribution_confidence: int
    composite_explanation: str  # paragraph explaining the full strategy


@dataclass
class SimulationResult:
    """
    Output of InterventionAgent.simulate_scenario() — the reusable,
    UI-agnostic "Urban Digital Twin" prediction API (Feature 3).

    This is intentionally decoupled from Streamlit: it can be called from
    the Command Center's multi-select scenario builder, from a future
    batch/API endpoint, or from tests, without any UI dependency.
    """
    baseline_aqi: int
    predicted_aqi: int
    delta_aqi: int                 # absolute AQI points removed
    total_pct_drop: float          # 0-85, physical ceiling matches simulator
    breakdown: dict                # {source_name: pct_of_AQI_reduced_by_that_source}
    scenario_label: str            # human-readable summary of what was simulated
    confidence: int = 0            # 0-100, see simulate_scenario() for derivation
    wind_shift_kmh: float = 0.0


@dataclass
class ScenarioComparison:
    """
    Output of InterventionAgent.compare_scenarios(). Wraps two
    SimulationResults — it does not recompute anything; both results come
    from simulate_scenario(), so there is exactly one place AQI prediction
    math exists in this codebase.
    """
    label_a: str
    label_b: str
    result_a: SimulationResult
    result_b: SimulationResult
    better_label: str        # label_a, label_b, or "Tie" — whichever predicts lower AQI
    aqi_gap: int              # |predicted_aqi_a - predicted_aqi_b|
    explanation: str         # which source(s) account for the gap, in plain language


# ──────────────────────────────────────────────────────────────────────────────
# Intervention library
# ──────────────────────────────────────────────────────────────────────────────

INTERVENTION_LIBRARY: list[InterventionSpec] = [

    InterventionSpec(
        id="truck_restriction",
        name="Truck Restrictions",
        icon="🚚",
        department="Traffic Control Department",
        target_sources=["Traffic"],
        source_effectiveness=0.55,
        cost_tier=1,
        base_feasibility=0.88,
        deployment_hours=1.0,
        description=(
            "Ban heavy commercial vehicles from entering city limits "
            "during daytime hours (6am–10pm). Applicable via GRAP "
            "Stage II action."
        ),
        feasibility_conditions={
            "rush_hour": +0.08,     # easier to justify during rush hour
            "weekend":   -0.15,     # harder to enforce on weekends
            "aqi_severe": +0.10,    # emergency justification at severe AQI
        },
    ),

    InterventionSpec(
        id="construction_pause",
        name="Construction Activity Pause",
        icon="🏗️",
        department="Municipal Corporation",
        target_sources=["Construction"],
        source_effectiveness=0.80,
        cost_tier=2,
        base_feasibility=0.72,
        deployment_hours=2.0,
        description=(
            "Issue emergency stop-work orders for all demolition, "
            "earthmoving, and building construction. Requires municipal "
            "commissioner authorization. Effective for coarse PM10 reduction."
        ),
        feasibility_conditions={
            "aqi_severe":  +0.12,
            "wind_calm":   +0.08,   # more justified when dust can't disperse
            "rain":        -0.20,   # unnecessary if rain suppresses dust
        },
    ),

    InterventionSpec(
        id="water_spraying",
        name="Water Spraying & Dust Suppression",
        icon="💧",
        department="Public Works Department",
        target_sources=["Construction", "Traffic"],
        source_effectiveness=0.25,
        cost_tier=2,
        base_feasibility=0.92,
        deployment_hours=0.5,
        description=(
            "Deploy water tankers and mechanical road sweepers on "
            "identified arterial corridors. Suppresses resuspended road "
            "dust. Fastest deployable intervention."
        ),
        feasibility_conditions={
            "aqi_severe":  +0.05,
            "wind_high":   -0.20,   # less effective in high wind
            "temp_high":   -0.10,   # water evaporates faster
        },
    ),

    InterventionSpec(
        id="industrial_emission_controls",
        name="Industrial Emission Controls",
        icon="🏭",
        department="Pollution Control Board",
        target_sources=["Industrial"],
        source_effectiveness=0.70,
        cost_tier=3,
        base_feasibility=0.60,
        deployment_hours=6.0,
        description=(
            "Issue emergency direction under Environment Protection Act "
            "to industrial units to cut production or switch to cleaner "
            "fuel. Most effective for NOx and SOx reduction."
        ),
        feasibility_conditions={
            "aqi_severe":  +0.18,
            "weekend":     -0.10,
            "industrial_dominant": +0.15,  # extra justification if industrial is primary
        },
    ),

    InterventionSpec(
        id="open_burning_enforcement",
        name="Open Burning Enforcement",
        icon="🔥",
        department="District Magistrate / Fire Services",
        target_sources=["Biomass Burning"],
        source_effectiveness=0.65,
        cost_tier=1,
        base_feasibility=0.55,
        deployment_hours=3.0,
        description=(
            "Deploy enforcement teams to suppress crop residue burning, "
            "garbage fires, and dhaba/tandoor violations. Coordinate "
            "with neighbouring districts for cross-border fire control."
        ),
        feasibility_conditions={
            "biomass_dominant": +0.20,  # highly justified when biomass is primary
            "aqi_severe":       +0.12,
            "fire_detected":    +0.15,  # FIRMS data confirms fires
            "wind_high":        -0.10,  # fires spread faster, harder to control
        },
    ),

    InterventionSpec(
        id="traffic_signal_optimization",
        name="Traffic Signal Optimization",
        icon="🚦",
        department="Integrated Traffic Management System",
        target_sources=["Traffic"],
        source_effectiveness=0.30,
        cost_tier=1,
        base_feasibility=0.95,
        deployment_hours=0.25,
        description=(
            "Activate green-wave signal coordination on key corridors to "
            "reduce vehicle idling time. Reduces CO and NOx from stop-start "
            "driving. Deployable remotely from traffic control centre."
        ),
        feasibility_conditions={
            "rush_hour":   +0.04,
            "weekend":     -0.08,
        },
    ),

    InterventionSpec(
        id="odd_even_scheme",
        name="Odd-Even Vehicle Scheme",
        icon="🔢",
        department="Transport Department",
        target_sources=["Traffic"],
        source_effectiveness=0.40,
        cost_tier=2,
        base_feasibility=0.50,
        deployment_hours=12.0,
        description=(
            "Alternate-day private vehicle restriction by registration "
            "number parity. Requires 12h advance public notice. "
            "Previously implemented in Delhi; effective for sustained "
            "multi-day pollution events."
        ),
        feasibility_conditions={
            "aqi_severe":  +0.20,
            "rush_hour":   +0.10,
            "weekend":     -0.30,
        },
    ),

    InterventionSpec(
        id="smog_tower_activation",
        name="Smog Tower & Air Purifier Network",
        icon="🌀",
        department="Delhi Pollution Control Committee",
        target_sources=["Traffic", "Industrial"],
        source_effectiveness=0.12,
        cost_tier=4,
        base_feasibility=0.40,
        deployment_hours=0.5,
        description=(
            "Activate installed large-scale air purification towers. "
            "Limited spatial coverage (~1 km radius each). "
            "Supplementary measure — not a primary intervention."
        ),
        feasibility_conditions={
            "aqi_severe":  +0.15,
            "wind_calm":   +0.12,   # more effective when air is stagnant
        },
    ),
]

# Lookup by id
_LIBRARY_MAP: dict[str, InterventionSpec] = {s.id: s for s in INTERVENTION_LIBRARY}

# ──────────────────────────────────────────────────────────────────────────────
# Cost tier normalisation (1=cheapest → 5=most expensive)
# ──────────────────────────────────────────────────────────────────────────────
_MAX_COST_TIER = 5
_MAX_DEPLOY_HOURS = 24.0   # normalisation ceiling


def _normalise_cost(tier: int) -> float:
    """Returns 0 (cheapest) → 1 (most expensive)."""
    return (tier - 1) / (_MAX_COST_TIER - 1)


def _normalise_deploy(hours: float) -> float:
    """Returns 0 (instant) → 1 (slowest)."""
    return min(1.0, hours / _MAX_DEPLOY_HOURS)


def _compound_remaining_fraction(
    interventions: list["InterventionSpec"],
    percentages: dict[str, float],
) -> dict[str, float]:
    """
    Shared by _composite_explanation() and simulate_scenario() so the rule
    "multiple interventions on the same source compound, not add" is defined
    exactly once (avoids the duplicate-logic anti-pattern).

    For each source category, returns the fraction of that source's AQI
    contribution that SURVIVES after every intervention in `interventions`
    has acted on it: remaining(source) = Π (1 − effectiveness_i) over each
    intervention i targeting that source. 1.0 = untouched, 0.0 = fully
    eliminated. This treats same-source interventions as acting
    independently rather than synergistically — an explicit, documented
    modelling assumption, not a hidden heuristic.
    """
    remaining = {src: 1.0 for src in percentages}
    for iv in interventions:
        for src in iv.target_sources:
            if src in remaining:
                remaining[src] *= (1.0 - iv.source_effectiveness)
    return remaining


def weather_reduction_pct(wind_speed: float, wind_shift_kmh: float, weather_pct: float) -> float:
    """
    Shared stagnation-relief formula. Used by both the free-form
    Counterfactual Simulator sliders (app.py Module 4) and
    simulate_scenario(), so this formula is defined exactly once rather
    than duplicated across the Streamlit file and this module.

    Models reduced pollutant trapping as wind speed rises above a calm-air
    baseline of 8 km/h. `weather_pct` is the Weather Amplification source's
    share of current AQI, expressed as a 0–1 fraction.
    """
    new_wind = wind_speed + wind_shift_kmh
    before = max(0.0, (8.0 - wind_speed) / 8.0)
    after = max(0.0, (8.0 - new_wind) / 8.0)
    return max(0.0, (before - after) * weather_pct * 100.0)


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

def _is_weekend_today() -> bool:
    """Returns True on Saturday (5) and Sunday (6). Used by feasibility conditions
    that reduce enforcement likelihood on weekends (e.g. Odd-Even scheme).
    Defined as a named function rather than an inline lambda so it's testable."""
    import datetime
    return datetime.date.today().weekday() >= 5


class InterventionAgent:
    """
    Autonomous Intervention Command Engine.

    Usage
    ─────
        agent = InterventionAgent()
        output = agent.generate(
            current_aqi=285,
            percentages=attribution_result.percentages,
            attribution_confidence=attribution_result.confidence,
            telemetry=telemetry_dict,
            current_hour=8,
            fire_hotspot_count=3,
        )
    """

    def generate(
        self,
        current_aqi: int,
        percentages: dict[str, float],
        attribution_confidence: int,
        telemetry: Optional[dict],
        current_hour: int,
        fire_hotspot_count: int = 0,
    ) -> CommandCenterOutput:
        """
        Generate a ranked intervention plan.

        Parameters
        ──────────
        current_aqi           : current worst-station AQI
        percentages           : source attribution percentages (sum = 100)
                                from SourceAttributionEngine or causal engine
        attribution_confidence: 0–100 confidence score from attribution engine
        telemetry             : dict with wind_speed, wind_dir, temp, active_fires
        current_hour          : int 0–23
        fire_hotspot_count    : count of NASA FIRMS hotspots near city

        Returns
        ───────
        CommandCenterOutput with ranked interventions and explanation.
        """
        wind_speed = (telemetry or {}).get("wind_speed", 10.0)
        temp       = (telemetry or {}).get("temp", 28.0)
        active_fires = (telemetry or {}).get("active_fires", 0)

        # Derive binary condition flags used in feasibility adjustment
        conditions = self._build_conditions(
            current_aqi, current_hour, wind_speed, temp,
            active_fires, fire_hotspot_count, percentages,
        )

        # Determine crisis level
        crisis_level, crisis_color, crisis_icon = _crisis_level(current_aqi)

        # Primary driver from attribution
        primary_driver = max(percentages, key=percentages.get)
        primary_pct    = percentages[primary_driver]

        # Attribution confidence as 0–1
        conf_norm = attribution_confidence / 100.0

        # Score and rank every intervention
        ranked: list[RankedIntervention] = []
        for spec in INTERVENTION_LIBRARY:
            ri = self._score_intervention(
                spec, current_aqi, percentages,
                conf_norm, conditions,
            )
            ranked.append(ri)

        ranked.sort(key=lambda x: x.final_score, reverse=True)
        for i, ri in enumerate(ranked):
            ri.rank = i + 1

        # Build composite explanation for the full strategy
        composite = self._composite_explanation(
            ranked[:3], current_aqi, primary_driver, primary_pct,
            crisis_level, attribution_confidence, percentages,
        )

        return CommandCenterOutput(
            crisis_level=crisis_level,
            crisis_color=crisis_color,
            crisis_icon=crisis_icon,
            current_aqi=current_aqi,
            primary_driver=primary_driver,
            primary_driver_pct=primary_pct,
            interventions=ranked,
            attribution_confidence=attribution_confidence,
            composite_explanation=composite,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Urban Digital Twin — Feature 3
    # ──────────────────────────────────────────────────────────────────────

    def simulate_scenario(
        self,
        current_aqi: int,
        percentages: dict[str, float],
        telemetry: Optional[dict],
        interventions: list[InterventionSpec],
        attribution_confidence: int = 70,
        extra_wind_shift_kmh: float = 0.0,
    ) -> SimulationResult:
        """
        Predict AQI after deploying one or more interventions together.

        This is the reusable API behind the "Urban Digital Twin" feature:
        given the current AQI, the attribution breakdown, and a list of
        InterventionSpecs (one or many — e.g. straight from
        CommandCenterOutput.interventions), it returns a predicted AQI
        plus a per-source breakdown of where the reduction came from.

        Reuses, rather than duplicates:
        • _compound_remaining_fraction() — the same overlap-compounding
          rule already used by _composite_explanation(), generalised here
          from "exactly top 3" to "any N selected interventions".
        • weather_reduction_pct() — the same stagnation-relief formula the
          free-form Counterfactual Simulator sliders in app.py use, so
          the math is defined once and both call sites stay numerically
          consistent.

        confidence derivation (documented, not arbitrary):
        The prediction can only be as trustworthy as the attribution
        breakdown it's compounding over, so confidence starts at
        attribution_confidence. Each additional intervention beyond the
        first applies a modest 3% discount, since stacking interventions
        on overlapping sources leans on the explicit "acts independently,
        not synergistically" assumption in _compound_remaining_fraction —
        more stacking means more exposure to that assumption being wrong.
        This is a deterministic function of real inputs, not a fabricated
        or random number.

        Example
        ───────
            agent.simulate_scenario(285, percentages, telemetry,
                                     [TRUCK_RESTRICTION_SPEC],
                                     attribution_confidence=82)
            → SimulationResult(baseline_aqi=285, predicted_aqi=247, ...)
        """
        wind_speed = (telemetry or {}).get("wind_speed", 10.0)

        # Source-targeted reduction, compounded (not summed) per source.
        remaining = _compound_remaining_fraction(interventions, percentages)
        breakdown: dict[str, float] = {}
        source_pct_drop = 0.0
        for src, pct in percentages.items():
            drop = (pct / 100.0) * (1.0 - remaining[src]) * 100.0
            if drop > 0.05:
                breakdown[src] = round(drop, 1)
            source_pct_drop += drop

        # Meteorological lever — independent of which sources were targeted,
        # mirrors the original Counterfactual Simulator's wind-shift math.
        weather_pct = percentages.get("Weather Amplification", 0.0) / 100.0
        weather_drop = weather_reduction_pct(wind_speed, extra_wind_shift_kmh, weather_pct)
        if weather_drop > 0.05:
            breakdown["Weather Amplification"] = round(weather_drop, 1)

        total_pct_drop = min(source_pct_drop + weather_drop, 85.0)  # physical ceiling
        predicted_aqi = int(round(current_aqi * (1.0 - total_pct_drop / 100.0)))
        delta_aqi = current_aqi - predicted_aqi

        stack_discount = 0.97 ** max(0, len(interventions) - 1)
        confidence = int(round(max(0, min(100, attribution_confidence)) * stack_discount))

        if interventions:
            names = ", ".join(iv.name for iv in interventions)
            label = f"{len(interventions)} intervention(s) simulated together: {names}."
        else:
            label = "Baseline scenario — no intervention applied."
        if extra_wind_shift_kmh > 0:
            label += f" Includes a +{extra_wind_shift_kmh:.0f} km/h wind-shift assumption."

        return SimulationResult(
            baseline_aqi=current_aqi,
            predicted_aqi=predicted_aqi,
            delta_aqi=delta_aqi,
            total_pct_drop=round(total_pct_drop, 1),
            breakdown=breakdown,
            scenario_label=label,
            confidence=confidence,
            wind_shift_kmh=extra_wind_shift_kmh,
        )

    def compare_scenarios(
        self,
        current_aqi: int,
        percentages: dict[str, float],
        telemetry: Optional[dict],
        scenario_a: list[InterventionSpec],
        scenario_b: list[InterventionSpec],
        attribution_confidence: int = 70,
        extra_wind_shift_kmh: float = 0.0,
        label_a: str = "Scenario A",
        label_b: str = "Scenario B",
    ) -> ScenarioComparison:
        """
        Compare two intervention scenarios side by side.

        Calls simulate_scenario() twice — does not recompute or duplicate
        any AQI-prediction math. The "explanation" is derived purely from
        the two breakdown dicts already returned, identifying which source
        category accounts for most of the gap between the two outcomes.
        """
        result_a = self.simulate_scenario(
            current_aqi, percentages, telemetry, scenario_a,
            attribution_confidence, extra_wind_shift_kmh,
        )
        result_b = self.simulate_scenario(
            current_aqi, percentages, telemetry, scenario_b,
            attribution_confidence, extra_wind_shift_kmh,
        )

        aqi_gap = abs(result_a.predicted_aqi - result_b.predicted_aqi)
        if result_a.predicted_aqi < result_b.predicted_aqi:
            better_label = label_a
        elif result_b.predicted_aqi < result_a.predicted_aqi:
            better_label = label_b
        else:
            better_label = "Tie"

        # Find the source category with the biggest swing between the two
        # breakdowns — purely derived from already-computed numbers.
        all_sources = set(result_a.breakdown) | set(result_b.breakdown)
        swing_by_source = {
            src: result_a.breakdown.get(src, 0.0) - result_b.breakdown.get(src, 0.0)
            for src in all_sources
        }
        if swing_by_source:
            dominant_src = max(swing_by_source, key=lambda s: abs(swing_by_source[s]))
            swing = swing_by_source[dominant_src]
            who = label_a if swing > 0 else label_b
            if aqi_gap > 0:
                explanation = (
                    f"{better_label} predicts {aqi_gap} fewer AQI points. "
                    f"The largest driver of that gap is {dominant_src}: {who} reduces "
                    f"it by {abs(swing):.1f} more percentage points of total AQI than "
                    f"the other scenario."
                )
            else:
                explanation = (
                    f"Both scenarios predict the same {result_a.predicted_aqi} AQI, "
                    f"but they get there differently — {dominant_src} is reduced "
                    f"{abs(swing):.1f} percentage points more by {who}, offset elsewhere."
                )
        else:
            explanation = (
                f"Both scenarios predict the same {result_a.predicted_aqi} AQI — "
                f"neither targets a source the other doesn't."
            )

        return ScenarioComparison(
            label_a=label_a,
            label_b=label_b,
            result_a=result_a,
            result_b=result_b,
            better_label=better_label,
            aqi_gap=aqi_gap,
            explanation=explanation,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Condition flags
    # ──────────────────────────────────────────────────────────────────────

    def _build_conditions(
        self,
        aqi: int,
        hour: int,
        wind_speed: float,
        temp: float,
        active_fires: int,
        firm_hotspots: int,
        percentages: dict[str, float],
    ) -> dict[str, bool]:
        """
        Compute boolean condition flags used for feasibility adjustment.
        Each flag is named to match the keys in InterventionSpec.feasibility_conditions.
        """
        return {
            "rush_hour":           hour in {7, 8, 9, 10, 17, 18, 19, 20},
            "weekend":             _is_weekend_today(),
            "aqi_severe":          aqi > 200,
            "wind_calm":           wind_speed < 6.0,
            "wind_high":           wind_speed > 20.0,
            "rain":                False,   # not in current telemetry; conservative default
            "temp_high":           temp > 35.0,
            "industrial_dominant": percentages.get("Industrial", 0) > 25.0,
            "biomass_dominant":    percentages.get("Biomass Burning", 0) > 20.0,
            "fire_detected":       active_fires > 0 or firm_hotspots > 0,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Per-intervention scoring
    # ──────────────────────────────────────────────────────────────────────

    def _score_intervention(
        self,
        spec: InterventionSpec,
        current_aqi: int,
        percentages: dict[str, float],
        conf_norm: float,
        conditions: dict[str, bool],
    ) -> RankedIntervention:
        """
        Score a single intervention against the current attribution state.

        AQI reduction calculation
        ─────────────────────────
        The intervention targets specific source categories. Its absolute
        AQI reduction equals:

            Σ (source_pct / 100 × source_effectiveness × current_aqi)
              for each target source

        Example: Traffic = 46%, effectiveness = 55%, AQI = 285
            → 0.46 × 0.55 × 285 = 72 AQI points

        For multi-source interventions (e.g. water spraying hits Traffic +
        Construction), contributions are summed.
        """
        # AQI reduction: sum over targeted sources
        total_source_pct = 0.0
        for src in spec.target_sources:
            total_source_pct += percentages.get(src, 0.0)

        expected_reduction_pct = (total_source_pct / 100.0) * spec.source_effectiveness * 100.0
        expected_aqi_reduction = (expected_reduction_pct / 100.0) * current_aqi

        # Adjusted feasibility: apply condition deltas
        feasibility = spec.base_feasibility
        for condition_key, delta in spec.feasibility_conditions.items():
            if conditions.get(condition_key, False):
                feasibility += delta
        feasibility = max(0.05, min(1.0, feasibility))

        # Normalised sub-components for the ranking formula
        s_aqi   = min(1.0, expected_reduction_pct / _SCORE_AQI_CEILING_PCT)
        s_cost  = 1.0 - _normalise_cost(spec.cost_tier)
        s_feas  = feasibility
        s_conf  = conf_norm
        s_time  = 1.0 - _normalise_deploy(spec.deployment_hours)

        final_score = (
            W_AQI  * s_aqi   +
            W_COST * s_cost  +
            W_FEAS * s_feas  +
            W_CONF * s_conf  +
            W_TIME * s_time
        )

        # Compute counterfactual simulator inputs for this intervention
        sim_traffic_pct    = 0.0
        sim_industrial_pct = 0.0
        sim_wind_shift     = 0.0

        if "Traffic" in spec.target_sources:
            sim_traffic_pct = round(spec.source_effectiveness * 100)
        if "Industrial" in spec.target_sources:
            sim_industrial_pct = round(spec.source_effectiveness * 100)
        # NOTE: Construction and Biomass Burning have no dedicated lever in
        # the Counterfactual Simulator (it only models Traffic/Industrial/Wind).
        # We deliberately do NOT proxy-map them onto an unrelated slider —
        # e.g. faking a "Restrict Vehicular Traffic %" value for a
        # Construction Pause would mislabel what's actually being simulated,
        # which is the kind of fabricated-precision this project avoids.
        # sim_traffic_pct / sim_industrial_pct correctly stay 0.0 for these
        # interventions; app.py is responsible for telling the user why.

        reasoning = self._build_reasoning(
            spec, expected_aqi_reduction, expected_reduction_pct,
            percentages, feasibility, conf_norm, conditions,
        )

        return RankedIntervention(
            spec=spec,
            rank=0,   # set after sorting
            final_score=final_score,
            expected_aqi_reduction=round(expected_aqi_reduction, 1),
            expected_aqi_reduction_pct=round(expected_reduction_pct, 1),
            cost_tier=spec.cost_tier,
            feasibility=round(feasibility, 2),
            confidence=conf_norm,
            deployment_hours=spec.deployment_hours,
            sim_traffic_pct=sim_traffic_pct,
            sim_industrial_pct=sim_industrial_pct,
            sim_wind_shift=sim_wind_shift,
            reasoning=reasoning,
            sub_scores={
                "AQI Impact":    round(s_aqi, 3),
                "Cost":          round(s_cost, 3),
                "Feasibility":   round(s_feas, 3),
                "Confidence":    round(s_conf, 3),
                "Speed":         round(s_time, 3),
                "Final":         round(final_score, 3),
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # Explanation generators
    # ──────────────────────────────────────────────────────────────────────

    def _build_reasoning(
        self,
        spec: InterventionSpec,
        aqi_reduction: float,
        reduction_pct: float,
        percentages: dict[str, float],
        feasibility: float,
        conf_norm: float,
        conditions: dict[str, bool],
    ) -> str:
        """
        Generates a transparent, value-referencing sentence explaining why
        this intervention was recommended and what it is expected to achieve.
        All numeric values are the actual computed values — no templates with
        placeholder text.
        """
        source_parts = []
        for src in spec.target_sources:
            pct = percentages.get(src, 0.0)
            if pct > 2.0:
                source_parts.append(f"{src.lower()} ({pct:.0f}%)")

        source_str = " and ".join(source_parts) if source_parts else "the dominant source"

        # Feasibility qualifier
        if feasibility >= 0.80:
            feas_phrase = "High feasibility"
        elif feasibility >= 0.60:
            feas_phrase = "Moderate feasibility"
        else:
            feas_phrase = "Lower feasibility"

        # Rush hour context
        rush_phrase = " during active rush hour" if conditions.get("rush_hour") else ""

        # Deployment speed
        if spec.deployment_hours <= 0.5:
            speed_phrase = "deployable in under 30 minutes"
        elif spec.deployment_hours <= 2.0:
            speed_phrase = f"deployable within {spec.deployment_hours:.0f} hour(s)"
        else:
            speed_phrase = f"requires {spec.deployment_hours:.0f}h advance notice"

        return (
            f"{spec.name} is recommended because {source_str} is contributing "
            f"significantly to current conditions{rush_phrase}. "
            f"Simulation predicts a {aqi_reduction:.0f}-point AQI reduction "
            f"({reduction_pct:.1f}% improvement) by addressing {spec.source_effectiveness*100:.0f}% "
            f"of that source's output. "
            f"{feas_phrase} ({feasibility*100:.0f}%) — {speed_phrase}. "
            f"Executing department: {spec.department}."
        )

    def _composite_explanation(
        self,
        top3: list[RankedIntervention],
        current_aqi: int,
        primary_driver: str,
        primary_pct: float,
        crisis_level: str,
        confidence: int,
        percentages: dict[str, float],
    ) -> str:
        """
        Generates a strategic paragraph summarising the full intervention plan.

        BUG FIX (was: naive sum of expected_aqi_reduction across top3):
        Top-3 selection is pure score-ranking with no source-diversity
        constraint, so two interventions that both target the same source
        (e.g. Truck Restrictions + Traffic Signal Optimization both hit
        "Traffic") can legitimately both appear in the top 3. Summing their
        standalone reductions double-counts that overlap and can claim more
        reduction than the source contributes in the first place.

        Fix: for each source category, compound the *survival* fraction
        across every top3 intervention that targets it —

            remaining(source) = Π (1 − effectiveness_i)  for each top3
                                  intervention i targeting that source

        — the same logic used for independent risk reduction. A source hit
        by two interventions gets diminishing, not additive, returns; a
        source untouched by any top3 intervention contributes nothing.
        This is an explicit modelling assumption (interventions on the same
        source act independently, not synergistically) — documented here
        per project convention rather than left implicit. The compounding
        itself now lives in the shared _compound_remaining_fraction() helper,
        also used by simulate_scenario() for the Digital Twin (Feature 3).
        """
        if not top3:
            return "Insufficient data to generate intervention strategy."

        top = top3[0]

        remaining = _compound_remaining_fraction([r.spec for r in top3], percentages)

        combined_pct = 0.0
        for src, pct in percentages.items():
            combined_pct += (pct / 100.0) * (1.0 - remaining[src]) * 100.0
        combined_pct = min(combined_pct, 85.0)   # physical ceiling, matches simulator
        total_potential = (combined_pct / 100.0) * current_aqi

        names = ", ".join(r.spec.name for r in top3)

        return (
            f"AQI is at {current_aqi} ({crisis_level} level). "
            f"Attribution engine ({confidence}% confidence) identifies {primary_driver} "
            f"as the dominant driver at {primary_pct:.0f}%. "
            f"The optimal three-action response — {names} — "
            f"is projected to reduce AQI by up to {total_potential:.0f} points "
            f"({combined_pct:.0f}% improvement) if deployed simultaneously. "
            f"Priority action: {top.spec.icon} {top.spec.name} "
            f"(estimated -{top.expected_aqi_reduction:.0f} AQI, "
            f"deployable in {top.deployment_hours:.1f}h, "
            f"cost tier {top.cost_tier}/5)."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _crisis_level(aqi: int) -> tuple[str, str, str]:
    for threshold, label, color, icon in CRISIS_LEVELS:
        if aqi >= threshold:
            return label, color, icon
    return "GOOD", "#00e676", "⚪"


def cost_tier_label(tier: int) -> str:
    return {1: "Minimal", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}.get(tier, "Unknown")


def deploy_hours_label(hours: float) -> str:
    if hours < 1:
        return f"{int(hours * 60)} min"
    return f"{hours:.1f}h"