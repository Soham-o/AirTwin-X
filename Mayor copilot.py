"""
mayor_copilot.py
─────────────────────────────────────────────────────────────────────────────
Mayor Copilot for AirTwin X — Feature 5.

This is an ORCHESTRATOR, not an AI model. It holds no pollution-science,
ranking, simulation, or health-economics logic of its own — every fact it
states is read directly off dataclasses already produced by:

    AttributionResult        (attribution_engine.py)
    CommandCenterOutput       (intervention_agent.py)
    SimulationResult / ScenarioComparison  (intervention_agent.py)
    HealthEconomicImpact      (health_economic_engine.py)

Design choice — deterministic intent matching, not an LLM
────────────────────────────────────────────────────────────────────────────
A free-form LLM chatbot bolted on top of this pipeline would risk exactly
what this feature is required not to do: invent numbers, paraphrase past
the point of accuracy, or answer confidently about something no upstream
module actually computed. So this Copilot instead:

  1. Classifies the question into one of a known set of supported intents
     using keyword matching (same "no randomness, deterministic" principle
     as InterventionAgent's reasoning generation — template-based, not a
     black box).
  2. Each intent handler is a pure function of DecisionContext: it pulls
     specific fields, formats them, and returns text plus a `sources` list
     naming exactly which upstream module/field backs every sentence.
  3. If the question doesn't match a known intent, the Copilot says so
     explicitly and lists what it CAN answer — never guesses, never
     fabricates an answer to look more capable.

This trades "can answer literally anything" for "every answer is provably
traceable to a computed value" — the explicit requirement here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

# These imports are for type hints ONLY — mayor_copilot.py never constructs
# any of these objects, it only reads fields off instances passed in by app.py.
# Keeping them TYPE_CHECKING-only means a missing/broken attribution_engine,
# intervention_agent, or health_economic_engine on the user's machine will NOT
# prevent the Copilot from loading. app.py's own import guards already handle
# whether those engines are available.
if TYPE_CHECKING:  # pragma: no cover
    from intervention_agent import (
        CommandCenterOutput, SimulationResult, ScenarioComparison, RankedIntervention,
    )
    from health_economic_engine import HealthEconomicImpact
    from attribution_engine import AttributionResult

# ─────────────────────────────────────────────────────────────────────────
# Context bundle — every fact the Copilot is allowed to talk about lives here
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class DecisionContext:
    """
    Snapshot of everything the pipeline has computed so far this session.
    Built once per Streamlit run from objects already in session_state —
    the Copilot never computes anything that should live here instead.
    """
    attribution: Optional["AttributionResult"] = None
    command_center: Optional["CommandCenterOutput"] = None
    telemetry: Optional[dict] = None
    # (label, SimulationResult) pairs — "" label for a single-scenario run,
    # named labels ("Scenario A"/"Scenario B") when a comparison was run.
    simulations: list[tuple[str, "SimulationResult"]] = field(default_factory=list)
    comparison: Optional["ScenarioComparison"] = None
    # (label, HealthEconomicImpact) pairs, same labelling convention.
    health_impacts: list[tuple[str, "HealthEconomicImpact"]] = field(default_factory=list)


@dataclass
class CopilotAnswer:
    text: str
    sources: list[str] = field(default_factory=list)   # e.g. ["AttributionResult.percentages", "CommandCenterOutput.composite_explanation"]
    intent: str = "unknown"


# ─────────────────────────────────────────────────────────────────────────
# Intent keyword table — deterministic matching, no ML/LLM classifier
# ─────────────────────────────────────────────────────────────────────────
# Each intent maps to a list of keyword groups; a question matches an
# intent if ANY group's keywords are ALL present (simple AND-within-OR).
_INTENT_KEYWORDS: dict[str, list[list[str]]] = {
    "why_aqi_increasing": [
        ["why", "aqi", "increas"], ["why", "aqi", "high"], ["why", "aqi", "rising"],
        ["why", "pollution", "increas"], ["why", "is", "aqi"],
        ["aqi", "high", "wrong"], ["what", "wrong"], ["why", "bad"],
        ["so", "high"], ["this", "bad"],
    ],
    "primary_source": [
        ["primary", "source"], ["main", "source"], ["biggest", "source"],
        ["dominant", "source"], ["what", "causing"], ["what", "is", "polluting"],
        ["causing", "pollution"], ["causing", "this"], ["what", "cause"],
    ],
    "why_intervention": [
        ["why", "recommend"], ["why", "suggest"], ["why", "top", "action"],
        ["why", "this", "intervention"], ["why", "choose"],
        ["should", "we", "do"], ["should", "do", "this"],
        ["why", "this", "action"], ["why", "this", "option"],
    ],
    "what_if_other": [
        ["what", "if", "instead"], ["what", "if", "we", "choose"],
        ["what", "if", "we", "pick"], ["compare"], ["alternative"], ["other", "option"],
        ["another", "intervention"], ["different", "intervention"],
        ["choose", "another"], ["pick", "another"], ["other", "scenario"],
    ],
    "how_many_benefit": [
        ["how", "many", "people", "benefit"], ["how", "many", "benefit"],
        ["population", "protected"], ["who", "benefits"],
        ["how", "many", "citizen"], ["citizen", "risk"], ["people", "risk"],
        ["how", "many", "affected"], ["residents", "affected"],
    ],
    "how_much_saved": [
        ["how", "much", "money"], ["how", "much", "saved"], ["savings"],
        ["cost", "saving"], ["economic", "benefit"], ["productivity"],
        ["cost", "us"], ["what", "does", "cost"], ["financial"],
        ["rupee"], ["lakh"], ["crore"],
    ],
    "why_confidence": [
        ["why", "confidence"], ["how", "confident"], ["confidence", "score"],
        ["how", "sure"], ["how", "certain"], ["how", "accurate"],
        ["trust", "this"], ["reliable"], ["certainty"],
    ],
    "health_impact": [
        ["hospitalization"], ["asthma"], ["health", "impact"], ["daly"],
        ["health", "benefit"], ["health", "effect"], ["medical"],
        ["hospital"], ["breathing"], ["respiratory"],
    ],
}


def _match_intent(question: str) -> str:
    q = question.lower()
    for intent, groups in _INTENT_KEYWORDS.items():
        for group in groups:
            if all(kw in q for kw in group):
                return intent
    return "unsupported"


class MayorCopilot:
    """
    Stateless orchestrator. Call ask(question, context) per query — no
    internal state carried between calls, mirroring InterventionAgent's
    and HealthEconomicEngine's design.
    """

    SUPPORTED_QUESTIONS = [
        "Why is AQI increasing?",
        "What is the primary pollution source today?",
        "Why are you recommending this intervention?",
        "What happens if we choose another intervention?",
        "How many people benefit?",
        "How much money could be saved?",
        "Why is the confidence score X%?",
        "What's the health impact?",
    ]

    def ask(self, question: str, context: DecisionContext) -> CopilotAnswer:
        intent = _match_intent(question)
        handler = getattr(self, f"_handle_{intent}", self._handle_unsupported)
        return handler(context)

    # ────────────────────────────────────────────────────────────────────
    # Intent handlers — each is a pure read of `context`, no new logic
    # ────────────────────────────────────────────────────────────────────

    def _handle_why_aqi_increasing(self, ctx: DecisionContext) -> CopilotAnswer:
        attr = ctx.attribution
        cc = ctx.command_center
        if attr is None or cc is None:
            return self._missing_data("Source Attribution Engine hasn't run yet for this location.")

        sorted_sources = sorted(attr.percentages.items(), key=lambda kv: -kv[1])
        top_two = sorted_sources[:2]
        breakdown_str = " and ".join(f"{name} ({pct:.0f}%)" for name, pct in top_two)

        text = (
            f"Current AQI is {cc.current_aqi} ({cc.crisis_level}). The Source "
            f"Attribution Engine identifies {breakdown_str} as the leading "
            f"contributors right now, at {attr.confidence}% confidence. "
            f"{attr.explanation}"
        )
        return CopilotAnswer(
            text=text,
            sources=["AttributionResult.percentages", "AttributionResult.explanation",
                      "CommandCenterOutput.current_aqi", "CommandCenterOutput.crisis_level"],
            intent="why_aqi_increasing",
        )

    def _handle_primary_source(self, ctx: DecisionContext) -> CopilotAnswer:
        attr = ctx.attribution
        if attr is None:
            return self._missing_data("Source Attribution Engine hasn't run yet for this location.")

        text = (
            f"The primary pollution source today is **{attr.primary_source}**, "
            f"contributing {attr.percentages.get(attr.primary_source, 0):.0f}% of "
            f"the current attribution breakdown, at {attr.confidence}% confidence. "
            f"Full breakdown: " + ", ".join(
                f"{name} {pct:.0f}%" for name, pct in
                sorted(attr.percentages.items(), key=lambda kv: -kv[1])
            ) + "."
        )
        return CopilotAnswer(
            text=text,
            sources=["AttributionResult.primary_source", "AttributionResult.percentages",
                      "AttributionResult.confidence"],
            intent="primary_source",
        )

    def _handle_why_intervention(self, ctx: DecisionContext) -> CopilotAnswer:
        cc = ctx.command_center
        if cc is None or not cc.interventions:
            return self._missing_data("the Intervention Command Engine hasn't produced a ranking yet.")

        top: RankedIntervention = cc.interventions[0]
        text = (
            f"**{top.spec.icon} {top.spec.name}** is ranked #1 with a composite "
            f"score of {top.final_score*100:.0f}/100, combining expected AQI impact, "
            f"cost ({top.cost_tier}/5), feasibility ({top.feasibility*100:.0f}%), "
            f"attribution confidence, and deployment speed "
            f"({top.deployment_hours:.1f}h). The engine's own reasoning: "
            f"\"{top.reasoning}\""
        )
        return CopilotAnswer(
            text=text,
            sources=["RankedIntervention.final_score", "RankedIntervention.reasoning",
                      "RankedIntervention.cost_tier", "RankedIntervention.feasibility",
                      "RankedIntervention.deployment_hours"],
            intent="why_intervention",
        )

    def _handle_what_if_other(self, ctx: DecisionContext) -> CopilotAnswer:
        if ctx.comparison is not None:
            c = ctx.comparison
            text = (
                f"Comparing **{c.label_a}** (predicted AQI {c.result_a.predicted_aqi}, "
                f"{c.result_a.confidence}% confidence) against **{c.label_b}** "
                f"(predicted AQI {c.result_b.predicted_aqi}, {c.result_b.confidence}% "
                f"confidence): {c.explanation}"
            )
            return CopilotAnswer(
                text=text,
                sources=["ScenarioComparison.result_a", "ScenarioComparison.result_b",
                          "ScenarioComparison.explanation"],
                intent="what_if_other",
            )
        if len(ctx.simulations) >= 2:
            (label_a, a), (label_b, b) = ctx.simulations[0], ctx.simulations[1]
            text = (
                f"**{label_a or 'Scenario 1'}** predicts AQI {a.predicted_aqi} "
                f"({a.total_pct_drop:.1f}% improvement). **{label_b or 'Scenario 2'}** "
                f"predicts AQI {b.predicted_aqi} ({b.total_pct_drop:.1f}% improvement). "
                f"Run the Digital Twin's 'Compare against a second scenario' option "
                f"for a detailed source-level explanation of the gap."
            )
            return CopilotAnswer(
                text=text,
                sources=["SimulationResult.predicted_aqi", "SimulationResult.total_pct_drop"],
                intent="what_if_other",
            )
        return self._missing_data(
            "no second scenario has been simulated yet — use the Digital Twin's "
            "'Compare against a second scenario' option, then ask again."
        )

    def _handle_how_many_benefit(self, ctx: DecisionContext) -> CopilotAnswer:
        if not ctx.health_impacts:
            return self._missing_data("the Health & Economic Impact Engine hasn't run for a scenario yet.")
        label, impact = ctx.health_impacts[0]
        prefix = f"For **{label.strip(': ')}**: " if label else ""
        text = (
            f"{prefix}an estimated {impact.population_protected:,} residents are "
            f"protected by this scenario's predicted {impact.delta_pm25:.1f} µg/m³ "
            f"PM2.5 reduction — translating to roughly {impact.hospitalizations_avoided} "
            f"avoided hospitalizations and {impact.asthma_attacks_avoided} avoided "
            f"childhood asthma attacks ({impact.dalys_reduced:.2f} DALYs reduced)."
        )
        return CopilotAnswer(
            text=text,
            sources=["HealthEconomicImpact.population_protected",
                      "HealthEconomicImpact.hospitalizations_avoided",
                      "HealthEconomicImpact.asthma_attacks_avoided",
                      "HealthEconomicImpact.dalys_reduced"],
            intent="how_many_benefit",
        )

    def _handle_how_much_saved(self, ctx: DecisionContext) -> CopilotAnswer:
        if not ctx.health_impacts:
            return self._missing_data("the Health & Economic Impact Engine hasn't run for a scenario yet.")
        label, impact = ctx.health_impacts[0]
        prefix = f"For **{label.strip(': ')}**: " if label else ""
        text = (
            f"{prefix}an estimated ₹{impact.healthcare_savings_inr:,.0f} in avoided "
            f"healthcare spending and ₹{impact.productivity_gains_inr:,.0f} in avoided "
            f"productivity loss — combined social benefit score: "
            f"{impact.social_benefit_score}/100. See the 'Model assumptions & sources' "
            f"panel for exactly which government and peer-reviewed figures these are based on."
        )
        return CopilotAnswer(
            text=text,
            sources=["HealthEconomicImpact.healthcare_savings_inr",
                      "HealthEconomicImpact.productivity_gains_inr",
                      "HealthEconomicImpact.social_benefit_score"],
            intent="how_much_saved",
        )

    def _handle_why_confidence(self, ctx: DecisionContext) -> CopilotAnswer:
        attr = ctx.attribution
        if attr is None:
            return self._missing_data("Source Attribution Engine hasn't run yet for this location.")

        text = (
            f"The {attr.confidence}% attribution confidence reflects data quality: "
            f"{', '.join(attr.data_sources_used) if attr.data_sources_used else 'a mix of live and heuristic data sources'}. "
        )
        if ctx.simulations:
            label, sr = ctx.simulations[0]
            text += (
                f" The Digital Twin's prediction confidence for "
                f"{label.strip(': ') or 'this scenario'} ({sr.confidence}%) is derived "
                f"from this same attribution confidence, discounted slightly for each "
                f"additional intervention stacked in the scenario (more stacking means "
                f"more reliance on the 'sources act independently' modelling assumption)."
            )
            sources = ["AttributionResult.confidence", "AttributionResult.data_sources_used",
                       "SimulationResult.confidence"]
        else:
            sources = ["AttributionResult.confidence", "AttributionResult.data_sources_used"]
        return CopilotAnswer(text=text, sources=sources, intent="why_confidence")

    def _handle_health_impact(self, ctx: DecisionContext) -> CopilotAnswer:
        if not ctx.health_impacts:
            return self._missing_data("the Health & Economic Impact Engine hasn't run for a scenario yet.")
        label, impact = ctx.health_impacts[0]
        return CopilotAnswer(text=impact.explanation, sources=["HealthEconomicImpact.explanation"],
                              intent="health_impact")

    def _handle_unsupported(self, ctx: DecisionContext) -> CopilotAnswer:
        examples = "\n".join(f"  • {q}" for q in self.SUPPORTED_QUESTIONS)
        return CopilotAnswer(
            text=(
                "I can only answer questions grounded in what the pipeline has "
                "actually computed — I won't guess. Try one of these:\n" + examples
            ),
            sources=[],
            intent="unsupported",
        )

    def _missing_data(self, reason: str) -> CopilotAnswer:
        return CopilotAnswer(
            text=f"I can't answer that yet — {reason}",
            sources=[],
            intent="missing_data",
        )