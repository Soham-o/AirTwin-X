"""
test_airtwin.py
────────────────────────────────────────────────────────────────
Regression tests for AirTwin X core modules.

Run with:  pytest test_airtwin.py -v

These tests guard the properties the project explicitly claims:
  1. Deterministic outputs — same inputs always produce same outputs
  2. Algorithm correctness — overlap-compounding, CPCB conversion, health math
  3. Edge case stability — AQI=0, all-zero percentages, empty interventions
  4. Copilot grounding — answers route to correct intents, never fabricate

They are not exhaustive; they are a regression baseline. Every future
change to core algorithms should be validated here before merging.
"""

import math
import datetime
import pytest

# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def typical_pct():
    return {
        "Traffic": 46.0, "Construction": 28.0, "Industrial": 18.0,
        "Biomass Burning": 8.0, "Weather Amplification": 0.0,
    }

@pytest.fixture
def typical_telemetry():
    return {"wind_speed": 5.0, "temp": 30, "active_fires": 0}

@pytest.fixture
def agent():
    from intervention_agent import InterventionAgent
    return InterventionAgent()

@pytest.fixture
def hee():
    from health_economic_engine import HealthEconomicEngine
    return HealthEconomicEngine()

@pytest.fixture
def copilot():
    from mayor_copilot import MayorCopilot
    return MayorCopilot()


# ─────────────────────────────────────────────────────────────────────────
# 1. Determinism — same inputs must produce exactly the same outputs
# ─────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_generate_is_deterministic(self, agent, typical_pct, typical_telemetry):
        a = agent.generate(285, typical_pct, 82, typical_telemetry, 8, 0)
        b = agent.generate(285, typical_pct, 82, typical_telemetry, 8, 0)
        assert [r.spec.name for r in a.interventions] == [r.spec.name for r in b.interventions]
        assert a.composite_explanation == b.composite_explanation

    def test_simulate_scenario_is_deterministic(self, agent, typical_pct, typical_telemetry):
        from intervention_agent import _LIBRARY_MAP
        truck = _LIBRARY_MAP["truck_restriction"]
        a = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck], 82)
        b = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck], 82)
        assert a.predicted_aqi == b.predicted_aqi
        assert a.confidence == b.confidence
        assert a.breakdown == b.breakdown

    def test_health_engine_is_deterministic(self, agent, hee, typical_pct, typical_telemetry):
        from intervention_agent import _LIBRARY_MAP
        from health_economic_engine import estimate_population_exposed
        truck = _LIBRARY_MAP["truck_restriction"]
        sim = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck], 82)
        pop = estimate_population_exposed(5.0)
        a = hee.assess(sim, pop)
        b = hee.assess(sim, pop)
        assert a.hospitalizations_avoided == b.hospitalizations_avoided
        assert a.healthcare_savings_inr == b.healthcare_savings_inr


# ─────────────────────────────────────────────────────────────────────────
# 2. Intervention ranking — algorithm correctness
# ─────────────────────────────────────────────────────────────────────────

class TestInterventionRanking:
    def test_weights_sum_to_one(self):
        from intervention_agent import W_AQI, W_COST, W_FEAS, W_CONF, W_TIME
        assert abs(W_AQI + W_COST + W_FEAS + W_CONF + W_TIME - 1.0) < 1e-9

    def test_ranks_are_sequential(self, agent, typical_pct, typical_telemetry):
        cc = agent.generate(285, typical_pct, 82, typical_telemetry, 8, 0)
        ranks = [r.rank for r in cc.interventions]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_scores_are_descending(self, agent, typical_pct, typical_telemetry):
        cc = agent.generate(285, typical_pct, 82, typical_telemetry, 8, 0)
        scores = [r.final_score for r in cc.interventions]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_all_8_interventions_ranked(self, agent, typical_pct, typical_telemetry):
        cc = agent.generate(285, typical_pct, 82, typical_telemetry, 8, 0)
        assert len(cc.interventions) == 8

    def test_aqi_reduction_bounded(self, agent, typical_pct, typical_telemetry):
        cc = agent.generate(285, typical_pct, 82, typical_telemetry, 8, 0)
        for ri in cc.interventions:
            assert ri.expected_aqi_reduction >= 0
            assert ri.expected_aqi_reduction <= 285  # can't reduce more than current AQI

    def test_feasibility_in_range(self, agent, typical_pct, typical_telemetry):
        cc = agent.generate(285, typical_pct, 82, typical_telemetry, 8, 0)
        for ri in cc.interventions:
            assert 0.05 <= ri.feasibility <= 1.0


# ─────────────────────────────────────────────────────────────────────────
# 3. Digital Twin / simulate_scenario
# ─────────────────────────────────────────────────────────────────────────

class TestDigitalTwin:
    def test_empty_intervention_returns_baseline(self, agent, typical_pct, typical_telemetry):
        sim = agent.simulate_scenario(285, typical_pct, typical_telemetry, [], 82)
        assert sim.predicted_aqi == 285
        assert sim.delta_aqi == 0
        assert sim.total_pct_drop == 0.0

    def test_physical_ceiling_respected(self, agent, typical_pct, typical_telemetry):
        """Even with all 8 interventions stacked, improvement must not exceed 85%."""
        from intervention_agent import _LIBRARY_MAP
        sim = agent.simulate_scenario(285, typical_pct, typical_telemetry,
                                       list(_LIBRARY_MAP.values()), 82)
        assert sim.total_pct_drop <= 85.0
        assert sim.predicted_aqi > 0

    def test_monotonic_more_interventions_more_reduction(self, agent, typical_pct, typical_telemetry):
        from intervention_agent import _LIBRARY_MAP
        truck = _LIBRARY_MAP["truck_restriction"]
        construction = _LIBRARY_MAP["construction_pause"]
        sim1 = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck], 82)
        sim2 = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck, construction], 82)
        assert sim2.total_pct_drop >= sim1.total_pct_drop

    def test_overlap_compounding_not_additive(self, agent, typical_pct, typical_telemetry):
        """Two Traffic interventions must compound, not add — prevents >100% reduction."""
        from intervention_agent import _LIBRARY_MAP
        truck  = _LIBRARY_MAP["truck_restriction"]
        signal = _LIBRARY_MAP["traffic_signal_optimization"]
        sim_truck  = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck], 82)
        sim_signal = agent.simulate_scenario(285, typical_pct, typical_telemetry, [signal], 82)
        sim_both   = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck, signal], 82)
        naive_sum = sim_truck.total_pct_drop + sim_signal.total_pct_drop
        assert sim_both.total_pct_drop < naive_sum, \
            "Two interventions on the same source must compound, not add"

    def test_confidence_discounts_with_stacking(self, agent, typical_pct, typical_telemetry):
        from intervention_agent import _LIBRARY_MAP
        truck = _LIBRARY_MAP["truck_restriction"]
        signal = _LIBRARY_MAP["traffic_signal_optimization"]
        water  = _LIBRARY_MAP["water_spraying"]
        sim1 = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck], 82)
        sim3 = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck, signal, water], 82)
        assert sim3.confidence < sim1.confidence

    def test_aqi_zero_stable(self, agent, typical_pct, typical_telemetry):
        sim = agent.simulate_scenario(0, typical_pct, typical_telemetry, [], 82)
        assert sim.predicted_aqi == 0
        assert sim.delta_aqi == 0

    def test_compare_scenarios_better_label(self, agent, typical_pct, typical_telemetry):
        from intervention_agent import _LIBRARY_MAP
        truck = _LIBRARY_MAP["truck_restriction"]
        smog  = _LIBRARY_MAP["smog_tower_activation"]
        cmp = agent.compare_scenarios(285, typical_pct, typical_telemetry,
                                       [truck], [smog], 82, label_a="A", label_b="B")
        assert cmp.better_label in ("A", "B", "Tie")
        assert cmp.aqi_gap == abs(cmp.result_a.predicted_aqi - cmp.result_b.predicted_aqi)


# ─────────────────────────────────────────────────────────────────────────
# 4. Health & Economic Impact Engine
# ─────────────────────────────────────────────────────────────────────────

class TestHealthEngine:
    def test_cpcb_aqi_pm25_breakpoints(self):
        """Spot-check the CPCB breakpoints against the published table."""
        from health_economic_engine import aqi_to_pm25
        # AQI 50 → PM2.5 30 µg/m³ (upper boundary of Good band)
        assert abs(aqi_to_pm25(50) - 30.0) < 0.5
        # AQI 100 → PM2.5 60 µg/m³ (upper boundary of Satisfactory band)
        assert abs(aqi_to_pm25(100) - 60.0) < 0.5

    def test_zero_improvement_zero_impact(self, agent, hee, typical_pct, typical_telemetry):
        from health_economic_engine import estimate_population_exposed
        sim = agent.simulate_scenario(285, typical_pct, typical_telemetry, [], 82)
        hi  = hee.assess(sim, estimate_population_exposed(5.0))
        assert hi.hospitalizations_avoided == 0
        assert hi.asthma_attacks_avoided == 0
        assert hi.healthcare_savings_inr == 0.0
        assert hi.population_protected == 0

    def test_social_benefit_score_bounded(self, agent, hee, typical_pct, typical_telemetry):
        from intervention_agent import _LIBRARY_MAP
        from health_economic_engine import estimate_population_exposed
        sim = agent.simulate_scenario(500, typical_pct, typical_telemetry,
                                       list(_LIBRARY_MAP.values()), 82)
        hi = hee.assess(sim, estimate_population_exposed(5.0))
        assert 0 <= hi.social_benefit_score <= 100

    def test_monotonic_larger_aqi_drop_more_health_benefit(self, agent, hee, typical_pct, typical_telemetry):
        from intervention_agent import _LIBRARY_MAP
        from health_economic_engine import estimate_population_exposed
        truck = _LIBRARY_MAP["truck_restriction"]
        water = _LIBRARY_MAP["water_spraying"]
        sim1 = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck], 82)
        sim2 = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck, water], 82)
        pop  = estimate_population_exposed(5.0)
        hi1  = hee.assess(sim1, pop)
        hi2  = hee.assess(sim2, pop)
        assert hi2.dalys_reduced >= hi1.dalys_reduced

    def test_assumptions_list_nonempty(self, agent, hee, typical_pct, typical_telemetry):
        from intervention_agent import _LIBRARY_MAP
        from health_economic_engine import estimate_population_exposed
        truck = _LIBRARY_MAP["truck_restriction"]
        sim  = agent.simulate_scenario(285, typical_pct, typical_telemetry, [truck], 82)
        hi   = hee.assess(sim, estimate_population_exposed(5.0))
        assert len(hi.assumptions) >= 8


# ─────────────────────────────────────────────────────────────────────────
# 5. Mayor Copilot — grounding and intent routing
# ─────────────────────────────────────────────────────────────────────────

class TestMayorCopilot:
    def test_all_supported_intents_route_correctly(self, copilot):
        from mayor_copilot import _match_intent, MayorCopilot
        cases = [
            ("Why is AQI increasing today?",         "why_aqi_increasing"),
            ("What is the primary pollution source?", "primary_source"),
            ("Why are you recommending this?",        "why_intervention"),
            ("What if we choose another option?",     "what_if_other"),
            ("How many people benefit?",              "how_many_benefit"),
            ("How much money could be saved?",        "how_much_saved"),
            ("Why is the confidence score 82%?",      "why_confidence"),
            ("What is the health impact?",            "health_impact"),
        ]
        for phrase, expected in cases:
            got = _match_intent(phrase)
            assert got == expected, f"'{phrase}' → got '{got}', expected '{expected}'"

    def test_unsupported_question_routes_to_unsupported(self, copilot):
        from mayor_copilot import _match_intent
        assert _match_intent("Tell me about cricket") == "unsupported"
        assert _match_intent("What is the capital of France?") == "unsupported"

    def test_missing_data_graceful_when_no_context(self, copilot):
        from mayor_copilot import DecisionContext
        ctx = DecisionContext()  # all fields None / empty
        for q in ["Why is AQI increasing?", "How many people benefit?", "How much saved?"]:
            ans = copilot.ask(q, ctx)
            assert ans.intent == "missing_data", \
                f"Expected missing_data for '{q}' with empty context, got {ans.intent}"

    def test_answers_have_sources_when_context_available(self, copilot, agent, hee, typical_pct, typical_telemetry):
        from mayor_copilot import DecisionContext
        from attribution_engine import AttributionResult, SourceScores
        from intervention_agent import _LIBRARY_MAP
        from health_economic_engine import estimate_population_exposed
        attr = AttributionResult(
            percentages=typical_pct, primary_source="Traffic", confidence=82,
            explanation="Traffic dominates.", data_sources_used=["OSM", "WAQI"],
            sub_scores=SourceScores(46, 28, 18, 8, 0),
        )
        cc  = agent.generate(285, typical_pct, 82, typical_telemetry, 8, 0)
        sim = agent.simulate_scenario(285, typical_pct, typical_telemetry,
                                       [_LIBRARY_MAP["truck_restriction"]], 82)
        hi  = hee.assess(sim, estimate_population_exposed(5.0))
        ctx = DecisionContext(attribution=attr, command_center=cc,
                              simulations=[("", sim)], health_impacts=[("", hi)])
        for q, expected in [
            ("Why is AQI increasing?",      "why_aqi_increasing"),
            ("How many people benefit?",    "how_many_benefit"),
            ("How much money could be saved?", "how_much_saved"),
        ]:
            ans = copilot.ask(q, ctx)
            assert ans.intent == expected
            assert len(ans.sources) > 0, f"Answer for '{q}' has no sources"
            assert len(ans.text) > 30


# ─────────────────────────────────────────────────────────────────────────
# 6. Weekend detection
# ─────────────────────────────────────────────────────────────────────────

class TestWeekendDetection:
    def test_is_weekend_returns_bool(self):
        from intervention_agent import _is_weekend_today
        result = _is_weekend_today()
        assert isinstance(result, bool)

    def test_is_weekend_consistent_with_python_weekday(self):
        from intervention_agent import _is_weekend_today
        expected = datetime.date.today().weekday() >= 5
        assert _is_weekend_today() == expected