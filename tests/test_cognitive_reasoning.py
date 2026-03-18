"""Tests for the cognitive_reasoning module."""

import pytest

from cognitive_reasoning.knowledge_base import KnowledgeBase
from cognitive_reasoning.reasoning_engine import ReasoningEngine
from cognitive_reasoning.decision_maker import DecisionMaker


# -------------------------------------------------------------------------
# KnowledgeBase
# -------------------------------------------------------------------------

class TestKnowledgeBase:
    def test_store_and_retrieve(self):
        kb = KnowledgeBase()
        kb.store("color", "blue")
        entry = kb.retrieve("color")
        assert entry is not None
        assert entry.value == "blue"
        assert entry.confidence == 1.0

    def test_update_existing_entry(self):
        kb = KnowledgeBase()
        kb.store("x", 1)
        kb.store("x", 2, confidence=0.5, source="test")
        entry = kb.retrieve("x")
        assert entry.value == 2
        assert entry.confidence == 0.5

    def test_retrieve_missing_returns_none(self):
        kb = KnowledgeBase()
        assert kb.retrieve("missing") is None

    def test_delete(self):
        kb = KnowledgeBase()
        kb.store("k", "v")
        assert kb.delete("k") is True
        assert kb.retrieve("k") is None
        assert kb.delete("k") is False  # already gone

    def test_search_by_prefix(self):
        kb = KnowledgeBase()
        kb.store("sensor.temp", 42)
        kb.store("sensor.humidity", 80)
        kb.store("config.timeout", 30)
        results = kb.search("sensor.")
        assert len(results) == 2

    def test_high_confidence_filter(self):
        kb = KnowledgeBase()
        kb.store("a", 1, confidence=0.9)
        kb.store("b", 2, confidence=0.5)
        results = kb.high_confidence(0.8)
        assert len(results) == 1
        assert results[0].key == "a"

    def test_len(self):
        kb = KnowledgeBase()
        assert len(kb) == 0
        kb.store("x", 1)
        assert len(kb) == 1

    def test_empty_key_raises(self):
        kb = KnowledgeBase()
        with pytest.raises(ValueError):
            kb.store("", "value")

    def test_clear(self):
        kb = KnowledgeBase()
        kb.store("x", 1)
        kb.clear()
        assert len(kb) == 0


# -------------------------------------------------------------------------
# ReasoningEngine
# -------------------------------------------------------------------------

class TestReasoningEngine:
    def test_reason_no_rules(self):
        engine = ReasoningEngine()
        chain = engine.reason("test query")
        assert chain.query == "test query"
        assert chain.confidence == 0.0
        assert chain.final_answer == {}

    def test_single_rule_fires(self):
        kb = KnowledgeBase()
        kb.store("temperature_c", 100)
        engine = ReasoningEngine(kb)
        engine.register_rule(
            "boiling",
            lambda ctx: ("state", "boiling") if ctx.get("temperature_c", 0) >= 100 else None,
        )
        chain = engine.reason("water state")
        assert chain.final_answer.get("state") == "boiling"
        assert chain.confidence == 1.0

    def test_rule_does_not_fire(self):
        kb = KnowledgeBase()
        kb.store("temperature_c", 20)
        engine = ReasoningEngine(kb)
        engine.register_rule(
            "boiling",
            lambda ctx: ("state", "boiling") if ctx.get("temperature_c", 0) >= 100 else None,
        )
        chain = engine.reason("water state")
        assert chain.final_answer.get("state") is None

    def test_context_overrides_kb(self):
        kb = KnowledgeBase()
        kb.store("value", 5)
        engine = ReasoningEngine(kb)
        chain = engine.reason("test", context={"value": 99})
        assert chain.final_answer["value"] == 99

    def test_unregister_rule(self):
        engine = ReasoningEngine()
        engine.register_rule("r", lambda ctx: ("x", 1))
        engine.unregister_rule("r")
        chain = engine.reason("q")
        assert "x" not in chain.final_answer

    def test_empty_rule_name_raises(self):
        engine = ReasoningEngine()
        with pytest.raises(ValueError):
            engine.register_rule("", lambda ctx: None)


# -------------------------------------------------------------------------
# DecisionMaker
# -------------------------------------------------------------------------

class TestDecisionMaker:
    def test_single_candidate_chosen(self):
        dm = DecisionMaker()
        decision = dm.decide(["only_action"])
        assert decision.chosen_action == "only_action"

    def test_scorer_influences_choice(self):
        dm = DecisionMaker()
        dm.register_scorer("prefer_b", lambda action, ctx: 1.0 if action == "b" else 0.0)
        decision = dm.decide(["a", "b", "c"])
        assert decision.chosen_action == "b"

    def test_multiple_scorers_aggregated(self):
        dm = DecisionMaker()
        dm.register_scorer("s1", lambda action, ctx: 1.0 if action == "x" else 0.0)
        dm.register_scorer("s2", lambda action, ctx: 2.0 if action == "x" else 0.0)
        decision = dm.decide(["x", "y"])
        assert decision.chosen_action == "x"
        assert decision.score == 3.0

    def test_empty_candidates_raises(self):
        dm = DecisionMaker()
        with pytest.raises(ValueError):
            dm.decide([])

    def test_context_passed_to_scorer(self):
        dm = DecisionMaker()
        dm.register_scorer("ctx_scorer",
                            lambda action, ctx: ctx.get("priority", 0) if action == "a" else 0)
        decision = dm.decide(["a", "b"], context={"priority": 5})
        assert decision.chosen_action == "a"
        assert decision.score == 5

    def test_unregister_scorer(self):
        dm = DecisionMaker()
        dm.register_scorer("s", lambda action, ctx: 10.0)
        dm.unregister_scorer("s")
        decision = dm.decide(["a"])
        assert decision.score == 0.0
