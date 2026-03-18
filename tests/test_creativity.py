"""
Tests for clawlippytm.creativity
"""
import pytest
from clawlippytm import CreativityEngine, CreativityAnnotation


class TestCreativityAnnotation:
    def test_to_dict(self):
        ann = CreativityAnnotation(analogies_added=2, narrative_framing=True)
        d = ann.to_dict()
        assert d["analogies_added"] == 2
        assert d["narrative_framing"] is True


class TestCreativityEngine:
    def test_enrich_returns_string(self):
        engine = CreativityEngine(temperature=0.5, seed=42)
        result = engine.enrich("This is a test response.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_enrich_at_zero_temperature(self):
        engine = CreativityEngine(temperature=0.0, seed=0)
        base = "Simple response."
        result = engine.enrich(base)
        # At zero temperature no enrichment beyond minimal substitution
        # The result should still be a string
        assert isinstance(result, str)

    def test_enrich_at_max_temperature(self):
        engine = CreativityEngine(temperature=1.0, seed=7)
        base = "This is an important response that we need to change and help people."
        result = engine.enrich(base)
        # At max temperature narrative framing should be applied
        narrative_hooks = [
            "Imagine", "Picture", "Consider", "Let's embark",
        ]
        assert any(hook in result for hook in narrative_hooks)

    def test_narrative_framing_threshold(self):
        engine_low = CreativityEngine(temperature=0.5, seed=1)
        engine_high = CreativityEngine(temperature=0.9, seed=1)
        base = "A thoughtful response."
        result_low = engine_low.enrich(base)
        result_high = engine_high.enrich(base)
        # Narrative hooks should appear in high-temp but not necessarily low-temp
        hooks = ["Imagine", "Picture", "Consider", "Let's embark"]
        assert any(h in result_high for h in hooks)
        assert not any(h in result_low for h in hooks)

    def test_lexical_substitution(self):
        # Use a response with known substitutable words
        engine = CreativityEngine(temperature=1.0, seed=0)
        base = "This is important and we need to help everyone and make things good."
        result = engine.enrich(base)
        # At temperature 1.0 some substitutions should occur
        substitutable = {
            "important": "pivotal",
            "help": "empower",
            "make": "craft",
            "good": "exceptional",
        }
        # At least one substitution should appear
        any_substituted = any(v in result for v in substitutable.values())
        assert any_substituted

    def test_summary(self):
        engine = CreativityEngine(temperature=0.7)
        engine.enrich("test")
        s = engine.summary()
        assert "temperature" in s
        assert "enrichments_applied" in s
        assert s["enrichments_applied"] == 1
        assert s["last_annotation"] is not None

    def test_temperature_clamped(self):
        engine_low = CreativityEngine(temperature=-0.5)
        engine_high = CreativityEngine(temperature=1.5)
        assert engine_low.temperature == 0.0
        assert engine_high.temperature == 1.0

    def test_seed_reproducibility(self):
        engine1 = CreativityEngine(temperature=0.8, seed=42)
        engine2 = CreativityEngine(temperature=0.8, seed=42)
        base = "What is the answer to everything?"
        assert engine1.enrich(base) == engine2.enrich(base)

    def test_no_error_on_empty_input(self):
        engine = CreativityEngine(temperature=0.7, seed=1)
        result = engine.enrich("")
        assert isinstance(result, str)
