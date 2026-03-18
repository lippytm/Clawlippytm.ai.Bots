"""Tests for the feedback_loops module."""

import pytest

from feedback_loops.feedback_manager import FeedbackManager, FeedbackPolarity, FeedbackRecord
from feedback_loops.evaluator import Evaluator
from feedback_loops.adapter import Adapter


# -------------------------------------------------------------------------
# FeedbackManager
# -------------------------------------------------------------------------

class TestFeedbackManager:
    def _record(self, source="agent-1", action="act", polarity=FeedbackPolarity.POSITIVE, score=0.8):
        return FeedbackRecord(source=source, action=action, polarity=polarity, score=score)

    def test_submit_and_retrieve(self):
        fm = FeedbackManager()
        fm.submit(self._record())
        records = fm.get_records()
        assert len(records) == 1

    def test_filter_by_source(self):
        fm = FeedbackManager()
        fm.submit(self._record(source="agent-1"))
        fm.submit(self._record(source="agent-2"))
        assert len(fm.get_records(source="agent-1")) == 1

    def test_filter_by_polarity(self):
        fm = FeedbackManager()
        fm.submit(self._record(polarity=FeedbackPolarity.POSITIVE))
        fm.submit(self._record(polarity=FeedbackPolarity.NEGATIVE))
        pos = fm.get_records(polarity=FeedbackPolarity.POSITIVE)
        assert len(pos) == 1
        assert pos[0].polarity == FeedbackPolarity.POSITIVE

    def test_handler_called_on_submit(self):
        fm = FeedbackManager()
        received = []
        fm.register_handler("collector", lambda r: received.append(r))
        fm.submit(self._record())
        assert len(received) == 1

    def test_unregister_handler(self):
        fm = FeedbackManager()
        received = []
        fm.register_handler("h", lambda r: received.append(r))
        fm.unregister_handler("h")
        fm.submit(self._record())
        assert len(received) == 0

    def test_average_score(self):
        fm = FeedbackManager()
        fm.submit(self._record(score=0.6))
        fm.submit(self._record(score=0.4))
        assert fm.average_score() == pytest.approx(0.5)

    def test_average_score_empty(self):
        fm = FeedbackManager()
        assert fm.average_score() is None

    def test_clear(self):
        fm = FeedbackManager()
        fm.submit(self._record())
        fm.clear()
        assert fm.get_records() == []

    def test_empty_handler_name_raises(self):
        fm = FeedbackManager()
        with pytest.raises(ValueError):
            fm.register_handler("", lambda r: None)


# -------------------------------------------------------------------------
# Evaluator
# -------------------------------------------------------------------------

class TestEvaluator:
    def test_exact_match_criterion(self):
        ev = Evaluator(pass_threshold=1.0)
        ev.register_criterion("exact", lambda out, exp: 1.0 if out == exp else 0.0)
        result = ev.evaluate(output="hello", expected="hello")
        assert result.passed
        assert result.overall_score == 1.0

    def test_failed_criterion(self):
        ev = Evaluator(pass_threshold=0.5)
        ev.register_criterion("exact", lambda out, exp: 1.0 if out == exp else 0.0)
        result = ev.evaluate(output="hello", expected="world")
        assert not result.passed
        assert result.overall_score == 0.0

    def test_no_criteria_returns_failed(self):
        ev = Evaluator()
        result = ev.evaluate(output="anything")
        assert not result.passed
        assert result.overall_score == 0.0

    def test_multiple_criteria_averaged(self):
        ev = Evaluator(pass_threshold=0.5)
        ev.register_criterion("c1", lambda out, exp: 1.0)
        ev.register_criterion("c2", lambda out, exp: 0.0)
        result = ev.evaluate(output="x")
        assert result.overall_score == pytest.approx(0.5)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            Evaluator(pass_threshold=1.5)

    def test_score_clamped_to_0_1(self):
        ev = Evaluator()
        ev.register_criterion("over", lambda out, exp: 5.0)
        result = ev.evaluate(output="x")
        assert result.overall_score <= 1.0

    def test_unregister_criterion(self):
        ev = Evaluator()
        ev.register_criterion("c", lambda out, exp: 1.0)
        ev.unregister_criterion("c")
        result = ev.evaluate("x")
        assert not result.passed


# -------------------------------------------------------------------------
# Adapter
# -------------------------------------------------------------------------

class TestAdapter:
    def _make_fm_with_records(self, n_negative=0, n_positive=0):
        fm = FeedbackManager()
        for _ in range(n_negative):
            fm.submit(FeedbackRecord("src", "act", FeedbackPolarity.NEGATIVE, score=0.1))
        for _ in range(n_positive):
            fm.submit(FeedbackRecord("src", "act", FeedbackPolarity.POSITIVE, score=0.9))
        return fm

    def test_no_strategy_no_change(self):
        fm = self._make_fm_with_records(n_negative=10)
        adapter = Adapter(fm)
        adapter.params = {"lr": 0.01}
        result = adapter.adapt()
        assert not result.changed

    def test_strategy_modifies_params(self):
        fm = self._make_fm_with_records(n_negative=6)
        adapter = Adapter(fm)
        adapter.params = {"lr": 0.01}

        def lr_decay(params, records):
            from feedback_loops.feedback_manager import FeedbackPolarity
            neg = [r for r in records if r.polarity == FeedbackPolarity.NEGATIVE]
            if len(neg) > 5:
                p = dict(params)
                p["lr"] = params["lr"] * 0.9
                return p
            return params

        adapter.register_strategy("lr_decay", lr_decay)
        result = adapter.adapt()
        assert result.changed
        assert adapter.params["lr"] == pytest.approx(0.009)

    def test_adapt_result_feedback_count(self):
        fm = self._make_fm_with_records(n_positive=3)
        adapter = Adapter(fm)
        result = adapter.adapt()
        assert result.feedback_count == 3

    def test_empty_strategy_name_raises(self):
        fm = FeedbackManager()
        adapter = Adapter(fm)
        with pytest.raises(ValueError):
            adapter.register_strategy("", lambda p, r: p)
