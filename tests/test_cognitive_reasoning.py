"""
Tests for clawlippytm.cognitive_reasoning
"""
import pytest
from clawlippytm import CognitiveReasoner, ReasoningOutput, ReasoningStep


class TestReasoningStep:
    def test_to_dict(self):
        step = ReasoningStep(depth=1, question="What?", answer="This.")
        d = step.to_dict()
        assert d["depth"] == 1
        assert d["question"] == "What?"
        assert d["sub_steps"] == []


class TestReasoningOutput:
    def test_to_dict(self):
        output = ReasoningOutput(prompt="p", response="r", trace=[])
        d = output.to_dict()
        assert d["prompt"] == "p"
        assert d["response"] == "r"
        assert d["refined"] is False
        assert d["self_critique"] is None


class TestCognitiveReasoner:
    def test_reason_returns_output(self):
        reasoner = CognitiveReasoner(depth=2, self_critique=False)
        output = reasoner.reason("What is deep learning?")
        assert isinstance(output, ReasoningOutput)
        assert isinstance(output.response, str)
        assert len(output.response) > 0

    def test_trace_has_steps(self):
        reasoner = CognitiveReasoner(depth=2, self_critique=False)
        output = reasoner.reason("How does the internet work?")
        assert len(output.trace) > 0
        for step in output.trace:
            assert isinstance(step, ReasoningStep)
            assert step.depth == 1

    def test_depth_affects_sub_steps(self):
        shallow = CognitiveReasoner(depth=1, self_critique=False)
        deep = CognitiveReasoner(depth=3, self_critique=False)
        s_out = shallow.reason("Why is the sky blue?")
        d_out = deep.reason("Why is the sky blue?")
        # Deep reasoner should produce sub-steps; shallow should not
        shallow_sub_steps = sum(len(s.sub_steps) for s in s_out.trace)
        deep_sub_steps = sum(len(s.sub_steps) for s in d_out.trace)
        assert shallow_sub_steps == 0
        assert deep_sub_steps > 0

    def test_self_critique_enabled(self):
        reasoner = CognitiveReasoner(depth=2, self_critique=True)
        output = reasoner.reason("Compare Python vs Java.")
        # A response that is reasonable length should not be refined
        # unless the critique found issues, so we just check types
        assert isinstance(output.refined, bool)

    def test_self_critique_disabled(self):
        reasoner = CognitiveReasoner(depth=2, self_critique=False)
        output = reasoner.reason("Explain recursion.")
        assert output.self_critique is None
        assert not output.refined

    def test_summary_keys(self):
        reasoner = CognitiveReasoner(depth=2)
        s = reasoner.summary()
        assert "depth" in s
        assert "self_critique_enabled" in s
        assert "calls" in s
        assert "total_steps" in s

    def test_reset_clears_counters(self):
        reasoner = CognitiveReasoner(depth=1)
        reasoner.reason("test")
        assert reasoner.summary()["calls"] == 1
        reasoner.reset()
        assert reasoner.summary()["calls"] == 0

    def test_what_question_decomposition(self):
        reasoner = CognitiveReasoner(depth=2, self_critique=False)
        output = reasoner.reason("What is artificial intelligence?")
        questions = [step.question for step in output.trace]
        assert any("core concept" in q or "components" in q for q in questions)

    def test_how_question_decomposition(self):
        reasoner = CognitiveReasoner(depth=2, self_critique=False)
        output = reasoner.reason("How do neural networks learn?")
        questions = [step.question for step in output.trace]
        assert any("high level" in q.lower() or "steps" in q.lower() for q in questions)

    def test_why_question_decomposition(self):
        reasoner = CognitiveReasoner(depth=2, self_critique=False)
        output = reasoner.reason("Why is diversity important?")
        questions = [step.question for step in output.trace]
        assert any("reason" in q.lower() or "motivation" in q.lower() for q in questions)

    def test_history_included_in_response(self):
        reasoner = CognitiveReasoner(depth=1, self_critique=False)
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "bot", "content": "Hello"},
        ]
        output = reasoner.reason("What did we discuss?", history=history)
        # The synthesise method notes how many prior turns were used.
        assert "prior conversation" in output.response or str(len(history)) in output.response
