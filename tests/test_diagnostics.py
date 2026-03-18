"""
Tests for clawlippytm.diagnostics
"""
import pytest
from clawlippytm import DiagnosticsSystem, DiagnosticResult, DiagnosticIssue


class TestDiagnosticIssue:
    def test_fields(self):
        issue = DiagnosticIssue(
            category="safety",
            severity="high",
            description="Something harmful.",
            suggestion="Fix it.",
        )
        assert issue.category == "safety"
        assert issue.severity == "high"


class TestDiagnosticResult:
    def test_has_issues_false_when_empty(self):
        result = DiagnosticResult(text="hello")
        assert not result.has_issues
        assert result.highest_severity is None

    def test_has_issues_true_when_issues_present(self):
        issue = DiagnosticIssue(category="tone", severity="low", description="meh")
        result = DiagnosticResult(text="hello", issues=[issue])
        assert result.has_issues
        assert result.highest_severity == "low"

    def test_highest_severity_ordering(self):
        issues = [
            DiagnosticIssue(category="a", severity="low", description="a"),
            DiagnosticIssue(category="b", severity="high", description="b"),
            DiagnosticIssue(category="c", severity="medium", description="c"),
        ]
        result = DiagnosticResult(text="test", issues=issues)
        assert result.highest_severity == "high"

    def test_to_dict(self):
        result = DiagnosticResult(text="test", iterations_run=2)
        d = result.to_dict()
        assert d["iterations_run"] == 2
        assert "issues" in d


class TestDiagnosticsSystem:
    def test_disabled_returns_clean_result(self):
        system = DiagnosticsSystem(enabled=False)
        result = system.analyse("This is fine.")
        assert not result.has_issues
        assert result.iterations_run == 0

    def test_safety_issue_detected(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=1)
        result = system.analyse("I want to harm people.")
        safety_issues = [i for i in result.issues if i.category == "safety"]
        assert len(safety_issues) > 0
        assert safety_issues[0].severity == "high"

    def test_tone_issue_detected(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=1)
        result = system.analyse("You are stupid and an idiot.")
        tone_issues = [i for i in result.issues if i.category == "tone"]
        assert len(tone_issues) > 0

    def test_clarity_issue_for_long_sentence(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=1)
        long_sentence = " ".join(["word"] * 60) + "."
        result = system.analyse(long_sentence)
        clarity_issues = [i for i in result.issues if i.category == "clarity"]
        assert len(clarity_issues) > 0

    def test_coherence_issue_for_empty_text(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=1)
        result = system.analyse("   ")
        coherence_issues = [i for i in result.issues if i.category == "coherence"]
        assert len(coherence_issues) > 0

    def test_feedback_loops_run(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=3)
        result = system.analyse("Normal text.")
        assert result.iterations_run == 3

    def test_deduplication_across_iterations(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=3)
        result = system.analyse("I want to harm you, you idiot.")
        # The same issues should not appear twice even with multiple loops
        descriptions = [i.description for i in result.issues]
        assert len(descriptions) == len(set(descriptions))

    def test_high_severity_triggers_extra_context(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=2)
        result = system.analyse("I will attack you.")
        assert result.has_issues
        assert result.highest_severity == "high"

    def test_reset_clears_history(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=1)
        system.analyse("test")
        assert system.summary()["analyses_run"] == 1
        system.reset()
        assert system.summary()["analyses_run"] == 0

    def test_summary_keys(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=2)
        s = system.summary()
        assert "enabled" in s
        assert "feedback_loops" in s
        assert "analyses_run" in s
        assert "total_issues_detected" in s

    def test_repetition_detection(self):
        system = DiagnosticsSystem(enabled=True, feedback_loops=2)
        context = [{"role": "bot", "content": "Hello there."}]
        result = system.analyse("Hello there.", context=context)
        coherence_issues = [i for i in result.issues if i.category == "coherence"]
        repetition_issues = [
            i for i in coherence_issues if "identical" in i.description
        ]
        assert len(repetition_issues) > 0
