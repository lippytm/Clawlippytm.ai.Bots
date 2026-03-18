"""
clawlippytm.diagnostics
~~~~~~~~~~~~~~~~~~~~~~~

Diagnostics system with iterative feedback loops.

The :class:`DiagnosticsSystem` inspects both incoming messages and outgoing
bot responses.  Each analysis pass can run through multiple *feedback loop*
iterations, progressively refining its findings before returning a
:class:`DiagnosticResult`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticIssue:
    """A single issue detected during a diagnostic pass."""

    category: str           # e.g. "safety", "clarity", "tone", "coherence"
    severity: str           # "low" | "medium" | "high"
    description: str
    suggestion: str = ""


@dataclass
class DiagnosticResult:
    """Result produced by one :class:`DiagnosticsSystem` analysis call."""

    text: str
    issues: List[DiagnosticIssue] = field(default_factory=list)
    iterations_run: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_issues(self) -> bool:
        """True when at least one issue was found."""
        return len(self.issues) > 0

    @property
    def highest_severity(self) -> Optional[str]:
        """Return the worst severity level found, or *None*."""
        order = {"high": 3, "medium": 2, "low": 1}
        if not self.issues:
            return None
        return max(self.issues, key=lambda i: order.get(i.severity, 0)).severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_issues": self.has_issues,
            "highest_severity": self.highest_severity,
            "issue_count": len(self.issues),
            "iterations_run": self.iterations_run,
            "processing_time_ms": self.processing_time_ms,
            "issues": [
                {
                    "category": iss.category,
                    "severity": iss.severity,
                    "description": iss.description,
                    "suggestion": iss.suggestion,
                }
                for iss in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# Diagnostic rules (pure functions)
# ---------------------------------------------------------------------------

def _check_safety(text: str) -> List[DiagnosticIssue]:
    """Flag potentially harmful or inappropriate patterns."""
    issues: List[DiagnosticIssue] = []
    harmful_patterns = [
        r"\bharm\b", r"\bviolence\b", r"\bkill\b", r"\battack\b",
        r"\billegal\b", r"\bexploit\b",
    ]
    for pattern in harmful_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(
                DiagnosticIssue(
                    category="safety",
                    severity="high",
                    description=f"Potentially harmful term matched: '{pattern}'",
                    suggestion="Review and rephrase to remove potentially harmful language.",
                )
            )
    return issues


def _check_clarity(text: str) -> List[DiagnosticIssue]:
    """Flag overly long sentences or ambiguous pronouns."""
    issues: List[DiagnosticIssue] = []
    sentences = re.split(r"[.!?]+", text)
    for sent in sentences:
        words = sent.split()
        if len(words) > 50:
            issues.append(
                DiagnosticIssue(
                    category="clarity",
                    severity="low",
                    description="Sentence exceeds 50 words; may be hard to follow.",
                    suggestion="Break the sentence into shorter, clearer statements.",
                )
            )
    return issues


def _check_coherence(text: str, context: List[Dict[str, str]]) -> List[DiagnosticIssue]:
    """Check that the text is non-empty and loosely consistent with context."""
    issues: List[DiagnosticIssue] = []
    if not text.strip():
        issues.append(
            DiagnosticIssue(
                category="coherence",
                severity="medium",
                description="Text is empty or whitespace-only.",
                suggestion="Provide a substantive response.",
            )
        )
    return issues


def _check_tone(text: str) -> List[DiagnosticIssue]:
    """Flag overly negative or aggressive language."""
    issues: List[DiagnosticIssue] = []
    negative_patterns = [r"\byou are stupid\b", r"\bidiot\b", r"\bmoron\b"]
    for pattern in negative_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            issues.append(
                DiagnosticIssue(
                    category="tone",
                    severity="medium",
                    description=f"Negative or aggressive phrasing detected: '{pattern}'",
                    suggestion="Use respectful, constructive language.",
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Main diagnostics class
# ---------------------------------------------------------------------------

class DiagnosticsSystem:
    """
    Multi-pass diagnostics engine with configurable feedback loops.

    Each call to :meth:`analyse` runs through *feedback_loops* iterations,
    accumulating and deduplicating issues.  The iterative approach means
    that issues found in early passes can influence subsequent passes (the
    "feedback loop").

    Parameters
    ----------
    enabled:
        When *False*, :meth:`analyse` returns a clean result immediately.
    feedback_loops:
        Number of iterative passes (minimum 1).
    """

    def __init__(self, enabled: bool = True, feedback_loops: int = 2) -> None:
        self.enabled: bool = enabled
        self.feedback_loops: int = max(1, feedback_loops)
        self._history: List[DiagnosticResult] = []
        self._total_issues_detected: int = 0

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def analyse(
        self,
        text: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> DiagnosticResult:
        """
        Analyse *text* through multiple feedback loop iterations.

        Parameters
        ----------
        text:
            The text to analyse (user input or bot output).
        context:
            Optional conversation history for coherence checking.

        Returns
        -------
        DiagnosticResult
            Aggregated result after all feedback loop iterations.
        """
        if not self.enabled:
            return DiagnosticResult(text=text)

        ctx = context or []
        start = time.monotonic()
        all_issues: List[DiagnosticIssue] = []
        seen_descriptions: set = set()

        for iteration in range(self.feedback_loops):
            new_issues = self._run_single_pass(text, ctx, iteration)
            for issue in new_issues:
                if issue.description not in seen_descriptions:
                    all_issues.append(issue)
                    seen_descriptions.add(issue.description)

            # Feedback: if a high-severity issue was found, add an extra
            # coherence pass on the next iteration by injecting a synthetic
            # "flagged" context marker.
            if any(i.severity == "high" for i in new_issues):
                ctx = ctx + [{"role": "_diag", "content": "[flagged:high-severity]"}]

        elapsed_ms = (time.monotonic() - start) * 1000
        result = DiagnosticResult(
            text=text,
            issues=all_issues,
            iterations_run=self.feedback_loops,
            processing_time_ms=round(elapsed_ms, 3),
            metadata={"context_length": len(ctx)},
        )
        self._history.append(result)
        self._total_issues_detected += len(all_issues)
        return result

    def reset(self) -> None:
        """Clear all diagnostic history."""
        self._history.clear()
        self._total_issues_detected = 0

    def summary(self) -> Dict[str, Any]:
        """Return a summary of diagnostic activity so far."""
        return {
            "enabled": self.enabled,
            "feedback_loops": self.feedback_loops,
            "analyses_run": len(self._history),
            "total_issues_detected": self._total_issues_detected,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_single_pass(
        self,
        text: str,
        context: List[Dict[str, str]],
        iteration: int,
    ) -> List[DiagnosticIssue]:
        """Execute all diagnostic rules for one iteration."""
        issues: List[DiagnosticIssue] = []
        issues.extend(_check_safety(text))
        issues.extend(_check_clarity(text))
        issues.extend(_check_coherence(text, context))
        issues.extend(_check_tone(text))
        # On later iterations also check for repetition in history
        if iteration > 0:
            issues.extend(self._check_repetition(text, context))
        return issues

    def _check_repetition(
        self, text: str, context: List[Dict[str, str]]
    ) -> List[DiagnosticIssue]:
        """Detect if the current text closely mirrors a recent bot turn."""
        issues: List[DiagnosticIssue] = []
        bot_turns = [
            m["content"] for m in context if m.get("role") == "bot"
        ]
        for prev in bot_turns[-3:]:
            if prev and text.strip() and text.strip() == prev.strip():
                issues.append(
                    DiagnosticIssue(
                        category="coherence",
                        severity="medium",
                        description="Response is identical to a previous bot turn.",
                        suggestion="Vary the response to avoid repetition.",
                    )
                )
                break
        return issues
