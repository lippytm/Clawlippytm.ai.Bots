"""Evaluator — scores agent outputs against defined criteria."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# A criterion function receives (output, expected) and returns a score 0.0–1.0.
CriterionFn = Callable[[Any, Any], float]


@dataclass
class EvaluationResult:
    output: Any
    expected: Any
    scores: Dict[str, float]
    overall_score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def criterion_names(self) -> List[str]:
        return list(self.scores.keys())


class Evaluator:
    """Evaluates agent outputs against registered criteria.

    Each criterion is a callable that returns a score in [0.0, 1.0].
    The overall score is the mean of all criterion scores.
    A result *passes* when the overall score meets or exceeds *pass_threshold*.

    Usage::

        ev = Evaluator(pass_threshold=0.7)
        ev.register_criterion("exact_match", lambda out, exp: 1.0 if out == exp else 0.0)
        result = ev.evaluate(output="hello", expected="hello")
        assert result.passed
    """

    def __init__(self, pass_threshold: float = 0.5) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be in [0.0, 1.0].")
        self.pass_threshold = pass_threshold
        self._criteria: Dict[str, CriterionFn] = {}

    def register_criterion(self, name: str, criterion_fn: CriterionFn) -> None:
        if not name:
            raise ValueError("Criterion name must not be empty.")
        self._criteria[name] = criterion_fn

    def unregister_criterion(self, name: str) -> None:
        self._criteria.pop(name, None)

    def evaluate(self, output: Any, expected: Any = None,
                 details: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Score *output* against all registered criteria.

        :param output: The actual output produced by an agent.
        :param expected: The reference value (may be None for open-ended tasks).
        :param details: Extra metadata attached to the result.
        """
        if not self._criteria:
            return EvaluationResult(output=output, expected=expected, scores={},
                                    overall_score=0.0, passed=False,
                                    details=details or {})
        scores: Dict[str, float] = {}
        for name, fn in self._criteria.items():
            raw = fn(output, expected)
            scores[name] = max(0.0, min(1.0, float(raw)))

        overall = sum(scores.values()) / len(scores)
        return EvaluationResult(
            output=output,
            expected=expected,
            scores=scores,
            overall_score=overall,
            passed=overall >= self.pass_threshold,
            details=details or {},
        )
