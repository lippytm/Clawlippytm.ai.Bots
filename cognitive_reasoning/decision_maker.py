"""Decision maker — selects the best action given a set of candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# A scoring function receives (action, context) and returns a float score.
ScoringFn = Callable[[str, Dict[str, Any]], float]


@dataclass
class Decision:
    chosen_action: str
    score: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


class DecisionMaker:
    """Selects the highest-scoring action from a list of candidates.

    Scoring functions can be registered by name and are applied to every
    candidate action.  The action with the highest aggregated score is chosen.

    Usage::

        dm = DecisionMaker()
        dm.register_scorer("urgency", lambda action, ctx: ctx.get("urgency", 0))
        decision = dm.decide(["action_a", "action_b"], context={"urgency": 7})
        print(decision.chosen_action)
    """

    def __init__(self) -> None:
        self._scorers: Dict[str, ScoringFn] = {}

    def register_scorer(self, name: str, scorer_fn: ScoringFn) -> None:
        if not name:
            raise ValueError("Scorer name must not be empty.")
        self._scorers[name] = scorer_fn

    def unregister_scorer(self, name: str) -> None:
        self._scorers.pop(name, None)

    def score_action(self, action: str, context: Dict[str, Any]) -> float:
        """Return the aggregated score for a single action."""
        if not self._scorers:
            return 0.0
        return sum(fn(action, context) for fn in self._scorers.values())

    def decide(self, candidates: List[str], context: Optional[Dict[str, Any]] = None,
               rationale: str = "") -> Decision:
        """Return a :class:`Decision` for the best candidate action.

        :param candidates: List of action identifiers to evaluate.
        :param context: Contextual data passed to every scorer.
        :param rationale: Optional free-text explanation attached to the decision.
        :raises ValueError: If *candidates* is empty.
        """
        if not candidates:
            raise ValueError("At least one candidate action is required.")
        ctx = context or {}
        scored = [{"action": a, "score": self.score_action(a, ctx)} for a in candidates]
        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]
        return Decision(
            chosen_action=best["action"],
            score=best["score"],
            alternatives=scored[1:],
            context=ctx,
            rationale=rationale,
        )
