"""Adapter — adjusts agent behaviour based on accumulated feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .feedback_manager import FeedbackManager, FeedbackPolarity, FeedbackRecord


# An adaptation strategy receives (current_params, feedback_records) and
# returns an updated params dict.
StrategyFn = Callable[[Dict[str, Any], List[FeedbackRecord]], Dict[str, Any]]


@dataclass
class AdaptationResult:
    previous_params: Dict[str, Any]
    updated_params: Dict[str, Any]
    strategy_applied: str
    feedback_count: int
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.previous_params != self.updated_params


class Adapter:
    """Adapts agent configuration parameters based on feedback signals.

    Strategies are registered by name and each receives the current parameter
    dictionary plus the list of relevant feedback records.  The first applicable
    strategy is applied; strategies should return a (possibly modified) copy of
    the params dict.

    Usage::

        adapter = Adapter(feedback_manager)
        adapter.params = {"learning_rate": 0.01, "temperature": 0.7}

        def lr_decay(params, records):
            negative = [r for r in records if r.polarity == FeedbackPolarity.NEGATIVE]
            if len(negative) > 5:
                params = dict(params)
                params["learning_rate"] *= 0.9
            return params

        adapter.register_strategy("lr_decay", lr_decay)
        result = adapter.adapt(source="agent-1")
        print(result.updated_params)
    """

    def __init__(self, feedback_manager: FeedbackManager) -> None:
        self._fm = feedback_manager
        self.params: Dict[str, Any] = {}
        self._strategies: Dict[str, StrategyFn] = {}

    def register_strategy(self, name: str, strategy_fn: StrategyFn) -> None:
        if not name:
            raise ValueError("Strategy name must not be empty.")
        self._strategies[name] = strategy_fn

    def unregister_strategy(self, name: str) -> None:
        self._strategies.pop(name, None)

    def adapt(self, source: Optional[str] = None,
              polarity: Optional[FeedbackPolarity] = None) -> AdaptationResult:
        """Run all registered strategies and return an :class:`AdaptationResult`.

        :param source: Filter feedback by this source name.
        :param polarity: Filter feedback by polarity.
        """
        records = self._fm.get_records(source=source, polarity=polarity)
        previous = dict(self.params)
        current = dict(self.params)
        strategy_applied = "none"

        for name, strategy_fn in self._strategies.items():
            updated = strategy_fn(current, records)
            if updated != current:
                current = updated
                strategy_applied = name

        self.params = current
        return AdaptationResult(
            previous_params=previous,
            updated_params=current,
            strategy_applied=strategy_applied,
            feedback_count=len(records),
        )
