"""Reasoning engine — chain-of-thought reasoning with registered inference rules."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .knowledge_base import KnowledgeBase


# A rule is a callable that receives a context dict and returns a (key, value) conclusion
# or None if the rule does not apply.
RuleFn = Callable[[Dict[str, Any]], Optional[tuple]]


@dataclass
class ReasoningStep:
    rule_name: str
    inputs: Dict[str, Any]
    conclusion_key: Optional[str]
    conclusion_value: Any
    applied: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReasoningChain:
    query: str
    steps: List[ReasoningStep]
    final_answer: Any
    confidence: float
    elapsed_ms: float
    timestamp: float = field(default_factory=time.time)


class ReasoningEngine:
    """Performs forward-chaining inference over a :class:`KnowledgeBase`.

    Rules are registered as callables that accept a context dictionary (current
    working memory) and return a ``(key, value)`` tuple when they fire, or
    ``None`` if they do not apply.

    Usage::

        kb = KnowledgeBase()
        kb.store("temperature_c", 100)
        engine = ReasoningEngine(kb)

        def boiling_rule(ctx):
            if ctx.get("temperature_c", 0) >= 100:
                return ("state", "boiling")
            return None

        engine.register_rule("boiling_rule", boiling_rule)
        chain = engine.reason("What is the water state?")
        print(chain.final_answer)  # {"state": "boiling"}
    """

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None) -> None:
        self._kb = knowledge_base if knowledge_base is not None else KnowledgeBase()
        self._rules: Dict[str, RuleFn] = {}

    @property
    def knowledge_base(self) -> KnowledgeBase:
        return self._kb

    def register_rule(self, name: str, rule_fn: RuleFn) -> None:
        """Register a named inference rule."""
        if not name:
            raise ValueError("Rule name must not be empty.")
        self._rules[name] = rule_fn

    def unregister_rule(self, name: str) -> None:
        self._rules.pop(name, None)

    def reason(self, query: str, context: Optional[Dict[str, Any]] = None,
               max_iterations: int = 10) -> ReasoningChain:
        """Run forward-chaining inference and return a :class:`ReasoningChain`.

        :param query: Natural-language description of the reasoning goal.
        :param context: Optional seed context; defaults to values from the KB.
        :param max_iterations: Safety cap on inference iterations.
        """
        start = time.perf_counter()
        working_memory: Dict[str, Any] = {e.key: e.value for e in self._kb.all_entries()}
        if context:
            working_memory.update(context)

        steps: List[ReasoningStep] = []
        new_facts = True
        iteration = 0

        while new_facts and iteration < max_iterations:
            new_facts = False
            iteration += 1
            for rule_name, rule_fn in self._rules.items():
                result = rule_fn(working_memory)
                applied = result is not None
                conclusion_key = conclusion_value = None
                if applied:
                    conclusion_key, conclusion_value = result
                    if working_memory.get(conclusion_key) != conclusion_value:
                        working_memory[conclusion_key] = conclusion_value
                        new_facts = True
                steps.append(ReasoningStep(
                    rule_name=rule_name,
                    inputs=dict(working_memory),
                    conclusion_key=conclusion_key,
                    conclusion_value=conclusion_value,
                    applied=applied,
                ))

        elapsed_ms = (time.perf_counter() - start) * 1000
        # Confidence: fraction of rules that fired at least once
        fired = {s.rule_name for s in steps if s.applied}
        confidence = len(fired) / len(self._rules) if self._rules else 0.0

        return ReasoningChain(
            query=query,
            steps=steps,
            final_answer=dict(working_memory),
            confidence=confidence,
            elapsed_ms=elapsed_ms,
        )
