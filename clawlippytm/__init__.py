"""
clawlippytm
~~~~~~~~~~~

AI Toolkit for Clawlippytm repositories.

Provides the :class:`~clawlippytm.bot.ClawBot` class along with its
constituent sub-systems:

- :class:`~clawlippytm.bot.BotAttributes` — all bot configuration knobs.
- :class:`~clawlippytm.diagnostics.DiagnosticsSystem` — multi-pass diagnostics
  with iterative feedback loops.
- :class:`~clawlippytm.cognitive_reasoning.CognitiveReasoner` — chain-of-thought
  reasoning engine with self-critique.
- :class:`~clawlippytm.creativity.CreativityEngine` — temperature-driven response
  enrichment (analogies, metaphors, narrative framing).
"""

from .bot import BotAttributes, ClawBot
from .cognitive_reasoning import CognitiveReasoner, ReasoningOutput, ReasoningStep
from .creativity import CreativityAnnotation, CreativityEngine
from .diagnostics import (
    DiagnosticIssue,
    DiagnosticResult,
    DiagnosticsSystem,
)

__all__ = [
    # Bot
    "ClawBot",
    "BotAttributes",
    # Diagnostics
    "DiagnosticsSystem",
    "DiagnosticResult",
    "DiagnosticIssue",
    # Cognitive reasoning
    "CognitiveReasoner",
    "ReasoningOutput",
    "ReasoningStep",
    # Creativity
    "CreativityEngine",
    "CreativityAnnotation",
]

__version__ = "1.0.0"
