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
  reasoning engine with self-critique and confidence scoring.
- :class:`~clawlippytm.creativity.CreativityEngine` — temperature-driven response
  enrichment (analogies, metaphors, narrative framing).
- :class:`~clawlippytm.agents.AgentOrchestrator` — multi-agent coordination for
  AI Full-Stack / DevOps / Synthetic Intelligence pipelines.
- :class:`~clawlippytm.devops.DevOpsEngine` — AI-powered CI/CD pipeline engine.
"""

from .bot import BotAttributes, ClawBot
from .cognitive_reasoning import CognitiveReasoner, ReasoningOutput, ReasoningStep
from .creativity import CreativityAnnotation, CreativityEngine
from .diagnostics import (
    DiagnosticIssue,
    DiagnosticResult,
    DiagnosticsSystem,
)
from .agents import (
    AgentOrchestrator,
    AgentResult,
    AgentRole,
    AgentTask,
    SyntheticAgent,
)
from .devops import (
    DevOpsEngine,
    PipelineRun,
    PipelineStage,
    StageResult,
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
    # Agents
    "AgentOrchestrator",
    "AgentResult",
    "AgentRole",
    "AgentTask",
    "SyntheticAgent",
    # DevOps
    "DevOpsEngine",
    "PipelineRun",
    "PipelineStage",
    "StageResult",
]

__version__ = "1.1.0"
