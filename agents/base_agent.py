"""Base agent class for AI bots in the swarm."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from diagnostics import DiagnosticsLogger, MetricsCollector
from cognitive_reasoning import DecisionMaker, KnowledgeBase, ReasoningEngine
from feedback_loops import Adapter, Evaluator, FeedbackManager


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    task_id: str
    agent_id: str
    output: Any
    success: bool
    elapsed_ms: float
    error: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base class for all agents in the swarm.

    Sub-classes must implement :meth:`execute`, which receives an
    :class:`AgentTask` and returns any output value.  The base class wires up
    diagnostics, reasoning, and feedback-loop infrastructure automatically.

    Usage::

        class EchoAgent(BaseAgent):
            def execute(self, task: AgentTask):
                return task.payload.get("message", "")

        agent = EchoAgent(agent_id="echo-1")
        result = agent.run(AgentTask(description="echo", payload={"message": "hi"}))
        print(result.output)  # "hi"
    """

    def __init__(self, agent_id: Optional[str] = None, name: str = "") -> None:
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.name: str = name or self.__class__.__name__
        self.status: AgentStatus = AgentStatus.IDLE

        # Diagnostics
        self.logger = DiagnosticsLogger(f"agent.{self.agent_id}")
        self.metrics = MetricsCollector()

        # Cognitive reasoning
        self.knowledge_base = KnowledgeBase()
        self.reasoning_engine = ReasoningEngine(self.knowledge_base)
        self.decision_maker = DecisionMaker()

        # Feedback loops
        self.feedback_manager = FeedbackManager()
        self.evaluator = Evaluator()
        self.adapter = Adapter(self.feedback_manager)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, task: AgentTask) -> Any:
        """Execute *task* and return the result output."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self, task: AgentTask) -> AgentResult:
        """Wrap :meth:`execute` with timing, logging, and metrics."""
        self.status = AgentStatus.RUNNING
        self.logger.info("Task started", extra={"task_id": task.task_id,
                                                "description": task.description})
        self.metrics.increment("tasks_started")
        start = time.perf_counter()
        error: Optional[str] = None
        output: Any = None
        success = False
        try:
            output = self.execute(task)
            success = True
            self.metrics.increment("tasks_succeeded")
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            self.status = AgentStatus.ERROR
            self.metrics.increment("tasks_failed")
            self.logger.error("Task failed", extra={"task_id": task.task_id, "error": error})
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics.observe("task_duration_ms", elapsed_ms)

        if success:
            self.status = AgentStatus.IDLE
            self.logger.info("Task completed", extra={"task_id": task.task_id,
                                                      "elapsed_ms": elapsed_ms})
        return AgentResult(task_id=task.task_id, agent_id=self.agent_id,
                           output=output, success=success,
                           elapsed_ms=elapsed_ms, error=error)

    def pause(self) -> None:
        self.status = AgentStatus.PAUSED
        self.logger.info("Agent paused")

    def resume(self) -> None:
        self.status = AgentStatus.IDLE
        self.logger.info("Agent resumed")

    def stop(self) -> None:
        self.status = AgentStatus.STOPPED
        self.logger.info("Agent stopped")

    def __repr__(self) -> str:
        return f"<{self.name} id={self.agent_id} status={self.status}>"
