"""Top-level orchestration engine for the AI Full Stack Generative AI DevOps
Synthetic Intelligence Engines Swarms Agents/Bots system.

The :class:`SyntheticIntelligenceEngine` wires together the three core
sub-systems — Diagnostics, Cognitive Reasoning, and Feedback Loops — and
exposes a single entry point for running agents within a managed lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents import BaseAgent, SwarmCoordinator
from agents.base_agent import AgentResult, AgentTask
from cognitive_reasoning import DecisionMaker, KnowledgeBase, ReasoningEngine
from cognitive_reasoning.reasoning_engine import ReasoningChain
from diagnostics import DiagnosticsLogger, HealthMonitor, MetricsCollector
from diagnostics.health_monitor import HealthCheckResult, HealthStatus, SystemHealthReport
from feedback_loops import Adapter, Evaluator, FeedbackManager
from feedback_loops.feedback_manager import FeedbackPolarity, FeedbackRecord


@dataclass
class EngineStatus:
    uptime_seconds: float
    health: SystemHealthReport
    metrics_snapshot: List[Any]
    swarm_report: Any
    timestamp: float = field(default_factory=time.time)


class SyntheticIntelligenceEngine:
    """Unified orchestration layer for all AI subsystems.

    Responsibilities
    ----------------
    * Bootstraps and wires up Diagnostics, Cognitive Reasoning, and
      Feedback-Loop sub-systems.
    * Manages a :class:`~agents.SwarmCoordinator` that owns agent lifecycles.
    * Runs a closed *Diagnose → Reason → Decide → Execute → Evaluate →
      Adapt* loop on demand.

    Usage::

        engine = SyntheticIntelligenceEngine()
        engine.register_agent(my_agent)

        task = AgentTask(description="process data", payload={"input": [1, 2, 3]})
        result = engine.run_task(task)

        status = engine.status()
        print(status.health.overall_status)
    """

    def __init__(self, engine_id: str = "sie-1") -> None:
        self.engine_id = engine_id
        self._start_time = time.time()

        # ----- Diagnostics -----
        self.logger = DiagnosticsLogger(f"engine.{engine_id}")
        self.metrics = MetricsCollector()
        self.health_monitor = HealthMonitor()

        # Register a basic engine self-health check
        self.health_monitor.register(
            "engine_alive",
            lambda: HealthCheckResult("engine_alive", HealthStatus.HEALTHY, "Engine is running"),
        )

        # ----- Cognitive Reasoning -----
        self.knowledge_base = KnowledgeBase()
        self.reasoning_engine = ReasoningEngine(self.knowledge_base)
        self.decision_maker = DecisionMaker()

        # ----- Feedback Loops -----
        self.feedback_manager = FeedbackManager()
        self.evaluator = Evaluator()
        self.adapter = Adapter(self.feedback_manager)

        # ----- Swarm / Agents -----
        self.swarm = SwarmCoordinator()

        self.logger.info("SyntheticIntelligenceEngine initialised",
                         extra={"engine_id": engine_id})

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the engine's swarm."""
        self.swarm.register(agent)
        self.health_monitor.register(
            f"agent.{agent.agent_id}",
            lambda a=agent: HealthCheckResult(
                f"agent.{a.agent_id}",
                HealthStatus.HEALTHY if a.status.value != "error" else HealthStatus.UNHEALTHY,
                f"Agent status: {a.status}",
            ),
        )
        self.metrics.increment("agents_registered")
        self.logger.info("Agent registered", extra={"agent_id": agent.agent_id,
                                                    "agent_name": agent.name})

    def unregister_agent(self, agent_id: str) -> None:
        self.swarm.unregister(agent_id)
        self.health_monitor.unregister(f"agent.{agent_id}")
        self.logger.info("Agent unregistered", extra={"agent_id": agent_id})

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def run_task(self, task: AgentTask,
                 agent_id: Optional[str] = None) -> Optional[AgentResult]:
        """Execute *task* on a specific agent or the first idle agent.

        Returns None if no suitable agent is available.
        """
        self.metrics.increment("tasks_submitted")
        if agent_id:
            result = self.swarm.dispatch(agent_id, task)
        else:
            result = self.swarm.dispatch_to_idle(task)

        if result is None:
            self.logger.warning("No agent available for task",
                                extra={"task_id": task.task_id})
            return None

        self._process_result(result)
        return result

    def broadcast_task(self, task: AgentTask) -> List[AgentResult]:
        """Broadcast *task* to all registered agents."""
        self.metrics.increment("tasks_broadcast")
        results = self.swarm.broadcast(task)
        for r in results:
            self._process_result(r)
        return results

    def _process_result(self, result: AgentResult) -> None:
        """Evaluate a result and feed back into the feedback-loop system."""
        eval_result = self.evaluator.evaluate(
            output=result.output,
            details={"task_id": result.task_id, "agent_id": result.agent_id},
        )
        polarity = (FeedbackPolarity.POSITIVE if result.success
                    else FeedbackPolarity.NEGATIVE)
        score = eval_result.overall_score if result.success else 0.0
        self.feedback_manager.submit(FeedbackRecord(
            source=result.agent_id,
            action=result.task_id,
            polarity=polarity,
            score=score,
        ))
        self.metrics.observe("result_score", score)
        self.adapter.adapt(source=result.agent_id)

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def reason(self, query: str,
               context: Optional[Dict[str, Any]] = None) -> ReasoningChain:
        """Run the cognitive reasoning engine for *query*."""
        self.metrics.increment("reasoning_requests")
        chain = self.reasoning_engine.reason(query, context=context)
        self.metrics.observe("reasoning_confidence", chain.confidence)
        return chain

    # ------------------------------------------------------------------
    # Status / health
    # ------------------------------------------------------------------

    def status(self) -> EngineStatus:
        """Return a comprehensive snapshot of engine health and metrics."""
        health = self.health_monitor.run_all_checks()
        metrics_snapshot = self.metrics.snapshot()
        swarm_report = self.swarm.report()
        uptime = time.time() - self._start_time
        return EngineStatus(
            uptime_seconds=uptime,
            health=health,
            metrics_snapshot=metrics_snapshot,
            swarm_report=swarm_report,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully stop all agents and flush diagnostics."""
        self.swarm.stop_all()
        self.logger.info("Engine shut down", extra={"engine_id": self.engine_id})
