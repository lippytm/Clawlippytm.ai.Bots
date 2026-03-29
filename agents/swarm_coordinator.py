"""Swarm coordinator — manages a pool of agents and distributes tasks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base_agent import AgentResult, AgentStatus, AgentTask, BaseAgent


@dataclass
class SwarmReport:
    total_agents: int
    active_agents: int
    idle_agents: int
    tasks_dispatched: int
    results: List[AgentResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class SwarmCoordinator:
    """Coordinates a heterogeneous pool of :class:`BaseAgent` instances.

    Tasks can be dispatched to a specific agent or broadcast to all agents.
    The coordinator is thread-safe and tracks cumulative dispatch statistics.

    Usage::

        coordinator = SwarmCoordinator()
        coordinator.register(my_agent)
        results = coordinator.broadcast(task)
        report = coordinator.report()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: Dict[str, BaseAgent] = {}
        self._tasks_dispatched: int = 0
        self._results: List[AgentResult] = []

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register(self, agent: BaseAgent) -> None:
        """Add an agent to the swarm."""
        with self._lock:
            self._agents[agent.agent_id] = agent

    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the swarm."""
        with self._lock:
            self._agents.pop(agent_id, None)

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        with self._lock:
            return self._agents.get(agent_id)

    @property
    def agent_ids(self) -> List[str]:
        with self._lock:
            return list(self._agents.keys())

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    def dispatch(self, agent_id: str, task: AgentTask) -> Optional[AgentResult]:
        """Dispatch a task to a specific agent.  Returns None if not found."""
        agent = self.get_agent(agent_id)
        if agent is None:
            return None
        result = agent.run(task)
        with self._lock:
            self._tasks_dispatched += 1
            self._results.append(result)
        return result

    def broadcast(self, task: AgentTask) -> List[AgentResult]:
        """Dispatch *task* to every registered agent and collect results."""
        with self._lock:
            agents = list(self._agents.values())
        results: List[AgentResult] = []
        for agent in agents:
            result = agent.run(task)
            results.append(result)
        with self._lock:
            self._tasks_dispatched += len(agents)
            self._results.extend(results)
        return results

    def dispatch_to_idle(self, task: AgentTask) -> Optional[AgentResult]:
        """Dispatch *task* to the first idle agent found."""
        with self._lock:
            idle_agents = [a for a in self._agents.values()
                           if a.status == AgentStatus.IDLE]
        if not idle_agents:
            return None
        return self.dispatch(idle_agents[0].agent_id, task)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> SwarmReport:
        """Return a summary of the swarm's current state."""
        with self._lock:
            agents = list(self._agents.values())
            dispatched = self._tasks_dispatched
            results = list(self._results)
        active = sum(1 for a in agents if a.status == AgentStatus.RUNNING)
        idle = sum(1 for a in agents if a.status == AgentStatus.IDLE)
        return SwarmReport(
            total_agents=len(agents),
            active_agents=active,
            idle_agents=idle,
            tasks_dispatched=dispatched,
            results=results,
        )

    def stop_all(self) -> None:
        """Stop every registered agent."""
        with self._lock:
            agents = list(self._agents.values())
        for agent in agents:
            agent.stop()
