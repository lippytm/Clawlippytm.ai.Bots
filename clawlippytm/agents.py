"""
clawlippytm.agents
~~~~~~~~~~~~~~~~~~

Multi-agent coordination system for Clawlippytm.Bots.

The :class:`AgentOrchestrator` coordinates a pool of :class:`SyntheticAgent`
instances, each specialised for a particular role (DevOps, FullStack,
Synthetic, Coordinator, General).  Agents process :class:`AgentTask` objects
and return :class:`AgentResult` records, enabling composable AI Full-Stack
and DevOps pipelines.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Agent roles
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    """Specialised role for a :class:`SyntheticAgent`."""

    COORDINATOR = "coordinator"
    DEVOPS = "devops"
    FULLSTACK = "fullstack"
    SYNTHETIC = "synthetic"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentTask:
    """A unit of work dispatched to an agent."""

    description: str
    role: AgentRole = AgentRole.GENERAL
    priority: int = 5                           # 1 (highest) – 10 (lowest)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "role": self.role.value,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }


@dataclass
class AgentResult:
    """Result produced by a :class:`SyntheticAgent` processing a task."""

    task_id: str
    agent_role: AgentRole
    output: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_role": self.agent_role.value,
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Synthetic agent
# ---------------------------------------------------------------------------

class SyntheticAgent:
    """
    A specialised synthetic intelligence agent.

    Each agent has a fixed :attr:`role` and processes :class:`AgentTask`
    objects, returning an :class:`AgentResult`.

    Parameters
    ----------
    role:
        The agent's specialisation.
    name:
        Optional display name; defaults to ``"<Role>Agent"``.
    """

    # Role-specific response templates
    _ROLE_TEMPLATES: Dict[AgentRole, str] = {
        AgentRole.COORDINATOR: (
            "Coordinator: task '{desc}' has been decomposed and dispatched "
            "to downstream agents for execution."
        ),
        AgentRole.DEVOPS: (
            "DevOps agent: analysed '{desc}'.  Pipeline stages validated; "
            "infrastructure configuration checked; deployment readiness confirmed."
        ),
        AgentRole.FULLSTACK: (
            "FullStack agent: processed '{desc}'.  Frontend, backend, and "
            "database layer considerations integrated into the response."
        ),
        AgentRole.SYNTHETIC: (
            "Synthetic Intelligence agent: '{desc}' evaluated across multiple "
            "synthetic reasoning dimensions; adaptive output generated."
        ),
        AgentRole.GENERAL: (
            "General agent: task '{desc}' processed with standard reasoning pipeline."
        ),
    }

    def __init__(
        self,
        role: AgentRole = AgentRole.GENERAL,
        name: Optional[str] = None,
    ) -> None:
        self.role: AgentRole = role
        self.name: str = name or f"{role.value.capitalize()}Agent"
        self._tasks_processed: int = 0

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def process(self, task: AgentTask) -> AgentResult:
        """
        Process *task* and return an :class:`AgentResult`.

        Parameters
        ----------
        task:
            The task to process.

        Returns
        -------
        AgentResult
        """
        try:
            template = self._ROLE_TEMPLATES.get(
                self.role, self._ROLE_TEMPLATES[AgentRole.GENERAL]
            )
            output = template.format(desc=task.description[:120])
            # Descriptions are truncated to 120 characters to keep agent output
            # concise and prevent excessively long template expansions.
            self._tasks_processed += 1
            return AgentResult(
                task_id=task.task_id,
                agent_role=self.role,
                output=output,
                success=True,
                metadata={
                    "agent_name": self.name,
                    "tasks_processed": self._tasks_processed,
                },
            )
        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                agent_role=self.role,
                output="",
                success=False,
                error=str(exc),
            )

    def summary(self) -> Dict[str, Any]:
        """Return a summary of agent activity."""
        return {
            "name": self.name,
            "role": self.role.value,
            "tasks_processed": self._tasks_processed,
        }


# ---------------------------------------------------------------------------
# Agent orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Coordinates a pool of :class:`SyntheticAgent` instances.

    The orchestrator maintains a registry of agents keyed by role, routes
    incoming tasks to the appropriate agent, and accumulates results.

    Parameters
    ----------
    max_agents:
        Maximum number of agents that can be registered (default 5).
    """

    def __init__(self, max_agents: int = 5) -> None:
        self.max_agents: int = max(1, max_agents)
        self._agents: Dict[AgentRole, SyntheticAgent] = {}
        self._results: List[AgentResult] = []
        self._register_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, agent: SyntheticAgent) -> None:
        """
        Register *agent* with the orchestrator.

        Raises
        ------
        RuntimeError
            When the agent pool is already at capacity.
        """
        if len(self._agents) >= self.max_agents:
            raise RuntimeError(
                f"Agent pool is at capacity ({self.max_agents}). "
                "Increase max_agents or remove an existing agent."
            )
        self._agents[agent.role] = agent

    def dispatch(self, task: AgentTask) -> AgentResult:
        """
        Dispatch *task* to the agent registered for ``task.role``.

        Falls back to the :attr:`AgentRole.GENERAL` agent when no exact
        role match is found.

        Parameters
        ----------
        task:
            The task to dispatch.

        Returns
        -------
        AgentResult
        """
        agent = self._agents.get(task.role) or self._agents.get(AgentRole.GENERAL)
        if agent is None:
            result = AgentResult(
                task_id=task.task_id,
                agent_role=task.role,
                output="",
                success=False,
                error=f"No agent registered for role '{task.role.value}'.",
            )
        else:
            result = agent.process(task)
        self._results.append(result)
        return result

    def dispatch_all(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """
        Dispatch a list of tasks in priority order (ascending priority value).

        Parameters
        ----------
        tasks:
            List of tasks to dispatch.

        Returns
        -------
        list[AgentResult]
        """
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)
        return [self.dispatch(t) for t in sorted_tasks]

    def summary(self) -> Dict[str, Any]:
        """Return a summary of orchestrator activity."""
        return {
            "registered_agents": [a.summary() for a in self._agents.values()],
            "tasks_dispatched": len(self._results),
            "tasks_succeeded": sum(1 for r in self._results if r.success),
            "tasks_failed": sum(1 for r in self._results if not r.success),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        """Populate the agent pool with one agent per built-in role."""
        for role in (
            AgentRole.COORDINATOR,
            AgentRole.DEVOPS,
            AgentRole.FULLSTACK,
            AgentRole.SYNTHETIC,
            AgentRole.GENERAL,
        ):
            if len(self._agents) < self.max_agents:
                self._agents[role] = SyntheticAgent(role=role)
