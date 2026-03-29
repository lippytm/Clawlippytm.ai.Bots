"""Tests for agents module and the top-level engine."""

import pytest

from agents.base_agent import AgentTask, AgentStatus, BaseAgent, AgentResult
from agents.swarm_coordinator import SwarmCoordinator
from engine import SyntheticIntelligenceEngine


# -------------------------------------------------------------------------
# Concrete test agent
# -------------------------------------------------------------------------

class EchoAgent(BaseAgent):
    """Simple agent that echoes its payload."""
    def execute(self, task: AgentTask):
        return task.payload.get("message", "")


class FailingAgent(BaseAgent):
    """Agent that always raises an exception."""
    def execute(self, task: AgentTask):
        raise RuntimeError("intentional failure")


# -------------------------------------------------------------------------
# BaseAgent
# -------------------------------------------------------------------------

class TestBaseAgent:
    def test_run_success(self):
        agent = EchoAgent(agent_id="echo-1")
        task = AgentTask(description="echo", payload={"message": "hello"})
        result = agent.run(task)
        assert result.success
        assert result.output == "hello"
        assert result.error is None

    def test_run_failure_captured(self):
        agent = FailingAgent(agent_id="fail-1")
        task = AgentTask(description="fail")
        result = agent.run(task)
        assert not result.success
        assert result.error is not None
        assert agent.status == AgentStatus.ERROR

    def test_status_returns_to_idle_after_success(self):
        agent = EchoAgent(agent_id="echo-2")
        task = AgentTask(description="echo", payload={"message": "hi"})
        agent.run(task)
        assert agent.status == AgentStatus.IDLE

    def test_pause_and_resume(self):
        agent = EchoAgent(agent_id="echo-3")
        agent.pause()
        assert agent.status == AgentStatus.PAUSED
        agent.resume()
        assert agent.status == AgentStatus.IDLE

    def test_stop(self):
        agent = EchoAgent(agent_id="echo-4")
        agent.stop()
        assert agent.status == AgentStatus.STOPPED

    def test_metrics_incremented(self):
        agent = EchoAgent(agent_id="echo-5")
        task = AgentTask(payload={"message": "x"})
        agent.run(task)
        assert agent.metrics.get_counter("tasks_started") == 1.0
        assert agent.metrics.get_counter("tasks_succeeded") == 1.0

    def test_repr(self):
        agent = EchoAgent(agent_id="echo-r")
        r = repr(agent)
        assert "echo-r" in r
        assert "EchoAgent" in r


# -------------------------------------------------------------------------
# SwarmCoordinator
# -------------------------------------------------------------------------

class TestSwarmCoordinator:
    def test_register_and_dispatch(self):
        coord = SwarmCoordinator()
        agent = EchoAgent(agent_id="e1")
        coord.register(agent)
        task = AgentTask(payload={"message": "ping"})
        result = coord.dispatch("e1", task)
        assert result is not None
        assert result.output == "ping"

    def test_dispatch_unknown_agent(self):
        coord = SwarmCoordinator()
        task = AgentTask()
        assert coord.dispatch("unknown", task) is None

    def test_broadcast(self):
        coord = SwarmCoordinator()
        coord.register(EchoAgent(agent_id="e1"))
        coord.register(EchoAgent(agent_id="e2"))
        task = AgentTask(payload={"message": "broadcast"})
        results = coord.broadcast(task)
        assert len(results) == 2
        assert all(r.output == "broadcast" for r in results)

    def test_dispatch_to_idle(self):
        coord = SwarmCoordinator()
        coord.register(EchoAgent(agent_id="e1"))
        task = AgentTask(payload={"message": "idle"})
        result = coord.dispatch_to_idle(task)
        assert result is not None
        assert result.output == "idle"

    def test_dispatch_to_idle_none_available(self):
        coord = SwarmCoordinator()
        agent = EchoAgent(agent_id="e1")
        agent.stop()
        coord.register(agent)
        task = AgentTask()
        assert coord.dispatch_to_idle(task) is None

    def test_report(self):
        coord = SwarmCoordinator()
        coord.register(EchoAgent(agent_id="e1"))
        task = AgentTask(payload={"message": "x"})
        coord.dispatch("e1", task)
        report = coord.report()
        assert report.total_agents == 1
        assert report.tasks_dispatched == 1

    def test_stop_all(self):
        coord = SwarmCoordinator()
        coord.register(EchoAgent(agent_id="e1"))
        coord.register(EchoAgent(agent_id="e2"))
        coord.stop_all()
        for aid in coord.agent_ids:
            assert coord.get_agent(aid).status == AgentStatus.STOPPED

    def test_unregister(self):
        coord = SwarmCoordinator()
        coord.register(EchoAgent(agent_id="e1"))
        coord.unregister("e1")
        assert coord.get_agent("e1") is None


# -------------------------------------------------------------------------
# SyntheticIntelligenceEngine
# -------------------------------------------------------------------------

class TestSyntheticIntelligenceEngine:
    def test_initialization(self):
        engine = SyntheticIntelligenceEngine(engine_id="test-engine")
        assert engine.engine_id == "test-engine"

    def test_register_and_run_task(self):
        engine = SyntheticIntelligenceEngine()
        engine.register_agent(EchoAgent(agent_id="e1"))
        task = AgentTask(payload={"message": "engine test"})
        result = engine.run_task(task)
        assert result is not None
        assert result.output == "engine test"

    def test_run_task_no_agent(self):
        engine = SyntheticIntelligenceEngine()
        task = AgentTask()
        assert engine.run_task(task) is None

    def test_broadcast_task(self):
        engine = SyntheticIntelligenceEngine()
        engine.register_agent(EchoAgent(agent_id="e1"))
        engine.register_agent(EchoAgent(agent_id="e2"))
        task = AgentTask(payload={"message": "bcast"})
        results = engine.broadcast_task(task)
        assert len(results) == 2

    def test_status_health(self):
        engine = SyntheticIntelligenceEngine()
        status = engine.status()
        from diagnostics.health_monitor import HealthStatus
        assert status.health.overall_status == HealthStatus.HEALTHY
        assert status.uptime_seconds >= 0

    def test_reason(self):
        engine = SyntheticIntelligenceEngine()
        engine.knowledge_base.store("sky", "blue")
        chain = engine.reason("What color is the sky?")
        assert chain.final_answer.get("sky") == "blue"

    def test_shutdown(self):
        engine = SyntheticIntelligenceEngine()
        engine.register_agent(EchoAgent(agent_id="e1"))
        engine.shutdown()
        agent = engine.swarm.get_agent("e1")
        assert agent.status == AgentStatus.STOPPED

    def test_unregister_agent(self):
        engine = SyntheticIntelligenceEngine()
        engine.register_agent(EchoAgent(agent_id="e1"))
        engine.unregister_agent("e1")
        assert engine.swarm.get_agent("e1") is None
