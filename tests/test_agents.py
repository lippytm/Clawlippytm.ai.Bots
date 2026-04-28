"""
Tests for clawlippytm.agents
"""
import pytest
from clawlippytm import (
    AgentOrchestrator,
    AgentResult,
    AgentRole,
    AgentTask,
    SyntheticAgent,
)


class TestAgentRole:
    def test_all_roles_exist(self):
        expected = {"coordinator", "devops", "fullstack", "synthetic", "general"}
        actual = {r.value for r in AgentRole}
        assert expected == actual

    def test_role_is_string_enum(self):
        assert AgentRole.DEVOPS == "devops"


class TestAgentTask:
    def test_defaults(self):
        task = AgentTask(description="Deploy service")
        assert task.description == "Deploy service"
        assert task.role == AgentRole.GENERAL
        assert task.priority == 5
        assert isinstance(task.task_id, str)
        assert len(task.task_id) > 0
        assert task.dependencies == []

    def test_to_dict(self):
        task = AgentTask(description="Build image", role=AgentRole.DEVOPS, priority=2)
        d = task.to_dict()
        assert d["description"] == "Build image"
        assert d["role"] == "devops"
        assert d["priority"] == 2
        assert "task_id" in d
        assert "dependencies" in d
        assert "metadata" in d


class TestAgentResult:
    def test_success_result(self):
        result = AgentResult(task_id="abc", agent_role=AgentRole.DEVOPS, output="done")
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        result = AgentResult(
            task_id="xyz",
            agent_role=AgentRole.GENERAL,
            output="",
            success=False,
            error="Something went wrong",
        )
        assert not result.success
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        result = AgentResult(task_id="t1", agent_role=AgentRole.FULLSTACK, output="ok")
        d = result.to_dict()
        assert d["task_id"] == "t1"
        assert d["agent_role"] == "fullstack"
        assert d["output"] == "ok"
        assert d["success"] is True
        assert d["error"] is None


class TestSyntheticAgent:
    def test_default_role(self):
        agent = SyntheticAgent()
        assert agent.role == AgentRole.GENERAL

    def test_custom_name(self):
        agent = SyntheticAgent(role=AgentRole.DEVOPS, name="PipelineBot")
        assert agent.name == "PipelineBot"

    def test_default_name_derived_from_role(self):
        agent = SyntheticAgent(role=AgentRole.SYNTHETIC)
        assert "synthetic" in agent.name.lower()

    def test_process_returns_result(self):
        agent = SyntheticAgent(role=AgentRole.DEVOPS)
        task = AgentTask(description="Run CI pipeline", role=AgentRole.DEVOPS)
        result = agent.process(task)
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert len(result.output) > 0
        assert result.task_id == task.task_id

    def test_process_increments_counter(self):
        agent = SyntheticAgent()
        task = AgentTask(description="Test task")
        agent.process(task)
        agent.process(task)
        assert agent.summary()["tasks_processed"] == 2

    @pytest.mark.parametrize("role", list(AgentRole))
    def test_all_roles_produce_output(self, role):
        agent = SyntheticAgent(role=role)
        task = AgentTask(description="Generic task", role=role)
        result = agent.process(task)
        assert result.success is True
        assert len(result.output) > 0

    def test_summary_keys(self):
        agent = SyntheticAgent(role=AgentRole.COORDINATOR)
        s = agent.summary()
        assert "name" in s
        assert "role" in s
        assert "tasks_processed" in s


class TestAgentOrchestrator:
    def test_default_agents_registered(self):
        orch = AgentOrchestrator()
        s = orch.summary()
        registered_roles = {a["role"] for a in s["registered_agents"]}
        assert "coordinator" in registered_roles
        assert "devops" in registered_roles
        assert "fullstack" in registered_roles
        assert "synthetic" in registered_roles
        assert "general" in registered_roles

    def test_dispatch_correct_role(self):
        orch = AgentOrchestrator()
        task = AgentTask(description="Deploy to production", role=AgentRole.DEVOPS)
        result = orch.dispatch(task)
        assert result.success is True
        assert result.agent_role == AgentRole.DEVOPS

    def test_dispatch_fallback_to_general(self):
        orch = AgentOrchestrator(max_agents=1)
        # Only register a GENERAL agent
        orch._agents = {AgentRole.GENERAL: SyntheticAgent(role=AgentRole.GENERAL)}
        task = AgentTask(description="Fullstack task", role=AgentRole.FULLSTACK)
        result = orch.dispatch(task)
        # Falls back to GENERAL
        assert result.success is True
        assert result.agent_role == AgentRole.GENERAL

    def test_dispatch_no_agent_returns_failure(self):
        orch = AgentOrchestrator(max_agents=1)
        orch._agents = {}
        task = AgentTask(description="Orphan task", role=AgentRole.DEVOPS)
        result = orch.dispatch(task)
        assert not result.success
        assert result.error is not None

    def test_dispatch_all_respects_priority(self):
        orch = AgentOrchestrator()
        tasks = [
            AgentTask(description="Low priority task", priority=9),
            AgentTask(description="High priority task", priority=1),
            AgentTask(description="Medium priority task", priority=5),
        ]
        results = orch.dispatch_all(tasks)
        assert len(results) == 3
        # All should succeed
        assert all(r.success for r in results)

    def test_summary_keys(self):
        orch = AgentOrchestrator()
        orch.dispatch(AgentTask(description="Test"))
        s = orch.summary()
        assert "registered_agents" in s
        assert "tasks_dispatched" in s
        assert "tasks_succeeded" in s
        assert "tasks_failed" in s
        assert s["tasks_dispatched"] == 1
        assert s["tasks_succeeded"] == 1

    def test_register_at_capacity_raises(self):
        orch = AgentOrchestrator(max_agents=1)
        orch._agents = {AgentRole.GENERAL: SyntheticAgent()}
        with pytest.raises(RuntimeError, match="capacity"):
            orch.register(SyntheticAgent(role=AgentRole.DEVOPS))

    def test_register_custom_agent(self):
        orch = AgentOrchestrator(max_agents=10)
        custom = SyntheticAgent(role=AgentRole.DEVOPS, name="CustomDevOps")
        orch.register(custom)
        task = AgentTask(description="Custom devops task", role=AgentRole.DEVOPS)
        result = orch.dispatch(task)
        assert result.success is True
        assert result.metadata.get("agent_name") == "CustomDevOps"
