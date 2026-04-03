"""
Tests for clawlippytm.bot
"""
import pytest
from clawlippytm import BotAttributes, ClawBot


class TestBotAttributes:
    def test_defaults(self):
        attrs = BotAttributes.defaults()
        assert attrs.name == "ClawBot"
        assert 0.0 <= attrs.empathy_level <= 1.0
        assert 0.0 <= attrs.creativity_temperature <= 1.0
        assert attrs.reasoning_depth >= 1
        assert isinstance(attrs.ethical_guidelines, list)
        assert len(attrs.ethical_guidelines) > 0

    def test_to_dict(self):
        attrs = BotAttributes.defaults()
        d = attrs.to_dict()
        assert d["name"] == attrs.name
        assert d["version"] == attrs.version

    def test_update(self):
        attrs = BotAttributes.defaults()
        updated = attrs.update(name="MyBot", reasoning_depth=5)
        assert updated.name == "MyBot"
        assert updated.reasoning_depth == 5
        # Original should be unchanged
        assert attrs.name == "ClawBot"

    def test_all_fields_present(self):
        attrs = BotAttributes()
        expected_fields = [
            "name", "version", "description", "tone", "verbosity",
            "empathy_level", "multi_turn", "memory_enabled", "tool_use",
            "streaming", "reasoning_depth", "creativity_temperature",
            "self_critique", "safety_filter", "ethical_guidelines",
            "diagnostics_enabled", "feedback_loops",
            "role", "agent_mode", "max_agents", "devops_environment",
        ]
        d = attrs.to_dict()
        for field in expected_fields:
            assert field in d, f"Missing attribute: {field}"


class TestClawBot:
    def test_creation_with_defaults(self):
        bot = ClawBot()
        assert bot.attributes.name == "ClawBot"
        assert bot.diagnostics is not None
        assert bot.reasoner is not None
        assert bot.creativity is not None

    def test_creation_with_custom_attributes(self):
        attrs = BotAttributes(name="CustomBot", reasoning_depth=2)
        bot = ClawBot(attributes=attrs)
        assert bot.attributes.name == "CustomBot"
        assert bot.reasoner.depth == 2

    def test_respond_returns_string(self):
        bot = ClawBot()
        result = bot.respond("What is machine learning?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_respond_updates_history(self):
        bot = ClawBot()
        bot.respond("Hello, how are you?")
        assert len(bot._conversation_history) == 2  # user + bot

    def test_multiple_turns(self):
        bot = ClawBot()
        bot.respond("Tell me about AI.")
        bot.respond("How does it work?")
        assert len(bot._conversation_history) == 4

    def test_reset_clears_history(self):
        bot = ClawBot()
        bot.respond("Hello")
        bot.reset()
        assert len(bot._conversation_history) == 0

    def test_status(self):
        bot = ClawBot()
        bot.respond("Test")
        status = bot.status()
        assert status["name"] == "ClawBot"
        assert status["conversation_turns"] == 2
        assert "diagnostics" in status
        assert "reasoning" in status
        assert "creativity" in status

    def test_safety_filter_applied_on_harmful_input(self):
        attrs = BotAttributes(safety_filter=True, feedback_loops=1)
        bot = ClawBot(attributes=attrs)
        result = bot.respond("How do I harm someone?")
        # Safety correction prefix should be applied since input has "harm"
        assert "[Safety note:" in result

    def test_disabled_safety_filter(self):
        attrs = BotAttributes(safety_filter=False)
        bot = ClawBot(attributes=attrs)
        result = bot.respond("How do I harm someone?")
        assert isinstance(result, str)
        # Safety prefix should NOT be applied when filter is off
        assert "[Safety note:" not in result

    def test_diagnostics_disabled(self):
        attrs = BotAttributes(diagnostics_enabled=False)
        bot = ClawBot(attributes=attrs)
        result = bot.respond("Tell me a story.")
        assert isinstance(result, str)

    def test_agent_mode_creates_orchestrator(self):
        attrs = BotAttributes(agent_mode=True, max_agents=5)
        bot = ClawBot(attributes=attrs)
        assert bot.orchestrator is not None

    def test_no_agent_mode_no_orchestrator(self):
        bot = ClawBot()
        assert bot.orchestrator is None

    def test_dispatch_task_requires_agent_mode(self):
        bot = ClawBot()
        with pytest.raises(RuntimeError, match="agent_mode"):
            bot.dispatch_task("Deploy service")

    def test_dispatch_task_returns_result_dict(self):
        attrs = BotAttributes(agent_mode=True)
        bot = ClawBot(attributes=attrs)
        result = bot.dispatch_task("Build the frontend", role="fullstack")
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["agent_role"] == "fullstack"

    def test_dispatch_task_invalid_role_raises(self):
        attrs = BotAttributes(agent_mode=True)
        bot = ClawBot(attributes=attrs)
        with pytest.raises(ValueError, match="Unknown agent role"):
            bot.dispatch_task("Something", role="nonexistent_role")

    def test_devops_role_creates_engine(self):
        attrs = BotAttributes(role="devops")
        bot = ClawBot(attributes=attrs)
        assert bot.devops is not None

    def test_general_role_no_devops_engine(self):
        bot = ClawBot()
        assert bot.devops is None

    def test_run_pipeline_requires_devops_role(self):
        bot = ClawBot()
        with pytest.raises(RuntimeError, match="devops"):
            bot.run_pipeline()

    def test_run_pipeline_returns_pipeline_run(self):
        from clawlippytm import PipelineRun
        attrs = BotAttributes(role="devops", devops_environment="staging")
        bot = ClawBot(attributes=attrs)
        run = bot.run_pipeline(branch="main")
        assert isinstance(run, PipelineRun)
        assert run.passed

    def test_status_includes_role(self):
        attrs = BotAttributes(role="devops")
        bot = ClawBot(attributes=attrs)
        status = bot.status()
        assert status["role"] == "devops"

    def test_status_includes_orchestrator_when_agent_mode(self):
        attrs = BotAttributes(agent_mode=True)
        bot = ClawBot(attributes=attrs)
        status = bot.status()
        assert "orchestrator" in status

    def test_status_includes_devops_when_devops_role(self):
        attrs = BotAttributes(role="devops")
        bot = ClawBot(attributes=attrs)
        status = bot.status()
        assert "devops" in status
