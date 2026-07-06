"""
Tests for clawlippytm.devops
"""
import pytest
from clawlippytm import DevOpsEngine, PipelineRun, PipelineStage, StageResult


class TestPipelineStage:
    def test_all_stages_exist(self):
        expected = {"build", "test", "lint", "deploy", "monitor"}
        actual = {s.value for s in PipelineStage}
        assert expected == actual

    def test_stage_is_string_enum(self):
        assert PipelineStage.BUILD == "build"


class TestStageResult:
    def test_fields(self):
        result = StageResult(
            stage=PipelineStage.BUILD,
            passed=True,
            duration_ms=12.5,
            message="OK",
        )
        assert result.stage == PipelineStage.BUILD
        assert result.passed is True
        assert result.duration_ms == 12.5

    def test_to_dict(self):
        result = StageResult(
            stage=PipelineStage.TEST,
            passed=False,
            duration_ms=3.14,
            message="Tests failed",
            metrics={"tests_failed": 2},
        )
        d = result.to_dict()
        assert d["stage"] == "test"
        assert d["passed"] is False
        assert d["message"] == "Tests failed"
        assert d["metrics"]["tests_failed"] == 2


class TestPipelineRun:
    def test_empty_run_not_passed(self):
        run = PipelineRun()
        assert not run.passed

    def test_all_passed(self):
        run = PipelineRun()
        run.stages = [
            StageResult(stage=PipelineStage.BUILD, passed=True, duration_ms=1.0),
            StageResult(stage=PipelineStage.TEST, passed=True, duration_ms=2.0),
        ]
        assert run.passed

    def test_one_failed_stage(self):
        run = PipelineRun()
        run.stages = [
            StageResult(stage=PipelineStage.BUILD, passed=True, duration_ms=1.0),
            StageResult(stage=PipelineStage.TEST, passed=False, duration_ms=2.0),
        ]
        assert not run.passed
        assert PipelineStage.TEST in run.failed_stages

    def test_total_duration_is_non_negative(self):
        run = PipelineRun()
        assert run.total_duration_ms >= 0

    def test_to_dict(self):
        run = PipelineRun(branch="feature-x")
        run.stages.append(
            StageResult(stage=PipelineStage.LINT, passed=True, duration_ms=0.5)
        )
        d = run.to_dict()
        assert d["branch"] == "feature-x"
        assert "run_id" in d
        assert "stages" in d
        assert len(d["stages"]) == 1


class TestDevOpsEngine:
    def test_default_stages(self):
        engine = DevOpsEngine()
        assert engine.stages == DevOpsEngine.DEFAULT_STAGES

    def test_custom_environment(self):
        engine = DevOpsEngine(environment="production")
        assert engine.environment == "production"

    def test_run_pipeline_returns_run(self):
        engine = DevOpsEngine()
        run = engine.run_pipeline(branch="main")
        assert isinstance(run, PipelineRun)
        assert run.branch == "main"

    def test_full_pipeline_passes(self):
        engine = DevOpsEngine()
        run = engine.run_pipeline()
        assert run.passed
        assert len(run.stages) == len(DevOpsEngine.DEFAULT_STAGES)

    def test_partial_pipeline(self):
        engine = DevOpsEngine(stages=[PipelineStage.BUILD, PipelineStage.TEST])
        run = engine.run_pipeline()
        assert run.passed
        stage_names = {s.stage for s in run.stages}
        assert PipelineStage.BUILD in stage_names
        assert PipelineStage.TEST in stage_names
        assert PipelineStage.DEPLOY not in stage_names

    def test_pipeline_records_all_stages(self):
        engine = DevOpsEngine()
        run = engine.run_pipeline()
        stages_executed = [s.stage for s in run.stages]
        assert stages_executed == DevOpsEngine.DEFAULT_STAGES

    def test_multiple_runs_tracked(self):
        engine = DevOpsEngine()
        engine.run_pipeline("main")
        engine.run_pipeline("feature-branch")
        assert engine.summary()["total_runs"] == 2

    def test_health_check_no_runs(self):
        engine = DevOpsEngine()
        health = engine.health_check()
        assert health["status"] == "no_runs"
        assert health["pass_rate"] is None

    def test_health_check_after_run(self):
        engine = DevOpsEngine()
        engine.run_pipeline()
        health = engine.health_check()
        assert health["status"] == "healthy"
        assert health["pass_rate"] == 100.0
        assert "last_run" in health

    def test_summary_keys(self):
        engine = DevOpsEngine()
        engine.run_pipeline()
        s = engine.summary()
        assert "environment" in s
        assert "configured_stages" in s
        assert "total_runs" in s
        assert "passed_runs" in s
        assert "failed_runs" in s
        assert s["passed_runs"] == 1
        assert s["failed_runs"] == 0

    def test_stage_metrics_present(self):
        engine = DevOpsEngine()
        run = engine.run_pipeline()
        # Test stage should have test metrics
        test_stage = next(s for s in run.stages if s.stage == PipelineStage.TEST)
        assert "tests_run" in test_stage.metrics
        assert "coverage_pct" in test_stage.metrics
        # Monitor stage should have uptime metric
        monitor_stage = next(s for s in run.stages if s.stage == PipelineStage.MONITOR)
        assert "uptime_pct" in monitor_stage.metrics

    def test_build_artifact_generated(self):
        engine = DevOpsEngine()
        run = engine.run_pipeline(branch="release")
        build_stage = next(s for s in run.stages if s.stage == PipelineStage.BUILD)
        assert len(build_stage.artifacts) > 0
        assert "release" in build_stage.artifacts[0]

    def test_finished_at_set_after_run(self):
        engine = DevOpsEngine()
        run = engine.run_pipeline()
        assert run.finished_at is not None
        assert run.total_duration_ms >= 0
