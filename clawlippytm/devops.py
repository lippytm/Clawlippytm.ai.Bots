"""
clawlippytm.devops
~~~~~~~~~~~~~~~~~~

AI-powered DevOps pipeline engine for Clawlippytm.Bots.

The :class:`DevOpsEngine` simulates a full CI/CD pipeline lifecycle:

- **Build**   — source compilation / dependency resolution checks.
- **Test**    — automated test-suite execution and coverage tracking.
- **Lint**    — static analysis and code-style enforcement.
- **Deploy**  — environment-targeted deployment orchestration.
- **Monitor** — runtime health checks and telemetry collection.

Pipeline runs produce a :class:`PipelineRun` record that accumulates
:class:`StageResult` objects for each stage executed.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pipeline stage definition
# ---------------------------------------------------------------------------

class PipelineStage(str, Enum):
    """The ordered stages of a CI/CD pipeline."""

    BUILD = "build"
    TEST = "test"
    LINT = "lint"
    DEPLOY = "deploy"
    MONITOR = "monitor"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Result from executing a single pipeline stage."""

    stage: PipelineStage
    passed: bool
    duration_ms: float
    message: str = ""
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 3),
            "message": self.message,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }


@dataclass
class PipelineRun:
    """A complete record of one pipeline execution."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    branch: str = "main"
    stages: List[StageResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None

    @property
    def passed(self) -> bool:
        """True when every stage passed."""
        return bool(self.stages) and all(s.passed for s in self.stages)

    @property
    def failed_stages(self) -> List[PipelineStage]:
        """Return the list of stages that did not pass."""
        return [s.stage for s in self.stages if not s.passed]

    @property
    def total_duration_ms(self) -> float:
        """Wall-clock time for the full run in milliseconds."""
        end = self.finished_at or time.monotonic()
        return round((end - self.started_at) * 1000, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "branch": self.branch,
            "passed": self.passed,
            "failed_stages": [s.value for s in self.failed_stages],
            "total_duration_ms": self.total_duration_ms,
            "stages": [s.to_dict() for s in self.stages],
        }


# ---------------------------------------------------------------------------
# Stage runners (heuristic / simulation)
# ---------------------------------------------------------------------------

def _run_build(branch: str) -> StageResult:
    """Simulate a build stage."""
    start = time.monotonic()
    artifacts = [f"dist/{branch}-build.tar.gz"]
    elapsed = (time.monotonic() - start) * 1000
    return StageResult(
        stage=PipelineStage.BUILD,
        passed=True,
        duration_ms=elapsed,
        message=f"Build succeeded for branch '{branch}'.",
        artifacts=artifacts,
        metrics={"artifact_count": len(artifacts)},
    )


def _run_test(branch: str) -> StageResult:
    """Simulate a test stage."""
    start = time.monotonic()
    elapsed = (time.monotonic() - start) * 1000
    metrics = {
        "tests_run": 42,
        "tests_passed": 42,
        "tests_failed": 0,
        "coverage_pct": 95.0,
    }
    return StageResult(
        stage=PipelineStage.TEST,
        passed=True,
        duration_ms=elapsed,
        message="All 42 tests passed. Coverage: 95.0 %.",
        metrics=metrics,
    )


def _run_lint(branch: str) -> StageResult:
    """Simulate a lint / static-analysis stage."""
    start = time.monotonic()
    elapsed = (time.monotonic() - start) * 1000
    metrics = {"files_checked": 12, "issues_found": 0}
    return StageResult(
        stage=PipelineStage.LINT,
        passed=True,
        duration_ms=elapsed,
        message="Lint passed. No issues found across 12 files.",
        metrics=metrics,
    )


def _run_deploy(branch: str, environment: str) -> StageResult:
    """Simulate a deployment stage."""
    start = time.monotonic()
    elapsed = (time.monotonic() - start) * 1000
    return StageResult(
        stage=PipelineStage.DEPLOY,
        passed=True,
        duration_ms=elapsed,
        message=f"Deployed '{branch}' to environment '{environment}' successfully.",
        artifacts=[f"deploy://{environment}/{branch}"],
        metrics={"environment": environment, "replicas": 3},
    )


def _run_monitor(environment: str) -> StageResult:
    """Simulate a post-deployment monitoring / health-check stage."""
    start = time.monotonic()
    elapsed = (time.monotonic() - start) * 1000
    metrics = {
        "health": "healthy",
        "uptime_pct": 99.9,
        "p99_latency_ms": 42.0,
        "error_rate_pct": 0.01,
    }
    return StageResult(
        stage=PipelineStage.MONITOR,
        passed=True,
        duration_ms=elapsed,
        message=f"Health check passed for '{environment}'. All services operational.",
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# DevOps engine
# ---------------------------------------------------------------------------

class DevOpsEngine:
    """
    AI-powered DevOps pipeline engine.

    Manages CI/CD pipeline runs for one or more branches and environments.
    Each call to :meth:`run_pipeline` produces a :class:`PipelineRun` record
    capturing the outcome of every configured stage.

    Parameters
    ----------
    environment:
        Default deployment target (e.g. ``"staging"`` or ``"production"``).
    stages:
        Ordered list of :class:`PipelineStage` values to execute.
        Defaults to the full pipeline: BUILD → TEST → LINT → DEPLOY → MONITOR.
    """

    DEFAULT_STAGES: List[PipelineStage] = [
        PipelineStage.BUILD,
        PipelineStage.TEST,
        PipelineStage.LINT,
        PipelineStage.DEPLOY,
        PipelineStage.MONITOR,
    ]

    def __init__(
        self,
        environment: str = "staging",
        stages: Optional[List[PipelineStage]] = None,
    ) -> None:
        self.environment: str = environment
        self.stages: List[PipelineStage] = stages or list(self.DEFAULT_STAGES)
        self._runs: List[PipelineRun] = []

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def run_pipeline(self, branch: str = "main") -> PipelineRun:
        """
        Execute the configured pipeline stages for *branch*.

        Stages are executed in order; the pipeline stops on the first
        failure (fail-fast behaviour).

        Parameters
        ----------
        branch:
            Source branch to build and deploy.

        Returns
        -------
        PipelineRun
            Complete pipeline run record.
        """
        run = PipelineRun(branch=branch)
        stage_runners = {
            PipelineStage.BUILD: lambda: _run_build(branch),
            PipelineStage.TEST: lambda: _run_test(branch),
            PipelineStage.LINT: lambda: _run_lint(branch),
            PipelineStage.DEPLOY: lambda: _run_deploy(branch, self.environment),
            PipelineStage.MONITOR: lambda: _run_monitor(self.environment),
        }
        for stage in self.stages:
            result = stage_runners[stage]()
            run.stages.append(result)
            if not result.passed:
                break
        run.finished_at = time.monotonic()
        self._runs.append(run)
        return run

    def health_check(self) -> Dict[str, Any]:
        """
        Return a current health snapshot based on recent pipeline runs.

        Returns
        -------
        dict
            Health summary including pass rate and last run status.
        """
        if not self._runs:
            return {"status": "no_runs", "pass_rate": None, "last_run": None}
        total = len(self._runs)
        passed = sum(1 for r in self._runs if r.passed)
        last = self._runs[-1]
        return {
            "status": "healthy" if last.passed else "degraded",
            "pass_rate": round(passed / total * 100, 1),
            "total_runs": total,
            "last_run": last.to_dict(),
        }

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all pipeline runs."""
        return {
            "environment": self.environment,
            "configured_stages": [s.value for s in self.stages],
            "total_runs": len(self._runs),
            "passed_runs": sum(1 for r in self._runs if r.passed),
            "failed_runs": sum(1 for r in self._runs if not r.passed),
        }
