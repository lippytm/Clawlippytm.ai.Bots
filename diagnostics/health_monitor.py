"""Health monitor — tracks system and component health."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    component: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


@dataclass
class SystemHealthReport:
    overall_status: HealthStatus
    results: List[HealthCheckResult]
    timestamp: float = field(default_factory=time.time)

    @property
    def healthy_count(self) -> int:
        return sum(1 for r in self.results if r.is_healthy())

    @property
    def unhealthy_count(self) -> int:
        return len(self.results) - self.healthy_count


# Type alias for a health-check callable
HealthCheckFn = Callable[[], HealthCheckResult]


class HealthMonitor:
    """Monitors health of registered system components.

    Usage::

        monitor = HealthMonitor()
        monitor.register("database", lambda: HealthCheckResult("database", HealthStatus.HEALTHY))
        report = monitor.run_all_checks()
        print(report.overall_status)
    """

    def __init__(self) -> None:
        self._checks: Dict[str, HealthCheckFn] = {}

    def register(self, name: str, check_fn: HealthCheckFn) -> None:
        """Register a named health-check callable."""
        if not name:
            raise ValueError("Health-check name must not be empty.")
        self._checks[name] = check_fn

    def unregister(self, name: str) -> None:
        """Remove a previously registered health check."""
        self._checks.pop(name, None)

    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a single named health check and return its result."""
        check_fn = self._checks.get(name)
        if check_fn is None:
            return None
        start = time.perf_counter()
        result = check_fn()
        result.latency_ms = (time.perf_counter() - start) * 1000
        result.timestamp = time.time()
        return result

    def run_all_checks(self) -> SystemHealthReport:
        """Run every registered health check and return an aggregated report."""
        results: List[HealthCheckResult] = []
        for name in self._checks:
            result = self.run_check(name)
            if result is not None:
                results.append(result)

        if not results:
            overall = HealthStatus.UNKNOWN
        elif all(r.status == HealthStatus.HEALTHY for r in results):
            overall = HealthStatus.HEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return SystemHealthReport(overall_status=overall, results=results)
