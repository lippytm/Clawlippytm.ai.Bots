"""Tests for the diagnostics module."""

import pytest

from diagnostics.health_monitor import HealthCheckResult, HealthMonitor, HealthStatus, SystemHealthReport
from diagnostics.metrics_collector import MetricsCollector
from diagnostics.logger import DiagnosticsLogger


# -------------------------------------------------------------------------
# HealthMonitor
# -------------------------------------------------------------------------

class TestHealthMonitor:
    def test_register_and_run_single_check(self):
        monitor = HealthMonitor()
        monitor.register("svc", lambda: HealthCheckResult("svc", HealthStatus.HEALTHY, "ok"))
        result = monitor.run_check("svc")
        assert result is not None
        assert result.status == HealthStatus.HEALTHY
        assert result.is_healthy()

    def test_run_all_checks_all_healthy(self):
        monitor = HealthMonitor()
        monitor.register("a", lambda: HealthCheckResult("a", HealthStatus.HEALTHY))
        monitor.register("b", lambda: HealthCheckResult("b", HealthStatus.HEALTHY))
        report = monitor.run_all_checks()
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.healthy_count == 2
        assert report.unhealthy_count == 0

    def test_run_all_checks_one_unhealthy(self):
        monitor = HealthMonitor()
        monitor.register("good", lambda: HealthCheckResult("good", HealthStatus.HEALTHY))
        monitor.register("bad", lambda: HealthCheckResult("bad", HealthStatus.UNHEALTHY))
        report = monitor.run_all_checks()
        assert report.overall_status == HealthStatus.UNHEALTHY

    def test_run_all_checks_degraded(self):
        monitor = HealthMonitor()
        monitor.register("ok", lambda: HealthCheckResult("ok", HealthStatus.HEALTHY))
        monitor.register("deg", lambda: HealthCheckResult("deg", HealthStatus.DEGRADED))
        report = monitor.run_all_checks()
        assert report.overall_status == HealthStatus.DEGRADED

    def test_empty_checks_returns_unknown(self):
        monitor = HealthMonitor()
        report = monitor.run_all_checks()
        assert report.overall_status == HealthStatus.UNKNOWN

    def test_run_check_unknown_name(self):
        monitor = HealthMonitor()
        assert monitor.run_check("nonexistent") is None

    def test_unregister(self):
        monitor = HealthMonitor()
        monitor.register("svc", lambda: HealthCheckResult("svc", HealthStatus.HEALTHY))
        monitor.unregister("svc")
        assert monitor.run_check("svc") is None

    def test_register_empty_name_raises(self):
        monitor = HealthMonitor()
        with pytest.raises(ValueError):
            monitor.register("", lambda: HealthCheckResult("", HealthStatus.HEALTHY))

    def test_latency_is_recorded(self):
        monitor = HealthMonitor()
        monitor.register("svc", lambda: HealthCheckResult("svc", HealthStatus.HEALTHY))
        result = monitor.run_check("svc")
        assert result.latency_ms >= 0.0


# -------------------------------------------------------------------------
# MetricsCollector
# -------------------------------------------------------------------------

class TestMetricsCollector:
    def test_counter_increment(self):
        mc = MetricsCollector()
        mc.increment("hits")
        mc.increment("hits", 4)
        assert mc.get_counter("hits") == 5.0

    def test_counter_negative_raises(self):
        mc = MetricsCollector()
        with pytest.raises(ValueError):
            mc.increment("hits", -1)

    def test_gauge_set_and_get(self):
        mc = MetricsCollector()
        mc.set_gauge("temperature", 98.6)
        assert mc.get_gauge("temperature") == 98.6

    def test_gauge_missing_returns_none(self):
        mc = MetricsCollector()
        assert mc.get_gauge("missing") is None

    def test_histogram_observe_and_percentile(self):
        mc = MetricsCollector()
        for v in range(1, 101):
            mc.observe("latency", float(v))
        p50 = mc.percentile("latency", 50)
        assert p50 is not None
        assert 45 <= p50 <= 55

    def test_histogram_empty_percentile(self):
        mc = MetricsCollector()
        assert mc.percentile("missing", 90) is None

    def test_snapshot_contains_metrics(self):
        mc = MetricsCollector()
        mc.increment("req")
        mc.set_gauge("mem", 1.0)
        mc.observe("rt", 10.0)
        snapshot = mc.snapshot()
        names = {s.name for s in snapshot}
        assert "req" in names
        assert "mem" in names
        assert "rt.avg" in names

    def test_reset_clears_all(self):
        mc = MetricsCollector()
        mc.increment("x")
        mc.reset()
        assert mc.get_counter("x") == 0.0


# -------------------------------------------------------------------------
# DiagnosticsLogger
# -------------------------------------------------------------------------

class TestDiagnosticsLogger:
    def test_logger_creation(self):
        logger = DiagnosticsLogger("test.logger")
        assert logger is not None

    def test_context_management(self):
        logger = DiagnosticsLogger("test.ctx")
        logger.set_context({"agent_id": "a1"})
        assert logger._context["agent_id"] == "a1"
        logger.update_context(run_id="r1")
        assert logger._context["run_id"] == "r1"
        logger.clear_context()
        assert logger._context == {}

    def test_log_methods_do_not_raise(self):
        logger = DiagnosticsLogger("test.emit")
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warn msg")
        logger.error("error msg")
        logger.critical("critical msg")
