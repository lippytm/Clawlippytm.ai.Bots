"""Metrics collector — captures counters, gauges, histograms and timings."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricSnapshot:
    name: str
    value: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Thread-safe collector for runtime metrics.

    Supports counters (monotonically increasing), gauges (arbitrary float),
    and histograms (list of observations for percentile analysis).

    Usage::

        mc = MetricsCollector()
        mc.increment("requests_total")
        mc.set_gauge("queue_depth", 42)
        mc.observe("response_time_ms", 123.4)
        snapshot = mc.snapshot()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._tags: Dict[str, Dict[str, str]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def increment(self, name: str, amount: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter by *amount* (default 1)."""
        if amount < 0:
            raise ValueError("Counter increment must be non-negative.")
        with self._lock:
            self._counters[name] += amount
            if tags:
                self._tags[name].update(tags)

    def get_counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    # ------------------------------------------------------------------
    # Gauges
    # ------------------------------------------------------------------

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge to an arbitrary float value."""
        with self._lock:
            self._gauges[name] = value
            if tags:
                self._tags[name].update(tags)

    def get_gauge(self, name: str) -> Optional[float]:
        with self._lock:
            return self._gauges.get(name)

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------

    def observe(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record an observation for a histogram metric."""
        with self._lock:
            self._histograms[name].append(value)
            if tags:
                self._tags[name].update(tags)

    def get_histogram(self, name: str) -> List[float]:
        with self._lock:
            return list(self._histograms.get(name, []))

    def percentile(self, name: str, p: float) -> Optional[float]:
        """Return the *p*-th percentile (0–100) of a histogram."""
        data = self.get_histogram(name)
        if not data:
            return None
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        idx = max(0, min(idx, len(sorted_data) - 1))
        return sorted_data[idx]

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> List[MetricSnapshot]:
        """Return a point-in-time snapshot of all tracked metrics."""
        ts = time.time()
        snapshots: List[MetricSnapshot] = []
        with self._lock:
            for name, value in self._counters.items():
                snapshots.append(MetricSnapshot(name=name, value=value, unit="count",
                                                tags=dict(self._tags.get(name, {})), timestamp=ts))
            for name, value in self._gauges.items():
                snapshots.append(MetricSnapshot(name=name, value=value,
                                                tags=dict(self._tags.get(name, {})), timestamp=ts))
            for name, values in self._histograms.items():
                if values:
                    avg = sum(values) / len(values)
                    snapshots.append(MetricSnapshot(name=f"{name}.avg", value=avg,
                                                    tags=dict(self._tags.get(name, {})), timestamp=ts))
        return snapshots

    def reset(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._tags.clear()
