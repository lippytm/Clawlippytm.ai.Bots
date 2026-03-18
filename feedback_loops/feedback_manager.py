"""Feedback manager — collects outcome feedback and routes it to registered handlers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class FeedbackPolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class FeedbackRecord:
    source: str
    action: str
    polarity: FeedbackPolarity
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# A feedback handler receives a FeedbackRecord and does something with it.
FeedbackHandlerFn = Callable[[FeedbackRecord], None]


class FeedbackManager:
    """Collects :class:`FeedbackRecord` instances and dispatches them to
    registered handler callbacks, enabling closed-loop learning.

    Usage::

        fm = FeedbackManager()
        fm.register_handler("log_feedback", lambda r: print(r))
        fm.submit(FeedbackRecord(source="agent-1", action="search",
                                 polarity=FeedbackPolarity.POSITIVE, score=0.9))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[FeedbackRecord] = []
        self._handlers: Dict[str, FeedbackHandlerFn] = {}

    def register_handler(self, name: str, handler_fn: FeedbackHandlerFn) -> None:
        """Register a named feedback handler."""
        if not name:
            raise ValueError("Handler name must not be empty.")
        with self._lock:
            self._handlers[name] = handler_fn

    def unregister_handler(self, name: str) -> None:
        with self._lock:
            self._handlers.pop(name, None)

    def submit(self, record: FeedbackRecord) -> None:
        """Record feedback and invoke all registered handlers."""
        with self._lock:
            self._records.append(record)
            handlers = list(self._handlers.values())
        for handler in handlers:
            handler(record)

    def get_records(self, source: Optional[str] = None,
                    polarity: Optional[FeedbackPolarity] = None) -> List[FeedbackRecord]:
        """Return stored feedback records, optionally filtered."""
        with self._lock:
            records = list(self._records)
        if source:
            records = [r for r in records if r.source == source]
        if polarity:
            records = [r for r in records if r.polarity == polarity]
        return records

    def average_score(self, source: Optional[str] = None) -> Optional[float]:
        """Return the mean feedback score, or None if no records exist."""
        records = self.get_records(source=source)
        if not records:
            return None
        return sum(r.score for r in records) / len(records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
