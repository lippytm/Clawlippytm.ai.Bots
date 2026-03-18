"""Structured diagnostics logger with severity levels and context support."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional


class DiagnosticsLogger:
    """Thin wrapper around the standard :mod:`logging` module that emits
    JSON-structured log records and supports an optional context dictionary
    that is merged into every log entry.

    Usage::

        logger = DiagnosticsLogger("my_agent")
        logger.set_context({"agent_id": "agent-1", "run_id": "abc123"})
        logger.info("Task started", extra={"task": "reasoning"})
    """

    def __init__(self, name: str, level: int = logging.DEBUG) -> None:
        self._name = name
        self._context: Dict[str, Any] = {}
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(_JsonFormatter())
            self._logger.addHandler(handler)
        self._logger.setLevel(level)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def set_context(self, context: Dict[str, Any]) -> None:
        """Replace the current context dictionary."""
        self._context = dict(context)

    def update_context(self, **kwargs: Any) -> None:
        """Merge *kwargs* into the current context dictionary."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Remove all context keys."""
        self._context.clear()

    # ------------------------------------------------------------------
    # Logging convenience methods
    # ------------------------------------------------------------------

    def _emit(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {"timestamp": time.time(), "logger": self._name, "message": message}
        payload.update(self._context)
        if extra:
            payload.update(extra)
        self._logger.log(level, message, extra={"_json_payload": payload})

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._emit(logging.DEBUG, message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._emit(logging.INFO, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._emit(logging.WARNING, message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._emit(logging.ERROR, message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self._emit(logging.CRITICAL, message, extra)


class _JsonFormatter(logging.Formatter):
    """Formats log records as JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "_json_payload", None)
        if payload:
            return json.dumps(payload)
        base = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(base)
