"""Knowledge base — in-memory storage for facts, context, and learned data."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class KnowledgeEntry:
    key: str
    value: Any
    source: str = "manual"
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update(self, value: Any, confidence: float = 1.0, source: str = "manual") -> None:
        self.value = value
        self.confidence = confidence
        self.source = source
        self.updated_at = time.time()


class KnowledgeBase:
    """Thread-safe in-memory knowledge base for reasoning agents.

    Stores arbitrary key-value facts with optional confidence scores and
    provenance tracking.

    Usage::

        kb = KnowledgeBase()
        kb.store("sky_color", "blue", confidence=0.99)
        entry = kb.retrieve("sky_color")
        print(entry.value)  # "blue"
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: Dict[str, KnowledgeEntry] = {}

    def store(self, key: str, value: Any, confidence: float = 1.0, source: str = "manual") -> KnowledgeEntry:
        """Insert or update a knowledge entry."""
        if not key:
            raise ValueError("Knowledge key must not be empty.")
        with self._lock:
            if key in self._store:
                self._store[key].update(value, confidence=confidence, source=source)
            else:
                self._store[key] = KnowledgeEntry(key=key, value=value,
                                                  confidence=confidence, source=source)
            return self._store[key]

    def retrieve(self, key: str) -> Optional[KnowledgeEntry]:
        """Return the entry for *key*, or ``None`` if not found."""
        with self._lock:
            return self._store.get(key)

    def delete(self, key: str) -> bool:
        """Remove *key* from the knowledge base. Returns True if it existed."""
        with self._lock:
            return self._store.pop(key, None) is not None

    def search(self, prefix: str) -> List[KnowledgeEntry]:
        """Return all entries whose key starts with *prefix*."""
        with self._lock:
            return [e for k, e in self._store.items() if k.startswith(prefix)]

    def high_confidence(self, threshold: float = 0.8) -> List[KnowledgeEntry]:
        """Return entries whose confidence is at or above *threshold*."""
        with self._lock:
            return [e for e in self._store.values() if e.confidence >= threshold]

    def all_entries(self) -> List[KnowledgeEntry]:
        with self._lock:
            return list(self._store.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __iter__(self) -> Iterator[KnowledgeEntry]:
        with self._lock:
            entries = list(self._store.values())
        return iter(entries)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
