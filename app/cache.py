"""Process-local LRU cache for LLM responses.

  ============================================================================
  NOT CURRENTLY WIRED IN. Translator and reranker bypass this module entirely
  during the early-stage development phase. See app/config.py for the
  rationale and the locations of the commented-out integration points.
  This file is kept as a working implementation for the re-enable path.
  ============================================================================

Why this exists:
- Each /search makes 2 LLM calls (translator + reranker).
- Free-tier daily quotas are tight. Identical inputs should not hit the API twice.
- This is also one half of the "determinism" answer: cached inputs return
  bit-identical outputs within a session.

Process-local on purpose: restarting uvicorn clears the cache, so a fresh
boot exercises the real LLM. We do NOT cache fallback responses — when the
LLM fails, the next call retries.
"""

import hashlib
import json
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional


class LRUCache:
    def __init__(self, maxsize: int = 512):
        self._maxsize = maxsize
        self._store: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            if len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0,
                "maxsize": self._maxsize,
            }


def make_key(*parts: Any) -> str:
    """Deterministic SHA1 hash of JSON-serializable inputs."""
    raw = json.dumps(parts, default=str, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()


# Shared caches — one per LLM stage so we can monitor independently.
translator_cache = LRUCache(maxsize=512)
reranker_cache = LRUCache(maxsize=512)
