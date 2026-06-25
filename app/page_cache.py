"""Single in-memory result cache for search.

Caches the full ranked pool for a query+settings combo. This makes both
repeat searches and page 2+ of the same search instant (no LLM calls). Only
clean runs are cached — the caller must NOT store a degraded result (e.g. one
where the reranker fell back after an LLM error), so a retry can hit the LLM
again. Process-local; cleared on restart. Separate from the on-disk catalog
embedding cache.
"""

from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from app.config import PAGE_CACHE_SIZE

_store: "OrderedDict[str, Any]" = OrderedDict()
_lock = Lock()


def make_key(query: str, settings: tuple) -> str:
    return repr((query.strip().lower(), settings))


def get(key: str) -> Optional[Any]:
    with _lock:
        if key in _store:
            _store.move_to_end(key)
            return _store[key]
        return None


def put(key: str, value: Any) -> None:
    with _lock:
        _store[key] = value
        _store.move_to_end(key)
        while len(_store) > PAGE_CACHE_SIZE:
            _store.popitem(last=False)


def clear() -> None:
    with _lock:
        _store.clear()
