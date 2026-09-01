"""In-memory LRU cache of the ranked pool per query+settings, so page 2+ is instant."""

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
    # callers must only store clean runs, so a degraded LLM fallback retries next time
    with _lock:
        _store[key] = value
        _store.move_to_end(key)
        while len(_store) > PAGE_CACHE_SIZE:
            _store.popitem(last=False)


def clear() -> None:
    with _lock:
        _store.clear()
