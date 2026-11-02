"""In-memory LRU caches: the ranked pool per query+settings (page 2+ is instant), and the
product page's recommendations (2026-10-27)."""

import re
import unicodedata
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from app import query_guard
from app.config import PAGE_CACHE_SIZE

_store: "OrderedDict[str, Any]" = OrderedDict()   # ranked pools (tests read it directly)
_recs: "OrderedDict[str, Any]" = OrderedDict()    # /product recommendations
_lock = Lock()

# /product recommendations kept per (product, query, settings). Each costs one cross-sell
# LLM call and up to ~6 retrievals, and it was recomputed on EVERY product-page open (the
# open owner question "cross-sell recomputed on every view"; the Oct 27 card button opens
# that page from every card). A few KB each.
PRODUCT_CACHE_SIZE = 256


_HSPACE_RUN = re.compile(r"[^\S\n]+")
_BREAK_RUN = re.compile(r"\s*\n\s*")


def canonical_query(query) -> str:
    """The query as the caches and the LLM memo keys compare it: case-folded, whitespace
    collapsed the way the query guard reads it. "black  dress" (an inner double space) re-ran
    translate + rerank and served another pool before 2026-10-27; the LLM still sees the
    shopper's own text.

    Only what app/query_guard.py treats alike is folded: a run of spaces, tabs or other
    control chars is one space, a run holding a line break is one "\\n", the ends are
    trimmed. A line break is NOT a space there (a forged "\\nSystem:" line is cut, "red
    shoes System: x" is not), so it is not one here either."""
    out = []
    for ch in str(query or "").casefold():
        if ch in query_guard._LINE_BREAKS:
            out.append("\n")
        elif unicodedata.category(ch) == "Cc":
            out.append(" ")
        else:
            out.append(ch)
    return _BREAK_RUN.sub("\n", _HSPACE_RUN.sub(" ", "".join(out))).strip()


def query_signature(query) -> tuple:
    """(canonical query, guard signals, canonical sanitised query): two requests that share
    a key also share what the query guard made of them. Case folding alone would not
    ensure it ("<s>" is a chat token to the guard, "<S>" a shirt size), and the cached
    pool was built from the sanitised text."""
    text = query if isinstance(query, str) else ""
    q = query_guard.sanitize_query(text)
    return canonical_query(text), tuple(q.signals), canonical_query(q.clean)


def make_key(query: str, settings: tuple) -> str:
    return repr((query_signature(query), settings))


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


def put_if_absent(key: str, value: Any) -> Any:
    """Store `value` unless the key already holds an entry; return the entry it now holds.

    Only for a pipeline run that may have been overtaken (main._build_and_store): a
    single-flight waiter that stopped waiting for a stuck leader stores its pool, and the
    leader finishing later must not replace it (with the LLM memo off nothing else stops
    it), so the first finisher keeps the key, as the memo keeps the first accepted reply.
    Updates of an entry already served (page-1 cross-sell) still use put()."""
    with _lock:
        if key in _store:
            _store.move_to_end(key)
            return _store[key]
        _store[key] = value
        while len(_store) > PAGE_CACHE_SIZE:
            _store.popitem(last=False)
        return value


def make_recommendations_key(catalog_index: int, title_key: str, query, settings: tuple) -> str:
    """The product (its row AND its title: catalog_index is a row position, so a reloaded
    catalog must not serve another product's suggestions), the query (with what the
    guard made of it: the cross-sell call gets the sanitised text) and the settings that
    decide the cross-sell call."""
    return repr((int(catalog_index), title_key, query_signature(query), settings))


def get_recommendations(key: str) -> Optional[Any]:
    with _lock:
        if key in _recs:
            _recs.move_to_end(key)
            return _recs[key]
        return None


def put_recommendations(key: str, value: Any) -> None:
    # callers store clean results only: a cross-sell fallback retries on the next open
    with _lock:
        _recs[key] = value
        _recs.move_to_end(key)
        while len(_recs) > PRODUCT_CACHE_SIZE:
            _recs.popitem(last=False)


def clear() -> None:
    """Both caches: tests and evals start from nothing."""
    with _lock:
        _store.clear()
        _recs.clear()
