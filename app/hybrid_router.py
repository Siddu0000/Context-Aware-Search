"""Hybrid search router — keyword-first, with a context-aware (CAS) fallback on a
keyword miss. The intent layer is injected so CAS stays swappable and testable."""

from typing import Callable, List, Optional

from app.keyword_search import KeywordSearchEngine

IntentSearchFn = Callable[[str, int], List[dict]]


def route(
    query: str,
    keyword_engine: KeywordSearchEngine,
    intent_search_fn: IntentSearchFn,
    *,
    k: int = 12,
    # 0.75 — Amazon titles are keyword-stuffed; a lenient 0.5 false-hit intent queries
    min_coverage: float = 0.75,
    min_results: int = 1,
) -> dict:
    """Return {query, path: "keyword"|"intent", reason, keyword_matched,
    keyword_coverage, results}; top-result term coverage decides hit vs miss."""
    kw_results, num_matched, _top_rel, top_coverage = keyword_engine.search(query, k=k)

    is_hit = num_matched >= min_results and top_coverage >= min_coverage
    if is_hit:
        return {
            "query": query,
            "path": "keyword",
            "reason": (
                f"keyword hit — top result covers {top_coverage:.0%} of query "
                f"terms across {num_matched} matches"
            ),
            "keyword_matched": num_matched,
            "keyword_coverage": round(top_coverage, 2),
            "results": kw_results,
        }

    intent_results = intent_search_fn(query, k) or []
    return {
        "query": query,
        "path": "intent",
        "reason": (
            f"keyword miss — top result covers only {top_coverage:.0%} of query "
            f"terms ({num_matched} lexical matches) -> context-aware fallback"
        ),
        "keyword_matched": num_matched,
        "keyword_coverage": round(top_coverage, 2),
        "results": intent_results,
    }
