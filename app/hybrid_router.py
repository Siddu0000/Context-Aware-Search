"""Hybrid search router — the "Swiggy model": one search bar, keyword-first,
with a fallback to context-aware (intent) search when the keyword engine
misses.

Flow:
    query -> keyword engine (lexical, fast)
          -> miss?  no  -> return keyword results        (the common case)
                    yes -> intent layer (CAS) -> return   (the catch layer)

The intent layer is injected (`intent_search_fn`) so this stays testable and
CAS stays swappable:
  * production: a callable that runs the real CAS pipeline and returns results
  * demo/tests: a lightweight stand-in

A "miss" is intentionally simple here (few lexical matches / weak top score).
In a real deployment this is where GA4/engagement signals would sharpen the
trigger — see the integration runbook.
"""

from typing import Callable, List, Optional

from app.keyword_search import KeywordSearchEngine

# intent_search_fn(query: str, k: int) -> List[dict]
IntentSearchFn = Callable[[str, int], List[dict]]


def route(
    query: str,
    keyword_engine: KeywordSearchEngine,
    intent_search_fn: IntentSearchFn,
    *,
    k: int = 12,
    # 0.75 = the top result may miss at most a quarter of the asked terms.
    # Amazon titles are keyword-stuffed ("...Breathable...Mothers Day...Gift"),
    # so a lenient 0.5 let attribute-style intent queries ("something
    # breathable for a humid day") false-hit on stuffed titles. A keyword HIT
    # should mean the result contains essentially everything asked; anything
    # less falls through to CAS, which costs latency but never quality.
    min_coverage: float = 0.75,
    min_results: int = 1,
) -> dict:
    """Return a unified payload:
        {
          "query", "path": "keyword"|"intent",
          "reason", "keyword_matched", "keyword_coverage", "results": [...]
        }

    A query is a keyword HIT when the top lexical result actually contains
    enough of what was asked (term coverage >= min_coverage) and at least
    min_results matched. Otherwise it's a miss and we fall back to the intent
    (CAS) layer. Coverage — not raw count — is the primary signal, so a precise
    query that legitimately matches only 1-2 products still counts as a hit.
    In production, GA4/engagement signals would refine this trigger."""
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

    # Miss -> hand off to the intent (CAS) layer.
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
