"""FastAPI service.

Endpoints:
    GET  /search?query=...    translate -> retrieve -> (optional) rerank
    POST /feedback            record thumbs-up / thumbs-down
    GET  /healthz             readiness probe
    GET  /stats               provider + key-rotator state (debug)

Note: in-process LLM caching is currently disabled. See app/config.py.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# CACHE DISABLED — early-stage dev. Uncomment to re-enable cache observability.
# from app.cache import reranker_cache, translator_cache
from app.config import (
    FINAL_TOP_K,
    LLM_PROVIDER,
    RECOMMEND_ENABLED,
    RERANK_ENABLED,
    RERANK_POOL_K,
    RETRIEVAL_TOP_K,
    SPONSORED_ENABLED,
    TRANSLATOR_MODE,
)
from app.feedback import record_feedback
from app.metrics import StageTimings
from app.recommendations import recommend as build_recommendations
from app.reranker import rerank as llm_rerank
from app.search import load_index, search_products
from app.sponsored import get_sponsored
from app.translator import translate_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Warming up: provider=%s, translator_mode=%s",
        LLM_PROVIDER,
        TRANSLATOR_MODE,
    )
    load_index()
    logger.info("Service ready.")
    yield


app = FastAPI(title="Context-Aware Agentic Search", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Quick visibility into provider state and (Gemini) key-rotator status."""
    out = {
        "llm_provider": LLM_PROVIDER,
        "translator_mode": TRANSLATOR_MODE,
        "cache": "disabled",  # cache is hard-off; see app/config.py
    }
    # Key rotator stats only meaningful for Gemini.
    if LLM_PROVIDER == "gemini":
        try:
            from app.llm_client import get_llm_client

            backend = get_llm_client()
            if hasattr(backend, "rotator"):
                out["gemini_keys"] = backend.rotator.stats()
        except Exception as e:  # noqa: BLE001
            out["gemini_keys"] = {"error": repr(e)}
    return out


def _paginate(ranked_pool, candidates, page: int, page_size: int):
    """Slice one page out of the ranked pool, with an embedding-order tail.

    The reranker scores a deep pool ONCE (RERANK_POOL_K items). Pages are
    sliced from that single ranked list, so paging costs no extra LLM call.
    Any deduped candidate not in the reranked pool forms an embedding-order
    tail for deep pages — each tail item is labelled in its `reason` so the
    UI can tell the user those rows are beyond the reranked window.

    Returns (page_items, full_ordered_list).
    """
    pool_titles = {p.get("Product_title", "") for p in ranked_pool}
    tail = []
    for c in candidates:
        if c.get("Product_title", "") in pool_titles:
            continue
        c = dict(c)
        c.setdefault("rerank_score", None)
        c.setdefault("bayesian_rating", None)
        c.setdefault("final_score", None)
        c["reason"] = "(beyond reranked pool — embedding-similarity order)"
        tail.append(c)

    full = list(ranked_pool) + tail
    start = (page - 1) * page_size
    return full[start : start + page_size], full


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, max_length=300),
    top_k: int = Query(FINAL_TOP_K, ge=1, le=50, description="Results per page."),
    page: int = Query(1, ge=1, description="1-based page number."),
    rerank: bool = Query(RERANK_ENABLED, description="Toggle the LLM rerank stage."),
    recommend: bool = Query(
        RECOMMEND_ENABLED, description="Include cross-sell/upsell (page 1 only)."
    ),
    sponsored: bool = Query(
        SPONSORED_ENABLED, description="Include sponsored placements (page 1 only)."
    ),
    translator_mode: Optional[str] = Query(
        None,
        description="Override TRANSLATOR_MODE for this request: query_expansion | hyde | hybrid",
    ),
):
    """Translate -> retrieve -> rerank -> paginate, plus optional sponsored
    and cross-sell layers. Every call hits the real LLM (cache disabled).

    Note: this is a stateless endpoint, so navigating to a new page re-runs
    translate + rerank. The reranked POOL is a fixed size (independent of the
    page number) and rerank is deterministic (temp 0 + seed), so the ordering
    and total_results are STABLE across pages — page 2 slices the same ranking
    page 1 did. To make deep paging free of per-page LLM cost, re-enable the
    (currently disabled) LLM cache or hold the ranked pool in server state."""
    # Validate the per-request override BEFORE the try block, so the 422 is
    # not swallowed by the catch-all and turned into a 500.
    if translator_mode is not None and translator_mode not in {
        "query_expansion",
        "hyde",
        "hybrid",
    }:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown translator_mode={translator_mode!r}; "
                "expected query_expansion | hyde | hybrid"
            ),
        )
    page_size = top_k
    timings = StageTimings()
    try:
        with timings.stage("translate"):
            intents = translate_query(query, mode=translator_mode)

        with timings.stage("retrieve"):
            candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)

        # Rerank a deep pool ONCE, then paginate within it. The pool size is
        # FIXED (does NOT grow with the page number) so the ranking — and thus
        # total_results / page boundaries — stays identical across pages. It's
        # at least page_size so page 1 is fully reranked (not padded from the
        # embedding tail).
        pool_k = max(RERANK_POOL_K, page_size)
        rerank_actually_ran = False
        if rerank and candidates:
            with timings.stage("rerank"):
                ranked_pool = llm_rerank(query, candidates, top_k=pool_k)
            rerank_actually_ran = any(
                r.get("rerank_score") is not None for r in ranked_pool
            )
        else:
            ranked_pool = []
            for p in candidates[:pool_k]:
                p = dict(p)
                p["rerank_score"] = None
                p["bayesian_rating"] = None
                p["final_score"] = None
                p["reason"] = "(rerank disabled)"
                ranked_pool.append(p)

        results, full = _paginate(ranked_pool, candidates, page, page_size)
        total_results = len(full)
        total_pages = max(1, -(-total_results // page_size))  # ceil

        # Sponsored + cross-sell are page-1 concerns only (keeps quota down and
        # matches how a storefront shows ads / "bought together" up top).
        sponsored_items = []
        if sponsored and page == 1:
            with timings.stage("sponsored"):
                sponsored_items = get_sponsored(query, results)

        recommendations = {"cross_sell": [], "upsell": []}
        if recommend and page == 1 and results:
            with timings.stage("recommend"):
                recommendations = build_recommendations(query, results, page=page)

        return {
            "user_query": query,
            "interpreted_as": intents,
            "translator_mode": translator_mode or TRANSLATOR_MODE,
            "rerank_requested": rerank,
            "rerank_succeeded": rerank_actually_ran,
            "results": results,
            "sponsored": sponsored_items,
            "recommendations": recommendations,
            "page": page,
            "page_size": page_size,
            "total_results": total_results,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "latency_ms": timings.to_dict(),
        }

    except Exception as e:  # noqa: BLE001
        logger.exception("Search failed for query=%r", query)
        raise HTTPException(status_code=500, detail=str(e))


# CACHE DISABLED — early-stage dev. Endpoint removed; cache is never populated.
# To re-enable: restore cache imports above and uncomment this block.
# @app.post("/cache/clear")
# def clear_cache():
#     """Clear both translator and reranker in-process caches."""
#     t_stats = translator_cache.stats()
#     r_stats = reranker_cache.stats()
#     translator_cache._store.clear()
#     reranker_cache._store.clear()
#     translator_cache.hits = translator_cache.misses = 0
#     reranker_cache.hits = reranker_cache.misses = 0
#     return {
#         "status": "cleared",
#         "translator": {"prev_size": t_stats["size"]},
#         "reranker": {"prev_size": r_stats["size"]},
#     }


# --- Feedback endpoint ----------------------------------------------------


class FeedbackIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=300)
    product_title: str = Field(..., min_length=1)
    rating: int = Field(..., ge=-1, le=1)
    rank: int = Field(..., ge=1)
    reason: Optional[str] = None


@app.post("/feedback")
def feedback(payload: FeedbackIn):
    if payload.rating == 0:
        raise HTTPException(status_code=422, detail="rating must be -1 or +1")
    record_feedback(
        query=payload.query,
        product_title=payload.product_title,
        rating=payload.rating,
        rank=payload.rank,
        reason=payload.reason,
    )
    return {"status": "recorded"}
