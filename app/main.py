"""FastAPI service.

Endpoints:
    GET  /search?query=...            translate -> retrieve -> rerank -> paginate
    GET  /product?catalog_index=...   single product + its cross-sell/upsell
    POST /feedback                    record thumbs-up / thumbs-down
    GET  /healthz, /stats             probes / debug
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app import page_cache
from app.config import (
    FINAL_TOP_K,
    GEMINI_MODEL,
    GROCERY_PER_INTENT_K,
    LLM_PROVIDER,
    MIN_RESULT_RELEVANCE,
    OPENAI_MODEL,
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
from app.search import best_score, get_product, load_index, search_products
from app.sponsored import sponsored_map
from app.translator import is_recipe_query, translate_query

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
    active_model = (
        GEMINI_MODEL if LLM_PROVIDER == "gemini"
        else OPENAI_MODEL if LLM_PROVIDER == "openai"
        else LLM_PROVIDER
    )
    out = {
        "llm_provider": LLM_PROVIDER,
        "active_model": active_model,
        "translator_mode": TRANSLATOR_MODE,
    }
    if LLM_PROVIDER == "gemini":
        try:
            from app.llm_client import get_llm_client

            backend = get_llm_client()
            if hasattr(backend, "rotator"):
                out["gemini_keys"] = backend.rotator.stats()
        except Exception as e:
            out["gemini_keys"] = {"error": repr(e)}
    return out


def _assemble_full(ranked_pool, candidates):
    """Full ordered result list: the reranked pool, then any remaining
    candidates as an embedding-order tail (labelled in `reason`)."""
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
    return list(ranked_pool) + tail


def _diversify_by_ingredient(full):
    """Round-robin results by their source ingredient so each ingredient
    appears once before any appears twice. Stable: preserves the existing
    (reranked) order within each ingredient group, and the relative order of
    first-appearances. Items without a source_intent are treated as their own
    group and kept in place."""
    groups = {}          # source_intent -> list of items (in current order)
    order = []           # first-seen order of group keys
    for item in full:
        key = item.get("source_intent") or item.get("Product_title")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)

    out = []
    while any(groups[k] for k in order):
        for k in order:
            if groups[k]:
                out.append(groups[k].pop(0))
    return out


def _boost_sponsored(full, smap):
    """Pull sponsored products that made the RERANKED pool to the top, ordered
    by bid (then final_score). Sponsored items outside the reranked pool are
    left in place — they weren't relevant enough to be boosted."""
    if not smap:
        return full
    boosted, rest = [], []
    for item in full:
        asin = item.get("parent_asin")
        asin = str(asin) if asin is not None else None
        if asin and asin in smap and item.get("rerank_score") is not None:
            item = dict(item)
            item["is_sponsored"] = True
            item["sponsor"] = smap[asin]["sponsor"]
            item["_bid"] = smap[asin]["bid"]
            boosted.append(item)
        else:
            rest.append(item)
    boosted.sort(key=lambda x: (x.get("_bid", 0.0), x.get("final_score") or 0.0), reverse=True)
    return boosted + rest


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
        description="Override TRANSLATOR_MODE: query_expansion | hyde | hybrid",
    ),
):
    """Translate -> retrieve -> rerank -> paginate, plus optional sponsored
    and cross-sell layers.

    Page 1 runs the full pipeline and caches the ranked pool. Page 2+ of the
    same search reuse that pool (no LLM call), so page transitions are fast.
    If the best match is below MIN_RESULT_RELEVANCE, the query is treated as
    having no real match and an empty result set is returned."""
    if translator_mode is not None and translator_mode not in {
        "query_expansion", "hyde", "hybrid",
    }:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown translator_mode={translator_mode!r}; "
                "expected query_expansion | hyde | hybrid"
            ),
        )
    page_size = top_k
    pool_k = max(RERANK_POOL_K, page_size)
    timings = StageTimings()

    cache_key = page_cache.make_key(
        query, (page_size, pool_k, rerank, sponsored, translator_mode or TRANSLATOR_MODE)
    )

    try:
        cached = page_cache.get(cache_key)
        if cached is not None:
            full = cached["full"]
            intents = cached["intents"]
            rerank_actually_ran = cached["rerank_succeeded"]
            no_match = cached["no_match"]
            start = (page - 1) * page_size
            results = full[start : start + page_size]
            sponsored_items = [r for r in results if r.get("is_sponsored")]
            recommendations = (
                cached["recommendations"]
                if page == 1
                else {"cross_sell": [], "upsell": []}
            )
            errors = []
        else:
            errors = []
            with timings.stage("translate"):
                intents = translate_query(query, mode=translator_mode, errors=errors)

            with timings.stage("retrieve"):
                if is_recipe_query(query):
                    candidates = search_products(
                        intents,
                        top_k=RETRIEVAL_TOP_K,
                        per_intent_quota=GROCERY_PER_INTENT_K,
                    )
                else:
                    candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)

            no_match = bool(candidates) and best_score(candidates) < MIN_RESULT_RELEVANCE
            if not candidates or no_match:
                empty_payload = {
                    "user_query": query,
                    "interpreted_as": intents,
                    "translator_mode": translator_mode or TRANSLATOR_MODE,
                    "rerank_requested": rerank,
                    "rerank_succeeded": False,
                    "no_match": True,
                    "message": "No matching products found. Try different or more general terms.",
                    "results": [],
                    "sponsored": [],
                    "recommendations": {"cross_sell": [], "upsell": []},
                    "errors": errors,
                    "page": 1,
                    "page_size": page_size,
                    "total_results": 0,
                    "total_pages": 1,
                    "has_prev": False,
                    "has_next": False,
                    "latency_ms": timings.to_dict(),
                }
                if not errors:
                    page_cache.put(
                        cache_key,
                        {
                            "full": [],
                            "intents": intents,
                            "rerank_succeeded": False,
                            "no_match": True,
                            "recommendations": {"cross_sell": [], "upsell": []},
                        },
                    )
                return empty_payload

            rerank_actually_ran = False
            if rerank and candidates:
                with timings.stage("rerank"):
                    ranked_pool = llm_rerank(query, candidates, top_k=pool_k, errors=errors)
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

            full = _assemble_full(ranked_pool, candidates)

            # Recipe queries: round-robin by source ingredient so page 1 shows
            # one of each ingredient (flour, eggs, butter, sugar...) before a
            # second of any — instead of three flours. Preserves rank within
            # each ingredient; sponsored items (boosted next) are unaffected.
            if is_recipe_query(query):
                full = _diversify_by_ingredient(full)

            if sponsored:
                with timings.stage("sponsored"):
                    full = _boost_sponsored(full, sponsored_map())

            start = (page - 1) * page_size
            results = full[start : start + page_size]
            sponsored_items = [r for r in results if r.get("is_sponsored")]

            recommendations = {"cross_sell": [], "upsell": []}
            if recommend and page == 1 and results:
                with timings.stage("recommend"):
                    recommendations = build_recommendations(query, results, page=page)

            if not errors:
                page_cache.put(
                    cache_key,
                    {
                        "full": full,
                        "intents": intents,
                        "rerank_succeeded": rerank_actually_ran,
                        "no_match": False,
                        "recommendations": recommendations,
                    },
                )

        cached_now = page_cache.get(cache_key)
        total_results = len(cached_now["full"]) if cached_now else len(full)
        total_pages = max(1, -(-total_results // page_size))

        return {
            "user_query": query,
            "interpreted_as": intents,
            "translator_mode": translator_mode or TRANSLATOR_MODE,
            "rerank_requested": rerank,
            "rerank_succeeded": rerank_actually_ran,
            "no_match": no_match,
            "results": results,
            "sponsored": sponsored_items,
            "recommendations": recommendations,
            "errors": errors,
            "page": page,
            "page_size": page_size,
            "total_results": total_results,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "latency_ms": timings.to_dict(),
        }

    except Exception as e:
        logger.exception("Search failed for query=%r", query)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/product")
def product(
    catalog_index: int = Query(..., ge=0, description="Catalog row id from a search result."),
    query: Optional[str] = Query(None, description="Original search query, for recommendation context."),
    recommend: bool = Query(
        RECOMMEND_ENABLED, description="Include this product's cross-sell/upsell."
    ),
):
    """Product detail view: one product plus recommendations anchored on it."""
    p = get_product(catalog_index)
    if p is None:
        raise HTTPException(status_code=404, detail="Product not found.")

    recommendations = {"cross_sell": [], "upsell": []}
    if recommend:
        recommendations = build_recommendations(
            query or p.get("Product_title", ""), [p], page=1
        )

    return {"product": p, "recommendations": recommendations}


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
