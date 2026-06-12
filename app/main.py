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
    RERANK_ENABLED,
    RETRIEVAL_TOP_K,
    TRANSLATOR_MODE,
)
from app.feedback import record_feedback
from app.metrics import StageTimings
from app.reranker import rerank as llm_rerank
from app.search import load_index, search_products
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


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, max_length=300),
    top_k: int = Query(FINAL_TOP_K, ge=1, le=50),
    rerank: bool = Query(RERANK_ENABLED, description="Toggle the LLM rerank stage."),
    translator_mode: Optional[str] = Query(
        None,
        description="Override TRANSLATOR_MODE for this request: query_expansion | hyde | hybrid",
    ),
):
    """Translate -> retrieve -> rerank pipeline. Every call hits the real LLM."""
    timings = StageTimings()
    try:
        with timings.stage("translate"):
            intents = translate_query(query, mode=translator_mode)

        with timings.stage("retrieve"):
            candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)

        rerank_actually_ran = False
        if rerank and candidates:
            with timings.stage("rerank"):
                results = llm_rerank(query, candidates, top_k=top_k)
            rerank_actually_ran = any(
                r.get("rerank_score") is not None for r in results
            )
        else:
            results = []
            for p in candidates[:top_k]:
                p = dict(p)
                p["rerank_score"] = None
                p["reason"] = "(rerank disabled)"
                results.append(p)

        return {
            "user_query": query,
            "interpreted_as": intents,
            "translator_mode": translator_mode or TRANSLATOR_MODE,
            "rerank_requested": rerank,
            "rerank_succeeded": rerank_actually_ran,
            "results": results,
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
