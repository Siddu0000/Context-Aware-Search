"""FastAPI service.

End-to-end pipeline:
    /search?query=...        translate -> retrieve -> (optional) rerank
    /feedback                record a thumbs-up / thumbs-down
    /healthz                 readiness probe

Every successful /search response carries per-stage latency.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import FINAL_TOP_K, RERANK_ENABLED, RETRIEVAL_TOP_K
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
    """Warm up the index BEFORE accepting traffic. First user pays nothing extra."""
    logger.info("Warming up index ...")
    load_index()
    logger.info("Service ready.")
    yield


app = FastAPI(title="Context-Aware Agentic Search", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, max_length=300),
    top_k: int = Query(FINAL_TOP_K, ge=1, le=50),
    rerank: bool = Query(RERANK_ENABLED, description="Toggle the LLM rerank stage."),
):
    """Translate -> retrieve -> rerank pipeline."""
    timings = StageTimings()
    try:
        with timings.stage("translate"):
            intents = translate_query(query)

        with timings.stage("retrieve"):
            # Pull a wider candidate pool than the final K so the reranker has
            # something to work with. RETRIEVAL_TOP_K is per intent.
            candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)

        if rerank and candidates:
            with timings.stage("rerank"):
                results = llm_rerank(query, candidates, top_k=top_k)
        else:
            # No rerank — just trim to top_k and tag the reason field for UI parity.
            results = []
            for p in candidates[:top_k]:
                p = dict(p)
                p["rerank_score"] = None
                p["reason"] = "(rerank disabled)"
                results.append(p)

        return {
            "user_query": query,
            "interpreted_as": intents,
            "rerank_enabled": rerank and bool(candidates),
            "results": results,
            "latency_ms": timings.to_dict(),
        }

    except Exception as e:
        logger.exception("Search failed for query=%r", query)
        raise HTTPException(status_code=500, detail=str(e))


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
