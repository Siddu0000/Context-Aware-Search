"""FastAPI service exposing the search, chat, product and feedback endpoints."""

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
from app.hybrid_router import route as hybrid_route
from app.keyword_search import KeywordSearchEngine
from app.metrics import StageTimings
from app.recommendations import recommend as build_recommendations
from app.reranker import rerank as llm_rerank
from app.search import best_score, get_dataframe, get_product, load_index, search_products
from app.sponsored import sponsored_map
from app.assistant import interpret
from app.translator import query_specifies_gender, understand_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


_keyword_engine: Optional[KeywordSearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Warming up: provider=%s, translator_mode=%s",
        LLM_PROVIDER,
        TRANSLATOR_MODE,
    )
    load_index()
    # Indexed on the same catalog so both search paths cover the same items
    global _keyword_engine
    _keyword_engine = KeywordSearchEngine()
    _keyword_engine.index(get_dataframe())
    logger.info("Keyword engine indexed; service ready.")
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
    """Reranked pool first, then remaining candidates as an embedding-order tail."""
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
    """Round-robin by source ingredient so each appears once before any appears twice."""
    groups = {}          # source_intent -> items, in current (reranked) order
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


def _recipe_slots_with_alternatives(full, max_options: int = 3):
    """One primary per component, each carrying the next same-component items as
    `alternatives` (brand A/B/C for the carousel), capped at max_options total."""
    groups = {}          # source_intent -> [items in rank order]
    order = []           # components in first-seen (rank) order
    for item in full:
        key = item.get("source_intent") or item.get("Product_title")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)

    slots = []
    for key in order:
        members = groups[key]
        primary = dict(members[0])
        primary["alternatives"] = [dict(m) for m in members[1:max_options]]
        slots.append(primary)
    return slots


def _product_gender(product: dict) -> str:
    """Infer gender from category, then title: 'women', 'men' or 'neutral'."""
    cat = (product.get("categ_lvl2_name") or "").lower()
    title = (product.get("Product_title") or "").lower()
    for text in (cat, title):
        if "women" in text or "ladies" in text or "girl" in text:
            return "women"
        if "men" in text or "boy" in text:   # 'women' already returned above
            return "men"
    return "neutral"


def _balance_by_gender(full):
    """Round-robin women/men/neutral so one gender can't crowd out the page."""
    # Can only reorder what retrieval put in the pool, never manufacture items
    women, men, neutral = [], [], []
    for item in full:
        g = _product_gender(item)
        (women if g == "women" else men if g == "men" else neutral).append(item)

    if not (women and men):
        return full   # nothing to balance (single gender or non-apparel)

    streams = [women, men, neutral]
    idx = [0, 0, 0]
    out = []
    while any(idx[i] < len(streams[i]) for i in range(3)):
        for i in range(3):
            if idx[i] < len(streams[i]):
                out.append(streams[i][idx[i]])
                idx[i] += 1
    return out


def _promote_sponsored_options(slots, smap):
    """Promote a sponsored option to primary WITHIN its slot; slot order unchanged."""
    if not smap:
        return slots
    out = []
    for slot in slots:
        # flat _boost_sponsored can't see a sponsored item nested in `alternatives`
        alternatives = slot.get("alternatives") or []
        options = [{k: v for k, v in slot.items() if k != "alternatives"}] + [
            dict(a) for a in alternatives
        ]
        sponsored_idxs = []
        for i, opt in enumerate(options):
            asin = opt.get("parent_asin")
            asin = str(asin) if asin is not None else None
            if asin and asin in smap:
                opt["is_sponsored"] = True
                opt["sponsor"] = smap[asin]["sponsor"]
                sponsored_idxs.append((i, smap[asin].get("bid", 0.0)))
        if sponsored_idxs and sponsored_idxs[0][0] != 0:
            best_i = max(sponsored_idxs, key=lambda t: t[1])[0]
            options.insert(0, options.pop(best_i))
        new_primary = dict(options[0])
        new_primary["alternatives"] = options[1:]
        out.append(new_primary)
    return out


def _balance_candidate_pool(candidates):
    """Gender-interleave the retrieval candidates before the reranker."""
    # Catalog skews ~4.5:1 women's; else the reranker's window sees no men's items
    women, men, neutral = [], [], []
    for item in candidates:
        g = _product_gender(item)
        (women if g == "women" else men if g == "men" else neutral).append(item)
    if not (women and men):
        return candidates
    streams = [women, men, neutral]
    idx = [0, 0, 0]
    out = []
    while any(idx[i] < len(streams[i]) for i in range(3)):
        for i in range(3):
            if idx[i] < len(streams[i]):
                out.append(streams[i][idx[i]])
                idx[i] += 1
    return out


def _boost_sponsored(full, smap):
    """Bid-order sponsored items that made the RERANKED pool to the top."""
    # rerank_score gates it: sponsored items outside the pool aren't relevant enough
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
    # _bid is sort-internal only — never expose sponsors' bid amounts to clients
    for item in boosted:
        item.pop("_bid", None)
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
    """Translate -> retrieve -> rerank -> paginate, plus sponsored and cross-sell."""
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

    # `recommend` must be in the key or /unified_search's recommend=False poisons it
    cache_key = page_cache.make_key(
        query,
        (page_size, pool_k, rerank, recommend, sponsored, translator_mode or TRANSLATOR_MODE),
    )

    try:
        # Page 1 caches the ranked pool; page 2+ reuse it, so no LLM call is made
        cached = page_cache.get(cache_key)
        if cached is not None:
            full = cached["full"]
            intents = cached["intents"]
            constraints = cached.get("constraints", [])
            bundle_type = cached.get("bundle_type")
            is_recipe = bundle_type == "recipe"
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
                # ONE LLM call: intents + bundle_type + typed constraints
                understanding = understand_query(
                    query, mode=translator_mode, errors=errors
                )
                intents = understanding["intents"]
                bundle_type = understanding.get("bundle_type")
                is_recipe = bundle_type == "recipe"
                constraints = understanding["constraints"]

            gender_stated = (
                any(c.get("type") == "gender" for c in constraints)
                or query_specifies_gender(query)  # regex belt-and-braces
            )

            with timings.stage("retrieve"):
                if bundle_type:
                    # Cap per intent so one bundle component can't crowd out the others
                    candidates = search_products(
                        intents,
                        top_k=RETRIEVAL_TOP_K,
                        per_intent_quota=GROCERY_PER_INTENT_K,
                    )
                elif not gender_stated:
                    # 2x deep so the skewed catalog's men's items reach the rerank window
                    candidates = search_products(intents, top_k=RETRIEVAL_TOP_K * 2)
                    candidates = _balance_candidate_pool(candidates)
                else:
                    candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)

            no_match = bool(candidates) and best_score(candidates) < MIN_RESULT_RELEVANCE
            if not candidates or no_match:
                empty_payload = {
                    "user_query": query,
                    "interpreted_as": intents,
                    "constraints": constraints,
                    "is_recipe": is_recipe,
            "bundle_type": bundle_type,
                    "bundle_type": bundle_type,
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
                            "constraints": constraints,
                            "is_recipe": is_recipe,
            "bundle_type": bundle_type,
                        "bundle_type": bundle_type,
                            "bundle_type": bundle_type,
                    "bundle_type": bundle_type,
                            "rerank_succeeded": False,
                            "no_match": True,
                            "recommendations": {"cross_sell": [], "upsell": []},
                        },
                    )
                return empty_payload

            rerank_actually_ran = False
            if rerank and candidates:
                with timings.stage("rerank"):
                    ranked_pool = llm_rerank(
                        query, candidates, top_k=pool_k, errors=errors,
                        constraints=constraints,
                    )
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

            if bundle_type:
                # One card per component (ingredient / garment slot / device) + options
                full = _recipe_slots_with_alternatives(full, max_options=3)
                if sponsored:
                    # Nested-aware: _boost_sponsored can't see slot `alternatives`
                    with timings.stage("sponsored"):
                        full = _promote_sponsored_options(full, sponsored_map())
            else:
                if not gender_stated:
                    # Re-balance the DISPLAY order after the reranker re-scored the pool
                    full = _balance_by_gender(full)
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
                        "constraints": constraints,
                        "is_recipe": is_recipe,
            "bundle_type": bundle_type,
                        "bundle_type": bundle_type,
                    "bundle_type": bundle_type,
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
            "constraints": constraints,
            # From the backend only: source_intent is on every result, not just recipes
            "is_recipe": is_recipe,
            "bundle_type": bundle_type,
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


def _boost_sponsored_keyword_rows(results, smap):
    """Label + bid-front sponsored items within a KEYWORD-ONLY result list."""
    # BM25 rows have no rerank_score, so keyword top-k membership IS the relevance gate
    if not smap:
        return results
    boosted, rest = [], []
    for item in results:
        asin = item.get("parent_asin")
        asin = str(asin) if asin is not None else None
        if asin and asin in smap:
            item = dict(item)
            item["is_sponsored"] = True
            item["sponsor"] = smap[asin]["sponsor"]
            boosted.append((smap[asin].get("bid", 0.0), item))
        else:
            rest.append(item)
    if not boosted:
        return results
    boosted.sort(key=lambda t: t[0], reverse=True)
    return [it for _, it in boosted] + rest


@app.get("/keyword_search")
def keyword_search(
    query: str = Query(..., min_length=1, max_length=300),
    top_k: int = Query(FINAL_TOP_K, ge=1, le=50),
    sponsored: bool = Query(SPONSORED_ENABLED),
):
    """Pure lexical BM25 search: never falls back to CAS, zero LLM calls."""
    if _keyword_engine is None:
        raise HTTPException(status_code=503, detail="Keyword engine not ready.")
    results, num_matched, top_relevance, top_coverage = _keyword_engine.search(
        query, k=top_k
    )
    if sponsored:
        results = _boost_sponsored_keyword_rows(results, sponsored_map())
    return {
        "user_query": query,
        "results": results,
        "no_match": num_matched == 0,
        "num_matched": num_matched,
        "top_relevance": round(top_relevance, 4),
        "top_coverage": round(top_coverage, 4),
    }


@app.get("/unified_search")
def unified_search(
    query: str = Query(..., min_length=1, max_length=300),
    top_k: int = Query(FINAL_TOP_K, ge=1, le=50),
    min_coverage: float = Query(
        0.75, ge=0.0, le=1.0,
        description="Keyword-hit threshold: the top result's title+category "
        "must cover this fraction of query terms, else fall back to "
        "context-aware search. 0.75 tolerates keyword-stuffed titles.",
    ),
    rerank: bool = Query(
        RERANK_ENABLED, description="Rerank toggle for the CAS fallback path."
    ),
    sponsored: bool = Query(
        SPONSORED_ENABLED, description="Sponsored toggle for the CAS fallback path."
    ),
):
    """Single entry point: keyword-first, context-aware fallback on a lexical miss."""
    if _keyword_engine is None:
        raise HTTPException(status_code=503, detail="Keyword engine not ready.")

    # Full CAS payload, so the intent path can surface intents / errors / no_match
    cas_context: dict = {}

    def _cas_intent(q: str, k: int):
        # search() is an endpoint fn: pass EVERY param or it keeps its Query() default
        payload = search(
            query=q,
            top_k=k,
            page=1,
            rerank=rerank,
            recommend=False,
            sponsored=sponsored,
            translator_mode=None,
        )
        cas_context.update(payload)
        return payload.get("results", [])

    try:
        result = hybrid_route(
            query,
            _keyword_engine,
            _cas_intent,
            k=top_k,
            min_coverage=min_coverage,
        )
        if result.get("path") == "intent" and cas_context:
            # Pass CAS context through so the UI reads the same as in non-hybrid mode
            result["interpreted_as"] = cas_context.get("interpreted_as", [])
            result["constraints"] = cas_context.get("constraints", [])
            result["is_recipe"] = cas_context.get("is_recipe", False)
            result["bundle_type"] = cas_context.get("bundle_type")
            result["errors"] = cas_context.get("errors", [])
            result["no_match"] = cas_context.get("no_match", not result["results"])
            result["message"] = cas_context.get("message", "")
            result["rerank_requested"] = cas_context.get("rerank_requested")
            result["rerank_succeeded"] = cas_context.get("rerank_succeeded")
            result["latency_ms"] = cas_context.get("latency_ms", {})
        elif result.get("path") == "keyword" and sponsored:
            result["results"] = _boost_sponsored_keyword_rows(
                result["results"], sponsored_map()
            )
        return result
    except HTTPException:
        # search() already logged and shaped it; re-wrapping nested "500: <msg>"
        raise
    except Exception as e:
        logger.exception("Unified search failed for query=%r", query)
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


class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., max_length=2000)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list)
    top_k: int = Field(6, ge=1, le=20)
    # Refinement context echoed back by the client each turn: the server is stateless
    last_search_query: str = Field("", max_length=500)
    exclusions: list[str] = Field(default_factory=list)


def _matches_exclusion(product: dict, terms: list[str]) -> bool:
    """True if the product matches any excluded term (title/category/intent)."""
    # Word boundary + naive plural fold: plain substring made "pants" kill "pantry"
    import re as _re

    hay = " ".join(
        str(product.get(f) or "")
        for f in ("Product_title", "categ_lvl2_name", "source_intent")
    ).lower()
    for t in terms:
        t = t.lower().strip()
        if not t:
            continue
        variants = {t}
        if t.endswith("s"):
            variants.add(t[:-1])
        else:
            variants.add(t + "s")
        for v in variants:
            if _re.search(rf"\b{_re.escape(v)}\b", hay):
                return True
    return False


@app.post("/chat")
def chat(payload: ChatIn):
    """On-site helper bot: one LLM call resolves the turn into reply/search/refine."""
    errors: list = []
    decision = interpret(
        payload.message,
        [t.model_dump() for t in payload.history],
        last_search_query=payload.last_search_query,
        errors=errors,
    )

    if decision["action"] == "reply":
        return {
            "reply": decision["reply"],
            "action": "reply",
            "new_topic": False,          # a chat reply never resets the page
            "search_query": payload.last_search_query,
            "exclusions": payload.exclusions,
            "results": [],
            "interpreted_as": [],
            "errors": errors,
        }

    # search() is an endpoint fn: pass EVERY param or it keeps its Query() default
    def _run_search(q: str, page: int = 1):
        return search(
            query=q,
            top_k=payload.top_k,
            page=page,
            rerank=RERANK_ENABLED,
            recommend=False,
            sponsored=SPONSORED_ENABLED,
            translator_mode=None,
        )

    if decision["action"] == "refine":
        all_excl = list(dict.fromkeys(
            [e.lower() for e in payload.exclusions] + decision["exclude_terms"]
        ))
        # Re-page the CACHED pool: re-retrieving here dropped unrelated items
        kept: list = []
        first_page = None
        for pg in range(1, 6):  # cap: 5 pages of the pool is plenty to backfill
            page_out = _run_search(payload.last_search_query, page=pg)
            if first_page is None:
                first_page = page_out
            kept.extend(
                r for r in page_out.get("results", [])
                if not _matches_exclusion(r, all_excl)
            )
            if len(kept) >= payload.top_k or not page_out.get("has_next"):
                break
        results = kept[: payload.top_k]
        reply = decision["reply"]
        if not results:
            reply = (
                "Removing that doesn't leave anything else from this search — "
                "want to try describing what you're after a bit differently?"
            )
        return {
            "reply": reply,
            "action": "refine",
            "new_topic": False,          # a refinement continues the topic
            # The base query stays on screen — future refinements stack on it
            "search_query": payload.last_search_query,
            "exclusions": all_excl,
            "results": results,
            "interpreted_as": (first_page or {}).get("interpreted_as", []),
            "constraints": (first_page or {}).get("constraints", []),
            "is_recipe": (first_page or {}).get("is_recipe", False),
            "bundle_type": (first_page or {}).get("bundle_type"),
            "no_match": not results,
            "errors": errors + list((first_page or {}).get("errors", [])),
        }

    # action == "search": fresh pipeline run, page replaced, exclusions reset
    payload_out = _run_search(decision["search_query"])
    results = payload_out.get("results", [])
    reply = decision["reply"]
    if payload_out.get("no_match") or not results:
        reply = (
            "I couldn't find anything matching that in our catalogue. "
            "Try describing it differently, or tell me more about what it's for."
        )
    return {
        "reply": reply,
        "action": "search",
        # True => unrelated goal: the client starts a FRESH page, no appending
        "new_topic": decision["new_topic"],
        "search_query": decision["search_query"],
        "exclusions": [],
        "results": results,
        "interpreted_as": payload_out.get("interpreted_as", []),
        "constraints": payload_out.get("constraints", []),
        "is_recipe": payload_out.get("is_recipe", False),
        "bundle_type": payload_out.get("bundle_type"),
        "no_match": payload_out.get("no_match", False),
        "errors": errors + list(payload_out.get("errors", [])),
    }


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
