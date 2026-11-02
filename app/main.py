"""FastAPI service exposing the search, chat, product and feedback endpoints."""

import logging
import math
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from itertools import groupby
from threading import Event, Lock
from typing import List, NamedTuple, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.params import Param
from pydantic import BaseModel, Field

import app.config as cfg
from app import device_role, llm_memo, page_cache, query_guard, tracking
from app import product_type as ptype
from app import attributes as attrs
# Knobs imported by value are fixed at import. The ones the type-aware ordering reads
# (TYPE_AWARE_RANKING, SPONSORED_MIN_RERANK_SCORE, RERANK_INPUT_K, REJECTS_AFTER_TAIL,
# VARIANT_COLLAPSE) go through `cfg.` at call time, so an in-process A/B
# (eval/eval_page_types.py --mode both, the battery replays) is real
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
from app.llm_client import memo_outcome, track_usage
from app.metrics import StageTimings
from app.recommendations import recommend as build_recommendations
from app.reranker import ROLE_FIELD_KEY
from app.reranker import rerank as llm_rerank
from app.scoring import bayesian_rating
from app.search import (
    best_score, get_dataframe, get_product, load_index, search_products, style_key, title_key,
)
from app.sponsored import sponsored_map
from app.assistant import interpret
from app.translator import query_specifies_gender, understand_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


_keyword_engine: Optional[KeywordSearchEngine] = None
# What /healthz reports: the LLM memo store on every backend, plus the Databricks
# dependencies on the databricks backend
_startup_checks: Optional[dict] = None
# The non-secret knobs this process runs with (effective_config), fixed at start-up
_effective_config: Optional[dict] = None


def _check_databricks_dependencies() -> dict:
    """One cheap call to each thing a search needs, as the identity the app runs as.

    In a Databricks App that is the app's service principal, and a missing grant
    otherwise shows up only as 500s (index, embedding) or as silent raw-query
    fallbacks (LLM). Never raises; costs ~2 embedding tokens and one tiny LLM call.
    """
    from app import vector_search
    from app.llm_client import generate_json, get_llm_client

    def llm_probe():
        # uvicorn binds its port only after start-up: a stalled endpoint must not hold it
        # for the openai defaults (600 s read timeout x 3 tries), so 20 s and no retries
        backend = get_llm_client()
        client = getattr(backend, "client", None)
        if hasattr(client, "with_options"):
            backend.client = client.with_options(timeout=20, max_retries=0)
        try:
            # memo=False: a replayed reply would report OK over a revoked CAN QUERY grant
            generate_json('Reply with the json object {"ok": true}', temperature=0.0,
                          memo=False)
        finally:
            if client is not None:
                backend.client = client

    checks = {"index": vector_search.index_health(len(get_dataframe()))}
    for name, probe in (
        ("embedding", lambda: vector_search.embed_texts(["health check"])),
        ("llm", llm_probe),
    ):
        try:
            probe()
            checks[name] = {"ok": True}
        except Exception as e:  # noqa: BLE001 -- reported in the log and /healthz
            checks[name] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:200]}"}
    return checks


def effective_config() -> dict:
    """The knobs this process runs with, none of them secret (no key, token or query).

    Logged once at start-up and served on /healthz so a laptop and the Databricks App can
    be compared line by line: an .env override is otherwise invisible (the local
    MIN_RESULT_RELEVANCE is 0.28, the App's default 0.5). Read as cfg.X: what the request
    path reads now."""
    from app import llm_client
    from app import search as search_module

    provider, model, base_url = _llm_identity()
    backend = (cfg.SEARCH_BACKEND or "local").lower()
    return {
        "llm_provider": provider,
        "llm_model": model,
        "llm_base_url": base_url if provider == "openai" else None,
        "deterministic": cfg.DETERMINISTIC,
        "seed_sent": llm_client.seed_sent(),
        "search_backend": backend,
        "embedding_model": (cfg.VS_EMBEDDING_MODEL if backend == "databricks"
                            else cfg.EMBEDDING_MODEL),
        "translator_mode": cfg.TRANSLATOR_MODE,
        "query_guard": cfg.QUERY_GUARD_ENABLED,
        "max_intents": cfg.MAX_INTENTS,
        "type_aware_ranking": cfg.TYPE_AWARE_RANKING,
        "outfit_bundles": cfg.OUTFIT_BUNDLES_ENABLED,
        "rejects_after_tail": cfg.REJECTS_AFTER_TAIL,
        "device_role_ranking": cfg.DEVICE_ROLE_RANKING,
        "variant_collapse": cfg.VARIANT_COLLAPSE,
        "weak_match_min_strong": cfg.WEAK_MATCH_MIN_STRONG,
        "rerank_enabled": cfg.RERANK_ENABLED,
        "rerank_input_k": cfg.RERANK_INPUT_K,
        "rerank_pool_k": cfg.RERANK_POOL_K,
        "retrieval_top_k": cfg.RETRIEVAL_TOP_K,
        "final_top_k": cfg.FINAL_TOP_K,
        "min_result_relevance": cfg.MIN_RESULT_RELEVANCE,
        "rating_boost_weight": cfg.RATING_BOOST_WEIGHT,
        "recommend_enabled": cfg.RECOMMEND_ENABLED,
        "recommend_max": cfg.RECOMMEND_MAX,
        "recommend_use_llm": cfg.RECOMMEND_USE_LLM,
        "sponsored_enabled": cfg.SPONSORED_ENABLED,
        "sponsored_min_rerank_score": cfg.SPONSORED_MIN_RERANK_SCORE,
        "page_cache_size": cfg.PAGE_CACHE_SIZE,
        "llm_memo": llm_memo.backend_kind(),
        "llm_memo_dir": llm_memo.location() if llm_memo.enabled() else None,
        "llm_memo_max_entries": cfg.LLM_MEMO_MAX_ENTRIES,
        "llm_memo_max_age_days": llm_memo.max_age_days(),
        "row_text_case": search_module.ROW_TEXT_CASE,
        # Content hash of the loaded catalog ("" before load_index)
        "catalog_fp": search_module.catalog_fingerprint(),
        "prompts_sha": llm_memo.prompts_digest(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Warming up: provider=%s, translator_mode=%s",
        LLM_PROVIDER,
        TRANSLATOR_MODE,
    )
    tracking.enable_autolog()   # no-op unless MLFLOW_ENABLED and mlflow installed
    load_index()
    global _keyword_engine, _startup_checks, _effective_config
    checks: dict = {}
    if cfg.SEARCH_BACKEND == "databricks":
        # Checked once at start-up, loudly: a deleted (paused) index, a mismatched
        # catalog or a missing grant otherwise surfaces only mid-demo
        checks.update(_check_databricks_dependencies())
    # Every backend: an unwritable .cache or a missing WRITE VOLUME grant on the App would
    # otherwise lose determinism silently (each search simply asks the LLM again)
    checks["memo"] = llm_memo.startup_check()
    _startup_checks = checks
    for name, check in checks.items():
        if check["ok"]:
            note = check.get("detail") if name == "memo" else ""
            logger.info("Start-up check %s: OK%s", name, f" ({note})" if note else "")
        else:
            logger.error("Start-up check %s FAILED: %s", name, check["detail"])
    _effective_config = effective_config()
    logger.info("Effective config: %s",
                " ".join(f"{k}={v}" for k, v in _effective_config.items()))
    # Indexed on the same catalog so both search paths cover the same items
    t0 = time.perf_counter()
    _keyword_engine = KeywordSearchEngine()
    _keyword_engine.index(get_dataframe())
    logger.info("Keyword engine indexed in %.0fs; service ready.", time.perf_counter() - t0)
    yield


app = FastAPI(title="Context-Aware Agentic Search", lifespan=lifespan)


@app.middleware("http")
async def _track_request(request, call_next):
    """Per-request REAL token usage (headers + log) and one MLflow trace per request."""
    t0 = time.perf_counter()
    with track_usage() as usage, tracking.request_span(
        f"{request.method} {request.url.path}"
    ) as span:
        response = await call_next(request)
        tracking.set_span_attributes(span, {f"llm.{k}": v for k, v in usage.items()})
    response.headers["X-LLM-Calls"] = str(usage["calls"])
    response.headers["X-LLM-Prompt-Tokens"] = str(usage["prompt_tokens"])
    response.headers["X-LLM-Completion-Tokens"] = str(usage["completion_tokens"])
    # Replies replayed from the LLM memo (0 provider calls): "calls 0, memo hits 2" is a
    # repeated search served exactly as before
    response.headers["X-LLM-Memo-Hits"] = str(usage["memo_hits"])
    if usage["calls"] or usage["memo_hits"]:
        # Path and counts only -- never the query text (safety.md: no PII in logs)
        logger.info(
            "%s %s llm_calls=%d memo_hits=%d prompt_tok=%d completion_tok=%d "
            "reasoning_tok=%d %.0fms",
            request.method, request.url.path, usage["calls"], usage["memo_hits"],
            usage["prompt_tokens"], usage["completion_tokens"], usage["reasoning_tokens"],
            (time.perf_counter() - t0) * 1000,
        )
    return response


@app.get("/healthz")
def healthz():
    out = {"status": "ok"}
    if _startup_checks is not None:
        out["checks"] = _startup_checks
    if _effective_config is not None:
        out["config"] = _effective_config
    # This process's memo hits / misses / commits since start (counts only)
    out["memo"] = llm_memo.counters()
    return out


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
    pool_titles = {title_key(p.get("Product_title")) for p in ranked_pool}
    tail = []
    for c in candidates:
        if title_key(c.get("Product_title")) in pool_titles:
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


def _group_by_slot(full):
    """({component: [items in rank order]}, components in first-seen (rank) order)."""
    groups = {}
    order = []
    for item in full:
        key = item.get("source_intent") or item.get("Product_title")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    return groups, order


def _slots_from_groups(groups, order, max_options: int):
    slots = []
    for key in order:
        members = groups[key]
        primary = dict(members[0])
        primary["alternatives"] = [dict(m) for m in members[1:max_options]]
        slots.append(primary)
    return slots


def _recipe_slots_with_alternatives(full, max_options: int = 3):
    """One primary per component, each carrying the next same-component items as
    `alternatives` (brand A/B/C for the carousel), capped at max_options total."""
    groups, order = _group_by_slot(full)
    return _slots_from_groups(groups, order, max_options)


def _row_type(row: dict) -> Optional[str]:
    """The row's product_type (app/product_type.py), or None for a non-Fashion row.

    Rows from an un-annotated catalog or a mock carry no type and read None. NaN (a
    CSV round-trip) is not a type either."""
    value = row.get(ptype.TYPE_COLUMN)
    return value if isinstance(value, str) and value else None


def _is_fashion_row(row: dict) -> bool:
    """Annotated rows say so by carrying a type; an un-annotated one is judged by its
    vertical, case-insensitively (the local backend served 'fashion' until 2026-10-15)."""
    return _row_type(row) is not None or ptype.is_fashion(row.get("bsns_vrtcl_name"))


# Whole words: the old substring test read 'replacement', 'ramen', 'yemen' and
# 'engagement' as men's and 'cowgirl' as women's
_WOMEN_WORDS_RE = re.compile(r"\b(women|women's|womens|woman|ladies|lady|girls?)\b")
_MEN_WORDS_RE = re.compile(r"\b(men|men's|mens|man|boys?|gentlemen)\b")


def _product_gender(product: dict) -> str:
    """'women', 'men' or 'neutral'.

    The enriched `gender` tag decides when present (unisex and kids are neutral). Empty,
    NaN and non-str values are absent: the local backend fills non-Fashion rows with ''.
    Without it only Fashion rows are gendered, by whole words in the category, then
    the title -- an Electronics 'men's shaver' replacement head is not men's apparel.

    Deliberately NOT the load-time `shopper_gender` (2026-10-19; the rail's gender filter
    reads that): measured offline on the 41-query battery (0 LLM, recorded scores), it
    moved 10 rerank windows (7 of them replay-exact today) for a page-1 tie (strict 0.519
    vs 0.520 imputed) and page-2 strict 0.325 -> 0.321. An owner / Measure item."""
    gender = product.get("gender")
    if isinstance(gender, str) and gender.strip():
        gender = gender.strip().casefold()
        return gender if gender in ("women", "men") else "neutral"
    if not _is_fashion_row(product):
        return "neutral"
    for field in ("categ_lvl2_name", "Product_title"):
        text = product.get(field)
        text = text.casefold() if isinstance(text, str) else ""
        if _WOMEN_WORDS_RE.search(text):
            return "women"
        if _MEN_WORDS_RE.search(text):
            return "men"
    return "neutral"


def _balance_by_gender(full):
    """Round-robin women/men/neutral so one gender can't crowd out the page.
    TYPE_AWARE_RANKING=false only; the type-aware display uses _interleave_gender."""
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
            # Slot membership is the bundle gate; the rerank floor isn't applied (owner call)
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
    """Gender-interleave the retrieval candidates before the reranker.
    TYPE_AWARE_RANKING=false only; the type-aware window is _flat_window."""
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


# ------------------------------------------------------------------ type-aware ordering
# TYPE_AWARE_RANKING (app/config.py). The catalog has no product-type field, and Fashion's
# categ_lvl2_name is only a gender bucket, so "cloths for a beach wedding" put sandals,
# earrings and bags in the rerank window and on page 1 ahead of 9,218 suitable garments.
# app/product_type.py types Fashion titles at load; the functions below order by it.

# Who the shopper is buying for names a gender too ("something for her to wear", "gift
# for my wife"): else the interleave filled half the page with men's tuxedos. Kept here,
# not in translator._GENDER_QUERY_RE, so the translator's inputs and the evals are unchanged
_GENDER_PEOPLE_RE = re.compile(
    r"\b(her|she|hers|wife|girlfriend|mom|mother|daughter|sister|niece|aunt|grandma|"
    r"bride|bridesmaids?|him|he|his|husband|boyfriend|dad|father|son|brother|nephew|"
    r"uncle|grandpa|groom|groomsmen)\b",
    re.IGNORECASE,
)


# A garment only women's wear names states the gender too: "Wedding dresses for a
# destination wedding in rajasthan" interleaved a men's dashiki (rr 55) into the second
# gendered slot, above women's dresses scored 85 and 78. Not "skirt steak", "bikini
# trimmer", a "car bra" or a unisex "dressing gown". Type-aware path only, like the
# pronouns. The singular 'dress' is judged by its neighbours (_dress_is_garment).
_WOMEN_GARMENT_RE = re.compile(
    r"\b(?:dresses|sundress(?:es)?|(?<!dressing )gowns?|sarees?|saris?|lehengas?|blouses?|"
    r"skirts?(?! steaks?\b)|bikinis?(?! (?:trimmers?|wax\w*|lines?)\b)|"
    r"(?<!car )bras?)\b",
    re.IGNORECASE,
)
_BARE_DRESS_RE = re.compile(r"\bdress\b", re.IGNORECASE)
# The singular 'dress' is a VERB after a subject or a 'to' ("how should I dress for an
# interview", "what to dress my toddler in") -- unless an article or possessive makes
# it the garment ("a dress to wear to a wedding", "my dress for prom")...
_DRESS_VERB_BEFORE = frozenset(
    "to i we you they he she should shall can could would will must do does did lets let's"
    .split())
_DRESS_NOUN_BEFORE = frozenset(
    "a an the this that my your her our their one another".split())
# ...or when an object or idiom follows it ("dress up", "dress for success")
_DRESS_VERB_AFTER_RE = re.compile(
    r"(?:up|down|like|my|your|his|our|their|him|them|me|us|\w+sel(?:f|ves)|"
    r"for success|to impress)\b", re.IGNORECASE)
# ...and a MODIFIER before another type noun ("dress shoes", "dress coat", "dress tie",
# "dress-shirt" -- men's or neutral wear) or one of these ("dress code", "dress forms")
_DRESS_COMPOUNDS = frozenset("code codes hanger hangers form forms".split())
_NEXT_WORD_RE = re.compile(r"[ -]([\w']+)")


def _dress_is_garment(text: str, match) -> bool:
    """Whether the singular 'dress' at `match` names the garment itself."""
    before = text[:match.start()].split()
    prev = before[-1].lower() if before else ""
    if prev in _DRESS_NOUN_BEFORE:
        return True
    if prev in _DRESS_VERB_BEFORE:
        return False
    rest = text[match.end():]
    if rest[:1] in (" ", "-") and _DRESS_VERB_AFTER_RE.match(rest[1:]):
        return False
    nxt = _NEXT_WORD_RE.match(rest)
    if nxt is None:
        return True
    word = nxt.group(1).lower()
    # product_type's own word reading, so a new type noun needs no edit here
    return word not in _DRESS_COMPOUNDS and ptype.classify_title(word)[0] == ptype.UNKNOWN


def _names_womens_garment(query: str) -> bool:
    text = " ".join(query.split())
    return bool(_WOMEN_GARMENT_RE.search(text)) or any(
        _dress_is_garment(text, m) for m in _BARE_DRESS_RE.finditer(text))


def _gender_stated(query: str, constraints: list, typed: bool) -> bool:
    """A gender constraint, a gender word, or (type-aware only) a person's gender or a
    women's-only garment. A women's fit word (petite, maternity) is NOT read here: it
    states women's only in a Fashion-majority pool (_women_fit_stated)."""
    return (
        any(c.get("type") == "gender" for c in constraints)
        or query_specifies_gender(query)  # regex belt-and-braces
        or (typed and bool(
            _GENDER_PEOPLE_RE.search(query) or _names_womens_garment(query)
        ))
    )


def _women_fit_stated(query: str, typed: bool, bundle_type) -> bool:
    """A women's fit word on a flat type-aware search (owner call 3, 2026-10-19: "petite
    work pants" read as unstated, so retrieval went 2x deep and 6 of 15 window seats went
    to men's pants, cargo shorts and a pea coat). Only a candidate: _ranked_pool keeps it
    only when the 1x pool is Fashion-majority, like every lexical reading -- "petite mints",
    "pregnancy tea" and "lactation cookies for breastfeeding" are Grocery, and retrieving
    them 1x deep halved their results (144 -> 78 on "petite mints")."""
    return typed and not bundle_type and ptype.implies_women(query)


def _interleave_gender(seq):
    """Alternate women's and men's items over the positions gendered items hold.

    Neutral items KEEP their positions, so shoes, bags, unisex and non-apparel rows
    are never pushed down by the balance (the old 3-stream round-robin moved every
    third slot to them). Each gender keeps its own order; the gender that appears
    first starts; when one runs out the other fills the rest. Unchanged unless both
    genders are present."""
    genders = [_product_gender(item) for item in seq]
    streams = {
        g: [item for item, ig in zip(seq, genders) if ig == g] for g in ("women", "men")
    }
    if not (streams["women"] and streams["men"]):
        return list(seq)
    turn = next(g for g in genders if g != "neutral")
    taken = {"women": 0, "men": 0}
    out = []
    for item, g in zip(seq, genders):
        if g == "neutral":
            out.append(item)
            continue
        other = "men" if turn == "women" else "women"
        pick = turn if taken[turn] < len(streams[turn]) else other
        out.append(streams[pick][taken[pick]])
        taken[pick] += 1
        turn = other
    return out


def _round_robin_by_intent(items, intents):
    """One item per intent per round, intents in the translator's order, each intent's
    items in their current order. Raw cosine is not comparable across intents (a short
    phrase scores higher), so score order let one intent fill the rerank window: the
    recorded "cloths for a beach wedding" window was 15 of 15 wedge sandals."""
    position = {}
    for i, term in enumerate(intents or []):
        position.setdefault(term, i)
    groups, order = {}, []
    for item in items:
        key = item.get("source_intent")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    # Stable: an intent the translator did not list keeps its first-seen place, last
    order.sort(key=lambda k: position.get(k, len(position)))
    out = []
    for depth in range(max((len(g) for g in groups.values()), default=0)):
        out.extend(groups[k][depth] for k in order if depth < len(groups[k]))
    return out


def _type_rank(item: dict, requested) -> int:
    """ptype.type_rank of a row, except that a NON-Fashion row under an active request
    is typed by its title: a mixed pool passes the gate (50-93% Fashion) for "kids
    backpack" or "heated gloves", and type_rank(None) = 2 sank every Electronics school
    backpack and USB heated glove below all Fashion rows, out of the rerank window. Such
    a row ranks 0 when the title rules give it a requested type outright; otherwise 2
    as before. A weak-word reading ("Laptop Cooling Pad 5 Fans" -> accessory) and any
    Grocery row ("Fruit Snack Pouches" -> bag, "pineapple rings") stay 2: food is never
    the garment, bag or ring asked for. product_type.py is unchanged: its title rules
    are only borrowed here."""
    row_type = _row_type(item)
    if not requested or row_type is not None or _is_fashion_row(item):
        return ptype.type_rank(row_type, requested)
    vertical = item.get("bsns_vrtcl_name")
    if isinstance(vertical, str) and vertical.strip().casefold() == "grocery":
        return 2
    title_type, subtype = ptype.classify_title(item.get("Product_title"))
    return 0 if title_type in requested and not subtype.startswith("weak") else 2


def _effective_request(query: str, candidates) -> Optional[frozenset]:
    """The product types the shopper named, kept only in a Fashion-majority pool.

    requested_types() is lexical, and its nouns name Electronics and Grocery products
    too ("tea bags", "ring light", "apple watch", "laptop bag"). There a request would
    rank the odd Fashion tote above every tea bag, tell the reranker every real
    candidate violates the ask and block their ads. So it counts only when at least
    half of the first RERANK_INPUT_K candidates, in RAW embedding order (search_products
    returns them score-sorted, before any reordering), are typed Fashion rows."""
    requested = ptype.requested_types(query)
    if not requested:
        return None
    if not _fashion_majority(candidates):
        # The flag only: the types come from the query text (safety.md)
        logger.info("Type-aware ranking: type request dropped (non_fashion_pool).")
        return None
    return requested


def _fashion_majority(candidates) -> bool:
    """At least half of the first RERANK_INPUT_K candidates, in RAW embedding order, are
    typed Fashion rows: the one gate every lexical reading of the query passes before it
    may act (the type request, the fit / subtype / age tiers, the guest constraint)."""
    window = list(candidates[: cfg.RERANK_INPUT_K])
    fashion = sum(1 for c in window if _row_type(c) is not None)
    return bool(window) and fashion / len(window) >= 0.5


# ------------------------------------------------------------------ fit / subtype / age
# What the shopper named beyond the 6-way type (2026-10-19, owner call 3; synthesis P3).
# The window tiered on product_type alone, so silk scarves and saree blouses competed with
# sarees, leggings with jeans, non-maternity robes with maternity dresses, and kids' shoes
# with adult running shoes. Read by app/product_type.py's lexical readers and gated like
# the type request (_fashion_majority), so "iPhone 14 Plus", "tall kitchen trash bags" and
# "baby formula" cannot act. Inside TYPE_AWARE_RANKING (its REVERT covers it).
class _Ask(NamedTuple):
    fit: Optional[frozenset]        # subset of {"petite", "maternity", "plus", "tall"}
    subtypes: Optional[frozenset]   # fine_subtype keys
    audience: Optional[frozenset]   # subset of {"adult", "kids", "baby"}; None = any age


# Age is a hard constraint for clothing and shoes only (owner call 3): a kids' backpack on
# an adult backpack ask is not demoted. 'unknown' rows are mostly model-named shoes.
_AGE_TYPES = frozenset({"apparel", "footwear", ptype.UNKNOWN})
_AUDIENCES = frozenset({"adult", "kids", "baby"})
# A shopper who names no age is shopping for an adult (the reranker already capped kids'
# rows on adult asks as "the implicit adult demographic": 30 of 555 battery window seats
# were kids' rows, mostly capped at exactly 40)...
_DEFAULT_AUDIENCE = frozenset({"adult"})


def _default_audience(candidates) -> Optional[frozenset]:
    """...unless the raw head itself is mostly children's clothing and shoes (2026-10-19
    review): a children's ask with no age word the reader knows ("first communion dress",
    "christening gown", "school uniform for my 8 yo" -- product_type reads those three
    now) retrieved 14-15 kids' rows in the raw top 15, and the default turned the missing
    age into a hard adult constraint -- 0 of 15 window seats were kids' (adult club
    dresses, bridal gowns, an exam gown), the first at #35-#52. This is the net for the
    next phrasing the reader misses: no age is assumed (None: the reranker reads the
    query as before). Counted over the first
    RERANK_INPUT_K candidates in RAW order, clothing and shoes with an audience only, and
    only a CLEAR children's head counts -- at least twice as many children's rows as
    adult ones and a third of the head: bare "running shoes" retrieves 8 kids' and 7
    adult rows (the catalog's shoe skew), and the age tier exists for it (x02). On the 40
    Fashion battery queries the reader's misses never reach it (x02 5 kids / 10 adult)."""
    head = candidates[: cfg.RERANK_INPUT_K]
    adult = young = 0
    for item in head:
        if _row_type(item) not in _AGE_TYPES:
            continue
        audience = item.get("audience")
        if audience == "adult":
            adult += 1
        elif audience in ("kids", "baby"):
            young += 1
    if young and young >= 2 * adult and 3 * young >= len(head):
        return None
    return _DEFAULT_AUDIENCE


def _effective_ask(query: str, candidates) -> Optional[_Ask]:
    """The fit, subtype and age readings of the query, or None outside a Fashion-majority
    pool (measured on the RAW order, like _effective_request)."""
    if not _fashion_majority(candidates):
        return None
    return _Ask(
        fit=ptype.requested_fit(query) or None,
        subtypes=ptype.requested_subtypes(query) or None,
        audience=ptype.requested_audience(query) or _default_audience(candidates),
    )


# ------------------------------------------------------------------ device asks
# DEVICE_ROLE_RANKING (2026-10-27, the Oct 27 meeting: "gift for a vintage watch collector"
# put watch boxes, winders and cleaning kits on page 1, the watches on page 2). Nothing
# here knew "device vs accessory": a wristwatch and its winder are both Fashion
# `accessory` rows, and an Electronics cleaning kit has no type at all, so the untyped
# window (1 seat per intent, then raw cosine) let whichever intent scored highest -- the
# winder -- fill the window and the unjudged tail. app/device_role.py reads the ask and the
# rows' load-time columns; inside TYPE_AWARE_RANKING (its REVERT covers this too), flat
# pages only. Offline (0 LLM, the real catalog, the recorded 10-15 intents): the window
# went from 0 watches to 12 of 15; the cross-vertical guard: build/NOTES_B1.md.
def _effective_device(query: str, candidates,
                      enabled: Optional[bool] = None) -> Optional[device_role.DeviceAsk]:
    """The device the shopper named, kept only where it can be acted on: the knob is on
    (`enabled`: the value search() read once for its cache key; None reads cfg), the ask
    is for the device itself (not an accessory for it), at least half of the first
    RERANK_INPUT_K candidates in RAW order are about the family (the _fashion_majority
    idiom: "dishwasher tablets" reads as a device, its pool does not), and
    the pool holds at least one core row ("airtag"-like supply gaps: nothing to promote,
    and promoting nothing would only demote the substitutes). A DEVICE of another family
    in the head is no evidence either way and is left out of the share (2026-10-27
    review): "iphone" with device-only intents ("apple iphone unlocked") retrieved 7 iPads,
    iPods and MacBooks among its 15 head rows, so only 7 of 15 named a phone, while a
    lexical collision's head is food, caps or clothing, never devices, so it still
    fails."""
    if not (cfg.DEVICE_ROLE_RANKING if enabled is None else enabled):
        return None
    ask = device_role.requested_device(query)
    if ask is None or ask.accessory_asked:
        return None
    head = list(candidates[: cfg.RERANK_INPUT_K])
    named = [device_role.mentions(c, ask.family) for c in head]
    counted = sum(1 for c, n in zip(head, named) if n or c.get("item_role") != "device")
    if not counted or sum(named) / counted < 0.5:
        # Flags only: the family comes from the query text (safety.md)
        logger.info("Device-role ranking: device ask dropped (minority_pool).")
        return None
    if not any(device_role.is_core(c, ask.family) for c in candidates):
        logger.info("Device-role ranking: device ask dropped (no_core_rows).")
        return None
    return ask


def _device_tier(item: dict, device: Optional[device_role.DeviceAsk]) -> int:
    """The row's role tier under an active device ask (CORE, UNKNOWN, ACCESSORY); 0 for
    every row without one, so every sort key it joins is unchanged."""
    if device is None:
        return 0
    return device_role.role(item, device.family)


def _is_addon(item: dict, device: Optional[device_role.DeviceAsk]) -> bool:
    return device is not None and device_role.role(item, device.family) == device_role.ACCESSORY


def _not_core(item: dict, device: Optional[device_role.DeviceAsk]) -> bool:
    return device is not None and not device_role.is_core(item, device.family)


# A gift ask keeps this many rerank-window seats for its add-on intents (owner call D1):
# storage and display pieces stay JUDGED -- with a constraint-respecting reranker they open
# page 2 behind every real watch -- and the reranker still sees what the translator meant.
# Without it the core tier filled all 15 seats whenever the pool held 15 watches.
_GIFT_RESERVE = 3


def _gift_reserve(out: list, intents, device: device_role.DeviceAsk) -> list:
    """The window's last _GIFT_RESERVE seats go to rows of the intents that do not name
    the device itself ("leather watch winder", "watch display box"), dealt one per intent
    in the translator's order, each intent's storage and display rows before its tools,
    care and parts (device_role.addon_rank), and seated in that order; the core-first
    order fills the seats before them. An intent's core rows are never reserved: they are
    in the core tier already. No add-on intent, no reserve."""
    family = device.family
    terms = [t for t in dict.fromkeys(intents or []) if not device_role.names_device(t, family)]
    pools = []
    for term in terms:
        rows = [r for r in out if r.get("source_intent") == term
                and not device_role.is_core(r, family)]
        pools.append(sorted(rows, key=lambda r: device_role.addon_rank(r, family)))
    reserve, depth = [], 0
    while len(reserve) < _GIFT_RESERVE and any(depth < len(p) for p in pools):
        for rows in pools:
            if depth < len(rows) and len(reserve) < _GIFT_RESERVE:
                reserve.append(rows[depth])
        depth += 1
    if not reserve:
        return out
    reserve.sort(key=lambda r: device_role.addon_rank(r, family))
    taken = {id(r) for r in reserve}
    head = [r for r in out if id(r) not in taken][: max(0, cfg.RERANK_INPUT_K - len(reserve))]
    taken |= {id(r) for r in head}
    return head + reserve + [r for r in out if id(r) not in taken]


def _with_role_fields(candidates: list, device: device_role.DeviceAsk) -> list:
    """The candidates with the reranker's transient ` | role=` field set on COPIES of the
    window rows (reranker.ROLE_FIELD_KEY): "device" on a core row, "accessory" on an
    add-on, nothing on the rest. A wristwatch's line said `type=accessory` next to a
    rubric that caps "an accessory FOR the named product" -- the recorded 10-15 rerank
    scored the only watch in its window 40. Rows past the window and the cached pool are
    never touched."""
    names = {device_role.CORE: "device", device_role.ACCESSORY: "accessory"}
    out = []
    for i, row in enumerate(candidates):
        name = names.get(device_role.role(row, device.family)) if i < cfg.RERANK_INPUT_K else None
        out.append(dict(row, **{ROLE_FIELD_KEY: name}) if name else row)
    return out


def _without_role_field(rows: list) -> list:
    """The reranker's rows without the transient role field (it copies its inputs)."""
    return [{k: v for k, v in r.items() if k != ROLE_FIELD_KEY} if ROLE_FIELD_KEY in r else r
            for r in rows]


def _title_fit(title: str) -> frozenset:
    """The fit words a TITLE carries, read by the same reader as the query ("Women's Plus
    Size Jeans" -> {"plus"}), so one hand-maintained vocabulary serves both sides. Only
    on a fit ask, only over the pool (<= ~180 titles), once per uncached search."""
    return ptype.requested_fit(title) or frozenset()


def _ask_misses(item: dict, ask: Optional[_Ask],
                device: Optional[device_role.DeviceAsk] = None) -> tuple:
    """(wrong age, number of named attributes missed) for a window row; (0, 0) = no
    demotion. Only Fashion rows are judged: an Electronics school backpack typed by its
    title under "kids backpack" has no age or subtype column and keeps its place. Under a
    device ask (2026-10-27 review) a named SUBTYPE is judged on every row: a Fashion
    wristwatch and an Electronics smartwatch or band compete for the same seats, and a
    row with no fine_subtype cannot be the named one ("dress watch for men" reads the
    dresses subtype, so every Fashion watch missed it and an Electronics replacement
    strap, judged on nothing, led the window ahead of 14 watches)."""
    if ask is None:
        return (0, 0)
    if not _is_fashion_row(item):
        return (0, 1 if device is not None and ask.subtypes else 0)
    wrong_age = 0
    if ask.audience and _row_type(item) in _AGE_TYPES:
        audience = item.get("audience")
        if isinstance(audience, str) and audience in _AUDIENCES and audience not in ask.audience:
            wrong_age = 1
    missed = 0
    if ask.fit:
        title = item.get("Product_title")
        if not (isinstance(title, str) and _title_fit(title) & ask.fit):
            missed += 1
    if ask.subtypes and item.get("fine_subtype") not in ask.subtypes:
        missed += 1
    return (wrong_age, missed)


def _partitioned(seq, ask: Optional[_Ask], interleave: bool, off=frozenset(),
                 device: Optional[device_role.DeviceAsk] = None) -> list:
    """seq stably partitioned by (wrong age, from an off-type intent, named fit / subtype
    missed, device role), each part gender-interleaved on its own: one interleave over the
    whole tier would pull a men's non-jeans row, or a model-named shoe from a shoe intent,
    up to fill a men's slot ahead of the women's jeans. The device role (core, unknown,
    accessory; 2026-10-27) comes LAST, so the type tiers, age and fit parts and the
    interleave stay what they were and only the order inside each part moves. No ask, no
    off-type intent and no device ask: one part, the interleave exactly as before."""
    def key(item):
        wrong_age, missed = _ask_misses(item, ask, device)
        return (wrong_age, item.get("source_intent") in off, missed,
                _device_tier(item, device))

    keyed = sorted(((key(item), item) for item in seq), key=lambda pair: pair[0])
    out = []
    for _, part in groupby(keyed, key=lambda pair: pair[0]):
        items = [item for _, item in part]
        out.extend(_interleave_gender(items) if interleave else items)
    return out


def _off_type_intents(intents, requested) -> frozenset:
    """Intents that name another Fashion type than the request outright ("black leather
    dress shoes" under a clothing ask). Under a request their rows keep no round-robin
    seat (synthesis P4): c09's shoe intent spent 5 of 15 seats on belts and a biker jacket
    typed apparel. A weak-word reading is not an outright type."""
    off = set()
    for term in intents or []:
        type_, subtype = ptype.classify_title(term)
        if type_ in ptype.FASHION_TYPES and type_ not in requested and \
                not subtype.startswith("weak"):
            off.add(term)
    return frozenset(off)


# Guaranteed rerank-window seats per intent for an UNTYPED flat query, then raw score
# order. Chosen OFFLINE (2026-10-15, 21 untyped context + cross-vertical queries, qwen3.8
# intents, gte-small; relevant rows in the 15-row window): score order 11.14, 1 seat
# 10.86, 2 seats 10.57, 3 seats 10.05, round-robin 8.81; power bank 12 / 11 / 10 / 8 / 5.
# Round-robin gave a weak paraphrase intent a third of the window (power bank: 9 real
# banks on page 1 fell to 5). 1 seat ties score order (0.28 of 15 rows) and still lets
# every intent reach the reranker (3.00 intents in the window vs 2.29), which is what
# fixed one-intent windows on gift queries ("gift for a coffee lover" had no pour-over
# seat in score order). 0 = plain score order.
_UNTYPED_INTENT_SEATS = 1


def _seat_intents(items, intents, seats: int):
    """Each intent's best `seats` items first (round by round, the translator's intent
    order), then every other item in its current (raw score) order."""
    if seats <= 0:
        return list(items)
    position = {}
    for i, term in enumerate(intents or []):
        position.setdefault(term, i)
    groups, order = {}, []
    for item in items:
        key = item.get("source_intent")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(item)
    order.sort(key=lambda k: position.get(k, len(position)))
    first = [groups[k][depth] for depth in range(seats) for k in order if depth < len(groups[k])]
    taken = {id(item) for item in first}
    return first + [item for item in items if id(item) not in taken]


def _flat_window(candidates, requested, intents, interleave: bool, ask: Optional[_Ask] = None,
                 device: Optional[device_role.DeviceAsk] = None):
    """Pre-rerank order. With a type request: the requested type, then untyped Fashion
    rows ('unknown', mostly shoes named only by model), then everything else, each tier
    round-robining the intents that do not name another type, then the ones that do
    (_off_type_intents). An untyped row within the raw top RERANK_INPUT_K competes in the
    first tier (synthesis P7: a Brooks Glycerin at raw #14 never reached the window of
    "running shoes for flat feet" behind 72 typed sandals and kids' shoes). Without a
    request: _UNTYPED_INTENT_SEATS per intent, then raw score order. Then, with `ask`,
    each tier but the other-types one is partitioned by age and the named fit / subtype
    (_partitioned), and with a `device` ask every tier also by role, last: the device
    itself, then rows that are neither, then accessories for it -- "laptop for college"
    seated 14 sleeves and backpacks and 1 laptop when the sleeve intent's cosine won. A
    gift ask then keeps _GIFT_RESERVE seats for its add-on intents (_gift_reserve). Either
    way, with no gender stated, women's and men's items alternate. The reranker sees only
    the first RERANK_INPUT_K, so this decides what it can score."""
    if not requested:
        out = _seat_intents(candidates, intents, _UNTYPED_INTENT_SEATS)
        out = _partitioned(out, ask, interleave, device=device)
    else:
        # `candidates` arrive in raw score order (search_products), so this is the raw head
        head = {id(item) for item in candidates[: cfg.RERANK_INPUT_K]}
        off = _off_type_intents(intents, requested)
        tiers = {}
        for item in candidates:
            rank = _type_rank(item, requested)
            if rank == 1 and id(item) in head:
                rank = 0
            tiers.setdefault(rank, []).append(item)
        out = []
        for rank in sorted(tiers):
            tier = _round_robin_by_intent(tiers[rank], intents)
            if rank == 2:
                # Other types: already behind everything asked for, so left as they were
                # (the device role only orders the inside of the tier)
                out.extend(_partitioned(tier, None, interleave, device=device))
            else:
                out.extend(_partitioned(tier, ask, interleave, off, device=device))
    if device is not None and device.gift:
        out = _gift_reserve(out, intents, device)
    return out


def _is_score(value) -> bool:
    return isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))


def _band(item) -> int:
    """0 = reranked above the violator ceiling, 1 = reranked at or below it, 2 = not
    reranked (the tail, rerank off, or a rerank fallback)."""
    score = item.get("rerank_score")
    if not _is_score(score):
        return 2
    return 0 if score > cfg.SPONSORED_MIN_RERANK_SCORE else 1


def _band_position(band: int) -> int:
    """Where a band is displayed. REJECTS_AFTER_TAIL (owner call 1, 2026-10-19): judged
    above the floor, then the never-judged tail, then the rejects -- graded on the
    apparel battery, rejected rows averaged 0.32 of 2 and tail rows 1.01. Off: the
    2026-10-14 order, rejects before the tail."""
    if cfg.REJECTS_AFTER_TAIL:
        return {0: 0, 2: 1, 1: 2}[band]
    return band


# Width of the rerank-score tiers the display interleave stays inside (REJECTS_AFTER_TAIL).
# The reranker's score is well calibrated against the graders (41-55: mean label 0.80,
# 71-85: 1.55, 86-100: 1.81), so alternating genders across a 30-point gap threw that
# away: a men's 55 sat above a women's 92
_INTERLEAVE_TIER_POINTS = 15


def _rerank_tier(item) -> int:
    """0 for 41-55, 1 for 56-70, ... above the floor (the band-0 rows only)."""
    return int((item["rerank_score"] - cfg.SPONSORED_MIN_RERANK_SCORE - 1)
               // _INTERLEAVE_TIER_POINTS)


def _flat_display(full, requested, interleave: bool,
                  device: Optional[device_role.DeviceAsk] = None):
    """Display order after the reranker. Stable sort by (mismatched type, band position,
    not the device asked for, type rank): a product of another type never precedes a
    requested or untyped one, even when it scored higher (the reranker gave wedge sandals
    92 on a clothing ask), while an untyped row competes inside its band -- a model-named
    oxford the reranker scored 90 is not buried under unreranked tail rows. Under a device
    ask (2026-10-27) the device itself leads each band and every other row keeps the
    reranker's order: a winder scored 88 no longer precedes a watch scored 70, and the
    unjudged tail shows its watches before its cleaning kits (with no device ask the
    element is False for every row). The gender interleave then runs within each (mismatch,
    band, core) group, so it can never undo the type, band or device order; with
    REJECTS_AFTER_TAIL, inside the top band it runs only within each run of rows in the
    same 15-point rerank tier, so it can no longer undo the reranker's order either."""
    def sort_key(item):
        rank = _type_rank(item, requested)
        return (rank == 2, _band_position(_band(item)), _not_core(item, device), rank)

    keyed = sorted(((sort_key(item), item) for item in full), key=lambda pair: pair[0])
    if not interleave:
        return [item for _, item in keyed]
    tiered = cfg.REJECTS_AFTER_TAIL
    out = []
    for (_, position, _), group in groupby(keyed, key=lambda pair: pair[0][:3]):
        items = [item for _, item in group]
        if tiered and position == _band_position(0):
            for _, run in groupby(items, key=_rerank_tier):
                out.extend(_interleave_gender(list(run)))
        else:
            out.extend(_interleave_gender(items))
    return out


# What the reranker is told a type request means. Deterministic text per family, never
# the shopper's words. The examples sit in parentheses like the prompt's own lists.
_TYPE_NAMES = {
    "apparel": ("clothing", "garments: dresses, tops, bottoms, suits, swimwear, outerwear"),
    "footwear": ("footwear", "shoes, boots, sandals, sneakers, slippers"),
    "jewelry": ("jewellery", "earrings, necklaces, bracelets, rings, anklets"),
    "bag": ("bags", "handbags, totes, clutches, backpacks, wallets"),
    "headwear": ("headwear", "hats, caps, beanies, hair accessories"),
    "accessory": ("accessories", "belts, scarves, sunglasses, watches, gloves"),
}
_TYPE_VIOLATORS = {
    "apparel": "clothing", "footwear": "shoes", "jewelry": "jewellery", "bag": "bags",
    "headwear": "hats", "accessory": "other accessories",
}


def _type_constraint(requested) -> dict:
    """{'type': 'product_type', 'value': ...} for the reranker's constraints block, e.g.
    apparel -> 'clothing only (garments: ...); shoes, jewellery, bags, hats and other
    accessories violate it'."""
    wanted = [t for t in ptype.FASHION_TYPES if t in requested]
    others = [_TYPE_VIOLATORS[t] for t in ptype.FASHION_TYPES if t not in requested]
    names = " or ".join(_TYPE_NAMES[t][0] for t in wanted)
    examples = "; ".join(_TYPE_NAMES[t][1] for t in wanted)
    value = f"{names} only ({examples})"
    if others:
        violators = others[0] if len(others) == 1 else ", ".join(others[:-1]) + " and " + others[-1]
        value += f"; {violators} violate it"
    return {"type": "product_type", "value": value}


# The rubric's guest rule (2026-10-15: cap bridal wear only for a PLAIN guest) fired on
# the wording, not the situation: "cloths for a beach wedding in summer" (the typo) let 4
# bridal gowns score 78-95, and two equivalent bridal gowns got 82 and 40 on m05. Owner
# call 6 (2026-10-19): the same rule, decided in code by product_type.is_wedding_guest_ask
# ('party or gala' kept) and stated to the reranker like the type request. Deterministic
# text, never the shopper's words; the examples sit in parentheses like the rubric's lists.
_GUEST_CONSTRAINT = {
    "type": "occasion",
    "value": ("the shopper is a GUEST at the event, not the bride, groom or wedding party; "
              "bridal wear (bridal gowns, wedding dresses, bridesmaid dresses), swimwear "
              "and cover-ups violate it"),
}
# A guest who ASKS for swimwear ("swimsuit for a beach wedding", "bikini for a wedding
# party"): the full text told the reranker that every row of the requested style -- the
# swim tier the window seats first -- violates a stated constraint, which it caps at 40
# (2026-10-19 review). Bridal wear still violates it; the swimwear asked for satisfies it
# (said outright: the rubric's own GUESTS line names swimwear too).
_GUEST_SWIM_CONSTRAINT = {
    "type": "occasion",
    "value": ("the shopper is a GUEST at the event, not the bride, groom or wedding party; "
              "bridal wear (bridal gowns, wedding dresses, bridesmaid dresses) violates it; "
              "the swimwear and cover-ups asked for satisfy it"),
}


def _guest_constraint(query: str) -> Optional[dict]:
    """The [occasion] constraint for a plain guest's ask (never a bridal one), else None;
    without the swimwear clause when the shopper names swimwear or a cover-up."""
    if not ptype.is_wedding_guest_ask(query):
        return None
    if "swim" in (ptype.requested_subtypes(query) or ()):
        return dict(_GUEST_SWIM_CONSTRAINT)
    return dict(_GUEST_CONSTRAINT)


# A setting is context, not a requirement: the translator turned "summer beach wedding
# outfit dress" into [other: wedding, other: summer, other: beach], which the reranker
# renders as STATED constraints to "enforce ALL" -- rewarding keyword-stuffed titles
# (18 of the 43 apparel titles saying both 'beach' and 'wedding' are bridal gowns).
_SETTING_WORDS = frozenset({
    "beach", "summer", "winter", "spring", "fall", "autumn", "wedding", "party", "gala",
    "humid", "hot", "cold", "warm", "cozy", "cosy", "formal", "casual", "breathable",
    "lightweight", "outdoor", "outdoors", "vacation", "holiday",
    "beaches", "weddings", "parties", "galas", "vacations", "holidays",
})
# Function words, and the generic nouns a setting phrase ends on ("a humid day")
_SETTING_FILLER = frozenset({
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "with", "my",
    "our", "day", "days", "night", "weather", "season", "time", "event", "occasion",
})


def _is_setting_only(constraint: dict) -> bool:
    if str(constraint.get("type", "")).strip().lower() != "other":
        return False
    words = re.findall(r"[a-z]+", str(constraint.get("value", "")).casefold())
    return (
        bool(words)
        and all(w in _SETTING_WORDS or w in _SETTING_FILLER for w in words)
        and any(w in _SETTING_WORDS for w in words)
    )


def _without_setting_constraints(constraints: list) -> list:
    """The translator's constraints minus 'other' ones made only of occasion, season,
    weather, setting or mood words. For the reranker only: the response and the
    cached constraints stay the translator's."""
    return [c for c in constraints if not _is_setting_only(c)]


# Setup slots naming a NON-Fashion product with a Fashion type word: a Fashion row of
# that type is never right there ("ring light" -> wedding rings, "apple watch" -> wrist
# watches, "zip ties" -> neckties, "tea bags" -> handbags). A slot whose Fashion rows can
# be right ("sleeping bag": a Fashion-catalogued one was reranked 92; "laptop backpack")
# is not listed.
_DEVICE_SLOT_RE = re.compile(
    r"\b(?:ring ?lights?|smart ?rings?|smart ?watch(?:es)?|apple ?watch(?:es)?|"
    r"galaxy ?watch(?:es)?|fitbits?|garmins?|watch ?chargers?|(?:cable|zip) ?ties?|"
    r"lens ?caps?|(?:tea|coffee|trash|garbage|sandwich|freezer|storage|ziploc|vacuum) ?bags?)\b"
)


def _bundle_keep(bundle_type: Optional[str]):
    """search_products keep= filter per bundle kind (None = keep everything).

    outfit: Fashion rows only, and a slot that names a Fashion type ("beach wedding
    sandals") keeps rows of that type or untyped ones -- its top hits were barefoot-
    sandal anklets, which filled the 4-row quota and the whole card. setup: other
    verticals always stay; a Fashion row stays when its type is one the slot accepts
    (ptype.slot_accepts: "fleece slippers" footwear, "sun hat" headwear, "beach tote
    bag" bags, "polarized sunglasses" accessories) or it is untyped (a Fashion-catalogued
    backpacking stove, reranked 85, in "camping stove and fuel"), unless the slot names
    a device (_DEVICE_SLOT_RE: a "ring light" slot drops wedding rings). Until
    2026-10-15 only clothing and footwear slots kept Fashion rows, which emptied "beach
    tote bag" (0/30 kept) and "polarized sunglasses" (0/29). recipe: no filter ("tea
    bags" and "pineapple rings" are ingredients). An outfit "accessories" slot accepts
    the whole accessory family (ptype.slot_accepts)."""
    if bundle_type not in ("outfit", "setup"):
        return None
    slot_types: dict = {}

    def slot_of(term):
        if term not in slot_types:
            device = bool(_DEVICE_SLOT_RE.search(" ".join(str(term).casefold().split())))
            slot_types[term] = (device, ptype.slot_accepts(term))
        return slot_types[term]

    def keep(row, term):
        rtype = _row_type(row)
        device, accepts = slot_of(term)
        if bundle_type == "outfit":
            if rtype is None:
                return False
            return not accepts or rtype in accepts or rtype == ptype.UNKNOWN
        if rtype is None:
            return True
        return not device and (rtype in accepts or rtype == ptype.UNKNOWN)

    return keep


def _refill_empty_slots(candidates, intents, **search_kwargs):
    """A setup slot keep= emptied gets that intent's UNFILTERED hits back: a card that
    vanished silently is worse than one the reranker can judge. Merged like
    search_products (score order, first title wins), so nothing already found moves."""
    filled = {c.get("source_intent") for c in candidates}
    missing = [term for term in dict.fromkeys(intents or []) if term not in filled]
    if not missing:
        return candidates
    # The flag and the count only: slot phrases are model text (safety.md)
    logger.info("Setup keep emptied %d slot(s); refilled unfiltered.", len(missing))
    extra = search_products(missing, **search_kwargs)
    seen = {title_key(c.get("Product_title")) for c in candidates}
    merged = list(candidates) + [
        c for c in extra if title_key(c.get("Product_title")) not in seen]
    merged.sort(key=lambda c: c.get("score") if _is_score(c.get("score")) else 0.0,
                reverse=True)
    return merged


# Outfit cards: garments first, then shoes, then the one accessory
_GARMENT_PRIORITY = {
    "apparel": 0, ptype.UNKNOWN: 1, "footwear": 2, "bag": 3, "headwear": 4,
    "accessory": 5, "jewelry": 6,
}


def _slot_rejected(primary: dict) -> bool:
    """The reranker judged the slot's best option a constraint violator (<= ceiling)."""
    score = primary.get("rerank_score")
    return _is_score(score) and score <= cfg.SPONSORED_MIN_RERANK_SCORE


def _typed_bundle_slots(full, bundle_type: str, intents, max_options: int = 3):
    """Bundle cards with type-aware options and slot order.

    Outfit: each slot's options put its own type first, then untyped, then the rest;
    every slot then prefers options of the first garment slot's gender (or neutral),
    since an outfit must not pair a women's top with men's trousers (not filtered:
    a slot with only the other gender still shows). Card order: garment priority,
    then rejected slots last, then the translator's order. Setup: rejected last, then
    the translator's order (it lists the most essential first). Recipe: rejected
    last, otherwise first-seen (rank) order as before."""
    groups, order = _group_by_slot(full)
    position = {}
    for i, term in enumerate(intents or []):
        position.setdefault(term, i)

    def slot_key(key):
        rejected = _slot_rejected(groups[key][0])
        if bundle_type == "outfit":
            priority = _GARMENT_PRIORITY.get(ptype.slot_type(key), len(_GARMENT_PRIORITY))
            return (priority, rejected, position.get(key, len(position)))
        if bundle_type == "setup":
            return (rejected, position.get(key, len(position)))
        return (rejected,)   # stable: first-seen order within each

    if bundle_type == "outfit":
        for key in order:
            accepts = ptype.slot_accepts(key)
            if accepts:
                groups[key] = sorted(
                    groups[key],
                    key=lambda item: 0 if _row_type(item) in accepts
                    else 1 if _row_type(item) == ptype.UNKNOWN else 2,
                )
        anchor_key = next(
            (k for k in sorted(order, key=slot_key) if ptype.slot_type(k) == "apparel"), None
        )
        anchor = _product_gender(groups[anchor_key][0]) if anchor_key is not None else "neutral"
        if anchor in ("women", "men"):
            for key in order:
                if key != anchor_key:
                    groups[key] = sorted(
                        groups[key],
                        key=lambda item: _product_gender(item) not in (anchor, "neutral"),
                    )

    return _slots_from_groups(groups, sorted(order, key=slot_key), max_options)


def _clears_sponsored_gate(item) -> bool:
    """Reranked AND scored above the constraint-violator ceiling; else it stays organic."""
    score = item.get("rerank_score")
    return score is not None and score > cfg.SPONSORED_MIN_RERANK_SCORE


def _boost_sponsored(full, smap, requested=None, device=None):
    """Bid-order sponsored items that made the RERANKED pool to the top.

    With a type request (type-aware ranking, flat results) the ad must also BE that
    type: a sandal ad scored 41 on a clothing ask must not lead the page. Under a device
    ask (2026-10-27) an ad for an accessory OF the device stays organic: a winder ad
    scored 75 on a watch gift must not stand above every watch the display put first."""
    # Pool membership alone let a low-scored off-topic ad (e.g. a women's dress on a
    # men's-shirt query, capped <=40 by the prompt) jump to position 1
    if not smap:
        return full
    boosted, rest = [], []
    for item in full:
        asin = item.get("parent_asin")
        asin = str(asin) if asin is not None else None
        if (
            asin and asin in smap and _clears_sponsored_gate(item)
            and (not requested or _type_rank(item, requested) == 0)
            and not _is_addon(item, device)
        ):
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


def _blocking_errors(errors: list) -> list:
    """Errors that mean a stage FELL BACK. A rerank 'partial' still used the LLM's
    ranking, so its pool is cached: else page 2 / a chat refine re-rank and reshuffle.
    A translate or cross-sell 'partial' (query guard) likewise used the screened LLM
    answer, and the guard is deterministic; translate 'guarded' made no LLM call, so
    it is blocking (cheap to redo: no LLM call is made)."""
    return [e for e in errors if e.get("code") != "partial"]


def _recs_pinnable(outcome: dict) -> bool:
    """Whether recommendations built inside `outcome` (llm_client.memo_outcome) may be
    cached: not when the cross-sell stage refused its reply (an empty or fully screened
    one: the next open asks again, as before the /product cache) or when a reply lost the
    memo race (the next open is rebuilt from the stored winner)."""
    return not outcome["superseded"] and "cross_sell" not in outcome["rejected"]


def _llm_identity() -> tuple:
    """(provider, model, base URL), read at CALL time so a model switch changes the key."""
    provider = cfg.LLM_PROVIDER
    model = {"gemini": cfg.GEMINI_MODEL, "anthropic": cfg.ANTHROPIC_MODEL}.get(
        provider, cfg.OPENAI_MODEL
    )
    return (provider, model, cfg.OPENAI_BASE_URL)


# ------------------------------------------------------------------ degraded pools
# A pool an LLM stage fell back on is never put in page_cache, so searching again
# retries the LLM. But the rail and the pager re-request the SAME search: uncached, each
# sort, filter or page click re-ran translate + rerank (10-20 s under Groq's TPM limit)
# and served another pool, so the facet counts and the results moved under the shopper
# (Oct 15 review). Such follow-ups (?follow_up=true; the UI sends it on everything but a
# Search submit) get the degraded pool back for a short while instead. Kept apart from
# page_cache: a plain request -- a submit, /chat, /unified_search, the evals -- never
# reads it, so their retry-on-fallback behaviour is unchanged.
_DEGRADED_TTL_S = 120.0
_DEGRADED_MAX = 32
_degraded: "OrderedDict[str, tuple]" = OrderedDict()   # key -> (stored at, entry, errors)
_degraded_lock = Lock()


def _remember_degraded(key: str, entry: dict, errors: list) -> None:
    with _degraded_lock:
        _degraded[key] = (time.monotonic(), entry, [dict(e) for e in errors])
        _degraded.move_to_end(key)
        while len(_degraded) > _DEGRADED_MAX:
            _degraded.popitem(last=False)


def _recent_degraded(key: str) -> Optional[tuple]:
    """(entry, errors) of a degraded pool stored less than _DEGRADED_TTL_S ago, else None."""
    with _degraded_lock:
        stored = _degraded.get(key)
        if stored is None:
            return None
        if time.monotonic() - stored[0] > _DEGRADED_TTL_S:
            del _degraded[key]
            return None
        return stored[1], [dict(e) for e in stored[2]]


def _forget_degraded(key: str) -> None:
    with _degraded_lock:
        _degraded.pop(key, None)


def clear_degraded_pools() -> None:
    """Tests and evals: start from nothing, as page_cache.clear() does."""
    with _degraded_lock:
        _degraded.clear()


# ------------------------------------------------------------------ single flight
# Two requests for one uncached key (two demo laptops, two tabs, a double submit) each ran
# the pipeline and the last page_cache.put won, so shopper A's page 2 came from shopper
# B's pool: 3 of A's page-1 items reappeared on it (stub-measured, 2026-10-27). Now one
# pipeline run per key at a time; the others wait and take the leader's entry, clean or
# degraded, with copies of its errors. Holds only keys in flight, so it stays small.
class _Flight:
    __slots__ = ("done", "result", "started")

    def __init__(self):
        self.done = Event()
        self.result: Optional[tuple] = None   # (entry, errors, cacheable) once finished
        self.started = time.monotonic()       # the bound below counts from here


_flights: dict = {}
_flights_lock = Lock()
# How long a pipeline in flight holds the requests waiting on it, counted from the
# FLIGHT's start (not each waiter's arrival); after it a waiter runs its own pipeline. A
# pipeline is 2 LLM calls (~10-30 s, more under 429 pacing), but the openai SDK's defaults
# (600 s read timeout x 3 tries) can hold a stuck leader for half an hour: without a bound
# every re-submit, and every other shopper's search for the same query, joined it. 60 s,
# not the 150 s of 2026-10-27, which outlasted the UI's 90 s /search timeout
# (pages/ai_search.py): a shopper who re-submitted after the UI gave up joined the same
# stuck leader and timed out again. Now a re-submit 60 s or more into a flight runs its
# own pipeline at once, and a request that arrived with the leader still has 30 s of the
# UI's 90 for its own 2 calls (Qwen ~10 s, Groq gpt-oss ~20-25 s per query; a later
# arrival has more). A healthy leader gets ~2x its usual time before a second run
# starts. /chat (120 s in the UI) spends one assistant call before its search.
_FLIGHT_WAIT_S = 60.0


def _build_and_store(key: str, build) -> tuple:
    """Run the pipeline for a key page_cache missed and store what it may: a clean entry
    in page_cache, a degraded one in the degraded store. A run whose LLM reply lost the
    memo race (llm_client.memo_outcome: another request stored another reply for the same
    prompt first) is stored nowhere and returned not cacheable: this request still shows
    it, the next one is rebuilt from the stored winner, so the page converges on what a
    restart would serve.

    A clean run that finds the key already stored (another run for it finished first:
    a waiter that stopped waiting for a stuck leader, or that leader once it returns)
    keeps that FIRST entry and returns it, as the memo keeps the first accepted reply.
    The late run's own pool is dropped: with LLM_MEMO=false it used to overwrite the
    first one, so the waiter's page 2 came from the leader's pool; and this request's
    page 2 and page-1 cross-sell (search() merges it into the stored entry) come from
    the stored entry, so its page 1 must too."""
    # A leader that finished between this request's cache miss and now
    entry = page_cache.get(key)
    if entry is not None:
        return entry, [], True
    errors: list = []
    with memo_outcome() as outcome:
        entry = build(errors)
    cacheable = not _blocking_errors(errors)
    if outcome["superseded"]:
        cacheable = False
    elif cacheable:
        first = page_cache.put_if_absent(key, entry)
        _forget_degraded(key)
        if first is not entry:
            return first, [], True
    else:
        _remember_degraded(key, entry, errors)
    return entry, errors, cacheable


def _pool_once(key: str, build) -> tuple:
    """(entry, errors, cacheable) for a key page_cache missed; `build(errors)` runs the
    pipeline (_build_and_store). A waiter whose leader raised tries again (as the next
    leader), so an exception is never shared; once the leader's flight is _FLIGHT_WAIT_S
    old a waiter runs its own pipeline (no flight: the leader still holds it)."""
    while True:
        with _flights_lock:
            flight = _flights.get(key)
            leader = flight is None
            if leader:
                flight = _flights[key] = _Flight()
        if not leader:
            # wait(0) still reports a flight that has just finished
            remaining = max(0.0, flight.started + _FLIGHT_WAIT_S - time.monotonic())
            if not flight.done.wait(remaining):
                # Counts only: the key carries the query (safety.md)
                logger.warning("Single flight: the pipeline in flight passed %.0f s; this "
                               "request runs its own pipeline.", _FLIGHT_WAIT_S)
                return _build_and_store(key, build)
            if flight.result is None:
                continue
            entry, errors, cacheable = flight.result
            return entry, [dict(e) for e in errors], cacheable
        try:
            entry, errors, cacheable = _build_and_store(key, build)
            flight.result = (entry, [dict(e) for e in errors], cacheable)
            return entry, errors, cacheable
        finally:
            with _flights_lock:
                _flights.pop(key, None)
            flight.done.set()


# ------------------------------------------------------------------ filters, sort, facets
# The shop's left rail and sort select (GET /search sort/min_price/max_price/min_rating/
# department/ptype). Applied to the CACHED ranked pool after the lookup and never part
# of the key, so changing a filter or the sort refetches page 1 with 0 LLM calls. A
# price or rating sort reorders rows within their relevance group (_relevance_groups).
# Flat results only: a bundle page (recipe/setup) is one card per component and ignores
# them.
SORTS = ("relevance", "price_asc", "price_desc", "rating")
_RATING_STEPS = (4, 3, 2, 1)


def _query_default(value):
    """search() is also called as a plain function (/chat, /unified_search, the evals and
    tests): a parameter it is not passed arrives as its Query(...) marker, not a value."""
    return value.default if isinstance(value, Param) else value


def _csv_values(text) -> list:
    """'Fashion, electronics,fashion' -> ['Fashion', 'electronics'] (stripped, first
    spelling kept, de-duplicated case-insensitively)."""
    if not isinstance(text, str):
        return []
    values = {}
    for value in text.split(","):
        value = value.strip()
        if value:
            values.setdefault(value.casefold(), value)
    return list(values.values())


def _number(value) -> Optional[float]:
    """A finite float, else None (None, NaN, '', unparseable). '$1,299.00' parses."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _price(row) -> Optional[float]:
    """The LISTED price. Never price_estimate: that kNN guess is shown as '≈ $X (est.)'
    and is never sorted, filtered or totalled."""
    price = _number(row.get("price"))
    return price if price is not None and price >= 0 else None


def _average_rating(row) -> Optional[float]:
    rating = _number(row.get("average_rating"))
    return rating if rating is not None and 0 <= rating <= 5 else None


def _department(row) -> Optional[str]:
    """The vertical as the catalog spells it ('Fashion'); a lowercase one ('fashion', the
    local backend's row shape until 2026-10-15) is Title-cased back for display."""
    value = row.get("bsns_vrtcl_name")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    return value.title() if value == value.lower() else value


def _on_target(row, requested) -> bool:
    """Not already ruled out by the ranking: of a requested (or untyped) type when the
    shopper named one, and not judged a violator by the reranker (<= the ceiling)."""
    return _type_rank(row, requested) != 2 and _band(row) != 1


# The 2026-10-19 rail (contract C2): shopper gender, finer style, colour family, fabric and
# brand, read from columns app/product_type.annotate derives at load (never embedded, so
# no re-encode). Keys and labels of style and colour come from product_type
# (FINE_SUBTYPE_LABELS, COLOR_FAMILIES); these are the ones main.py owns.
_SHOPPER_GENDER, _FINE_SUBTYPE, _COLOR_FAMILY = "shopper_gender", "fine_subtype", "color_family"
_GENDER_LABELS = {"women": "Women", "men": "Men", "kids": "Kids",
                  "unisex": "Unisex / unspecified"}
# `material` is one rule-set value per row, and it mixes fabrics with "elastic", "metal",
# "rubber" or "stainless steel" (catalog audit: 46.6% of Fashion carries a value, 40.9% a
# fabric one). The Fabric section offers only these, and only on a clothing pool
_FABRICS = frozenset({
    "acrylic", "bamboo", "canvas", "cashmere", "chiffon", "cotton", "denim", "down",
    "faux leather", "fleece", "jersey", "knit", "lace", "leather", "linen", "mesh", "modal",
    "neoprene", "nylon", "polycotton", "polyester", "rayon", "satin", "silk", "spandex",
    "suede", "velvet", "wool",
})
# Listed prices only (never price_estimate). A band holds [low, high): a $25.00 item is in
# "$25 to $50" only (157 catalog rows sit exactly on $25; overlapping bands counted them
# twice). min_price/max_price filter inclusively, so a band sends max = high - 1 cent,
# and its count is what sending its bounds returns (every catalog price has 2 decimals)
_PRICE_EDGES = (25, 50, 100, 200)
_STORE_FACET_TOP = 30
_MAX_STORES, _MAX_STORE_CHARS = 20, 120


def _tag(row, column) -> Optional[str]:
    """A derived label column as a casefolded str, else None (None, NaN, '', absent)."""
    value = row.get(column)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().casefold()


def _store_fold(value) -> str:
    """A seller name for matching: casefolded, whitespace collapsed ('' when missing).
    606 catalog store names contain a comma ("Topsville, Inc."), hence a repeated param."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.casefold().split())


def _price_bands(prices) -> list:
    """[{label, min, max, n}] over listed prices: up to $25, $25-50, ... $200 & above,
    each [low, high) with max = high - 0.01 (see _PRICE_EDGES)."""
    bands, low = [], None
    for high in _PRICE_EDGES + (None,):
        if low is None:
            label = f"Up to ${high}"
        elif high is None:
            label = f"${low} & above"
        else:
            label = f"${low} to ${high}"
        n = sum(1 for p in prices
                if (low is None or p >= low) and (high is None or p < high))
        top = None if high is None else round(high - 0.01, 2)
        bands.append({"label": label, "min": low, "max": top, "n": n})
        low = high
    return bands


def _facets(full, requested=None) -> dict:
    """Counts over the UNFILTERED pool, so a count is what ticking that box returns
    (before VARIANT_COLLAPSE: a count is rows, the filtered total is cards).

    A value is OFFERED only when at least one of its rows is on target (_on_target): the
    "wedding dresses ... in rajasthan" rail offered 'Electronics (7)' and 'Grocery (5)',
    and ticking one showed photo backdrops and cake toppers -- rows the display order had
    already put last as another type (Oct 15 review). Price and rating cover every row:
    they are not options a shopper ticks. Fabric is offered on clothing pools only."""
    departments, types, prices = {}, {}, []
    offered_departments, offered_types = set(), set()
    ratings = {str(k): 0 for k in _RATING_STEPS}
    genders, subtypes, colors, materials = {}, {}, {}, {}
    offered = {"gender": set(), "subtype": set(), "color": set(), "material": set()}
    stores = {}            # fold -> {"n": rows, "names": {spelling: rows}, "offered": bool}
    clothing = False
    subtype_labels = ptype.FINE_SUBTYPE_LABELS
    color_hex, color_labels = ptype.COLOR_FAMILIES, ptype.COLOR_LABELS
    for row in full:
        on_target = _on_target(row, requested)
        name = _department(row)
        if name:
            departments[name] = departments.get(name, 0) + 1
            if on_target:
                offered_departments.add(name)
        row_type = _row_type(row)
        if row_type in ptype.FASHION_TYPES:
            types[row_type] = types.get(row_type, 0) + 1
            if on_target:
                offered_types.add(row_type)
        clothing = clothing or (on_target and row_type == "apparel")
        for column, counts, kind, valid in (
                (_SHOPPER_GENDER, genders, "gender", _GENDER_LABELS),
                (_FINE_SUBTYPE, subtypes, "subtype", subtype_labels),
                (_COLOR_FAMILY, colors, "color", color_hex)):
            value = _tag(row, column)
            if value in valid:
                counts[value] = counts.get(value, 0) + 1
                if on_target:
                    offered[kind].add(value)
        material = _tag(row, "material")
        if material in _FABRICS:
            materials[material] = materials.get(material, 0) + 1
            if on_target and row_type == "apparel":
                offered["material"].add(material)
        store = attrs.brand_name(row)    # `brand`, else the store (Fashion: the store)
        fold = _store_fold(store)
        if fold:
            entry = stores.setdefault(fold, {"n": 0, "names": {}, "offered": False})
            entry["n"] += 1
            spelling = " ".join(store.split())
            entry["names"][spelling] = entry["names"].get(spelling, 0) + 1
            entry["offered"] = entry["offered"] or on_target
        price = _price(row)
        if price is not None:
            prices.append(price)
        rating = _average_rating(row)
        if rating is not None:
            for k in _RATING_STEPS:
                if rating >= k:
                    ratings[str(k)] += 1

    # Electronics / Grocery sections (2026-10-27), described by app/attributes.py's
    # registry -- labels included -- and the department half the on-target rows share
    sections, primary = attrs.facet_sections(
        full, lambda r: _on_target(r, requested), _department)

    def by_count(counts, kind):
        return sorted((k for k in counts if k in offered[kind]), key=lambda k: (-counts[k], k))

    # The most frequent spelling names a store (the first seen on a tie). A name the
    # `store` param refuses (over _MAX_STORE_CHARS: two catalog "stores" are VHS credit
    # lines of 126 and 130 characters) is not offered -- ticking it returned a 422
    named = ((max(e["names"].items(), key=lambda kv: kv[1])[0], e["n"])
             for e in stores.values() if e["offered"])
    offered_stores = sorted(
        ({"name": name, "n": n} for name, n in named if len(name) <= _MAX_STORE_CHARS),
        key=lambda s: (-s["n"], s["name"].casefold()),
    )
    return {
        "departments": dict(sorted(
            ((name, n) for name, n in departments.items() if name in offered_departments),
            key=lambda kv: (-kv[1], kv[0]),
        )),
        "product_types": {t: types[t] for t in ptype.FASHION_TYPES if t in offered_types},
        "price": {
            "min": min(prices) if prices else None,
            "max": max(prices) if prices else None,
            "n_priced": len(prices),
            "bands": _price_bands(prices),
        },
        "rating": ratings,
        "genders": [{"key": g, "label": _GENDER_LABELS[g], "n": genders[g]}
                    for g in _GENDER_LABELS if g in offered["gender"]],
        "subtypes": [{"key": s, "label": subtype_labels[s], "n": subtypes[s]}
                     for s in by_count(subtypes, "subtype")],
        "colors": [{"key": c, "label": color_labels.get(c) or c.replace("_", " ").title(),
                    "hex": color_hex[c], "n": colors[c]} for c in by_count(colors, "color")],
        "materials": ([{"key": m, "label": m.title(), "n": materials[m]}
                       for m in by_count(materials, "material")] if clothing else []),
        "stores": offered_stores[:_STORE_FACET_TOP],
        "n_stores": len(offered_stores),
        "attributes": sections,
        "primary_department": primary,
    }


def _active_filters(min_price, max_price, min_rating, departments, types, *, genders=(),
                    subtypes=(), colors=(), materials=(), stores=(), attributes=None) -> dict:
    """The filters a request set, echoed as `filters_applied` (unset ones left out).
    `attributes` = attrs.parse_filters' {param: [keys]}, echoed under each param."""
    active = {}
    for name, value in (("min_price", min_price), ("max_price", max_price),
                        ("min_rating", min_rating)):
        if value is not None:
            active[name] = float(value)
    if departments:
        active["department"] = departments
    if types:
        active["ptype"] = types
    for name, values in (("gender", genders), ("subtype", subtypes), ("color", colors),
                         ("material", materials), ("store", stores)):
        if values:
            active[name] = list(values)
    for name, values in (attributes or {}).items():
        if values:
            active[name] = list(values)
    return active


def _passes(row, active: dict) -> bool:
    if "min_price" in active or "max_price" in active:
        price = _price(row)
        if price is None:
            return False      # an unpriced row can't be said to be in any range
        if price < active.get("min_price", 0.0) or price > active.get("max_price", math.inf):
            return False
    if "min_rating" in active:
        rating = _average_rating(row)
        if rating is None or rating < active["min_rating"]:
            return False
    if "department" in active:
        department = row.get("bsns_vrtcl_name")
        wanted = {d.casefold() for d in active["department"]}
        if not isinstance(department, str) or department.strip().casefold() not in wanted:
            return False
    if "ptype" in active and _row_type(row) not in active["ptype"]:
        return False          # a non-Fashion row has no type: it never matches
    # The derived labels are None off Fashion (colour: wherever `color` is set), so a
    # row without one never matches, like ptype
    for name, column in (("gender", _SHOPPER_GENDER), ("subtype", _FINE_SUBTYPE),
                         ("color", _COLOR_FAMILY), ("material", "material")):
        if name in active and _tag(row, column) not in active[name]:
            return False
    # The Brands facet's names: `brand`, else the store (attrs.brand_name)
    if "store" in active and _store_fold(attrs.brand_name(row)) not in \
            {_store_fold(s) for s in active["store"]}:
        return False
    # Electronics / Grocery sections: a row of another department never matches
    if any(p in active for p in attrs.PARAMS) and \
            not attrs.passes(row, active, _department(row)):
        return False
    return True


def _relevance_groups(full, requested, device=None) -> dict:
    """{id(row): (type mismatch, band)} for a flat pool in relevance order -- the key
    _flat_display leads with, so a price or rating sort reorders rows WITHIN a group,
    never across one (Oct 15 review). Sorting the whole pool put the never-reranked
    tail first: 12 cake toppers on "birthday gift for a 10 year old", earbud cleaning
    kits on "gift for a vintage watch collector", cables and no power bank on "power
    bank", and under a type request a bridal bracelet before the first garment. Under a
    device ask the display's third element too (not the device): a price sort cannot put
    a $12 sleeve above every laptop on "laptop for college".

    With nothing reranked (rerank off, a fallback, a guarded query) every band reads 2,
    so the first RERANK_INPUT_K rows -- the ones the reranker would have judged -- stand
    in for band 0. Groups follow the display's band order (_band_position)."""
    scored = any(_is_score(row.get("rerank_score")) for row in full)
    groups = {}
    for position, row in enumerate(full):
        band = _band(row) if scored else (0 if position < cfg.RERANK_INPUT_K else 2)
        group = (_type_rank(row, requested) == 2, _band_position(band))
        groups[id(row)] = group + (_not_core(row, device),) if device is not None else group
    return groups


# ------------------------------------------------------------------ variants, match quality
def _variant_key(row) -> str:
    """search.style_key for an organic Fashion row; '' (never merged) for any other row.
    A sponsored row is never merged either way (safety.md: ads apart from organic): it
    absorbed its organic siblings as swatches of a paid card, and under a price sort an
    organic sibling fronted the card and the ad lost its Sponsored label as a swatch."""
    if not _is_fashion_row(row) or row.get("is_sponsored"):
        return ""
    return style_key(row.get("store"), row.get("Product_title"))


def _variant_view(row) -> dict:
    def clean(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return None if isinstance(value, str) and not value.strip() else value
    return {"catalog_index": clean(row.get("catalog_index")),
            "title": clean(row.get("Product_title")), "color": clean(row.get("color")),
            "color_family": clean(row.get(_COLOR_FAMILY)), "img_url": clean(row.get("img_url"))}


def _collapse_variants(rows) -> list:
    """One card per style (VARIANT_COLLAPSE, owner call 2): the first row of each
    search.style_key group in the CURRENT order stays, carrying `variants` (every row of
    the group, itself first) and `n_variants`; the rest leave the list. Runs after the
    filters and the sort, before paging, so a page holds 12 different products. Copies
    the rows it annotates: the cached pool is never written to."""
    if not cfg.VARIANT_COLLAPSE:
        return list(rows)
    out, first, members = [], {}, {}
    for row in rows:
        key = _variant_key(row)
        if key and key in members:
            members[key].append(row)
            continue
        if key:
            members[key], first[key] = [row], len(out)
        out.append(row)
    for key, group in members.items():
        if len(group) > 1:
            out[first[key]] = dict(out[first[key]], variants=[_variant_view(r) for r in group],
                                   n_variants=len(group))
    return out


# "No exact matches" (synthesis P8): x02, p03, p07 and m04 filled page 1 with kids' shoes,
# black canvas shoes, silk bras and robes, under a header implying every row was a match.
# A page is weak (contract C2) when fewer than this many rows were judged on target
# (above the floor)...
_WEAK_MIN_GOOD = 4
# ...or when the best judged score is at most this (41-55 rows averaged 0.80 of 2 with the
# graders, below the 1.01 of unjudged tail rows)
_WEAK_MAX_BEST = 55
# A strong row scored at least this: the 71-85 band averaged 1.55 of 2 (strict 0.58), the
# 56-70 band 1.13. Counted always; it makes a page weak only with WEAK_MATCH_MIN_STRONG > 0
# (app/config.py: stricter than C2, the replay numbers and why it is not the default)
_STRONG_MIN = 71


def _judged_share(minimum: int, judged: int, window: int) -> int:
    """`minimum` good rows of a full window, scaled to the rows the reranker returned
    (rounded up): 4 of 15 when it judged all 15, 1 when it returned 3. A reply may list
    fewer entries than asked ("wedding dress": 95 / 85 / 70, 3 of 15 returned) and is not
    a partial (only invalid entries are), so its unreturned window rows are unknown, not
    misses: counted as misses they put the "No exact matches" banner over gowns scored 95
    and 85 (2026-10-19 review)."""
    if window <= 0 or judged >= window:
        return minimum
    return -(-minimum * judged // window)


def _match_quality(full, requested, device=None) -> dict:
    """{band0, strong, best_rerank, weak, unjudged} over the unfiltered flat pool. band0
    counts rows the reranker scored above the floor that the display does not demote as
    another type, strong those scored >= _STRONG_MIN; best_rerank is the best score among
    the rows of a requested (or untyped) type. weak = band0 < 4 or best <= 55 (C2), or
    strong < WEAK_MATCH_MIN_STRONG, both counts against the rows the reranker actually
    judged (_judged_share: a short reply scales them down). Under a device ask an
    accessory for the device never counts as on type (2026-10-27): "iphone" scored the
    one handset 100 and 14 cases and chargers at 40-90 by turns, and a page of them is no
    exact match. Nothing scored (rerank off, a fallback, a guarded query): unjudged,
    never weak."""
    on_type = [r for r in full if _type_rank(r, requested) != 2 and not _is_addon(r, device)]
    scores = [float(r["rerank_score"]) for r in on_type if _is_score(r.get("rerank_score"))]
    judged = sum(1 for r in full if _is_score(r.get("rerank_score")))
    if not judged:
        return {"band0": 0, "strong": 0, "best_rerank": None, "weak": False,
                "unjudged": True}
    window = min(cfg.RERANK_INPUT_K, len(full))
    band0 = sum(1 for r in on_type if _band(r) == 0)
    strong = sum(1 for s in scores if s >= _STRONG_MIN)
    best = max(scores) if scores else None
    weak = (band0 < _judged_share(_WEAK_MIN_GOOD, judged, window)
            or best is None or best <= _WEAK_MAX_BEST
            or strong < _judged_share(cfg.WEAK_MATCH_MIN_STRONG, judged, window))
    return {"band0": band0, "strong": strong, "best_rerank": best, "weak": weak,
            "unjudged": False}


def _sorted_view(rows, sort: str, groups: Optional[dict] = None) -> list:
    """Stable, so ties keep the relevance order. With `groups` (_relevance_groups) a
    non-relevance sort orders rows within their relevance group. Unpriced rows go LAST
    (within the group) in both price orders. A sponsored row keeps its label but sorts
    like any other row."""
    def group(row):
        return groups[id(row)] if groups else ()

    if sort == "price_asc":
        return sorted(rows, key=lambda r: (group(r), _price(r) is None, _price(r) or 0.0))
    if sort == "price_desc":
        return sorted(rows, key=lambda r: (group(r), _price(r) is None, -(_price(r) or 0.0)))
    if sort == "rating":
        return sorted(rows, key=lambda r: (group(r), -bayesian_rating(
            r.get("average_rating"), r.get("rating_number"))))
    return list(rows)


def _ranked_pool(query, llm_query, translator_mode, rerank, sponsored, pool_k, typed,
                 timings, errors, device_roles: Optional[bool] = None) -> dict:
    """Translate -> retrieve -> rerank -> order: the page_cache entry for one query and
    its settings (every page size, filter and sort is served from it). `typed` and
    `device_roles` are the knob values search() put in the cache key, so a knob flipped
    mid-request (an eval, a notebook) cannot store one mode's pool under the other's key
    (None reads cfg)."""
    with timings.stage("translate"):
        # ONE LLM call: intents + bundle_type + typed constraints
        understanding = understand_query(query, mode=translator_mode, errors=errors)
        intents = understanding["intents"]
        bundle_type = understanding.get("bundle_type")
        constraints = understanding["constraints"]
    entry = {
        "full": [],
        "intents": intents,
        "constraints": constraints,
        "is_recipe": bundle_type == "recipe",
        "bundle_type": bundle_type,
        "rerank_succeeded": False,
        "no_match": True,
        # {page_size: cross-sell/upsell of THAT page 1}; a size not here is built on use
        "recommendations": {},
        # The type request the display order used: a sort and the facets keep to it
        "requested": None,
        # The device ask the display order used (device_role.DeviceAsk): the sorts, the
        # match quality and the ad gate keep to it
        "device": None,
    }

    gender_stated = _gender_stated(query, constraints, typed)
    # A fit word states women's only if the pool turns out Fashion-majority
    women_fit = not gender_stated and _women_fit_stated(query, typed, bundle_type)

    with timings.stage("retrieve"):
        if bundle_type:
            # Cap per intent so one bundle component can't crowd out the others;
            # keep= is passed only when there is a filter, so a search_products
            # stand-in without the parameter (tests, old callers) still works
            keep = _bundle_keep(bundle_type) if typed else None
            bundle_search = dict(top_k=RETRIEVAL_TOP_K, per_intent_quota=GROCERY_PER_INTENT_K)
            candidates = search_products(
                intents, **bundle_search, **({"keep": keep} if keep is not None else {})
            )
            if keep is not None and bundle_type == "setup":
                candidates = _refill_empty_slots(candidates, intents, **bundle_search)
        elif not (gender_stated or women_fit):
            # 2x deep so the skewed catalog's men's items reach the rerank window
            candidates = search_products(intents, top_k=RETRIEVAL_TOP_K * 2)
            if not typed:
                candidates = _balance_candidate_pool(candidates)
        else:
            candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)
            if women_fit and not _fashion_majority(candidates):
                # Not a Fashion pool ("petite mints"): the fit word states nothing, so
                # the unstated-gender retrieval, exactly as without it (one more search)
                women_fit = False
                candidates = search_products(intents, top_k=RETRIEVAL_TOP_K * 2)
    gender_stated = gender_stated or women_fit

    if not candidates or best_score(candidates) < MIN_RESULT_RELEVANCE:
        return entry

    # What the reranker is told; the response and the cache keep the translator's
    rerank_constraints = constraints
    requested = None   # flat only: a bundle's slots carry their own types
    device = None      # flat only, like the type request
    if typed:
        rerank_constraints = _without_setting_constraints(constraints)
        if bundle_type:
            # Every slot gets reranked options inside RERANK_INPUT_K
            candidates = _round_robin_by_intent(candidates, intents)
        else:
            # Measured on the RAW score order, before the window reorders it. The
            # sanitised query: the shopper's ask is the text before any instruction
            # marker (identical for an ordinary query; '' when guarded names none)
            requested = _effective_request(llm_query, candidates)
            ask = _effective_ask(llm_query, candidates)
            device = _effective_device(llm_query, candidates, enabled=device_roles)
            guest = _guest_constraint(llm_query) if ask is not None else None
            candidates = _flat_window(
                candidates, requested, intents, interleave=not gender_stated, ask=ask,
                device=device,
            )
            if requested:
                rerank_constraints = rerank_constraints + [_type_constraint(requested)]
            if guest:
                rerank_constraints = rerank_constraints + [guest]
            if device:
                # Code-written like the type request: deterministic text per family, never
                # the shopper's words; the response keeps the translator's constraints
                rerank_constraints = rerank_constraints + [device_role.constraint(device)]

    rerank_actually_ran = False
    if rerank and candidates and llm_query:
        # Under a device ask the window rows carry a transient ` | role=` field, on copies
        rerank_input = _with_role_fields(candidates, device) if device else candidates
        with timings.stage("rerank"):
            ranked_pool = llm_rerank(
                llm_query, rerank_input, top_k=pool_k, errors=errors,
                constraints=rerank_constraints,
            )
        if device:
            ranked_pool = _without_role_field(ranked_pool)
        rerank_actually_ran = any(r.get("rerank_score") is not None for r in ranked_pool)
    else:
        # Rerank off, or the guard refused the query (translate 'guarded'): no
        # LLM may judge it, so embedding order -- and 'guarded' is never cached
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
        if typed:
            full = _typed_bundle_slots(full, bundle_type, intents, max_options=3)
        else:
            full = _recipe_slots_with_alternatives(full, max_options=3)
        if sponsored:
            # Nested-aware: _boost_sponsored can't see slot `alternatives`
            with timings.stage("sponsored"):
                full = _promote_sponsored_options(full, sponsored_map())
    else:
        if typed:
            full = _flat_display(full, requested, interleave=not gender_stated, device=device)
        elif not gender_stated:
            # Re-balance the DISPLAY order after the reranker re-scored the pool
            full = _balance_by_gender(full)
        if sponsored:
            with timings.stage("sponsored"):
                full = _boost_sponsored(full, sponsored_map(), requested, device)

    return {**entry, "full": full, "rerank_succeeded": rerank_actually_ran, "no_match": False,
            "requested": requested, "device": device}


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
    sort: str = Query(
        "relevance", description="relevance (default) | price_asc | price_desc | rating"
    ),
    min_price: Optional[float] = Query(
        None, ge=0, description="Listed price floor; unpriced rows are excluded when set."
    ),
    max_price: Optional[float] = Query(
        None, ge=0, description="Listed price ceiling; unpriced rows are excluded when set."
    ),
    min_rating: Optional[float] = Query(
        None, ge=0, le=5, description="average_rating floor; unrated rows are excluded."
    ),
    department: Optional[str] = Query(
        None, max_length=300,
        description="Comma-separated departments (bsns_vrtcl_name), case-insensitive.",
    ),
    ptype_filter: Optional[str] = Query(
        None, alias="ptype", max_length=300,
        description="Comma-separated product types: apparel, footwear, jewelry, bag, "
        "headwear, accessory. Rows of other departments never match.",
    ),
    gender: Optional[str] = Query(
        None, max_length=300,
        description="Comma-separated shopper genders: women, men, kids, unisex.",
    ),
    subtype: Optional[str] = Query(
        None, max_length=300,
        description="Comma-separated style keys (facets.subtypes[].key).",
    ),
    color: Optional[str] = Query(
        None, max_length=300,
        description="Comma-separated colour families (facets.colors[].key).",
    ),
    material: Optional[str] = Query(
        None, max_length=300, description="Comma-separated materials, case-insensitive.",
    ),
    store: Optional[List[str]] = Query(
        None,
        description="Brand names (facets.stores[].name: the listing's brand, else its "
        "seller), REPEATED (store=a&store=b): names contain commas. Case- and "
        "whitespace-insensitive; at most 20, each up to 120 characters.",
    ),
    # The per-department sections of facets.attributes (app/attributes.py). Declared one
    # by one, not as a catch-all: FastAPI silently drops an undeclared param, so a filter
    # would vanish with no 422, and OpenAPI documents them for a client integration.
    # Lists, so a REPEATED param (dietary=vegan&dietary=keto, the `store` pattern) means
    # its csv: as a str, FastAPI kept only the LAST value with no 422, which dropped a
    # dietary restriction and widened the page. No Query(max_length=): on a list that
    # caps the item count; attrs.parse_filters caps the joined csv at MAX_CHARS (a 422)
    category: Optional[List[str]] = Query(
        None,
        description="Category keys: Electronics groups or Grocery categories "
        "(facets.attributes[param=category].values[].key). Comma-separated or repeated; "
        f"at most {attrs.MAX_CHARS} characters in all.",
    ),
    item_type: Optional[List[str]] = Query(
        None,
        description="Electronics item types: device, accessory, part, consumable. "
        f"Comma-separated or repeated; at most {attrs.MAX_CHARS} characters in all.",
    ),
    works_with: Optional[List[str]] = Query(
        None,
        description="Ecosystems an Electronics accessory names "
        "(facets.attributes[param=works_with].values[].key). Comma-separated or "
        f"repeated; at most {attrs.MAX_CHARS} characters in all.",
    ),
    condition: Optional[List[str]] = Query(
        None,
        description="Electronics conditions: renewed, used. Comma-separated or repeated; "
        f"at most {attrs.MAX_CHARS} characters in all.",
    ),
    dietary: Optional[List[str]] = Query(
        None,
        description="Grocery diet keys; a row must carry EVERY one. Comma-separated or "
        f"repeated; at most {attrs.MAX_CHARS} characters in all.",
    ),
    follow_up: bool = Query(
        False,
        description="A page, filter, sort or page-size change of results already shown: "
        "a pool an LLM stage fell back on is reused for a short while instead of "
        "re-running the LLM. Leave false for a new search (it retries the LLM).",
    ),
):
    """Translate -> retrieve -> rerank -> paginate, plus sponsored and cross-sell.
    Filters, sort and facets are applied to the cached pool (no LLM call)."""
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
    sort = _query_default(sort) or "relevance"
    if sort not in SORTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown sort={sort!r}; expected {' | '.join(SORTS)}",
        )
    types = [t.casefold() for t in _csv_values(_query_default(ptype_filter))]
    unknown = [t for t in types if t not in ptype.FASHION_TYPES]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown ptype {unknown!r}; expected {', '.join(ptype.FASHION_TYPES)}",
        )
    keyed = {}
    for name, value, allowed in (("gender", gender, _GENDER_LABELS),
                                 ("subtype", subtype, ptype.FINE_SUBTYPE_LABELS),
                                 ("color", color, ptype.COLOR_FAMILIES)):
        values = [v.casefold() for v in _csv_values(_query_default(value))]
        unknown = [v for v in values if v not in allowed]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown {name} {unknown!r}; expected one of {', '.join(allowed)}",
            )
        keyed[name] = values
    materials = _query_default(material)
    if isinstance(materials, str) and len(materials) > 300:
        raise HTTPException(status_code=422, detail="material is longer than 300 characters")
    stores = _query_default(store) or []
    if isinstance(stores, str):
        stores = [stores]
    stores = list({_store_fold(s): " ".join(s.split()) for s in stores
                   if isinstance(s, str) and s.strip()}.values())
    if len(stores) > _MAX_STORES or any(len(s) > _MAX_STORE_CHARS for s in stores):
        raise HTTPException(
            status_code=422,
            detail=f"store: at most {_MAX_STORES} values of up to {_MAX_STORE_CHARS} characters",
        )
    try:
        attributes = attrs.parse_filters({
            "category": _query_default(category), "item_type": _query_default(item_type),
            "works_with": _query_default(works_with), "condition": _query_default(condition),
            "dietary": _query_default(dietary),
        })
    except ValueError as e:      # the message names the param and keys, never the query
        raise HTTPException(status_code=422, detail=str(e))
    active = _active_filters(
        _query_default(min_price), _query_default(max_price), _query_default(min_rating),
        _csv_values(_query_default(department)), types,
        genders=keyed["gender"], subtypes=keyed["subtype"], colors=keyed["color"],
        materials=[m.casefold() for m in _csv_values(materials)], stores=stores,
        attributes=attributes,
    )

    page_size = top_k
    pool_k = max(RERANK_POOL_K, page_size)
    timings = StageTimings()
    # The rerank and cross-sell LLMs get only what the translator's guard lets an LLM
    # see (the same pure function); "" = the query opened with an instruction
    llm_query = query_guard.sanitize_query(query).clean
    # Read ONCE per request, so the window, the display and the cache key agree even if
    # an eval flips it mid-run
    typed = cfg.TYPE_AWARE_RANKING
    outfit_bundles = getattr(cfg, "OUTFIT_BUNDLES_ENABLED", False)

    # The POOL does not depend on the page size (pool_k = RERANK_POOL_K up to 30 per
    # page), so page_size is not in the key: moving the "Results per page" slider re-ran
    # both LLM calls and served a different page (Oct 15 review). Page 1's cross-sell is
    # the one page-size-dependent part, kept per size inside the entry. `recommend` must
    # be in the key or /unified_search's recommend=False poisons it; the LLM identity
    # must be too, or a model switch serves the old model's ranking; and the guard,
    # band-order, outfit and ordering states, or toggling one in-process serves the other
    # mode's pool. The ordering knob stays LAST (tests/test_ranking.py checks the key's
    # end). VARIANT_COLLAPSE is applied after the lookup, so it is not in the key.
    # DEVICE_ROLE_RANKING (2026-10-27) orders the window and the display, right after the
    # band-order knob.
    rejects_after_tail = cfg.REJECTS_AFTER_TAIL
    device_roles = cfg.DEVICE_ROLE_RANKING
    cache_key = page_cache.make_key(
        query,
        (
            pool_k, rerank, recommend, sponsored,
            translator_mode or TRANSLATOR_MODE, _llm_identity(), cfg.QUERY_GUARD_ENABLED,
            rejects_after_tail, device_roles, outfit_bundles, typed,
        ),
    )

    try:
        # The first request caches the ranked pool; later pages, page sizes up to the
        # pool, filters and sorts reuse it, so no LLM call is made
        entry = page_cache.get(cache_key)
        errors: list = []
        cacheable = True
        if entry is None and _query_default(follow_up):
            recent = _recent_degraded(cache_key)
            if recent is not None:
                # The fallback errors come back too: the page still says it is degraded
                entry, errors = recent
                cacheable = False
        if entry is None:
            entry, errors, cacheable = _pool_once(cache_key, lambda errs: _ranked_pool(
                query, llm_query, translator_mode, rerank, sponsored, pool_k, typed,
                timings, errs, device_roles=device_roles,
            ))

        full = entry["full"]
        intents = entry["intents"]
        constraints = entry.get("constraints", [])
        bundle_type = entry.get("bundle_type")
        is_recipe = bundle_type == "recipe"
        flat = not bundle_type
        if not flat:
            active = {}      # a bundle is one card per component: nothing to filter
        common = {
            "user_query": query,
            "interpreted_as": intents,
            "constraints": constraints,
            # From the backend only: source_intent is on every result, not just recipes
            "is_recipe": is_recipe,
            "bundle_type": bundle_type,
            "translator_mode": translator_mode or TRANSLATOR_MODE,
            "rerank_requested": rerank,
            "errors": errors,
            "page_size": page_size,
            "sort": sort if flat else "relevance",
            "filters_applied": active,
        }

        if entry["no_match"] or not full:
            return {
                **common,
                "rerank_succeeded": False,
                "no_match": True,
                "message": "No matching products found. Try different or more general terms.",
                "results": [],
                "sponsored": [],
                "recommendations": {"cross_sell": [], "upsell": []},
                "page": 1,
                "total_results": 0,
                "total_pages": 1,
                "has_prev": False,
                "has_next": False,
                "facets": None,
                "match_quality": None,
                "total_unfiltered": 0,
                "pool_rows": 0,
                "latency_ms": timings.to_dict(),
            }

        recommendations = {"cross_sell": [], "upsell": []}
        # Page 1 only, on the relevance-ordered page 1 (a filter or sort never costs an
        # LLM call). A guarded query (llm_query '') reaches no LLM, cross-sell included
        if recommend and page == 1 and llm_query:
            stored = (entry.get("recommendations") or {}).get(page_size)
            if stored is not None:
                recommendations = stored
            else:
                with timings.stage("recommend"), memo_outcome() as rec_outcome:
                    recommendations = build_recommendations(
                        llm_query, full[:page_size], page=1
                    )
                # A cross-sell fallback isn't pinned (the next page 1 retries ONLY that
                # stage), but the clean POOL is: else page 2 re-ranks (unseeded on
                # Databricks) and pages overlap. A cross-sell 'partial' (guard-screened
                # phrases) still used the LLM's answer: pinned like a clean one. Not
                # pinned either: a reply the stage refused (nothing usable) or one that
                # lost the memo race (llm_client.memo_outcome)
                if (cacheable and not _blocking_errors(recommendations.get("errors") or [])
                        and _recs_pinnable(rec_outcome)):
                    latest = page_cache.get(cache_key) or entry
                    page_cache.put(cache_key, {
                        **latest,
                        "recommendations": {
                            **(latest.get("recommendations") or {}),
                            page_size: recommendations,
                        },
                    })

        if flat:
            # cached pool -> filters -> sort (within relevance groups) -> variant
            # collapse -> page; the facets and match quality read the unfiltered pool
            requested = entry.get("requested")
            device = entry.get("device")
            facets = _facets(full, requested)
            match_quality = _match_quality(full, requested, device)
            groups = _relevance_groups(full, requested, device) if sort != "relevance" else None
            view = _sorted_view([row for row in full if _passes(row, active)], sort, groups)
            view = _collapse_variants(view)
            # Cards, like total_results: the header says "(filtered from N)" whenever the
            # two differ, so rows here would announce a filter nobody set
            unfiltered = len(_collapse_variants(full)) if cfg.VARIANT_COLLAPSE else len(full)
        else:
            facets, view, match_quality, unfiltered = None, full, None, len(full)
        start = (page - 1) * page_size
        results = view[start : start + page_size]
        total_results = len(view)
        total_pages = max(1, -(-total_results // page_size))

        payload = {
            **common,
            "rerank_succeeded": entry["rerank_succeeded"],
            "no_match": False,
            "results": results,
            "sponsored": [r for r in results if r.get("is_sponsored")],
            "recommendations": recommendations,
            "page": page,
            "total_results": total_results,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "facets": facets,
            "match_quality": match_quality,
            "total_unfiltered": unfiltered,
            # Rows, the unit of every facet count (facets.price.n_priced included)
            "pool_rows": len(full),
            "latency_ms": timings.to_dict(),
        }
        if not view:
            payload["message"] = "No products match these filters."
        return payload

    except Exception as e:
        logger.exception("Search failed")  # never the query text: safety.md, shared App logs
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
    top_k: int = Query(FINAL_TOP_K, ge=1, le=50, description="Results per page."),
    sponsored: bool = Query(SPONSORED_ENABLED, description="Sponsored labels (page 1 only)."),
    page: int = Query(1, ge=1, description="1-based page number."),
):
    """Pure lexical BM25 search: never falls back to CAS, zero LLM calls."""
    if _keyword_engine is None:
        raise HTTPException(status_code=503, detail="Keyword engine not ready.")
    # A plain-function caller that omits `page` gets its Query(...) marker (see search())
    page = _query_default(page)
    results, num_matched, top_relevance, top_coverage = _keyword_engine.search(
        query, k=top_k, offset=(page - 1) * top_k
    )
    # Page 1 only: BM25 top-k membership is the ad's relevance gate (rows have no
    # rerank score), and page 1 IS the top k -- as on /search, ads lead page 1
    if sponsored and page == 1:
        results = _boost_sponsored_keyword_rows(results, sponsored_map())
    total_pages = max(1, -(-num_matched // top_k))
    return {
        "user_query": query,
        "results": results,
        "no_match": num_matched == 0,
        "num_matched": num_matched,
        "top_relevance": round(top_relevance, 4),
        "top_coverage": round(top_coverage, 4),
        "page": page,
        "page_size": top_k,
        "total_results": num_matched,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
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
        logger.exception("Unified search failed")  # never the query text (safety.md)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/product")
def product(
    catalog_index: int = Query(..., ge=0, description="Catalog row id from a search result."),
    # Same cap as /search: this text is inserted into the cross-sell LLM prompt
    query: Optional[str] = Query(
        None, max_length=300, description="Original search query, for recommendation context."
    ),
    recommend: bool = Query(
        RECOMMEND_ENABLED, description="Include this product's cross-sell/upsell."
    ),
):
    """Product detail view: one product plus recommendations anchored on it."""
    p = get_product(catalog_index)
    if p is None:
        raise HTTPException(status_code=404, detail="Product not found.")

    recommendations = {"cross_sell": [], "upsell": [], "cross_sell_source": None, "errors": []}
    if recommend:
        # Opened from every card (the Oct 27 "Related items" button): computed once per
        # product, query and setting, so a second open shows the same suggestions with
        # no LLM call. Keyed on the shopper's query as sent (case and spacing folded as
        # the guard reads them) and on what the guard made of it; the cross-sell call
        # itself gets the sanitised text. Clean results only: a cross-sell fallback, a
        # refused reply and a reply that lost the memo race are asked again next open
        rec_key = page_cache.make_recommendations_key(
            catalog_index, title_key(p.get("Product_title")), query,
            (_llm_identity(), cfg.RECOMMEND_MAX, cfg.RECOMMEND_USE_LLM,
             cfg.QUERY_GUARD_ENABLED),
        )
        cached = page_cache.get_recommendations(rec_key)
        if cached is not None:
            recommendations = cached
        else:
            # Sanitised as /search sends it; a query the guard refuses anchors on the product
            llm_query = query_guard.sanitize_query(query).clean if query else ""
            with memo_outcome() as outcome:
                recommendations = build_recommendations(
                    llm_query or p.get("Product_title", ""), [p], page=1
                )
            if (not _blocking_errors(recommendations.get("errors") or [])
                    and _recs_pinnable(outcome)):
                page_cache.put_recommendations(rec_key, recommendations)

    # errors mirrored top-level like /search and /chat: a cross-sell fallback is visible
    return {
        "product": p,
        "recommendations": recommendations,
        "cross_sell_source": recommendations.get("cross_sell_source"),
        "errors": list(recommendations.get("errors") or []),
    }


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
        if len(t) < 2:
            continue
        variants = {t}
        if t.endswith("s"):
            variants.add(t[:-1])
        else:
            variants.add(t + "s")
        for v in variants:
            # "s" -> "" would make \b\b match EVERY product and wipe the results
            if len(v) < 2:
                continue
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
        page_errors: list = []
        for pg in range(1, 6):  # cap: 5 pages of the pool is plenty to backfill
            page_out = _run_search(payload.last_search_query, page=pg)
            if first_page is None:
                first_page = page_out
            kept.extend(
                r for r in page_out.get("results", [])
                if not _matches_exclusion(r, all_excl)
            )
            page_errors.extend(page_out.get("errors") or [])
            if _blocking_errors(page_out.get("errors") or []):
                # Degraded runs aren't cached, so every further page would re-run
                # translate + rerank: stop and keep what we have
                break
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
            "errors": errors + page_errors,
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
