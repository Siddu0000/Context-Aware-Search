"""Centralized configuration. All env vars are read here exactly once.

Why centralize: scattered os.getenv() calls become hard to audit. A single
config module makes it obvious what knobs exist and what their defaults are,
and lets eval scripts override them programmatically.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"

for d in (CACHE_DIR, LOGS_DIR, EVAL_RESULTS_DIR):
    d.mkdir(exist_ok=True)

PRODUCTS_CSV = DATA_DIR / "products.csv"
EVAL_QUERIES_JSON = DATA_DIR / "eval_queries.json"
FEEDBACK_LOG = LOGS_DIR / "feedback.jsonl"

# --- LLM provider --------------------------------------------------------
# Which LLM provider to use for translator + reranker.
# Options: "gemini" | "openai" | "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Gemini config -----------------------------------------------------------
# Primary key + up to 4 backup keys for automatic 429 failover.
# Niharika's suggestion: create extra free keys with personal Gmail IDs.
GOOGLE_API_KEYS = [
    k for k in [
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GOOGLE_API_KEY_2"),
        os.getenv("GOOGLE_API_KEY_3"),
        os.getenv("GOOGLE_API_KEY_4"),
        os.getenv("GOOGLE_API_KEY_5"),
    ] if k
]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-lite")

# OpenAI config -----------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Anthropic config --------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

# --- Translator strategy -------------------------------------------------
# "query_expansion": N short product-search phrases (original / Niharika's preference)
# "hyde":            1 long hypothetical product listing (classical HyDE)
# "hybrid":          1 hypothetical doc + (N-1) short phrases
#
# CHOSEN DEFAULT: query_expansion
# Backed by eval/compare_translators.py run on 2026-06-10 against the
# Amazon catalog with rerank OFF (so the comparison is purely translator):
#
#   mode             P@1   P@10  MRR   NDCG
#   query_expansion  1.000 0.875 1.000 0.904  <-- clear winner
#   hyde             0.750 0.700 0.819 0.716
#   hybrid           0.833 0.758 0.857 0.768
#
# HyDE's hypothetical-document approach drifts lexically away from
# short Amazon titles ("girls dress for kids" -> NDCG=0.000). Hybrid
# inherits HyDE's failure modes without recovering them. Keep both
# strategies in code for future re-benchmarking, but do not change
# the default without re-running compare_translators first.
TRANSLATOR_MODE = os.getenv("TRANSLATOR_MODE", "query_expansion").lower()

# --- Embedding model -----------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Retrieval params ----------------------------------------------------
NUM_INTENTS = int(os.getenv("NUM_INTENTS", "3"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "30"))  # per intent
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))

# --- Reranker toggle -----------------------------------------------------
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in {"1", "true", "yes"}

# --- Determinism ---------------------------------------------------------
# When true: temperature is forced to 0 and a fixed seed is sent. This is
# what makes repeated identical queries reproducible.
# Set false for "creative" mode (no seed, requested temperature used).
DETERMINISTIC = os.getenv("DETERMINISTIC", "true").lower() in {"1", "true", "yes"}
LLM_SEED = int(os.getenv("LLM_SEED", "42"))

# Temperature sweep override. None = normal behaviour (see effective_temperature).
# When set to a float, EVERY LLM call uses that temperature, ignoring the
# per-call value and ignoring DETERMINISTIC. Used by eval/compare_temperature.py
# to answer "why temperature=0 specifically — did we test other values?".
# Not intended for production use. Set back to None when done.
TEMPERATURE_OVERRIDE = None


def effective_temperature(requested: float) -> float:
    """Resolve the temperature a backend should actually use.

    Precedence:
      1. TEMPERATURE_OVERRIDE if set (sweep mode) — wins over everything.
      2. 0.0 if DETERMINISTIC (the production default).
      3. otherwise the caller's requested temperature.

    Read dynamically by app.llm_client so eval scripts can mutate the module
    globals at runtime and have it take effect immediately.
    """
    if TEMPERATURE_OVERRIDE is not None:
        return float(TEMPERATURE_OVERRIDE)
    return 0.0 if DETERMINISTIC else float(requested)


# --- LLM response caching ------------------------------------------------
# DISABLED in current build. The in-process LRU caches in app/cache.py are
# NOT wired into translator.py or reranker.py — every /search hits the real
# LLM. This is intentional for the early development phase: cached responses
# create uncertainty during debugging ("did the LLM actually run, or did I
# get a stale cached answer?") that distracts from real engineering work.
#
# To re-enable: uncomment the cache blocks in app/translator.py and
# app/reranker.py, then change the value below to read from the env var.
# CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
CACHE_ENABLED = False  # hard-off; ignore env var until cache is rewired

# --- Rating-aware scoring ------------------------------------------------
# How much weight the Bayesian-adjusted rating gets in the final ranking,
# relative to the LLM's semantic relevance score. Range [0, 1].
# 0.0  = pure relevance (rating ignored).
# 0.05 = relevance dominates (95%), rating is a faint tie-breaker.  <-- CURRENT
# 0.15 = rating nudges 15%.
#
# Niharika (2026-06-11): rating must be "as minimal as possible — not our
# primary filter." Lowered 0.15 -> 0.05. It now only breaks near-ties
# between products of essentially equal semantic relevance.
RATING_BOOST_WEIGHT = float(os.getenv("RATING_BOOST_WEIGHT", "0.05"))

# --- Pagination ----------------------------------------------------------
# /search returns ONE page of results. `top_k` is the page size; `page`
# (1-based) selects which slice. To keep LLM quota sane, the reranker scores a
# single deep pool ONCE (RERANK_POOL_K items) and every page is sliced from
# that one ranked pool — so paging never costs an extra LLM call. Pages that
# run past the reranked pool fall back to embedding-similarity order and are
# labelled as such in each item's `reason`.
RERANK_POOL_K = int(os.getenv("RERANK_POOL_K", "30"))

# --- Cross-sell / upsell recommendations ---------------------------------
# After organic results are ranked, an optional recommender suggests
# (a) COMPLEMENTARY items ("frequently bought together") and (b) an UPSELL
# (a higher-rated alternative in the same category). Amazon's `bought_together`
# field is empty across this dataset (verified 2026-06-12), so complements are
# produced by an LLM and then GROUNDED in the real catalog via embedding
# retrieval. Set RECOMMEND_USE_LLM=false to skip the LLM and use pure
# embedding similarity ("more like this") instead. Computed only for page 1.
RECOMMEND_ENABLED = os.getenv("RECOMMEND_ENABLED", "true").lower() in {"1", "true", "yes"}
RECOMMEND_MAX = int(os.getenv("RECOMMEND_MAX", "4"))  # max complementary items
RECOMMEND_USE_LLM = os.getenv("RECOMMEND_USE_LLM", "true").lower() in {"1", "true", "yes"}

# --- Featured / sponsored (paid-ad) prioritization -----------------------
# Sponsored products are kept STRICTLY separate from organic ranking for
# auditability (see .claude/rules/safety.md). There is no real ad inventory in
# this PoC, so placements are read from a curated config file, targeted by
# keyword/vertical, and returned under a dedicated `sponsored` response key —
# never blended into `results`. Each carries is_sponsored=true + sponsor name.
SPONSORED_ENABLED = os.getenv("SPONSORED_ENABLED", "true").lower() in {"1", "true", "yes"}
SPONSORED_MAX = int(os.getenv("SPONSORED_MAX", "2"))  # max sponsored slots
SPONSORED_CONFIG = DATA_DIR / "sponsored.json"

# --- Networking ----------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
