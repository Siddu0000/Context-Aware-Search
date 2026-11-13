"""Centralized configuration: every env knob is read here exactly once."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"

for d in (CACHE_DIR, LOGS_DIR, EVAL_RESULTS_DIR):
    try:
        d.mkdir(exist_ok=True)
    except OSError:
        # A Databricks App's source folder may be read-only. Only /feedback writes
        # (to LOGS_DIR) on the serving path, and it tolerates a missing dir; the
        # rest are local-dev caches and eval output, so the import must not fail
        pass

# The catalog the app serves. The cleaned + enriched copy (scripts/enrich_catalog.py)
# is the default whenever it exists: on the context eval set it lifted P@10 0.408 ->
# 0.521 with query expansion (3 runs each, 2026-10-09). Set PRODUCTS_CSV=data/products.csv
# for the raw catalog. Relative paths resolve from the project root.
_enriched = PROJECT_ROOT / "data" / "products_enriched.csv"
_products = Path(os.getenv(
    "PRODUCTS_CSV", "data/products_enriched.csv" if _enriched.exists() else "data/products.csv"))
PRODUCTS_CSV = _products if _products.is_absolute() else PROJECT_ROOT / _products


def eval_queries_path(value: str | None = None) -> Path:
    """The eval query set: EVAL_QUERIES_JSON env (relative = repo root), else the default.

    data/eval_queries.json is keyword-only and SATURATES (raw retrieval ~0.95);
    data/eval_queries_context.json needs the product to be INFERRED.
    """
    value = value if value is not None else os.getenv("EVAL_QUERIES_JSON")
    if not value:
        return DATA_DIR / "eval_queries.json"
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


EVAL_QUERIES_JSON = eval_queries_path()
FEEDBACK_LOG = LOGS_DIR / "feedback.jsonl"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
# How the OpenAI-compatible backend authenticates. "key" (default): the static
# OPENAI_API_KEY -- Groq locally, the token notebook Cell 2 mints on Databricks.
# "databricks": a fresh workspace token from databricks-sdk before EVERY call. A
# Databricks App runs as a service principal whose OAuth tokens last one hour, so
# a key fixed at start-up would 401 an hour into a demo and every LLM stage would
# silently fall back. Needs openai>=1.106 (callable api_key).
OPENAI_AUTH = os.getenv("OPENAI_AUTH", "key").lower()
GROQ_REASONING_FORMAT = os.getenv("GROQ_REASONING_FORMAT", "hidden")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

TRANSLATOR_MODE = os.getenv("TRANSLATOR_MODE", "query_expansion").lower()

# gte-small won both eval rounds; switching invalidates the on-disk embedding cache
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-small")

NUM_INTENTS = int(os.getenv("NUM_INTENTS", "3"))

# Deterministic prompt-injection guard (app/query_guard.py). The prompts' "the query
# is DATA" sentence did not stop Qwen3-Next obeying an injected payload (databricks/08,
# 2026-10-12, 3 of 3), so the query is cut at its first instruction marker before any
# LLM sees it and every LLM-written phrase is screened before it is searched or shown.
# A query with no instruction signal reaches the LLM byte-identical (normalisation is
# for detection only), so eval prompts are the same either way; a clipped or screened
# model phrase is labelled translate:partial, so the guard never changes a run silently.
# false = the pre-2026-10-12 behaviour: the in-process A/B switch and the REVERT.
QUERY_GUARD_ENABLED = os.getenv("QUERY_GUARD_ENABLED", "true").lower() in {"1", "true", "yes"}
# Most intents one query may fan out to: each is one retrieval call, so an injected
# "list 500 products" can't become 500 vector searches. Sized by what models WRITE,
# not by data/recipe_eval.json's 6-8 canonical ingredients: RECIPE_PROMPT asks for
# "every core ingredient", and a chicken biryani reply listed 22. The cap only applies
# while the guard is enabled; a clipped reply shows as translate:partial (over_cap).
MAX_INTENTS = int(os.getenv("MAX_INTENTS", "24"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "30"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "12"))

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in {"1", "true", "yes"}

DETERMINISTIC = os.getenv("DETERMINISTIC", "true").lower() in {"1", "true", "yes"}
LLM_SEED = int(os.getenv("LLM_SEED", "42"))

TEMPERATURE_OVERRIDE = None


def effective_temperature(requested: float) -> float:
    """Sweep override wins; otherwise the caller's tuned temperature is used as-is."""
    if TEMPERATURE_OVERRIDE is not None:
        return float(TEMPERATURE_OVERRIDE)
    return float(requested)


RATING_BOOST_WEIGHT = float(os.getenv("RATING_BOOST_WEIGHT", "0.05"))

# Absolute "no products match" floor on the retrieval score. Tuned on gte-small
# cosine; the Vector Search score scale differs -- recalibrate per backend with
# eval/calibrate_relevance.py (databricks/07) before trusting it there.
MIN_RESULT_RELEVANCE = float(os.getenv("MIN_RESULT_RELEVANCE", "0.5"))

RERANK_POOL_K = int(os.getenv("RERANK_POOL_K", "30"))

RERANK_INPUT_K = int(os.getenv("RERANK_INPUT_K", "15"))

PAGE_CACHE_SIZE = int(os.getenv("PAGE_CACHE_SIZE", "64"))

# Persistent memo of ACCEPTED LLM replies (app/llm_memo.py), 2026-10-27: "the same search
# must give the same results" (Oct 27 meeting). With a fake LLM the pipeline is
# byte-identical run to run; the LLM is not -- Groq's seed is best effort (the same
# translator prompt gave 3 different replies in 3 cold runs) and Databricks gets no seed --
# and page_cache dies with the process. On: a reply a stage accepted is replayed for the
# byte-identical request (0 provider calls), so the page repeats across restarts. A
# fallback is never stored. OFF in every eval harness (eval/*.py call
# llm_memo.disable_for_eval(): an eval must sample the model) and in databricks/08.
# false = every search asks the LLM again (the pre-2026-10-27 behaviour). Read as cfg.X at
# call time.
LLM_MEMO = os.getenv("LLM_MEMO", "true").lower() in {"1", "true", "yes"}
# Where: a local directory (relative = the project root; SQLite inside, gitignored under
# .cache/) or a UC Volume path /Volumes/<catalog>/<schema>/<volume>/... (one JSON file per
# key; the FUSE mount in a notebook, the Files API in a Databricks App, whose own disk is
# lost on restart). The App sets /Volumes/<CAS_CATALOG>/<CAS_SCHEMA>/app_state/llm_memo
# (databricks/app_bundle/app.yaml).
LLM_MEMO_DIR = os.getenv("LLM_MEMO_DIR", ".cache/llm_memo")
# Bound, least recently used out (SQLite on every new entry; a Volume store on
# `python -m app.llm_memo purge --to-limit`). A search adds ~2-3 replies of 1-3 KB.
LLM_MEMO_MAX_ENTRIES = int(os.getenv("LLM_MEMO_MAX_ENTRIES", "50000"))
# Age limit in days (0 = never expire, the default: the memo exists to repeat a page).
# The stored replies echo shopper text (intents, rerank reasons, assistant rewrites), and
# on the App they sit in a shared Volume: a limit bounds how long they are kept. An older
# entry is a miss and is deleted when read; `python -m app.llm_memo purge --expired` (or
# --to-limit) deletes the ones never read again.
LLM_MEMO_MAX_AGE_DAYS = float(os.getenv("LLM_MEMO_MAX_AGE_DAYS", "0"))

GROCERY_PER_INTENT_K = int(os.getenv("GROCERY_PER_INTENT_K", "4"))

RECOMMEND_ENABLED = os.getenv("RECOMMEND_ENABLED", "true").lower() in {"1", "true", "yes"}
RECOMMEND_MAX = int(os.getenv("RECOMMEND_MAX", "4"))
RECOMMEND_USE_LLM = os.getenv("RECOMMEND_USE_LLM", "true").lower() in {"1", "true", "yes"}

SPONSORED_ENABLED = os.getenv("SPONSORED_ENABLED", "true").lower() in {"1", "true", "yes"}
SPONSORED_CONFIG = DATA_DIR / "sponsored.json"
# A FLAT-results ad is promoted only if its raw 0-100 rerank_score is ABOVE this.
# The rerank prompt caps constraint violators at 40, so a women's dress scored 40
# on a men's-shirt query stays organic instead of jumping to position 1. Bundles
# (_promote_sponsored_options) still gate on slot membership only: owner decision.
# With TYPE_AWARE_RANKING on, main.py also reads it for the organic display band
# (`_band`) and the rejected bundle-slot test (`_slot_rejected`): moving the ad floor
# moves both.
SPONSORED_MIN_RERANK_SCORE = float(os.getenv("SPONSORED_MIN_RERANK_SCORE", "40"))

# Type-aware ordering (app/main.py, types from app/product_type.py), 2026-10-14. The demo's
# "cloths for a beach wedding in summer" led with sandals, earrings and bags although the
# catalog holds 9,218 suitable garments: Fashion has no product-type field (categ_lvl2_name
# is only a gender bucket), and score order let one intent fill the rerank window. When on,
# a flat query that NAMES a kind of product ranks that type first in the rerank window and
# on the page -- only inside a Fashion-majority pool, so "tea bags" / "ring light" are
# untouched -- the reranker is told the type, an ad must be that type, outfit/setup slots
# keep their own type, and outfit cards put garments first.
# false restores the old ordering AND the old reranker inputs from main.py: the old gender
# balancers, the score-order rerank window and display, the old flat ad gate, the old bundle
# retrieval, window and slot order, no [product_type] constraint, the translator's
# setting-only 'other' constraints kept, and no pronoun gender ('for her' is no longer a
# stated gender). It does NOT revert the word-bounded gender classifier, the setup routing
# regex, the translator prompt and reranker rubric edits, the `type=` candidate field or the
# same-type upsell filter -- those need a git revert. Read as cfg.TYPE_AWARE_RANKING at call time and part of the
# page_cache key, so flipping it in-process (eval/eval_page_types.py --mode both) is real.
TYPE_AWARE_RANKING = os.getenv("TYPE_AWARE_RANKING", "true").lower() in {"1", "true", "yes"}

# Outfit bundles ("beach wedding outfit" -> one card per garment slot: dress, sandals, hat).
# OFF since Niharika's review (2026-10-15): the combinations did not tell a coherent story,
# a bundle page shows only 3-5 products with no page 2, and one rerank window capped the
# outfit's own shoes and bag at <=40. Off, an outfit-worded query is an ordinary clothing
# search (garment intents, paginated, type-aware). Gated in the translator, the only
# producer of bundle_type, so main.py's outfit code is simply unreachable. true restores the
# outfit route AND the previous general prompt byte-identical. Read as cfg.X at call time and
# part of the page_cache key. Recipe and setup bundles are unaffected.
OUTFIT_BUNDLES_ENABLED = os.getenv("OUTFIT_BUNDLES_ENABLED", "false").lower() in {"1", "true", "yes"}

# Display band order of a flat page (app/main.py _flat_display / _relevance_groups),
# 2026-10-19 owner call 1 (Sai). The 41-query apparel battery (two grader lenses) found
# rows the reranker REJECTED (<= SPONSORED_MIN_RERANK_SCORE) average 0.32 on a 0-2 label
# while never-reranked tail rows average 1.01, yet the 2026-10-14 order showed the rejects
# first: 41 of them sat on page 1 across 13 queries. true = reranked > floor, then the
# unreranked tail, then the rejects; and the gender interleave stays inside 15-point rerank
# tiers of the top band, so a men's 55 is never lifted above a women's 92. false = the
# 2026-10-14 order (rejects before the tail, interleave across the whole band). It changes
# the CACHED pool, so it is part of the page_cache key; read as cfg.X at call time.
REJECTS_AFTER_TAIL = os.getenv("REJECTS_AFTER_TAIL", "true").lower() in {"1", "true", "yes"}

# Device asks (2026-10-27, the Oct 27 meeting: "gift for a vintage watch collector" put
# watch boxes, winders and cleaning kits on page 1 and the watches on page 2; "redo the
# electronics category"). When a flat query names a device itself -- watch, laptop,
# headphones, phone, tablet, camera or speaker (app/device_role.py), or a gift for someone
# who collects or loves one -- and the family leads the raw head with at least one real
# device in the pool: the rerank window seats the device first and accessories for it
# last (a gift keeps 3 seats for its add-on intents), the device leads each display band
# and every sort group, an accessory ad stays organic, an accessory never counts toward
# match_quality, the reranker gets a code-written [product_type] text and a ` | role=`
# field on its window rows. Only inside TYPE_AWARE_RANKING (its false turns this off
# too). false = the 2026-10-19 window, display, ads, banner and reranker inputs, except
# for watches: app/product_type.py types a watch or watch band `accessory`, no longer
# `jewelry` (500 Fashion rows change their `type=` field and type tier), and reads "apple
# watch bracelet" (accessory) and "dress watch" (no dresses subtype) as their titles read,
# with this knob on or off (knob off, 10 of the 15 window rows of "women's bracelet watch"
# are not the 2026-10-19 ones). That typing, the prompt and rubric lines (QE DEVICE /
# GIFT bullets, the rubric's GIFTS sentence) and the `_format_candidate` blank-value fix
# need a git revert. It changes the CACHED pool, so it is part of the page_cache key;
# read as cfg.X at call time.
DEVICE_ROLE_RANKING = os.getenv("DEVICE_ROLE_RANKING", "true").lower() in {"1", "true", "yes"}

# Colour/size variants of one product as ONE card (2026-10-19 owner call 2). Fashion rows
# sharing search.style_key (store + title minus colour, size and pack words; the catalog
# audit's strict key: 0 false merges on the audited page-1 duplicates) collapse onto the
# best-placed one, which lists them under `variants`. Graders tagged 60 duplicate cards in
# 32 of 41 battery queries ("Hubby & Wifey" tee at #1-3). Applied AFTER the cache lookup
# (filters -> sort -> collapse -> paginate), so it is NOT in the page_cache key and costs no
# LLM call. Fashion only: the key drops 1-2 digit numbers, which name models elsewhere
# ("Series 7" / "Series 8"). false = one card per row, as before.
VARIANT_COLLAPSE = os.getenv("VARIANT_COLLAPSE", "true").lower() in {"1", "true", "yes"}

# The "No exact matches" banner (/search match_quality.weak). Contract C2 (2026-10-19):
# weak = fewer than 4 rows judged above the floor, or a best score <= 55. N > 0 ALSO calls
# a page weak when fewer than N rows scored >= 71 ("strong"). Why it exists: replayed on
# the recorded apparel battery (0 LLM), C2 alone flagged only p07 of its four target pages
# (x02, p03, p07, m04: sandals at 48-55 and kids' shoes count as accepts); N = 4 flagged
# m04, p07, x02 and c07 and none of the 20 best (p03 still passes, its canvas shoes scored
# 70-92). Not an owner call yet, so 0 = C2 exactly. Applied after the cache lookup: NOT in
# the page_cache key; read as cfg.X at call time.
WEAK_MATCH_MIN_STRONG = int(os.getenv("WEAK_MATCH_MIN_STRONG", "0"))

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Retrieval backend. "local" = in-process gte-small + numpy (dev laptop);
# "databricks" = Mosaic AI Vector Search over gte-large-en vectors (databricks/02).
# Read DYNAMICALLY via cfg.SEARCH_BACKEND so notebooks/evals can switch it.
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "local").lower()
CAS_CATALOG = os.getenv("CAS_CATALOG", "dev")
CAS_SCHEMA = os.getenv("CAS_SCHEMA", "cas")
VS_ENDPOINT = os.getenv("CAS_ENDPOINT", "cas-search")
VS_INDEX = os.getenv("VS_INDEX", f"{CAS_CATALOG}.{CAS_SCHEMA}.products_index")
VS_TABLE = f"{CAS_CATALOG}.{CAS_SCHEMA}.products"
# Embeddings are computed ONCE by databricks/02 embed() and kept in Delta, so a
# rebuilt index never calls the embedding model again (an index cannot be paused;
# deleting it between runs is what stops the endpoint billing). Two tables:
#   VS_EMBEDDINGS_TABLE -- the store, keyed by (embedding_model, text_hash): only
#     new or changed product text is ever embedded again; never deleted.
#   VS_VECTORS_TABLE -- the index source: one row per catalog_index, re-derived from
#     the store whenever the catalog changes.
VS_EMBEDDINGS_TABLE = f"{CAS_CATALOG}.{CAS_SCHEMA}.products_embeddings"
VS_VECTORS_TABLE = f"{CAS_CATALOG}.{CAS_SCHEMA}.products_vectors"
# The pay-per-token serving endpoint that embeds BOTH the catalog (02) and every
# query (app/vector_search.py) -- the two must match or retrieval is meaningless.
VS_EMBEDDING_MODEL = os.getenv("VS_EMBEDDING_MODEL", "databricks-gte-large-en")
# What embed_texts CALLS when the workspace has "Enforce Unity Gateway" on: that
# setting disables every databricks-* pay-per-token endpoint ("Please use Unity
# Gateway", 2026-11-13), and the same model is then a Unity Gateway model service
# ("system.ai.gte-large-en") at AI_GATEWAY_URL/embeddings. "" (default) = call
# VS_EMBEDDING_MODEL as a pay-per-token endpoint, as before. VS_EMBEDDING_MODEL stays the
# LABEL of the stored vectors (02 resume() refuses a mismatch), so another route to the
# same model re-embeds nothing -- once the vector check shows the same vectors.
VS_EMBEDDING_SERVICE = os.getenv("VS_EMBEDDING_SERVICE", "").strip()
# The gateway's OpenAI-compatible base, https://<workspace URL>/ai-gateway/mlflow/v1.
# "" = the databricks-sdk host + that path. Wrong in a NOTEBOOK: its SDK host is the
# regional control plane (https://eastus-c3...), where /ai-gateway/ answers 404, so
# Cell 2 sets it from spark.databricks.workspaceUrl (both measured 2026-11-13). In an
# App the host is the injected DATABRICKS_HOST (assumed the workspace URL: unverified;
# the App's start-up embedding check says).
AI_GATEWAY_URL = os.getenv("AI_GATEWAY_URL", "").strip().rstrip("/")
# "vector": embed the query here and send query_vector -- required by the
# self-managed index 02 builds. "text": send query_text, only for an index built
# with MANAGED embeddings (the pre-2026-10-09 index).
VS_QUERY_MODE = os.getenv("VS_QUERY_MODE", "vector").lower()
# Catalog source when Spark is unavailable (Databricks Apps), and the file
# databricks/01 loads. Row position must equal catalog_index -- guaranteed by
# the dense row_number() in databricks/01. The App points this at the export
# databricks/09 export_catalog() writes FROM the Delta table, which carries the
# real catalog_index column, so the App hydrates exactly the rows the evals did.
VS_CATALOG_CSV = os.getenv(
    "VS_CATALOG_CSV", f"/Volumes/{CAS_CATALOG}/{CAS_SCHEMA}/raw/products_enriched.csv"
)

# The Databricks App that serves the UI (databricks/09_deploy_app.py). Lowercase
# letters, digits and hyphens only; its URL is derived from it and cannot change.
APP_NAME = os.getenv("CAS_APP_NAME", "cas-app")

# Experiment tracking. Opt-in: needs MLFLOW_ENABLED=true AND mlflow installed;
# otherwise every tracking call is a no-op, so local dev never needs mlflow.
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "false").lower() in {"1", "true", "yes"}
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")  # e.g. /Users/<you>/cas-evals
