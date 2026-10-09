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
    d.mkdir(exist_ok=True)

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
GROQ_REASONING_FORMAT = os.getenv("GROQ_REASONING_FORMAT", "hidden")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

TRANSLATOR_MODE = os.getenv("TRANSLATOR_MODE", "query_expansion").lower()

# gte-small won both eval rounds; switching invalidates the on-disk embedding cache
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-small")

NUM_INTENTS = int(os.getenv("NUM_INTENTS", "3"))
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
SPONSORED_MIN_RERANK_SCORE = float(os.getenv("SPONSORED_MIN_RERANK_SCORE", "40"))

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
# "vector": embed the query here and send query_vector -- required by the
# self-managed index 02 builds. "text": send query_text, only for an index built
# with MANAGED embeddings (the pre-2026-10-09 index).
VS_QUERY_MODE = os.getenv("VS_QUERY_MODE", "vector").lower()
# Catalog source when Spark is unavailable (Databricks Apps), and the file
# databricks/01 loads. Row position must equal catalog_index -- guaranteed by
# the dense row_number() in databricks/01.
VS_CATALOG_CSV = os.getenv(
    "VS_CATALOG_CSV", f"/Volumes/{CAS_CATALOG}/{CAS_SCHEMA}/raw/products_enriched.csv"
)

# Experiment tracking. Opt-in: needs MLFLOW_ENABLED=true AND mlflow installed;
# otherwise every tracking call is a no-op, so local dev never needs mlflow.
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "false").lower() in {"1", "true", "yes"}
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")  # e.g. /Users/<you>/cas-evals
