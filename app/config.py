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
# When true: temperature=0 + fixed seed + aggressive caching.
# Set false for "creative" mode; default true so demos and evals reproduce.
DETERMINISTIC = os.getenv("DETERMINISTIC", "true").lower() in {"1", "true", "yes"}
LLM_SEED = int(os.getenv("LLM_SEED", "42"))

# --- Networking ----------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
