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

# --- LLM -----------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")

# --- Embedding model -----------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Retrieval params ----------------------------------------------------
NUM_INTENTS = int(os.getenv("NUM_INTENTS", "3"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "30"))  # per intent
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))

# --- Reranker toggle -----------------------------------------------------
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in {"1", "true", "yes"}

# --- Networking ----------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
