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

PRODUCTS_CSV = DATA_DIR / "products.csv"
EVAL_QUERIES_JSON = DATA_DIR / "eval_queries.json"
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

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
