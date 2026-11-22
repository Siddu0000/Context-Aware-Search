"""Compare embedding models on the same eval set, retrieval only.

Local backend only: a Databricks index holds the vectors databricks/02 stored, so every
arm would read the same gte-large-en index under another model's name. SEARCH_TEXT_COMPACT
is off for the run: the compact text exists for the model it was built with only, so with
it on that arm would embed other text than the rest.
"""

import argparse
import logging
import time
from statistics import mean

import app.config as cfg
import app.embeddings as embeddings_module
from app import llm_memo
from app.embeddings import Embedder
from eval.run_eval import evaluate

logging.basicConfig(level=logging.WARNING)


def run_for_model(model_name: str):
    """Swap the default embedder and run eval (rerank disabled)."""
    embeddings_module._default_embedder = Embedder(model_name)

    from app import search as search_module

    # clear the cached df/embeddings so this model actually re-encodes
    search_module._df = None
    search_module._embeddings = None
    search_module._csv_hash_of = None   # it holds the old frame too
    search_module._loaded_fields = ()

    t0 = time.perf_counter()
    out_path = evaluate(
        rerank_on=False,
        translate_on=False,
        tag=f"embed_{model_name.replace('/', '_')}",
    )
    return out_path, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "thenlper/gte-small",
            "intfloat/e5-small-v2",
        ],
        help="Embedding model names. text-embedding-3-* triggers the OpenAI path.",
    )
    args = p.parse_args()
    if (cfg.SEARCH_BACKEND or "local").lower() != "local":
        raise SystemExit("compare_embeddings needs SEARCH_BACKEND=local: on Databricks every "
                         "arm reads the one gte-large-en index.")
    # No LLM call today (translate_on=False), but never a replay if that changes
    llm_memo.disable_for_eval()
    from app import search as search_module

    compact, cfg.SEARCH_TEXT_COMPACT = getattr(cfg, "SEARCH_TEXT_COMPACT", False), False
    try:
        print("\n=== Embedding-model comparison ===")
        for model in args.models:
            print(f"\n--- {model} ---")
            try:
                out_path, elapsed = run_for_model(model)
                print(f"completed in {elapsed:.1f}s ({search_module.search_text_mode()} text), "
                      f"results -> {out_path}")
            except Exception as e:
                print(f"FAILED for {model}: {e!r}")
    finally:
        cfg.SEARCH_TEXT_COMPACT = compact


if __name__ == "__main__":
    main()
