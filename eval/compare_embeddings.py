"""Compare embedding backends on the same eval set.

Runs the eval pipeline once per model, swapping the global embedder. Pure
retrieval comparison — reranker is disabled so we isolate the effect of the
embedding model.

Default contenders:
  - all-MiniLM-L6-v2          (384-d, fast, free)
  - all-mpnet-base-v2         (768-d, stronger, free, slower)
  - text-embedding-3-small    (1536-d, OpenAI, requires OPENAI_API_KEY)

Usage:
    python -m eval.compare_embeddings
    python -m eval.compare_embeddings --models all-MiniLM-L6-v2 all-mpnet-base-v2

The function caches embeddings under a key that includes the model name,
so each model only re-embeds the first time you run it.
"""

import argparse
import logging
import time
from statistics import mean

import app.embeddings as embeddings_module
from app.embeddings import Embedder
from eval.run_eval import evaluate

logging.basicConfig(level=logging.WARNING)


def run_for_model(model_name: str):
    """Swap the default embedder and run eval (rerank disabled)."""
    # Forcibly reset the default embedder + clear cached search index so the
    # next load_index() rebuilds for this model.
    embeddings_module._default_embedder = Embedder(model_name)

    # Reset the search module's loaded state so it rebuilds the index.
    from app import search as search_module

    search_module._df = None
    search_module._embeddings = None
    search_module._loaded_fields = ()

    t0 = time.perf_counter()
    out_path = evaluate(rerank_on=False, tag=f"embed_{model_name.replace('/', '_')}")
    return out_path, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        help="Embedding model names. text-embedding-3-* triggers the OpenAI path.",
    )
    args = p.parse_args()

    print("\n=== Embedding-model comparison ===")
    for model in args.models:
        print(f"\n--- {model} ---")
        try:
            out_path, elapsed = run_for_model(model)
            print(f"completed in {elapsed:.1f}s, results -> {out_path}")
        except Exception as e:
            print(f"FAILED for {model}: {e!r}")


if __name__ == "__main__":
    main()
