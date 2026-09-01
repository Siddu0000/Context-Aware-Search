"""Compare embedding models on the same eval set, retrieval only."""

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
    embeddings_module._default_embedder = Embedder(model_name)

    from app import search as search_module

    # clear the cached df/embeddings so this model actually re-encodes
    search_module._df = None
    search_module._embeddings = None
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
