"""Compare embedding backends on the same eval set.

Runs the eval pipeline once per model, swapping the global embedder. Pure
retrieval comparison — reranker is disabled so we isolate the embedding model.

Contenders (production default stays all-MiniLM-L6-v2 until this says otherwise):
  - all-MiniLM-L6-v2        384-d, fast, free, CPU. Current production model.
  - all-mpnet-base-v2       768-d, stronger, free, slower, CPU.
  - BAAI/bge-m3             1024-d, MIT, multilingual; strong RAG baseline.
                            Runs on CPU via sentence-transformers (no installer),
                            ~2GB download, noticeably slower than MiniLM.
  - Qwen/Qwen3-Embedding-8B top of the open MTEB leaderboard, but an 8B model:
                            ~16GB RAM and very slow to encode 60K rows on CPU.
                            Practical only on a GPU/cloud box — or swap to the
                            lighter Qwen/Qwen3-Embedding-0.6B locally. Listed
                            here so it's in the lineup; expect to run it elsewhere.

All run through sentence-transformers (pip + HF download, no system installer).
Anything starting with 'text-embedding-' would instead use the OpenAI path.

Usage:
    python -m eval.compare_embeddings
    python -m eval.compare_embeddings --models all-MiniLM-L6-v2 BAAI/bge-m3

Embeddings are cached per model name, so each model only re-embeds once.
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
    embeddings_module._default_embedder = Embedder(model_name)

    from app import search as search_module

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
            "all-mpnet-base-v2",
            "BAAI/bge-m3",
            "Qwen/Qwen3-Embedding-8B",
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
