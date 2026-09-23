"""Compare translator modes on the same eval set, reranker off."""

import argparse
import logging
import time

import app.config as cfg
import app.translator as translator_module
from eval.run_eval import evaluate

logging.basicConfig(level=logging.WARNING)


def run_for_mode(mode: str):
    """Set TRANSLATOR_MODE for this run and execute the eval."""
    original = translator_module.TRANSLATOR_MODE
    translator_module.TRANSLATOR_MODE = mode
    try:
        t0 = time.perf_counter()
        # The CSV has no model column, so the name must say which LLM translated
        model = {"openai": cfg.OPENAI_MODEL, "gemini": cfg.GEMINI_MODEL,
                 "anthropic": cfg.ANTHROPIC_MODEL}.get(cfg.LLM_PROVIDER, "?")
        tag = f"translator_{mode}_{model.replace('/', '_')}"
        out_path = evaluate(rerank_on=False, tag=tag)
        elapsed = time.perf_counter() - t0
        return out_path, elapsed
    finally:
        translator_module.TRANSLATOR_MODE = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["query_expansion", "hyde", "hybrid"],
        help="Translator modes to compare.",
    )
    args = parser.parse_args()

    print("\n=== Translator-strategy comparison ===")
    print("Reranker is DISABLED so differences are purely from translation.")
    for mode in args.modes:
        print(f"\n--- mode={mode} ---")
        try:
            out_path, elapsed = run_for_mode(mode)
            print(f"completed in {elapsed:.1f}s, results -> {out_path}")
        except Exception as e:
            print(f"FAILED for mode={mode}: {e!r}")


if __name__ == "__main__":
    main()
