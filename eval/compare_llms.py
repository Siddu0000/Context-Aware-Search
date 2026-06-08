"""Compare LLM providers on the same eval set.

Runs the full eval (with rerank ON, since this is where the LLM is doing
the most work) once per configured provider. Each provider needs the
matching API key in .env:

  gemini      -> GOOGLE_API_KEY
  openai      -> OPENAI_API_KEY  (`pip install openai` first)
  anthropic   -> ANTHROPIC_API_KEY  (`pip install anthropic` first)

Outputs side-by-side CSVs under eval_results/llm_<provider>_<timestamp>.csv.
Mean lines printed at the end let you eyeball precision/latency tradeoff.

Usage:
    python -m eval.compare_llms
    python -m eval.compare_llms --providers gemini openai
"""

import argparse
import logging
import time

import app.config as config_module
import app.llm_client as llm_client_module
from eval.run_eval import evaluate

logging.basicConfig(level=logging.WARNING)


def run_for_provider(provider: str):
    """Swap the LLM provider for this eval run."""
    original = config_module.LLM_PROVIDER
    config_module.LLM_PROVIDER = provider
    # Reset the cached singleton so the next get_llm_client() picks up the
    # new provider.
    llm_client_module._singleton = None
    llm_client_module._singleton_provider = None

    try:
        t0 = time.perf_counter()
        out_path = evaluate(rerank_on=True, tag=f"llm_{provider}")
        elapsed = time.perf_counter() - t0
        return out_path, elapsed
    finally:
        config_module.LLM_PROVIDER = original
        llm_client_module._singleton = None
        llm_client_module._singleton_provider = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["gemini", "openai", "anthropic"],
        help="LLM providers to compare.",
    )
    args = parser.parse_args()

    print("\n=== LLM-provider comparison ===")
    print("Rerank ON. Each provider sees the same eval set.")
    print()

    for provider in args.providers:
        print(f"--- provider={provider} ---")
        try:
            out_path, elapsed = run_for_provider(provider)
            print(f"completed in {elapsed:.1f}s, results -> {out_path}\n")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED for {provider}: {e!r}\n")


if __name__ == "__main__":
    main()
