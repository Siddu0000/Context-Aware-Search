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
import importlib.util
import logging
import time

import app.config as config_module
import app.llm_client as llm_client_module
from eval.run_eval import evaluate

logging.basicConfig(level=logging.WARNING)


def provider_availability(provider: str) -> tuple[bool, str]:
    """Check a provider can actually run BEFORE we spend any LLM calls.

    Returns (available, reason). Reason explains the skip when unavailable —
    e.g. on a Gemini-only machine, openai/anthropic skip with a one-line note
    rather than failing deep inside the eval (or burning the first few calls).
    """
    if provider == "gemini":
        if not config_module.GOOGLE_API_KEYS:
            return False, "no GOOGLE_API_KEY in .env"
        return True, ""
    if provider == "openai":
        if not config_module.OPENAI_API_KEY:
            return False, "no OPENAI_API_KEY in .env"
        if importlib.util.find_spec("openai") is None:
            return False, "openai SDK not installed (`pip install openai`)"
        return True, ""
    if provider == "anthropic":
        if not config_module.ANTHROPIC_API_KEY:
            return False, "no ANTHROPIC_API_KEY in .env"
        if importlib.util.find_spec("anthropic") is None:
            return False, "anthropic SDK not installed (`pip install anthropic`)"
        return True, ""
    return False, f"unknown provider {provider!r}"


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

    ran, skipped, failed = [], [], []
    for provider in args.providers:
        print(f"--- provider={provider} ---")
        available, reason = provider_availability(provider)
        if not available:
            print(f"SKIP {provider}: {reason}\n")
            skipped.append((provider, reason))
            continue
        try:
            out_path, elapsed = run_for_provider(provider)
            print(f"completed in {elapsed:.1f}s, results -> {out_path}\n")
            ran.append((provider, out_path))
        except Exception as e:  # noqa: BLE001
            print(f"FAILED for {provider}: {e!r}\n")
            failed.append((provider, repr(e)))

    print("=== Comparison summary ===")
    for p, path in ran:
        print(f"  ran     {p:10s} -> {path.name}")
    for p, reason in skipped:
        print(f"  skipped {p:10s} ({reason})")
    for p, err in failed:
        print(f"  failed  {p:10s} ({err})")
    if not ran:
        print("\n  No providers ran. Add keys/SDKs above, then re-run.")
    elif len(ran) == 1:
        print(
            f"\n  Only '{ran[0][0]}' ran — nothing to compare against yet. "
            "Add another provider's key + SDK to get a side-by-side."
        )


if __name__ == "__main__":
    main()
