"""Compare LLMs on the same eval set, rerank ON."""

import argparse
import importlib.util
import logging
import time

import app.config as config_module
import app.llm_client as llm_client_module
from eval.run_eval import evaluate

logging.basicConfig(level=logging.WARNING)

BUILTIN = {"gemini", "openai", "anthropic"}


def availability(name: str) -> tuple[bool, str]:
    """Pre-flight a model before spending LLM calls; returns (ok, skip_reason)."""
    if name == "gemini":
        if not config_module.GOOGLE_API_KEYS:
            return False, "no GOOGLE_API_KEY in .env"
        return True, ""
    if name == "anthropic":
        if importlib.util.find_spec("anthropic") is None:
            return False, "anthropic SDK not installed (`pip install anthropic`)"
        if not config_module.ANTHROPIC_API_KEY:
            return False, "no ANTHROPIC_API_KEY in .env"
        return True, ""
    if importlib.util.find_spec("openai") is None:
        return False, "openai SDK not installed (`pip install openai`)"
    if not config_module.OPENAI_API_KEY:
        return False, "no OPENAI_API_KEY in .env"
    if name != "openai" and not config_module.OPENAI_BASE_URL:
        return False, "set OPENAI_BASE_URL to an OpenAI-compatible endpoint (e.g. Groq)"
    return True, ""


def run_for_model(name: str):
    """Run the eval once for one model, restoring config afterwards."""
    orig_provider = config_module.LLM_PROVIDER
    orig_openai_model = config_module.OPENAI_MODEL

    if name in BUILTIN:
        config_module.LLM_PROVIDER = name
    else:
        config_module.LLM_PROVIDER = "openai"
        config_module.OPENAI_MODEL = name

    llm_client_module._singleton = None
    llm_client_module._singleton_provider = None
    try:
        t0 = time.perf_counter()
        out_path = evaluate(rerank_on=True, tag=f"llm_{name.replace('/', '_')}")
        return out_path, time.perf_counter() - t0
    finally:
        config_module.LLM_PROVIDER = orig_provider
        config_module.OPENAI_MODEL = orig_openai_model
        llm_client_module._singleton = None
        llm_client_module._singleton_provider = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        help="Built-ins (gemini/openai/anthropic) or any OpenAI-compatible model id.",
    )
    args = parser.parse_args()

    print("\n=== LLM comparison (rerank ON; same eval set) ===\n")
    ran, skipped, failed = [], [], []
    for name in args.models:
        print(f"--- {name} ---")
        ok, reason = availability(name)
        if not ok:
            print(f"SKIP {name}: {reason}\n")
            skipped.append((name, reason))
            continue
        try:
            out_path, elapsed = run_for_model(name)
            print(f"completed in {elapsed:.1f}s, results -> {out_path}\n")
            ran.append((name, out_path))
        except Exception as e:
            print(f"FAILED for {name}: {e!r}\n")
            failed.append((name, repr(e)))

    print("=== Summary ===")
    for n, path in ran:
        print(f"  ran     {n:28s} -> {path.name}")
    for n, reason in skipped:
        print(f"  skipped {n:28s} ({reason})")
    for n, err in failed:
        print(f"  failed  {n:28s} ({err})")
    if not ran:
        print("\n  No models ran. Add keys/SDKs above, then re-run.")
    elif len(ran) == 1:
        print(f"\n  Only '{ran[0][0]}' ran — add another model for a side-by-side.")


if __name__ == "__main__":
    main()
