"""Compare core LLMs on the same eval set (rerank ON — where the LLM works hardest).

Two kinds of model name are accepted in --models:

  Built-in providers (use the model configured for that provider in .env):
    gemini      -> GOOGLE_API_KEY                       (current production)
    openai      -> OPENAI_API_KEY      (+ pip install openai)
    anthropic   -> ANTHROPIC_API_KEY   (+ pip install anthropic)

  ANY other string is treated as a model ID served through an OpenAI-COMPATIBLE
  endpoint (set OPENAI_BASE_URL + OPENAI_API_KEY). Nothing is installed locally.
  So you can compare any model your provider serves, e.g.:
    python -m eval.compare_llms --models gemini openai/gpt-oss-120b openai/gpt-oss-20b

Recommended FREE, no-credit-card endpoint — Groq (groq.com):
    OPENAI_API_KEY   = gsk_...
    OPENAI_BASE_URL  = https://api.groq.com/openai/v1
  Current Groq models (as of mid-2026): openai/gpt-oss-120b (flagship open model),
  openai/gpt-oss-20b (lighter), qwen/qwen3.6-27b. NOTE: the older
  llama-3.3-70b-versatile, llama-3.1-8b-instant, and qwen/qwen3-32b were
  deprecated on 2026-06-17 — avoid them. GPT-OSS are reasoning models; if one
  returns empty/invalid JSON under strict JSON mode, that's the cause (Groq's
  reasoning_format would need wiring in). Lower RERANK_INPUT_K if you hit 429s.

NOTE: with OPENAI_BASE_URL set, the plain `openai` entry also targets that
endpoint — so compare gemini + compatible models in one run, not real OpenAI too.

Outputs side-by-side CSVs under eval_results/llm_<name>_<timestamp>.csv.

Usage:
    python -m eval.compare_llms
    python -m eval.compare_llms --models gemini openai/gpt-oss-120b openai/gpt-oss-20b
"""

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
    """Check a model can run BEFORE spending any LLM calls.
    Returns (available, reason); reason explains a skip in one line.
    Built-ins use their own key; any other name is an OpenAI-compatible model
    ID and needs the openai SDK + OPENAI_API_KEY + OPENAI_BASE_URL."""
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
    """Run the eval once for a built-in provider or a compatible model id.
    Restores config afterwards."""
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
