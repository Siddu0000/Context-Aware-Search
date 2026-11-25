"""Compare which catalog fields are concatenated into each product's search_text.

    python -m eval.compare_search_text [--only all_fields title_only ...] [--translate]

Field SELECTION, not chunking: each variant still embeds one string per product (one
vector each); it only changes which columns go into that string.

Local backend only: a Databricks index holds the one search_text databricks/01 built, so
every variant would read the same vectors. SEARCH_TEXT_COMPACT is turned off for the run
(the compact text exists for the app's own field list only; mixing it into one arm would
compare two things at once).

Retrieval only by default (no translator, no rerank), so the variants differ by the
embedded text alone. --translate translates each eval query ONCE and gives every variant
the same intents: a fresh sample per variant would mix translator noise (+-0.4 P@10 per
query on a small model: .claude/rules/eval.md) into the field comparison.

COST: every variant but the app's own text (all_fields, run first) is a new embedding
cache key, so it encodes the whole catalog once on CPU (hours each on the enriched
catalog; cached in .cache/ after). Use --only to pick variants.
"""

import argparse
import json
import logging
import time
from typing import Dict, List, Tuple

import app.config as cfg
from app import llm_client, llm_memo
from app import search as search_module
from eval import run_eval
from eval.run_eval import evaluate

logging.basicConfig(level=logging.WARNING)


VARIANTS: dict[str, Tuple[str, ...]] = {
    # The app's own text, first (its cache already exists): equal to DEFAULT_SEARCH_FIELDS,
    # so effective_fields() adds the enriched tags the catalog has (gender, use_case,
    # dietary, specs, ...). Every other variant is used as given (an ablation).
    "all_fields": search_module.DEFAULT_SEARCH_FIELDS,
    "title_only": ("Product_title",),
    "title_plus_desc": ("Product_title", "prod_description"),
    "title_color": ("Product_title", "color"),
    "title_attrs": ("Product_title", "color", "material", "occasion"),
}


def _release_index() -> None:
    """Drop the loaded frame and vectors before the next variant loads its own, so two
    catalogs are never in memory at once (load_index reads the new frame first)."""
    search_module._df = None
    search_module._embeddings = None
    search_module._csv_hash_of = None


def run_for_variant(name: str, fields: Tuple[str, ...], translate_on: bool = False):
    """Run the eval on an index whose search_text embeds exactly `fields`.

    The fields are PASSED to evaluate(). Until 2026-11-20 this rebound
    search_module.DEFAULT_SEARCH_FIELDS and called evaluate() without them; but
    evaluate() and load_index() bind that default when they are defined, so the rebind
    never reached them: on the base catalog (2026-06-10) all five variants embedded the
    same 7 fields; on the enriched catalog the four ablations embedded the 7 base fields
    (the rebind made effective_fields() drop the enriched tags) and all_fields the app's
    12."""
    _release_index()
    t0 = time.perf_counter()
    tag = f"fields_{name}" + ("_shared_intents" if translate_on else "")
    out_path = evaluate(rerank_on=False, tag=tag, fields=fields, translate_on=translate_on)
    return out_path, time.perf_counter() - t0


def _shared_translations() -> Tuple[Dict[str, Tuple[List[str], list]], dict]:
    """Each eval query translated once: ({query: (intents, the stage errors of that call)},
    the provider-reported usage of those calls). The variant CSVs record 0 LLM calls --
    the variants replay these intents -- so this usage is the comparison's whole cost."""
    with open(run_eval.EVAL_QUERIES_JSON, encoding="utf-8") as f:   # the set _evaluate reads
        queries = [q["query"] for q in json.load(f)]
    out = {}
    with llm_client.track_usage() as usage:
        for q in queries:
            errors: list = []
            out[q] = (list(run_eval.translate_query(q, errors=errors)), errors)
    return out, dict(usage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=list(VARIANTS), help="variants to run")
    parser.add_argument("--translate", action="store_true",
                        help="translate each query once and share the intents across variants")
    args = parser.parse_args(argv)
    if (cfg.SEARCH_BACKEND or "local").lower() != "local":
        raise SystemExit("compare_search_text needs SEARCH_BACKEND=local: a Databricks index "
                         "embeds one fixed search_text, so every variant would read it.")
    # No memo: a replayed reply is not a sample of the model (evaluate() does this too)
    llm_memo.disable_for_eval()
    compact, cfg.SEARCH_TEXT_COMPACT = getattr(cfg, "SEARCH_TEXT_COMPACT", False), False
    original_translate = run_eval.translate_query
    try:
        if args.translate:
            shared, usage = _shared_translations()
            degraded = sum(1 for _, errors in shared.values() if errors)
            print(f"translated {len(shared)} queries once ({degraded} with a stage error; "
                  f"{usage['calls']} LLM calls, {usage['total_tokens']:,} tokens, "
                  f"{usage['reasoning_tokens']:,} reasoning); every variant gets these "
                  "intents, so its CSV records 0 LLM calls")

            def replay(query, errors=None):
                intents, errs = shared[query]
                if errors is not None:
                    errors.extend(errs)     # fell_back stays truthful in every variant CSV
                return list(intents)
            run_eval.translate_query = replay
        print("\n=== Search-text field comparison ===")
        for name in args.only or list(VARIANTS):
            fields = VARIANTS[name]
            print(f"\n--- {name}: {fields} ---")
            try:
                out_path, elapsed = run_for_variant(name, fields, translate_on=args.translate)
                print(f"completed in {elapsed:.1f}s, embedded {search_module.embedded_fields()}"
                      f" ({search_module.search_text_mode()} text), results -> {out_path}")
            except Exception as e:
                print(f"FAILED for {name}: {e!r}")
    finally:
        run_eval.translate_query = original_translate
        cfg.SEARCH_TEXT_COMPACT = compact


if __name__ == "__main__":
    main()
