"""Step 6: compare intent-generation STRATEGIES on Databricks.

Rerank is OFF, so differences come purely from how the query is translated
into search intents -- this isolates intent generation from reranking, which
05 measures together.

Modes (app/translator.py):
  query_expansion -- N literal product-search intents. The production default.
  hyde            -- generate a hypothetical product listing, search with that.
  hybrid          -- both, concatenated.

Locally (Groq, gte-small, 2026-06-10) query_expansion won decisively:
P@1 1.000 / NDCG 0.904 vs HyDE 0.750 / 0.716 -- HyDE drifts lexically from
short Amazon titles. Worth re-checking here because gte-large-en is a stronger
embedding model and may tolerate HyDE's verbosity better; that result was
measured on a model we no longer use.

Cost: 1 LLM call per query x 18 queries = 18 calls per mode.

Usage in a notebook:
    exec(open("databricks/06_compare_translators.py").read())
    run()                                    # all three modes
    run(["query_expansion", "hyde"])         # or a subset
"""

import sys

sys.path.insert(0, "databricks")
import vs_shim  # noqa: E402


def run(modes=None):
    """Compare translator modes with retrieval and reranking held constant."""
    from pyspark.sql import SparkSession

    import eval.compare_translators as cmp

    modes = modes or ["query_expansion", "hyde", "hybrid"]

    spark = SparkSession.builder.getOrCreate()
    vs_shim.patch_eval_harness(spark)

    print("\n=== Intent-generation comparison on Databricks (rerank OFF) ===")
    print("Same index, same embeddings, same LLM -- only the strategy differs.\n")

    ran, failed = [], []
    for mode in modes:
        print(f"--- mode={mode} ---")
        try:
            out_path, elapsed = cmp.run_for_mode(mode)
            print(f"completed in {elapsed:.1f}s -> {out_path}\n")
            ran.append((mode, out_path))
        except Exception as e:  # noqa: BLE001 -- report and continue to next mode
            print(f"FAILED mode={mode}: {e!r}\n")
            failed.append((mode, repr(e)))

    print("=== Summary ===")
    for m, p in ran:
        print(f"  ran    {m:20s} -> {p.name}")
    for m, e in failed:
        print(f"  failed {m:20s} {e[:80]}")
    print(
        "\nCheck fell_back before comparing: a mode whose LLM call errored "
        "degrades to the raw query and will look identical across modes."
    )
    return ran, failed
