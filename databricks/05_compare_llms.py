"""Step 5: compare Databricks FM API models on rerank + intent generation.

Rerank is ON, so each run exercises BOTH LLM stages: query expansion
(intent generation) and reranking-with-reasons. The retrieval layer is held
constant -- same Vector Search index, same gte-large-en embeddings -- so any
difference is attributable to the LLM.

Cost: 2 LLM calls per query x 18 queries = 36 calls per model.

Model names are Databricks SERVING ENDPOINT names, not vendor model ids. Run
list_models() first -- do not guess them.

Usage in a notebook:
    exec(open("databricks/05_compare_llms.py").read())
    list_models()                       # see what this workspace actually has
    run(["databricks-gpt-oss-120b", "databricks-gpt-oss-20b"])
"""

import sys

sys.path.insert(0, "databricks")
import vs_shim  # noqa: E402


def list_models():
    """Print the serving endpoints available in THIS workspace."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    names = []
    for e in w.serving_endpoints.list():
        state = getattr(getattr(e, "state", None), "ready", None)
        print(f"  {e.name:44s} {state or ''}")
        names.append(e.name)
    print(f"\n{len(names)} endpoints. Chat models are the ones to pass to run().")
    return names


def run(models):
    """Compare `models` on the full pipeline (intent generation + rerank)."""
    from pyspark.sql import SparkSession

    import eval.compare_llms as cmp

    spark = SparkSession.builder.getOrCreate()
    vs_shim.patch_eval_harness(spark)

    # availability() checks OPENAI_API_KEY / OPENAI_BASE_URL, both of which point
    # at this workspace -- so every Databricks endpoint reads as "available".
    # The real check is whether the endpoint name exists; a bad name raises and
    # compare_llms records it under `failed`.
    print("\n=== LLM comparison on Databricks (rerank ON) ===")
    print("Retrieval is identical across runs; differences are the LLM alone.\n")

    ran, failed = [], []
    for name in models:
        print(f"--- {name} ---")
        try:
            out_path, elapsed = cmp.run_for_model(name)
            print(f"completed in {elapsed:.1f}s -> {out_path}\n")
            ran.append((name, out_path))
        except Exception as e:  # noqa: BLE001 -- report and continue to next model
            print(f"FAILED {name}: {e!r}\n")
            failed.append((name, repr(e)))

    print("=== Summary ===")
    for n, p in ran:
        print(f"  ran    {n:44s} -> {p.name}")
    for n, e in failed:
        print(f"  failed {n:44s} {e[:80]}")
    print(
        "\nIMPORTANT: check each CSV's fell_back column before comparing. A model "
        "whose calls errored falls back to embedding order and will score like "
        "the retrieval-only baseline, not like a working reranker."
    )
    return ran, failed
