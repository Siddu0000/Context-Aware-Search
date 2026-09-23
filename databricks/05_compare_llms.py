"""Step 5: compare Databricks FM API models on rerank + intent generation.

Rerank is ON, so each run exercises BOTH LLM stages: query expansion
(intent generation) and reranking-with-reasons. The retrieval layer is held
constant -- same Vector Search index, same gte-large-en embeddings -- so any
difference is attributable to the LLM.

Cost: 2 LLM calls per query -- 36 per model on the 18-query keyword set, 32 on
the 16-query context set. control() makes none.

Model names are Databricks SERVING ENDPOINT names, not vendor model ids. Run
list_models() first -- do not guess them.

The keyword set saturates (a no-LLM control scores the same), so compare on the
CONTEXT set, and always next to control():
    os.environ["EVAL_QUERIES_JSON"] = "data/eval_queries_context.json"

Usage in a notebook:
    exec(open("databricks/05_compare_llms.py").read())
    list_models()                       # see what this workspace actually has
    run(["databricks-qwen3-next-80b-a3b-instruct", "databricks-gpt-oss-120b"])
    control()                           # the no-LLM baseline on the same set
    os.environ.pop("EVAL_QUERIES_JSON", None)  # else every later eval runs the context set
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
    # run_for_model swaps cfg's model after this header prints, so name them all
    vs_shim.patch_eval_harness(spark, llm=f"one run each: {', '.join(models)}")

    # availability() checks OPENAI_API_KEY / OPENAI_BASE_URL, both of which point
    # at this workspace -- so every Databricks endpoint reads as "available".
    # A mistyped endpoint name does NOT raise: each call 404s inside translate
    # and rerank, which catch it, so the model is listed under `ran`. It shows as
    # run_eval's "!! N/N QUERIES RAN DEGRADED" banner, fell_back on every CSV
    # row, and MLflow tag valid=false.
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


def control():
    """The NO-LLM control (raw query, rerank off) on the current eval set.

    Without it a model comparison cannot say whether either LLM added anything.
    """
    from pyspark.sql import SparkSession

    import eval.run_eval as run_eval

    vs_shim.patch_eval_harness(SparkSession.builder.getOrCreate(),
                               llm="off (no-LLM control)")
    tag = f"databricks_no_llm_control_{run_eval.EVAL_QUERIES_JSON.stem}"
    return run_eval.evaluate(rerank_on=False, tag=tag, translate_on=False)
