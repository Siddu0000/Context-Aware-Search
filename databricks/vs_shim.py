"""Point the eval harness at Databricks through the app's OWN search backend.

An earlier version monkey-patched eval.run_eval with a separate retrieval shim.
That tested code the app never runs, and it returned only 11 index columns --
harmless for the eval (which compares titles), but it would have blanked product
images and silently broken sponsored-ad matching (keyed by parent_asin) in the
app. app/search.py now has a native SEARCH_BACKEND=databricks path, so the eval
exercises exactly the code the deployed app will run.

Shared by 04 (retrieval eval), 05 (LLM comparison), 06 (translator modes) and
07 (threshold calibration). Named without a digit prefix so it imports.

Do NOT add databricks/__init__.py: this folder merges with the pip `databricks`
namespace package, and a regular package here would shadow databricks.sdk and
databricks.vector_search.
"""


def patch_eval_harness(spark, verbose=True):
    """Switch app.search to Vector Search and send eval CSVs to the UC Volume."""
    from pathlib import Path

    import app.config as cfg
    import eval.run_eval as run_eval
    from app import search, tracking, vector_search

    cfg.SEARCH_BACKEND = "databricks"
    vector_search.set_spark(spark)
    search.load_index()           # raises loudly if catalog_index is not dense

    out = Path(f"/Volumes/{cfg.CAS_CATALOG}/{cfg.CAS_SCHEMA}/evals")
    out.mkdir(parents=True, exist_ok=True)
    run_eval.EVAL_RESULTS_DIR = out

    tracked = tracking.enable_autolog()
    if verbose:
        print(f"catalog : {len(search.get_dataframe()):,} rows")
        print(f"index   : {cfg.VS_INDEX} ({cfg.VS_EMBEDDING_MODEL})")
        print(f"results : {out}")
        print(f"mlflow  : {'ON -> ' + str(cfg.MLFLOW_EXPERIMENT) if tracked else 'off'}")
    return vector_search.get_index()
