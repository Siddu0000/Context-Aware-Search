"""Point the eval harness at Databricks through the app's OWN search backend.

An earlier version monkey-patched eval.run_eval with a separate retrieval shim.
That tested code the app never runs, and it returned only 11 index columns --
harmless for the eval (which compares titles), but it would have blanked product
images and silently broken sponsored-ad matching (keyed by parent_asin) in the
app. app/search.py now has a native SEARCH_BACKEND=databricks path, so the eval
exercises exactly the code the deployed app will run.

Shared by 04 (retrieval eval), 05 (LLM comparison), 06 (translator modes) and
07 (threshold calibration); 08 (LLM smoke) uses only llm_model(). Named
without a digit prefix so it imports.

Do NOT add databricks/__init__.py: this folder merges with the pip `databricks`
namespace package, and a regular package here would shadow databricks.sdk and
databricks.vector_search.
"""


def llm_model():
    """The model the configured provider will call, read LIVE from cfg."""
    import app.config as cfg

    return {
        "openai": cfg.OPENAI_MODEL,
        "gemini": cfg.GEMINI_MODEL,
        "anthropic": cfg.ANTHROPIC_MODEL,
    }.get(cfg.LLM_PROVIDER, "?")


def patch_eval_harness(spark, verbose=True, llm=None):
    """Switch app.search to Vector Search and send eval CSVs to the UC Volume.

    `llm` overrides the header's llm line for callers where cfg's model is not
    what will run (05 swaps models per run; its control() uses none).
    """
    from pathlib import Path

    import app.config as cfg
    import app.llm_client as llm_client
    import eval.run_eval as run_eval
    from app import search, tracking, vector_search

    # Belt-and-braces: get_llm_client() already rebuilds on a provider/model/base-URL
    # change in cfg. An os.environ edit is NOT picked up (cfg reads env once): restartPython
    llm_client._singleton = None

    cfg.SEARCH_BACKEND = "databricks"
    vector_search.set_spark(spark)
    search.load_index()           # raises loudly if catalog_index is not dense

    # Read the env NOW: a notebook usually imported run_eval before setting it
    run_eval.EVAL_QUERIES_JSON = cfg.eval_queries_path()

    out = Path(f"/Volumes/{cfg.CAS_CATALOG}/{cfg.CAS_SCHEMA}/evals")
    out.mkdir(parents=True, exist_ok=True)
    run_eval.EVAL_RESULTS_DIR = out

    tracked = tracking.enable_autolog()
    if verbose:
        print(f"catalog : {len(search.get_dataframe()):,} rows")
        print(f"index   : {cfg.VS_INDEX} ({cfg.VS_EMBEDDING_MODEL})")
        print(f"results : {out}")
        print(f"eval set: {run_eval.EVAL_QUERIES_JSON.name}")
        print(f"llm     : {llm or f'{cfg.LLM_PROVIDER} / {llm_model()}'}")
        print(f"mlflow  : {'ON -> ' + str(cfg.MLFLOW_EXPERIMENT) if tracked else 'off'}")
    return vector_search.get_index()
