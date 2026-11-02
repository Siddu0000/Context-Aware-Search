"""Point the eval harness at Databricks through the app's OWN search backend.

An earlier version monkey-patched eval.run_eval with a separate retrieval shim.
That tested code the app never runs, and it returned only 11 index columns --
harmless for the eval (which compares titles), but it would have blanked product
images and silently broken sponsored-ad matching (keyed by parent_asin) in the
app. app/search.py now has a native SEARCH_BACKEND=databricks path, so the eval
exercises exactly the code the deployed app will run.

Shared by 04 (retrieval eval), 05 (LLM comparison), 06 (translator modes),
07 (threshold calibration), 10 (page eval) and 12 (determinism check); 08 (LLM
smoke) uses only llm_model(). 00 (preflight), 09 (deploy) and 12 read the App's
memo volume from here -- 12 to refuse it as a check's memo folder -- and 00 the
App's dependency pins too, so the notebooks cannot disagree on them.
Named without a digit prefix so it imports.

Do NOT add databricks/__init__.py: this folder merges with the pip `databricks`
namespace package, and a regular package here would shadow databricks.sdk and
databricks.vector_search.
"""

import os
import re

# The Databricks App's own requirements.txt (09 stage() ships it as the App's root
# requirements.txt). Relative to the repo root, which Cell 2's os.chdir(REPO) makes cwd.
APP_REQUIREMENTS = os.path.join("databricks", "app_bundle", "requirements.txt")

# The UC Volume that holds the App's persistent LLM reply memo (determinism,
# 2026-10-27). Not the catalog's `raw` volume: the App writes here, and that one is
# READ VOLUME only. Its own volume also keeps the grant narrow (WRITE VOLUME on
# memo files, never on the catalog file the App hydrates from).
MEMO_VOLUME_NAME = "app_state"
MEMO_SUBDIR = "llm_memo"


def llm_model():
    """The model the configured provider will call, read LIVE from cfg."""
    import app.config as cfg

    return {
        "openai": cfg.OPENAI_MODEL,
        "gemini": cfg.GEMINI_MODEL,
        "anthropic": cfg.ANTHROPIC_MODEL,
    }.get(cfg.LLM_PROVIDER, "?")


def memo_volume():
    """The memo volume's full UC name, e.g. dev.cas.app_state (read LIVE from cfg)."""
    import app.config as cfg

    return f"{cfg.CAS_CATALOG}.{cfg.CAS_SCHEMA}.{MEMO_VOLUME_NAME}"


def memo_volume_path():
    """The memo volume's /Volumes path, e.g. /Volumes/dev/cas/app_state."""
    import app.config as cfg

    return f"/Volumes/{cfg.CAS_CATALOG}/{cfg.CAS_SCHEMA}/{MEMO_VOLUME_NAME}"


def app_memo_dir():
    """The App's LLM_MEMO_DIR: app.yaml must say exactly this (09 _check_bundle)."""
    return f"{memo_volume_path()}/{MEMO_SUBDIR}"


def bundle_pins(path=APP_REQUIREMENTS):
    """{distribution name (lower case, '-' separators): pinned version} of the App.

    Only exact `name==version` lines count (extras dropped: uvicorn[standard] ->
    uvicorn); comments and blank lines are skipped. The App's interpreter is the
    platform's (Python 3.11), not the notebook's, so these are what the App runs.
    """
    pins = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([^\s;]+)", line)
            if m:
                pins[re.sub(r"[._]+", "-", m.group(1)).lower()] = m.group(2)
    return pins


def _memo_line(cfg):
    """The header's memo line. Never prints keys or replies: the state and the dir only."""
    if not getattr(cfg, "LLM_MEMO", False):
        return "off (every LLM call is a fresh sample)"
    where = str(getattr(cfg, "LLM_MEMO_DIR", "?"))
    note = ""
    if where.rstrip("/") == app_memo_dir():
        note = " -- the APP's memo: these calls read and pin the App's replies"
    elif not where.startswith("/Volumes/"):
        note = " -- not a /Volumes path: on a notebook, use a scratch Volume dir"
    # Said here because the header is all a notebook user reads: every eval harness
    # (llm_memo.disable_for_eval / eval_page_types.run) switches it off again
    return (f"ON -> {where} (replays accepted replies){note}; only for app calls made "
            "directly in this notebook -- run_eval, eval_page_types and the other eval "
            "harnesses turn it off at their start")


def patch_eval_harness(spark, verbose=True, llm=None, memo=False):
    """Switch app.search to Vector Search and send eval CSVs to the UC Volume.

    `llm` overrides the header's llm line for callers where cfg's model is not
    what will run (05 swaps models per run; its control() uses none).

    `memo` (default False) sets cfg.LLM_MEMO and the header prints it. An eval
    compares models, prompts and repeats; a warm memo would silently replay old
    samples (and make a temperature sweep measure nothing), so it is off. memo=True
    reaches ONLY app calls made directly in the notebook (app.main.search, the
    translator...): run_eval.evaluate() and eval_page_types.run() -- hence 04-06 and
    10 -- calibrate_relevance.run() (07) and every other eval/ harness turn it off
    again before their first LLM
    call, and the header says so. A determinism check is eval/check_determinism.py
    with its own --memo on / --memo-dir (a scratch dir, never the App's).
    """
    from pathlib import Path

    import app.config as cfg
    import app.llm_client as llm_client
    import eval.run_eval as run_eval
    from app import search, tracking, vector_search

    # Belt-and-braces: get_llm_client() already rebuilds on a provider/model/base-URL
    # change in cfg. An os.environ edit is NOT picked up (cfg reads env once): restartPython
    llm_client._singleton = None
    cfg.LLM_MEMO = bool(memo)

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
        print(f"index   : {cfg.VS_INDEX} ({cfg.VS_EMBEDDING_MODEL}, query={cfg.VS_QUERY_MODE})")
        print(f"results : {out}")
        print(f"eval set: {run_eval.EVAL_QUERIES_JSON.name}")
        print(f"llm     : {llm or f'{cfg.LLM_PROVIDER} / {llm_model()}'}")
        print(f"memo    : {_memo_line(cfg)}")
        print(f"mlflow  : {'ON -> ' + str(cfg.MLFLOW_EXPERIMENT) if tracked else 'off'}")
    # Fail HERE, not 18 queries into an eval, when the index is missing
    info = vector_search.index_info()
    if info is None:
        raise RuntimeError(f"{cfg.VS_INDEX} does not exist: run 02's resume() first.")
    return info
