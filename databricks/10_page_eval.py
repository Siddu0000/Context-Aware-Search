"""Step 10: the app-path page-type eval on Databricks (gte-large-en + the workspace LLM).

eval/eval_page_types.py calls app.main.search in-process -- the code the App serves --
and scores page 1 as a shopper sees it: product types in the top 4 / top 12 of a
clothing ask, context fit (no bridal gowns or wool at a beach wedding), outfit slot
order, setup components, and that type-vocabulary collisions ("ring light", "tea
bags", "apple watch") keep their vertical. 04-06 score the reranker's own list and
cannot see main.py's ordering, which is where the demo complaint lived. The demo ran
on the App (gte-large-en + Qwen) while the diagnosis traces are local (gte-small +
gpt-oss), so this is the run that measures what the stakeholder saw.

mode="both" runs TYPE_AWARE_RANKING off then on and replays the translator (and any
identical rerank call) into the second arm. "off" is NOT the pre-change code: the
prompts, routing, the gender fix and the type= field sit outside the knob. For a true
before, run this from a Git folder at the previous commit with only
eval/eval_page_types.py, data/page_type_queries.json and app/product_type.py copied
in (the script types rows itself there and runs one 'head' arm).

Cost: 2 LLM calls per query run; "both" adds only the rerank calls it cannot replay:
34 queries ~ 70-100 calls per repeat. No index rebuild; the index must be ONLINE.

app.main is a FastAPI module and the runbook's Cell 1 does not install FastAPI. If
the import fails:
    %pip install "fastapi==0.136.3"     # the App's pin (app_bundle/requirements.txt); no --upgrade
    dbutils.library.restartPython()

Usage in a notebook:
    exec(open("databricks/10_page_eval.py").read())
    run()                                             # off + on, 1 repeat, rerank on
    run(repeats=3)                                    # the stakeholder before/after
    run(relevance="data/eval_queries_context.json")   # served-page P@4 / P@12, context set
"""

import sys

sys.path.insert(0, "databricks")
import vs_shim  # noqa: E402


def run(mode="both", repeats=1, rerank=True, queries=None, relevance=None):
    """Page-type eval through app.main.search on Vector Search; the CSV goes to the UC
    Volume next to 04's. Returns (csv_path, rows)."""
    try:
        import fastapi  # noqa: F401 -- app.main needs it
    except ImportError as e:
        raise RuntimeError(
            'FastAPI is missing: %pip install "fastapi==0.136.3" (no --upgrade), then '
            "dbutils.library.restartPython() and run Cell 2 again."
        ) from e
    import logging

    from pyspark.sql import SparkSession

    import app.config as cfg
    import eval.eval_page_types as page_eval
    import eval.run_eval as run_eval

    # One INFO line per LLM request otherwise (the URL, never the query, but noise)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    spark = SparkSession.builder.getOrCreate()
    vs_shim.patch_eval_harness(spark)

    # The CSV name carries the model (eval_page_types adds it) and the index mode:
    # "vector" = the stored-vector index (2026-10-09 on), "text" = the old managed
    # one; their numbers are not comparable
    tag = f"databricks_gte_large_{cfg.VS_QUERY_MODE}"
    return page_eval.run(mode=mode, repeats=repeats, rerank=rerank, queries=queries,
                         relevance=relevance, out_dir=run_eval.EVAL_RESULTS_DIR, tag=tag)
