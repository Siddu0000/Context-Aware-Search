"""Step 7: calibrate MIN_RESULT_RELEVANCE for the Vector Search score scale.

The floor was tuned on gte-small cosine. If Vector Search scores sit on a
different scale, the deployed app either says "no matching products" for
everything or never says it at all -- and the eval would not catch either,
because run_eval never applies the floor.

Usage in a notebook (after Cell 2):
    exec(open("databricks/07_calibrate_threshold.py").read())
    result = run(translate=True)     # faithful; ~28 LLM calls
    # then set MIN_RESULT_RELEVANCE=<result["recommended"]> for the deployed app
"""

import sys

sys.path.insert(0, "databricks")
import vs_shim  # noqa: E402


def run(translate=True, margin=0.02):
    from pyspark.sql import SparkSession

    from eval import calibrate_relevance

    vs_shim.patch_eval_harness(SparkSession.builder.getOrCreate())
    return calibrate_relevance.run(translate=translate, margin=margin)
