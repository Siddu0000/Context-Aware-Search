"""Step 4: run eval/run_eval.py against Vector Search instead of local numpy.

Re-benchmarking is MANDATORY after an embedding swap: numbers measured on
thenlper/gte-small at 384 dims do not transfer to a 1024-dim model.

Retrieval-only by default (rerank off) so the embedding change is isolated.
Pass --rerank for the full pipeline; 05 compares LLMs, 06 compares translator
modes. The shim lives in vs_shim.py and is shared by all three.

Usage in a notebook:
    exec(open("databricks/04_run_eval.py").read())
"""

import sys

sys.path.insert(0, "databricks")
import vs_shim  # noqa: E402


def main():
    from pyspark.sql import SparkSession

    import eval.run_eval as run_eval

    spark = SparkSession.builder.getOrCreate()
    vs_shim.patch_eval_harness(spark)

    rerank_on = "--rerank" in sys.argv
    tag = "databricks_gte_large_" + ("rerank_on" if rerank_on else "rerank_off")
    return run_eval.evaluate(rerank_on=rerank_on, tag=tag)


if __name__ == "__main__":
    main()
