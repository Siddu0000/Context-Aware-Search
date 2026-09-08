"""Step 4: run eval/run_eval.py against Vector Search instead of local numpy.

Re-benchmarking is MANDATORY after the embedding swap: every published number
(NDCG 0.955 Fashion, 0.894 3-vertical, P@1/MRR 1.000) was measured on
thenlper/gte-small at 384 dims. databricks-gte-large-en is a different model at
1024 dims, so those numbers do not transfer.

Nothing in eval/ or app/ is edited. The harness is redirected in-process, so the
committed pipeline stays the single source of truth.

Run as a Databricks notebook or job task, with the repo root on sys.path.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------- config
ENDPOINT = "cas-search"
INDEX = "main.cas.products_index"
TABLE = "main.cas.products"
RESULTS_VOLUME = "/Volumes/main/cas/evals"   # durable; the repo dir is not

# Every column app/reranker.py reads, plus the primary key. Miss one and the
# reranker silently sees empty strings and scores worse for the wrong reason.
COLUMNS = [
    "catalog_index",
    "Product_title",
    "bsns_vrtcl_name",
    "categ_lvl2_name",
    "color",
    "material",
    "occasion",
    "price",
    "prod_description",
    "average_rating",
    "rating_number",
]


# ------------------------------------------------------- retrieval shim
def make_vs_search(index):
    """Return a drop-in replacement for app.search.search_products."""

    def search_products(search_terms, top_k=30, per_intent_quota=None):
        rows, seen = [], set()
        for term in search_terms:
            res = index.similarity_search(
                query_text=term, columns=COLUMNS, num_results=top_k
            )
            data = (res.get("result") or {}).get("data_array") or []
            kept = 0
            for r in data:
                if per_intent_quota is not None and kept >= per_intent_quota:
                    break
                item = dict(zip(COLUMNS, r))
                # similarity is appended after the requested columns
                item["score"] = float(r[len(COLUMNS)]) if len(r) > len(COLUMNS) else 0.0
                item["source_intent"] = term
                title = item.get("Product_title")
                if title in seen:
                    continue          # mirrors the local dedup-by-title
                seen.add(title)
                rows.append(item)
                kept += 1
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows

    return search_products


def main():
    assert os.environ.get("DATABRICKS_TOKEN"), "DATABRICKS_TOKEN not set"

    from databricks.vector_search.client import VectorSearchClient
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    # Relevance sets need the WHOLE catalog in pandas: build_relevance_set
    # filters on title/color/material/occasion/category across all rows.
    # 300K x ~13 cols fits the driver comfortably.
    df = spark.table(TABLE).toPandas()
    print(f"catalog: {len(df):,} rows for relevance sets")

    index = VectorSearchClient().get_index(
        endpoint_name=ENDPOINT, index_name=INDEX
    )

    import eval.run_eval as run_eval

    # Patch the MODULE THAT IMPORTED the names. run_eval does
    # `from app.search import search_products`, which binds into its own
    # namespace at import time -- patching app.search here would be ignored
    # and you would silently benchmark the local model instead.
    run_eval.search_products = make_vs_search(index)
    run_eval.load_index = lambda *a, **k: None      # no local .npy to load
    run_eval.get_dataframe = lambda: df

    out = Path(RESULTS_VOLUME)
    out.mkdir(parents=True, exist_ok=True)
    run_eval.EVAL_RESULTS_DIR = out

    rerank_on = "--rerank" in sys.argv
    tag = "databricks_gte_large_" + ("rerank_on" if rerank_on else "rerank_off")
    run_eval.evaluate(rerank_on=rerank_on, tag=tag)


if __name__ == "__main__":
    main()
