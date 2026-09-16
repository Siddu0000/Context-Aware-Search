"""Redirect the eval harness at Mosaic AI Vector Search instead of local numpy.

Shared by 04 (retrieval eval), 05 (LLM comparison) and 06 (translator modes).
Named without a digit prefix so it can actually be imported.

Nothing in eval/ or app/ is edited: the committed pipeline stays the single
source of truth and only its retrieval entry point is swapped.
"""

import os

# Every column app/reranker.py reads, plus the primary key. Miss one and the
# reranker sees empty strings and scores worse for the wrong reason.
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

CATALOG = os.getenv("CAS_CATALOG", "dev")
SCHEMA = os.getenv("CAS_SCHEMA", "cas")
ENDPOINT = os.getenv("CAS_ENDPOINT", "cas-search")
INDEX = f"{CATALOG}.{SCHEMA}.products_index"
TABLE = f"{CATALOG}.{SCHEMA}.products"
RESULTS_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/evals"


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


def patch_eval_harness(spark, verbose=True):
    """Point eval.run_eval at Vector Search + the Delta table. Returns the index.

    Patches the MODULE THAT IMPORTED the names: run_eval does
    `from app.search import search_products`, which binds into its own namespace
    at import time, so patching app.search would be ignored and the eval would
    silently measure the LOCAL model while reporting it as Databricks.

    compare_llms and compare_translators both call run_eval.evaluate(), which
    resolves these names from run_eval's globals -- so this one patch covers all
    three harnesses.
    """
    from pathlib import Path

    from databricks.vector_search.client import VectorSearchClient

    import eval.run_eval as run_eval

    df = spark.table(TABLE).toPandas()
    if verbose:
        print(f"catalog: {len(df):,} rows for relevance sets")

    index = VectorSearchClient(disable_notice=True).get_index(
        endpoint_name=ENDPOINT, index_name=INDEX
    )

    run_eval.search_products = make_vs_search(index)
    run_eval.load_index = lambda *a, **k: None      # no local .npy to load
    run_eval.get_dataframe = lambda: df

    out = Path(RESULTS_VOLUME)
    out.mkdir(parents=True, exist_ok=True)
    run_eval.EVAL_RESULTS_DIR = out

    if verbose:
        print(f"index  : {INDEX}")
        print(f"results: {out}")
    return index
