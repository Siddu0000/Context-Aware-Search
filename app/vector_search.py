"""Mosaic AI Vector Search retrieval, used when SEARCH_BACKEND=databricks.

The index returns ONLY the key and the similarity. Full rows are hydrated from
the catalog DataFrame by app/search.py, so both backends return identical row
shapes -- including img_url, store and parent_asin, which the index was never
asked for. (Returning index columns instead would blank every product image and
silently stop sponsored ads matching, since those are keyed by parent_asin.)

Imports of databricks/pyspark are lazy so local dev never needs them.
"""

import logging
from typing import List, Tuple

import app.config as cfg

logger = logging.getLogger(__name__)

_index = None
_spark = None


def set_spark(spark) -> None:
    """Inject the notebook's session; avoids guessing at Spark Connect internals."""
    global _spark
    _spark = spark


def set_index(index) -> None:
    """Inject an index handle (tests, or a notebook that already has one)."""
    global _index
    _index = index


def get_index():
    global _index
    if _index is None:
        from databricks.vector_search.client import VectorSearchClient

        _index = VectorSearchClient(disable_notice=True).get_index(
            endpoint_name=cfg.VS_ENDPOINT, index_name=cfg.VS_INDEX
        )
    return _index


def _active_spark():
    if _spark is not None:
        return _spark
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    try:
        # getActiveSession, never getOrCreate: that would START a local Spark
        return SparkSession.getActiveSession()
    except Exception:  # noqa: BLE001
        return None


def load_catalog():
    """The catalog the index was built from, with row position == catalog_index."""
    import pandas as pd

    spark = _active_spark()
    if spark is not None:
        df = spark.table(cfg.VS_TABLE).orderBy("catalog_index").toPandas()
        source = cfg.VS_TABLE
    else:
        # Databricks Apps run plain Python with no Spark: read the source CSV
        df = pd.read_csv(cfg.VS_CATALOG_CSV)
        df["catalog_index"] = range(len(df))
        source = cfg.VS_CATALOG_CSV

    df = df.reset_index(drop=True)
    keys = pd.to_numeric(df["catalog_index"], errors="coerce")
    if keys.isna().any() or not (keys.astype("int64").values == range(len(df))).all():
        raise RuntimeError(
            f"catalog_index in {source} is not dense 0..N-1. Hydration and "
            "GET /product use it as a ROW POSITION, so a sparse key returns the "
            "WRONG products. Re-run databricks/01_catalog_to_delta.py (dense "
            "row_number) and rebuild the index."
        )
    logger.info("Catalog loaded from %s: %d rows.", source, len(df))
    return df


def query(term: str, top_k: int) -> List[Tuple[int, float]]:
    """[(catalog_index, similarity), ...] for one intent, best first."""
    res = get_index().similarity_search(
        query_text=term, columns=["catalog_index"], num_results=top_k
    )
    rows = (res.get("result") or {}).get("data_array") or []
    out = []
    for r in rows:
        if not r:
            continue
        # key may come back as int, float or numeric string depending on the client
        idx = int(float(r[0]))
        score = float(r[-1]) if len(r) > 1 else 0.0
        out.append((idx, score))
    return out
