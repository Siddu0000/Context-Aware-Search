"""Mosaic AI Vector Search retrieval, used when SEARCH_BACKEND=databricks.

The index returns ONLY the key and the similarity. Full rows are hydrated from
the catalog DataFrame by app/search.py, so both backends return identical row
shapes -- including img_url, store and parent_asin, which the index was never
asked for. (Returning index columns instead would blank every product image and
silently stop sponsored ads matching, since those are keyed by parent_asin.)

Queries are embedded HERE (embed_texts) and sent as query_vector: the index
databricks/02 builds holds self-managed vectors, so it has no query-time model.
databricks/02 embeds the catalog through the same embed_texts, so document and
query vectors cannot drift apart (different endpoint, different preprocessing).

Imports of databricks/pyspark are lazy so local dev never needs them.
"""

import logging
from typing import List, Tuple

import app.config as cfg

logger = logging.getLogger(__name__)

_index = None
_spark = None
_workspace = {}


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
        df = pd.read_csv(cfg.VS_CATALOG_CSV, low_memory=False)
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


def _workspace_client(batch: bool = False):
    """databricks-sdk client; authenticates itself in notebooks (driver) and in Apps.

    The SDK retries 429/503 on its own until retry_timeout_seconds (default 300s)
    runs out. The budget is PER CALL and /product makes up to ~6 sequential
    retrievals against a 60s UI timeout, so the app's client gives up fast; the
    catalog batch (02 embed(), batch=True) can afford to wait.
    """
    if batch not in _workspace:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.core import Config

        timeouts = (60, 60) if batch else (10, 15)
        _workspace[batch] = WorkspaceClient(config=Config(
            retry_timeout_seconds=timeouts[0], http_timeout_seconds=timeouts[1]))
    return _workspace[batch]


def embed_texts(texts: List[str], batch: bool = False) -> Tuple[List[List[float]], int]:
    """(unit vectors in input order, prompt tokens billed) from VS_EMBEDDING_MODEL.

    The ONLY embedding call on the Databricks path: the catalog (02 embed()) and
    every query go through it. Vectors are L2-normalised because gte-large-en
    "does not generate normalized embeddings" while the index ranks by L2
    distance (score 1/(1+d^2)); on unit vectors that ranking equals cosine:
    https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/supported-models
    """
    if not texts:
        return [], 0
    resp = _workspace_client(batch).serving_endpoints.query(
        name=cfg.VS_EMBEDDING_MODEL, input=list(texts)
    )
    import numpy as np

    data = list(resp.data or [])
    # Input order is not documented: restore it from each element's index
    if all(getattr(d, "index", None) is not None for d in data):
        data.sort(key=lambda d: d.index)
    vecs = [list(d.embedding or []) for d in data]
    if len(vecs) != len(texts) or not all(vecs) or len({len(v) for v in vecs}) != 1:
        # A short, empty or ragged reply would silently mis-align vectors and products
        raise RuntimeError(
            f"{cfg.VS_EMBEDDING_MODEL} returned {len(vecs)} embeddings "
            f"({sum(1 for v in vecs if not v)} empty) for {len(texts)} inputs"
        )
    arr = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise RuntimeError(f"{cfg.VS_EMBEDDING_MODEL} returned a zero vector")
    usage = getattr(resp, "usage", None)
    # Plain Python floats: the client JSON-encodes query_vector and drops falsy ones
    return (arr / norms).tolist(), int(getattr(usage, "prompt_tokens", 0) or 0)


def query_many(terms: List[str], top_k: int) -> List[List[Tuple[int, float]]]:
    """Per term [(catalog_index, similarity), ...], best first; ONE embedding call."""
    if not terms:
        return []
    mode = cfg.VS_QUERY_MODE
    if mode == "text":
        return [_search(top_k, query_text=t) for t in terms]
    if mode != "vector":
        raise ValueError(f"VS_QUERY_MODE must be 'vector' or 'text', got {mode!r}")
    vecs, _ = embed_texts(terms)
    return [_search(top_k, query_vector=v) for v in vecs]


def query(term: str, top_k: int) -> List[Tuple[int, float]]:
    """[(catalog_index, similarity), ...] for one intent, best first."""
    return query_many([term], top_k)[0]


def _search(top_k: int, **query_arg) -> List[Tuple[int, float]]:
    res = get_index().similarity_search(
        columns=["catalog_index"], num_results=top_k,
        disable_notice=True,  # printed per call otherwise; the client flag doesn't cover it
        **query_arg,
    )
    rows = (res.get("result") or {}).get("data_array") or []
    # The score column is appended after the requested ones and named "score"
    # (not in the REST docs; databricks-ai-bridge reads it this way)
    names = [c.get("name") for c in (res.get("manifest") or {}).get("columns") or []]
    at = names.index("score") if "score" in names else -1
    out = []
    for r in rows:
        if not r:
            continue
        # key may come back as int, float or numeric string depending on the client
        idx = int(float(r[0]))
        score = float(r[at]) if len(r) > 1 else 0.0
        out.append((idx, score))
    return out
