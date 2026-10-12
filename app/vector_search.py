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

Vector Search is reached through its REST API with the databricks-sdk client
every runtime already ships -- NOT the databricks-vectorsearch package (now a
deprecated shim over databricks-ai-search). That package pins protobuf<6, and
installing it on DBR 19 (protobuf 6.33.5) downgraded protobuf until the notebook
kernel could not start (Spark Connect gencode 6.33.5 vs runtime 5.29.6, 2026-10-10).
REST paths: https://docs.databricks.com/api/workspace/vectorsearchindexes

Imports of databricks/pyspark are lazy so local dev never needs them.
"""

import logging
import os
import shutil
import tempfile
from typing import List, Tuple

import app.config as cfg

logger = logging.getLogger(__name__)

_spark = None
_workspace = {}
_VS_API = "/api/2.0/vector-search"


def set_spark(spark) -> None:
    """Inject the notebook's session; avoids guessing at Spark Connect internals."""
    global _spark
    _spark = spark


def rest(method: str, path: str, body: dict = None, query: dict = None) -> dict:
    """One Vector Search REST call (path relative to /api/2.0/vector-search), as a dict.

    Same JSON the old VectorSearchClient returned (it wrapped these endpoints), so
    callers parse what they always parsed. Auth and 429/503 retries: the SDK's.
    """
    out = _workspace_client().api_client.do(
        method, f"{_VS_API}{path}", body=body, query=query,
        # what the SDK's typed vector-search methods send; a raw do() would not
        headers={"Accept": "application/json", "Content-Type": "application/json"})
    return out if isinstance(out, dict) else {}


def is_not_found(e: Exception) -> bool:
    """The SDK's 404 for a missing endpoint / index -- and nothing looser.

    Not ENDPOINT_NOT_FOUND (a wrong API path, also a NotFound), and no message
    matching ("Catalog ... does not exist" is a 400): misreading either as
    "absent" would let pause() report "nothing to delete" over a live index.
    """
    try:
        from databricks.sdk.errors import NotFound
    except ImportError:
        return False
    return isinstance(e, NotFound) and getattr(e, "error_code", None) != "ENDPOINT_NOT_FOUND"


def endpoint_info(name: str = None):
    """GET the Vector Search endpoint, or None if it does not exist."""
    try:
        return rest("GET", f"/endpoints/{name or cfg.VS_ENDPOINT}")
    except Exception as e:  # noqa: BLE001
        if is_not_found(e):
            return None
        raise


def index_info(name: str = None):
    """GET the index (status.detailed_state, indexed_row_count, spec), or None."""
    try:
        return rest("GET", f"/indexes/{name or cfg.VS_INDEX}")
    except Exception as e:  # noqa: BLE001
        if is_not_found(e):
            return None
        raise


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
        # Databricks Apps run plain Python with no Spark: read the CSV
        source = cfg.VS_CATALOG_CSV
        local = _local_copy(source)
        try:
            df = pd.read_csv(local, low_memory=False)
        finally:
            if local != source:
                os.remove(local)  # the App's disk is small; the DataFrame is the copy
        if "catalog_index" in df.columns:
            # databricks/09 export_catalog() writes the table's OWN key, which
            # makes the density check below a real check of the export
            df = df.sort_values("catalog_index", kind="stable")
        else:
            # The raw CSV 01 loaded: row order IS the key (01's row_number over file order)
            df["catalog_index"] = range(len(df))

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


def _local_copy(path: str) -> str:
    """A path pandas can open: `path` itself where /Volumes is mounted (notebooks),
    else a Files API download -- a Databricks App has no /Volumes mount, so a plain
    open() raises FileNotFoundError there:
    https://kb.databricks.com/en_US/unity-catalog/queries-to-access-files-on-a-unity-catalog-volume-fail-with-filenotfound-error
    """
    if os.path.exists(path):
        return path
    dest = os.path.join(tempfile.gettempdir(), "cas_" + os.path.basename(path))
    src = _workspace_client(batch=True).files.download(path).contents
    try:
        # Streamed to disk: the raw bytes and the parsed frame never sit in RAM together
        with open(dest, "wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
    finally:
        src.close()
    logger.info("Downloaded %s through the Files API (%d MB).", path, os.path.getsize(dest) >> 20)
    return dest


def index_health(expected_rows: int) -> dict:
    """Is the index there, ONLINE and built from a catalog of this size? Never raises.

    Without it a paused index (02 pause() deletes it) shows shoppers a generic
    "could not reach the search service"; with it the App's log and /healthz say why.
    """
    try:
        info = index_info()
        if info is None:
            return {"ok": False, "index": cfg.VS_INDEX,
                    "detail": "index does not exist (02 pause() deleted it? run resume())"}
        st = info.get("status") or {}
        state = str(st.get("detailed_state", "?"))
        rows = int(st.get("indexed_row_count") or 0)
    except Exception as e:  # noqa: BLE001 -- permissions, network, an odd reply: report it
        return {"ok": False, "index": cfg.VS_INDEX, "detail": f"{type(e).__name__}: {str(e)[:200]}"}
    ok = "ONLINE" in state and rows == expected_rows
    detail = "" if ok else (f"state {state}, {rows:,} vectors vs {expected_rows:,} catalog rows"
                            " -- a different catalog would hydrate the WRONG products")
    return {"ok": ok, "index": cfg.VS_INDEX, "state": state, "rows": rows, "detail": detail}


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
    # Plain Python floats: the request body is json.dumps'ed (no numpy, no NaN)
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
    res = rest("POST", f"/indexes/{cfg.VS_INDEX}/query",
               body={"columns": ["catalog_index"], "num_results": top_k, **query_arg})
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
