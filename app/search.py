"""Retrieval over the product catalog: local numpy cosine or Databricks Vector Search.

Every consumer (main, recommendations, keyword engine, the eval suite) goes
through this module, so SEARCH_BACKEND switches the whole app in one place.
Both backends return identical row dicts; only where the (row, score) hits come
from differs.
"""

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import app.config as cfg
from app.config import CACHE_DIR, PRODUCTS_CSV
from app.embeddings import embed_text, get_default_embedder, search_vectors

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_FIELDS: Tuple[str, ...] = (
    "bsns_vrtcl_name",
    "categ_lvl2_name",
    "Product_title",
    "prod_description",
    "color",
    "material",
    "occasion",
)

_df: Optional[pd.DataFrame] = None
_embeddings: Optional[np.ndarray] = None
_loaded_backend: Optional[str] = None
_loaded_fields: Tuple[str, ...] = ()


def _file_hash(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _cache_key(fields: Tuple[str, ...]) -> str:
    embedder = get_default_embedder()
    parts = [
        _file_hash(PRODUCTS_CSV),
        embedder.model_name.replace("/", "_"),
        "-".join(sorted(fields)),
    ]
    return "_".join(parts)


def _build_search_text(df: pd.DataFrame, fields: Tuple[str, ...]) -> pd.Series:
    for col in fields:
        if col not in df.columns:
            logger.warning("Missing column in CSV: %s. Filling with ''.", col)
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.lower()
    return df[list(fields)].agg(" ".join, axis=1)


def _backend() -> str:
    return (cfg.SEARCH_BACKEND or "local").lower()


def _ready() -> bool:
    if _df is None or _loaded_backend != _backend():
        return False
    return _backend() == "databricks" or _embeddings is not None


def load_index(fields: Tuple[str, ...] = DEFAULT_SEARCH_FIELDS) -> None:
    """Build the in-memory index once. Rebuilds if the field set or backend changes."""
    global _df, _embeddings, _loaded_fields, _loaded_backend

    if _backend() == "databricks":
        if _df is not None and _loaded_backend == "databricks":
            return
        if fields != DEFAULT_SEARCH_FIELDS:
            # The managed index embeds one search_text column, fixed at build time
            logger.warning(
                "SEARCH_BACKEND=databricks ignores fields=%r; the index embeds "
                "the search_text column it was built with.", fields,
            )
        from app import vector_search

        _df = vector_search.load_catalog()
        _embeddings = None
        _loaded_fields = DEFAULT_SEARCH_FIELDS
        _loaded_backend = "databricks"
        logger.info("Index ready: %d products via Vector Search %s.", len(_df), cfg.VS_INDEX)
        return

    if (
        _df is not None
        and _embeddings is not None
        and _loaded_fields == fields
        and _loaded_backend == "local"
    ):
        return

    if not PRODUCTS_CSV.exists():
        raise FileNotFoundError(f"Catalog not found: {PRODUCTS_CSV}")

    logger.info("Loading catalog from %s", PRODUCTS_CSV)
    df = pd.read_csv(PRODUCTS_CSV)
    df["search_text"] = _build_search_text(df, fields)

    cache_path = CACHE_DIR / f"embeddings_{_cache_key(fields)}.npy"
    if cache_path.exists():
        logger.info("Cache hit: %s", cache_path.name)
        embeddings = np.load(cache_path)
    else:
        logger.info(
            "Cache miss. Encoding %d products with %s ...",
            len(df),
            get_default_embedder().model_name,
        )
        embeddings = embed_text(df["search_text"].tolist())
        np.save(cache_path, embeddings)
        logger.info("Persisted embeddings to %s", cache_path.name)

    _df = df
    _embeddings = embeddings
    _loaded_fields = fields
    _loaded_backend = "local"
    logger.info("Index ready: %d products, dim=%d.", len(df), embeddings.shape[1])


def get_dataframe() -> pd.DataFrame:
    if not _ready():
        load_index()
    return _df


def get_product(catalog_index: int) -> Optional[dict]:
    """Return a single catalog row as a JSON-safe dict (NaN -> None)."""
    if not _ready():
        load_index()
    if catalog_index < 0 or catalog_index >= len(_df):
        return None
    row = _df.iloc[int(catalog_index)].to_dict()
    item = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
    item["catalog_index"] = int(catalog_index)
    return item


def search_products(
    search_terms: List[str],
    top_k: int = 30,
    per_intent_quota: Optional[int] = None,
) -> List[dict]:
    """Scatter-gather: top_k per intent, merged, deduped by title, score-sorted."""
    if not _ready():
        load_index()
    if not search_terms:
        return []

    rows: List[dict] = []
    for term, hits in zip(search_terms, _raw_hits(search_terms, top_k)):
        kept = 0
        for idx, score in hits:
            # grocery mode: cap per intent so one ingredient can't crowd the rest out
            if per_intent_quota is not None and kept >= per_intent_quota:
                break
            item = _df.iloc[int(idx)].to_dict()
            item["score"] = float(score)
            item["catalog_index"] = int(idx)
            item["source_intent"] = term
            rows.append(item)
            kept += 1

    if not rows:
        return []

    final = (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .drop_duplicates(subset="Product_title")
        .replace({np.nan: None})
    )
    return final.to_dict(orient="records")


def _raw_hits(search_terms: List[str], top_k: int) -> List[List[Tuple[int, float]]]:
    """Per-intent [(row_position, score)] from the active backend, best first."""
    if _backend() == "databricks":
        from app import vector_search

        return [vector_search.query(t, top_k) for t in search_terms]
    intent_vecs = embed_text(search_terms)
    out = []
    for vec in intent_vecs:
        indices, scores = search_vectors(vec, _embeddings, top_k=top_k)
        out.append(list(zip(indices, scores)))
    return out


def best_score(candidates: List[dict]) -> float:
    """Top embedding similarity in a candidate list (0.0 if empty)."""
    scores = [
        c["score"] for c in candidates
        if isinstance(c.get("score"), (int, float))
    ]
    return max(scores) if scores else 0.0
