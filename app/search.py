"""Retrieval over the product catalog.

Pipeline at boot:
    1. Load CSV
    2. Build per-product `search_text` from configured fields
    3. Load embeddings from disk if cache hit, else compute + persist

The cache key is a hash of (CSV content, embedding model name, field set).
Any change to any of those auto-invalidates. No manual cache busting needed.
"""

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.config import CACHE_DIR, PRODUCTS_CSV
from app.embeddings import embed_text, get_default_embedder, search_vectors

logger = logging.getLogger(__name__)

# Fields that go into the searchable text for each product.
# Centralized here so the "chunking" comparison eval can swap them.
DEFAULT_SEARCH_FIELDS: Tuple[str, ...] = (
    "bsns_vrtcl_name",
    "categ_lvl2_name",
    "Product_title",
    "prod_description",
    "color",
    "material",
    "occasion",
)

# Module-level state populated by load_index().
_df: Optional[pd.DataFrame] = None
_embeddings: Optional[np.ndarray] = None
_loaded_fields: Tuple[str, ...] = ()


def _file_hash(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _cache_key(fields: Tuple[str, ...]) -> str:
    """Stable key combining CSV hash + model name + field set."""
    embedder = get_default_embedder()
    parts = [
        _file_hash(PRODUCTS_CSV),
        embedder.model_name.replace("/", "_"),
        "-".join(sorted(fields)),
    ]
    return "_".join(parts)


def _build_search_text(df: pd.DataFrame, fields: Tuple[str, ...]) -> pd.Series:
    """Combine the configured fields into one lowercase string per row.

    `fillna("")` BEFORE str-casting is critical: otherwise NaN cells become
    the literal string 'nan' and pollute the embedding space.
    """
    for col in fields:
        if col not in df.columns:
            logger.warning("Missing column in CSV: %s. Filling with ''.", col)
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.lower()
    return df[list(fields)].agg(" ".join, axis=1)


def load_index(fields: Tuple[str, ...] = DEFAULT_SEARCH_FIELDS) -> None:
    """Idempotent: builds the in-memory index once.

    Reloads if the field set differs from what's currently loaded — that's
    how the chunking-comparison eval triggers a rebuild without restarting.
    """
    global _df, _embeddings, _loaded_fields

    if _df is not None and _embeddings is not None and _loaded_fields == fields:
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
    logger.info("Index ready: %d products, dim=%d.", len(df), embeddings.shape[1])


def get_dataframe() -> pd.DataFrame:
    if _df is None:
        load_index()
    return _df


def get_product(catalog_index: int) -> Optional[dict]:
    """Return a single catalog row as a JSON-safe dict (NaN -> None).

    catalog_index is the positional row index we attach to every search
    result, so the UI can round-trip a product into a detail view.
    """
    if _df is None:
        load_index()
    if catalog_index < 0 or catalog_index >= len(_df):
        return None
    row = _df.iloc[int(catalog_index)].to_dict()
    item = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
    item["catalog_index"] = int(catalog_index)
    return item


def intent_similarity(terms: List[str], catalog_indices: List[int]) -> dict:
    """Max cosine similarity between any of `terms` and each catalog row.

    Same basis as retrieval (scatter-gather over the query intents), so the
    scores are directly comparable to the `score` field on organic results.
    The sponsored layer uses this to gate ads on real relevance to the query.
    Returns {catalog_index: similarity}.
    """
    if _df is None or _embeddings is None:
        load_index()
    if not terms or not catalog_indices:
        return {}
    term_vecs = embed_text(terms)
    out = {}
    for idx in catalog_indices:
        if 0 <= idx < len(_embeddings):
            sims = cosine_similarity([_embeddings[idx]], term_vecs)[0]
            out[idx] = float(sims.max())
    return out


def search_products(search_terms: List[str], top_k: int = 30) -> List[dict]:
    """Scatter-gather retrieval: top_k candidates per intent, then merge.

    Returns up to (len(search_terms) * top_k) candidates, deduped by Product_title,
    sorted by best embedding score. This is the candidate pool for the reranker
    (which then trims to FINAL_TOP_K).
    """
    if _df is None or _embeddings is None:
        load_index()
    if not search_terms:
        return []

    intent_vecs = embed_text(search_terms)

    rows: List[dict] = []
    for vec in intent_vecs:
        indices, scores = search_vectors(vec, _embeddings, top_k=top_k)
        for idx, score in zip(indices, scores):
            item = _df.iloc[int(idx)].to_dict()
            item["score"] = float(score)
            item["catalog_index"] = int(idx)
            rows.append(item)

    if not rows:
        return []

    final = (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .drop_duplicates(subset="Product_title")
        .replace({np.nan: None})  # JSON-safe
    )
    return final.to_dict(orient="records")
