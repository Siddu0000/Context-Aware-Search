"""Retrieval over the product catalog: local numpy cosine or Databricks Vector Search.

Every consumer (main, recommendations, keyword engine, the eval suite) goes
through this module, so SEARCH_BACKEND switches the whole app in one place.
Both backends return identical row dicts; only where the (row, score) hits come
from differs.
"""

import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

import app.config as cfg
from app import catalog_taxonomy, product_type
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

# Embedded too when the catalog has them (scripts/enrich_catalog.py); the base
# catalog lacks them, so its search text -- and embedding cache -- is unchanged
ENRICHED_SEARCH_FIELDS: Tuple[str, ...] = (
    "gender", "use_case", "dietary", "specs", "gen_description",
)

# The case of the row text the LLM stages read (rerank candidate lines, the cross-sell
# anchor): the seller's, on both backends. Until 2026-10-15 the LOCAL backend lowercased
# those columns in place, so a local rerank-on run from before then sent the model other
# text with the same prompts_sha and catalog_fp. Eval provenance (param `row_text_case`;
# an older run lacks it): change it whenever the served row text changes.
# "+fine_fields" (2026-10-19): a typed Fashion rerank line also carries ' | subtype=<s> |
# audience=<a>' (reranker._type_field), with no catalog or embedding change.
# "+role" (2026-10-27): under an acted device ask the window's lines carry ' | role=device'
# or ' | role=accessory' (main._with_role_fields), and a None / NaN field renders '' (was
# "None" / "nan" on Databricks lines; local lines unchanged).
ROW_TEXT_CASE = "original+fine_fields+role"

_df: Optional[pd.DataFrame] = None
_embeddings: Optional[np.ndarray] = None
_loaded_backend: Optional[str] = None
_loaded_fields: Tuple[str, ...] = ()
_embedded_fields: Tuple[str, ...] = ()
# (frame, md5[:12]) of the CSV a local load_index read that frame from, when the hash is
# known to name the bytes read. catalog_fingerprint() returns it: the API's start-up
# config line hashed the 228 MB file a second time (~0.75-0.8 s per start, 2026-10-30
# review), right after load_index had. Kept WITH its frame, so a frame set any other way
# (a test, a Databricks load) never reports this file's hash.
_csv_hash_of: Optional[Tuple[pd.DataFrame, str]] = None


def effective_fields(columns, fields: Tuple[str, ...] = DEFAULT_SEARCH_FIELDS) -> Tuple[str, ...]:
    """The fields actually embedded. The enriched tags join the DEFAULT set only;
    an explicit field list (an ablation such as --exclude-description) is used as
    given, so it measures what it says. databricks/01 applies the same rule."""
    fields = tuple(fields)
    if fields != DEFAULT_SEARCH_FIELDS:
        return fields
    return fields + tuple(c for c in ENRICHED_SEARCH_FIELDS if c in columns and c not in fields)


def embedded_fields() -> Tuple[str, ...]:
    """What the loaded index embedded (for eval provenance); () before load."""
    return _embedded_fields


def catalog_fingerprint() -> str:
    """Short content hash of the loaded catalog, so evals on different catalog
    text (base vs enriched, before vs after a re-enrichment) are distinguishable.

    Local: the hash load_index took of the bytes it read (the same md5[:12] as hashing
    the file), so a CSV rewritten after the load no longer names a catalog that is not
    served; the file is hashed now only when that hash is unknown (a frame not loaded by
    load_index, or a catalog that changed while it was read). Databricks: unchanged."""
    if _df is None:
        return ""
    if _loaded_backend == "databricks":
        if "text_hash" not in _df.columns:
            return "no-text_hash"
        return hashlib.md5("\n".join(_df["text_hash"].astype(str)).encode()).hexdigest()[:12]
    if _csv_hash_of is not None and _csv_hash_of[0] is _df:
        return _csv_hash_of[1]
    return _file_hash(PRODUCTS_CSV)


def _file_hash(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _file_state(path) -> Optional[tuple]:
    """(size, mtime_ns), or None when the file cannot be stat'ed: equal before and after
    a read means the bytes read are the bytes hashed."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def _cache_key(fields: Tuple[str, ...], csv_hash: Optional[str] = None) -> str:
    """The embedding cache file's key; `csv_hash` saves re-reading the CSV when the
    caller already hashed it (load_index)."""
    embedder = get_default_embedder()
    parts = [
        csv_hash or _file_hash(PRODUCTS_CSV),
        embedder.model_name.replace("/", "_"),
        "-".join(sorted(fields)),
    ]
    return "_".join(parts)


def _build_search_text(df: pd.DataFrame, fields: Tuple[str, ...]) -> pd.Series:
    """The lowercased, space-joined text that gets embedded, one string per row.

    Lowercases COPIES: the shopper sees these columns, and lowercasing them in place
    served "kalawen watch winder for rolex ..." while the seller wrote it in title case.
    The text itself is byte-identical to the old in-place version (each column
    lowercased, then joined; tests/test_search_text.py pins it). It must stay so: the
    embedding cache key hashes the CSV file, not this text, so a changed text would be
    served the OLD vectors without a re-encode. The one in-place step is unchanged and
    deliberate: missing fields become '' (str), the row shape the local app has always
    served (_product_gender and the reranker read '' as absent; BM25 tokenises it to
    nothing)."""
    for col in fields:
        if col not in df.columns:
            logger.warning("Missing column in CSV: %s. Filling with ''.", col)
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    return df[list(fields)].apply(lambda col: col.str.lower()).agg(" ".join, axis=1)


def _annotate_types(df: pd.DataFrame, csv_hash: Optional[str] = None) -> None:
    """Derive product_type from the titles (app/product_type.py), then the Electronics /
    Grocery taxonomy (app/catalog_taxonomy.py). Only counts and the elapsed time are
    logged, never titles. Both are ranking and rail signals and deliberately NOT
    embedded: search_text and the embedding cache key are unchanged. `csv_hash` (local
    backend only) lets the taxonomy columns come from .cache/ (see _annotate_taxonomy)."""
    started = time.perf_counter()
    product_type.annotate(df)
    logger.info(
        "Product types derived in %.1fs: %s",
        time.perf_counter() - started, product_type.type_counts(df),
    )
    # The 2026-10-19 columns (gender, audience, style, colour family): value counts only.
    # A catalog without `gender` / `color` (an App or notebook load) shows up here as a
    # gender read from titles alone or an all-'none' colour family, not as a silent rail
    logger.info("Annotations derived: %s", product_type.annotation_counts(df))
    _annotate_taxonomy(df, csv_hash)


# ------------------------------------------------------------------ taxonomy cache
# The 2026-10-27 columns cost 2-3x product_type.annotate on the laptop (~6-16 s for
# 299,972 rows, CPU-load dependent), over the 1.5x boot budget, so the LOCAL backend
# keeps them in .cache/, keyed by the CSV's md5 (as the embedding cache) and a digest
# of app/catalog_taxonomy.py itself (every rule lives there: it imports no app module;
# only while the file is the one this process imported, see _taxonomy_digest),
# this format and the Python minor version (its `re` parser derives the rule index): any
# edit of a rule, the catalog or this format is a miss, and a miss (or an unreadable
# file) just derives them again. They stay derived data: never written to the CSV, never
# embedded. Databricks derives them at every load (no csv_hash there). Old files are
# not pruned, like the embedding cache's (~10 MB each).
_TAXONOMY_CACHE_FORMAT = "1"


def _taxonomy_digest() -> Optional[str]:
    """The rules part of the cache key, or None (no cache: derive). The key must name the
    rules that RUN, and the file on disk is those rules only while it is no newer than
    catalog_taxonomy.IMPORTED_AT: an edit after the import (a ~20 s boot, a notebook that
    imported the module, agents editing rules while others boot) filed the old rules'
    columns under the new rules' name and every later boot served them (2026-10-30
    review). Read first, then stat, so a read racing an edit is refused too. An unreadable
    file is no cache either, never a failed boot."""
    try:
        path = Path(catalog_taxonomy.__file__)
        source = path.read_bytes()
        if path.stat().st_mtime > catalog_taxonomy.IMPORTED_AT:
            return None
    except Exception:  # noqa: BLE001 -- no digest is no cache, never a failed boot
        return None
    salt = f"{_TAXONOMY_CACHE_FORMAT}|py{sys.version_info[0]}.{sys.version_info[1]}"
    return hashlib.md5(source + salt.encode()).hexdigest()[:12]


def _taxonomy_cache_path(csv_hash: str) -> Optional[Path]:
    digest = _taxonomy_digest()
    return CACHE_DIR / f"taxonomy_{csv_hash}_{digest}.npz" if digest else None


def _encode_column(values) -> tuple:
    """(int32 codes, distinct values): ~300K str / tuple / None cells stored compactly."""
    codes, uniques, seen = np.empty(len(values), dtype=np.int32), [], {}
    for i, value in enumerate(values):
        code = seen.get(value)
        if code is None:
            code = seen[value] = len(uniques)
            uniques.append(value)
        codes[i] = code
    return codes, uniques


def _decode_column(codes, uniques, as_tuples: bool, index) -> pd.Series:
    table = np.empty(len(uniques), dtype=object)
    for i, value in enumerate(uniques):   # element-wise: a tuple must stay one cell
        table[i] = tuple(value) if as_tuples else value
    return pd.Series(table[codes] if len(codes) else [], index=index, dtype=object)


_TUPLE_COLUMNS = (catalog_taxonomy.DIET_COLUMN, catalog_taxonomy.WORKS_WITH_COLUMN)


def _load_taxonomy_cache(df: pd.DataFrame, path: Path) -> bool:
    """Set the taxonomy columns from a cache file; False (nothing set) on any doubt. The
    file is plain arrays + JSON (np.load without pickle): no code runs from .cache/."""
    names = catalog_taxonomy.ADDED_COLUMNS
    try:
        with np.load(path, allow_pickle=False) as stored:
            meta = json.loads(stored["meta"].tobytes().decode("utf-8"))
            if meta.get("rows") != len(df) or tuple(meta.get("columns", ())) != names:
                return False
            decoded = {name: _decode_column(stored[f"codes_{i}"], meta["uniques"][i],
                                            name in _TUPLE_COLUMNS, df.index)
                       for i, name in enumerate(names)}
    except Exception as exc:  # noqa: BLE001 -- a bad cache is a miss, never a failed boot
        logger.warning("Taxonomy cache unreadable (%s); deriving again.", type(exc).__name__)
        return False
    for name in names:
        df[name] = decoded[name]
    return True


def _save_taxonomy_cache(df: pd.DataFrame, path: Path) -> None:
    names = catalog_taxonomy.ADDED_COLUMNS
    try:
        encoded = [_encode_column(df[name].tolist()) for name in names]
        meta = {"rows": len(df), "columns": list(names),
                "uniques": [[list(v) if isinstance(v, tuple) else v for v in uniques]
                            for _codes, uniques in encoded]}
        arrays = {f"codes_{i}": codes for i, (codes, _uniques) in enumerate(encoded)}
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        # per process: two boots writing at once each rename a whole file into place
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
    except Exception as exc:  # noqa: BLE001 -- a cache that cannot be written is no error
        logger.warning("Taxonomy cache not written (%s).", type(exc).__name__)


def _annotate_taxonomy(df: pd.DataFrame, csv_hash: Optional[str] = None) -> None:
    started = time.perf_counter()
    path = _taxonomy_cache_path(csv_hash) if csv_hash else None
    if csv_hash and path is None:
        logger.warning("Taxonomy cache off: app/catalog_taxonomy.py is unreadable or newer "
                       "than its import; deriving.")
    hit = path is not None and path.exists() and _load_taxonomy_cache(df, path)
    if not hit:
        catalog_taxonomy.annotate_catalog(df)
        if path is not None:
            _save_taxonomy_cache(df, path)
    elapsed = time.perf_counter() - started
    if path is not None:
        logger.info("Taxonomy cache %s: %s", "hit" if hit else "miss", path.name)
    logger.info("Catalog taxonomy derived in %.1fs: %s", elapsed,
                catalog_taxonomy.taxonomy_counts(df))


def _backend() -> str:
    return (cfg.SEARCH_BACKEND or "local").lower()


def _ready() -> bool:
    if _df is None or _loaded_backend != _backend():
        return False
    return _backend() == "databricks" or _embeddings is not None


def load_index(fields: Tuple[str, ...] = DEFAULT_SEARCH_FIELDS) -> None:
    """Build the in-memory index once. Rebuilds if the field set or backend changes."""
    global _df, _embeddings, _loaded_fields, _loaded_backend, _embedded_fields, _csv_hash_of

    if _backend() == "databricks":
        if _df is not None and _loaded_backend == "databricks":
            return
        if fields != DEFAULT_SEARCH_FIELDS:
            # The index holds vectors of the search_text column databricks/01 built
            logger.warning(
                "SEARCH_BACKEND=databricks ignores fields=%r; the index holds the "
                "search_text vectors it was built from.", fields,
            )
        from app import vector_search

        df = vector_search.load_catalog()
        embedded = effective_fields(df.columns)
        _annotate_types(df)
        _df = df
        _embeddings = None
        _loaded_fields = DEFAULT_SEARCH_FIELDS
        _embedded_fields = embedded
        _loaded_backend = "databricks"
        _csv_hash_of = None          # no CSV here: the fingerprint hashes text_hash
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
    read_state = _file_state(PRODUCTS_CSV)
    # low_memory=False: the enriched catalog's sparse tag columns otherwise get
    # per-chunk mixed dtypes (DtypeWarning)
    df = pd.read_csv(PRODUCTS_CSV, low_memory=False)
    embedded = effective_fields(df.columns, fields)
    csv_hash = _file_hash(PRODUCTS_CSV)
    # The taxonomy cache files its columns under this hash, so the hash must name the
    # bytes read: a catalog rewritten while it was read or hashed (an enrich run during a
    # boot) gives the taxonomy no key -- derive, write nothing (2026-10-30 review, as for
    # a rules edit after import). The embedding cache below is unchanged.
    taxonomy_hash = csv_hash if read_state and _file_state(PRODUCTS_CSV) == read_state else None
    if taxonomy_hash is None:
        logger.warning("Taxonomy cache off: the catalog changed while it was read; deriving.")
    # BEFORE _build_search_text, which fills missing text fields with '' in place;
    # annotate() should read the catalog as loaded (it is case-insensitive anyway).
    _annotate_types(df, taxonomy_hash)
    df["search_text"] = _build_search_text(df, embedded)

    cache_path = CACHE_DIR / f"embeddings_{_cache_key(embedded, csv_hash)}.npy"
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
    _embedded_fields = embedded
    _loaded_backend = "local"
    # The verified hash only (taxonomy_hash): when the catalog changed while it was read,
    # catalog_fingerprint() hashes the file as it always did
    _csv_hash_of = (df, taxonomy_hash) if taxonomy_hash else None
    logger.info("Index ready: %d products, dim=%d.", len(df), embeddings.shape[1])


def title_key(title) -> str:
    """What two rows must share to count as one listing: the case-folded title, ''
    when missing. The catalog sells some listings under several casings ("AIMERDAY
    Racerback Sports Bras ..." / "Aimerday ..."); the local backend once lowercased
    titles in place, which merged them by accident, so every title dedup (here, main.py,
    recommendations.py) compares this key, on both backends."""
    if title is None or (isinstance(title, float) and math.isnan(title)):
        return ""
    return str(title).casefold()


# ------------------------------------------------------------------ style key (variants)
# The catalog audit's STRICT variant key (2026-10-19, scratch audit variants.py): a seller
# lists one dress as many rows that differ only by colour, size or pack words ("HUHOT
# Floral Summer Dress ... XX-Large / Small / Medium"; one maxi dress in 12 rows), and
# parent_asin is unique per row, so nothing else groups them. It kept model numbers of 3+
# digits and checked 12 of 12 sampled groups as true variants, with 0 false merges among
# the battery's page-1 duplicates. Word lists are English-only and hand-maintained.
_VARIANT_COLOURS = (
    "black white red blue navy green yellow pink purple brown beige grey gray gold golden "
    "silver rose orange khaki tan ivory cream maroon burgundy wine teal turquoise aqua coral "
    "peach mint olive lavender lilac nude camel charcoal multicolor multicolour multicolored "
    "multi apricot coffee champagne fuchsia magenta violet plum mustard emerald sky royal hot "
    "light dark deep bright army jade cyan indigo taupe sand stone mocha chocolate rust copper "
    "bronze gunmetal denim heather leopard camo camouflage clear transparent rainbow pastel "
    "neon off"
).split()
_VARIANT_SIZES = (
    "xxs xs s m l xl xxl xxxl xxxxl 1x 2x 3x 4x 5x 1xl 2xl 3xl 4xl 5xl 6xl small medium "
    "large x-large xx-large xxx-large x-small xx-small xlarge xxlarge xsmall regular "
    "plus-size one-size onesize os osfa osfm us uk eu size sizes sz"
).split()
_V_COLOUR_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, _VARIANT_COLOURS)) + r")\b")
_V_SIZE_RE = re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, _VARIANT_SIZES)) + r")(?![\w-])")
_V_PACK_RE = re.compile(
    r"\b(?:pack of \d+|\d+ ?(?:-? ?pack|pk|pcs?|pieces?|pairs?|count|ct|set))\b")
_V_AGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:-|to|/)\s*\d+(?:\.\d+)?\s*(?:months?|mos?|years?|yrs?|y|t|m)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:months?|years?|yrs?|t|y)\b")
_V_SHOE_SIZE_RE = re.compile(
    r"\b\d{1,2}(?:\.5)?\s*(?:b|m|w|d|e{1,4}|c)?\s*(?:\(m\)|\(w\)|\(b\))?(?=[\s,)/]|$)")
_V_PUNCT_RE = re.compile(r"[^\w\s]")
_V_WS_RE = re.compile(r"\s+")


def _variant_title_key(title) -> str:
    """The title with pack counts, ages, colour words, size tokens and punctuation removed
    (the audit's strict_key, step for step: colour and size words go twice, because
    dropping punctuation can expose new ones)."""
    if title is None or (isinstance(title, float) and math.isnan(title)):
        return ""
    t = str(title).casefold().replace("’", "'").replace("&", " and ")
    t = _V_PACK_RE.sub(" ", t)
    t = _V_AGE_RE.sub(" ", t)
    t = _V_COLOUR_RE.sub(" ", t)
    t = _V_SIZE_RE.sub(" ", t)
    t = _V_SHOE_SIZE_RE.sub(" ", t)
    t = _V_PUNCT_RE.sub(" ", t)
    t = _V_COLOUR_RE.sub(" ", t)
    t = _V_SIZE_RE.sub(" ", t)
    return _V_WS_RE.sub(" ", t).strip()


def _variant_store_key(store) -> str:
    if store is None or (isinstance(store, float) and math.isnan(store)):
        return ""
    return _V_WS_RE.sub(" ", _V_PUNCT_RE.sub(" ", str(store).casefold())).strip()


@lru_cache(maxsize=65536)
def _style_key_cached(store: str, title: str) -> str:
    stripped = _variant_title_key(title)
    if not stripped:
        return ""
    return _variant_store_key(store) + "\x1f" + stripped


def style_key(store, title) -> str:
    """What two rows must share to be colour/size variants of one product: the seller
    (case and punctuation folded) plus the title without colour, size and pack words.
    '' when nothing of the title is left (such a row is never merged). A row without a
    store groups by its title alone, as a shopper would see it. Pure; NaN reads as
    missing. main.py collapses Fashion rows only (VARIANT_COLLAPSE)."""
    store = "" if store is None or (isinstance(store, float) and math.isnan(store)) else str(store)
    title = "" if title is None or (isinstance(title, float) and math.isnan(title)) else str(title)
    return _style_key_cached(store, title)


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
    keep: Optional[Callable[[dict, str], bool]] = None,
) -> List[dict]:
    """Scatter-gather: top_k per intent, merged, deduped by title, score-sorted.

    `keep(row, term)` (bundles) drops a hit before it counts toward
    per_intent_quota, so a slot's quota fills with rows of the right kind rather
    than being spent on, say, barefoot-sandal anklets in a 'sandals' slot. It only
    filters the top_k hits already fetched for that intent; it never fetches deeper.
    None (the default) leaves the loop exactly as before."""
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
            if keep is not None and not keep(item, term):
                continue
            rows.append(item)
            kept += 1

    if not rows:
        return []

    final = pd.DataFrame(rows).sort_values("score", ascending=False)
    # the best-scoring casing of a title wins, keeping the seller's display text
    final = final[~final["Product_title"].map(title_key).duplicated()]
    return final.replace({np.nan: None}).to_dict(orient="records")


def _raw_hits(search_terms: List[str], top_k: int) -> List[List[Tuple[int, float]]]:
    """Per-intent [(row_position, score)] from the active backend, best first."""
    if _backend() == "databricks":
        from app import vector_search

        return vector_search.query_many(search_terms, top_k)
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
