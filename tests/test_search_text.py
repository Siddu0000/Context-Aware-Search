"""Offline tests for the local catalog's display casing (app/search.py) and the API log
config (uvicorn_log.json).

No network, no LLM, no embedding model (the embedder and retrieval are faked). Runs
under pytest (`python -m pytest tests/test_search_text.py`) or plainly
(`python tests/test_search_text.py`).

Pinned here:
- _build_search_text returns BYTE-IDENTICAL text to the old in-place version. The
  embedding cache key hashes the CSV file, not this text, so a changed text would be
  served the old vectors silently: this test is the only guard.
- The display columns keep the seller's casing (the local app used to show
  "kalawen watch winder for rolex ..."); missing text fields still read '' (the
  local row shape _product_gender, the reranker and BM25 were written against).
- load_index serves original-case rows from the same cache file, with no re-encode.
- The load-time columns (product_type's and, since 2026-10-27, app/catalog_taxonomy.py's)
  never reach the embedded fields or the text; a second local boot reads the taxonomy
  columns from their own .cache/ file.
- catalog_fingerprint() returns the hash load_index took of the bytes it read (no second
  read of the 228 MB file at API start, 2026-10-30), and hashes the file only when that
  hash is unknown; a Databricks load keeps its text_hash fingerprint.
- uvicorn_log.json: loads through uvicorn's own Config, creates logs/, sends app and
  uvicorn.error records to logs/api.log, and keeps uvicorn.access (URLs carry the
  query) on the console only.
CAS_SLOW_TESTS=1 adds the old-vs-new text comparison on the real catalog (~1-2 min).
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import app.config as cfg  # noqa: E402
from app import search as S  # noqa: E402

LOG_CONFIG = ROOT / "uvicorn_log.json"


class _Skipped(Exception):
    """Raised by _skip() under the plain runner (it reports it), pytest or not."""


_PLAIN_RUNNER = False  # set by __main__: pytest.skip() there would abort the whole run


def _skip(reason):
    if _PLAIN_RUNNER:
        raise _Skipped(reason)
    try:
        import pytest
    except ImportError:
        raise _Skipped(reason)
    pytest.skip(reason)


@contextlib.contextmanager
def patched(obj, attrs):
    """Set attributes for one test and always restore them."""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _old_build_search_text(df, fields):
    """app/search.py::_build_search_text as of f75b3a6 (2026-10-15), verbatim but for
    the log line: it lowercased the display columns IN PLACE."""
    for col in fields:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.lower()
    return df[list(fields)].agg(" ".join, axis=1)


def _md5(series):
    return hashlib.md5("\n".join(series.tolist()).encode("utf-8")).hexdigest()


NAN = float("nan")


def _frame(with_occasion=True):
    """Catalog-shaped rows: Title-case verticals, NaN gaps, numbers in a text field,
    and Unicode whose lowercasing changes length or depends on the next letter
    (dotted capital I, final sigma, fullwidth, titlecase digraphs)."""
    rows = {
        "bsns_vrtcl_name": ["Fashion", "Electronics", "Grocery", "Fashion", NAN, "Fashion"],
        "categ_lvl2_name": ["Women's Clothing", "Portable Power", NAN, "Men's Clothing",
                            "", "Women's Clothing"],
        "Product_title": [
            "KALAWEN Watch Winder for Rolex, Black PU Leather",
            "Anker PowerCore 10000 Portable Charger",
            "Lindt EXCELLENCE 70% Cocoa Dark Chocolate Bar",
            "ΟΔΟΣ Men's Linen Shirt ΣΑΣ",
            NAN,
            "İstanbul STRASSE ß Dress ＡＢＣ ǅemal",
        ],
        "prod_description": ["Fits Automatic WATCHES.", NAN, "Rich & Smooth", "", "x", NAN],
        "color": ["Black", NAN, "", "White", NAN, "Navy Blue"],
        "material": ["PU Leather", NAN, NAN, "Linen", NAN, "Cotton"],
        "gender": ["Women", NAN, NAN, "men", NAN, "women"],
        "use_case": ["Gift", "Travel", NAN, NAN, NAN, "Wedding Guest"],
        "dietary": [NAN, NAN, "Gluten-Free", NAN, NAN, NAN],
        "specs": [1.5, NAN, "Weight: 3.5 OZ", NAN, 7, NAN],
        "gen_description": [NAN] * 6,
        "price": [39.99, NAN, 4.5, 20.0, NAN, 59.0],
        "average_rating": [4.4, 4.7, NAN, 3.9, NAN, 4.1],
        "rating_number": [120, 9000, NAN, 15, NAN, 33],
        "parent_asin": ["B0001", "B0002", "B0003", "B0004", "B0005", "B0006"],
    }
    if with_occasion:
        rows["occasion"] = ["Formal", NAN, NAN, "Beach", NAN, "Wedding"]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the text itself
def test_search_text_is_byte_identical_to_the_old_in_place_version():
    for with_occasion in (True, False):
        old_df, new_df = _frame(with_occasion), _frame(with_occasion)
        fields = S.effective_fields(new_df.columns)
        assert "occasion" in fields  # a DEFAULT field: added as '' when the CSV lacks it
        old = _old_build_search_text(old_df, fields)
        new = S._build_search_text(new_df, fields)
        assert new.tolist() == old.tolist()
        assert _md5(new) == _md5(old)
        # an explicit (ablation) field list, as eval --exclude-description passes
        ablation = tuple(f for f in S.DEFAULT_SEARCH_FIELDS if f != "prod_description")
        assert (S._build_search_text(_frame(with_occasion), ablation).tolist()
                == _old_build_search_text(_frame(with_occasion), ablation).tolist())


def test_load_time_columns_never_reach_search_text():
    """Every column load_index derives (product_type's and, since 2026-10-27, the
    taxonomy's: item_category, brand, ...) stays out of the embedded fields and leaves
    the text byte-identical: the embedding cache key would not notice a change."""
    from app import catalog_taxonomy, product_type
    for with_occasion in (True, False):
        plain, derived = _frame(with_occasion), _frame(with_occasion)
        fields = S.effective_fields(plain.columns)
        with tempfile.TemporaryDirectory() as tmp, patched(S, {"CACHE_DIR": Path(tmp)}):
            S._annotate_types(derived)
        added = set(derived.columns) - set(plain.columns)
        assert set(catalog_taxonomy.ADDED_COLUMNS) | {product_type.TYPE_COLUMN} <= added
        assert S.effective_fields(derived.columns) == fields
        assert not added & set(fields), added & set(fields)
        pd.testing.assert_frame_equal(derived[plain.columns], plain)   # sources untouched
        text_plain = S._build_search_text(plain, fields)
        text_derived = S._build_search_text(derived, fields)
        assert text_derived.tolist() == text_plain.tolist()
        assert _md5(text_derived) == _md5(_old_build_search_text(_frame(with_occasion), fields))


def test_search_text_is_lowercase():
    df = _frame()
    text = S._build_search_text(df, S.effective_fields(df.columns))
    assert text[0].startswith("fashion women's clothing kalawen watch winder for rolex")
    assert all(t == t.lower() for t in text)


def test_display_columns_keep_their_case():
    raw = _frame()
    df = _frame()
    fields = S.effective_fields(df.columns)
    S._build_search_text(df, fields)
    assert df.at[0, "Product_title"] == "KALAWEN Watch Winder for Rolex, Black PU Leather"
    assert df.at[3, "Product_title"] == "ΟΔΟΣ Men's Linen Shirt ΣΑΣ"
    assert list(df["bsns_vrtcl_name"]) == ["Fashion", "Electronics", "Grocery", "Fashion", "",
                                           "Fashion"]
    for col in fields:
        expected = raw[col].fillna("").astype(str) if col in raw.columns else ""
        assert (df[col] == expected).all(), col


def test_missing_text_fields_still_read_as_empty_strings():
    """The one in-place step kept: the local row shape. A missing DEFAULT column is
    added as ''; columns that are not embedded (price, rating) are left alone."""
    df = _frame(with_occasion=False)
    S._build_search_text(df, S.effective_fields(df.columns))
    assert (df["occasion"] == "").all()
    assert df.at[1, "color"] == "" and df.at[4, "Product_title"] == ""
    assert df.at[0, "specs"] == "1.5"
    assert math.isnan(df.at[1, "price"]) and math.isnan(df.at[2, "average_rating"])
    assert df["parent_asin"].tolist() == _frame()["parent_asin"].tolist()


# ------------------------------------------------------------------ load_index
class _FakeEmbedder:
    model_name = "fake/model"


_INDEX_STATE = ("_df", "_embeddings", "_loaded_backend", "_loaded_fields", "_embedded_fields",
                "_csv_hash_of")


@contextlib.contextmanager
def _isolated_index():
    """Run a load_index test without disturbing a real loaded index."""
    saved = {k: getattr(S, k) for k in _INDEX_STATE}
    old_backend = cfg.SEARCH_BACKEND
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(S, k, v)
        cfg.SEARCH_BACKEND = old_backend


def _no_encode(texts):
    raise AssertionError(f"re-encoded {len(texts)} texts: the cache should have been hit")


def test_load_index_serves_original_case_from_the_same_cache_file():
    with tempfile.TemporaryDirectory() as tmp, _isolated_index():
        csv = Path(tmp) / "catalog.csv"
        _frame().to_csv(csv, index=False)
        cfg.SEARCH_BACKEND = "local"
        S._df = None
        fake_embed = lambda texts: np.ones((len(texts), 4), dtype=np.float32)  # noqa: E731
        common = {"PRODUCTS_CSV": csv, "CACHE_DIR": Path(tmp),
                  "get_default_embedder": lambda: _FakeEmbedder()}
        with patched(S, dict(common, embed_text=fake_embed)):
            S.load_index()
            fields = S.effective_fields(pd.read_csv(csv).columns)
            cache = Path(tmp) / f"embeddings_{S._cache_key(fields)}.npy"
            assert cache.exists()
            first_text = S._df["search_text"].tolist()

        # The text the vectors were encoded from is the old in-place text
        reference = _frame()
        assert first_text == _old_build_search_text(reference, fields).tolist()

        df = S._df
        assert df.at[0, "Product_title"] == "KALAWEN Watch Winder for Rolex, Black PU Leather"
        assert df.at[0, "bsns_vrtcl_name"] == "Fashion" and df.at[0, "color"] == "Black"
        item = S.get_product(0)
        assert item["Product_title"].startswith("KALAWEN") and item["bsns_vrtcl_name"] == "Fashion"
        assert S.get_product(1)["price"] is None and S.get_product(1)["color"] == ""

        with patched(S, {"_raw_hits": lambda terms, k: [[(2, 0.9), (0, 0.8)] for _ in terms]}):
            rows = S.search_products(["anything"], top_k=2)
        assert [r["Product_title"] for r in rows] == [
            "Lindt EXCELLENCE 70% Cocoa Dark Chocolate Bar",
            "KALAWEN Watch Winder for Rolex, Black PU Leather"]

        first_taxonomy = {c: S._df[c].tolist() for c in S.catalog_taxonomy.ADDED_COLUMNS}
        assert first_taxonomy["item_category"] == [None, "power", "candy", None, None, None]
        assert first_taxonomy["watch_kind"][0] == "storage"   # the winder, on Fashion too

        # A second boot reads both cache files: no encode, no taxonomy pass, same text
        def _no_taxonomy(df):
            raise AssertionError("taxonomy derived again: its .cache/ file should have been hit")

        S._df = None
        with patched(S, dict(common, embed_text=_no_encode)), \
                patched(S.catalog_taxonomy, {"annotate_catalog": _no_taxonomy}):
            S.load_index()
        assert S._df["search_text"].tolist() == first_text
        assert S._df.at[0, "Product_title"].startswith("KALAWEN")
        assert {c: S._df[c].tolist() for c in first_taxonomy} == first_taxonomy


def test_catalog_fingerprint_reuses_the_hash_load_index_took():
    """The API's start-up config line (main.effective_config) and run_eval read
    catalog_fingerprint() right after load_index, which had just hashed the CSV: the 228 MB
    file was read a second time at every start (~0.75-0.8 s; 2026-10-30 review). Now the
    load's own hash is returned, the same md5[:12], and it names the bytes LOADED. The file
    is still hashed when that hash is unknown: a frame set another way, or a catalog that
    changed while it was read. A Databricks load keeps its text_hash fingerprint."""
    from app import vector_search
    with tempfile.TemporaryDirectory() as tmp, _isolated_index():
        csv = Path(tmp) / "catalog.csv"
        _frame().to_csv(csv, index=False)
        loaded = S._file_hash(csv)
        real_hash, hashed = S._file_hash, []

        def counting_hash(path):
            hashed.append(Path(path).name)
            return real_hash(path)

        cfg.SEARCH_BACKEND = "local"
        S._df = None
        fake_embed = lambda texts: np.ones((len(texts), 4), dtype=np.float32)  # noqa: E731
        common = {"PRODUCTS_CSV": csv, "CACHE_DIR": Path(tmp), "_file_hash": counting_hash,
                  "get_default_embedder": lambda: _FakeEmbedder(), "embed_text": fake_embed}
        with patched(S, common):
            S.load_index()
            assert hashed == ["catalog.csv"]              # load_index's one md5 of the file
            assert S.catalog_fingerprint() == loaded
            assert hashed == ["catalog.csv"]              # reused: not read again

            # a catalog rewritten after the load: the fingerprint names what is served
            _frame().iloc[:3].to_csv(csv, index=False)
            assert real_hash(csv) != loaded and S.catalog_fingerprint() == loaded
            assert hashed == ["catalog.csv"]

            # a frame load_index did not read (a test, an eval patch): hashed as before
            S._df = S._df.copy()
            assert S.catalog_fingerprint() == real_hash(csv)
            assert hashed == ["catalog.csv"] * 2

            # a catalog that changed while it was read: no verified hash, so the file is
            # hashed at fingerprint time, as it always was
            S._df = None
            ticks = iter(range(10))
            with patched(S, {"_file_state": lambda path: (next(ticks), 0)}):
                S.load_index()
            assert S._csv_hash_of is None
            del hashed[:]
            assert S.catalog_fingerprint() == real_hash(csv) and hashed == ["catalog.csv"]

            # back to a clean load, then Databricks: the CSV's hash never names its frame
            S._df = None
            S.load_index()
            assert S._csv_hash_of is not None and S.catalog_fingerprint() == real_hash(csv)
            remote = _frame().assign(text_hash=[f"h{i}" for i in range(6)])
            cfg.SEARCH_BACKEND = "databricks"
            del hashed[:]
            with patched(vector_search, {"load_catalog": lambda: remote}):
                S.load_index()
            assert S._loaded_backend == "databricks" and S._csv_hash_of is None
            text_hashes = "\n".join(f"h{i}" for i in range(6))
            expected = hashlib.md5(text_hashes.encode()).hexdigest()[:12]
            assert S.catalog_fingerprint() == expected and hashed == []


# ------------------------------------------------------------------ title dedup
# The catalog sells 36 titles under two casings (products_enriched.csv: 297,176 exact vs
# 297,140 case-folded titles). The in-place lowercasing merged them by accident; with
# display case kept, every title dedup must compare case-folded keys (review 2026-10-15:
# both casings were served at ranks 1 and 2, and the "Similar items" strip offered the
# anchor's own listing under its other casing).
VARIANT_A = "AIMERDAY Racerback Sports Bras for Women High Impact"
VARIANT_B = "Aimerday Racerback Sports Bras for Women High Impact"


def _variant_frame():
    return pd.DataFrame({
        "bsns_vrtcl_name": ["Fashion"] * 4,
        "Product_title": [VARIANT_A, VARIANT_B, "Women's Running Shorts", NAN],
        "average_rating": [4.0, 4.9, 4.5, 4.0], "rating_number": [80, 5000, 300, 10],
        "parent_asin": ["B07S2HFSSM", "B07RXB7L7G", "B0SHORTS", "B0NOTITLE"],
    })


def test_title_key():
    assert S.title_key(VARIANT_A) == S.title_key(VARIANT_B) == VARIANT_A.casefold()
    assert S.title_key("STRASSE") == S.title_key("Straße")
    assert S.title_key(None) == S.title_key(NAN) == S.title_key("") == ""


def test_search_products_dedups_case_variants_keeping_the_display_case():
    df = _variant_frame()
    hits = {"bra": [(1, 0.93), (0, 0.92), (2, 0.80)], "sports bra": [(0, 0.95), (3, 0.5)]}
    with patched(S, {"_df": df, "_ready": lambda: True,
                     "_raw_hits": lambda terms, k: [hits[t] for t in terms]}):
        one = S.search_products(["bra"])
        both = S.search_products(["bra", "sports bra"])
    # the best-scoring casing wins, as the seller wrote it
    assert [r["Product_title"] for r in one] == [VARIANT_B, "Women's Running Shorts"]
    assert [r["Product_title"] for r in both] == [VARIANT_A, "Women's Running Shorts", None]
    assert [r["catalog_index"] for r in both] == [0, 2, 3]


def test_main_and_recommendations_compare_title_keys():
    import app.main as main
    import app.recommendations as recs

    anchor = dict(_variant_frame().iloc[0], catalog_index=0, Product_title=VARIANT_A)
    variant = dict(anchor, Product_title=VARIANT_B, catalog_index=1, parent_asin="B07RXB7L7G",
                   average_rating=4.9, rating_number=5000, score=0.99)
    other = {"Product_title": "Women's Running Shorts", "catalog_index": 2, "score": 0.9,
             "average_rating": 3.0, "rating_number": 60, "parent_asin": "B0SHORTS"}

    # the reranked pool's title is not repeated in the embedding-order tail
    full = main._assemble_full([anchor], [variant, other])
    assert [r["Product_title"] for r in full] == [VARIANT_A, other["Product_title"]]
    # nor is a setup refill's
    with patched(main, {"search_products": lambda terms, **kw: [variant, other]}):
        merged = main._refill_empty_slots([dict(anchor, source_intent="a", score=0.5)],
                                          ["a", "b"])
    assert [r["Product_title"] for r in merged] == [other["Product_title"], VARIANT_A]

    # the anchor's other casing is neither a "similar item" nor its "higher-rated pick"
    with patched(recs, {"search_products": lambda terms, top_k=30: [variant, other]}):
        out = recs.recommend("sports bra", [anchor], use_llm=False)
    assert [r["Product_title"] for r in out["cross_sell"]] == [other["Product_title"]]
    assert out["upsell"] == [], out["upsell"]   # the variant (4.9 from 5,000) is excluded


# ------------------------------------------------------------------ style key (2026-10-19)
# search.style_key is the catalog audit's STRICT variant key (VARIANT_COLLAPSE, app/main.py):
# store + title without colour, size, age and pack words. Variants of one listing share it;
# other products, other sellers and model numbers of 3+ digits keep theirs.
def test_style_key_groups_colour_and_size_variants():
    k = S.style_key
    huhot = k("HUHOT", "HUHOT Floral Summer Dresses for Women Casual Beach Sundress XX-Large")
    assert huhot == k("Huhot", "HUHOT Floral Summer Dresses for Women Casual Beach Sundress Small")
    assert huhot == k("HUHOT ", "HUHOT Floral Summer Dresses for Women Casual Beach Sundress, Blue")
    assert huhot == "huhot\x1fhuhot floral summer dresses for women casual beach sundress", huhot
    # pack counts, ages, shoe sizes and the curly apostrophe
    assert k("Hanes", "Hanes Men's Crew Socks, 6 Pack, Black") == \
        k("Hanes", "Hanes Men’s Crew Socks 12 pack White")
    assert k("Carter's", "Carter's Baby Girls' Dress 3-6 Months Pink") == \
        k("Carter's", "Carter's Baby Girls' Dress 12 Months Ivory")
    assert k("Brooks", "Brooks Ghost 14 Running Shoe 9.5 D") == \
        k("Brooks", "Brooks Ghost 14 Running Shoe 11 D")


def test_style_key_keeps_other_products_apart():
    k = S.style_key
    dress = k("HUHOT", "HUHOT Floral Summer Dress Small")
    assert dress != k("HUHOT", "HUHOT Linen Pants Small")                # another product
    assert dress != k("Zattcas", "HUHOT Floral Summer Dress Small")      # another seller
    assert k("Casio", "Casio Watch F-91W") != k("Casio", "Casio Watch A-158W")  # models
    assert k(None, "Floral Summer Dress Red") == k(float("nan"), "Floral Summer Dress Blue")
    # nothing of the title left: never merged
    for title in ("", None, float("nan"), "Black / White - XL", "S M L"):
        assert k("Acme", title) == "", title


def test_style_key_matches_the_audit_reference():
    """Ported from the audit's variants.py strict_key + store_key; the 2026-10-19 battery
    numbers were measured with that file. Checked against it when it is on this disk."""
    ref_path = os.getenv("CAS_AUDIT_VARIANTS_PY")
    if not ref_path or not Path(ref_path).exists():
        _skip("set CAS_AUDIT_VARIANTS_PY to the audit's variants.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit_variants", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    titles = ["HUHOT Floral Summer Dresses XX-Large", "Men's 3-Pack Boxer Briefs, Navy/Grey, L",
              "Girls 2-3T Tutu Dress (Toddler/Little Kid)", "Nike Air Max 270 Size 10.5",
              "Women's Plus-Size Off-White Blouse 1X", "ASICS GT-2000 9 Women's 8.5 B(M)",
              "Pack of 6 Crew Socks", "SAMPLE & CO. Tote — Black", ""]
    for store in ("HUHOT", "Topsville, Inc.", None, "lxg, inc."):
        for title in titles:
            strict = ref.strict_key(title)
            expected = (ref.store_key(store) + "\x1f" + strict) if strict else ""
            assert S.style_key(store, title) == expected, (store, title)


# ------------------------------------------------------------------ the real catalog
def test_real_catalog_text_is_unchanged():
    """The whole enriched catalog through both versions: same text, so the cached
    vectors still describe it. Opt-in, it reads the full CSV twice."""
    if os.getenv("CAS_SLOW_TESTS") != "1":
        _skip("set CAS_SLOW_TESTS=1 (reads the full catalog twice, ~1-2 min)")
    if not cfg.PRODUCTS_CSV.exists():
        _skip(f"no catalog at {cfg.PRODUCTS_CSV}")
    old_df = pd.read_csv(cfg.PRODUCTS_CSV, low_memory=False)
    new_df = pd.read_csv(cfg.PRODUCTS_CSV, low_memory=False)
    fields = S.effective_fields(new_df.columns)
    # as load_index does: every load-time column first (product_type and, since
    # 2026-10-27, the taxonomy; derived here, never read from .cache/), then the text
    S._annotate_types(new_df)
    assert S.effective_fields(new_df.columns) == fields
    old, new = _old_build_search_text(old_df, fields), S._build_search_text(new_df, fields)
    print(f"\n  search_text md5 {_md5(new)[:8]} over {len(new):,} rows")
    assert _md5(new) == _md5(old)
    for col in fields:  # the display columns differ from the old frame only by case
        assert new_df[col].str.lower().equals(old_df[col]), col


# ------------------------------------------------------------------ uvicorn_log.json
_LOG_PROBE = textwrap.dedent("""
    import json, logging, sys
    sys.path.insert(0, sys.argv[2])  # the repo root, as `uvicorn` run from it has
    import uvicorn
    uvicorn.Config(app="app.main:app", log_config=sys.argv[1])  # configures logging only
    logging.getLogger("app.translator").warning("wp4-canary-app %s", "translator")
    logging.getLogger("uvicorn.error").info("wp4-canary-uvicorn")
    logging.getLogger("httpx").info("wp4-canary-root")
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1:50000", "GET",
        "/search?query=wp4-canary-access", "1.1", 200)
    access = logging.getLogger("uvicorn.access")
    print(json.dumps({"access_handlers": [type(h).__name__ for h in access.handlers],
                      "access_propagate": access.propagate}))
    logging.shutdown()
""")


def test_api_log_config_routes_records():
    """Through uvicorn's own Config (what --log-config does), in a fresh directory with
    no logs/: the config creates it, app / uvicorn.error / root records reach the file,
    the access record (its URL carries the query) reaches only the console."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        _skip("uvicorn not installed")
    with tempfile.TemporaryDirectory() as tmp:
        probe = subprocess.run(
            [sys.executable, "-c", _LOG_PROBE, str(LOG_CONFIG), str(ROOT)],
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
        assert probe.returncode == 0, probe.stderr[-2000:]
        log_file = Path(tmp) / "logs" / "api.log"
        assert log_file.exists(), "logs/api.log was not created"
        text = log_file.read_text(encoding="utf-8")
        assert "WARNING app.translator: wp4-canary-app translator" in text
        assert "wp4-canary-uvicorn" in text and "wp4-canary-root" in text
        assert "wp4-canary-access" not in text
        assert "wp4-canary-access" in probe.stdout  # console only
        info = json.loads(probe.stdout.strip().splitlines()[-1])
        assert info == {"access_handlers": ["StreamHandler"], "access_propagate": False}


def test_api_log_config_shape():
    raw = LOG_CONFIG.read_bytes()
    assert all(b < 128 for b in raw)  # uvicorn opens it with the locale's encoding
    conf = json.loads(raw)
    handler = conf["handlers"]["file"]
    assert handler["class"] == "app.log_handlers.SharedRotatingFileHandler"
    assert handler["filename"] == "logs/api.log"
    assert handler["maxBytes"] == 5 * 1024 * 1024 and handler["backupCount"] == 3
    assert "file" not in conf["loggers"]["uvicorn.access"]["handlers"]
    assert conf["loggers"]["uvicorn.access"]["propagate"] is False
    assert "file" in conf["root"]["handlers"] and "file" in conf["loggers"]["uvicorn"]["handlers"]


def _emit_all(handlers, n):
    """n records, alternating over the handlers; returns the messages sent."""
    sent = []
    for i in range(n):
        msg = f"record {i:03d} " + "x" * 20
        record = logging.LogRecord("app.translator", logging.INFO, __file__, 0, msg, None, None)
        handlers[i % len(handlers)].emit(record)
        sent.append(msg)
    return sent


def _shared_handler(path, failures, **kwargs):
    from app.log_handlers import SharedRotatingFileHandler

    handler = SharedRotatingFileHandler(str(path), encoding="utf-8", **kwargs)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.handleError = failures.append   # what would print '--- Logging error ---'
    return handler


def test_shared_log_file_rotation_loses_no_record():
    """Two writers on one file, as uvicorn --reload (parent + child), --workers or two
    servers from the repo root have. On Windows the stock RotatingFileHandler cannot
    rename a file another handle holds open: past maxBytes every record printed a
    traceback and was lost, and each retry shifted the backups away (Oct 15 review:
    169 of 200 records lost). Two handlers in one process are two OS handles, which is
    what the rename trips on."""
    with tempfile.TemporaryDirectory() as tmp:
        failures = []
        handlers = [_shared_handler(Path(tmp) / "api.log", failures, maxBytes=400,
                                    backupCount=100) for _ in range(2)]
        try:
            sent = _emit_all(handlers, 120)
        finally:
            for handler in handlers:
                handler.close()
        assert failures == [], f"{len(failures)} logging errors"
        lines = [line for f in Path(tmp).glob("api.log*")
                 for line in f.read_text(encoding="utf-8").splitlines()]
        assert sorted(lines) == sorted(sent)          # every record, exactly once
        assert not list(Path(tmp).glob("*.rotating"))


def test_single_writer_rotates_like_the_stock_handler():
    with tempfile.TemporaryDirectory() as tmp:
        failures = []
        handler = _shared_handler(Path(tmp) / "api.log", failures, maxBytes=400, backupCount=3)
        try:
            sent = _emit_all([handler], 200)
        finally:
            handler.close()
        assert failures == []
        files = {f.name: f for f in Path(tmp).iterdir()}
        assert sorted(files) == ["api.log", "api.log.1", "api.log.2", "api.log.3"], sorted(files)
        assert all(f.stat().st_size <= 400 for f in files.values())
        newest = files["api.log"].read_text(encoding="utf-8").splitlines()
        before = files["api.log.1"].read_text(encoding="utf-8").splitlines()
        assert newest[-1] == sent[-1] and before[-1] == sent[-len(newest) - 1]


def test_logs_dir_is_gitignored():
    lines = {ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "logs/" in lines


# ------------------------------------------------------------------ plain runner
if __name__ == "__main__":
    _PLAIN_RUNNER = True
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.ERROR)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except _Skipped as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except Exception as e:  # noqa: BLE001 -- report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:500]}")
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
