"""Offline tests for GET /search filters, sort and facets, and /keyword_search paging.

No LLM, no catalog load, no embedding model: main.search() runs with the translator,
retrieval, the reranker and cross-sell faked, and the keyword engine indexes a small
frame. Filters and sort are applied to the CACHED pool, so every filter/sort call after
the first must make no translator or rerank call -- the fakes count them.

Runs under pytest (`python -m pytest tests/test_api_filters.py`) or plainly
(`python tests/test_api_filters.py`).
"""

import contextlib
import inspect
import math
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_ENABLED", "false")

import pandas as pd  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import app.config as cfg  # noqa: E402
import app.main as main  # noqa: E402
from app import attributes as attrs  # noqa: E402
from app import catalog_taxonomy as ct  # noqa: E402
from app import page_cache  # noqa: E402
from app import product_type as pt  # noqa: E402
from app.keyword_search import KeywordSearchEngine  # noqa: E402
from app.scoring import bayesian_rating  # noqa: E402

_ABSENT = object()


@contextlib.contextmanager
def patched(obj, attrs: dict):
    """setattr for the duration; an attribute the object lacks is removed again after."""
    old = {k: getattr(obj, k, _ABSENT) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is _ABSENT:
                delattr(obj, k)
            else:
                setattr(obj, k, v)


# The style and colour vocabularies the 2026-10-19 rail validates against (contract C1,
# owned by app/product_type.py). These tests are about main.py's contract, so they pin
# small stand-ins; the readers main.py calls on a Fashion pool are product_type's own
# (their effects are tested in tests/test_ranking.py and tests/test_product_type.py).
FAKE_SUBTYPES = {"dresses": "Dresses", "tops_tees": "Tops & tees", "jeans": "Jeans",
                 "sandals": "Sandals"}
FAKE_COLORS = {"black": "#111111", "white": "#FFFFFF", "blue": "#1F5FBF",
               "multicolor": "multicolor"}


def c1_patches() -> dict:
    return {"FINE_SUBTYPE_LABELS": FAKE_SUBTYPES, "COLOR_FAMILIES": FAKE_COLORS}


def item(title, vertical, ptype, price, rating, n, score, gender=None):
    return {
        "Product_title": title, "bsns_vrtcl_name": vertical, "categ_lvl2_name": vertical,
        "product_type": ptype, "price": price, "price_estimate": 9.99,
        "average_rating": rating, "rating_number": n, "gender": gender,
        "parent_asin": "A" + title.replace(" ", "")[:9].upper(), "score": score,
        "source_intent": "beach day", "catalog_index": int(score * 1000),
    }


# Relevance order once reranked: the rerank score is the list order (95, 90, ...)
POOL = [
    item("Linen Midi Dress", "Fashion", "apparel", 39.99, 4.5, 200, 0.95, "women"),
    item("Leather Sandals", "Fashion", "footwear", 25.0, 4.1, 50, 0.94),
    item("Straw Tote Bag", "Fashion", "bag", None, 4.8, 5, 0.93),
    item("Gold Hoop Earrings", "Fashion", "jewelry", 12.5, None, None, 0.92),
    item("Bluetooth Speaker", "Electronics", None, 59.0, 3.2, 1000, 0.91),
    item("Sun Hat", "Fashion", "headwear", float("nan"), 2.5, 10, 0.90),
    item("Beach Snack Mix", "Grocery", None, "$8.99", 4.0, 30, 0.89),
    item("Cotton Sundress", "Fashion", "apparel", 19.0, 1.5, 3, 0.88, "women"),
    # a lowercase vertical (the local shape until 2026-10-15): one department in the facets
    item("Portable Charger", "electronics", None, 15.0, 4.4, 300, 0.87),
    item("Model 7 Sneaker", "Fashion", "unknown", 49.0, 3.9, 80, 0.86),
]
TITLES = [p["Product_title"] for p in POOL]
SPONSORED = {"ABLUETOOTH": {"sponsor": "SoundCo", "bid": 2.0}}   # the speaker


class Fakes:
    def __init__(self, bundle_type=None, min_relevance=0.0, pool=None, scores=None,
                 rerank_error=None):
        self.translate = self.rerank = self.recs = 0
        self.bundle_type = bundle_type
        self.min_relevance = min_relevance
        self.pool = POOL if pool is None else pool
        # {title: rerank score}; default: the POOL order (95, 90, ...)
        self.scores = scores or {t: 95 - 5 * i for i, t in enumerate(TITLES)}
        self.rerank_error = rerank_error   # e.g. "429": the reranker falls back

    def understand_query(self, q, mode=None, errors=None):
        self.translate += 1
        return {"intents": ["beach day"], "bundle_type": self.bundle_type, "constraints": [],
                "is_recipe": self.bundle_type == "recipe", "query": q}

    def search_products(self, terms, top_k=30, per_intent_quota=None, **kwargs):
        return [dict(p) for p in self.pool]

    def llm_rerank(self, query, candidates, top_k, errors=None, constraints=None):
        self.rerank += 1
        window = candidates[: cfg.RERANK_INPUT_K]
        if self.rerank_error:
            # app/reranker.py's fallback: the window in embedding order, unscored
            errors.append({"stage": "rerank", "code": self.rerank_error, "detail": "fake"})
            return [dict(c, rerank_score=None, bayesian_rating=None, final_score=None,
                         reason="(rerank unavailable — embedding score only)")
                    for c in window[:top_k]]
        out = [dict(c, rerank_score=self.scores[c["Product_title"]], bayesian_rating=4.0,
                    final_score=float(self.scores[c["Product_title"]]), reason="ok")
               for c in window]
        return sorted(out, key=lambda c: -c["final_score"])[:top_k]

    def build_recommendations(self, q, results, page=1):
        self.recs += 1
        return {"cross_sell": [], "upsell": [], "cross_sell_source": None, "errors": []}

    def patches(self):
        return {
            "understand_query": self.understand_query,
            "search_products": self.search_products,
            "llm_rerank": self.llm_rerank,
            "build_recommendations": self.build_recommendations,
            "MIN_RESULT_RELEVANCE": self.min_relevance,
            "sponsored_map": lambda: dict(SPONSORED),
        }


@contextlib.contextmanager
def app_fakes(knobs=None, **kwargs):
    fakes = Fakes(**kwargs)
    knobs = {**dict(TYPE_AWARE_RANKING=True, SPONSORED_MIN_RERANK_SCORE=40.0,
                    RERANK_INPUT_K=15, QUERY_GUARD_ENABLED=True,
                    OUTFIT_BUNDLES_ENABLED=False, REJECTS_AFTER_TAIL=True,
                    VARIANT_COLLAPSE=True, WEAK_MATCH_MIN_STRONG=0), **(knobs or {})}
    page_cache.clear()
    main.clear_degraded_pools()
    try:
        with patched(cfg, knobs), patched(main, fakes.patches()), patched(pt, c1_patches()):
            yield fakes
    finally:
        page_cache.clear()
        main.clear_degraded_pools()


def search(**params):
    """main.search() as /chat and /unified_search call it: the new params only if given."""
    args = dict(query="beach day", top_k=4, page=1, rerank=True, recommend=False,
                sponsored=True, translator_mode=None)
    args.update(params)
    return main.search(**args)


def titles(out):
    return [r["Product_title"] for r in out["results"]]


def all_pages(**params):
    out, rows, page = None, [], 1
    while True:
        out = search(page=page, **params)
        rows += titles(out)
        if not out["has_next"]:
            return rows, out
        page += 1


# ------------------------------------------------------------------ relevance + facets
def test_relevance_default_and_facets_over_the_unfiltered_pool():
    with app_fakes() as fakes:
        out = search()
        assert (fakes.translate, fakes.rerank) == (1, 1)
        # relevance = today's order: the sponsored speaker (rr 75 > 40) leads, labelled
        assert titles(out) == ["Bluetooth Speaker", "Linen Midi Dress", "Leather Sandals",
                               "Straw Tote Bag"]
        assert out["results"][0]["is_sponsored"] and out["sponsored"] == [out["results"][0]]
        assert out["sort"] == "relevance" and out["filters_applied"] == {}
        assert out["total_results"] == out["total_unfiltered"] == len(POOL)
        assert (out["total_pages"], out["has_next"]) == (3, True)
        f = out["facets"]
        assert f["departments"] == {"Fashion": 7, "Electronics": 2, "Grocery": 1}, f
        assert f["product_types"] == {"apparel": 2, "footwear": 1, "jewelry": 1, "bag": 1,
                                      "headwear": 1}, f   # 'unknown' is not filterable
        # priced: 39.99, 25, 12.5, 59, '$8.99', 19, 15, 49 (None and NaN are not)
        assert {k: v for k, v in f["price"].items() if k != "bands"} == \
            {"min": 8.99, "max": 59.0, "n_priced": 8}, f
        # bands over the same listed prices, [low, high): $25 is in "$25 to $50" only,
        # and max is high - 1 cent, so sending a band's bounds returns its count
        assert [(b["label"], b["min"], b["max"], b["n"]) for b in f["price"]["bands"]] == [
            ("Up to $25", None, 24.99, 4), ("$25 to $50", 25, 49.99, 3),
            ("$50 to $100", 50, 99.99, 1), ("$100 to $200", 100, 199.99, 0),
            ("$200 & above", 200, None, 0)], f["price"]["bands"]
        for band in f["price"]["bands"]:
            sent = {k: v for k, v in (("min_price", band["min"]), ("max_price", band["max"]))
                    if v is not None}
            assert search(top_k=50, **sent)["total_results"] == band["n"], band
        # rows without the derived columns offer no gender, style, colour or brand
        assert (f["genders"], f["subtypes"], f["colors"], f["stores"], f["n_stores"]) == \
            ([], [], [], [], 0), f
        # cumulative: ratings 4.5 4.1 4.8 3.2 2.5 4.0 1.5 4.4 3.9 (earrings unrated)
        assert f["rating"] == {"4": 5, "3": 7, "2": 8, "1": 9}, f
        # no "message" on a non-empty page
        assert "message" not in out and out["no_match"] is False


def test_filters_and_sorts_reuse_the_cached_pool():
    with app_fakes() as fakes:
        search()
        before = (fakes.translate, fakes.rerank)
        for params in ({"sort": "price_asc"}, {"sort": "rating", "page": 2},
                       {"min_price": 10, "max_price": 40}, {"department": "Grocery"},
                       {"ptype_filter": "bag,jewelry"}, {"top_k": 8}, {"min_rating": 4}):
            search(**params)
        assert (fakes.translate, fakes.rerank) == before == (1, 1)


# ------------------------------------------------------------------ sort
def test_price_sorts_put_unpriced_rows_last_and_never_use_the_estimate():
    with app_fakes():
        rows, out = all_pages(sort="price_asc")
        assert rows == ["Beach Snack Mix", "Gold Hoop Earrings", "Portable Charger",
                        "Cotton Sundress", "Leather Sandals", "Linen Midi Dress",
                        "Model 7 Sneaker", "Bluetooth Speaker",
                        "Straw Tote Bag", "Sun Hat"], rows   # unpriced last, relevance order
        assert out["sort"] == "price_asc"
        rows, _ = all_pages(sort="price_desc")
        assert rows[:3] == ["Bluetooth Speaker", "Model 7 Sneaker", "Linen Midi Dress"]
        assert rows[-2:] == ["Straw Tote Bag", "Sun Hat"], rows
        # a sponsored row keeps its label but is not forced to the front
        first = search(sort="price_asc")
        assert titles(first)[0] == "Beach Snack Mix"
        speaker = next(r for r in search(sort="price_asc", page=2)["results"]
                       if r["Product_title"] == "Bluetooth Speaker")
        assert speaker["is_sponsored"] is True


def test_rating_sort_is_bayesian_and_stable():
    with app_fakes():
        rows, _ = all_pages(sort="rating")
        shrunk = {p["Product_title"]: bayesian_rating(p["average_rating"], p["rating_number"])
                  for p in POOL}
        assert [shrunk[t] for t in rows] == sorted(shrunk.values(), reverse=True)
        # 4.5 from 200 reviews (4.48) beats 4.8 from 5 (4.27, shrunk toward 4.0); the
        # unrated earrings (prior 4.0) tie the snack mix (4.0 from 30) and keep the
        # relevance order; the sponsored speaker (3.2) is not forced to the front
        assert rows == ["Linen Midi Dress", "Portable Charger", "Straw Tote Bag",
                        "Leather Sandals", "Gold Hoop Earrings", "Beach Snack Mix",
                        "Model 7 Sneaker", "Cotton Sundress", "Sun Hat",
                        "Bluetooth Speaker"], rows


# ------------------------------------------------------------------ sorts keep relevance groups
# Oct 15 review: a price or rating sort over the WHOLE cached pool put never-reranked
# tail rows (and, under a type request, other types) on page 1 -- 12 cake toppers on
# "birthday gift for a 10 year old", no power bank at all on "power bank". The tail here
# is cheaper and better rated than every row the reranker judged, as it was there.
# "linen dress" asks for apparel; the reranker sees RERANK_INPUT_K = 4 rows.
GROUPED = [
    item("Linen Maxi Dress", "Fashion", "apparel", 80.0, 4.0, 40, 0.95),
    item("Cotton Shirt Dress", "Fashion", "apparel", 45.0, 4.1, 40, 0.94),
    item("Wool Ball Gown", "Fashion", "apparel", 30.0, 4.2, 40, 0.93),
    item("Linen Wrap Dress", "Fashion", "apparel", 60.0, 3.9, 40, 0.92),
    item("Gauze Beach Dress", "Fashion", "apparel", 3.0, 4.9, 900, 0.91),          # tail
    item("Gold Bridal Bracelet", "Fashion", "jewelry", 2.0, 5.0, 900, 0.90),       # tail
    item("Wedding Photo Backdrop", "Electronics", None, 1.0, 5.0, 900, 0.89),      # tail
    item("Wedding Cake Topper", "Grocery", None, 0.5, 5.0, 900, 0.88),             # tail
]
# 20 = a violator (<= SPONSORED_MIN_RERANK_SCORE, 40); the tail is never scored
GROUPED_SCORES = {"Linen Maxi Dress": 90, "Cotton Shirt Dress": 85, "Wool Ball Gown": 20,
                  "Linen Wrap Dress": 70}
OTHER_TYPES = ["Gold Bridal Bracelet", "Wedding Photo Backdrop", "Wedding Cake Topper"]


def grouped(knobs=None, **kwargs):
    return app_fakes(pool=GROUPED, scores=GROUPED_SCORES, knobs=knobs or {"RERANK_INPUT_K": 4},
                     **kwargs)


def test_sorts_reorder_within_relevance_groups_under_a_type_request():
    with grouped():
        rows, _ = all_pages(query="linen dress")
        # relevance: reranked > 40, the unscored dress, the rejected gown
        # (REJECTS_AFTER_TAIL, 2026-10-19: the reject used to precede the tail), other types
        assert rows == ["Linen Maxi Dress", "Cotton Shirt Dress", "Linen Wrap Dress",
                        "Gauze Beach Dress", "Wool Ball Gown"] + OTHER_TYPES, rows
        rows, _ = all_pages(query="linen dress", sort="price_asc")
        assert rows == ["Cotton Shirt Dress", "Linen Wrap Dress", "Linen Maxi Dress",
                        "Gauze Beach Dress", "Wool Ball Gown", "Wedding Cake Topper",
                        "Wedding Photo Backdrop", "Gold Bridal Bracelet"], rows
        rows, _ = all_pages(query="linen dress", sort="price_desc")
        assert rows[:3] == ["Linen Maxi Dress", "Linen Wrap Dress", "Cotton Shirt Dress"]
        assert rows[3:5] == ["Gauze Beach Dress", "Wool Ball Gown"], rows
        assert rows[5:] == OTHER_TYPES, rows
        rows, _ = all_pages(query="linen dress", sort="rating")
        shrunk = {p["Product_title"]: bayesian_rating(p["average_rating"], p["rating_number"])
                  for p in GROUPED}
        judged = ["Linen Maxi Dress", "Cotton Shirt Dress", "Linen Wrap Dress"]
        assert rows[:3] == sorted(judged, key=lambda t: -shrunk[t]), rows
        assert rows[3:5] == ["Gauze Beach Dress", "Wool Ball Gown"], rows
        assert rows[5:] == OTHER_TYPES, rows   # 5.0 from 900 reviews, still last
    # the 2026-10-14 band order, for contrast
    with grouped(knobs={"RERANK_INPUT_K": 4, "REJECTS_AFTER_TAIL": False}):
        rows, _ = all_pages(query="linen dress", sort="price_asc")
        assert rows[3:5] == ["Wool Ball Gown", "Gauze Beach Dress"], rows


def test_sorts_keep_the_judged_rows_first_without_a_type_request():
    """"power bank" / "gift for ..." are untyped: the unscored tail still follows the
    judged rows, and the rejected gown comes last (REJECTS_AFTER_TAIL)."""
    with grouped():
        rows, _ = all_pages(sort="price_asc")
        assert rows[:3] == ["Cotton Shirt Dress", "Linen Wrap Dress", "Linen Maxi Dress"], rows
        assert rows[3:] == ["Wedding Cake Topper", "Wedding Photo Backdrop",
                            "Gold Bridal Bracelet", "Gauze Beach Dress", "Wool Ball Gown"], rows


def test_sorts_after_a_rerank_fallback_keep_the_window_first():
    """Nothing reranked: the rows the reranker would have judged stand in for band 0."""
    with grouped(rerank_error="429"):
        out = search(query="linen dress", sort="price_asc")
        assert [(e["stage"], e["code"]) for e in out["errors"]] == [("rerank", "429")]
        rows, _ = all_pages(query="linen dress", sort="price_asc")
        assert rows == ["Wool Ball Gown", "Cotton Shirt Dress", "Linen Wrap Dress",
                        "Linen Maxi Dress", "Gauze Beach Dress", "Wedding Cake Topper",
                        "Wedding Photo Backdrop", "Gold Bridal Bracelet"], rows


def test_facets_offer_only_options_with_a_row_on_target():
    """The wedding-dress rail offered 'Electronics (7)' and 'Grocery (5)': backdrops and
    cake toppers the display had already put last as another type. Counts stay over the
    whole pool, so a count is what ticking it returns."""
    with grouped():
        out = search(query="linen dress")
        assert out["facets"]["departments"] == {"Fashion": 6}, out["facets"]
        assert out["facets"]["product_types"] == {"apparel": 5}, out["facets"]
        assert out["facets"]["price"]["n_priced"] == len(GROUPED)   # every row
        assert out["total_unfiltered"] == len(GROUPED)
        rows, filtered = all_pages(query="linen dress", department="Fashion")
        assert len(rows) == filtered["total_results"] == 6
    # untyped: only the reranker's rejection rules a row out
    scores = {t: 95 - 5 * i for i, t in enumerate(TITLES)}
    scores.update({"Bluetooth Speaker": 20, "Portable Charger": 35})
    with app_fakes(scores=scores):
        facets = search()["facets"]
        assert facets["departments"] == {"Fashion": 7, "Grocery": 1}, facets
        assert facets["product_types"] == {"apparel": 2, "footwear": 1, "jewelry": 1,
                                           "bag": 1, "headwear": 1}, facets


# ------------------------------------------------------------------ degraded pools
def test_a_degraded_pool_serves_follow_ups_and_a_new_search_retries():
    """Oct 15 review: a fallback pool is never cached, so every sort, filter or page
    click re-ran both LLM calls and served another pool (the facets moved). A follow-up
    now reuses it for _DEGRADED_TTL_S; a new search still retries the LLM."""
    with app_fakes(rerank_error="429") as fakes:
        base = search()
        assert (fakes.translate, fakes.rerank) == (1, 1)
        assert [(e["stage"], e["code"]) for e in base["errors"]] == [("rerank", "429")]
        for params in ({"sort": "price_asc"}, {"ptype_filter": "apparel"}, {"page": 2},
                       {"top_k": 8}, {}):
            out = search(follow_up=True, **params)
            assert (fakes.translate, fakes.rerank) == (1, 1), params
            # still labelled degraded, same pool
            assert out["errors"] == base["errors"] and out["rerank_succeeded"] is False
            assert out["total_unfiltered"] == base["total_unfiltered"], params
        assert out["facets"] == base["facets"]
        search()                                     # a new search: retried
        assert (fakes.translate, fakes.rerank) == (2, 2)
        fakes.rerank_error = None                    # the reranker answers again
        clean = search()
        assert (fakes.translate, fakes.rerank) == (3, 3) and clean["errors"] == []
        search(follow_up=True, sort="rating")
        search(page=2)
        assert (fakes.translate, fakes.rerank) == (3, 3)   # the clean pool is cached
        assert not main._degraded                          # and the degraded one gone

    with app_fakes(rerank_error="429") as fakes:
        search(follow_up=True)                      # nothing stored yet: a real run
        assert (fakes.translate, fakes.rerank) == (1, 1)
        with patched(main, {"_DEGRADED_TTL_S": -1.0}):
            search(follow_up=True, page=2)          # expired: retried
        assert (fakes.translate, fakes.rerank) == (2, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    client = TestClient(main.app)
    with app_fakes(rerank_error="429") as fakes:
        for follow_up in ("false", "true", "true"):
            r = client.get("/search", params={"query": "beach day", "page": 2,
                                              "follow_up": follow_up})
            assert r.status_code == 200 and r.json()["errors"], r.text
        assert (fakes.translate, fakes.rerank) == (1, 1)


def test_a_translator_failure_is_a_flat_page_not_a_one_card_bundle():
    """Oct 15 review: the fallback kept the regex's setup / recipe with one intent (the
    query), i.e. ONE card and no page 2 for "home office setup"."""
    from app import translator as tr

    def boom(*_a, **_k):
        raise RuntimeError("Error code: 429 (simulated)")

    with app_fakes() as fakes, patched(main, {"understand_query": tr.understand_query}), \
            patched(tr, {"generate_json": boom}):
        for query in ("home office setup", "gaming setup under $500",
                      "ingredients to make pancakes"):
            out = search(query=query)
            assert out["bundle_type"] is None and out["is_recipe"] is False, query
            assert out["interpreted_as"] == [query], out["interpreted_as"]
            assert [e["stage"] for e in out["errors"]] == ["translate"], out["errors"]
            assert (out["total_results"], len(out["results"]), out["has_next"]) == \
                (len(POOL), 4, True), query
        assert fakes.rerank == 3


# ------------------------------------------------------------------ filters
def test_price_and_rating_filters_exclude_rows_without_the_value():
    with app_fakes():
        rows, out = all_pages(min_price=10, max_price=40)
        assert rows == ["Linen Midi Dress", "Leather Sandals", "Gold Hoop Earrings",
                        "Cotton Sundress", "Portable Charger"], rows
        assert out["filters_applied"] == {"min_price": 10.0, "max_price": 40.0}
        assert (out["total_results"], out["total_unfiltered"]) == (5, len(POOL))
        assert out["total_pages"] == 2
        # either bound alone also drops unpriced rows
        rows, _ = all_pages(max_price=1000)
        assert "Straw Tote Bag" not in rows and "Sun Hat" not in rows and len(rows) == 8
        rows, _ = all_pages(min_rating=4)
        assert rows == ["Linen Midi Dress", "Leather Sandals", "Straw Tote Bag",
                        "Beach Snack Mix", "Portable Charger"], rows
        assert "Gold Hoop Earrings" not in rows               # unrated: excluded


def test_department_and_ptype_filters():
    with app_fakes():
        rows, out = all_pages(department="electronics, GROCERY")
        assert rows == ["Bluetooth Speaker", "Beach Snack Mix", "Portable Charger"], rows
        assert out["filters_applied"] == {"department": ["electronics", "GROCERY"]}
        rows, out = all_pages(ptype_filter="Bag, jewelry")
        assert rows == ["Straw Tote Bag", "Gold Hoop Earrings"], rows
        assert out["filters_applied"] == {"ptype": ["bag", "jewelry"]}
        # non-Fashion rows never match a ptype filter; 'unknown' is no type
        rows, _ = all_pages(ptype_filter="apparel,footwear,jewelry,bag,headwear,accessory")
        assert "Bluetooth Speaker" not in rows and "Model 7 Sneaker" not in rows
        # combined, with paging over the FILTERED list
        out = search(department="Fashion", min_price=0, top_k=2, page=2)
        # Fashion AND priced: linen, sandals, earrings, sundress, sneaker (the tote and
        # the hat have no price); page 2 at 2 per page is the earrings and the sundress
        assert titles(out) == ["Gold Hoop Earrings", "Cotton Sundress"], titles(out)
        assert (out["total_results"], out["total_pages"], out["has_prev"], out["has_next"]) \
            == (5, 3, True, True)
        # the facets never move with the filters
        assert out["facets"] == search()["facets"]


def test_empty_filtered_list_says_so():
    with app_fakes():
        out = search(min_price=1000)
        assert out["results"] == [] and out["no_match"] is False
        assert out["message"] == "No products match these filters."
        assert (out["total_results"], out["total_pages"], out["has_next"]) == (0, 1, False)
        assert out["total_unfiltered"] == len(POOL) and out["facets"] is not None


def test_unknown_sort_or_ptype_is_422():
    with app_fakes() as fakes:
        for params in ({"sort": "cheapest"}, {"ptype_filter": "shoes"},
                       {"ptype_filter": "bag,unknown"}):
            try:
                search(**params)
            except HTTPException as e:
                assert e.status_code == 422, params
            else:
                raise AssertionError(f"no 422 for {params}")
        assert (fakes.translate, fakes.rerank) == (0, 0)   # refused before any work


def test_http_params_alias_and_validation():
    """Over HTTP: ?ptype= reaches the filter, bad values are 422, the facets come back."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    client = TestClient(main.app)   # no `with`: the lifespan (catalog load) never runs
    with app_fakes() as fakes:
        r = client.get("/search", params={"query": "beach day", "top_k": 4, "ptype": "bag",
                                          "sort": "price_desc", "min_rating": 4,
                                          "recommend": "false"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["filters_applied"] == {"min_rating": 4.0, "ptype": ["bag"]}
        assert [x["Product_title"] for x in body["results"]] == ["Straw Tote Bag"]
        assert body["sort"] == "price_desc" and body["facets"]["departments"]
        for params in ({"sort": "cheap"}, {"ptype": "shoes"}, {"min_price": -1},
                       {"min_rating": 6}, {"max_price": "abc"}):
            bad = client.get("/search", params={"query": "beach day", **params})
            assert bad.status_code == 422, (params, bad.status_code)
        assert fakes.translate == 1


def test_bundles_ignore_filters_and_sort():
    with app_fakes(bundle_type="recipe"):
        out = search(min_price=1000, sort="price_asc", ptype_filter="bag")
        assert out["facets"] is None and out["filters_applied"] == {}
        assert out["sort"] == "relevance" and out["results"] and "message" not in out
        assert out["total_unfiltered"] == out["total_results"]


def test_plain_callers_get_the_defaults():
    """/chat and /unified_search call search() without the new params: the Query(...)
    markers must read as their defaults."""
    with app_fakes():
        out = search()
        assert out["sort"] == "relevance" and out["filters_applied"] == {}
        assert out["facets"] is not None


def test_no_match_payload():
    with app_fakes(min_relevance=0.99) as fakes:
        out = search(min_price=5)
        assert out["no_match"] is True and out["results"] == []
        assert out["message"].startswith("No matching products found")
        assert out["facets"] is None and out["total_unfiltered"] == 0
        # the clean no-match is cached: the second call makes no LLM call
        again = search()
        assert again["no_match"] is True and fakes.translate == 1


# ------------------------------------------------------------------ 2026-10-19 rail (C2)
def rail_item(title, ptype, gender, subtype, color_family, material, store, score,
              price=None, vertical="Fashion", color=None):
    out = item(title, vertical, ptype, price, 4.2, 40, score)
    out.update({"shopper_gender": gender, "fine_subtype": subtype, "color_family": color_family,
                "material": material, "store": store, "color": color,
                "img_url": f"https://img.example/{title.replace(' ', '_')}.jpg"})
    return out


# rerank scores R1 95 ... R6 70, R7 30 (rejected); stores spelled two ways
RAIL = [
    rail_item("Linen Shirt Dress", "apparel", "women", "dresses", "white", "linen",
              "Topsville, Inc.", 0.95, price=45.0),
    rail_item("Silk Slip Dress", "apparel", "women", "dresses", "black", "Silk",
              "topsville,  inc.", 0.94, price=120.0),
    rail_item("Graphic Tee", "apparel", "men", "tops_tees", "black", "cotton", "Acme",
              0.93, price=12.0),
    rail_item("Kids Rain Jacket", "apparel", "kids", "coats_jackets", "blue", "polyester",
              "Acme", 0.92),
    rail_item("Leather Sandals", "footwear", "unisex", "sandals", "brown", "leather",
              "Zattcas", 0.91, price=25.0),
    rail_item("Bluetooth Speaker", None, None, None, "black", "plastic", "SoundCo", 0.90,
              price=59.0, vertical="Electronics"),
    rail_item("Wool Coat", "apparel", "women", "coats_jackets", "multicolor", "wool",
              "Zattcas", 0.89),
]
RAIL_SCORES = {r["Product_title"]: s for r, s in zip(RAIL, (95, 90, 85, 80, 75, 70, 30))}


def rail(**kwargs):
    return app_fakes(pool=RAIL, scores=RAIL_SCORES, **kwargs)


def test_rail_facets_count_the_unfiltered_pool():
    with rail() as fakes:
        f = search(top_k=12)["facets"]
        # every gender with a row on target; the rejected Wool Coat still counts
        assert f["genders"] == [
            {"key": "women", "label": "Women", "n": 3}, {"key": "men", "label": "Men", "n": 1},
            {"key": "kids", "label": "Kids", "n": 1},
            {"key": "unisex", "label": "Unisex / unspecified", "n": 1}], f["genders"]
        # styles from product_type's labels; 'coats_jackets' is not one of them here
        assert f["subtypes"] == [{"key": "dresses", "label": "Dresses", "n": 2},
                                 {"key": "sandals", "label": "Sandals", "n": 1},
                                 {"key": "tops_tees", "label": "Tops & tees", "n": 1}]
        # colour on every vertical; 'multicolor' only on the rejected coat: not offered
        assert f["colors"] == [
            {"key": "black", "label": "Black", "hex": "#111111", "n": 3},
            {"key": "blue", "label": "Blue", "hex": "#1F5FBF", "n": 1},
            {"key": "white", "label": "White", "hex": "#FFFFFF", "n": 1}], f["colors"]
        # fabrics carried by an on-target garment (leather: only the sandals; wool: only
        # the rejected coat; plastic is no fabric)
        assert f["materials"] == [{"key": k, "label": k.title(), "n": 1}
                                  for k in ("cotton", "linen", "polyester", "silk")]
        # folded brands, the most frequent spelling (first seen on a tie) names one
        assert f["stores"] == [{"name": "Acme", "n": 2}, {"name": "Topsville, Inc.", "n": 2},
                               {"name": "Zattcas", "n": 2}, {"name": "SoundCo", "n": 1}]
        assert f["n_stores"] == 4
        # 12 | 25, 45 | 59 | 120: the $25 sandals are in the second band only
        assert [b["n"] for b in f["price"]["bands"]] == [1, 2, 1, 1, 0]
        assert fakes.rerank == 1
    # no garment on target: no Fabric section
    with app_fakes(pool=[RAIL[4], RAIL[5]], scores=RAIL_SCORES):
        assert search()["facets"]["materials"] == []


def test_rail_filters_match_the_derived_columns():
    with rail() as fakes:
        search(top_k=12)
        cases = [
            ({"gender": "women"}, ["Linen Shirt Dress", "Silk Slip Dress", "Wool Coat"]),
            ({"gender": "Kids, unisex"}, ["Kids Rain Jacket", "Leather Sandals"]),
            ({"subtype": "dresses"}, ["Linen Shirt Dress", "Silk Slip Dress"]),
            # the speaker is sponsored (SPONSORED): an ad keeps its place at the front
            ({"color": "black"}, ["Bluetooth Speaker", "Silk Slip Dress", "Graphic Tee"]),
            ({"material": "LINEN,silk"}, ["Linen Shirt Dress", "Silk Slip Dress"]),
            ({"store": ["TOPSVILLE,  Inc."]}, ["Linen Shirt Dress", "Silk Slip Dress"]),
            ({"store": ["acme", "zattcas"]},
             ["Graphic Tee", "Kids Rain Jacket", "Leather Sandals", "Wool Coat"]),
            ({"gender": "women", "color": "white"}, ["Linen Shirt Dress"]),
        ]
        for params, expected in cases:
            rows, out = all_pages(top_k=12, **params)
            assert rows == expected, (params, rows)
        _, out = all_pages(top_k=12, gender="Women", store=["Acme", " acme "], material="Silk")
        assert out["filters_applied"] == {"gender": ["women"], "material": ["silk"],
                                          "store": ["acme"]}, out["filters_applied"]
        assert (fakes.translate, fakes.rerank) == (1, 1)   # every filter: 0 LLM calls


def test_rail_params_are_validated():
    with rail() as fakes:
        for params in ({"gender": "female"}, {"subtype": "gowns"}, {"color": "chartreuse"},
                       {"store": [f"s{i}" for i in range(21)]}, {"store": ["x" * 121]},
                       {"material": "x" * 301}):
            try:
                search(**params)
            except HTTPException as e:
                assert e.status_code == 422, params
            else:
                raise AssertionError(f"no 422 for {params}")
        assert (fakes.translate, fakes.rerank) == (0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    client = TestClient(main.app)
    with rail():
        # a repeated param: names with commas stay whole
        r = client.get("/search", params=[("query", "beach day"), ("top_k", "12"),
                                           ("store", "Topsville, Inc."), ("store", "SoundCo")])
        assert r.status_code == 200, r.text
        assert [x["Product_title"] for x in r.json()["results"]] == [
            "Bluetooth Speaker", "Linen Shirt Dress", "Silk Slip Dress"]
        for params in ({"gender": "female"}, {"material": "x" * 301}, {"color": "teal"}):
            assert client.get("/search", params={"query": "beach day", **params}).status_code \
                == 422, params


def test_every_offered_brand_is_a_valid_store_filter():
    """2026-10-19 review: two catalog "stores" are VHS credit lines longer than the store
    param's 120 characters (the 130-character one below is real); the facet offered them and
    ticking one returned a 422 that blanked the results and the rail."""
    long_store = ("Steve Coogan (Actor, Writer), John Thomson (Actor, Writer), Dominic "
                  "Brigstocke (Director) & 0 more Rated: Unrated Format: VHS Tape")
    assert len(long_store) > main._MAX_STORE_CHARS
    edge = "E" * main._MAX_STORE_CHARS                     # exactly at the cap: offered
    pool = RAIL + [rail_item("Comedy VHS", None, None, None, None, None, long_store, 0.88,
                             vertical="Electronics"),
                   rail_item("Drama VHS", None, None, None, None, None, edge, 0.87,
                             vertical="Electronics")]
    with app_fakes(pool=pool, scores={**RAIL_SCORES, "Comedy VHS": 80, "Drama VHS": 80}):
        f = search(top_k=12)["facets"]
        names = [s["name"] for s in f["stores"]]
        assert long_store not in names and edge in names, names
        assert f["n_stores"] == len(names) == 5
        for name in names:                                  # ticking any of them works
            assert search(top_k=12, store=[name])["results"], name


# ------------------------------------------------------------------ variant collapse
HUHOT = [
    rail_item("HUHOT Floral Summer Dress Small", "apparel", "women", "dresses", "blue",
              "cotton", "HUHOT", 0.95, price=30.0, color="Blue"),
    rail_item("HUHOT Floral Summer Dress XX-Large", "apparel", "women", "dresses", "black",
              "cotton", "HUHOT", 0.94, price=20.0, color="Black"),
    rail_item("Huhot Linen Pants", "apparel", "women", None, "white", "linen", "HUHOT", 0.93),
    rail_item("HUHOT Floral Summer Dress Medium Red", "apparel", "women", "dresses", None,
              "cotton", "huhot", 0.92, color="Red"),
    rail_item("Anker Power Bank 10000mAh Black", None, None, None, "black", None, "Anker",
              0.91, vertical="Electronics"),
    rail_item("Anker Power Bank 10000mAh White", None, None, None, "white", None, "Anker",
              0.90, vertical="Electronics"),
]


HUHOT_SCORES = {r["Product_title"]: 95 - 5 * i for i, r in enumerate(HUHOT)}


def test_variant_collapse_one_card_per_style_after_filter_and_sort():
    with app_fakes(pool=HUHOT, scores=HUHOT_SCORES) as fakes:
        out = search(top_k=12)
        titles_ = titles(out)
        # the three dress rows become one card (sizes and colours); Electronics never merge
        assert titles_ == ["HUHOT Floral Summer Dress Small", "Huhot Linen Pants",
                           "Anker Power Bank 10000mAh Black",
                           "Anker Power Bank 10000mAh White"], titles_
        card = out["results"][0]
        assert card["n_variants"] == 3
        assert [v["title"] for v in card["variants"]] == [
            "HUHOT Floral Summer Dress Small", "HUHOT Floral Summer Dress XX-Large",
            "HUHOT Floral Summer Dress Medium Red"]
        assert card["variants"][0] == {
            "catalog_index": HUHOT[0]["catalog_index"], "title": HUHOT[0]["Product_title"],
            "color": "Blue", "color_family": "blue", "img_url": HUHOT[0]["img_url"]}
        assert card["variants"][2]["color_family"] is None
        assert "variants" not in out["results"][1]
        # cards, not rows: the header's totals agree, the facets still count rows
        assert (out["total_results"], out["total_unfiltered"], out["pool_rows"]) == (4, 4, 6)
        assert out["facets"]["genders"] == [{"key": "women", "label": "Women", "n": 4}]
        # paging is over cards
        rows, last = all_pages(top_k=2)
        assert len(rows) == 4 and last["total_pages"] == 2
        # sort, then collapse: the cheapest variant fronts its card
        first = search(top_k=12, sort="price_asc")["results"][0]
        assert first["Product_title"] == "HUHOT Floral Summer Dress XX-Large"
        assert first["variants"][0]["title"] == first["Product_title"]
        # filter, then collapse: a colour filter leaves one row, no variant list
        blue = search(top_k=12, color="blue")
        assert titles(blue) == ["HUHOT Floral Summer Dress Small"]
        assert "variants" not in blue["results"][0] and blue["total_results"] == 1
        # the cached pool is never written to
        entry = next(iter(page_cache._store.values()))
        assert not any("variants" in r for r in entry["full"])
        assert (fakes.translate, fakes.rerank) == (1, 1)
    with app_fakes(pool=HUHOT, scores=HUHOT_SCORES, knobs={"VARIANT_COLLAPSE": False}):
        out = search(top_k=12)
        assert len(out["results"]) == 6 and out["total_results"] == out["total_unfiltered"] == 6
        assert not any("variants" in r for r in out["results"])


def test_variant_collapse_never_merges_an_ad_with_organic_rows():
    """2026-10-19 review (safety.md, ads apart from organic): keyed on the style alone, the
    ad card absorbed its organic siblings as swatches, and under a price sort the cheaper
    organic sibling fronted the card while the ad became an unlabelled swatch."""
    ad = dict(HUHOT[0], parent_asin="ASPONSORED")        # $30, the organic XX-Large is $20
    pool = [ad] + HUHOT[1:]
    smap = {"ASPONSORED": {"sponsor": "HUHOT", "bid": 1.0}}
    with app_fakes(pool=pool, scores=HUHOT_SCORES), \
            patched(main, {"sponsored_map": lambda: dict(smap)}):
        for sort in ("relevance", "price_asc", "rating"):
            out = search(top_k=12, sort=sort)
            ads = [r for r in out["results"] if r.get("is_sponsored")]
            assert [r["Product_title"] for r in ads] == [ad["Product_title"]], (sort, ads)
            assert "variants" not in ads[0] and out["sponsored"] == ads, sort
            organic = [r for r in out["results"]
                       if r["Product_title"].startswith("HUHOT") and not r.get("is_sponsored")]
            # the two organic sizes are still one card, without the ad in its swatches
            assert len(organic) == 1 and organic[0]["n_variants"] == 2, (sort, organic)
            assert ad["catalog_index"] not in \
                [v["catalog_index"] for v in organic[0]["variants"]], sort
            assert out["total_results"] == 5, sort
        assert titles(search(top_k=12))[0] == ad["Product_title"]   # the ad still leads


# ------------------------------------------------------------------ match quality (P8)
def test_match_quality_flags_a_weak_page():
    with rail() as fakes:
        mq = search()["match_quality"]
        # 6 rows above the floor (the speaker included: no type request), 5 of them strong
        # (>= 71), best 95
        assert mq == {"band0": 6, "strong": 5, "best_rerank": 95.0, "weak": False,
                      "unjudged": False}, mq
    weak_scores = dict(RAIL_SCORES, **{"Linen Shirt Dress": 55, "Silk Slip Dress": 50,
                                       "Graphic Tee": 52, "Kids Rain Jacket": 45,
                                       "Leather Sandals": 30, "Bluetooth Speaker": 20})
    with app_fakes(pool=RAIL, scores=weak_scores):
        mq = search()["match_quality"]
        assert mq == {"band0": 4, "strong": 0, "best_rerank": 55.0, "weak": True,
                      "unjudged": False}, mq
    # enough accepts, none strong (the x02 shape: sandals accepted at 48-55 for running
    # shoes): NOT weak under contract C2 (band0 >= 4, best > 55), the default; weak only
    # with the stricter WEAK_MATCH_MIN_STRONG, an owner call (2026-10-19 review)
    lukewarm = dict(zip(RAIL_SCORES, (70, 69, 68, 67, 66, 65, 30)))
    for min_strong, weak in ((0, False), (4, True)):
        with app_fakes(pool=RAIL, scores=lukewarm, knobs={"WEAK_MATCH_MIN_STRONG": min_strong}):
            mq = search()["match_quality"]
            assert (mq["band0"], mq["strong"], mq["best_rerank"], mq["weak"]) == \
                (6, 0, 70.0, weak), (min_strong, mq)
    # six accepts, best 90, three strong: C2 says fine (the knob at 4 would not)
    six = dict(zip(RAIL_SCORES, (90, 85, 80, 70, 69, 68, 30)))
    with app_fakes(pool=RAIL, scores=six):
        mq = search()["match_quality"]
        assert (mq["band0"], mq["strong"], mq["weak"]) == (6, 3, False), mq
    few = dict(RAIL_SCORES, **{"Graphic Tee": 40, "Kids Rain Jacket": 35,
                               "Leather Sandals": 30, "Bluetooth Speaker": 20})
    with app_fakes(pool=RAIL, scores=few):
        mq = search()["match_quality"]
        assert mq["band0"] == 2 and mq["weak"] is True and mq["best_rerank"] == 95.0, mq
    # under a type request a mismatched type is not a match, whatever it scored
    with grouped():
        mq = search(query="linen dress")["match_quality"]
        assert mq["band0"] == 3 and mq["weak"] is True, mq
    # rerank fell back or was off: unjudged, never weak
    with rail(rerank_error="429"):
        assert search()["match_quality"] == {"band0": 0, "strong": 0, "best_rerank": None,
                                             "weak": False, "unjudged": True}
    with rail():
        assert search(rerank=False)["match_quality"]["unjudged"] is True
    # bundles and no-match pages carry none
    with app_fakes(bundle_type="recipe"):
        assert search()["match_quality"] is None
    with app_fakes(min_relevance=0.99):
        assert search()["match_quality"] is None
    assert fakes.translate == 1


def test_match_quality_counts_only_what_the_reranker_judged():
    """2026-10-19 review: a reply that lists fewer entries than asked is valid and no
    partial (only invalid entries are), so main.py keeps it -- "wedding dress" came back
    95 / 85 / 70, 3 of 15. Its 12 unreturned window rows counted as misses, so band0 3 < 4
    put "No exact matches" over gowns the model scored 95 and 85. The C2 thresholds now
    scale to the rows actually judged (4 of 15 -> 1 of 3); a full reply is unchanged."""
    gowns = [item(f"Lace Bridal Gown Style {i:02d}", "Fashion", "apparel", 99.0, 4.5, 100,
                  0.95 - 0.001 * i, "women") for i in range(60)]
    titles_ = [g["Product_title"] for g in gowns]

    def quality(scores, **knob_values):
        reply = dict(zip(titles_, scores))

        def short_rerank(query, candidates, top_k, errors=None, constraints=None):
            out = [dict(c, rerank_score=reply[c["Product_title"]], bayesian_rating=4.0,
                        final_score=float(reply[c["Product_title"]]), reason="ok")
                   for c in candidates[: cfg.RERANK_INPUT_K] if c["Product_title"] in reply]
            return sorted(out, key=lambda c: -c["final_score"])[:top_k]

        with app_fakes(pool=gowns, knobs=knob_values),                 patched(main, {"llm_rerank": short_rerank}):
            return search(query="wedding dress")["match_quality"]

    mq = quality((95, 85, 70))
    assert mq == {"band0": 3, "strong": 2, "best_rerank": 95.0, "weak": False,
                  "unjudged": False}, mq
    assert quality((95, 85, 70), WEAK_MATCH_MIN_STRONG=4)["weak"] is False   # 1 of 3 needed
    assert quality((95, 30))["weak"] is False             # 1 good of 2 judged: 1 needed
    assert quality((30, 20, 10))["weak"] is True          # every judged row rejected
    assert quality((55, 50, 45))["weak"] is True          # the best is still <= 55
    # a full reply (15 of 15) keeps C2 exactly: 3 above the floor of 15 is weak
    assert quality((95, 85, 70) + (30,) * 12)["weak"] is True
    assert quality((95, 85, 70, 60) + (30,) * 11)["weak"] is False
    assert main._judged_share(4, 15, 15) == 4 and main._judged_share(4, 8, 15) == 3
    assert main._judged_share(4, 3, 15) == 1 and main._judged_share(0, 3, 15) == 0
    assert main._judged_share(4, 3, 3) == 4             # a 3-row pool, fully judged


# ------------------------------------------------------------------ 2026-10-27 per-department rail
# facets.attributes + primary_department, the csv params category / item_type /
# works_with / condition / dietary (app/attributes.py's registry over the load-time
# columns of contract 3.1), and the Brands facet / `store` filter reading `brand`.
def attr_item(title, vertical, score, *, category=None, role=None, works=(), condition=None,
              diet=(), store="Volt", brand=None, price=None, rating=4.2):
    """A row carrying the load-time columns catalog_taxonomy.annotate_catalog adds, SET
    BY HAND: these tests are about main.py and app/attributes.py, so the readings are
    pinned stand-ins (the real readers: test_real_taxonomy_pools_* below and
    tests/test_catalog_taxonomy.py)."""
    out = item(title, vertical, None, price, rating, 40, score)
    out.update({"item_category": category, "item_role": role, "works_with": tuple(works),
                "item_condition": condition, "diet_tags": tuple(diet), "store": store,
                "brand": brand or store})
    return out


ATTR = [
    attr_item("Noise Cancelling Headphones", "Electronics", 0.95, category="headphones",
              role="device", store="Sony", price=199.0),
    attr_item("Wireless Earbuds", "Electronics", 0.94, category="headphones", role="device",
              store="JBL", price=49.0),
    attr_item("iPhone 14 Case", "Electronics", 0.93, category="cases", role="accessory",
              works=("apple",), store="Spigen", price=12.0),
    attr_item("Galaxy Watch Charger", "Electronics", 0.92, category="power", role="accessory",
              works=("samsung", "fitness")),
    # a chip vendor's store; the specs named the maker (catalog_taxonomy.brand_of)
    attr_item("IdeaPad Laptop (Renewed)", "Electronics", 0.91, category="computers",
              role="device", condition="renewed", store="intel", brand="Lenovo", price=450.0),
    # a protection plan: in no rail group and never an item type
    attr_item("Laptop Protection Plan", "Electronics", 0.90, category="service",
              role="service", store="Asurion"),
    attr_item("GF Crackers", "Grocery", 0.89, category="snacks", role="food",
              diet=("gluten free", "vegan"), store="Pantry Co", price=4.99),
    attr_item("GF Cookies", "Grocery", 0.88, category="bakery", role="food",
              diet=("gluten free",), store="Pantry Co"),
    attr_item("Vegan Jerky", "Grocery", 0.87, category="snacks", role="food",
              diet=("vegan", "high protein"), store="Trail", price=9.0),
    attr_item("Dog Treat Bowl", "Grocery", 0.86, category="non_food", role="non_food",
              store="Pawz"),
    attr_item("Mystery Snack Box", "Grocery", 0.85, category="unknown", store="Pantry Co"),
    # no taxonomy column at all (a Fashion row): never matches, never offers a value
    rail_item("Linen Midi Dress", "apparel", "women", "dresses", "white", "linen", "Zattcas",
              0.84, price=39.0),
    # rejected by the reranker (30): keto and the cheapest price are on no on-target row
    attr_item("Rejected Keto Chips", "Grocery", 0.83, category="snacks", role="food",
              diet=("keto",), store="Pantry Co", price=1.99),
]
ATTR_SCORES = {r["Product_title"]: s
               for r, s in zip(ATTR, (95, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 30))}
_OLD_FACET_KEYS = {"departments", "product_types", "price", "rating", "genders", "subtypes",
                   "colors", "materials", "stores", "n_stores"}


def attr_pool(**kwargs):
    return app_fakes(pool=ATTR, scores=ATTR_SCORES, **kwargs)


def _sections(facets, department):
    return {s["param"]: s for s in facets["attributes"] if s["department"] == department}


def _values(section):
    return [(v["key"], v["label"], v["n"]) for v in section["values"]]


def _refused(**params) -> str:
    try:
        search(**params)
    except HTTPException as e:
        assert e.status_code == 422, (params, e.status_code)
        return str(e.detail)
    raise AssertionError(f"no 422 for {params}")


def test_attribute_facets_follow_the_registry_count_rows_and_offer_on_target_only():
    with attr_pool() as fakes:
        f = search(top_k=12)["facets"]
        # on target: 6 Electronics, 5 Grocery, 1 Fashion rows -- exactly half is primary
        assert f["primary_department"] == "Electronics", f["primary_department"]
        # the registry's order, every department's (the UI shows the focus department's)
        assert [(s["param"], s["department"]) for s in f["attributes"]] == [
            ("category", "Electronics"), ("item_type", "Electronics"),
            ("works_with", "Electronics"), ("condition", "Electronics"),
            ("category", "Grocery"), ("dietary", "Grocery")], f["attributes"]
        elec, groc = _sections(f, "Electronics"), _sections(f, "Grocery")
        # shop groups, by count then the groups' order; "service" is in no group
        assert _values(elec["category"]) == [
            ("headphones", "Headphones & earbuds", 2),
            ("computers_monitors", "Computers & monitors", 1),
            ("cases_protection", "Cases & screen protection", 1),
            ("power_batteries", "Chargers, power & batteries", 1)], elec["category"]
        assert (elec["category"]["label"], elec["category"]["match"], elec["category"]["hint"],
                elec["category"]["n_values"]) == ("Category", "any", "", 4)
        assert _values(elec["item_type"]) == [("device", "Devices", 3),
                                              ("accessory", "Accessories", 2)]
        assert _values(elec["works_with"]) == [("apple", "Apple", 1),
                                               ("samsung", "Samsung", 1),
                                               ("fitness", "Fitbit & Garmin", 1)]
        assert (elec["condition"]["label"], elec["condition"]["hint"]) == \
            ("Condition", "As the listing words it")
        assert _values(elec["condition"]) == [("renewed", "Renewed / refurbished", 1)]
        # n counts every row (the rejected chips too); 'unknown' is no value
        assert _values(groc["category"]) == [
            ("snacks", "Chips, crackers & snacks", 3), ("bakery", "Cookies, cakes & bakery", 1),
            ("non_food", "Kitchen, pet & other non-food", 1)], groc["category"]
        diet = groc["dietary"]
        assert (diet["label"], diet["match"], diet["hint"]) == \
            ("Dietary", "all", "As the listing words it, not a certification")
        # keto sits only on the rejected row: not offered
        assert _values(diet) == [("gluten free", "Gluten free", 2), ("vegan", "Vegan", 2),
                                 ("high protein", "High protein", 1)], diet
        # Brands name the `brand` (Lenovo, never the chip vendor), else the store
        assert f["stores"] == [{"name": "Pantry Co", "n": 4}] + [
            {"name": n, "n": 1} for n in ("Asurion", "JBL", "Lenovo", "Pawz", "Sony", "Spigen",
                                          "Trail", "Volt", "Zattcas")], f["stores"]
        assert f["departments"] == {"Electronics": 6, "Grocery": 6, "Fashion": 1}
        assert (fakes.translate, fakes.rerank) == (1, 1)
    # values past MAX_VALUES_SHOWN are counted, not listed
    with attr_pool(), patched(attrs, {"MAX_VALUES_SHOWN": 2}):
        elec = _sections(search(top_k=12)["facets"], "Electronics")
        assert [v["key"] for v in elec["category"]["values"]] == \
            ["headphones", "computers_monitors"] and elec["category"]["n_values"] == 4
    # Fashion primary; no majority -> None
    dresses = [rail_item(f"Dress {i}", "apparel", "women", "dresses", "black", "cotton",
                         "Zattcas", 0.99 - 0.01 * i) for i in range(3)]
    with app_fakes(pool=dresses + ATTR[:2], scores={**ATTR_SCORES, **{
            d["Product_title"]: 90 for d in dresses}}):
        assert search()["facets"]["primary_department"] == "Fashion"
    with app_fakes(pool=dresses[:1] + ATTR[:1] + ATTR[6:7],
                   scores={**ATTR_SCORES, "Dress 0": 90}):
        f = search()["facets"]     # one row each: no department holds half
        assert f["primary_department"] is None and len(f["attributes"]) == 4, f


def test_attribute_sections_follow_the_type_request():
    """Under a type request ("linen dress": apparel) the Electronics backdrop and the
    Grocery cake topper are another type: no value of theirs is offered and they do not
    count toward primary_department -- the rule test_facets_offer_only_options_with_a_
    row_on_target pins for departments (review round 5: _facets' attribute sections
    reading _on_target(r, None) passed the suite)."""
    pool = [dict(r) for r in GROUPED]
    for r in pool:
        if r["Product_title"] == "Wedding Photo Backdrop":
            r.update(item_category="backdrops", item_role="accessory", works_with=("camera",),
                     item_condition=None, diet_tags=())
        elif r["Product_title"] == "Wedding Cake Topper":
            r.update(item_category="cake_decor", item_role="food", works_with=(),
                     item_condition=None, diet_tags=("vegan",))
    with app_fakes(pool=pool, scores=GROUPED_SCORES, knobs={"RERANK_INPUT_K": 4}) as fakes:
        f = search(query="linen dress")["facets"]
        assert f["attributes"] == [], f["attributes"]
        assert f["primary_department"] == "Fashion" and f["departments"] == {"Fashion": 6}
        # a key stays a valid filter when it is not offered (as a department does)
        for params, expected in (({"category": "camera_accessories"}, ["Wedding Photo Backdrop"]),
                                 ({"dietary": "vegan"}, ["Wedding Cake Topper"])):
            assert titles(search(query="linen dress", **params)) == expected, params
        assert (fakes.translate, fakes.rerank) == (1, 1)
    with app_fakes(pool=pool, scores=GROUPED_SCORES, knobs={"RERANK_INPUT_K": 4}):
        f = search()["facets"]          # untyped: both rows are on target again
        assert {(s["department"], s["param"]) for s in f["attributes"]} == {
            ("Electronics", "category"), ("Electronics", "item_type"),
            ("Electronics", "works_with"), ("Grocery", "category"),
            ("Grocery", "dietary")}, f["attributes"]
        assert f["primary_department"] == "Fashion"      # 5 of the 7 on-target rows


def test_primary_department_tie_and_rows_without_a_department():
    """Exactly half each: the department seen first in relevance order wins (the
    docstring's rule; the UI's focus department reads it). A row with no department is
    still an on-target row: the share is of every on-target row (contract 3.3), so one
    Electronics row beside two unlabelled ones is no majority (review round 5: a
    last-seen tie-break and a denominator without those rows both passed the suite)."""
    elec, groc = ATTR[0:2], ATTR[6:8]
    unlabelled = [item(f"Unlabelled Thing {i}", None, None, 5.0, 4.0, 10, 0.80 - 0.01 * i)
                  for i in range(2)]
    for pool, expected in ((elec + groc, "Electronics"), (groc + elec, "Grocery"),
                           (elec[:1] + unlabelled, None), (elec + unlabelled, "Electronics"),
                           (unlabelled + elec[:1], None)):
        scores = {r["Product_title"]: 90 - i for i, r in enumerate(pool)}
        with app_fakes(pool=pool, scores=scores):
            f = search(top_k=12)["facets"]
            assert f["primary_department"] == expected, ([r["Product_title"] for r in pool], f)
    # the pure function: an exact tie goes to the first seen, however the rest interleave
    for order, expected in (("GEEG", "Grocery"), ("EGGE", "Electronics"),
                            ("EEGG", "Electronics"), ("GGEE", "Grocery")):
        rows = [{"d": {"E": "Electronics", "G": "Grocery"}[c]} for c in order]
        assert attrs.facet_sections(rows, lambda r: True, lambda r: r["d"])[1] == expected, \
            order


def test_attribute_filters_any_all_and_zero_llm_calls():
    with attr_pool() as fakes:
        search(top_k=12)
        cases = [
            ({"category": "headphones"}, ["Noise Cancelling Headphones", "Wireless Earbuds"]),
            # ANY of a section's values; a group covers its fine keys
            ({"category": "computers_monitors,cases_protection"},
             ["iPhone 14 Case", "IdeaPad Laptop (Renewed)"]),
            ({"category": "snacks"}, ["GF Crackers", "Vegan Jerky", "Rejected Keto Chips"]),
            # one param, two departments' keys: each row reads its own department's
            ({"category": "SNACKS, headphones"},
             ["Noise Cancelling Headphones", "Wireless Earbuds", "GF Crackers", "Vegan Jerky",
              "Rejected Keto Chips"]),
            ({"item_type": "accessory"}, ["iPhone 14 Case", "Galaxy Watch Charger"]),
            # sections AND together
            ({"item_type": "device", "category": "headphones,computers_monitors"},
             ["Noise Cancelling Headphones", "Wireless Earbuds", "IdeaPad Laptop (Renewed)"]),
            ({"works_with": "fitness,apple"}, ["iPhone 14 Case", "Galaxy Watch Charger"]),
            ({"condition": "renewed"}, ["IdeaPad Laptop (Renewed)"]),
            ({"dietary": "gluten free"}, ["GF Crackers", "GF Cookies"]),
            ({"dietary": "Gluten Free,vegan"}, ["GF Crackers"]),           # ALL
            ({"dietary": "vegan", "min_price": 5}, ["Vegan Jerky"]),
            ({"dietary": "keto"}, ["Rejected Keto Chips"]),     # a valid key filters, offered or not
            ({"category": "headphones", "department": "Grocery"}, []),
            ({"item_type": "device", "dietary": "vegan"}, []),   # no row is in both
            # the Brands names: the brand matches, the chip vendor no longer does
            ({"store": ["lenovo"]}, ["IdeaPad Laptop (Renewed)"]),
            ({"store": ["intel"]}, []),
            ({"store": ["Zattcas"]}, ["Linen Midi Dress"]),     # no `brand`: the store
        ]
        for params, expected in cases:
            rows, out = all_pages(top_k=12, **params)
            assert rows == expected, (params, rows)
            if not expected:
                assert out["message"] == "No products match these filters."
        _, out = all_pages(top_k=12, dietary="vegan, Gluten free", category="snacks")
        assert out["filters_applied"] == {"category": ["snacks"],
                                          "dietary": ["vegan", "gluten free"]}, out
        assert out["total_unfiltered"] == len(ATTR)
        assert (fakes.translate, fakes.rerank) == (1, 1)     # every filter: 0 LLM calls


def test_attribute_params_are_validated_before_any_llm_call():
    with attr_pool() as fakes:
        for params in ({"dietary": "paleo"}, {"category": "computers"},   # a fine key, no group
                       {"item_type": "service"}, {"item_type": "food"}, {"condition": "new"},
                       {"works_with": "apple,teleport"},
                       {"category": "x" * (attrs.MAX_CHARS + 1)},
                       {"dietary": 3}, {"dietary": ["vegan", 3]}):        # not strings
            detail = _refused(**params)
            assert next(iter(params)) in detail, detail
        assert "expected one of" in _refused(dietary="paleo")
        assert (fakes.translate, fakes.rerank) == (0, 0)    # refused before any work
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    client = TestClient(main.app)
    with attr_pool() as fakes:
        r = client.get("/search", params={"query": "beach day", "top_k": 12,
                                          "dietary": "vegan"})
        assert r.status_code == 200 and [x["Product_title"] for x in r.json()["results"]] == \
            ["GF Crackers", "Vegan Jerky"], r.text
        assert r.json()["filters_applied"] == {"dietary": ["vegan"]}
        # every key a param accepts, at once, is never refused for its length
        everything = ",".join(attrs.vocabulary("category"))
        assert len(everything) > 300
        r = client.get("/search", params={"query": "beach day", "top_k": 12,
                                          "category": everything})
        assert r.status_code == 200, r.text
        assert [x["Product_title"] for x in r.json()["results"]] == [
            t["Product_title"] for t in ATTR
            if t.get("item_category") not in (None, "unknown", "service")], r.text
        for params in ({"dietary": "paleo"}, {"category": "x" * (attrs.MAX_CHARS + 1)},
                       {"condition": "used,mint"}):
            assert client.get("/search", params={"query": "beach day", **params}) \
                .status_code == 422, params
        assert fakes.translate == 1


def test_a_key_of_another_param_is_refused():
    """Contract 3.3: an unknown value is a 422 -- also a key only ANOTHER attribute param
    knows. Accepted, `dietary=headphones` would filter to an empty page with no error
    (review round 5: a parse_filters reading every param's vocabulary passed the suite)."""
    with attr_pool() as fakes:
        for params in ({"item_type": "snacks"}, {"dietary": "headphones"},
                       {"condition": "vegan"}, {"works_with": "renewed"},
                       {"category": "device"}, {"category": "gluten free"},
                       {"item_type": "apple"}):
            detail = _refused(**params)
            assert next(iter(params)) in detail and "expected one of" in detail, detail
        assert (fakes.translate, fakes.rerank) == (0, 0)
    # every key of every param, sent to each param that does not know it
    n_refused = 0
    for param in attrs.PARAMS:
        own = attrs.vocabulary(param)
        for other in attrs.PARAMS:
            for key in attrs.vocabulary(other):
                if key in own:
                    continue
                try:
                    attrs.parse_filters({param: key})
                except ValueError as e:
                    assert param in str(e), (param, key, e)
                    n_refused += 1
                    continue
                raise AssertionError((param, key))
    assert n_refused > 200, n_refused


def test_parse_filters_caps_a_long_but_valid_value():
    """/chat, /unified_search and the evals call search() plainly, and a list param gets
    no per-value Query max_length: parse_filters' cap on the JOINED csv is the only one.
    A value of valid keys over MAX_CHARS is refused for its length (the 501 x's above are
    an unknown key too, so they cannot tell the two reasons apart)."""
    long_valid = ",".join(["snacks"] * (attrs.MAX_CHARS // 7 + 2))   # 510 characters
    assert len(long_valid) > attrs.MAX_CHARS
    for value in (long_valid, long_valid.split(",")):     # csv, and repeated
        try:
            attrs.parse_filters({"category": value})
        except ValueError as e:
            assert "longer than" in str(e) and "category" in str(e), e
            continue
        raise AssertionError("no ValueError for an over-long value of valid keys")
    with attr_pool() as fakes:
        assert "longer than" in _refused(category=long_valid)
        assert "longer than" in _refused(dietary=["vegan"] * 84)   # 503 joined
        assert (fakes.translate, fakes.rerank) == (0, 0)
    # just under the cap, repeated: accepted (one key, listed once)
    assert attrs.parse_filters({"category": ["snacks"] * 71}) == {"category": ["snacks"]}


def test_a_repeated_attribute_param_means_its_csv():
    """Review round 5: declared as str, a REPEATED param kept only its LAST value with no
    422 -- dietary=vegan&dietary=keto served the keto chips (dietary is ALL: a dropped
    restriction widened the page past what the shopper asked). As lists, the repeated
    form (`store`'s pattern) and the csv the UI sends filter alike."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    client = TestClient(main.app)

    def get(qs):
        return client.get("/search?query=beach+day&top_k=12&" + qs)

    with attr_pool() as fakes:
        for repeated, csv, expected in (
                ("dietary=vegan&dietary=keto", "dietary=vegan,keto", []),
                ("dietary=keto&dietary=vegan", "dietary=keto,vegan", []),
                ("dietary=gluten+free&dietary=vegan", "dietary=gluten+free,vegan",
                 ["GF Crackers"]),
                ("category=snacks&category=headphones", "category=snacks,headphones",
                 ["Noise Cancelling Headphones", "Wireless Earbuds", "GF Crackers",
                  "Vegan Jerky", "Rejected Keto Chips"]),
                ("item_type=device&item_type=accessory", "item_type=device,accessory",
                 ["Noise Cancelling Headphones", "Wireless Earbuds", "iPhone 14 Case",
                  "Galaxy Watch Charger", "IdeaPad Laptop (Renewed)"]),
                ("works_with=apple&works_with=fitness", "works_with=apple,fitness",
                 ["iPhone 14 Case", "Galaxy Watch Charger"]),
                ("condition=renewed&condition=", "condition=renewed",
                 ["IdeaPad Laptop (Renewed)"])):
            a, b = get(repeated), get(csv)
            assert a.status_code == b.status_code == 200, (repeated, a.text, b.text)
            assert titles(a.json()) == titles(b.json()) == expected, (repeated, titles(a.json()))
            assert a.json()["filters_applied"] == b.json()["filters_applied"], repeated
        assert get("dietary=vegan&dietary=keto").json()["filters_applied"] == \
            {"dietary": ["vegan", "keto"]}
        # a bad key in any repetition, or a joined csv over the cap: a 422
        for qs in ("dietary=vegan&dietary=paleo", "&".join(["dietary=vegan"] * 84)):
            r = get(qs)
            assert r.status_code == 422, (qs[:40], r.text)
        assert "longer than" in get("&".join(["dietary=vegan"] * 84)).text
        assert (fakes.translate, fakes.rerank) == (1, 1)     # 0 LLM calls past the first
    # a plain caller's list is the repeated form too
    with attr_pool():
        assert titles(search(top_k=12, dietary=["gluten free", "vegan"])) == ["GF Crackers"]
    # OpenAPI says so: an array of strings for each param
    documented = {p["name"]: p for p in main.app.openapi()["paths"]["/search"]["get"]
                  ["parameters"]}
    for param in attrs.PARAMS:
        schema = documented[param]["schema"]
        assert {"type": "array", "items": {"type": "string"}} in schema["anyOf"], schema
        assert "repeated" in documented[param]["description"], param


def test_every_registry_param_is_a_declared_search_param():
    """FastAPI silently drops an undeclared query param: a section whose param /search
    does not declare would filter nothing, with no 422."""
    sig = inspect.signature(main.search)
    assert attrs.PARAMS == ("category", "item_type", "works_with", "condition", "dietary")
    assert set(attrs.PARAMS) <= set(sig.parameters), attrs.PARAMS
    for param in attrs.PARAMS:
        assert len(",".join(attrs.vocabulary(param))) <= attrs.MAX_CHARS, param
        for f in attrs.FACETS:
            if f.param == param:
                assert f.match in ("any", "all") and set(f.vocab) <= set(attrs.vocabulary(param))
    # one param, two departments: their keys are disjoint, so a key names its department
    elec, groc = (set(f.vocab) for f in attrs.FACETS if f.param == "category")
    assert not elec & groc and elec == set(ct.ELECTRONICS_RAIL_GROUPS) and \
        groc == set(ct.GROCERY_CATEGORIES)
    # every fine Electronics key but a protection plan belongs to a group
    assert set(ct.ELECTRONICS_CATEGORIES) - set(ct.ELECTRONICS_GROUP_OF) == {"service"}
    # the contract's labels (spec 3.3)
    assert attrs.vocabulary("item_type") == {
        "device": "Devices", "accessory": "Accessories", "part": "Parts & components",
        "consumable": "Batteries & supplies"}
    assert attrs.vocabulary("condition") == {"renewed": "Renewed / refurbished",
                                             "used": "Used"}
    assert attrs.vocabulary("works_with") == ct.WORKS_WITH_LABELS and \
        list(ct.WORKS_WITH_LABELS) == ["apple", "samsung", "google", "amazon", "microsoft",
                                       "playstation", "nintendo", "pc", "camera", "fitness"]
    diet = attrs.vocabulary("dietary")
    assert list(diet) == list(ct.DIETARY_VALUES) and len(diet) == 15
    assert diet["sugar free"] == "Sugar free / no added sugar"
    assert ct.GROCERY_CATEGORIES["non_food"] == "Kitchen, pet & other non-food"


def test_attribute_rows_read_tuples_lists_and_strings():
    """The columns are tuples in the frame, lists once a pool went through JSON, strings
    for one value; NaN / None / absent are no value; anything off the vocabulary is
    dropped (never seller text); the department is read case-insensitively."""
    elec_cat, item_type, works, cond, groc_cat, diet = attrs.FACETS
    row = {"item_category": "monitors", "works_with": ["apple", "Samsung", "bogus"],
           "diet_tags": "vegan,  gluten free"}
    assert attrs.row_keys(row, elec_cat, "Electronics") == {"computers_monitors"}
    assert attrs.row_keys(row, elec_cat, "electronics") == {"computers_monitors"}
    assert attrs.row_keys(row, works, "Electronics") == {"apple", "samsung"}
    assert attrs.row_keys(row, elec_cat, "Grocery") == frozenset()       # another department
    assert attrs.row_keys(row, groc_cat, "Grocery") == frozenset()       # not a Grocery key
    assert attrs.row_keys(row, diet, "Grocery") == {"vegan", "gluten free"}
    for empty in ({}, {"item_condition": None}, {"item_condition": float("nan")},
                  {"item_condition": ""}, {"item_condition": "mint"}):
        assert attrs.row_keys(empty, cond, "Electronics") == frozenset(), empty
    assert attrs.row_keys({"item_role": "service"}, item_type, "Electronics") == frozenset()
    assert attrs.passes(row, {"dietary": ["vegan"], "min_price": 3.0}, "Grocery")
    assert not attrs.passes(row, {"dietary": ["vegan", "keto"]}, "Grocery")   # ALL
    assert not attrs.passes(row, {"dietary": ["vegan"]}, None)                # no department
    assert attrs.passes({}, {"gender": ["women"]}, "Fashion")    # other filters: not ours
    assert attrs.parse_filters({"category": None, "dietary": " , "}) == {}
    assert attrs.parse_filters({"category": "Snacks,snacks, HEADPHONES"}) == \
        {"category": ["snacks", "headphones"]}
    for bad in ({"colour": "black"}, {"dietary": 3}):
        try:
            attrs.parse_filters(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    # brand, else the store; Fashion rows carry brand == store
    assert attrs.brand_name({"brand": "Lenovo", "store": "intel"}) == "Lenovo"
    for blank in (None, "", "  ", float("nan")):
        assert attrs.brand_name({"brand": blank, "store": "intel"}) == "intel", blank
    assert attrs.brand_name({"store": "Acme"}) == "Acme"
    assert attrs.brand_name({"store": float("nan")}) is None and attrs.brand_name({}) is None


def test_every_offered_attribute_value_is_a_valid_filter():
    """Like the Brands check: ticking any offered value alone returns its count (rows =
    cards off Fashion), never a 422 or an empty page."""
    with attr_pool() as fakes:
        f = search(top_k=12)["facets"]
        n_checked = 0
        for section in f["attributes"]:
            for value in section["values"]:
                out = search(top_k=30, **{section["param"]: value["key"]})
                assert out["total_results"] == value["n"] >= 1, (section["param"], value)
                assert all(r["bsns_vrtcl_name"] == section["department"]
                           for r in out["results"]), (section["param"], value)
                n_checked += 1
        assert n_checked == 16 and (fakes.translate, fakes.rerank) == (1, 1)


def test_a_sort_with_an_attribute_filter_stays_within_relevance_groups():
    """The rejected $1.99 chips are the cheapest snack: a whole-pool price sort would put
    them first; within relevance groups they stay after every accepted row."""
    with attr_pool() as fakes:
        for sort, expected in (
                ("price_asc", ["GF Crackers", "Vegan Jerky", "Rejected Keto Chips"]),
                ("price_desc", ["Vegan Jerky", "GF Crackers", "Rejected Keto Chips"]),
                ("relevance", ["GF Crackers", "Vegan Jerky", "Rejected Keto Chips"])):
            rows, out = all_pages(top_k=2, sort=sort, category="snacks")
            assert rows == expected, (sort, rows)
            assert out["sort"] == sort and out["total_results"] == 3
        rows, _ = all_pages(top_k=12, sort="price_asc", item_type="device")
        assert rows == ["Wireless Earbuds", "Noise Cancelling Headphones",
                        "IdeaPad Laptop (Renewed)"], rows
        assert (fakes.translate, fakes.rerank) == (1, 1)


def test_bundles_and_plain_callers_ignore_attribute_filters():
    with app_fakes(bundle_type="recipe", pool=ATTR, scores=ATTR_SCORES):
        out = search(dietary="vegan", category="snacks")
        assert out["facets"] is None and out["filters_applied"] == {}
        assert len(out["results"]) > 0
    with attr_pool():
        out = search()       # /chat, /unified_search and the evals: no attribute param
        assert out["filters_applied"] == {} and out["facets"]["attributes"]
        assert out["total_results"] == len(ATTR)


def test_an_older_client_sees_the_2026_10_19_payload_unchanged():
    """The new keys are additive. A client that never sends the new params and never
    reads `attributes` / `primary_department` (an older UI, a client integration) gets
    the same results and the same value under every key it knew: the taxonomy columns
    change none of them. (`brand` does change Brands, by design, where it is not the
    store: kept in both arms here, tested above.)"""
    stripped = [{k: v for k, v in r.items()
                 if k not in ct.ADDED_COLUMNS or k == ct.BRAND_COLUMN} for r in ATTR]
    assert not any("item_category" in r for r in stripped)
    with attr_pool():
        new = search(top_k=12)
    with app_fakes(pool=stripped, scores=ATTR_SCORES):
        old = search(top_k=12)
    assert set(new["facets"]) == _OLD_FACET_KEYS | {"attributes", "primary_department"}
    assert {k: new["facets"][k] for k in _OLD_FACET_KEYS} == \
        {k: old["facets"][k] for k in _OLD_FACET_KEYS}
    assert set(new) == set(old) and titles(new) == titles(old)
    assert (new["total_results"], new["total_unfiltered"], new["filters_applied"]) == \
        (old["total_results"], old["total_unfiltered"], old["filters_applied"])
    assert old["facets"]["attributes"] == [] and new["facets"]["attributes"]
    # a frame without the columns at all (the pre-2026-10-27 POOL) still answers
    with app_fakes():
        f = search()["facets"]
        assert f["attributes"] == [] and f["primary_department"] == "Fashion"
    # over HTTP: plain JSON, the shape of contract 3.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    client = TestClient(main.app)
    with attr_pool():
        body = client.get("/search", params={"query": "beach day", "top_k": 12}).json()
    assert body["facets"]["primary_department"] == "Electronics"
    assert {k: body["facets"][k] for k in ("departments", "stores", "n_stores")} == \
        {k: new["facets"][k] for k in ("departments", "stores", "n_stores")}
    for s in body["facets"]["attributes"]:
        assert set(s) == {"param", "label", "department", "match", "hint", "values",
                          "n_values"}, s
        assert all(set(v) == {"key", "label", "n"} and isinstance(v["n"], int)
                   for v in s["values"]), s
    assert body["facets"]["attributes"] == new["facets"]["attributes"]


def test_the_rail_draws_every_section_and_sends_back_what_the_api_accepts():
    """The contract both halves were built against (spec 3.3): ui_common draws every
    section the API sends for the focus department -- none dropped as malformed or as a
    param it owns -- and what a tick sends (sorted csv) is accepted, 0 LLM calls."""
    import ui_common as ui
    with attr_pool() as fakes:
        facets = search(top_k=12)["facets"]
        assert ui.focus_department(facets, {}) == facets["primary_department"]
        assert ui.focus_department(facets, {"department": ["Grocery"]}) == "Grocery"
        for department in ("Electronics", "Grocery"):
            sent = [s for s in facets["attributes"] if s["department"] == department]
            assert ui.visible_attributes(facets, department) == sent, department
            for s in sent:
                keys = [v["key"] for v in s["values"]]
                params = ui.filter_params({"attr": {s["param"]: keys}}, None)
                assert params == {s["param"]: ",".join(sorted(keys))}, params
                out = search(top_k=30, **params)
                assert out["filters_applied"][s["param"]] == sorted(keys)
                if s["match"] == "any":   # every value ticked: every row carrying one
                    assert out["total_results"] == len(
                        [r for r in ATTR if attrs.row_keys(
                            r, next(f for f in attrs.FACETS if f.param == s["param"]
                                    and f.department == department),
                            main._department(r))]), s["param"]
                else:                     # dietary: rows carrying them ALL (none here)
                    assert out["total_results"] == 0
        assert ui.visible_attributes(facets, "Fashion") == []
        assert (fakes.translate, fakes.rerank) == (1, 1)


# Real titles through the REAL readers (catalog_taxonomy.annotate_catalog on a frame, as
# search.load_index does on both backends): the registry, main.py and A1's columns
# together. Expectations come from the annotated frame's own columns; only a few
# unmistakable readings are pinned (the readers' own tables: test_catalog_taxonomy.py).
_REAL_ELECTRONICS = [
    ("Electronics", "Sony WH-1000XM5 Wireless Noise Canceling Headphones", "Sony", None, None),
    ("Electronics", "JBL Tune 230NC True Wireless Earbuds", "JBL", None, None),
    ("Electronics", "Silicone Case for iPhone 14 Pro Max", "Spigen", None, None),
    ("Electronics", "Fast Charger Cable for Samsung Galaxy S23", "Volt", None, None),
    ("Electronics", "Anker Portable Charger 10000mAh Power Bank", "Anker", None, None),
    ("Electronics", "Lenovo IdeaPad 3 15.6 inch Laptop, 8GB RAM, 256GB SSD (Renewed)",
     "intel", None, "Brand: Lenovo; Series: IdeaPad"),
    ("Electronics", "Watch Winder for Automatic Watches", "Versa", None, None),
    ("Electronics", "Replacement Band for Fitbit Charge 5", "Bandco", None, None),
    ("Electronics", "3-Year Protection Plan for Laptops", "Asurion", None, None),
    ("Electronics", "Bose SoundLink Flex Bluetooth Portable Speaker", "Bose", None, None),
    ("Electronics", "Wireless Mouse for Dell Laptop", "Logi", None, None),
]
_REAL_GROCERY = [
    ("Grocery", "KIND Bars, Dark Chocolate Nuts & Sea Salt, Gluten Free Snack Bars", "KIND",
     "gluten free, non-gmo", None),
    ("Grocery", "Lavazza Super Crema Whole Bean Coffee Blend, 2.2 lb", "Lavazza", "", None),
    ("Grocery", "Organic Green Tea Bags, 100 Count", "Bigelow", "organic, kosher", None),
    ("Grocery", "Pirate's Booty Aged White Cheddar Puffs, Gluten Free", "Pirate",
     "Gluten-Free, vegan", None),
    ("Grocery", "Ceramic Coffee Mug 12 oz", "MugCo", None, None),
    ("Grocery", "Ghirardelli Dark Chocolate Squares", "Ghirardelli", None, None),
    ("Grocery", "Starbucks Medium Roast Ground Coffee", "Starbucks", "kosher", None),
]


def taxonomy_rows(rows, fashion=()):
    """Pool rows carrying the REAL load-time columns, in the given (relevance) order."""
    frame = pd.DataFrame(rows, columns=["bsns_vrtcl_name", "Product_title", "store",
                                        "dietary", "specs"])
    ct.annotate_catalog(frame)
    out = []
    for i, rec in enumerate(frame.to_dict("records")):
        row = item(rec["Product_title"], rec["bsns_vrtcl_name"], None, None, 4.2, 40,
                   0.95 - 0.01 * i)
        row.update(rec)
        out.append(row)
    for j, title in enumerate(fashion):   # never annotated: no taxonomy column
        out.append(rail_item(title, "apparel", "women", "tops_tees", "black", "cotton",
                             "Zattcas", 0.50 - 0.01 * j))
    return out


def _expected_keys(row, param, department) -> set:
    """What a row offers for a section, read straight from its columns."""
    if main._department(row) != department:
        return set()
    if param == "category":
        key = row.get("item_category")
        if department == "Electronics":
            return {ct.ELECTRONICS_GROUP_OF[key]} if key in ct.ELECTRONICS_GROUP_OF else set()
        return {key} if key in ct.GROCERY_CATEGORIES else set()
    if param == "item_type":
        role = row.get("item_role")
        return {role} if role in ("device", "accessory", "part", "consumable") else set()
    if param == "condition":
        return {row["item_condition"]} if row.get("item_condition") else set()
    column = {"works_with": "works_with", "dietary": "diet_tags"}[param]
    return set(row.get(column) or ())


def _check_real_pool(pool, primary):
    scores = {r["Product_title"]: 90 - i for i, r in enumerate(pool)}
    with app_fakes(pool=pool, scores=scores) as fakes:
        out = search(top_k=30)
        f = out["facets"]
        assert f["primary_department"] == primary, f["primary_department"]
        order = titles(out)
        for s in f["attributes"]:
            facet = next(x for x in attrs.FACETS
                         if (x.param, x.department) == (s["param"], s["department"]))
            assert (s["label"], s["match"], s["hint"]) == (facet.label, facet.match, facet.hint)
            for v in s["values"]:
                assert v["label"] == facet.vocab[v["key"]], v
                carrying = [r["Product_title"] for r in pool
                            if v["key"] in _expected_keys(r, s["param"], s["department"])]
                assert v["n"] == len(carrying) >= 1, (s["param"], v, carrying)
                got = titles(search(top_k=30, **{s["param"]: v["key"]}))
                assert got == [t for t in order if t in carrying], (s["param"], v, got)
        assert (fakes.translate, fakes.rerank) == (1, 1)
        return f


def test_real_taxonomy_pools_electronics_grocery_and_mixed():
    f = _check_real_pool(taxonomy_rows(_REAL_ELECTRONICS), "Electronics")
    elec = _sections(f, "Electronics")
    assert set(elec) == {"category", "item_type", "works_with", "condition"}, set(elec)
    assert ("headphones", "Headphones & earbuds", 2) in _values(elec["category"])
    assert {"cases_protection", "watch_accessories", "keyboards_components"} <= \
        {v["key"] for v in elec["category"]["values"]}
    assert "service" not in {v["key"] for v in elec["item_type"]["values"]}
    assert {"apple", "samsung", "fitness", "pc"} <= {v["key"] for v in elec["works_with"]["values"]}
    assert _values(elec["condition"]) == [("renewed", "Renewed / refurbished", 1)]
    names = {s["name"] for s in f["stores"]}
    assert "Lenovo" in names and "intel" not in names        # the specs' maker
    assert not _sections(f, "Grocery")

    f = _check_real_pool(taxonomy_rows(_REAL_GROCERY), "Grocery")
    groc = _sections(f, "Grocery")
    assert set(groc) == {"category", "dietary"}, set(groc)
    assert ("coffee", "Coffee", 2) in _values(groc["category"])
    assert ("gluten free", "Gluten free", 2) in _values(groc["dietary"])  # "Gluten-Free" folds
    assert {"non_food", "tea", "candy"} <= {v["key"] for v in groc["category"]["values"]}

    # "gift for a coffee lover": mostly Grocery, an Electronics and a Fashion minority
    mixed = taxonomy_rows(_REAL_GROCERY + _REAL_ELECTRONICS[9:10] + _REAL_ELECTRONICS[0:1],
                          fashion=("Coffee Lover Graphic T-Shirt",))
    f = _check_real_pool(mixed, "Grocery")
    assert set(_sections(f, "Grocery")) == {"category", "dietary"}
    assert set(_sections(f, "Electronics")) == {"category", "item_type"}
    assert f["product_types"] == {"apparel": 1}       # Fashion's own facets, as before


# ------------------------------------------------------------------ helpers
def test_number_and_department_helpers():
    assert main._number("$1,299.50") == 1299.5 and main._number(" 12 ") == 12.0
    for bad in (None, float("nan"), float("inf"), "", "n/a", True):
        assert main._number(bad) is None, bad
    assert main._price({"price": -3}) is None and main._price({"price": 0}) == 0.0
    assert main._price({"price": None, "price_estimate": 20.0}) is None
    assert main._department({"bsns_vrtcl_name": "fashion"}) == "Fashion"
    assert main._department({"bsns_vrtcl_name": "AMAZON FASHION"}) == "AMAZON FASHION"
    assert main._department({"bsns_vrtcl_name": float("nan")}) is None
    assert main._csv_values(" a, B ,,b, c") == ["a", "B", "c"]
    assert main._csv_values(None) == []
    assert not math.isnan(main._average_rating({"average_rating": "4.5"}))


# ------------------------------------------------------------------ /keyword_search
def _engine():
    rows = []
    for i in range(7):
        rows.append({"Product_title": f"Red Running Shoe Model {i}", "categ_lvl2_name": "Shoes",
                     "average_rating": 4.0, "rating_number": 10 * (7 - i),
                     "parent_asin": f"S{i}", "bsns_vrtcl_name": "Fashion"})
    rows.append({"Product_title": "Blue Kettle", "categ_lvl2_name": "Kitchen",
                 "average_rating": 4.0, "rating_number": 5, "parent_asin": "K1",
                 "bsns_vrtcl_name": "Electronics"})
    engine = KeywordSearchEngine()
    engine.index(pd.DataFrame(rows))
    return engine


def test_keyword_search_pages():
    engine = _engine()
    full, n, *_ = engine.search("red running shoe", k=50)
    order = [r["Product_title"] for r in full]
    assert n == 7 and len(order) == 7
    # the sponsored row sits on page 2 of the BM25 order: labelled there? no -- page 1 only
    smap = {"S5": {"sponsor": "ShoeCo", "bid": 1.0}, "S0": {"sponsor": "Acme", "bid": 0.5}}
    with patched(main, {"_keyword_engine": engine, "sponsored_map": lambda: dict(smap)}):
        p1 = main.keyword_search(query="red running shoe", top_k=3, sponsored=True, page=1)
        p2 = main.keyword_search(query="red running shoe", top_k=3, sponsored=True, page=2)
        p3 = main.keyword_search(query="red running shoe", top_k=3, sponsored=True, page=3)
        p4 = main.keyword_search(query="red running shoe", top_k=3, sponsored=True, page=4)
        # a plain call without `page` (as before the param existed) is page 1
        p1_plain = main.keyword_search(query="red running shoe", top_k=3, sponsored=True)
        none = main.keyword_search(query="zzzz", top_k=3, sponsored=True, page=1)
    got = [[r["Product_title"] for r in p["results"]] for p in (p1, p2, p3, p4)]
    # page 1: its own top 3, with the sponsored S0 labelled (bid-front within the page)
    assert set(got[0]) == set(order[:3]) and p1["results"][0]["is_sponsored"]
    assert got[1] == order[3:6] and got[2] == order[6:] and got[3] == []
    assert not any(r.get("is_sponsored") for r in p2["results"])   # S5 is not boosted
    for p, page in ((p1, 1), (p2, 2), (p3, 3)):
        assert (p["page"], p["page_size"], p["total_results"], p["total_pages"]) == \
            (page, 3, 7, 3)
        assert p["has_prev"] == (page > 1) and p["has_next"] == (page < 3)
        assert p["num_matched"] == 7 and p["no_match"] is False
    assert p4["has_next"] is False and p4["results"] == []
    assert [r["Product_title"] for r in p1_plain["results"]] == got[0]
    assert none["no_match"] and none["total_results"] == 0 and none["total_pages"] == 1
    # offsets build only the rows returned, with the same catalog_index
    rows, *_ = engine.search("red running shoe", k=2, offset=5)
    assert [r["Product_title"] for r in rows] == order[5:7]
    assert all(isinstance(r["catalog_index"], int) for r in rows)


# ------------------------------------------------------------------ plain runner
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 -- report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:500]}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
