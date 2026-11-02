"""Offline tests for the type-aware ordering in app/main.py (TYPE_AWARE_RANKING) and the
same-type upsell in app/recommendations.py.

No LLM, no catalog load, no embedding model. main.search() runs with the translator and
the reranker faked, and retrieval runs the REAL app.search.search_products over a small
annotated frame (only _raw_hits is faked), so keep=, the per-intent quota and the
NaN -> None row shape are the real ones. Every fixture is built in BOTH row shapes: the
Databricks one (gender None off Fashion) and the local one, pushed through
app.search._build_search_text (gender '' off Fashion; until 2026-10-15 it also lowercased
the vertical, title and category IN PLACE, so the code under test must read either case).

Runs under pytest (`python -m pytest tests/test_ranking.py`) or plainly
(`python tests/test_ranking.py`). CAS_TRACE_JSON=<trace_out.json> adds the replay of the
recorded live traces (before/after page-1 type sequences of the stakeholder queries).
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import app.config as cfg  # noqa: E402
import app.main as main  # noqa: E402
from app import catalog_taxonomy as ct  # noqa: E402
from app import device_role as dr  # noqa: E402
from app import page_cache  # noqa: E402
from app import product_type as pt  # noqa: E402
from app import recommendations as recs  # noqa: E402
from app import search as S  # noqa: E402

SHAPES = ("databricks", "local")



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


# ------------------------------------------------------------------ frozen pre-change code
# Pasted from app/main.py at HEAD (before TYPE_AWARE_RANKING). The knob-false path must
# order exactly like these wherever the old and new gender classifiers agree.
def _old_product_gender(product: dict) -> str:
    """Infer gender from category, then title: 'women', 'men' or 'neutral'."""
    cat = (product.get("categ_lvl2_name") or "").lower()
    title = (product.get("Product_title") or "").lower()
    for text in (cat, title):
        if "women" in text or "ladies" in text or "girl" in text:
            return "women"
        if "men" in text or "boy" in text:   # 'women' already returned above
            return "men"
    return "neutral"


def _old_balance_by_gender(full):
    """Round-robin women/men/neutral so one gender can't crowd out the page."""
    women, men, neutral = [], [], []
    for item in full:
        g = _old_product_gender(item)
        (women if g == "women" else men if g == "men" else neutral).append(item)

    if not (women and men):
        return full   # nothing to balance (single gender or non-apparel)

    streams = [women, men, neutral]
    idx = [0, 0, 0]
    out = []
    while any(idx[i] < len(streams[i]) for i in range(3)):
        for i in range(3):
            if idx[i] < len(streams[i]):
                out.append(streams[i][idx[i]])
                idx[i] += 1
    return out


def _old_balance_candidate_pool(candidates):
    """Gender-interleave the retrieval candidates before the reranker."""
    women, men, neutral = [], [], []
    for item in candidates:
        g = _old_product_gender(item)
        (women if g == "women" else men if g == "men" else neutral).append(item)
    if not (women and men):
        return candidates
    streams = [women, men, neutral]
    idx = [0, 0, 0]
    out = []
    while any(idx[i] < len(streams[i]) for i in range(3)):
        for i in range(3):
            if idx[i] < len(streams[i]):
                out.append(streams[i][idx[i]])
                idx[i] += 1
    return out


# ------------------------------------------------------------------ helpers
@contextlib.contextmanager
def patched(obj, attrs: dict):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


@contextlib.contextmanager
def knobs(**values):
    """Pin the knobs these tests read; .env must not decide what a test checks.
    VARIANT_COLLAPSE is off here: it is a post-cache display step with its own tests
    (tests/test_api_filters.py), and these fixtures' numbered titles ("... Model 3") are
    exactly what it merges."""
    pinned = dict(TYPE_AWARE_RANKING=True, SPONSORED_MIN_RERANK_SCORE=40.0,
                  RERANK_INPUT_K=15, QUERY_GUARD_ENABLED=True, OUTFIT_BUNDLES_ENABLED=False,
                  REJECTS_AFTER_TAIL=True, VARIANT_COLLAPSE=False, DEVICE_ROLE_RANKING=True)
    pinned.update(values)
    with patched(cfg, pinned):
        yield


@contextlib.contextmanager
def c1_fakes(**fakes):
    """Install product_type readers (contract C1) for one test, whatever the module
    defines, and restore them after."""
    with patched(pt, fakes):
        yield


@contextlib.contextmanager
def captured_logs(name="app"):
    records, logger = [], logging.getLogger(name)

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler, level = _H(level=logging.DEBUG), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def row(title, vertical="Fashion", cat=None, gender=None, rating=4.2, n=120):
    """One catalog row with Title-case values, as databricks/01 stores it."""
    if cat is None:
        cat = {"Fashion": "Women's Clothing"}.get(vertical, vertical)
    return {
        "bsns_vrtcl_name": vertical, "categ_lvl2_name": cat, "Product_title": title,
        "prod_description": "", "color": "", "material": "", "occasion": "",
        "gender": gender,
        "parent_asin": "B" + hashlib.md5(title.encode()).hexdigest()[:9].upper(),
        "average_rating": rating, "rating_number": n,
    }


def frame(rows, shape):
    """The served frame: annotated first (load_index's order: product_type, then the
    2026-10-27 catalog taxonomy -- watch_kind, item_category, item_role ...), then for the
    local shape pushed through the real _build_search_text (whatever it does to the
    display columns)."""
    df = pd.DataFrame(rows)
    pt.annotate(df)
    ct.annotate_catalog(df)
    if shape == "local":
        S._build_search_text(df, S.effective_fields(df.columns))
    return df


def as_rows(rows, shape):
    """Row dicts in a backend's shape, NaN -> None like search_products returns them."""
    df = frame(rows, shape)
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in df.to_dict(orient="records")]


@contextlib.contextmanager
def served(rows, hits, shape):
    """The real search_products over `rows`; hits = {term: [title | (title, score)]}, best
    first. Default scores interleave intents rank by rank, so the raw order is fixed."""
    df = frame(rows, shape)
    position = {r["Product_title"].casefold(): i for i, r in enumerate(rows)}

    def raw_hits(terms, top_k):
        out = []
        for ti, term in enumerate(terms):
            pairs = []
            for rank, hit in enumerate(hits.get(term, [])):
                title, score = hit if isinstance(hit, tuple) else (
                    hit, 0.95 - 0.002 * rank - 0.0001 * ti)
                pairs.append((position[title.casefold()], score))
            out.append(pairs[:top_k])
        return out

    with patched(S, {"_df": df, "_ready": lambda: True, "_raw_hits": raw_hits}):
        yield df


class FakeRerank:
    """Stands in for llm_rerank: records the window and the constraints, returns the
    candidates it has a score for (by title), best first; the others become the tail."""

    def __init__(self, scores=None):
        self.scores = {k.casefold(): v for k, v in (scores or {}).items()}
        self.calls = []

    def __call__(self, query, candidates, top_k, errors=None, constraints=None):
        window = candidates[: cfg.RERANK_INPUT_K]
        self.calls.append({
            "query": query,
            "window": [c["Product_title"].casefold() for c in window],
            "window_rows": [dict(c) for c in window],
            "constraints": [dict(c) for c in (constraints or [])],
        })
        out = []
        for c in window:
            score = self.scores.get(c["Product_title"].casefold())
            if score is None:
                continue
            out.append(dict(c, rerank_score=score, bayesian_rating=4.0,
                            final_score=float(score), reason="ok"))
        out.sort(key=lambda p: p["final_score"], reverse=True)
        return out[:top_k]


class Run:
    def __init__(self, out, rerank, searches, translated=(), recs=()):
        self.out, self.rerank, self.searches = out, rerank, searches
        self.translated, self.recs = list(translated), list(recs)

    @property
    def results(self):
        return self.out["results"]

    @property
    def titles(self):
        return [r["Product_title"].casefold() for r in self.results]

    @property
    def types(self):
        return [main._row_type(r) for r in self.results]


_UNSET = object()


def run(query, rows, hits, *, shape, intents, bundle_type=None, constraints=(), scores=None,
        typed=True, rerank=True, smap=None, top_k=12, requested=_UNSET, clear=True,
        recommend=False, knob_values=None, sort="relevance"):
    """One main.search() call over the fakes; every endpoint parameter explicit."""
    fake_rerank = FakeRerank(scores)
    searches, translated, recs = [], [], []
    real_search = S.search_products

    def search_products(terms, top_k=30, per_intent_quota=None, **kwargs):
        searches.append({"top_k": top_k, "quota": per_intent_quota, "keep": "keep" in kwargs})
        return real_search(terms, top_k=top_k, per_intent_quota=per_intent_quota, **kwargs)

    def understand_query(q, mode=None, errors=None):
        translated.append(q)
        return {"intents": list(intents), "bundle_type": bundle_type,
                "constraints": [dict(c) for c in constraints],
                "is_recipe": bundle_type == "recipe", "query": q}

    def build_recommendations(q, results, page=1):
        recs.append([r["Product_title"].casefold() for r in results])
        return {"cross_sell": [], "upsell": [], "cross_sell_source": None, "errors": []}

    fakes = {
        "understand_query": understand_query,
        "search_products": search_products,
        "llm_rerank": fake_rerank,
        "MIN_RESULT_RELEVANCE": 0.0,
        "build_recommendations": build_recommendations,
        "sponsored_map": lambda: dict(smap or {}),
    }
    type_patch = {} if requested is _UNSET else {"requested_types": lambda q: requested}
    if clear:
        page_cache.clear()
    with knobs(TYPE_AWARE_RANKING=typed, **(knob_values or {})), served(rows, hits, shape), \
            patched(main, fakes), patched(pt, type_patch):
        out = main.search(query=query, top_k=top_k, page=1, rerank=rerank, recommend=recommend,
                          sponsored=bool(smap), translator_mode=None, sort=sort)
        if clear:
            # the cached entry, read before it is dropped (device / requested / full)
            run.last_entries = [dict(e) for e in page_cache._store.values()]
    if clear:
        page_cache.clear()
    return Run(out, fake_rerank, searches, translated, recs)


def _no_mismatch_before_match(types, requested):
    """No row of another type precedes a requested-type or untyped one."""
    ranks = [pt.type_rank(t, requested) for t in types]
    first_mismatch = next((i for i, r in enumerate(ranks) if r == 2), len(ranks))
    return all(r == 2 for r in ranks[first_mismatch:])


# ------------------------------------------------------------------ fixtures
# "cloths for a beach wedding in summer" in miniature: the sandals score highest in raw
# embedding order (as recorded: the live window was 15 of 15 wedge sandals), an untyped
# model-named shoe, a pair of earrings and a grocery row ride along.
BEACH_QUERY = "cloths for a beach wedding in summer"
BEACH_INTENTS = ["lightweight summer dress", "linen blazer", "wedge sandals"]
BEACH_ROWS = [
    row("Women's Floral Summer Maxi Dress"),
    row("Women's Linen Midi Sundress"),
    row("Women's Cotton Wrap Dress"),
    row("Women's Off Shoulder Beach Dress"),
    row("Pearl Drop Earrings"),
    row("Summer Party Snack Mix", vertical="Grocery"),
    row("Men's Linen Blazer Lightweight", cat="Men's Clothing"),
    row("Women's Linen Blazer Casual"),
    row("Men's Linen Sport Coat", cat="Men's Clothing"),
    row("Men's Short Sleeve Linen Shirt", cat="Men's Clothing"),
    row("Women's Wedge Sandals Espadrille"),
    row("Women's Platform Wedge Sandals"),
    row("Rhinestone Wedge Sandal for Women"),
    row("Cork Wedge Sandals", cat="Unisex Clothing"),
    row("Strappy Gladiator Sandals for Women"),
    row("Clarks Pitney Limit", cat="Men's Clothing"),
]
BEACH_HITS = {
    "lightweight summer dress": [
        ("Women's Floral Summer Maxi Dress", 0.889), ("Women's Linen Midi Sundress", 0.888),
        ("Women's Cotton Wrap Dress", 0.886), ("Women's Off Shoulder Beach Dress", 0.884),
        ("Pearl Drop Earrings", 0.870), ("Summer Party Snack Mix", 0.860)],
    "linen blazer": [
        ("Men's Linen Blazer Lightweight", 0.887), ("Women's Linen Blazer Casual", 0.885),
        ("Men's Linen Sport Coat", 0.883), ("Men's Short Sleeve Linen Shirt", 0.881)],
    "wedge sandals": [
        ("Women's Wedge Sandals Espadrille", 0.919), ("Women's Platform Wedge Sandals", 0.905),
        ("Rhinestone Wedge Sandal for Women", 0.903), ("Cork Wedge Sandals", 0.902),
        ("Strappy Gladiator Sandals for Women", 0.900), ("Clarks Pitney Limit", 0.899)],
}
# What a reranker does without the type constraint: sandals win (recorded: 92, 88, 84)
BEACH_SCORES = {
    "Rhinestone Wedge Sandal for Women": 92, "Women's Wedge Sandals Espadrille": 88,
    "Cork Wedge Sandals": 84, "Women's Platform Wedge Sandals": 80,
    "Strappy Gladiator Sandals for Women": 78, "Clarks Pitney Limit": 75,
    "Women's Floral Summer Maxi Dress": 86, "Women's Off Shoulder Beach Dress": 82,
    "Women's Linen Midi Sundress": 79, "Women's Cotton Wrap Dress": 70,
    "Men's Linen Blazer Lightweight": 72, "Men's Linen Sport Coat": 66,
    "Women's Linen Blazer Casual": 64, "Men's Short Sleeve Linen Shirt": 60,
    "Pearl Drop Earrings": 30, "Summer Party Snack Mix": 5,
}


def _collision_pool(query, own, fashion, vertical):
    """A pool where `own` (the query's real vertical) dominates the raw top 15 and a few
    Fashion rows the query's noun collides with sit in the middle of it."""
    rows = [row(t, vertical=vertical) for t in own] + [row(t) for t in fashion]
    order = list(own[:4]) + list(fashion) + list(own[4:])
    hits = {query: [(t, 0.95 - 0.003 * i) for i, t in enumerate(order)]}
    return rows, hits


COLLISIONS = [
    ("organic green tea bags", "Grocery",
     [f"Organic Green Tea Bags {i} Count" for i in range(12)],
     ["Canvas Tote Bag", "Straw Clutch Handbag", "Leather Chest Sling Bag"]),
    ("ring light", "Electronics",
     [f"LED Ring Light {i} inch with Tripod Stand" for i in range(12)],
     ["Silicone Wedding Ring for Women", "Stackable Promise Ring", "Gold Hoop Earrings"]),
    ("apple watch", "Electronics",
     [f"Apple Watch Series {i} GPS 41mm" for i in range(12)],
     ["Trail Loop Band for Apple Watch", "Men's Watch", "Canvas Tote Bag"]),
    ("laptop bag", "Electronics",
     [f"Laptop Bag {i} inch Water Resistant" for i in range(11)],
     ["Leather Briefcase Messenger Bag", "Canvas Tote Bag", "Straw Clutch Handbag",
      "Leather Chest Sling Bag"]),
    ("camera bag", "Electronics",
     [f"DSLR Camera Bag {i} with Rain Cover" for i in range(12)],
     ["Leather Chest Sling Bag", "Canvas Tote Bag", "Straw Clutch Handbag"]),
    ("smart ring", "Electronics",
     [f"Smart Ring Sleep Tracker Size {i}" for i in range(12)],
     ["Silicone Wedding Ring for Women", "Stackable Promise Ring", "Gold Hoop Earrings"]),
    ("watch movies on my bedroom wall", "Electronics",
     [f"Mini Projector {i} 1080P Portable" for i in range(13)],
     ["Men's Watch", "Trail Loop Band for Apple Watch"]),
]


# ------------------------------------------------------------------ C1: gender
_GENDER_CASES = [
    # (row, expected new, expected old) -- old = the frozen substring classifier
    (row("Floral Midi Dress"), "women", "women"),
    (row("Linen Shirt", cat="Men's Clothing"), "men", "men"),
    (row("Tote Bag", cat="Unisex Clothing", gender="women"), "women", "neutral"),
    (row("Wide Brim Sun Hat", cat="Unisex Clothing"), "neutral", "neutral"),
    # the intended C1 differences: trap words, and the enriched tag beating the title
    (row("Women's Wide Brim Sun Hat", cat="Unisex Clothing", gender="unisex"), "neutral", "women"),
    (row("Girls Floral Sundress", cat="Kids", gender="kids"), "neutral", "women"),
    (row("Cowgirl Boots Western", cat="Unisex Clothing"), "neutral", "women"),
    (row("Diamond Engagement Ring Solitaire", cat="Unisex Clothing"), "neutral", "men"),
    (row("Replacement Remote for Samsung TV", vertical="Electronics"), "neutral", "men"),
    (row("Instant Ramen Noodles Spicy", vertical="Grocery"), "neutral", "men"),
    (row("Yemen Mocha Coffee Beans", vertical="Grocery"), "neutral", "men"),
    (row("DecalGirl Skin for iPad", vertical="Electronics"), "neutral", "women"),
    (row("Men's Electric Shaver", vertical="Electronics"), "neutral", "men"),
    (row("Women's Running Shoes Lightweight", cat="Unisex Clothing"), "women", "women"),
]


def test_product_gender_both_shapes():
    rows = [r for r, _, _ in _GENDER_CASES]
    for shape in SHAPES:
        served_rows = as_rows(rows, shape)
        got = [main._product_gender(r) for r in served_rows]
        assert got == [new for _, new, _ in _GENDER_CASES], (shape, got)
    # the local shape really has '' gender off Fashion (and, before 2026-10-15, a
    # lowercased vertical; either case must classify the same)
    local = as_rows(rows, "local")
    grocery = local[rows.index(_GENDER_CASES[9][0])]
    assert grocery["bsns_vrtcl_name"].casefold() == "grocery" and grocery["gender"] == ""
    assert main._product_gender(grocery) == "neutral"
    lowered = {k: (v.lower() if isinstance(v, str) else v) for k, v in grocery.items()}
    assert main._product_gender(lowered) == "neutral"
    # absent gender values: None, NaN, '', whitespace, non-str
    for value in (None, float("nan"), "", "  ", 3):
        r = dict(rows[0], gender=value)
        assert main._product_gender(r) == "women", value
    assert main._product_gender({"Product_title": "Women's Dress"}) == "neutral"  # no vertical


def test_intended_gender_differences_are_listed():
    """C1 is NOT behind the kill switch: these rows order differently with the knob off
    than before the change, on purpose. Every other row agrees."""
    for shape in SHAPES:
        served_rows = as_rows([r for r, _, _ in _GENDER_CASES], shape)
        for r, (_, new, old) in zip(served_rows, _GENDER_CASES):
            assert _old_product_gender(r) == old, (shape, r["Product_title"])
            assert main._product_gender(r) == new, (shape, r["Product_title"])
    differ = [r["Product_title"] for r, new, old in _GENDER_CASES if new != old]
    assert differ == [
        "Tote Bag", "Women's Wide Brim Sun Hat", "Girls Floral Sundress",
        "Cowgirl Boots Western", "Diamond Engagement Ring Solitaire",
        "Replacement Remote for Samsung TV", "Instant Ramen Noodles Spicy",
        "Yemen Mocha Coffee Beans", "DecalGirl Skin for iPad", "Men's Electric Shaver",
    ], differ


# ------------------------------------------------------------------ C2: pronouns
def test_pronouns_state_gender():
    stated = ["something for her to wear to a black-tie gala", "gift for my wife",
              "suit for the groom", "his and hers towels", "outfit for my daughter"]
    unstated = ["the best headphones", "hershey chocolate", "beach wedding outfit",
                "cloths for a beach wedding in summer", "shoes"]
    for q in stated:
        assert main._gender_stated(q, [], typed=True), q
        assert not main._gender_stated(q, [], typed=False), q  # gated
    for q in unstated:
        assert not main._gender_stated(q, [], typed=True), q
    assert main._gender_stated("shoes", [{"type": "gender", "value": "men"}], typed=False)
    assert main._gender_stated("men's shoes", [], typed=False)

    rows = [row("Women's Silk Evening Gown"), row("Women's Satin Maxi Dress"),
            row("Men's Slim Fit Tuxedo", cat="Men's Clothing"),
            row("Men's Velvet Dinner Jacket", cat="Men's Clothing")]
    hits = {"formal evening gown": [r["Product_title"] for r in rows]}
    raw = [r["Product_title"].casefold() for r in rows]
    q = "something for her to wear to a black-tie gala"
    for shape in SHAPES:
        on = run(q, rows, hits, shape=shape, intents=["formal evening gown"])
        off = run(q, rows, hits, shape=shape, intents=["formal evening gown"], typed=False)
        # stated: one retrieval depth and no interleave (raw order kept); off: 2x + balance
        assert on.searches[0]["top_k"] == main.RETRIEVAL_TOP_K, on.searches
        assert off.searches[0]["top_k"] == main.RETRIEVAL_TOP_K * 2, off.searches
        assert on.rerank.calls[0]["window"] == raw
        assert off.rerank.calls[0]["window"] == [raw[0], raw[2], raw[1], raw[3]]
        assert pt.requested_types(q) == frozenset({"apparel"})
        # "gift for my wife" names no type: no type constraint reaches the reranker
        gift = run("gift for my wife", rows, hits, shape=shape, intents=["formal evening gown"])
        assert pt.requested_types("gift for my wife") is None
        assert all(c["type"] != "product_type" for c in gift.rerank.calls[0]["constraints"])


# ------------------------------------------------------------------ C3: interleave
def _g(title, gender):
    return {"Product_title": title, "bsns_vrtcl_name": "Fashion", "gender": gender,
            "categ_lvl2_name": "", pt.TYPE_COLUMN: "apparel"}


def test_interleave_gender():
    W1, W2, W3 = _g("w1", "women"), _g("w2", "women"), _g("w3", "women")
    M1, M2 = _g("m1", "men"), _g("m2", "men")
    N1, N2 = _g("n1", "unisex"), {"Product_title": "n2 lamp", "bsns_vrtcl_name": "electronics",
                                    "gender": ""}
    seq = [W1, N1, W2, W3, M1, N2, M2]
    out = main._interleave_gender(seq)
    assert [x["Product_title"] for x in out] == ["w1", "n1", "m1", "w2", "m2", "n2 lamp", "w3"]
    # neutral items keep their positions; men first when a man's item comes first
    out = main._interleave_gender([N1, M1, W1, W2, W3])
    assert [x["Product_title"] for x in out] == ["n1", "m1", "w1", "w2", "w3"]
    out = main._interleave_gender([M1, M2, W1, N1, W2])
    assert [x["Product_title"] for x in out] == ["m1", "w1", "m2", "n1", "w2"]
    # single gender (or none): unchanged
    assert main._interleave_gender([W1, N1, W2]) == [W1, N1, W2]
    assert main._interleave_gender([]) == []


def test_round_robin_by_intent():
    def it(t, intent):
        return {"Product_title": t, "source_intent": intent}

    items = [it("c1", "c"), it("a1", "a"), it("a2", "a"), it("b1", "b"), it("c2", "c"),
             it("a3", "a")]
    out = main._round_robin_by_intent(items, ["b", "a"])  # c is not a listed intent: last
    assert [x["Product_title"] for x in out] == ["b1", "a1", "c1", "a2", "c2", "a3"]
    assert main._round_robin_by_intent([], ["a"]) == []


# ------------------------------------------------------------------ C4: pool gate
def test_pool_gate_neutralises_collisions():
    for query, vertical, own, fashion in COLLISIONS:
        requested = pt.requested_types(query)
        assert requested, (query, "documented collision: the noun names a Fashion type")
        rows, hits = _collision_pool(query, own, fashion, vertical)
        scores = {t: 90 - i for i, t in enumerate(own)}
        scores.update({t: 50 for t in fashion})
        smap = {r["parent_asin"]: {"sponsor": "Acme", "bid": 1.0} for r in rows[:2]}
        for shape in SHAPES:
            with captured_logs("app.main") as logs:
                gated = run(query, rows, hits, shape=shape, intents=[query], scores=scores,
                            smap=smap)
            plain = run(query, rows, hits, shape=shape, intents=[query], scores=scores,
                        smap=smap, requested=None)
            assert gated.rerank.calls[0]["window"] == plain.rerank.calls[0]["window"], query
            assert gated.rerank.calls[0]["constraints"] == plain.rerank.calls[0]["constraints"]
            assert gated.titles == plain.titles, (query, shape)
            assert [r.get("is_sponsored") for r in gated.results] == \
                [r.get("is_sponsored") for r in plain.results]
            # No TYPE constraint reaches the reranker. "apple watch" is also a DEVICE ask
            # (2026-10-27, decided: app/device_role.py reads it, the pool is Electronics
            # watches), so it gets the code-written device text instead -- the same with
            # or without the type reading, which the pool gate drops either way
            device = dr.requested_device(query)
            expected = [dr.constraint(device)] if query == "apple watch" else []
            assert device is None or device.accessory_asked or query == "apple watch", query
            assert gated.rerank.calls[0]["constraints"] == expected, query
            # the gate logs a flag, never the query
            assert any("non_fashion_pool" in m for m in logs), (query, logs)
            assert not any(query in m for m in logs), logs
            # ...and it mattered wherever the own rows do not READ as the requested type:
            # without it the Fashion rows would lead the window. Since 2026-10-15 an own
            # row whose title classifies to the type ranks with them ("Organic Green Tea
            # Bags" -> bag, "LED Ring Light" -> jewelry), so there it is moot
            window = gated.rerank.calls[0]["window_rows"]
            if any(main._type_rank(c, requested) == 2 for c in window):
                assert main._flat_window(window, requested, [query], True) != window, query
            else:
                assert main._flat_window(window, requested, [query], True) == window, query


def test_pool_gate_keeps_a_fashion_majority_request():
    for shape in SHAPES:
        r = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                scores=BEACH_SCORES)
        type_constraints = [c for c in r.rerank.calls[0]["constraints"]
                            if c["type"] == "product_type"]
        assert type_constraints == [{"type": "product_type", "value": (
            "clothing only (garments: dresses, tops, bottoms, suits, swimwear, outerwear); "
            "shoes, jewellery, bags, hats and other accessories violate it")}], type_constraints
        assert BEACH_QUERY not in json.dumps(r.rerank.calls[0]["constraints"])
        # the response keeps the translator's constraints (none here)
        assert r.out["constraints"] == []


def test_type_constraint_text():
    both = main._type_constraint(frozenset({"apparel", "footwear"}))["value"]
    assert both.startswith("clothing or footwear only (garments: ") and \
        "; jewellery, bags, hats and other accessories violate it" in both, both
    every = main._type_constraint(frozenset(pt.FASHION_TYPES))["value"]
    assert "violate" not in every and "{" not in every


# ------------------------------------------------------------------ C5: window
def _typed(title, ptype_, intent, gender=None, score=None):
    return {"Product_title": title, pt.TYPE_COLUMN: ptype_, "source_intent": intent,
            "bsns_vrtcl_name": "Fashion" if ptype_ else "Grocery", "gender": gender,
            "categ_lvl2_name": "", "rerank_score": score}


def test_window_tiers_round_robin_and_interleave():
    cands = [
        _typed("sandal1", "footwear", "sandals", "women"),
        _typed("sandal2", "footwear", "sandals", "women"),
        _typed("clarks", "unknown", "sandals", "men"),
        _typed("dress1", "apparel", "dress", "women"),
        _typed("blazer1", "apparel", "blazer", "women"),
        _typed("dress2", "apparel", "dress", "women"),
        _typed("blazer2", "apparel", "blazer", "men"),
        _typed("snack", None, "dress"),
        _typed("earring", "jewelry", "dress", "women"),
        _typed("blazer3", "apparel", "blazer", "men"),
    ]
    intents = ["dress", "blazer", "sandals"]
    with knobs():
        out = main._flat_window(cands, frozenset({"apparel"}), intents, interleave=False)
    assert [c["Product_title"] for c in out] == [
        "dress1", "blazer1", "dress2", "blazer2", "blazer3",   # tier 0, intents round-robin
        # untyped Fashion, but within the raw top RERANK_INPUT_K: competes in tier 0
        # (2026-10-19), after the rows of intents that do not name another type ('sandals'
        # names footwear: its rows keep no seat ahead of the other intents')
        "clarks",
        "snack", "sandal1", "earring", "sandal2",              # tier 2, round-robin
    ]
    # outside the raw head the untyped row is tier 1 again, still after every typed row
    with knobs(RERANK_INPUT_K=2):
        out = main._flat_window(cands, frozenset({"apparel"}), intents, interleave=False)
    assert [c["Product_title"] for c in out][4:7] == ["blazer3", "clarks", "snack"], out
    with knobs():
        out = main._flat_window(cands, frozenset({"apparel"}), intents, interleave=True)
    assert [c["Product_title"] for c in out][:6] == [
        "dress1", "blazer2", "blazer1", "blazer3", "dress2",   # women/men alternate
        "clarks"], out   # the off-type intent's men's row is not pulled into a men's slot
    # no request: one tier, intents round-robin only
    out = main._flat_window(cands, None, intents, interleave=False)
    assert [c["Product_title"] for c in out][:3] == ["dress1", "blazer1", "sandal1"]

    for shape in SHAPES:
        r = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                scores=BEACH_SCORES)
        assert r.searches[0]["top_k"] == main.RETRIEVAL_TOP_K * 2   # 2x deep kept
        assert not r.searches[0]["keep"]
        types = [main._row_type(c) for c in r.rerank.calls[0]["window_rows"]]
        assert types[:8] == ["apparel"] * 8 and types[8] == "unknown", types
        old = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                  scores=BEACH_SCORES, typed=False)
        old_types = [main._row_type(c) for c in old.rerank.calls[0]["window_rows"]]
        assert old_types[:5].count("footwear") >= 3, old_types   # the recorded failure


# ------------------------------------------------------------------ C6: display
def test_display_puts_no_mismatch_first_and_unknown_competes_in_band():
    full = [
        _typed("sandal95", "footwear", "s", score=95),
        _typed("dress60", "apparel", "d", score=60),
        _typed("oxford90", "unknown", "s", score=90),
        _typed("dress30", "apparel", "d", score=30),
        _typed("dress_tail", "apparel", "d"),
        _typed("unknown_tail", "unknown", "s"),
        _typed("sandal_tail", "footwear", "s"),
        _typed("dress40", "apparel", "d", score=40),
    ]
    with knobs():
        out = main._flat_display(full, frozenset({"apparel"}), interleave=False)
        assert [x["Product_title"] for x in out] == [
            "dress60", "oxford90",              # band 0 (reranked > 40)
            "dress_tail", "unknown_tail",       # band 2 (not reranked): REJECTS_AFTER_TAIL
            "dress30", "dress40",               # band 1 (<= the ceiling), after the tail
            "sandal95", "sandal_tail",          # mismatches last, whatever they scored
        ]
    # the 2026-10-14 order: the rejects before the tail
    with knobs(REJECTS_AFTER_TAIL=False):
        out = main._flat_display(full, frozenset({"apparel"}), interleave=False)
        assert [x["Product_title"] for x in out] == [
            "dress60", "oxford90", "dress30", "dress40", "dress_tail", "unknown_tail",
            "sandal95", "sandal_tail"]
    with knobs():
        # without a request: band order only, stable
        out = main._flat_display(full, None, interleave=False)
        assert [x["Product_title"] for x in out][:3] == ["sandal95", "dress60", "oxford90"]
        # the ceiling is read at call time
        with patched(cfg, {"SPONSORED_MIN_RERANK_SCORE": 25.0}):
            out = main._flat_display(full, frozenset({"apparel"}), interleave=False)
            assert [x["Product_title"] for x in out][:4] == [
                "dress60", "dress30", "dress40", "oxford90"]

    for shape in SHAPES:
        r = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                scores=BEACH_SCORES)
        requested = frozenset({"apparel"})
        assert _no_mismatch_before_match(r.types, requested), r.types
        assert r.types[:4] == ["apparel"] * 4, r.types
        # rerank off: the window order, still type-tiered
        off = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                  rerank=False)
        assert _no_mismatch_before_match(off.types, requested) and off.types[0] == "apparel"


# ------------------------------------------------------------------ C8: setting backstop
def test_setting_backstop():
    setting = [{"type": "other", "value": v} for v in (
        "wedding", "summer", "beach", "beach wedding", "Beach-Wedding", "a humid day",
        "warm", "formal", "cozy")]
    hard = [{"type": "color", "value": "warm"}, {"type": "other", "value": "machine washable"},
            {"type": "other", "value": "beach towel"}, {"type": "other", "value": "the"},
            {"type": "material", "value": "linen"}]
    assert main._without_setting_constraints(setting + hard) == hard

    constraints = setting + hard
    for shape in SHAPES:
        for bundle in (None, "outfit"):
            r = run("summer beach wedding outfit dress", BEACH_ROWS, BEACH_HITS, shape=shape,
                    intents=BEACH_INTENTS, bundle_type=bundle, constraints=constraints,
                    scores=BEACH_SCORES)
            # the code-written ones ([product_type], the 2026-10-19 guest [occasion]) aside
            sent = [c for c in r.rerank.calls[0]["constraints"]
                    if c["type"] not in ("product_type", "occasion")]
            assert sent == hard, (bundle, sent)
            assert r.out["constraints"] == constraints   # the response is the translator's
        off = run("summer beach wedding outfit dress", BEACH_ROWS, BEACH_HITS, shape=shape,
                  intents=BEACH_INTENTS, constraints=constraints, scores=BEACH_SCORES,
                  typed=False)
        assert off.rerank.calls[0]["constraints"] == constraints


# ------------------------------------------------------------------ C9: sponsored
def test_sponsored_type_gate():
    asin = {r["Product_title"]: r["parent_asin"] for r in BEACH_ROWS}
    smap = {asin["Rhinestone Wedge Sandal for Women"]: {"sponsor": "ShoeCo", "bid": 9.0},
            asin["Women's Linen Midi Sundress"]: {"sponsor": "DressCo", "bid": 1.0}}
    for shape in SHAPES:
        on = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                 scores=BEACH_SCORES, smap=smap)
        boosted = [r["Product_title"].casefold() for r in on.results if r.get("is_sponsored")]
        assert boosted == ["women's linen midi sundress"], boosted
        assert on.titles[0] == "women's linen midi sundress"
        assert on.types[0] == "apparel"
        # no request -> the old gate: the higher bid leads
        plain = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                    scores=BEACH_SCORES, smap=smap, requested=None)
        assert plain.titles[:2] == ["rhinestone wedge sandal for women",
                                    "women's linen midi sundress"], plain.titles
        off = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                  scores=BEACH_SCORES, smap=smap, typed=False)
        assert off.titles[0] == "rhinestone wedge sandal for women"
        for r in (on, plain, off):
            assert all("_bid" not in x for x in r.results)


# ------------------------------------------------------------------ C10: bundles
OUTFIT_ROWS = [
    row("Starfish Barefoot Sandals Beach Wedding Foot Jewelry", cat="Unisex Clothing"),
    row("Beaded Barefoot Sandal Anklet Toe Ring", cat="Unisex Clothing"),
    row("Shell Barefoot Sandals Bridal Anklet", cat="Unisex Clothing"),
    row("Pearl Barefoot Sandal Foot Chain"),
    row("Women's Platform Wedge Sandals"),
    row("Women's Flat Leather Sandals"),
    row("Strappy Gladiator Sandals for Women"),
    row("Cork Footbed Sandals Women"),
    row("Women's Off Shoulder Beach Dress"),
    row("Silk Wrap Midi Dress"),
    row("Straw Clutch Handbag"),
    row("Gold Pearl Necklace"),
    row("Women's Wide Brim Sun Hat"),
]
OUTFIT_INTENTS = ["pearl necklace", "beach wedding sandals", "summer beach wedding dress",
                  "woven straw clutch"]
OUTFIT_HITS = {
    "beach wedding sandals": [r["Product_title"] for r in OUTFIT_ROWS[:8]],
    "summer beach wedding dress": ["Women's Wide Brim Sun Hat", "Women's Off Shoulder Beach Dress",
                                   "Silk Wrap Midi Dress"],
    "woven straw clutch": ["Straw Clutch Handbag"],
    "pearl necklace": ["Gold Pearl Necklace"],
}


def _cards(out):
    """[(slot type, primary type, [option types])] of a bundle page."""
    return [(pt.slot_type(c["source_intent"]), main._row_type(c),
             [main._row_type(a) for a in c.get("alternatives") or []]) for c in out["results"]]


def test_outfit_keep_fills_a_slot_with_its_own_type():
    """The 'sandals' slot's ranks 1-4 are barefoot-sandal jewellery, 5-8 real sandals."""
    for shape in SHAPES:
        r = run("summer beach wedding outfit dress", OUTFIT_ROWS, OUTFIT_HITS, shape=shape,
                intents=OUTFIT_INTENTS, bundle_type="outfit")
        assert r.searches[0]["keep"] and r.searches[0]["quota"] == main.GROCERY_PER_INTENT_K
        cards = {c["source_intent"]: c for c in r.results}
        sandals = cards["beach wedding sandals"]
        assert main._row_type(sandals) == "footwear"
        assert [main._row_type(a) for a in sandals["alternatives"]] == ["footwear", "footwear"]
        # the sun hat never reaches the dress slot
        dress = cards["summer beach wedding dress"]
        assert {main._row_type(x) for x in [dress] + dress["alternatives"]} == {"apparel"}
        # off: no keep, the quota fills with the anklets
        off = run("summer beach wedding outfit dress", OUTFIT_ROWS, OUTFIT_HITS, shape=shape,
                  intents=OUTFIT_INTENTS, bundle_type="outfit", typed=False)
        assert not off.searches[0]["keep"]
        off_cards = {c["source_intent"]: c for c in off.results}
        assert main._row_type(off_cards["beach wedding sandals"]) == "jewelry"


def test_outfit_slot_order_garment_first_and_no_type_request():
    constraints = [{"type": "other", "value": "wedding"}, {"type": "color", "value": "white"}]
    for shape in SHAPES:
        r = run("summer beach wedding outfit dress", OUTFIT_ROWS, OUTFIT_HITS, shape=shape,
                intents=OUTFIT_INTENTS, bundle_type="outfit", constraints=constraints,
                scores={"Gold Pearl Necklace": 95, "Straw Clutch Handbag": 90})
        assert [c[0] for c in _cards(r.out)] == ["apparel", "footwear", "bag", "jewelry"]
        # 'dress' names a type, but a bundle never gets the query-level request
        assert pt.requested_types("summer beach wedding outfit dress") == frozenset({"apparel"})
        assert r.rerank.calls[0]["constraints"] == [{"type": "color", "value": "white"}]
        # every slot reached the rerank window (round-robin by intent, translator order)
        window_slots = [c["source_intent"] for c in r.rerank.calls[0]["window_rows"]]
        assert window_slots[:4] == OUTFIT_INTENTS, window_slots


def test_outfit_rejected_garment_slot_after_good_garments():
    rows = [row("Linen Midi Dress"), row("Linen Shorts for Women"),
            row("Women's Flat Leather Sandals"), row("Gold Hoop Earrings")]
    hits = {"linen midi dress": ["Linen Midi Dress"], "linen shorts": ["Linen Shorts for Women"],
            "leather sandals": ["Women's Flat Leather Sandals"],
            "gold hoop earrings": ["Gold Hoop Earrings"]}
    intents = ["gold hoop earrings", "leather sandals", "linen midi dress", "linen shorts"]
    scores = {"Linen Midi Dress": 30, "Linen Shorts for Women": 80,
              "Women's Flat Leather Sandals": 90, "Gold Hoop Earrings": 95}
    for shape in SHAPES:
        r = run("linen outfit", rows, hits, shape=shape, intents=intents, bundle_type="outfit",
                scores=scores)
        # garment priority first: a rejected dress is still a garment, but after the shorts
        assert [c["source_intent"] for c in r.results] == [
            "linen shorts", "linen midi dress", "leather sandals", "gold hoop earrings"]


def test_outfit_dress_belt_slot_is_an_accessory():
    """'dress belt' was typed apparel, so the belt card sorted before the shoes and the
    keep filter dropped the plain belts typed accessory from its own slot."""
    rows = [row("Men's Slim Fit Navy Suit", cat="Men's Clothing"),
            row("Men's White Dress Shirt", cat="Men's Clothing"),
            row("Men's Black Oxford Shoes", cat="Men's Shoes"),
            row("Men's Dress Pants with Belt", cat="Men's Clothing"),
            row("Men's Full Grain Black Leather Belt with Silver Buckle", cat="Men's Clothing"),
            row("Men's Classic Stitched Leather Dress Belt", cat="Men's Clothing")]
    intents = ["men's navy suit", "men's white dress shirt", "men's black leather dress belt",
               "men's black oxford shoes"]
    hits = {"men's navy suit": ["Men's Slim Fit Navy Suit"],
            "men's white dress shirt": ["Men's White Dress Shirt"],
            "men's black leather dress belt": [
                "Men's Dress Pants with Belt",
                "Men's Full Grain Black Leather Belt with Silver Buckle",
                "Men's Classic Stitched Leather Dress Belt"],
            "men's black oxford shoes": ["Men's Black Oxford Shoes"]}
    for shape in SHAPES:
        r = run("outfit for a job interview", rows, hits, shape=shape, intents=intents,
                bundle_type="outfit")
        assert [c["source_intent"] for c in r.results] == [
            "men's navy suit", "men's white dress shirt", "men's black oxford shoes",
            "men's black leather dress belt"], [c["source_intent"] for c in r.results]
        belt = next(c for c in r.results if c["source_intent"] == intents[2])
        options = [belt] + belt["alternatives"]
        assert [main._row_type(o) for o in options] == ["accessory", "accessory"], options
        assert "men's dress pants with belt" not in [o["Product_title"].casefold() for o in options]


def test_outfit_accessories_slot_takes_the_whole_family():
    """A bare 'accessories' slot typed unknown: it sorted before the shoes and its primary
    was a slip dress. It is an accessory slot now, open to jewellery, bags and hats."""
    rows = [row("Linen Midi Dress"), row("Women's Flat Leather Sandals"),
            row("Gold Satin Slip Dress"), row("Gold Pearl Necklace"),
            row("Straw Clutch Handbag")]
    intents = ["linen midi dress", "delicate gold accessories", "flat leather sandals"]
    hits = {"linen midi dress": ["Linen Midi Dress"],
            "delicate gold accessories": ["Gold Satin Slip Dress", "Gold Pearl Necklace",
                                          "Straw Clutch Handbag"],
            "flat leather sandals": ["Women's Flat Leather Sandals"]}
    for shape in SHAPES:
        r = run("linen outfit", rows, hits, shape=shape, intents=intents, bundle_type="outfit")
        assert [c["source_intent"] for c in r.results] == [
            "linen midi dress", "flat leather sandals", "delicate gold accessories"]
        card = r.results[2]
        assert [main._row_type(o) for o in [card] + card["alternatives"]] == ["jewelry", "bag"]


def test_accessories_ask_is_the_family_and_a_context_noun_is_no_ask():
    """'accessories for a black dress' asked for clothing only (the dress after 'for' was
    read as the request); 'something to go with my black dress' likewise."""
    rows = [row("Gold Drop Earrings"), row("Black Satin Clutch Bag"),
            row("Skinny Gold Waist Belt for Dresses"), row("Women's Little Black Dress"),
            row("Black Bodycon Party Dress"), row("Black Wrap Midi Dress")]
    intents = ["gold earrings", "black clutch", "black dress"]
    hits = {"gold earrings": ["Gold Drop Earrings"],
            "black clutch": ["Black Satin Clutch Bag", "Skinny Gold Waist Belt for Dresses"],
            "black dress": ["Women's Little Black Dress", "Black Bodycon Party Dress",
                            "Black Wrap Midi Dress"]}
    # a reranker that scores the dresses highest: the request must still put them last
    scores = {"Women's Little Black Dress": 95, "Black Bodycon Party Dress": 93,
              "Black Wrap Midi Dress": 91, "Gold Drop Earrings": 70,
              "Black Satin Clutch Bag": 65, "Skinny Gold Waist Belt for Dresses": 60}
    for shape in SHAPES:
        r = run("accessories for a black dress", rows, hits, shape=shape, intents=intents,
                scores=scores)
        family = pt.ACCESSORY_FAMILY
        assert r.types[:3] == ["jewelry", "bag", "accessory"], r.types
        assert _no_mismatch_before_match(r.types, family), r.types
        sent = [c for c in r.rerank.calls[0]["constraints"] if c["type"] == "product_type"]
        assert sent == [main._type_constraint(family)], sent
        assert sent[0]["value"].endswith("; clothing and shoes violate it"), sent

        ctx = run("something to go with my black dress", rows, hits, shape=shape,
                  intents=intents, scores=scores)
        assert ctx.rerank.calls[0]["constraints"] == []          # no type request at all
        assert ctx.types[:3] == ["apparel"] * 3                   # the reranker decides


def test_occasion_noun_after_a_wear_ask_is_no_jewelry_request():
    """'what to wear to a ring ceremony' was read as a jewellery request ('ring' won
    over the clothing ask): the window held only rings and earrings, the reranker was
    told garments violate the ask, and no garment reached page 1."""
    query = "what to wear to a ring ceremony"
    garments = ["Women's Silk Saree for Engagement", "Men's Kurta Pajama Set for Engagement",
                "Anarkali Dress for Women Party Wear", "Men's Sherwani Kurta Set for Wedding",
                "Lehenga Choli for Engagement Ceremony"]
    jewelry = ["Couple Rings for Ring Ceremony", "Diamond Engagement Ring Solitaire",
               "Kundan Earrings for Engagement", "Bridal Necklace Set for Ring Ceremony"]
    rows = [row(t) for t in garments] + [row(t) for t in jewelry]
    intents = ["silk saree for ring ceremony", "men's kurta set for engagement", "anarkali dress"]
    # the rings score first in raw embedding order, as the recorded pool did
    hits = {intents[0]: [jewelry[0], garments[0], jewelry[3]],
            intents[1]: [jewelry[1], garments[1], garments[3]],
            intents[2]: [jewelry[2], garments[2], garments[4]]}
    # a reranker that likes the rings: the request must still keep them after the garments
    scores = {t: 90 for t in jewelry}
    scores.update({t: 80 - i for i, t in enumerate(garments)})
    assert pt.requested_types(query) == frozenset({"apparel"})
    for shape in SHAPES:
        r = run(query, rows, hits, shape=shape, intents=intents, scores=scores)
        sent = [c for c in r.rerank.calls[0]["constraints"] if c["type"] == "product_type"]
        assert sent == [main._type_constraint(frozenset({"apparel"}))], sent
        assert "jewellery only" not in json.dumps(sent)
        window = [main._row_type(c) for c in r.rerank.calls[0]["window_rows"]]
        assert window[:5] == ["apparel"] * 5 and set(window[5:]) == {"jewelry"}, window
        assert r.types[:5] == ["apparel"] * 5, r.types                # no garment demoted
        assert _no_mismatch_before_match(r.types, frozenset({"apparel"})), r.types
        # the misfire, for contrast: a jewellery request buries every garment
        old = run(query, rows, hits, shape=shape, intents=intents, scores=scores,
                  requested=frozenset({"jewelry"}))
        assert old.types[:4] == ["jewelry"] * 4 and old.types[4:] == ["apparel"] * 5, old.types


def test_untyped_product_in_a_conjunction_drops_the_request():
    """'hiking boots and a headlamp': a footwear request ranked the Electronics headlamp
    below every Fashion row, so with 16 boots retrieved it never reached the rerank
    window. The untyped part drops the request: the page is the no-request page."""
    query = "hiking boots and a headlamp"
    boots = [f"{g} Waterproof Hiking Boots Model {i}"
             for i in range(8) for g in ("Men's", "Women's")]
    lamps = ["Rechargeable LED Headlamp 1200 Lumen", "Fenix HL22 LED Headlamp"]
    rows = [row(t, cat="Men's Clothing" if t.startswith("Men") else None) for t in boots]
    rows += [row(t, vertical="Electronics") for t in lamps]
    intents = ["waterproof hiking boots", "rechargeable LED headlamp"]
    hits = {intents[0]: boots, intents[1]: lamps}
    scores = {t: 88 - i for i, t in enumerate(boots)}
    scores.update({lamps[0]: 86, lamps[1]: 84})
    assert pt.requested_types(query) is None
    for shape in SHAPES:
        r = run(query, rows, hits, shape=shape, intents=intents, scores=scores)
        assert all(c["type"] != "product_type" for c in r.rerank.calls[0]["constraints"])
        assert lamps[0].casefold() in r.rerank.calls[0]["window"], r.rerank.calls[0]["window"]
        assert lamps[0].casefold() in r.titles, r.titles                # page 1
        plain = run(query, rows, hits, shape=shape, intents=intents, scores=scores,
                    requested=None)
        assert r.rerank.calls[0]["window"] == plain.rerank.calls[0]["window"]
        assert r.titles == plain.titles, (r.titles, plain.titles)
        # the old request, for contrast (the pool is Fashion-majority, so the gate kept
        # it): neither headlamp reaches the window or page 1
        old = run(query, rows, hits, shape=shape, intents=intents, scores=scores,
                  requested=frozenset({"footwear"}))
        assert not {t.casefold() for t in lamps} & set(old.rerank.calls[0]["window"])
        assert not {t.casefold() for t in lamps} & set(old.titles), old.titles


def test_outfit_next_to_a_type_and_a_type_before_accessorize_e2e():
    """Review 2026-10-15, both on the knob-off flat path. 'outfit and shoes for a
    wedding' came back {footwear}: the reranker was told clothing violates it and the
    dresses (scored 92) sorted below the shoes (70). 'boots to accessorize my outfit'
    came back as the accessory family: the boots sorted after every earring."""
    dresses = [f"Women's Satin Midi Dress {i}" for i in range(3)]
    suits = [f"Men's Slim Fit Wedding Suit {i}" for i in range(3)]
    shoes = [f"Women's Strappy Heeled Sandals {i}" for i in range(6)]
    boots = [f"Women's Suede Ankle Boots {i}" for i in range(3)]
    earrings = [f"Gold Hoop Earrings {i}" for i in range(4)]
    rows = [row(t) for t in dresses + shoes + boots + earrings]
    rows += [row(t, cat="Men's Clothing") for t in suits]

    query = "outfit and shoes for a wedding"
    intents = ["wedding guest midi dress", "wedding suit", "heeled sandals"]
    hits = {intents[0]: dresses, intents[1]: suits, intents[2]: shoes}
    scores = {t: 92 - i for i, t in enumerate(dresses + suits)}
    scores.update({t: 70 - i for i, t in enumerate(shoes)})
    assert pt.requested_types(query) is None
    for shape in SHAPES:
        r = run(query, rows, hits, shape=shape, intents=intents, scores=scores)
        assert all(c["type"] != "product_type" for c in r.rerank.calls[0]["constraints"])
        assert r.types[:6] == ["apparel"] * 6, r.titles     # the 88-92 garments lead

    query = "boots to accessorize my outfit"
    intents = ["ankle boots", "gold earrings"]
    hits = {intents[0]: boots, intents[1]: earrings}
    scores = {t: 90 - i for i, t in enumerate(earrings)}
    scores.update({t: 60 - i for i, t in enumerate(boots)})
    assert pt.requested_types(query) == frozenset({"footwear"})
    for shape in SHAPES:
        r = run(query, rows, hits, shape=shape, intents=intents, scores=scores)
        assert r.types[:3] == ["footwear"] * 3, r.titles     # the boots asked for lead


def test_adjective_conjunction_keeps_the_clothing_request():
    """'light, breezy clothes for a beach wedding': the untyped-part rule read 'light'
    (and 'breezy', 'boho', 'stretchy', ...) as a second, untyped product and dropped the
    request, so the sandals the reranker prefers led page 1 again. An adjective part
    before the garment, after a comma or after another modifier is no product."""
    requested = frozenset({"apparel"})
    for query in ("light, breezy clothes for a beach wedding",
                  "light and airy clothes for a beach wedding in summer",
                  "beach wedding clothes, light and airy", "boho and chic dresses"):
        assert pt.requested_types(query) == requested, query
        for shape in SHAPES:
            r = run(query, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                    scores=BEACH_SCORES)
            sent = [c for c in r.rerank.calls[0]["constraints"] if c["type"] == "product_type"]
            assert sent == [main._type_constraint(requested)], (query, sent)
            window = [main._row_type(c) for c in r.rerank.calls[0]["window_rows"]]
            assert window[:8] == ["apparel"] * 8, (query, window)
            assert r.types[:4] == ["apparel"] * 4, (query, r.types)
            assert _no_mismatch_before_match(r.types, requested), (query, r.types)
            # the regression, for contrast: no request lets the sandals lead
            dropped = run(query, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                          scores=BEACH_SCORES, requested=None)
            assert dropped.types[0] == "footwear", (query, dropped.types)


def test_dress_code_attire_is_clothing_not_neckties():
    """'black tie attire' was read as {accessory}: the title rules classify the pair
    'tie attire' by its weak 'tie', and that pair won over the clothing noun. The
    window held only ties and the reranker was told garments violate the ask."""
    tuxes = ["Men's Slim Fit Tuxedo Black", "Men's Two Piece Tuxedo Suit",
             "Men's Formal Tuxedo Jacket Shawl Lapel", "Men's Black Tie Dinner Suit",
             "Men's Classic Tuxedo with Satin Trim"]
    ties = ["Men's Black Silk Bow Tie", "Black Skinny Necktie for Men",
            "Pre-Tied Black Bow Tie Formal", "Black Satin Neck Tie and Pocket Square Set"]
    rows = [row(t, cat="Men's Clothing") for t in tuxes + ties]
    intents = ["men's tuxedo", "men's black tie suit", "formal tuxedo jacket"]
    hits = {intents[0]: [ties[0], tuxes[0], tuxes[1]],
            intents[1]: [ties[1], tuxes[3], ties[3]],
            intents[2]: [ties[2], tuxes[2], tuxes[4]]}
    scores = {t: 90 for t in ties}
    scores.update({t: 80 - i for i, t in enumerate(tuxes)})
    requested = frozenset({"apparel"})
    for query in ("black tie attire for men", "what to wear to a black tie"):
        assert pt.requested_types(query) == requested, query
        for shape in SHAPES:
            r = run(query, rows, hits, shape=shape, intents=intents, scores=scores)
            sent = [c for c in r.rerank.calls[0]["constraints"] if c["type"] == "product_type"]
            assert sent == [main._type_constraint(requested)], (query, sent)
            assert r.types[:5] == ["apparel"] * 5, (query, r.types)
            old = run(query, rows, hits, shape=shape, intents=intents, scores=scores,
                      requested=frozenset({"accessory"}))
            assert old.types[:4] == ["accessory"] * 4, (query, old.types)


def test_non_fashion_product_named_with_a_fashion_noun_drops_the_request():
    """'gym shorts and an apple watch band' asked for {apparel, accessory}: the pool is
    Fashion-majority, so the gate kept it, and the Electronics watch bands (no Fashion
    type) ranked below every Fashion row, out of the window and off page 1."""
    query = "gym shorts and an apple watch band"
    shorts = [f"{g} Athletic Gym Shorts Model {i}" for i in range(8) for g in ("Men's", "Women's")]
    bands = ["Sport Band Compatible with Apple Watch 45mm", "Apple Watch Nylon Loop Band 41mm"]
    rows = [row(t, cat="Men's Clothing" if t.startswith("Men") else None) for t in shorts]
    rows += [row(t, vertical="Electronics") for t in bands]
    intents = ["men's gym shorts", "women's athletic shorts", "apple watch sport band"]
    hits = {intents[0]: shorts[0::2], intents[1]: shorts[1::2], intents[2]: bands}
    scores = {t: 88 - i for i, t in enumerate(shorts)}
    scores.update({bands[0]: 86, bands[1]: 84})
    assert pt.requested_types(query) is None
    for shape in SHAPES:
        r = run(query, rows, hits, shape=shape, intents=intents, scores=scores)
        assert all(c["type"] != "product_type" for c in r.rerank.calls[0]["constraints"])
        assert bands[0].casefold() in r.rerank.calls[0]["window"], r.rerank.calls[0]["window"]
        assert bands[0].casefold() in r.titles, r.titles
        plain = run(query, rows, hits, shape=shape, intents=intents, scores=scores,
                    requested=None)
        assert r.rerank.calls[0]["window"] == plain.rerank.calls[0]["window"]
        assert r.titles == plain.titles, (r.titles, plain.titles)
        # the old request, for contrast: the Electronics bands (no Fashion type) ranked
        # below every Fashion row, out of the window. Since 2026-10-15 a non-Fashion row
        # is typed by its title under a request, and 'watch band' reads as an accessory,
        # so even that request keeps them in the window and on page 1
        old = run(query, rows, hits, shape=shape, intents=intents, scores=scores,
                  requested=frozenset({"apparel", "accessory"}))
        assert {t.casefold() for t in bands} <= set(old.rerank.calls[0]["window"])
        assert {t.casefold() for t in bands} <= set(old.titles), old.titles


def test_setup_keep_and_order():
    rows = [
        row("Silicone Wedding Ring for Women"), row("Stackable Promise Ring"),
        row("LED Ring Light 10 inch with Tripod", vertical="Electronics"),
        row("Ring Light 18 inch Dimmable", vertical="Electronics"),
        row("Fleece Lined Slippers for Women"), row("Memory Foam House Slippers"),
        row("Gold Pearl Necklace"),
        row("USB Heated Foot Warmer Slippers", vertical="Electronics"),
        row("Warm White Desk Lamp", vertical="Electronics"),
        row("Fairy String Lights 33ft", vertical="Electronics"),
        row("Felt Desk Mat Large", vertical="Electronics"),
        row("Bluetooth Speaker Portable", vertical="Electronics"),
    ]
    hits = {
        "ring light": ["Silicone Wedding Ring for Women", "Stackable Promise Ring",
                       "LED Ring Light 10 inch with Tripod", "Ring Light 18 inch Dimmable"],
        "fleece slippers": ["Gold Pearl Necklace", "Fleece Lined Slippers for Women",
                            "USB Heated Foot Warmer Slippers", "Memory Foam House Slippers"],
        "warm desk lamp": [("Warm White Desk Lamp", 0.80)],
        "string lights": [("Fairy String Lights 33ft", 0.81)],
        "desk mat": [("Felt Desk Mat Large", 0.82)],
        "bluetooth speaker": [("Bluetooth Speaker Portable", 0.97)],
    }
    by_title = {r["Product_title"]: r for r in rows}
    for shape in SHAPES:
        r = run("cozy streaming desk setup", rows, hits, shape=shape,
                intents=["ring light", "fleece slippers"], bundle_type="setup")
        cards = {c["source_intent"]: c for c in r.results}
        ring = [cards["ring light"]] + cards["ring light"]["alternatives"]
        assert all(by_title_vertical(by_title, x) == "Electronics" for x in ring), ring
        slip = [cards["fleece slippers"]] + cards["fleece slippers"]["alternatives"]
        kinds = sorted((by_title_vertical(by_title, x), main._row_type(x)) for x in slip)
        assert kinds == [("Electronics", None), ("Fashion", "footwear"), ("Fashion", "footwear")]

        # translator order (most essential first), whatever the scores; rejected last
        mood = ["warm desk lamp", "string lights", "desk mat", "bluetooth speaker"]
        r = run("cozy desk setup ideas", rows, hits, shape=shape, intents=mood,
                bundle_type="setup",
                scores={"Bluetooth Speaker Portable": 95, "Felt Desk Mat Large": 20,
                        "Fairy String Lights 33ft": 70, "Warm White Desk Lamp": 60})
        assert [c["source_intent"] for c in r.results] == [
            "warm desk lamp", "string lights", "bluetooth speaker", "desk mat"]
        off = run("cozy desk setup ideas", rows, hits, shape=shape, intents=mood,
                  bundle_type="setup", typed=False,
                  scores={"Bluetooth Speaker Portable": 95, "Felt Desk Mat Large": 20,
                          "Fairy String Lights 33ft": 70, "Warm White Desk Lamp": 60})
        assert [c["source_intent"] for c in off.results][0] == "bluetooth speaker"


def by_title_vertical(by_title, item):
    """The Title-case vertical of a served row (the local shape lowercased it)."""
    for title, r in by_title.items():
        if title.casefold() == item["Product_title"].casefold():
            return r["bsns_vrtcl_name"]
    raise KeyError(item["Product_title"])


def test_recipe_order_and_no_keep():
    rows = [row(t, vertical="Grocery") for t in (
        "Organic Black Tea Bags", "Pineapple Rings in Juice", "Basmati Rice 10 lb",
        "Fresh Garlic Bulbs")]
    rows.append(row("Pineapple Toe Ring Silver"))   # in the slot's top hits, typed jewellery
    hits = {"black tea bags": ["Organic Black Tea Bags"],
            "pineapple rings": ["Pineapple Toe Ring Silver", "Pineapple Rings in Juice"],
            "basmati rice": ["Basmati Rice 10 lb"], "garlic": ["Fresh Garlic Bulbs"]}
    intents = ["black tea bags", "pineapple rings", "basmati rice", "garlic"]
    # garlic is not reranked (tail), so first-seen order has it AFTER the rejected tea bags
    scores = {"Organic Black Tea Bags": 20, "Pineapple Toe Ring Silver": 10,
              "Pineapple Rings in Juice": 85, "Basmati Rice 10 lb": 90}
    for shape in SHAPES:
        r = run("ingredients for a tea party", rows, hits, shape=shape, intents=intents,
                bundle_type="recipe", scores=scores)
        assert not r.searches[0]["keep"]                 # recipes are never filtered
        # first-seen (rank) order as before, with the rejected tea bags last
        assert [c["source_intent"] for c in r.results] == [
            "basmati rice", "pineapple rings", "garlic", "black tea bags"]
        off = run("ingredients for a tea party", rows, hits, shape=shape, intents=intents,
                  bundle_type="recipe", scores=scores, typed=False)
        assert [c["source_intent"] for c in off.results] == [
            "basmati rice", "pineapple rings", "black tea bags", "garlic"]
        pineapple = next(c for c in r.results if c["source_intent"] == "pineapple rings")
        assert pineapple["Product_title"].casefold() == "pineapple rings in juice"  # no type sort
        assert r.out["is_recipe"] is True and r.out["bundle_type"] == "recipe"


def test_outfit_gender_coherence():
    rows = [row("Linen Midi Dress", gender="women"),
            row("Men's Leather Sandals", cat="Men's Clothing", gender="men"),
            row("Women's Flat Leather Sandals", gender="women"),
            row("Unisex Slide Sandals", cat="Unisex Clothing", gender="unisex")]
    hits = {"linen midi dress": ["Linen Midi Dress"],
            "leather sandals": ["Men's Leather Sandals", "Women's Flat Leather Sandals",
                                "Unisex Slide Sandals"]}
    intents = ["linen midi dress", "leather sandals"]
    scores = {"Linen Midi Dress": 85, "Men's Leather Sandals": 90,
              "Women's Flat Leather Sandals": 80, "Unisex Slide Sandals": 70}
    for shape in SHAPES:
        r = run("linen outfit", rows, hits, shape=shape, intents=intents, bundle_type="outfit",
                scores=scores)
        sandals = r.results[1]
        options = [sandals] + sandals["alternatives"]
        assert [o["Product_title"].casefold() for o in options] == [
            "women's flat leather sandals", "unisex slide sandals", "men's leather sandals"]
        off = run("linen outfit", rows, hits, shape=shape, intents=intents,
                  bundle_type="outfit", scores=scores, typed=False)
        off_sandals = next(c for c in off.results if c["source_intent"] == "leather sandals")
        assert off_sandals["Product_title"].casefold() == "men's leather sandals"
        # no anchor gender (a unisex garment): option order untouched
        neutral_rows = [dict(rows[0], gender="unisex", categ_lvl2_name="Unisex Clothing")]
        neutral_rows += rows[1:]
        r = run("linen outfit", neutral_rows, hits, shape=shape, intents=intents,
                bundle_type="outfit", scores=scores)
        sandals = next(c for c in r.results if c["source_intent"] == "leather sandals")
        assert sandals["Product_title"].casefold() == "men's leather sandals"


# ------------------------------------------------------------------ kill switch
def test_knob_false_matches_the_frozen_pre_change_ordering():
    rows = [
        row("Floral Midi Dress"), row("Linen Shirt", cat="Men's Clothing"),
        row("Wrap Dress"), row("Chino Shorts", cat="Men's Clothing"),
        row("Wedge Sandals"), row("Canvas Sneakers", cat="Men's Clothing"),
        row("Straw Sun Hat", cat="Unisex Clothing"),
        row("Bluetooth Speaker Portable", vertical="Electronics"),
        row("Pleated Skirt"), row("Polo Shirt", cat="Men's Clothing"),
    ]
    intents = ["summer clothes", "summer shoes"]
    hits = {"summer clothes": ["Floral Midi Dress", "Wrap Dress", "Pleated Skirt",
                               "Straw Sun Hat", "Linen Shirt", "Chino Shorts", "Polo Shirt"],
            "summer shoes": ["Wedge Sandals", "Bluetooth Speaker Portable", "Canvas Sneakers"]}
    scores = {r["Product_title"]: 90 - 3 * i for i, r in enumerate(rows) if i % 3}
    for shape in SHAPES:
        served_rows = as_rows(rows, shape)
        assert [main._product_gender(r) for r in served_rows] == \
            [_old_product_gender(r) for r in served_rows], shape   # inputs both agree on
        # the balancers themselves
        assert main._balance_by_gender(served_rows) == _old_balance_by_gender(served_rows)
        assert main._balance_candidate_pool(served_rows) == \
            _old_balance_candidate_pool(served_rows)
        # the whole flat path
        r = run("summer clothes", rows, hits, shape=shape, intents=intents, scores=scores,
                typed=False)
        with knobs(), served(rows, hits, shape):
            raw = S.search_products(intents, top_k=main.RETRIEVAL_TOP_K * 2)
            window = _old_balance_candidate_pool(raw)
            ranked = FakeRerank(scores)("summer clothes", window, top_k=main.RERANK_POOL_K)
            expected = _old_balance_by_gender(main._assemble_full(ranked, window))
        assert r.rerank.calls[0]["window"] == [c["Product_title"].casefold() for c in window]
        assert r.titles == [c["Product_title"].casefold() for c in expected][:12], shape
        assert r.searches == [{"top_k": main.RETRIEVAL_TOP_K * 2, "quota": None, "keep": False}]
        # the bundle path: score-order window, first-seen slots, no keep=
        b = run("summer outfit", rows, hits, shape=shape, intents=intents,
                bundle_type="outfit", scores=scores, typed=False)
        with knobs(), served(rows, hits, shape):
            raw = S.search_products(intents, top_k=main.RETRIEVAL_TOP_K,
                                    per_intent_quota=main.GROCERY_PER_INTENT_K)
            ranked = FakeRerank(scores)("summer outfit", raw, top_k=main.RERANK_POOL_K)
            slots = main._recipe_slots_with_alternatives(main._assemble_full(ranked, raw))
        assert b.rerank.calls[0]["window"] == [c["Product_title"].casefold() for c in raw][:15]
        assert [c["Product_title"] for c in b.results] == [c["Product_title"] for c in slots]
        assert not b.searches[0]["keep"]


def test_flipping_the_knob_changes_order_and_cache_key():
    shape = "local"
    common = dict(shape=shape, intents=BEACH_INTENTS, scores=BEACH_SCORES, clear=False)
    page_cache.clear()
    try:
        on = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, **common)
        keys_after_on = list(page_cache._store)
        off = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, typed=False, **common)
        keys = list(page_cache._store)
    finally:
        page_cache.clear()
    assert on.titles != off.titles
    assert on.types[0] == "apparel" and off.types[0] == "footwear", (on.types, off.types)
    assert len(keys_after_on) == 1 and len(keys) == 2
    assert keys[0].endswith("True))") and keys[1].endswith("False))"), keys
    assert keys[0][: -len("True))")] == keys[1][: -len("False))")]


def test_keep_is_passed_only_to_typed_outfit_and_setup():
    for bundle, typed, expected in [(None, True, False), ("recipe", True, False),
                                    ("outfit", True, True), ("setup", True, True),
                                    ("outfit", False, False), ("setup", False, False)]:
        r = run("x", BEACH_ROWS, BEACH_HITS, shape="local", intents=BEACH_INTENTS,
                bundle_type=bundle, typed=typed)
        assert r.searches[0]["keep"] is expected, (bundle, typed)


# ------------------------------------------------------------------ C11: upsell
def test_upsell_keeps_a_typed_anchor_to_its_type():
    neighbours = [
        {"Product_title": "Gladiator Sandals", pt.TYPE_COLUMN: "footwear",
         "average_rating": 4.9, "rating_number": 5000, "parent_asin": "S1"},
        {"Product_title": "Linen Maxi Dress", pt.TYPE_COLUMN: "apparel",
         "average_rating": 4.6, "rating_number": 3000, "parent_asin": "D1"},
        {"Product_title": "Model-named Shoe", pt.TYPE_COLUMN: float("nan"),
         "average_rating": 5.0, "rating_number": 9000, "parent_asin": "U1"},
    ]
    anchor = {"Product_title": "Linen Midi Dress", "average_rating": 4.0,
              "rating_number": 100, "parent_asin": "A1"}
    with patched(recs, {"search_products": lambda terms, top_k=30: [dict(n) for n in neighbours]}):
        def pick(**extra):
            up = recs._upsell(dict(anchor, **extra), set())
            return up[0]["Product_title"] if up else None

        assert pick(**{pt.TYPE_COLUMN: "apparel"}) == "Linen Maxi Dress"
        assert pick(**{pt.TYPE_COLUMN: "footwear"}) == "Gladiator Sandals"
        # 'unknown', None (other verticals), NaN, missing: the plain neighbour scope
        for value in ("unknown", None, float("nan")):
            assert pick(**{pt.TYPE_COLUMN: value}) == "Model-named Shoe", value
        assert pick() == "Model-named Shoe"


def test_upsell_keeps_a_device_or_a_watch_to_its_kind():
    """Review 2026-10-30: an anchor with a load-time item_role upsells only that role, a
    watch-related one only its watch_kind. A phone's look-alikes are mostly its batteries,
    cables and cases, and a wristwatch's its bands: each won 'Better-rated alternative'
    on rating alone. None (and watch_kind 'other') keep the plain neighbour scope."""
    role, watch = ct.ROLE_COLUMN, ct.WATCH_COLUMN

    def row(title, rating, n, asin, **cols):
        return {"Product_title": title, "average_rating": rating, "rating_number": n,
                "parent_asin": asin, **cols}

    phones = [
        row("Phone Case", 4.9, 9000, "C1", **{role: "accessory"}),
        row("Phone Battery Pack", 4.8, 7000, "B1", **{role: "consumable"}),
        row("Better Phone", 4.6, 900, "P2", **{role: "device"}),
    ]
    watches = [
        row("Watch Band", 4.9, 9000, "W1", **{pt.TYPE_COLUMN: "accessory", watch: "band"}),
        row("Better Watch", 4.6, 900, "W2", **{pt.TYPE_COLUMN: "accessory", watch: "watch"}),
    ]
    groceries = [
        row("Coffee Grinder", 4.9, 9000, "G1", **{role: "non_food"}),
        row("Better Coffee Beans", 4.6, 900, "G2", **{role: "food"}),
    ]

    def pick(neighbours, **anchor):
        base = row("Anchor", 4.0, 100, "A1")
        with patched(recs, {"search_products":
                            lambda terms, top_k=30: [dict(n) for n in neighbours]}):
            up = recs._upsell(dict(base, **anchor), set())
        return up[0]["Product_title"] if up else None

    assert pick(phones, **{role: "device"}) == "Better Phone"
    assert pick(phones, **{role: "accessory"}) == "Phone Case"
    assert pick(watches, **{pt.TYPE_COLUMN: "accessory", watch: "watch"}) == "Better Watch"
    assert pick(watches, **{pt.TYPE_COLUMN: "accessory", watch: "band"}) == "Watch Band"
    assert pick(groceries, **{role: "food"}) == "Better Coffee Beans"
    # No better row of its own kind: nothing, never the accessory
    assert pick(phones[:2], **{role: "device"}) is None
    # No role, no watch_kind, an 'other' watch_kind: the plain neighbour scope (as before)
    for anchor in ({}, {role: None}, {watch: None}, {watch: "other"}):
        assert pick(phones, **anchor) == "Phone Case", anchor
    assert pick(watches, **{pt.TYPE_COLUMN: "accessory"}) == "Watch Band"


# ------------------------------------------------------------------ trace replay
def _trace_case(entry):
    """Rows, per-intent hits and rerank scores recorded live (local backend, gte-small,
    gpt-oss), in the local row shape. Fashion rows' gender tags were not recorded, so
    their gender comes from category/title words."""
    rows, hits, seen = [], {}, {}

    def add(title, cat, label, occasion, intent, score):
        if title not in seen:
            vertical = ("electronics" if label.startswith("electronics")
                        else "grocery" if label.startswith("grocery") else "fashion")
            seen[title] = len(rows)
            rows.append({"bsns_vrtcl_name": vertical, "categ_lvl2_name": cat or "",
                         "Product_title": title, "occasion": occasion or "",
                         "gender": "", "parent_asin": f"T{len(rows)}"})
        listed = hits.setdefault(intent, [])
        if all(t != title for t, _ in listed):
            listed.append((title, score))

    for block in entry["per_intent"]:
        for h in block["hits"]:
            add(h["title"], h.get("cat"), h.get("label", ""), h.get("occasion"),
                block["intent"], h["score"])
    for r in entry["rerank"]:   # rows the live window held beyond the recorded top 10
        add(r["title"], "", r.get("label", ""), "", r["intent"], r["emb"])
    for listed in hits.values():
        listed.sort(key=lambda pair: -pair[1])
    scores = {r["title"]: r["rerank_score"] for r in entry["rerank"]
              if r.get("rerank_score") is not None}
    return rows, hits, scores


@contextlib.contextmanager
def _served_local_rows(rows, hits):
    df = pd.DataFrame(rows)
    pt.annotate(df)
    position = {r["Product_title"]: i for i, r in enumerate(rows)}

    def raw_hits(terms, top_k):
        return [[(position[t], s) for t, s in hits.get(term, [])][:top_k] for term in terms]

    with patched(S, {"_df": df, "_ready": lambda: True, "_raw_hits": raw_hits}):
        yield


def _replay(entry, typed):
    rows, hits, scores = _trace_case(entry)
    tr = entry["translator"]
    fake_rerank = FakeRerank(scores)

    def understand_query(q, mode=None, errors=None):
        return {"intents": list(tr["intents"]), "bundle_type": tr["bundle_type"],
                "constraints": [dict(c) for c in tr["constraints"]],
                "is_recipe": tr["bundle_type"] == "recipe", "query": q}

    page_cache.clear()
    with knobs(TYPE_AWARE_RANKING=typed), _served_local_rows(rows, hits), patched(main, {
        "understand_query": understand_query, "llm_rerank": fake_rerank,
        "MIN_RESULT_RELEVANCE": 0.0, "sponsored_map": lambda: {},
    }):
        out = main.search(query=entry["query"], top_k=12, page=1, rerank=True,
                          recommend=False, sponsored=False, translator_mode=None)
    page_cache.clear()
    return out


def _flat_line(results):
    out = []
    for r in results:
        t = pt.classify_title(r["Product_title"])[0] if main._row_type(r) is None and \
            pt.is_fashion(r.get("bsns_vrtcl_name")) else (main._row_type(r) or "non-fashion")
        score = r.get("rerank_score")
        out.append(f"{t}{'' if score is not None else '~'}")
    return out


def _card_line(results):
    return [f"{pt.slot_type(c['source_intent'])}->{main._row_type(c) or 'non-fashion'}"
            f"{'' if c.get('rerank_score') is not None else '~'}" for c in results]


def test_trace_replay():
    path = os.getenv("CAS_TRACE_JSON")
    if not path or not Path(path).exists():
        _skip("set CAS_TRACE_JSON to the recorded trace_out.json")
    trace = {e["query"]: e for e in json.loads(Path(path).read_text(encoding="utf-8"))}
    print()
    for query in ("cloths for a beach wedding in summer", "beach wedding outfit",
                  "summer beach wedding outfit dress"):
        entry = trace[query]
        live = entry["variants"]["full"]["rows"]
        before_off = _replay(entry, typed=False)
        after = _replay(entry, typed=True)
        print(f"== {query!r}  (bundle={entry['translator']['bundle_type']}; "
              f"'~' = no recorded rerank score)")
        if entry["translator"]["bundle_type"]:
            recorded = [f"{pt.slot_type(r['slot'])}->{pt.classify_title(r['title'])[0]}"
                        for r in live]
            print("   before (recorded live):   ", recorded)
            print("   before (replay, knob off):", _card_line(before_off["results"]))
            print("   after  (replay, knob on): ", _card_line(after["results"]))
            slots = [pt.slot_type(c["source_intent"]) for c in after["results"]]
            assert slots[0] == "apparel", slots
            first_other = next((i for i, s in enumerate(slots) if s != "apparel"), len(slots))
            assert all(s != "apparel" for s in slots[first_other:]), slots
            for c in after["results"]:
                stype = pt.slot_type(c["source_intent"])
                for o in [c] + c.get("alternatives", []):
                    if stype in pt.FASHION_TYPES:
                        assert main._row_type(o) in (stype, pt.UNKNOWN), (stype, o["Product_title"])
        else:
            recorded = [pt.classify_title(r["title"])[0] for r in live][:12]
            print("   before (recorded live):   ", recorded)
            before_types = _flat_line(before_off["results"])
            after_types = _flat_line(after["results"])
            print("   before (replay, knob off):", before_types)
            print("   after  (replay, knob on): ", after_types)
            for name, seq in (("recorded", recorded), ("knob off", before_types),
                              ("knob on", after_types)):
                seq = [s.rstrip("~") for s in seq]
                print(f"   {name:9s} non-apparel in top 4: {sum(s != 'apparel' for s in seq[:4])}"
                      f", top 12: {sum(s != 'apparel' for s in seq[:12])}")
            assert all(s.rstrip("~") == "apparel" for s in after_types[:12]), after_types


# ------------------------------------------------------------------ 2026-10-15 (WP2)
def test_page_size_not_in_cache_key():
    """The "Results per page" slider re-ran both LLM calls and served a different page
    (Oct 15 review): page_size was in the page_cache key although the pool is the same
    for every size up to RERANK_POOL_K. Page 1's cross-sell is kept per size instead."""
    common = dict(shape="local", intents=BEACH_INTENTS, scores=BEACH_SCORES, clear=False)
    page_cache.clear()
    try:
        first = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, top_k=12, **common)
        keys = list(page_cache._store)
        pool = [r["Product_title"].casefold() for r in page_cache._store[keys[0]]["full"]]
        second = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, top_k=24, **common)
        assert list(page_cache._store) == keys          # the same entry served it
        deeper = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, top_k=31, **common)
        keys_after = list(page_cache._store)
    finally:
        page_cache.clear()
    assert len(first.translated) == 1 and len(first.rerank.calls) == 1
    assert second.translated == [] and second.rerank.calls == []   # 0 LLM calls
    assert second.titles == pool[:24] and first.titles == pool[:12]
    assert first.out["total_results"] == second.out["total_results"] == len(pool)
    assert (first.out["total_pages"], second.out["total_pages"]) == (2, 1)
    # a page bigger than the pool raises pool_k: that IS a different pool, a new key
    assert len(deeper.translated) == 1 and len(keys_after) == 2

    # recommend=True: page 1's cross-sell is per size inside the one entry
    page_cache.clear()
    try:
        a = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, top_k=12, recommend=True, **common)
        b = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, top_k=8, recommend=True, **common)
        c = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, top_k=12, recommend=True, **common)
        n_keys = len(page_cache._store)
    finally:
        page_cache.clear()
    assert len(a.translated) == 1 and b.translated == [] and c.translated == []
    assert [len(x) for x in a.recs] == [12] and [len(x) for x in b.recs] == [8]
    assert c.recs == [] and n_keys == 1                 # size 12's cross-sell was kept
    assert a.recs[0] == pool[:12] and b.recs[0] == pool[:8]


def test_outfit_knob_in_cache_key():
    """OUTFIT_BUNDLES_ENABLED decides the translator's route: flipping it in-process must
    not serve the other mode's pool. It sits BEFORE the ordering knob, which stays last."""
    common = dict(shape="local", intents=BEACH_INTENTS, scores=BEACH_SCORES, clear=False)
    page_cache.clear()
    try:
        run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, knob_values={"OUTFIT_BUNDLES_ENABLED": False},
            **common)
        on = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS,
                 knob_values={"OUTFIT_BUNDLES_ENABLED": True}, **common)
        keys = list(page_cache._store)
    finally:
        page_cache.clear()
    assert len(on.translated) == 1 and len(keys) == 2
    assert keys[0].endswith("False, True))") and keys[1].endswith("True, True))"), keys
    assert keys[0][: -len("False, True))")] == keys[1][: -len("True, True))")]


def test_women_garment_nouns_state_gender():
    stated = ["Wedding dresses for a destination wedding in rajasthan", "maxi dresses",
              "red dress for a party", "wedding dress", "evening gowns", "silk sarees",
              "saree", "lehenga for a sangeet", "pencil skirts", "chiffon blouses",
              "strapless bras", "bikinis for a beach holiday"]
    unstated = ["dress shoes", "Dress  Shoes", "dress shirt", "dress shirts for work",
                "dress pants", "dress socks", "dress up costume", "skirt steak",
                "bikini trimmer", "car bra", "shoes", "beach wedding outfit",
                "cloths for a beach wedding in summer", "address book",
                # 'dress' the VERB (review 2026-10-15: these skipped the 2x retrieval
                # and the interleave, so a job-interview ask served 0 men's items)
                "how should I dress for a job interview", "how to dress for a wedding in summer",
                "dress for success", "what to dress my toddler in for a wedding",
                "dress to impress", "dress like a pirate",
                # 'dress' the MODIFIER of another type noun, with a space or a hyphen
                "dress coat for winter", "dress jacket for a wedding", "navy dress blazer",
                "dress suit for prom", "dress vest", "dress sneakers", "dress loafers",
                "dress gloves", "dress tie", "dress hat", "dress-shirt for work", "dress-shoes",
                "kids dress-up costumes", "dress code for a gala",
                "dressing gown"]                              # a unisex robe, not a gown
    for q in stated:
        assert main._gender_stated(q, [], typed=True), q
        assert not main._gender_stated(q, [], typed=False), q   # type-aware path only
    for q in unstated:
        assert not main._gender_stated(q, [], typed=True), q
    # the singular garment itself still states it: after an article or a possessive,
    # before no type noun ("dress for a beach wedding", "dress, size 8")
    for q in ("a dress to wear to a wedding", "my dress for prom", "a dress like a princess's",
              "dress for a beach wedding", "little black dress", "dress, size 8",
              "dress with pockets", "sundress"):
        assert main._gender_stated(q, [], typed=True), q
    # a men's 'dress' ask is stated by its men word, as before
    for q in ("dress shoes for men", "men's dress shirt"):
        assert main._gender_stated(q, [], typed=True) and main._gender_stated(q, [], typed=False)


def test_dress_verb_keeps_the_gender_balance_end_to_end():
    """'how should I dress for ...' asks what to wear: the same 2x-deep retrieval and
    interleave as 'what should I wear to ...' (review 2026-10-15: it served 0 men's
    items on page 1 when the verb counted as a women's garment)."""
    women = [f"Women's Interview Blazer {i}" for i in range(12)]
    men = [f"Men's Interview Suit Jacket {i}" for i in range(4)]
    rows = [row(t) for t in women] + [row(t, cat="Men's Clothing") for t in men]
    intents = ["interview blazer"]
    hits = {intents[0]: [(t, 0.95 - 0.001 * i) for i, t in enumerate(women + men)]}
    scores = {t: 90 - i for i, t in enumerate(women + men)}
    for shape in SHAPES:
        pages = {}
        for q in ("what should I wear to a job interview",
                  "how should I dress for a job interview", "dress coat for winter"):
            r = run(q, rows, hits, shape=shape, intents=intents, scores=scores)
            assert r.searches[0]["top_k"] == main.RETRIEVAL_TOP_K * 2, (q, r.searches)
            pages[q] = [main._product_gender(x) for x in r.results]
            # the men's 78-75 interleave inside the 71-85 tier only (REJECTS_AFTER_TAIL),
            # below the women's 90-86: 3 of them on page 1
            assert pages[q].count("men") == 3, (shape, q, pages[q])
            assert pages[q][:5] == ["women"] * 5, (shape, q, pages[q])
            # across the whole band (the 2026-10-14 display) they took 4 early slots
            old = run(q, rows, hits, shape=shape, intents=intents, scores=scores,
                      knob_values={"REJECTS_AFTER_TAIL": False})
            assert [main._product_gender(x) for x in old.results].count("men") == 4
        assert len(set(map(tuple, pages.values()))) == 1, pages


# The Oct 15 rajasthan page as cached live (repro runs/raj_k12_a.json): intents, gender
# tags, categories and rerank scores as recorded; titles shortened
RAJ_QUERY = "Wedding dresses for a destination wedding in rajasthan"
RAJ_INTENTS = ["Traditional Rajasthani embroidered wedding dress",
               "Lightweight silk wedding gown with desert-inspired detailing",
               "Floor-length lehenga with gold threadwork for Rajasthan wedding"]
RAJ = [  # (title, intent, gender, category, rerank score)
    ("anantdiva women's net saree indian embroidery wedding party wear", 0, "women",
     "Women's Clothing", 95),
    ("sakala party wear designer bollywood replica lehenga choli", 2, "unisex",
     "Unisex Clothing", 92),
    ("daskall dashiki shirt for men printed button up shirt long sleeve", 0, "men",
     "Men's Clothing", 55),
    ("wedding wear designer salwar dress heavy embroidery salwar kameez", 2, "women",
     "Women's Clothing", 85),
    ("lyrur women's sheath feast dress party wedding guest knee length", 2, "women",
     "Women's Clothing", 78),
    ("craft trade dupatta for indian wear rajasthani hand embroidered", 0, "unisex",
     "Unisex Clothing", 60),
    ("geshun womens desert rose mother of the bride dress with lace", 1, "women",
     "Women's Clothing", 70),
    ("faybox bridesmaid peacock short kimono robe wedding satin", 1, "unisex",
     "Unisex Clothing", 40),
    ("numbersea chiffon mother of the bride dresses with jacket", 1, "women",
     "Women's Clothing", 40),
]


def test_rajasthan_dashiki_no_longer_interleaved_above_womens_dresses():
    rows = [row(t, cat=cat, gender=g) for t, _, g, cat, _ in RAJ]
    hits = {}
    for t, i, *_ in RAJ:
        hits.setdefault(RAJ_INTENTS[i], []).append(t)
    scores = {t: s for t, *_, s in RAJ}
    dashiki, salwar, sheath = RAJ[2][0], RAJ[3][0], RAJ[4][0]
    assert pt.requested_types(RAJ_QUERY) == frozenset({"apparel"})
    never = __import__("re").compile(r"(?!x)x")
    for shape in SHAPES:
        r = run(RAJ_QUERY, rows, hits, shape=shape, intents=RAJ_INTENTS, scores=scores)
        at = {t: i for i, t in enumerate(r.titles)}
        assert at[dashiki] > at[salwar] and at[dashiki] > at[sheath], r.titles
        assert r.searches[0]["top_k"] == main.RETRIEVAL_TOP_K      # stated: 1x deep
        # the recorded failure, for contrast: without the cue the interleave across the
        # whole band (the 2026-10-14 display) gives the lone men's row the second gendered
        # position (#3 on the live page)
        with patched(main, {"_WOMEN_GARMENT_RE": never}):
            old = run(RAJ_QUERY, rows, hits, shape=shape, intents=RAJ_INTENTS, scores=scores,
                      knob_values={"REJECTS_AFTER_TAIL": False})
            tiered = run(RAJ_QUERY, rows, hits, shape=shape, intents=RAJ_INTENTS,
                         scores=scores)
        assert old.titles.index(dashiki) == 2, old.titles
        assert old.titles.index(dashiki) < old.titles.index(sheath)
        # ...and the 15-point tiers alone (2026-10-19) keep a 55 below every dress scored
        # 70-95 even when the gender reads unstated
        at = {t: i for i, t in enumerate(tiered.titles)}
        assert at[dashiki] > max(at[salwar], at[sheath], at[RAJ[6][0]]), tiered.titles


# "power bank" in miniature (xvertical.md): three paraphrase intents, one of which
# retrieves junk at a cosine the others' real banks share
PB_INTENTS = ["portable power bank", "high capacity power bank", "fast charging power bank"]
PB_REAL = [f"Portable Power Bank {i} 10000mAh" for i in range(12)]
PB_FAST = [f"Fast Charging Power Bank PD {i}" for i in range(3)]
PB_JUNK = ["Car Audio Amplifier 4 Channel", "ATX Power Supply 650W", "Pure Sine Inverter 1000W",
           "Universal Travel Plug Adapter", "Firstbuy", "24V Battery Charger"]


def _pb_case():
    rows = [row(t, vertical="Electronics") for t in PB_REAL + PB_FAST + PB_JUNK]
    hits = {
        PB_INTENTS[0]: [(t, 0.931 - 0.003 * i) for i, t in enumerate(PB_REAL)],
        PB_INTENTS[1]: [(t, 0.8955 - 0.003 * i) for i, t in enumerate(PB_JUNK)],
        PB_INTENTS[2]: [(t, 0.9145 - 0.003 * i) for i, t in enumerate(PB_FAST)],
    }
    return rows, hits


def test_untyped_window_one_seat_per_intent_then_score_order():
    rows, hits = _pb_case()
    junk = {t.casefold() for t in PB_JUNK}
    real, fast = [t.casefold() for t in PB_REAL], [t.casefold() for t in PB_FAST]
    assert pt.requested_types("power bank") is None
    for shape in SHAPES:
        r = run("power bank", rows, hits, shape=shape, intents=PB_INTENTS)
        window = r.rerank.calls[0]["window"]
        # one seat per intent, in the translator's order (the junk intent gets ONE, its
        # best row), then raw score order: the real banks and the fast-charging ones
        assert window == [real[0], PB_JUNK[0].casefold(), fast[0]] + real[1:7] + [
            fast[1], real[7], fast[2], real[8], real[9], real[10]], window
        assert sum(t in junk for t in window) == 1
    # round-robin (2026-10-14..15), for contrast: the junk intent held 6 of 15 seats
    with knobs(), served(rows, hits, "local"):
        raw = S.search_products(PB_INTENTS, top_k=main.RETRIEVAL_TOP_K * 2)
    rr = main._round_robin_by_intent(raw, PB_INTENTS)[:15]
    assert sum(c["Product_title"].casefold() in junk for c in rr) == 6
    # ...and plain score order (before 2026-10-14) let the strongest intent fill it
    assert main._seat_intents(raw, PB_INTENTS, 0) == raw
    assert raw[0]["source_intent"] == PB_INTENTS[0]
    # a typed request keeps the tiers AND the round-robin (one tier here: no row is a bag)
    typed_window = main._flat_window(raw, frozenset({"bag"}), PB_INTENTS, False)
    assert typed_window == main._round_robin_by_intent(raw, PB_INTENTS)


def test_non_fashion_row_typed_by_title_under_a_request():
    """'kids backpack' passes the pool gate (Fashion-majority) and asks for bags: the
    Electronics school backpacks ranked 2 (type_rank(None)) below every Fashion row and
    out of the window. Typed by title they rank with the Fashion bags; a non-Fashion
    row whose title reads as no requested type still ranks 2."""
    query = "kids backpack"
    fashion = [f"Kids Backpack Cartoon Print Model {i}" for i in range(14)]
    school = ["Kids School Backpack with Laptop Compartment",
              "Boys Backpack for School Water Resistant"]
    noise = ["Laptop Cooling Pad 5 Fans", "Fruit Snack Pouches for Kids"]
    rows = [row(t) for t in fashion] + [row(t, vertical="Electronics") for t in school]
    rows += [row(noise[0], vertical="Electronics"), row(noise[1], vertical="Grocery")]
    order = [school[0], fashion[0], school[1]] + fashion[1:9] + [noise[0], noise[1]] + fashion[9:]
    hits = {query: [(t, 0.95 - 0.002 * i) for i, t in enumerate(order)]}
    scores = {t: 70 for t in fashion}
    scores.update({school[0]: 92, school[1]: 88, noise[0]: 10, noise[1]: 5})
    assert pt.requested_types(query) == frozenset({"bag"})
    for shape in SHAPES:
        served_rows = {r["Product_title"].casefold(): r for r in as_rows(rows, shape)}
        bag = frozenset({"bag"})
        assert main._type_rank(served_rows[school[0].casefold()], bag) == 0
        assert pt.type_rank(None, bag) == 2                     # what it was before
        assert main._type_rank(served_rows[noise[0].casefold()], bag) == 2
        assert main._type_rank(served_rows[noise[1].casefold()], bag) == 2
        assert main._type_rank(served_rows[fashion[0].casefold()], bag) == 0
        assert main._type_rank(served_rows[school[0].casefold()], None) == 0
        r = run(query, rows, hits, shape=shape, intents=[query], scores=scores)
        window = r.rerank.calls[0]["window"]
        assert {t.casefold() for t in school} <= set(window), window
        assert not {t.casefold() for t in noise} & set(window), window
        assert r.titles[:2] == [t.casefold() for t in school], r.titles
        # the other 2d examples: USB heated gloves on an accessory ask
        gloves = dict(served_rows[school[0].casefold()],
                      Product_title="USB Heated Gloves for Men Women")
        assert main._type_rank(gloves, frozenset({"accessory"})) == 0


def test_setup_keep_accepts_the_slot_type_and_refills_an_empty_slot():
    rows = [
        row("Women's Linen Midi Dress"), row("Canvas Beach Tote Bag"),
        row("Straw Beach Tote with Zipper"),
        row("Waterproof Phone Pouch", vertical="Electronics"),
        row("Wide Brim Straw Sun Hat"), row("Silicone Wedding Ring for Women"),
        row("LED Ring Light 10 inch with Tripod", vertical="Electronics"),
        row("Redcamp Portable Backpacking Stove with Piezo Ignition"),
        row("Campingaz Fuel Cartridge", vertical="Electronics"),
        row("Wearable Blanket Hoodie Sherpa"),
    ]
    by_title = {r["Product_title"]: r for r in rows}
    intents = ["beach tote bag", "sun hat", "ring light", "camping stove and fuel",
               "throw blanket"]
    hits = {
        "beach tote bag": ["Women's Linen Midi Dress", "Canvas Beach Tote Bag",
                           "Straw Beach Tote with Zipper", "Waterproof Phone Pouch"],
        "sun hat": ["Silicone Wedding Ring for Women", "Wide Brim Straw Sun Hat"],
        "ring light": ["Silicone Wedding Ring for Women", "LED Ring Light 10 inch with Tripod"],
        "camping stove and fuel": ["Redcamp Portable Backpacking Stove with Piezo Ignition",
                                   "Campingaz Fuel Cartridge"],
        "throw blanket": ["Wearable Blanket Hoodie Sherpa"],
    }
    assert pt.classify_title(rows[7]["Product_title"])[0] == pt.UNKNOWN   # the stove
    assert pt.slot_accepts("throw blanket") == frozenset()
    for shape in SHAPES:
        r = run("beach day setup", rows, hits, shape=shape, intents=intents,
                bundle_type="setup")
        cards = {c["source_intent"]: [c] + c["alternatives"] for c in r.results}

        def kinds(slot):
            return [(by_title_vertical(by_title, x), main._row_type(x)) for x in cards[slot]]

        # a Fashion row stays when its type is the slot's (before: 0 Fashion bags/hats)
        assert kinds("beach tote bag") == [("Fashion", "bag"), ("Fashion", "bag"),
                                           ("Electronics", None)], kinds("beach tote bag")
        assert kinds("sun hat") == [("Fashion", "headwear")]
        # the device veto still drops the wedding ring from the ring light slot
        assert kinds("ring light") == [("Electronics", None)]
        # an untyped Fashion row is kept: the Fashion-catalogued stove (reranked 85 live)
        assert ("Fashion", pt.UNKNOWN) in kinds("camping stove and fuel")
        # keep= emptied "throw blanket" (an apparel row in an untyped slot): refilled
        # from the unfiltered hits, so the card is there for the reranker to judge
        assert kinds("throw blanket") == [("Fashion", "apparel")]
        assert len(r.searches) == 2 and r.searches[0]["keep"] and not r.searches[1]["keep"]
        assert r.searches[1]["quota"] == main.GROCERY_PER_INTENT_K
        # knob off: no keep, no refill
        off = run("beach day setup", rows, hits, shape=shape, intents=intents,
                  bundle_type="setup", typed=False)
        assert len(off.searches) == 1 and not off.searches[0]["keep"]


# ------------------------------------------------------------------ 2026-10-19 (WP-R)
def _attr(title, ptype_, intent, audience=None, fine=None, gender="women", score=None,
          vertical="Fashion"):
    """A served row with the C1 columns set by hand (no annotate: these tests are about
    main.py's use of them, not product_type's reading of titles)."""
    return {"Product_title": title, pt.TYPE_COLUMN: ptype_, "source_intent": intent,
            "bsns_vrtcl_name": vertical, "gender": gender, "categ_lvl2_name": "",
            "audience": audience, "fine_subtype": fine, "rerank_score": score}


def _fit_reader(text):
    """A stand-in requested_fit: 'petite' / 'maternity' words, query or title."""
    words = set(str(text).casefold().replace("'", " ").split())
    found = frozenset(w for w in ("petite", "maternity") if w in words)
    return found or None


def _titles(rows):
    return [r["Product_title"] for r in rows]


def test_window_partitions_by_age_fit_and_subtype():
    """'petite work pants' (synthesis P3, owner call 3): inside each type tier the rows
    of the right age come first, then the ones matching more of the named fit and
    style; each part keeps its round-robin order and is interleaved on its own."""
    intents = ["petite work pants"]
    cands = [
        _attr("Girls Petite Pants", "apparel", intents[0], "kids", "pants"),
        _attr("Women's Blazer", "apparel", intents[0], "adult", "suits_blazers"),
        _attr("Women's Work Pants", "apparel", intents[0], "adult", "pants"),
        _attr("Women's Petite Blazer", "apparel", intents[0], "adult", "suits_blazers"),
        _attr("Women's Petite Work Pants", "apparel", intents[0], "adult", "pants"),
        _attr("Kids Petite Backpack", "bag", intents[0], "kids", None),
    ]
    apparel = frozenset({"apparel"})
    ask = main._Ask(fit=frozenset({"petite"}), subtypes=frozenset({"pants"}),
                    audience=frozenset({"adult"}))
    with knobs(), c1_fakes(requested_fit=_fit_reader):
        out = main._flat_window(cands, apparel, intents, interleave=False, ask=ask)
        assert _titles(out) == [
            "Women's Petite Work Pants",                       # all three named
            "Women's Work Pants", "Women's Petite Blazer",     # one miss each, raw order
            "Women's Blazer",                                  # two misses
            "Girls Petite Pants",                              # the wrong age, last
            "Kids Petite Backpack",                            # tier 2 (another type)
        ], _titles(out)
        # age is judged on clothing and shoes only: the kids' bag is not demoted for it
        assert main._ask_misses(cands[5], ask) == (0, 1)
        # no ask (a non-Fashion pool, or the knob off): the tiers alone, raw order
        assert _titles(main._flat_window(cands, apparel, intents, interleave=False)) == \
            _titles(cands[:5]) + ["Kids Petite Backpack"]
        # the untyped window is partitioned the same way
        out = main._flat_window(cands, None, intents, interleave=False, ask=ask)
        assert _titles(out)[0] == "Women's Petite Work Pants" and \
            _titles(out)[-1] == "Girls Petite Pants", _titles(out)
        # a non-Fashion row (typed by its title under a request) is never demoted
        school = _attr("Kids School Backpack", None, intents[0], vertical="Electronics")
        assert main._ask_misses(school, ask) == (0, 0)
        # a kids' ask turns it round
        kids = ask._replace(audience=frozenset({"kids"}), fit=None, subtypes=None)
        out = main._flat_window(cands, apparel, intents, interleave=False, ask=kids)
        assert _titles(out)[0] == "Girls Petite Pants", _titles(out)


def test_ask_is_gated_like_the_type_request_and_defaults_to_adult():
    fashion = [_attr(f"Women's Dress {i}", "apparel", "d", "adult", "dresses") for i in range(8)]
    other = [_attr(f"Phone Case {i}", None, "d", vertical="Electronics") for i in range(8)]
    readers = dict(requested_fit=lambda q: frozenset({"plus"}) if "plus size" in q else None,
                   requested_subtypes=lambda q: frozenset({"dresses"}) if "dress" in q else None,
                   requested_audience=lambda q: frozenset({"kids"}) if "girls" in q else None)
    with knobs(), c1_fakes(**readers):
        ask = main._effective_ask("plus size dress", fashion + other[:7])
        assert ask == main._Ask(frozenset({"plus"}), frozenset({"dresses"}),
                                frozenset({"adult"})), ask         # no age named: adult
        assert main._effective_ask("girls dress", fashion)[2] == frozenset({"kids"})
        # a pool the shopper's words do not lead to Fashion: nothing acts ("iPhone 14
        # Plus", "tall kitchen trash bags", "baby formula")
        assert main._effective_ask("plus size dress", other[:8] + fashion[:7]) is None
        assert main._effective_ask("plus size dress", []) is None


def test_fit_words_state_women():
    """'petite work pants' read as gender-unstated: 2x retrieval and a men's interleave
    put 6 men's rows in its 15-seat window (owner call 3: petite/maternity state women's).
    A candidate cue only: it acts in a Fashion-majority pool (next test)."""
    with c1_fakes(implies_women=lambda q: bool(_fit_reader(q))):
        for q in ("petite work pants", "maternity dress for a baby shower"):
            assert main._women_fit_stated(q, typed=True, bundle_type=None), q
            assert not main._women_fit_stated(q, typed=False, bundle_type=None), q
            assert not main._women_fit_stated(q, typed=True, bundle_type="setup"), q
        # not a pool-blind gender cue any more (2026-10-19 review); a dress still is one
        assert not main._gender_stated("petite work pants", [], typed=True)
        assert main._gender_stated("maternity dress for a baby shower", [], typed=True)
        assert not main._women_fit_stated("work pants", typed=True, bundle_type=None)
        women = [f"Women's Petite Trousers {i}" for i in range(4)]
        men = [f"Men's Work Pants {i}" for i in range(4)]
        rows = [row(t) for t in women] + [row(t, cat="Men's Clothing") for t in men]
        hits = {"petite work pants": [(t, 0.95 - 0.01 * i) for i, t in enumerate(women + men)]}
        for shape in SHAPES:
            r = run("petite work pants", rows, hits, shape=shape, intents=["petite work pants"])
            assert [s["top_k"] for s in r.searches] == [main.RETRIEVAL_TOP_K], r.searches  # 1x
            window = r.rerank.calls[0]["window"]
            assert window[:4] == [t.casefold() for t in women], window       # no interleave


def test_fit_words_outside_fashion_change_nothing():
    """2026-10-19 review: implies_women ran before the pool gate, so "petite mints",
    "pregnancy tea" or "lactation cookies for breastfeeding" (Grocery) retrieved 1x deep
    and lost half their results (144 -> 78 on "petite mints"). Now the 1x pool is checked
    and a non-Fashion one is retrieved 2x, exactly as with the fit word read as nothing."""
    cases = [("petite mints", "Grocery", "Petite Mints Tin"),
             ("pregnancy test strips", "Electronics", "Pregnancy Test Strip"),
             ("lactation cookies for breastfeeding", "Grocery", "Lactation Cookie")]
    for query, vertical, stem in cases:
        assert pt.implies_women(query), query       # the real reader fires on each
        rows = [row(f"{stem} {i} Pack", vertical=vertical) for i in range(45)] + \
            [row(f"Women's Petite Pants {i}") for i in range(3)]
        order = [r["Product_title"] for r in rows]
        hits = {query: [(t, 0.95 - 0.001 * i) for i, t in enumerate(order)]}
        for shape in SHAPES:
            r = run(query, rows, hits, shape=shape, intents=[query])
            with c1_fakes(implies_women=lambda q: False):
                off = run(query, rows, hits, shape=shape, intents=[query])
            assert [s["top_k"] for s in r.searches] == \
                [main.RETRIEVAL_TOP_K, main.RETRIEVAL_TOP_K * 2], (query, r.searches)
            assert [s["top_k"] for s in off.searches] == [main.RETRIEVAL_TOP_K * 2]
            assert r.rerank.calls[0]["window"] == off.rerank.calls[0]["window"], query
            assert r.titles == off.titles and \
                r.out["total_results"] == off.out["total_results"] == 48, query


def test_an_untyped_row_in_the_raw_head_competes_in_the_first_tier():
    """'running shoes for flat feet' (synthesis P7): a Brooks Glycerin typed 'unknown' at
    raw #14 sat behind 72 typed footwear rows, sandals and kids' shoes included."""
    intent = "running shoes for flat feet"
    cands = [_attr(f"Running Sandal {i}", "footwear", intent) for i in range(5)]
    cands.append(_attr("Brooks Glycerin 15", "unknown", intent))           # raw #6
    cands += [_attr(f"Running Sandal {i}", "footwear", intent) for i in range(5, 16)]
    cands.append(_attr("Brooks Adrenaline GTS 18", "unknown", intent))     # raw #18
    footwear = frozenset({"footwear"})
    with knobs():
        out = _titles(main._flat_window(cands, footwear, [intent], interleave=False))
    assert out.index("Brooks Glycerin 15") == 5, out                # in place, tier 0
    assert out.index("Brooks Adrenaline GTS 18") == len(out) - 1    # outside: tier 1
    # the display still ranks a typed row before an untyped one inside a band
    assert main._type_rank(cands[5], footwear) == 1


def test_off_type_intents_keep_no_window_seats():
    """'funeral attire for men' (synthesis P4): the 'black leather dress shoes' intent's
    rows typed apparel (a biker jacket, belts read as dress) took 5 of 15 seats."""
    intents = ["men's black suit", "black leather dress shoes", "white dress shirt"]
    assert main._off_type_intents(intents, frozenset({"apparel"})) == \
        frozenset({"black leather dress shoes"})
    assert main._off_type_intents(intents, frozenset({"apparel", "footwear"})) == frozenset()
    assert main._off_type_intents(BEACH_INTENTS, frozenset({"apparel"})) == \
        frozenset({"wedge sandals"})
    assert main._off_type_intents(["cozy warm desk lamp"], frozenset({"apparel"})) == frozenset()
    suit = [_attr(f"Men's Black Suit {i}", "apparel", intents[0], gender="men") for i in range(3)]
    shoes = [_attr("Men's Biker Leather Jacket", "apparel", intents[1], gender="men"),
             _attr("Men's Dress Vest", "apparel", intents[1], gender="men")]
    shirt = [_attr(f"Men's White Dress Shirt {i}", "apparel", intents[2], gender="men")
             for i in range(2)]
    cands = [shoes[0], suit[0], shirt[0], shoes[1], suit[1], shirt[1], suit[2]]
    with knobs():
        out = _titles(main._flat_window(cands, frozenset({"apparel"}), intents, False))
    assert out == _titles([suit[0], shirt[0], suit[1], shirt[1], suit[2], shoes[0], shoes[1]])


def test_display_interleaves_inside_rerank_tiers_only():
    rows = [_attr("w92", "apparel", "d", score=92), _attr("w90", "apparel", "d", score=90),
            _attr("m55", "apparel", "d", gender="men", score=55),
            _attr("m50", "apparel", "d", gender="men", score=50),
            _attr("w_tail", "apparel", "d"), _attr("w30", "apparel", "d", score=30)]
    apparel = frozenset({"apparel"})
    with knobs():
        out = main._flat_display(rows, apparel, interleave=True)
        assert _titles(out) == ["w92", "w90", "m55", "m50", "w_tail", "w30"], _titles(out)
        # equal tiers still alternate (86-100 here)
        close = [_attr("w95", "apparel", "d", score=95), _attr("w93", "apparel", "d", score=93),
                 _attr("m90", "apparel", "d", gender="men", score=90)]
        assert _titles(main._flat_display(close, apparel, True)) == ["w95", "m90", "w93"]
        groups = main._relevance_groups(rows, apparel)
        assert [groups[id(r)] for r in rows] == [(False, 0)] * 4 + [(False, 1), (False, 2)]
    with knobs(REJECTS_AFTER_TAIL=False):
        out = main._flat_display(rows, apparel, interleave=True)
        assert _titles(out) == ["w92", "m55", "w90", "m50", "w30", "w_tail"], _titles(out)
        groups = main._relevance_groups(rows, apparel)
        assert [groups[id(r)][1] for r in rows][4:] == [2, 1]


def test_guest_constraint_goes_to_the_reranker_only():
    """'cloths for a beach wedding in summer' (owner call 6): the typo kept the rubric's
    phrase-based guest rule from firing; product_type decides it now."""
    guest = c1_fakes(is_wedding_guest_ask=lambda q: q == BEACH_QUERY)
    with guest:
        for shape in SHAPES:
            r = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                    scores=BEACH_SCORES)
            sent = r.rerank.calls[0]["constraints"]
            assert [c["type"] for c in sent] == ["product_type", "occasion"], sent
            assert sent[1] == main._GUEST_CONSTRAINT and BEACH_QUERY not in sent[1]["value"]
            assert "(bridal gowns, wedding dresses, bridesmaid dresses)" in sent[1]["value"]
            assert r.out["constraints"] == []                # the response is the translator's
            off = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, shape=shape, intents=BEACH_INTENTS,
                      scores=BEACH_SCORES, typed=False)
            assert off.rerank.calls[0]["constraints"] == []
        # not a guest (the fake says so for every other query): nothing added
        r = run("wedding dress", BEACH_ROWS, BEACH_HITS, shape="local", intents=BEACH_INTENTS,
                scores=BEACH_SCORES)
        assert all(c["type"] != "occasion" for c in r.rerank.calls[0]["constraints"])
    # a pool that is not Fashion-majority never gets it
    query, vertical, own, fashion = COLLISIONS[1]                # "ring light"
    rows, hits = _collision_pool(query, own, fashion, vertical)
    with c1_fakes(is_wedding_guest_ask=lambda q: True):
        r = run("ring light", rows, hits, shape="local", intents=["ring light"])
        assert r.rerank.calls[0]["constraints"] == []


SWIM_GUEST_ASKS = ["swimsuit for a beach wedding", "cover up for a beach wedding",
                   "swim trunks for a destination wedding", "wedding guest swimsuit",
                   "bikini for a wedding party", "beach cover up for a wedding",
                   "one piece swimsuit for a wedding weekend", "swim suit for a beach wedding",
                   "bikini to wear to a beach wedding", "beach cover up for a wedding guest",
                   "bathing suit to wear to a party",
                   "one piece swimsuit for a destination wedding"]


def test_guest_constraint_keeps_the_swimwear_the_shopper_asked_for():
    """2026-10-19 review: on "swimsuit for a beach wedding" the reranker was told, next to
    [product_type] clothing (swimwear), that swimwear violates the [occasion] constraint,
    so every row of the requested style was a stated violator (capped at 40). The real
    product_type readers decide it here, no fakes."""
    for q in SWIM_GUEST_ASKS:
        assert pt.is_wedding_guest_ask(q), q        # still a guest: bridal wear violates
        sent = main._guest_constraint(q)
        assert sent == main._GUEST_SWIM_CONSTRAINT, q
        assert "swimwear and cover-ups violate" not in sent["value"]
        assert "(bridal gowns, wedding dresses, bridesmaid dresses) violates it" in sent["value"]
    for q in (BEACH_QUERY, "wedding guest dress", "beach wedding outfit",
              "dress for a beach wedding, not a swimsuit"):
        assert main._guest_constraint(q) == main._GUEST_CONSTRAINT, q
    for q in ("swimsuit for a pool party", "bridal swimsuit", "swimsuit"):
        assert main._guest_constraint(q) is None, q
    # a guest who names only accessories or shoes asks for no clothing: no [occasion]
    # constraint at all (2026-10-19 review: they got the full guest text)
    for q in ("earrings for a wedding guest", "wedding guest clutch", "wedding guest shoes",
              "jewelry for a wedding guest", "wedding guest heels"):
        assert main._guest_constraint(q) is None, q
    # end to end: what the reranker is sent for a Fashion swim pool
    swim = [row(f"Women's One Piece Swimsuit {i}") for i in range(8)] + \
        [row(f"Women's Floral Beach Cover Up {i}") for i in range(4)] + \
        [row(f"Women's Linen Maxi Dress {i}") for i in range(4)]
    order = [r["Product_title"] for r in swim]
    q = SWIM_GUEST_ASKS[0]
    hits = {q: [(t, 0.95 - 0.001 * i) for i, t in enumerate(order)]}
    for shape in SHAPES:
        r = run(q, swim, hits, shape=shape, intents=[q])
        sent = r.rerank.calls[0]["constraints"]
        assert [c["type"] for c in sent] == ["product_type", "occasion"], sent
        assert sent[1] == main._GUEST_SWIM_CONSTRAINT, sent
    # and the reranker renders it whole (nothing screened out of the constraints block)
    import app.reranker as rr
    prompts = []

    def fake(prompt, temperature=None):
        prompts.append(prompt)
        return {"ranked": [{"idx": 0, "score": 90, "reason": "fits"}]}

    with patched(rr, {"generate_json": fake}):
        rr.rerank(q, [{"Product_title": "Women's One Piece Swimsuit", pt.TYPE_COLUMN: "apparel"}],
                  top_k=5, errors=[], constraints=[main._GUEST_SWIM_CONSTRAINT])
    assert "- [occasion] " + main._GUEST_SWIM_CONSTRAINT["value"] in prompts[0]


def test_band_order_knob_is_in_the_cache_key_and_collapse_is_not():
    common = dict(shape="local", intents=BEACH_INTENTS, scores=BEACH_SCORES, clear=False)
    page_cache.clear()
    try:
        run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, **common)
        run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, knob_values={"REJECTS_AFTER_TAIL": False},
            **common)
        again = run(BEACH_QUERY, BEACH_ROWS, BEACH_HITS, knob_values={"VARIANT_COLLAPSE": True},
                    **common)
        keys = list(page_cache._store)
    finally:
        page_cache.clear()
    assert len(keys) == 2 and again.rerank.calls == []   # collapse: post-cache, 0 LLM calls
    # (..., guard, REJECTS_AFTER_TAIL, DEVICE_ROLE_RANKING, OUTFIT_BUNDLES_ENABLED,
    # TYPE_AWARE_RANKING); the LRU moved the re-read entry last
    for tail in (", True, True, True, False, True))", ", True, False, True, False, True))"):
        assert sum(k.endswith(tail) for k in keys) == 1, (tail, keys)


def test_reranker_rows_carry_subtype_and_audience():
    import app.reranker as rr
    base = {"Product_title": "Girls Party Dress", pt.TYPE_COLUMN: "apparel"}
    assert rr._type_field(dict(base, fine_subtype="dresses", audience="kids")) == \
        " | type=apparel | subtype=dresses | audience=kids"
    assert rr._type_field(dict(base, fine_subtype="unknown", audience="Adult")) == \
        " | type=apparel | audience=adult"
    for subtype, audience in ((None, None), (float("nan"), float("nan")), ("", "teen"),
                              ("  ", "")):
        assert rr._type_field(dict(base, fine_subtype=subtype, audience=audience)) == \
            " | type=apparel"
    # never on an untyped or non-Fashion row, whatever its columns say
    for ptype_ in ("unknown", None, float("nan")):
        assert rr._type_field({pt.TYPE_COLUMN: ptype_, "fine_subtype": "dresses",
                               "audience": "kids"}) == ""
    rubric = " ".join(rr.RERANK_PROMPT_TEMPLATE.split())
    for phrase in ('A "Unisex" category means the gender is UNSPECIFIED',
                   "when the title names a fabric, the title wins",
                   "AGE, for clothing and shoes only",
                   "Novelty, slogan and licensed-character prints",
                   "neither satisfies nor violates a budget",
                   "`type` and `subtype` fields",
                   # pinned by tests/test_routing.py: still there
                   "reliable for GENDER only", "only when the shopper is plainly a guest",
                   "never cap bridal wear", '"wedding costume"', "bridesmaid"):
        assert phrase in rubric, phrase
    assert "json" in rr.RERANK_PROMPT_TEMPLATE.lower()
    # renders end to end (a stray brace would 500 every /search)
    prompts = []

    def fake(prompt, temperature=None):
        prompts.append(prompt)
        return {"ranked": [{"idx": 0, "score": 90, "reason": "fits"}]}

    with patched(rr, {"generate_json": fake}):
        out = rr.rerank("party dress for my daughter",
                        [dict(base, fine_subtype="dresses", audience="kids")], top_k=5,
                        errors=[], constraints=[main._GUEST_CONSTRAINT])
    assert out[0]["rerank_score"] == 90
    assert "| type=apparel | subtype=dresses | audience=kids |" in prompts[0]
    assert "     - [occasion] the shopper is a GUEST" in prompts[0]


# ------------------------------------------------------------------ 2026-10-19 review fixes
def test_a_clear_childrens_head_assumes_no_age():
    """'first communion dress', 'christening gown', 'school uniform' name no age, but the
    raw head was 14-15 of 15 kids' rows: the default adult audience sent every one of
    them out of the window behind adult club dresses (first kids' row #35-#52). A clear
    children's head (>= 2x the adult rows, >= a third of the head) now assumes no age;
    a mixed head ('running shoes': 8 kids' / 7 adult rows) stays adult."""
    kids = [_attr(f"Girls Communion Dress {i}", "apparel", "d", "kids", "dresses") for i in range(10)]
    adults = [_attr(f"Women's Club Dress {i}", "apparel", "d", "adult", "dresses") for i in range(5)]
    readers = dict(requested_fit=lambda q: None, requested_subtypes=lambda q: None,
                   requested_audience=lambda q: None)
    apparel = frozenset({"apparel"})
    with knobs(), c1_fakes(**readers):
        head = adults[:3] + kids + adults[3:]                      # raw order, 10 / 5
        ask = main._effective_ask("first communion dress", head)
        assert ask is not None and ask.audience is None, ask
        out = main._flat_window(head, apparel, ["d"], interleave=False, ask=ask)
        assert _titles(out) == _titles(head)                       # nobody demoted
        assert main._ask_misses(kids[0], ask) == (0, 0)
        # the old reading, for contrast: every kids' row behind every adult one
        old = main._flat_window(head, apparel, ["d"], interleave=False,
                                ask=ask._replace(audience=main._DEFAULT_AUDIENCE))
        assert _titles(old)[:5] == _titles(adults), _titles(old)
        # mixed heads stay adult: 8 / 7, and 9 kids' rows that are no third of a 30-row head
        shoes = kids[:8] + adults + [_attr(f"Men's Shoe {i}", "footwear", "d", "adult")
                                     for i in range(2)]
        assert main._effective_ask("running shoes", shoes).audience == frozenset({"adult"})
        sparse = kids[:4] + [_attr(f"Phone Case {i}", "accessory", "d", "adult") for i in range(11)]
        assert main._default_audience(sparse) == main._DEFAULT_AUDIENCE    # 4 < 15 / 3
        # 9 kids' / 6 adult: a kids' majority, but not twice the adult rows
        assert main._default_audience(kids[:9] + adults + adults[:1]) == main._DEFAULT_AUDIENCE
        # a stated age wins either way
    with knobs(), c1_fakes(requested_fit=lambda q: None, requested_subtypes=lambda q: None,
                           requested_audience=lambda q: frozenset({"adult"})):
        assert main._effective_ask("women's dress", head).audience == frozenset({"adult"})


def test_childrens_occasion_asks_seat_kids_rows_first():
    """The real readers, end to end: a child's occasion, a grade or 'N yo' reads kids, so
    the kids' rows lead the window although adult dresses score higher raw."""
    adults = [f"Women's Sexy Bodycon Club Dress {i}" for i in range(15)]
    kids = [f"Girls White Tulle Communion Dress Size 8 {i}" for i in range(6)]
    rows = [row(t) for t in adults] + [row(t, cat="Girls", gender="kids") for t in kids]
    for query in ("first communion dress", "communion dress for my 8 yo"):
        hits = {query: adults + kids}
        for shape in SHAPES:
            r = run(query, rows, hits, shape=shape, intents=[query])
            window = r.rerank.calls[0]["window"]
            assert window[:6] == [t.casefold() for t in kids], (query, window)


def test_black_tie_dress_code_seats_formal_garments_first():
    """'black tie event men' (2026-10-19 review): an apparel request alone let the jacket
    intent seat windbreakers and hi-vis jackets in a third of the window. The dress code
    now names formal subtypes, so tuxedos and trousers come first in the tier."""
    wind = [f"Men's Black Hooded Windbreaker Jacket {i}" for i in range(6)]
    tux = [f"Men's Black Slim Fit Tuxedo Jacket {i}" for i in range(4)]
    trousers = [f"Men's Black Dress Trousers {i}" for i in range(8)]
    rows = [row(t, cat="Men's Clothing") for t in wind + tux + trousers]
    intents = ["black tuxedo jacket", "black tuxedo trousers"]
    hits = {intents[0]: wind + tux, intents[1]: trousers}
    formal = {t.casefold() for t in tux + trousers}
    query = "black tie event men"
    assert pt.requested_subtypes(query) == pt._MEN_FORMAL_SUBTYPES
    for shape in SHAPES:
        r = run(query, rows, hits, shape=shape, intents=intents)
        window = r.rerank.calls[0]["window"]
        assert set(window[:12]) == formal and len(window) == 15, window
        sent = [c for c in r.rerank.calls[0]["constraints"] if c["type"] == "product_type"]
        assert sent == [main._type_constraint(frozenset({"apparel"}))]
        # without the dress-code reading: the windbreakers take every other seat
        with c1_fakes(requested_subtypes=lambda q: None):
            old = run(query, rows, hits, shape=shape, intents=intents)
        assert sum(t in formal for t in old.rerank.calls[0]["window"]) == 9, old.rerank.calls


# ------------------------------------------------------------------ 2026-10-27: device asks
# DEVICE_ROLE_RANKING (app/device_role.py). "gift for a vintage watch collector" in
# miniature, with the recorded failure's intents: three ACCESSORY intents and no watch
# intent (qwen3.8-27b, 2026-10-27), but gte-small's pool still holds leather-strap watches
# under "leather watch winder" and "vintage watch repair kit" (the live pool held 49).
GIFT_QUERY = "gift for a vintage watch collector"
GIFT_INTENTS = ["leather watch winder", "vintage watch repair kit", "vintage watch display case"]
GIFT_MEN = [f"Men's Vintage Automatic Mechanical Watch Edition {i}" for i in range(7)]
GIFT_WOMEN = [f"Women's Retro Quartz Analog Watch Edition {i}" for i in range(7)]
GIFT_STORAGE = ["Automatic Watch Winder for Single Watch", "Double Watch Winder Wood Finish",
                "Watch Box for Men 12 Slots Leather Watch Case Organizer",
                "Wooden Watch Display Case for 10 Watches"]
GIFT_CARE = ["SOONAN DIY Watch Repair Tool Kit",
             "Watch Polishing Cloth for Stainless Steel Watches"]
GIFT_BAND = "Watch Band Strap Quick Release Leather Replacement"
GIFT_CLEANER = "Screen Cleaner Kit, Laptop Computer Monitor Cleaner"   # Electronics tools


def _gift_rows():
    rows = [dict(row(t, cat="Men's Clothing"), price=89.0) for t in GIFT_MEN]
    rows += [dict(row(t), price=79.0) for t in GIFT_WOMEN]
    rows += [dict(row(t, cat="Unisex Clothing"), price=12.0)
             for t in GIFT_STORAGE + GIFT_CARE + [GIFT_BAND]]
    rows.append(dict(row(GIFT_CLEANER, vertical="Electronics"), price=9.0))
    return rows


GIFT_HITS = {
    "leather watch winder": [(t, 0.93 - 0.001 * i) for i, t in enumerate(GIFT_STORAGE[:2])]
    + [(t, 0.88 - 0.001 * i) for i, t in enumerate(GIFT_MEN)],
    "vintage watch repair kit": [(t, 0.92 - 0.001 * i) for i, t in
                                 enumerate(GIFT_CARE + [GIFT_CLEANER, GIFT_BAND])]
    + [(t, 0.87 - 0.001 * i) for i, t in enumerate(GIFT_WOMEN)],
    "vintage watch display case": [(t, 0.91 - 0.001 * i) for i, t in enumerate(GIFT_STORAGE[2:])],
}
# What the recorded reranker did with winders in its window (88 / 85, boxes 72, kits
# 55-60) and what it might do with a watch it reads as "not an accessory" (70)
GIFT_SCORES = {**{t: 70 - i for i, t in enumerate(GIFT_MEN + GIFT_WOMEN)},
               GIFT_STORAGE[0]: 88, GIFT_STORAGE[1]: 85, GIFT_STORAGE[2]: 72,
               GIFT_STORAGE[3]: 71, GIFT_CARE[0]: 60, GIFT_CARE[1]: 55, GIFT_BAND: 58,
               GIFT_CLEANER: 57}
_WATCHES = {t.casefold() for t in GIFT_MEN + GIFT_WOMEN}


def _gift_run(shape, **kw):
    kw.setdefault("scores", GIFT_SCORES)
    return run(GIFT_QUERY, _gift_rows(), GIFT_HITS, shape=shape, intents=GIFT_INTENTS, **kw)


def test_device_gift_window_seats_the_watches_and_reserves_three_addon_seats():
    gift = dr.DeviceAsk("watch", True, False)
    assert dr.requested_device(GIFT_QUERY) == gift
    for shape in SHAPES:
        on = _gift_run(shape)
        off = _gift_run(shape, knob_values={"DEVICE_ROLE_RANKING": False})
        window, before = on.rerank.calls[0]["window"], off.rerank.calls[0]["window"]
        # Before: one seat per intent then raw cosine -- all 8 add-ons lead the window
        assert not set(before[:8]) & _WATCHES and len(before) == 15, before
        # After: 12 watches, then the 3 reserved add-on seats, storage and display before
        # care, one per add-on intent in the translator's order
        assert set(window[:12]) <= _WATCHES and len(window) == 15, window
        assert window[12:] == [GIFT_STORAGE[0].casefold(), GIFT_STORAGE[2].casefold(),
                               GIFT_BAND.casefold()], window[12:]
        # the core tier keeps the gender interleave (no gender stated)
        genders = ["men" if t.startswith("men's") else "women" for t in window[:12]]
        assert genders[:4] in (["men", "women"] * 2, ["women", "men"] * 2), genders
        # the reranker gets the gift text; the response keeps the translator's (none)
        assert on.rerank.calls[0]["constraints"] == [dr.constraint(gift)]
        assert off.rerank.calls[0]["constraints"] == []
        assert on.out["constraints"] == [] and GIFT_QUERY not in json.dumps(
            on.rerank.calls[0]["constraints"])


def test_device_role_field_rides_only_on_window_copies():
    import app.reranker as rr
    for shape in SHAPES:
        on = _gift_run(shape)
        rows = on.rerank.calls[0]["window_rows"]
        roles = {r["Product_title"].casefold(): r.get(rr.ROLE_FIELD_KEY) for r in rows}
        assert all(roles[t] == "device" for t in roles if t in _WATCHES), roles
        assert all(roles[t] == "accessory" for t in roles if t not in _WATCHES), roles
        # never cached, never served
        entry = run.last_entries[0]
        assert not any(rr.ROLE_FIELD_KEY in r for r in entry["full"])
        assert not any(rr.ROLE_FIELD_KEY in r for r in on.results)
        assert entry["device"] == dr.DeviceAsk("watch", True, False)
        line = rr._format_candidate(0, rows[0])
        assert "| type=accessory | audience=adult | role=device | color=" in line, line
        off = _gift_run(shape, knob_values={"DEVICE_ROLE_RANKING": False})
        assert not any(rr.ROLE_FIELD_KEY in r for r in off.rerank.calls[0]["window_rows"])


def test_device_display_puts_the_device_first_inside_each_band():
    for shape in SHAPES:
        on = _gift_run(shape)
        off = _gift_run(shape, knob_values={"DEVICE_ROLE_RANKING": False})
        # the winder the reranker scored 88 no longer leads a page of watches scored 70
        assert off.titles[0] == GIFT_STORAGE[0].casefold(), off.titles
        assert set(on.titles[:12]) <= _WATCHES, on.titles
        # inside band 0 the judged add-ons (the 3 reserved seats) keep the reranker's
        # order and open page 2; the unjudged tail follows, its watches first
        deep = run(GIFT_QUERY, _gift_rows(), GIFT_HITS, shape=shape, intents=GIFT_INTENTS,
                   scores=GIFT_SCORES, top_k=24)
        second = deep.results[12:]
        judged = [r["rerank_score"] for r in second if r.get("rerank_score") is not None]
        assert judged == [88, 72, 58], judged
        assert [r["Product_title"].casefold() in _WATCHES for r in second[3:5]] == [True, True]


def test_device_sort_groups_keep_the_device_first():
    """A price sort reorders WITHIN (mismatch, band, not the device): the $12 winders
    stay behind every $79-89 watch, as a $12 sleeve behind every laptop."""
    for shape in SHAPES:
        on = _gift_run(shape, sort="price_asc")
        off = _gift_run(shape, sort="price_asc", knob_values={"DEVICE_ROLE_RANKING": False})
        assert set(on.titles[:12]) <= _WATCHES, on.titles
        prices = [r.get("price") for r in on.results]
        assert prices == sorted(prices), prices                 # sorted inside the group
        assert off.titles[0] not in _WATCHES, off.titles        # before: $9 cleaner kit first


def test_device_ask_refuses_an_accessory_ad():
    rows = _gift_rows()
    by_title = {r["Product_title"]: r for r in rows}
    winder_asin = by_title[GIFT_STORAGE[0]]["parent_asin"]
    watch_asin = by_title[GIFT_MEN[0]]["parent_asin"]
    smap = {winder_asin: {"sponsor": "Acme", "bid": 2.0},
            watch_asin: {"sponsor": "Acme", "bid": 1.0}}
    for shape in SHAPES:
        on = run(GIFT_QUERY, rows, GIFT_HITS, shape=shape, intents=GIFT_INTENTS,
                 scores=GIFT_SCORES, smap=smap)
        off = run(GIFT_QUERY, rows, GIFT_HITS, shape=shape, intents=GIFT_INTENTS,
                  scores=GIFT_SCORES, smap=smap, knob_values={"DEVICE_ROLE_RANKING": False})
        ads_on = [r["Product_title"] for r in on.results if r.get("is_sponsored")]
        ads_off = [r["Product_title"] for r in off.results if r.get("is_sponsored")]
        assert ads_on == [GIFT_MEN[0]] and on.titles[0] == GIFT_MEN[0].casefold(), ads_on
        assert ads_off == [GIFT_STORAGE[0], GIFT_MEN[0]], ads_off       # bid order before


IPHONE_HANDSET = "Apple iPhone 12 Mini, 64GB, Black - Unlocked (Renewed)"
IPHONE_ACC = ([f"iPhone 12 Case Clear Shockproof Style {i}" for i in range(8)]
              + [f"Fast Charger for iPhone 12 USB C Pack {i}" for i in range(6)])


def test_device_match_quality_counts_no_accessory_as_on_type():
    """"iphone": 3 handsets among 2,696 iPhone-titled rows; the reranker scored the one
    it saw 100 and the cases 40-90 by turns. A page of cases is no exact match."""
    rows = [row(t, vertical="Electronics") for t in IPHONE_ACC + [IPHONE_HANDSET]]
    hits = {"iphone smartphone": [(t, 0.95 - 0.002 * i) for i, t in
                                  enumerate(IPHONE_ACC + [IPHONE_HANDSET])]}
    scores = {**{t: 90 - 2 * i for i, t in enumerate(IPHONE_ACC)}, IPHONE_HANDSET: 100}
    for shape in SHAPES:
        on = run("iphone", rows, hits, shape=shape, intents=["iphone smartphone"], scores=scores)
        off = run("iphone", rows, hits, shape=shape, intents=["iphone smartphone"],
                  scores=scores, knob_values={"DEVICE_ROLE_RANKING": False})
        # the handset at raw #15 takes window seat 1
        assert on.rerank.calls[0]["window"][0] == IPHONE_HANDSET.casefold()
        assert off.rerank.calls[0]["window"][-1] == IPHONE_HANDSET.casefold()
        mq_on, mq_off = on.out["match_quality"], off.out["match_quality"]
        assert (mq_on["band0"], mq_on["best_rerank"], mq_on["weak"]) == (1, 100, True), mq_on
        assert (mq_off["band0"], mq_off["weak"]) == (15, False), mq_off
        assert on.rerank.calls[0]["constraints"] == [
            dr.constraint(dr.DeviceAsk("phone", False, False))]


def test_typed_watch_ask_orders_the_type_tier_by_role():
    """'mens automatic watch' is also a type request ({accessory}): the role partitions
    the inside of the accessory tier, watches before winders and bands."""
    query = "mens automatic watch"
    assert pt.requested_types(query) == frozenset({"accessory"})
    intents = ["men's automatic watch", "automatic watch winder"]
    winders = GIFT_STORAGE[:2] + [GIFT_BAND]
    rows = [row(t, cat="Men's Clothing") for t in GIFT_MEN] + \
        [row(t, cat="Unisex Clothing") for t in winders]
    hits = {intents[0]: GIFT_MEN, intents[1]: winders}
    for shape in SHAPES:
        on = run(query, rows, hits, shape=shape, intents=intents)
        off = run(query, rows, hits, shape=shape, intents=intents,
                  knob_values={"DEVICE_ROLE_RANKING": False})
        window = on.rerank.calls[0]["window"]
        assert window == [t.casefold() for t in GIFT_MEN + winders], window
        assert off.rerank.calls[0]["window"][1] == winders[0].casefold()   # round-robin
        sent = on.rerank.calls[0]["constraints"]
        assert sent == [main._type_constraint(frozenset({"accessory"})),
                        dr.constraint(dr.DeviceAsk("watch", False, False))], sent


PADS = [f"Replacement Ear Pads for Sony WH-1000XM4 Headphones Style {i}" for i in range(8)]
HEADPHONES = ["Sony WH-1000XM4 Wireless Noise Canceling Overhead Headphones",
              "Bose QuietComfort 45 Bluetooth Wireless Noise Cancelling Headphones"]


def test_device_ask_does_nothing_where_it_cannot_act():
    """No change from the knob-off page: an accessory ask, an owner need, a pool the
    family does not lead, a pool with no core row, a bundle, and an ordinary query."""
    laptops = ["Lenovo IdeaPad 3 Laptop 15.6 inch Intel Core i5 8GB RAM"]
    sleeves = [f"Laptop Sleeve 13 inch Compatible with MacBook Air Color {i}" for i in range(6)]
    totes = [f"Women's Leather Tote Bag Large Style {i}" for i in range(9)]
    # The bundle row must read a device ask over a pool the gate passes, so that only the
    # flat-pages-only guard keeps it inert: the regex routes this query to 'setup' (as the
    # QUERY_EXPANSION reply may route a device ask). Its old query, "gaming laptop setup",
    # read no ask once the reader failed closed on the tail word "setup" (2026-10-27
    # review), so device ranking on bundles passed it too (Phase D review)
    setup_query, setup_rows = "laptop for a gaming setup", [
        row(t, vertical="Electronics") for t in laptops + sleeves]
    setup_hits = {"gaming laptop": laptops, "laptop sleeve": sleeves}
    laptop_ask = dr.DeviceAsk("laptop", False, False)
    assert dr.requested_device(setup_query) == laptop_ask
    for shape in SHAPES:
        flat = run(setup_query, setup_rows, setup_hits, shape=shape, intents=list(setup_hits))
        assert flat.rerank.calls[0]["constraints"] == [dr.constraint(laptop_ask)], shape
        assert run.last_entries[0]["device"] == laptop_ask, shape
    cases = [
        # (query, rows, hits, intents, bundle, flag logged)
        ("watch winder", _gift_rows(), GIFT_HITS, GIFT_INTENTS, None, None),
        ("stop my laptop from overheating",
         [row(t, vertical="Electronics") for t in laptops + sleeves],
         {"laptop cooling pad": laptops + sleeves}, ["laptop cooling pad"], None, None),
        ("laptop for college",
         [row(t, vertical="Electronics") for t in laptops + sleeves] + [row(t) for t in totes],
         {"laptop tote": totes + laptops + sleeves[:2]}, ["laptop tote"], None, "minority_pool"),
        ("laptop for college", [row(t, vertical="Electronics") for t in sleeves],
         {"laptop sleeve": sleeves}, ["laptop sleeve"], None, "no_core_rows"),
        (setup_query, setup_rows, setup_hits, list(setup_hits), "setup", None),
        (BEACH_QUERY, BEACH_ROWS, BEACH_HITS, BEACH_INTENTS, None, None),
        # 2026-10-27 review: an add-on compound the accessory list does not know ("pads")
        # in a pool the gate WOULD pass (every row names headphones, two are the device):
        # the reader fails closed, so nothing moves
        ("headphone pads", [row(t, vertical="Electronics") for t in PADS + HEADPHONES],
         {"headphone ear pads": PADS + HEADPHONES}, ["headphone ear pads"], None, None),
        ("phone or tablet stand", [row(t, vertical="Electronics") for t in IPHONE_ACC +
                                   [IPHONE_HANDSET]],
         {"phone tablet stand": IPHONE_ACC + [IPHONE_HANDSET]}, ["phone tablet stand"],
         None, None),
    ]
    for query, rows, hits, intents, bundle, flag in cases:
        for shape in SHAPES:
            with captured_logs("app.main") as logs:
                on = run(query, rows, hits, shape=shape, intents=intents, bundle_type=bundle)
            # no ask kept, so no sort, match quality or ad gate reads one either
            assert run.last_entries[0]["device"] is None, (query, shape)
            off = run(query, rows, hits, shape=shape, intents=intents, bundle_type=bundle,
                      knob_values={"DEVICE_ROLE_RANKING": False})
            assert on.rerank.calls[0]["window"] == off.rerank.calls[0]["window"], query
            assert on.rerank.calls[0]["constraints"] == off.rerank.calls[0]["constraints"]
            assert on.titles == off.titles, (query, shape)
            assert on.out["match_quality"] == off.out["match_quality"], query
            assert not any("role=" in json.dumps(r, default=str) or "_device_role" in r
                           for r in on.rerank.calls[0]["window_rows"]), query
            if flag:
                assert any(flag in m for m in logs), (query, logs)
            # the gate logs a flag, never the query
            assert not any(query in m for m in logs), logs


def test_device_knob_is_in_the_cache_key_after_the_band_order_knob():
    common = dict(shape="local", intents=GIFT_INTENTS, scores=GIFT_SCORES, clear=False)
    page_cache.clear()
    try:
        run(GIFT_QUERY, _gift_rows(), GIFT_HITS, **common)
        flipped = run(GIFT_QUERY, _gift_rows(), GIFT_HITS,
                      knob_values={"DEVICE_ROLE_RANKING": False}, **common)
        keys = list(page_cache._store)
    finally:
        page_cache.clear()
    assert len(keys) == 2 and len(flipped.rerank.calls) == 1   # a real second pipeline run
    # (..., guard, REJECTS_AFTER_TAIL, DEVICE_ROLE_RANKING, OUTFIT_BUNDLES_ENABLED, typed)
    assert keys[0].endswith(", True, True, True, False, True))"), keys
    assert keys[1].endswith(", True, True, False, False, True))"), keys


def test_device_ranking_lives_inside_type_aware_ranking():
    """TYPE_AWARE_RANKING=false is still the frozen pre-change ordering: no device tier,
    no gift reserve, no device constraint, no role field."""
    for shape in SHAPES:
        off = _gift_run(shape, typed=False)
        also = _gift_run(shape, typed=False, knob_values={"DEVICE_ROLE_RANKING": False})
        assert off.rerank.calls[0]["constraints"] == []
        assert not any("_device_role" in r for r in off.rerank.calls[0]["window_rows"])
        assert off.rerank.calls[0]["window"] == also.rerank.calls[0]["window"]
        assert off.titles == also.titles and off.out["match_quality"] == also.out["match_quality"]
        with knobs(), served(_gift_rows(), GIFT_HITS, shape):
            raw = S.search_products(GIFT_INTENTS, top_k=main.RETRIEVAL_TOP_K * 2)
            window = main._balance_candidate_pool(raw)
        assert off.rerank.calls[0]["window"] == [c["Product_title"].casefold()
                                                 for c in window][:15]


def _identity_cands():
    def t(title, ptype_, intent, gender, score, audience="adult"):
        # every device column set: a column alone must never act without an ask
        return dict(_typed(title, ptype_, intent, gender, score), audience=audience,
                    watch_kind="watch", item_category="watches", item_role="device")
    return [t("w1", "apparel", "a", "women", 95), t("w2", "apparel", "a", "women", 85),
            t("m1", "apparel", "b", "men", 80), t("k1", "apparel", "b", "women", 75, "kids"),
            t("w3", "apparel", "a", "women", None), t("m2", "footwear", "b", "men", None),
            t("w4", "jewelry", "a", "women", 30), t("m3", "apparel", "b", "men", 20)]


def test_device_helpers_are_identity_without_a_device_ask():
    """With no device ask the window, display, sort groups and match quality are what
    they were before 2026-10-27. The expected values are the 10-26 snapshot's own outputs
    on this fixture (build/b1_r5/identity_probe.py ran it through both trees: equal), so
    a key element the device code adds without guarding it shows up here."""
    names = lambda rows: [r["Product_title"] for r in rows]   # noqa: E731
    cands = _identity_cands()
    ask = main._Ask(fit=None, subtypes=None, audience=frozenset({"adult"}))
    apparel = frozenset({"apparel"})
    with knobs():
        assert names(main._partitioned(cands, ask, True)) == \
            ["w1", "m1", "w2", "m2", "w3", "m3", "w4", "k1"]
        assert names(main._partitioned(cands, None, True)) == \
            ["w1", "m1", "w2", "m2", "k1", "m3", "w3", "w4"]
        assert names(main._flat_window(cands, None, ["a", "b"], True, ask=ask)) == \
            ["w1", "m1", "w2", "m2", "w3", "m3", "w4", "k1"]
        assert names(main._flat_window(cands, apparel, ["a", "b"], True, ask=ask)) == \
            ["w1", "m1", "w2", "m3", "w3", "k1", "w4", "m2"]
        assert main._match_quality(cands, apparel) == {
            "band0": 4, "strong": 4, "best_rerank": 95.0, "weak": False, "unjudged": False}
    expected = {
        True: (["w1", "w2", "m1", "k1", "w3", "m2", "w4", "m3"],
               ["w1", "w2", "m1", "k1", "w3", "m3", "m2", "w4"],
               [(False, 0)] * 4 + [(False, 1), (True, 1), (True, 2), (False, 2)]),
        False: (["w1", "m1", "w2", "k1", "w4", "m3", "w3", "m2"],
                ["w1", "m1", "w2", "k1", "m3", "w3", "w4", "m2"],
                [(False, 0)] * 4 + [(False, 2), (True, 2), (True, 1), (False, 1)]),
    }
    for rejects_after_tail, (display, typed_display, groups) in expected.items():
        with knobs(REJECTS_AFTER_TAIL=rejects_after_tail):
            assert names(main._flat_display(cands, None, True)) == display
            assert names(main._flat_display(cands, apparel, True)) == typed_display
            got = main._relevance_groups(cands, apparel)
            assert [got[id(c)] for c in cands] == groups, rejects_after_tail


def test_device_display_keeps_a_gendered_addon_behind_every_device():
    """The display groups by (mismatch, band, NOT core) BEFORE it interleaves genders:
    grouped by (mismatch, band) only, the interleave lifted a men's-category watch box
    scored 92 above three women's watches scored 93-95 (no gender stated), under either
    band order (2026-10-27 review: a mutant of that grouping passed every test)."""
    query = "vintage watch"
    watches = [f"Women's Retro Quartz Analog Watch Edition {i}" for i in range(3)]
    box = "Men's Leather Watch Box 6 Slots"
    rows = [row(t) for t in watches] + [row(box, cat="Men's Clothing")]
    hits = {"vintage wristwatch": watches, "vintage watch box": [box]}
    scores = {watches[0]: 95, watches[1]: 94, watches[2]: 93, box: 92}
    device = dr.DeviceAsk("watch", False, False)
    assert dr.requested_device(query) == device
    for rejects_after_tail in (True, False):
        for shape in SHAPES:
            with knobs(REJECTS_AFTER_TAIL=rejects_after_tail):
                annotated = as_rows(rows, shape)
            for r in annotated:
                r["rerank_score"] = scores[r["Product_title"]]
            with knobs(REJECTS_AFTER_TAIL=rejects_after_tail):
                shown = main._flat_display(annotated, None, True, device)
            assert [r["Product_title"] for r in shown][-1] == box, (rejects_after_tail, shape)
            assert main._product_gender(shown[-1]) == "men"     # the interleave would lift it
            on = run(query, rows, hits, shape=shape, intents=list(hits), scores=scores,
                     knob_values={"REJECTS_AFTER_TAIL": rejects_after_tail})
            assert on.titles == [t.casefold() for t in watches + [box]], on.titles


IPAD_IPOD_MAC = ["Apple iPad Air 2, 16GB, Space Gray (Renewed)",
                 "Apple iPad Mini 4 128GB Wi-Fi Gold",
                 "Apple iPod Touch 32GB Blue (6th Generation)",
                 "Apple iPod touch 16GB (5th Generation) Silver", "Apple iPod Nano 8GB Silver",
                 "Apple MacBook Air MD711LL/A 11.6-Inch Laptop",
                 "Apple MacBook Pro 13-inch 8GB RAM 256GB SSD"]
IPHONES = ["Apple iPhone 12 Mini, 64GB, Black - Unlocked (Renewed)",
           "Apple iPhone 7, 32GB, Black - Unlocked Smartphone (Renewed)",
           "Apple iPhone X, 64GB, Silver - Fully Unlocked (Renewed)"]


def test_device_gate_leaves_other_devices_out_of_the_share():
    """"iphone" with the device-only intents the DEVICE bullet asks for retrieves Apple's
    other devices too (the 2026-10-27 review, real catalog: 7 iPads, iPods and MacBooks in
    a 15-row head): 7 of 15 head rows named a phone and the gate dropped the ask. A device
    of another family is evidence of nothing, so it leaves the share (7 of 8 here)."""
    query, intents = "iphone", ["apple iphone unlocked", "apple iphone renewed",
                                "iphone smartphone"]
    accessories = [f"iPhone 12 Case Clear Shockproof Style {i}" for i in range(4)]
    shirt = "Apple Logo Graphic Print T-Shirt"
    rows = [row(t, vertical="Electronics") for t in IPAD_IPOD_MAC + accessories + IPHONES]
    rows.append(row(shirt))
    ordered = IPAD_IPOD_MAC + [shirt] + accessories + IPHONES
    hits = {intents[0]: [(t, 0.97 - 0.002 * i) for i, t in enumerate(ordered)]}
    scores = {**{t: 95 for t in IPHONES}, **{t: 60 for t in accessories},
              **{t: 30 for t in IPAD_IPOD_MAC + [shirt]}}
    for shape in SHAPES:
        head = as_rows(rows, shape)
        named = sum(dr.mentions(r, "phone") for r in head)
        assert (named, sum(r.get("item_role") == "device" and not dr.mentions(r, "phone")
                           for r in head)) == (7, 7)      # 7 of 15: the old gate's minority
        with captured_logs("app.main") as logs:
            on = run(query, rows, hits, shape=shape, intents=intents, scores=scores)
        assert not any("minority_pool" in m for m in logs), logs
        assert on.rerank.calls[0]["constraints"] == [
            dr.constraint(dr.DeviceAsk("phone", False, False))]
        assert set(on.rerank.calls[0]["window"][:3]) == {t.casefold() for t in IPHONES}
        assert set(on.titles[:3]) == {t.casefold() for t in IPHONES}


def test_device_gate_counts_the_raw_head_at_half():
    """The share is >= 0.5 of the RAW RERANK_INPUT_K head: 8 watch rows of 15 act, 7 do
    not, and watch rows deeper in the pool never count (a head of belts with 20 watches
    behind it is a belt pool)."""
    query, intents = "vintage watch", ["vintage watch"]
    watch = "Men's Vintage Automatic Mechanical Watch Edition 0"
    winders = [f"Automatic Watch Winder for Single Watch Finish {i}" for i in range(30)]
    belts = [f"Men's Leather Belt Classic Buckle Style {i}" for i in range(15)]
    cases = [
        # (head: watch rows, belts; then deep watch rows) -> acts
        ([watch] + winders[:7], belts[:7], [], True),
        ([watch] + winders[:6], belts[:8], [], False),
        ([watch] + winders[:3], belts[:11], winders[3:23], False),
    ]
    for head_watch, head_belts, deep, acts in cases:
        ordered = head_watch + head_belts + deep
        rows = [row(t, cat="Men's Clothing") for t in ordered]
        hits = {intents[0]: ordered}
        for shape in SHAPES:
            with captured_logs("app.main") as logs:
                on = run(query, rows, hits, shape=shape, intents=intents)
            sent = on.rerank.calls[0]["constraints"]
            assert (dr.constraint(dr.DeviceAsk("watch", False, False)) in sent) is acts, (
                len(head_watch), len(deep), sent)
            assert any("minority_pool" in m for m in logs) is (not acts), logs


def test_device_ask_orders_the_other_types_tier_too():
    """Under a type request the device role also orders the other-types tier (tier 2):
    an Electronics fitness tracker -- a core watch the title rules do not type -- before
    the Fitbit charging cable the raw order put first."""
    query, intents = "mens automatic watch", ["men's automatic watch", "fitbit charge"]
    watches = GIFT_MEN[:3]
    cable, tracker = "USB Charging Cable for Fitbit Charge 5", "Fitbit Charge 5 Fitness Tracker"
    rows = [row(t, cat="Men's Clothing") for t in watches] + \
        [row(t, vertical="Electronics") for t in (cable, tracker)]
    hits = {intents[0]: watches, intents[1]: [cable, tracker]}
    for shape in SHAPES:
        on = run(query, rows, hits, shape=shape, intents=intents)
        off = run(query, rows, hits, shape=shape, intents=intents,
                  knob_values={"DEVICE_ROLE_RANKING": False})
        assert on.rerank.calls[0]["window"][3:] == [tracker.casefold(), cable.casefold()]
        assert off.rerank.calls[0]["window"][3:] == [cable.casefold(), tracker.casefold()]


def test_a_misread_subtype_cannot_seat_a_band_before_the_watches():
    """"dress watch for men" reads the dresses subtype (a product_type misread, A1
    Request): every Fashion watch misses it, while a non-Fashion row was judged on
    nothing, so an Electronics replacement strap led the window ahead of 14 watches.
    Under a device ask a named subtype is judged on every row, and the role decides. The
    misread is pinned with a fake reader, so the test keeps checking main.py's rule once
    product_type reads "dress watch" right (any subtype no row carries does the same)."""
    query = "dress watch for men"
    intents = ["men's dress watch", "leather watch band"]
    watches = [f"Men's Classic Dress Watch Leather Strap Quartz Analog Model {i}"
               for i in range(12)]
    band = "GELISHI Compatible Bands for Fitbit Versa Watch Genuine Leather Replacement Strap"
    winders = GIFT_STORAGE[:2]
    rows = [row(t, cat="Men's Clothing") for t in watches + winders] + \
        [row(band, vertical="Electronics")]
    hits = {intents[0]: [(t, 0.95 - 0.002 * i) for i, t in enumerate(watches)],
            intents[1]: [(band, 0.97)] + [(t, 0.90) for t in winders]}
    misread = {"requested_subtypes": lambda q: frozenset({"dresses"})}
    for shape in SHAPES:
        with c1_fakes(**misread):
            on = run(query, rows, hits, shape=shape, intents=intents)
            off = run(query, rows, hits, shape=shape, intents=intents,
                      knob_values={"DEVICE_ROLE_RANKING": False})
        window = on.rerank.calls[0]["window"]
        assert set(window[:12]) == {t.casefold() for t in watches}, window
        assert window.index(band.casefold()) >= 12
        assert off.rerank.calls[0]["window"][0] == band.casefold()   # the misread, knob off
        sent = on.rerank.calls[0]["constraints"]
        assert sent[-1] == dr.constraint(dr.DeviceAsk("watch", False, False)), sent


def test_device_knob_is_read_once_per_request():
    """search() reads DEVICE_ROLE_RANKING once, for the cache key AND the pipeline: a knob
    flipped while the pipeline runs (an eval, a notebook) cannot store a pool built one
    way under the other mode's key."""
    real, flipped = S.search_products, []

    def flipping(*args, **kwargs):
        if not flipped:
            flipped.append(True)
            cfg.DEVICE_ROLE_RANKING = False      # mid-request; knobs() restores it after
        return real(*args, **kwargs)

    page_cache.clear()
    try:
        with patched(S, {"search_products": flipping}):
            on = _gift_run("local", clear=False)
        keys = list(page_cache._store)
    finally:
        page_cache.clear()
    assert flipped and len(keys) == 1
    assert keys[0].endswith(", True, True, True, False, True))"), keys
    assert on.rerank.calls[0]["constraints"] == [dr.constraint(dr.DeviceAsk("watch", True,
                                                                            False))]


# The recorded 2026-10-15 live capture of the meeting's query (qwen3.8-27b, the "on" arm of
# oct15/xvertical/pt_capture_xvertical.json): the 15-row window main.py sent, in the local
# row shape of that day (lowercased), with each row's raw cosine and its rerank score (None:
# returned but not on page 1, so not recorded). Intents: three accessory intents, no watch.
CAPTURE_INTENTS = ["vintage watch repair kit", "leather watch winder", "vintage watch display case"]
CAPTURE_1015 = [
    ('electronics', 'computers', 'vintage watch repair kit', 'soonan diy watch repair tool kit, professional 168 in 1 watches band strap link pins battery remover replacement adjustment repair tools kits with carrying case & manual', 0.8995, 65),
    ('fashion', 'unisex clothing', 'leather watch winder', 'deluxe watch winder for single automatic watch with built in ic timer battery or ac operation black finish by tech swiss', 0.9005, 75),
    ('fashion', 'unisex clothing', 'vintage watch display case', 'watch storage case for 10 watches', 0.8831, 72),
    ('fashion', 'unisex clothing', 'vintage watch repair kit', 'mandala crafts watch repair tool kit to replace batteries, remove links, open backs', 0.8982, 58),
    ('fashion', 'unisex clothing', 'leather watch winder', 'heiden design dual watch winder - cherry wood', 0.8999, 78),
    ('electronics', 'all electronics', 'vintage watch display case', 'new fashion retro stylish style quartz wristwatch with pu leather band -dark brown', 0.8688, 40),
    ('fashion', 'unisex clothing', 'vintage watch repair kit', 'jcc 5-piece watch repair band sizing tool kit set bracelet link pin remover removal tool', 0.8816, 55),
    ('fashion', "women's clothing", 'leather watch winder', "automatic watch winder with quiet japanese motor,ac adapter or battery powered,wood shell piano finish, silent motor, flexible watch pillow for men's and women's watches (black piano paint-113bw)", 0.8992, 82),
    ('electronics', 'all electronics', 'vintage watch display case', 'huanlong apple watch pc plated cover case slim premium super/exact fit/plastic cover snap on hard protective case for apple watch series 2 (gold 38mm)', 0.8667, None),
    ('fashion', 'unisex clothing', 'vintage watch repair kit', 'se tools 16 pcs watch tool kit plus adjustable watch wrench all in one', 0.8780, 60),
    ('fashion', 'unisex clothing', 'leather watch winder', 'melancy watch winder for double automatic watches adjustable watch pillows wood shell piano paint exterior led lighting and touch screen', 0.8970, 85),
    ('fashion', 'unisex clothing', 'vintage watch display case', '38mm metal case for apple watch 3, heavy duty rugged cover with stainless steel band for iwatch series 3 2 1 strap 38mm full protective bumper case tempered glass protector (38mm, silver)', 0.8666, None),
    ('fashion', 'unisex clothing', 'vintage watch repair kit', 'sodial(r) watch strap band spring bar link pin remover repair new', 0.8664, 52),
    ('fashion', 'unisex clothing', 'leather watch winder', 'double watch winder, love nest high-grade japanese mabuchi motor wood automatic watch winder burgundy and black inside piano finish pure handmade with quiet watch winder box [power included]', 0.8954, 88),
    ('electronics', 'computers', 'vintage watch display case', 'julk hard pc case compatible with apple watch series 7 45mm with tempered glass screen protector, ultra-thin h', 0.8664, None),
]
_CAPTURE_WATCH = 'new fashion retro stylish style quartz wristwatch with pu leather band -dark brown'


def _capture_case():
    rows = [{"bsns_vrtcl_name": v, "categ_lvl2_name": c, "Product_title": t, "gender": "",
             "occasion": "", "parent_asin": f"C{i}"} for i, (v, c, _, t, _, _) in
            enumerate(CAPTURE_1015)]
    hits = {}
    for _, _, intent, title, cos, _ in CAPTURE_1015:
        hits.setdefault(intent, []).append((title, cos))
    for listed in hits.values():
        listed.sort(key=lambda pair: -pair[1])
    recorded = {t: s for _, _, _, t, _, s in CAPTURE_1015 if s is not None}
    return rows, hits, recorded


def test_replay_of_the_recorded_1015_watch_capture():
    """The meeting's failure through the new code, 0 LLM: the recorded window, intents
    and scores. The pool held one real watch (rr 40: "not an accessory for a collector").
    The code reads the gift ask, seats that watch FIRST, sends the gift text and marks
    every row device / accessory -- what the model judged it from. It never overrides a
    judged reject: with the recorded scores the watch stays in the rejected band; with a
    reranker that follows the new text (the watch 90, add-ons 50-70, care 35) it leads."""
    import app.reranker as rr
    rows, hits, recorded = _capture_case()
    for shape in SHAPES:
        # one page of 24 holds the whole 15-row pool
        on = run(GIFT_QUERY, rows, hits, shape=shape, intents=CAPTURE_INTENTS, scores=recorded,
                 top_k=24)
        off = run(GIFT_QUERY, rows, hits, shape=shape, intents=CAPTURE_INTENTS, scores=recorded,
                  knob_values={"DEVICE_ROLE_RANKING": False}, top_k=24)
        call = on.rerank.calls[0]
        # The core row leads the window: the real watch, the pool's only one. (Until
        # catalog_taxonomy's 2026-10-30 fix the SE Tools "watch tool kit PLUS adjustable
        # watch wrench" read as a watch too and took seat 2 as `role=device`.)
        core = [r["Product_title"] for r in call["window_rows"] if dr.is_core(r, "watch")]
        assert core == [_CAPTURE_WATCH], core
        assert call["window"][:1] == core, call["window"]
        assert off.rerank.calls[0]["window"].index(_CAPTURE_WATCH) > 2
        assert call["constraints"] == [dr.constraint(dr.DeviceAsk("watch", True, False))]
        roles = [r.get(rr.ROLE_FIELD_KEY) for r in call["window_rows"]]
        assert roles == ["device"] + ["accessory"] * 14, roles
        # the reserve: one add-on per intent, storage first (all three intents are add-ons)
        tail = call["window"][12:]
        kinds = [dr.addon_rank(r, "watch") for r in call["window_rows"][12:]]
        assert kinds == sorted(kinds) and len(tail) == 3, (tail, kinds)
        # recorded scores: the reject stays a reject (last), the winders lead band 0
        assert on.titles[-1] == _CAPTURE_WATCH and off.titles[-1] == _CAPTURE_WATCH
        policy = {t: (90 if t == _CAPTURE_WATCH else 35 if "tool" in t or "remover" in t
                      else 60) for t in (r["Product_title"] for r in rows)}
        led = run(GIFT_QUERY, rows, hits, shape=shape, intents=CAPTURE_INTENTS, scores=policy)
        assert led.titles[0] == _CAPTURE_WATCH, led.titles
        # the 3 rows the 10-15 reply returned without a page-1 score stay unjudged here
        assert sum(r.get("rerank_score") is None for r in on.results) == 3


# ------------------------------------------------------------------ plain runner
if __name__ == "__main__":
    _PLAIN_RUNNER = True
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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
