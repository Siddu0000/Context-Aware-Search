"""Offline tests for the type-aware ordering in app/main.py (TYPE_AWARE_RANKING) and the
same-type upsell in app/recommendations.py.

No LLM, no catalog load, no embedding model. main.search() runs with the translator and
the reranker faked, and retrieval runs the REAL app.search.search_products over a small
annotated frame (only _raw_hits is faked), so keep=, the per-intent quota and the
NaN -> None row shape are the real ones. Every fixture is built in BOTH row shapes: the
Databricks one (Title-case, gender None off Fashion) and the local one, pushed through
app.search._build_search_text (lowercase vertical/title/category, gender '' off Fashion).

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
    """Pin the knobs these tests read; .env must not decide what a test checks."""
    pinned = dict(TYPE_AWARE_RANKING=True, SPONSORED_MIN_RERANK_SCORE=40.0,
                  RERANK_INPUT_K=15, QUERY_GUARD_ENABLED=True)
    pinned.update(values)
    with patched(cfg, pinned):
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
    """The served frame: annotated first (load_index's order), then for the local shape
    lowercased IN PLACE by the real _build_search_text."""
    df = pd.DataFrame(rows)
    pt.annotate(df)
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
    def __init__(self, out, rerank, searches):
        self.out, self.rerank, self.searches = out, rerank, searches

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
        typed=True, rerank=True, smap=None, top_k=12, requested=_UNSET, clear=True):
    """One main.search() call over the fakes; every endpoint parameter explicit."""
    fake_rerank = FakeRerank(scores)
    searches = []
    real_search = S.search_products

    def search_products(terms, top_k=30, per_intent_quota=None, **kwargs):
        searches.append({"top_k": top_k, "quota": per_intent_quota, "keep": "keep" in kwargs})
        return real_search(terms, top_k=top_k, per_intent_quota=per_intent_quota, **kwargs)

    def understand_query(q, mode=None, errors=None):
        return {"intents": list(intents), "bundle_type": bundle_type,
                "constraints": [dict(c) for c in constraints],
                "is_recipe": bundle_type == "recipe", "query": q}

    fakes = {
        "understand_query": understand_query,
        "search_products": search_products,
        "llm_rerank": fake_rerank,
        "MIN_RESULT_RELEVANCE": 0.0,
        "build_recommendations": lambda q, r, page=1: {
            "cross_sell": [], "upsell": [], "cross_sell_source": None, "errors": []},
        "sponsored_map": lambda: dict(smap or {}),
    }
    type_patch = {} if requested is _UNSET else {"requested_types": lambda q: requested}
    if clear:
        page_cache.clear()
    with knobs(TYPE_AWARE_RANKING=typed), served(rows, hits, shape), \
            patched(main, fakes), patched(pt, type_patch):
        out = main.search(query=query, top_k=top_k, page=1, rerank=rerank, recommend=False,
                          sponsored=bool(smap), translator_mode=None)
    if clear:
        page_cache.clear()
    return Run(out, fake_rerank, searches)


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
    # the local shape really is lowercase with '' gender off Fashion
    local = as_rows(rows, "local")
    grocery = local[rows.index(_GENDER_CASES[9][0])]
    assert grocery["bsns_vrtcl_name"] == "grocery" and grocery["gender"] == ""
    assert main._product_gender(grocery) == "neutral"
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
            assert gated.rerank.calls[0]["constraints"] == []
            # the gate logs a flag, never the query
            assert any("non_fashion_pool" in m for m in logs), (query, logs)
            assert not any(query in m for m in logs), logs
            # ...and it mattered: without it the Fashion rows would lead the window
            window = gated.rerank.calls[0]["window_rows"]
            assert main._flat_window(window, requested, [query], True) != window, query


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
    out = main._flat_window(cands, frozenset({"apparel"}), intents, interleave=False)
    assert [c["Product_title"] for c in out] == [
        "dress1", "blazer1", "dress2", "blazer2", "blazer3",   # tier 0, intents round-robin
        "clarks",                                              # tier 1: untyped Fashion
        "snack", "sandal1", "earring", "sandal2",              # tier 2, round-robin
    ]
    out = main._flat_window(cands, frozenset({"apparel"}), intents, interleave=True)
    assert [c["Product_title"] for c in out][:5] == [
        "dress1", "blazer2", "blazer1", "blazer3", "dress2"], out  # women/men alternate
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
            "dress30", "dress40",               # band 1 (<= the ceiling)
            "dress_tail", "unknown_tail",       # band 2 (not reranked)
            "sandal95", "sandal_tail",          # mismatches last, whatever they scored
        ]
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
            sent = [c for c in r.rerank.calls[0]["constraints"] if c["type"] != "product_type"]
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
        old = run(query, rows, hits, shape=shape, intents=intents, scores=scores,
                  requested=frozenset({"apparel", "accessory"}))
        assert not {t.casefold() for t in bands} & set(old.rerank.calls[0]["window"])
        assert not {t.casefold() for t in bands} & set(old.titles), old.titles


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
