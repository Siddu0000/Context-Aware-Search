"""Offline tables for app/exclusions.py (2026-11-19): which stated exclusions a query names
and which rows show them. No LLM, no catalog, no model.

The Databricks App review: "Find me a summer dress that isn't floral, isn't too short, and
is suitable for a brunch date" served floral dresses (main.py's use of it: tests/
test_ranking.py, test_exclusions_*). A new vocabulary word goes in app/exclusions.py AND
these tables; a phrasing the reader must NOT act on goes in NOT_READ.

Runs under pytest (`python -m pytest tests/test_exclusions.py`) or plainly
(`python tests/test_exclusions.py`).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import exclusions as ex  # noqa: E402

REVIEW_QUERY = ("Find me a summer dress that isn't floral, isn't too short, and is suitable "
                "for a brunch date.")

# query -> the keys it excludes
READ = {
    REVIEW_QUERY: {"floral", "mini"},
    "dress that is not black": {"black"},
    "dress that isnt black": {"black"},
    "summer dress that isn’t floral": {"floral"},        # curly apostrophe
    "no sequins or lace": {"sequin", "lace"},
    "not floral, striped or plaid": {"floral", "striped", "plaid"},
    "non-floral maxi dress": {"floral"},                       # maxi is wanted here
    "anything but black": {"black"},
    "nothing in black": {"black"},
    "not made of polyester": {"polyester"},
    "isn't too floral": {"floral"},
    "jacket without a hood": {"hooded"},
    "not too long": {"maxi"},
    "no flower prints please": {"floral"},
    "black and white dress, not too short": {"mini"},
    "no leather and no suede": {"leather", "suede"},
    "except for red": {"red"},
    "nothing pink or purple": {"pink", "purple"},
    "I hate florals": {"floral"},
    "something that is not see-through": {"sheer"},
    "no logo hoodie": {"graphic"},
    "summer dress not grey": {"grey"},
    "summer dress not gray": {"grey"},
    "I don't want anything floral": {"floral"},
    "a top without ruffles or embroidery": {"ruffle", "embroidered"},
    "no prints": {"patterned"},
}

# Phrasings the reader must leave alone: a wanted "free-from" attribute, a product name, a
# negation of something outside the vocabulary, or no negation at all
NOT_READ = [
    "no-show socks", "no added sugar", "non-stick pan", "non-alcoholic beer",
    "I don't want a flagship-priced phone", "without wasting ingredients",
    "dress with no sleeves", "not short sleeve tee", "knot dress", "forget me not seeds",
    "Not Your Daughter's Jeans", "lace-up boots", "flower girl dress", "gluten free snacks",
    "floral summer dress", "black dress", "short sleeve floral blouse", "no iron dress shirt",
    "non-slip work shoes", "nothing bundt cake", "not too spicy salsa", "", None, 7,
]


def test_reader_table():
    for query, keys in READ.items():
        assert ex.requested(query) == frozenset(keys), (query, ex.requested(query))
        assert keys <= ex.KEYS, query


def test_reader_leaves_other_phrasings_alone():
    for query in NOT_READ:
        assert ex.requested(query) == frozenset(), (query, ex.requested(query))


def _row(title, vertical="Fashion", **cols):
    return {"Product_title": title, "bsns_vrtcl_name": vertical, **cols}


def test_rows_that_show_an_excluded_key():
    shows = [
        (_row("HUHOT Floral Summer Dresses Women's Midi Sundress Flower-32 Medium"), "floral"),
        (_row("MSBASIC Sunflower Dress Thin Strap V Neck Short Summer Dress"), "floral"),
        (_row("Women Vintage Botanical Print Wrap Dress"), "floral"),
        (_row("Womens Bodycon Mini Dress Party"), "mini"),
        (_row("Solid Midi Dress", color="black"), "black"),               # colour column
        (_row("Plain Shift Dress", color_family="black"), "black"),       # colour family
        (_row("Basic Tee", material="Polyester"), "polyester"),           # material column
        (_row("Leather Phone Case for iPhone 12", vertical="Electronics"), "leather"),
        (_row("Women's Lace Trim Cami Top"), "lace"),
        (_row("Striped Linen Shirt"), "patterned"),
        (_row("Slate Grey Knit Dress"), "grey"),
        (_row("Slate Gray Knit Dress"), "grey"),
    ]
    for row, key in shows:
        assert ex.matches(row, {key}), (row, key)


def test_rows_that_do_not_show_it():
    clear = [
        (_row("Flower Girl Dress for Wedding"), "floral"),         # a role, not a print
        (_row("Women's Lace-Up Ankle Boots"), "lace"),             # a closure, not a trim
        (_row("Women's Short Sleeve Midi Dress"), "mini"),         # no bare "short"
        (_row("Summer Dresses for Women Casual Midi Length T-Shirt Dress - Mid Grey"), "floral"),
        # Grocery is the reranker's: a free-from listing is what a food negation wants
        (_row("Peanut-Free Black Bean Chips", vertical="Grocery"), "black"),
        (_row("Floral Honey 12 oz", vertical="Grocery"), "floral"),
    ]
    for row, key in clear:
        assert not ex.matches(row, {key}), (row, key)
    assert not ex.matches(_row("Floral Dress"), frozenset())
    assert not ex.matches({}, {"floral"})


def test_constraint_text_is_the_vocabularys_own():
    c = ex.constraint(ex.requested(REVIEW_QUERY))
    assert c == {"type": "exclude",
                 "value": "NOT floral prints (flowers, botanical, blossom); NOT mini length "
                          "(too short): a product that shows any of them violates it"}
    # deterministic order, never the shopper's words
    assert ex.constraint({"mini", "floral"}) == c
    assert "brunch" not in c["value"] and "isn't" not in c["value"]


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
    print(f"\n{len(tests) - failed} passed, {failed} failed, 0 skipped")
    sys.exit(1 if failed else 0)
