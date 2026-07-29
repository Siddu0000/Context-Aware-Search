"""Runnable demo of the Swiggy-model hybrid search — no catalog or LLM keys
needed. Proves the routing: precise queries are served by the keyword engine;
intent queries the keyword engine can't match fall back to the intent layer.

The intent layer here is a LIGHTWEIGHT STAND-IN (concept expansion + BM25 over
the expanded query). It mimics the shape of CAS (expand -> retrieve) so the
demo runs offline. In production the router calls the real CAS pipeline
instead (see app/main.py `/search`), which does LLM query expansion + gte-small
vector retrieval + LLM rerank.

Run:  python -m demo.hybrid_demo
"""

import pandas as pd

from app.hybrid_router import route
from app.keyword_search import KeywordSearchEngine, _tokenize


# ---- 1. Synthetic catalog (titles deliberately avoid intent words like
#         "breathable"/"humid" so keyword search misses them) --------------
def build_catalog() -> pd.DataFrame:
    rows = [
        # Apparel — cotton/linen items are breathable, but titles never say so
        ("A1", "Men's Cotton Short-Sleeve Oxford Shirt", "Men's Clothing", "blue", "cotton", "casual", 4.5, 320),
        ("A2", "Linen Button-Down Beach Shirt", "Men's Clothing", "white", "linen", "casual", 4.4, 210),
        ("A3", "Lightweight Cotton Chino Shorts", "Men's Clothing", "beige", "cotton", "casual", 4.2, 150),
        ("A4", "Women's Linen Wrap Summer Dress", "Women's Clothing", "green", "linen", "casual", 4.6, 480),
        ("A5", "Genuine Leather Belt for Men", "Men's Clothing", "brown", "leather", "formal", 4.7, 900),
        ("A6", "Men's Reversible Leather Dress Belt", "Men's Clothing", "black", "leather", "formal", 4.5, 640),
        ("A7", "Merino Wool Crew-Neck Sweater", "Men's Clothing", "grey", "wool", "winter", 4.3, 275),
        ("A8", "Women's Chunky Knit Wool Cardigan", "Women's Clothing", "cream", "wool", "winter", 4.4, 190),
        ("A9", "Waterproof Trail Running Shoes", "Shoes", "black", "synthetic", "sport", 4.1, 410),
        ("A10", "Polyester Gym Training T-Shirt", "Men's Clothing", "red", "polyester", "sport", 4.0, 130),
        # Grocery — pantry ingredients; plus a "cake" decoration to bait keyword search
        ("G1", "All-Purpose Wheat Flour 2kg", "Baking", None, None, None, 4.6, 700),
        ("G2", "Granulated White Sugar 1kg", "Baking", None, None, None, 4.5, 520),
        ("G3", "Unsalted Butter 500g", "Dairy", None, None, None, 4.7, 610),
        ("G4", "Large Free-Range Eggs (12 pack)", "Dairy", None, None, None, 4.6, 830),
        ("G5", "Baking Powder 200g", "Baking", None, None, None, 4.4, 240),
        ("G6", "Pure Vanilla Extract 100ml", "Baking", None, None, None, 4.8, 300),
        ("G7", "Happy Birthday Cake Topper Decoration", "Party Supplies", None, None, None, 4.1, 95),
        ("G8", "Dark Roast Ground Coffee 500g", "Beverages", None, None, None, 4.5, 560),
    ]
    return pd.DataFrame(rows, columns=[
        "parent_asin", "Product_title", "categ_lvl2_name", "color",
        "material", "occasion", "average_rating", "rating_number",
    ])


# ---- 2. Lightweight intent-layer stand-in (concept expansion + BM25) ------
#         Stands in for the real CAS pipeline so the demo runs offline.
_CONCEPTS = {
    "breathable": ["cotton", "linen", "lightweight"],
    "humid": ["cotton", "linen", "short-sleeve"],
    "hot": ["cotton", "linen", "shorts", "short-sleeve"],
    "summer": ["linen", "cotton", "shorts"],
    "warm": ["wool", "knit", "sweater", "cardigan"],
    "cold": ["wool", "knit", "sweater"],
    "winter": ["wool", "cardigan"],
    "cozy": ["wool", "knit"],
    "bake": ["flour", "sugar", "butter", "eggs", "baking", "vanilla"],
    "baking": ["flour", "sugar", "butter", "eggs", "vanilla"],
    "cake": ["flour", "sugar", "butter", "eggs", "baking", "vanilla"],
    "outfit": ["shirt", "dress", "shorts"],
    "wear": ["shirt", "dress", "shorts"],
}


class LocalIntentLayer:
    """Demo stand-in: expand the query via a concept map, then retrieve with
    BM25 over the expanded terms. This is the offline analogue of CAS's
    'LLM query expansion -> semantic retrieval'."""

    def __init__(self, engine: KeywordSearchEngine):
        self.engine = engine

    def _expand(self, query: str) -> str:
        terms = _tokenize(query)
        expanded = list(terms)
        for t in terms:
            expanded.extend(_CONCEPTS.get(t, []))
        return " ".join(expanded)

    def search(self, query: str, k: int = 12):
        expanded = self._expand(query)
        results, _, _, _ = self.engine.search(expanded, k=k)
        for r in results:
            r["intent_expanded_query"] = expanded
        return results


# ---- 3. Run the demo ------------------------------------------------------
def _show(payload):
    print(f"\nQUERY: {payload['query']!r}")
    print(f"  PATH : {payload['path'].upper()}  ({payload['reason']})")
    if payload["path"] == "intent" and payload["results"]:
        print(f"  expanded -> {payload['results'][0].get('intent_expanded_query')!r}")
    if not payload["results"]:
        print("  (no results)")
    for r in payload["results"][:5]:
        print(f"    - {r['Product_title']}  [bm25={r.get('keyword_score')}]")


def main():
    df = build_catalog()
    engine = KeywordSearchEngine()
    engine.index(df)
    intent = LocalIntentLayer(engine)
    intent_fn = lambda q, k: intent.search(q, k)

    print("=" * 70)
    print("SWIGGY-MODEL HYBRID SEARCH DEMO  (keyword-first, intent fallback)")
    print("=" * 70)

    # Precise queries -> keyword engine handles them (the common case)
    for q in ["genuine leather belt for men", "ground coffee", "wool sweater"]:
        _show(route(q, engine, intent_fn, k=12, min_coverage=0.5))

    # Intent queries -> keyword misses -> CAS fallback catches them
    for q in [
        "something breathable to wear in hot humid weather",
        "an outfit for a cold winter day",
        "ingredients to bake a cake",
    ]:
        _show(route(q, engine, intent_fn, k=12, min_coverage=0.5))

    print("\n" + "=" * 70)
    print("Note: the intent layer here is an offline stand-in. In production the")
    print("router calls the real CAS pipeline (LLM expansion + gte-small vectors")
    print("+ LLM rerank) via the same interface — see app/main.py `/search`.")


if __name__ == "__main__":
    main()
