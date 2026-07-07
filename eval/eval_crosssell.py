"""Cross-sell quality eval.

The recommendation engine suggests COMPLEMENTARY products (things bought
ALONGSIDE the anchor), grounded to real catalog items. This eval checks, per
anchor query, whether the cross-sell results actually look complementary.

Caveat (be honest about this): "complementary" is fuzzy and there is no ground
truth in the data (no co-purchase signal — bought_together is empty). So this
is a PROXY: for each anchor we list the complement categories a human would
expect, and measure how many cross-sell results match one. It tells you the
cross-sell is in the right ballpark, not that it is provably optimal.

Per case it reports:
  - complement_relevance: fraction of cross-sell items whose title matches an
    expected complement term (higher is better).
  - substitute_rate: fraction whose title matches a "substitute" term, i.e.
    the engine suggested another of the same thing instead of a complement
    (lower is better; optional per case).

Needs the catalog loaded and the LLM configured (cross-sell makes an LLM call),
same as eval/compare_llms.

Usage:
    python -m eval.eval_crosssell
    python -m eval.eval_crosssell --top-k 12
"""

import argparse
import json
import logging
from statistics import mean

from app.config import DATA_DIR, FINAL_TOP_K, RETRIEVAL_TOP_K
from app.recommendations import recommend
from app.search import load_index, search_products
from app.translator import translate_query

logging.basicConfig(level=logging.WARNING)

CROSS_SELL_FILE = DATA_DIR / "cross_sell_eval.json"


def _cross_sell_titles(query: str, top_k: int) -> list[str]:
    intents = translate_query(query)
    candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)
    results = candidates[:top_k]
    rec = recommend(query, results)
    return [c.get("Product_title", "") for c in rec.get("cross_sell", [])]


def _any_match(terms: list[str], title_lower: str) -> bool:
    return any(t.lower() in title_lower for t in terms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=FINAL_TOP_K,
                        help="How many search results to anchor cross-sell on.")
    args = parser.parse_args()

    load_index()
    spec = json.loads(CROSS_SELL_FILE.read_text(encoding="utf-8"))
    cases = spec["cases"]

    rel_scores, sub_scores = [], []
    print(f"\n=== Cross-sell eval — {len(cases)} anchors ===")
    for case in cases:
        query = case["query"]
        complement = case.get("expect_complement_any", [])
        substitute = case.get("avoid_substitute_any", [])

        titles = _cross_sell_titles(query, args.top_k)
        if not titles:
            print(f"\nANCHOR: {query!r}\n  (no cross-sell returned — LLM/quota issue or empty pool)")
            continue

        rel_hits, sub_hits, rows = 0, 0, []
        for t in titles:
            tl = t.lower()
            is_comp = _any_match(complement, tl)
            is_sub = bool(substitute) and _any_match(substitute, tl)
            rel_hits += is_comp
            sub_hits += is_sub
            tag = "OK " if is_comp else ("SUB" if is_sub else " ? ")
            rows.append(f"    [{tag}] {t[:70]}")

        relevance = rel_hits / len(titles)
        rel_scores.append(relevance)
        line = f"  complement_relevance: {rel_hits}/{len(titles)} = {relevance:.0%}"
        if substitute:
            sub_rate = sub_hits / len(titles)
            sub_scores.append(sub_rate)
            line += f"   substitute_rate: {sub_hits}/{len(titles)} = {sub_rate:.0%}"
        print(f"\nANCHOR: {query!r}")
        print(line)
        for r in rows:
            print(r)

    print("\n" + "=" * 60)
    if rel_scores:
        print(f"mean complement_relevance: {mean(rel_scores):.1%}")
    if sub_scores:
        print(f"mean substitute_rate:      {mean(sub_scores):.1%}  (lower is better)")
    print("Note: proxy metric — 'complementary' is judgment-based, no co-purchase ground truth.")


if __name__ == "__main__":
    main()
