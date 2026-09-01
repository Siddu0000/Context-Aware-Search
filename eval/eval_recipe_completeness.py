"""Grocery eval: % of a dish's ingredients that appear in the top-K results."""

import argparse
import json
import logging

import pandas as pd

from app.config import DATA_DIR, EVAL_RESULTS_DIR
from app.search import load_index, search_products
from app.translator import translate_query

logging.basicConfig(level=logging.WARNING)

RECIPE_FILE = DATA_DIR / "recipe_eval.json"


def _ingredient_terms(ingredient: str, synonyms: dict) -> list[str]:
    terms = [ingredient.lower()]
    for syn in synonyms.get(ingredient, []):
        terms.append(syn.lower())
    return terms


def _found_in_titles(terms: list[str], titles_blob: str) -> bool:
    return any(t in titles_blob for t in terms)


def _search_titles(dish_query: str, top_k: int) -> list[str]:
    # rerank is skipped on purpose: it reorders but never adds products, and it costs quota
    intents = translate_query(dish_query)
    candidates = search_products(intents, top_k=top_k)
    return [c.get("Product_title", "") for c in candidates[:top_k]]


def evaluate(top_k: int, query_template: str):
    load_index()
    spec = json.loads(RECIPE_FILE.read_text(encoding="utf-8"))
    target = spec.get("target_coverage", 0.7)
    dishes = spec["dishes"]

    all_rows = []
    coverages = []

    for d in dishes:
        dish = d["dish"]
        ingredients = d["ingredients"]
        synonyms = d.get("synonyms", {})

        query = query_template.format(dish=dish)
        titles = _search_titles(query, top_k)
        titles_blob = " || ".join(titles).lower()

        found_flags = {}
        for ing in ingredients:
            terms = _ingredient_terms(ing, synonyms)
            found_flags[ing] = _found_in_titles(terms, titles_blob)

        n_found = sum(found_flags.values())
        coverage = n_found / len(ingredients) if ingredients else 0.0
        coverages.append(coverage)

        print("\n" + "=" * 70)
        print(f"DISH: {dish}   (query: {query!r})")
        print(f"  coverage: {n_found}/{len(ingredients)} = {coverage:.0%}"
              f"   {'PASS' if coverage >= target else 'BELOW TARGET'}")
        print("  ingredient grid:")
        for ing, ok in found_flags.items():
            print(f"    [{'FOUND' if ok else '  -  '}] {ing}")

        for ing, ok in found_flags.items():
            all_rows.append(
                {"dish": dish, "ingredient": ing, "found": int(ok), "coverage": round(coverage, 3)}
            )

    overall = sum(coverages) / len(coverages) if coverages else 0.0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for d, cov in zip(dishes, coverages):
        bar = "#" * int(cov * 20)
        print(f"  {d['dish']:<22} {cov:>5.0%}  {bar}")
    print(f"\n  Overall mean coverage: {overall:.0%}   (target {target:.0%})")
    n_pass = sum(1 for c in coverages if c >= target)
    print(f"  Dishes meeting target: {n_pass}/{len(dishes)}")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EVAL_RESULTS_DIR / f"recipe_completeness_{ts}.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\n  Grid saved to: {out}")
    return overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=30, help="Results to inspect per dish.")
    p.add_argument(
        "--query-template",
        default="ingredients to make {dish}",
        help="Template for the search query. Must contain {dish}.",
    )
    args = p.parse_args()
    evaluate(top_k=args.top_k, query_template=args.query_template)


if __name__ == "__main__":
    main()
