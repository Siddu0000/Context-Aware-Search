"""Recipe-completeness eval for the grocery use case.

  (2026-06-11) specified this QC: pick a few dishes, and check that
the grocery search surfaces the INGREDIENTS needed to cook them. Target:
70-80% of a dish's ingredients should appear in the top results.

Methodology, per dish:
  1. Query the pipeline with "ingredients to make <dish>".
  2. Take the top-K result titles (default K=30 — recipes have many
     ingredients, and   noted "the ingredients themselves would be
     more than 10", so top-10 alone is too tight a window).
  3. For each canonical ingredient, mark it FOUND if the ingredient word
     (or any of its synonyms) appears in any result title.
  4. coverage = found / total ingredients.

Output: per-dish per-ingredient hit/miss grid, per-dish coverage, and an
overall mean. Flags dishes below the target threshold.

Notes / caveats this surfaces (worth telling  ):
  - The "70 paneer sellers" problem: if the search returns 30 variants of
    the headline ingredient, coverage of the OTHER ingredients suffers.
    The per-ingredient grid makes this visible — you'll see paneer FOUND
    but peas/cumin/cream MISSING. That argues for diversity/dedup in
    retrieval (a separate P2 item).
  - Requires the grocery catalog (Grocery_and_Gourmet_Food) to be loaded
    into products.csv. If grocery rows are absent, coverage will be ~0 and
    that's a data problem, not a ranking problem.

Usage:
    python -m eval.eval_recipe_completeness
    python -m eval.eval_recipe_completeness --top-k 20
    python -m eval.eval_recipe_completeness --query-template "buy ingredients for {dish}"
"""

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
    """All title-substrings that count as a match for this ingredient."""
    terms = [ingredient.lower()]
    for syn in synonyms.get(ingredient, []):
        terms.append(syn.lower())
    return terms


def _found_in_titles(terms: list[str], titles_blob: str) -> bool:
    """True if any term appears as a substring in the concatenated titles."""
    return any(t in titles_blob for t in terms)


def _search_titles(dish_query: str, top_k: int) -> list[str]:
    """Run the pipeline (translate -> retrieve) and return top-K titles.

    Rerank is intentionally skipped here: ingredient coverage is about what
    the retrieval surfaces, and skipping rerank keeps the eval light on LLM
    quota. (The reranker reorders but does not add new products.)
    """
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

    # Save the grid for sharing.
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
