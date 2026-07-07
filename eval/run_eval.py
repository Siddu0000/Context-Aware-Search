"""End-to-end retrieval evaluation.

Reads queries with relevance criteria from data/eval_queries.json, runs the
pipeline on each, and computes:
  - Precision@K (K=1,5,10)
  - Recall@K   (K=10)
  - MRR        (Mean Reciprocal Rank)
  - NDCG@10
  - Per-query latency per stage
  - Token-cost proxy

Usage:
    python -m eval.run_eval                      # rerank ON, all fields
    python -m eval.run_eval --no-rerank          # baseline HyDE-only
    python -m eval.run_eval --exclude-description  # measure label-leakage
    python -m eval.run_eval --tag custom         # name the output CSV

Outputs land in eval_results/<tag>_<timestamp>.csv plus a stdout summary.

Note on label leakage:
The synthetic `prod_description` column was generated as a template from
the color/material/occasion attributes — the same ones used here for
relevance. Use --exclude-description to remove that column from the
embedded search text and see how much of the precision was from leakage.
"""

import argparse
import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import List, Set, Tuple

import pandas as pd

from app.config import EVAL_QUERIES_JSON, EVAL_RESULTS_DIR, FINAL_TOP_K, RETRIEVAL_TOP_K
from app.metrics import StageTimings, approx_tokens
from app.reranker import rerank as llm_rerank
from app.search import (
    DEFAULT_SEARCH_FIELDS,
    get_dataframe,
    load_index,
    search_products,
)
from app.translator import translate_query
from eval.metrics_ir import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

logging.basicConfig(level=logging.WARNING)


def build_relevance_set(df: pd.DataFrame, criteria: dict) -> Set[str]:
    """Resolve a query's relevance criteria against the catalog.

    Supported keys:
      - title_must_include:  all terms must appear in Product_title
      - title_any_of:        any term may appear in Product_title
      - color_in:            color matches one of these
      - material_in:         material matches one of these
      - occasion_in:         occasion matches one of these
      - category_in:         bsns_vrtcl_name or categ_lvl2_name matches
    """
    mask = pd.Series([True] * len(df))

    if "title_must_include" in criteria:
        for term in criteria["title_must_include"]:
            mask &= df["Product_title"].astype(str).str.lower().str.contains(
                re.escape(term.lower())
            )

    if "title_any_of" in criteria:
        any_mask = pd.Series([False] * len(df))
        for term in criteria["title_any_of"]:
            any_mask |= df["Product_title"].astype(str).str.lower().str.contains(
                re.escape(term.lower())
            )
        mask &= any_mask

    if "color_in" in criteria:
        colors = [c.lower() for c in criteria["color_in"]]
        mask &= df["color"].astype(str).str.lower().isin(colors)

    if "material_in" in criteria:
        materials = [m.lower() for m in criteria["material_in"]]
        mask &= df["material"].astype(str).str.lower().isin(materials)

    if "occasion_in" in criteria:
        occasions = [o.lower() for o in criteria["occasion_in"]]
        mask &= df["occasion"].astype(str).str.lower().isin(occasions)

    if "category_in" in criteria:
        cats = [c.lower() for c in criteria["category_in"]]
        cat_mask = df["bsns_vrtcl_name"].astype(str).str.lower().isin(cats) | df[
            "categ_lvl2_name"
        ].astype(str).str.lower().isin(cats)
        mask &= cat_mask

    return set(df.loc[mask, "Product_title"].astype(str).tolist())


def evaluate(
    rerank_on: bool,
    tag: str,
    fields: Tuple[str, ...] = DEFAULT_SEARCH_FIELDS,
    translate_on: bool = True,
) -> Path:
    load_index(fields=fields)
    df = get_dataframe()

    with open(EVAL_QUERIES_JSON, "r", encoding="utf-8") as f:
        queries = json.load(f)

    rows = []
    for q in queries:
        query = q["query"]
        relevant = build_relevance_set(df, q.get("relevance", {}))
        if not relevant:
            print(f"[skip] No relevant products for query: {query!r}")
            continue

        timings = StageTimings()
        tokens_in = 0

        with timings.stage("translate"):
            if translate_on:
                intents = translate_query(query)
                tokens_in += approx_tokens(query)
            else:
                intents = [query]

        with timings.stage("retrieve"):
            candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)

        if rerank_on and candidates:
            with timings.stage("rerank"):
                final = llm_rerank(query, candidates, top_k=FINAL_TOP_K)
                tokens_in += sum(
                    approx_tokens(c.get("Product_title", ""))
                    for c in candidates[: max(FINAL_TOP_K * 3, 30)]
                )
        else:
            final = candidates[:FINAL_TOP_K]

        retrieved_titles = [p.get("Product_title", "") for p in final]

        rows.append(
            {
                "query": query,
                "n_relevant_in_catalog": len(relevant),
                "P@1": precision_at_k(retrieved_titles, relevant, 1),
                "P@5": precision_at_k(retrieved_titles, relevant, 5),
                "P@10": precision_at_k(retrieved_titles, relevant, 10),
                "R@10": recall_at_k(retrieved_titles, relevant, 10),
                "MRR": reciprocal_rank(retrieved_titles, relevant),
                "NDCG@10": ndcg_at_k(retrieved_titles, relevant, 10),
                "ms_translate": timings.timings_ms.get("translate", 0),
                "ms_retrieve": timings.timings_ms.get("retrieve", 0),
                "ms_rerank": timings.timings_ms.get("rerank", 0),
                "ms_total": timings.total_ms,
                "tokens_in_approx": tokens_in,
            }
        )

        print(
            f"{query!r:55s} P@10={rows[-1]['P@10']:.2f} MRR={rows[-1]['MRR']:.2f} "
            f"NDCG={rows[-1]['NDCG@10']:.2f} total={rows[-1]['ms_total']}ms"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_RESULTS_DIR / f"{tag}_{ts}.csv"
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\n=== Summary ===")
    if rows:
        for col in ("P@1", "P@5", "P@10", "R@10", "MRR", "NDCG@10", "ms_total"):
            print(f"  mean_{col:9s} = {mean(r[col] for r in rows):.3f}")
    print(f"  rerank_on    = {rerank_on}")
    print(f"  fields       = {fields}")
    print(f"  saved to     = {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-rerank", action="store_true", help="Disable LLM rerank.")
    p.add_argument(
        "--exclude-description",
        action="store_true",
        help="Exclude prod_description from search text — diagnoses label leakage.",
    )
    p.add_argument("--tag", default=None, help="Filename tag for the output CSV.")
    args = p.parse_args()

    fields = DEFAULT_SEARCH_FIELDS
    if args.exclude_description:
        fields = tuple(f for f in DEFAULT_SEARCH_FIELDS if f != "prod_description")

    rerank_on = not args.no_rerank
    default_tag = f"{'rerank_on' if rerank_on else 'rerank_off'}"
    if args.exclude_description:
        default_tag += "_no_desc"
    tag = args.tag or default_tag
    evaluate(rerank_on=rerank_on, tag=tag, fields=fields)


if __name__ == "__main__":
    main()
