"""End-to-end retrieval evaluation.

Reads queries with relevance criteria from data/eval_queries.json, runs the
pipeline on each, and computes:
  - Precision@K (K=1,5,10)
  - Recall@K (K=10)
  - MRR (Mean Reciprocal Rank)
  - NDCG@10
  - Per-query latency per stage
  - Token-cost proxy

Usage:
    python -m eval.run_eval                 # default: rerank ON
    python -m eval.run_eval --no-rerank     # baseline: HyDE retrieval only
    python -m eval.run_eval --tag rerank_on # name the output CSV

Outputs land in eval_results/<tag>_<timestamp>.csv plus a summary printed
to stdout.
"""

import argparse
import csv
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set

import pandas as pd

from app.config import EVAL_QUERIES_JSON, EVAL_RESULTS_DIR, FINAL_TOP_K, RETRIEVAL_TOP_K
from app.metrics import StageTimings, approx_tokens
from app.reranker import rerank as llm_rerank
from app.search import get_dataframe, load_index, search_products
from app.translator import translate_query
from eval.metrics_ir import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

logging.basicConfig(level=logging.WARNING)


def _matches(text: str, terms: List[str], mode: str = "all") -> bool:
    text = text.lower()
    needles = [t.lower() for t in terms]
    if mode == "all":
        return all(n in text for n in needles)
    return any(n in text for n in needles)


def build_relevance_set(df: pd.DataFrame, criteria: dict) -> Set[str]:
    """Resolve a query's relevance criteria against the catalog.

    Supported criteria keys:
      - title_must_include: list[str] — all must appear in Product_title 
      - title_any_of: list[str] — any may appear in Product_title 
      - color_in: list[str] — color matches one of these (case-insensitive)
      - category_in: list[str] — bsns_vrtcl_name or categ_lvl2_name match
    """
    mask = pd.Series([True] * len(df))

    if "title_must_include" in criteria:
        for term in criteria["title_must_include"]:
            mask &= df["Product_title "].astype(str).str.lower().str.contains(
                re.escape(term.lower())
            )

    if "title_any_of" in criteria:
        any_mask = pd.Series([False] * len(df))
        for term in criteria["title_any_of"]:
            any_mask |= df["Product_title "].astype(str).str.lower().str.contains(
                re.escape(term.lower())
            )
        mask &= any_mask

    if "color_in" in criteria:
        colors = [c.lower() for c in criteria["color_in"]]
        mask &= df["color"].astype(str).str.lower().isin(colors)

    if "category_in" in criteria:
        cats = [c.lower() for c in criteria["category_in"]]
        cat_mask = df["bsns_vrtcl_name"].astype(str).str.lower().isin(cats) | df[
            "categ_lvl2_name"
        ].astype(str).str.lower().isin(cats)
        mask &= cat_mask

    return set(df.loc[mask, "Product_title "].astype(str).tolist())


def evaluate(rerank_on: bool, tag: str) -> Path:
    load_index()
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
            intents = translate_query(query)
            tokens_in += approx_tokens(query)

        with timings.stage("retrieve"):
            candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)

        if rerank_on and candidates:
            with timings.stage("rerank"):
                final = llm_rerank(query, candidates, top_k=FINAL_TOP_K)
                tokens_in += sum(approx_tokens(c.get("Product_title ", "")) for c in candidates[: max(FINAL_TOP_K * 3, 30)])
        else:
            final = candidates[:FINAL_TOP_K]

        retrieved_titles = [p.get("Product_title ", "") for p in final]

        p_at_1 = precision_at_k(retrieved_titles, relevant, 1)
        p_at_5 = precision_at_k(retrieved_titles, relevant, 5)
        p_at_10 = precision_at_k(retrieved_titles, relevant, 10)
        r_at_10 = recall_at_k(retrieved_titles, relevant, 10)
        mrr = reciprocal_rank(retrieved_titles, relevant)
        ndcg = ndcg_at_k(retrieved_titles, relevant, 10)

        rows.append(
            {
                "query": query,
                "n_relevant_in_catalog": len(relevant),
                "P@1": p_at_1,
                "P@5": p_at_5,
                "P@10": p_at_10,
                "R@10": r_at_10,
                "MRR": mrr,
                "NDCG@10": ndcg,
                "ms_translate": timings.timings_ms.get("translate", 0),
                "ms_retrieve": timings.timings_ms.get("retrieve", 0),
                "ms_rerank": timings.timings_ms.get("rerank", 0),
                "ms_total": timings.total_ms,
                "tokens_in_approx": tokens_in,
            }
        )

        print(
            f"{query!r:60s} P@10={p_at_10:.2f} MRR={mrr:.2f} NDCG={ndcg:.2f} "
            f"total={timings.total_ms}ms"
        )

    # ---- Persist + summarize ----
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
    print(f"  rerank_on = {rerank_on}")
    print(f"  saved to  {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-rerank", action="store_true", help="Disable LLM rerank stage.")
    p.add_argument("--tag", default=None, help="Filename tag for the output CSV.")
    args = p.parse_args()

    rerank_on = not args.no_rerank
    tag = args.tag or ("rerank_on" if rerank_on else "rerank_off")
    evaluate(rerank_on=rerank_on, tag=tag)


if __name__ == "__main__":
    main()
