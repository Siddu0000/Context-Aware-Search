"""Stress test: measures graceful handling of edge/nonsense queries, not precision."""

import argparse
import json
import logging
import sys
import time

import pandas as pd

from app.config import DATA_DIR, EVAL_RESULTS_DIR
from app.search import load_index, search_products
from app.translator import translate_query

logging.basicConfig(level=logging.ERROR)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

STRESS_FILE = DATA_DIR / "stress_queries.json"


def _run_one(query: str, rerank_on: bool, top_k: int) -> dict:
    """Run a single query end-to-end, catching everything. Returns a report row."""
    t0 = time.perf_counter()
    row = {
        "query": query if query.strip() else "(whitespace/empty)",
        "status": "ok",
        "n_results": 0,
        "translator_fell_back": False,
        "n_intents": 0,
        "ms": 0,
        "note": "",
    }
    try:
        intents = translate_query(query)
        row["n_intents"] = len(intents)
        row["translator_fell_back"] = intents == [query]

        candidates = search_products(intents, top_k=top_k)
        row["n_results"] = len(candidates)

        if rerank_on and candidates:
            from app.reranker import rerank as llm_rerank

            ranked = llm_rerank(query, candidates, top_k=top_k)
            row["n_results"] = len(ranked)
            if ranked and ranked[0].get("rerank_score") is None:
                row["note"] = "rerank fell back to embedding order"
    except Exception as e:
        row["status"] = "ERROR"
        row["note"] = f"{type(e).__name__}: {e}"[:120]
    row["ms"] = round((time.perf_counter() - t0) * 1000)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rerank", action="store_true", help="Include the rerank stage.")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    load_index()
    spec = json.loads(STRESS_FILE.read_text(encoding="utf-8"))
    queries = spec["queries"]

    print("\n" + "=" * 80)
    print(f"STRESS TEST — {len(queries)} edge-case queries (rerank={args.rerank})")
    print("Goal: every query handled gracefully (no crash); injection strings inert.")
    print("=" * 80)

    rows = []
    for q in queries:
        row = _run_one(q, args.rerank, args.top_k)
        rows.append(row)
        status_icon = "OK " if row["status"] == "ok" else "ERR"
        fb = " [translator fell back]" if row["translator_fell_back"] else ""
        note = f"  <{row['note']}>" if row["note"] else ""
        disp = row["query"][:48]
        print(f"  [{status_icon}] n={row['n_results']:>3} {row['ms']:>6}ms  {disp}{fb}{note}")

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_err = sum(1 for r in rows if r["status"] == "ERROR")
    n_fb = sum(1 for r in rows if r["translator_fell_back"])
    n_empty = sum(1 for r in rows if r["n_results"] == 0)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  handled without crashing : {n_ok}/{len(rows)}")
    print(f"  hard errors              : {n_err}")
    print(f"  translator fell back     : {n_fb}  (couldn't expand -> used raw query)")
    print(f"  returned zero results    : {n_empty}")
    if n_err == 0:
        print("\n  PASS — no crashes. Injection-like strings were treated as plain text.")
    else:
        print("\n  ATTENTION — some queries errored. See the <...> notes above.")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EVAL_RESULTS_DIR / f"stress_test_{ts}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Report saved to: {out}")


if __name__ == "__main__":
    main()
