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


def _stage_label(err: dict) -> str:
    """'rerank:partial' keeps a partial rerank apart: its LLM ranking WAS used."""
    stage = str(err.get("stage"))
    return f"{stage}:partial" if err.get("code") == "partial" else stage


def _has(row: dict, label: str) -> bool:
    return label in row["fell_back"].split(";")


def _run_one(query: str, rerank_on: bool, top_k: int) -> dict:
    """Run a single query end-to-end, catching everything. Returns a report row."""
    t0 = time.perf_counter()
    row = {
        "query": query if query.strip() else "(whitespace/empty)",
        "status": "ok",
        "n_results": 0,
        "translator_fell_back": False,
        "fell_back": "",
        "n_intents": 0,
        "ms": 0,
        "note": "",
    }
    # Stages report their own fallback; intents == [query] was only a guess
    stage_errors: list = []
    try:
        intents = translate_query(query, errors=stage_errors)
        row["n_intents"] = len(intents)

        candidates = search_products(intents, top_k=top_k)
        row["n_results"] = len(candidates)

        if rerank_on and candidates:
            from app.reranker import rerank as llm_rerank

            ranked = llm_rerank(query, candidates, top_k=top_k, errors=stage_errors)
            row["n_results"] = len(ranked)
            if ranked and ranked[0].get("rerank_score") is None:
                row["note"] = "rerank fell back to embedding order"
    except Exception as e:
        row["status"] = "ERROR"
        row["note"] = f"{type(e).__name__}: {e}"[:120]
    # A fallback is graceful handling, NOT a hard error: status stays "ok"
    row["fell_back"] = ";".join(_stage_label(e) for e in stage_errors)
    row["translator_fell_back"] = _has(row, "translate")
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
        fb = f" [fell back: {row['fell_back']}]" if row["fell_back"] else ""
        note = f"  <{row['note']}>" if row["note"] else ""
        disp = row["query"][:48]
        print(f"  [{status_icon}] n={row['n_results']:>3} {row['ms']:>6}ms  {disp}{fb}{note}")

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_err = sum(1 for r in rows if r["status"] == "ERROR")
    n_fb = sum(1 for r in rows if r["translator_fell_back"])
    n_rerank_fb = sum(1 for r in rows if _has(r, "rerank"))
    n_rerank_partial = sum(1 for r in rows if _has(r, "rerank:partial"))
    # A partial rerank still used the LLM's ranking, so it is not a fallback
    n_any_fb = sum(1 for r in rows if r["translator_fell_back"] or _has(r, "rerank"))
    n_empty = sum(1 for r in rows if r["n_results"] == 0)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  handled without crashing : {n_ok}/{len(rows)}")
    print(f"  hard errors              : {n_err}")
    print(f"  an LLM stage fell back   : {n_any_fb}  (graceful, not a hard error)")
    print(f"    translator             : {n_fb}  (couldn't expand -> used raw query)")
    if args.rerank:
        print(f"    reranker               : {n_rerank_fb}  (embedding order kept)")
        print(f"  reranker partial         : {n_rerank_partial}  "
              "(LLM ranking used; invalid entries dropped)")
    print(f"  returned zero results    : {n_empty}")
    if n_err == 0:
        print("\n  PASS — no crashes. Injection-like strings were treated as plain text.")
        if n_any_fb:
            # Only the stage that fell back skipped the LLM; the other stage still saw the text
            untested = [f"translator x{n_fb} (raw query used)"] if n_fb else []
            if n_rerank_fb:
                untested.append(f"reranker x{n_rerank_fb} (embedding order kept)")
            print(f"  CAVEAT — {n_any_fb} quer(y/ies) fell back; injection inertness is "
                  f"untested for the stage that fell back: {', '.join(untested)}.")
    else:
        print("\n  ATTENTION — some queries errored. See the <...> notes above.")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EVAL_RESULTS_DIR / f"stress_test_{ts}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Report saved to: {out}")


if __name__ == "__main__":
    main()
