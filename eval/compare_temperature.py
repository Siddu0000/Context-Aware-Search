"""Temperature sweep — eval quality and run-to-run ordering stability per temp."""

import argparse
import json
import logging
import time
from statistics import mean

import pandas as pd

import app.config as cfg
from app.search import get_dataframe, load_index, search_products
from app.translator import translate_query
from eval.run_eval import evaluate

logging.basicConfig(level=logging.WARNING)


def _run_quality_at(temp: float, rerank_on: bool):
    """Run the full eval set once at a fixed temperature."""
    orig_det, orig_ovr = cfg.DETERMINISTIC, cfg.TEMPERATURE_OVERRIDE
    # fixed seed off, temperature forced — else the sweep isn't observable
    cfg.DETERMINISTIC = False
    cfg.TEMPERATURE_OVERRIDE = temp
    try:
        t0 = time.perf_counter()
        out_path = evaluate(rerank_on=rerank_on, tag=f"temp_{temp:.2f}")
        return out_path, time.perf_counter() - t0
    finally:
        cfg.DETERMINISTIC, cfg.TEMPERATURE_OVERRIDE = orig_det, orig_ovr


def _pipeline_titles(query: str, rerank_on: bool, top_k: int = 10):
    """Run one query through the pipeline and return its ordered result titles."""
    intents = translate_query(query)
    candidates = search_products(intents, top_k=max(top_k * 3, 30))
    if rerank_on and candidates:
        from app.reranker import rerank as llm_rerank

        ranked = llm_rerank(query, candidates, top_k=top_k)
        return tuple(p.get("Product_title", "") for p in ranked[:top_k])
    return tuple(c.get("Product_title", "") for c in candidates[:top_k])


def _run_stability_at(temp: float, queries: list[str], repeats: int, rerank_on: bool):
    """Returns query -> distinct top-10 orderings across repeats (1 == stable)."""
    orig_det, orig_ovr = cfg.DETERMINISTIC, cfg.TEMPERATURE_OVERRIDE
    cfg.DETERMINISTIC = False
    cfg.TEMPERATURE_OVERRIDE = temp
    out = {}
    try:
        for q in queries:
            seen = set()
            for _ in range(repeats):
                seen.add(_pipeline_titles(q, rerank_on))
            out[q] = len(seen)
    finally:
        cfg.DETERMINISTIC, cfg.TEMPERATURE_OVERRIDE = orig_det, orig_ovr
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3],
        help="Temperatures to sweep. Default: 0.0 0.1 0.2 0.3",
    )
    p.add_argument(
        "--rerank",
        action="store_true",
        help="Include the rerank stage (doubles LLM calls). Default off.",
    )
    p.add_argument(
        "--no-stability",
        action="store_true",
        help="Skip the run-to-run stability test (quality only).",
    )
    p.add_argument(
        "--stability-queries",
        type=int,
        default=3,
        help="How many eval queries to use for the stability test. Default 3.",
    )
    p.add_argument(
        "--stability-repeats",
        type=int,
        default=3,
        help="How many times to repeat each stability query. Default 3.",
    )
    args = p.parse_args()

    rerank_on = args.rerank

    print("\n" + "=" * 78)
    print("TEMPERATURE SWEEP")
    print(f"  temps         = {args.temps}")
    print(f"  rerank        = {rerank_on}  (off = translator-only, lighter on quota)")
    print(f"  seed          = DISABLED during sweep (so temperature is observable)")
    print("=" * 78)

    quality_rows = []
    for temp in args.temps:
        print(f"\n--- quality @ temperature={temp:.2f} ---")
        try:
            out_path, elapsed = _run_quality_at(temp, rerank_on)
            df = pd.read_csv(out_path)
            quality_rows.append(
                {
                    "temperature": temp,
                    "P@1": round(df["P@1"].mean(), 3),
                    "P@10": round(df["P@10"].mean(), 3),
                    "MRR": round(df["MRR"].mean(), 3),
                    "NDCG@10": round(df["NDCG@10"].mean(), 3),
                    "ms_total": round(df["ms_total"].mean(), 0),
                    "csv": out_path.name,
                }
            )
            print(f"   done in {elapsed:.1f}s -> {out_path.name}")
        except Exception as e:
            print(f"   FAILED at temp={temp}: {e!r}")

    stability_summary = {}
    if not args.no_stability:
        load_index()
        df_all = get_dataframe()
        with open(cfg.EVAL_QUERIES_JSON, encoding="utf-8") as f:
            all_queries = [q["query"] for q in json.load(f)]
        stab_queries = all_queries[: args.stability_queries]

        print("\n" + "=" * 78)
        print(
            f"STABILITY: {len(stab_queries)} quer(y/ies) x {args.stability_repeats} "
            f"repeats per temperature"
        )
        print("(distinct top-10 orderings; 1 = perfectly stable/deterministic)")
        print("=" * 78)
        for temp in args.temps:
            res = _run_stability_at(
                temp, stab_queries, args.stability_repeats, rerank_on
            )
            stability_summary[temp] = res
            avg = mean(res.values()) if res else 0
            print(f"\n  temperature={temp:.2f}  (avg distinct orderings: {avg:.2f})")
            for q, n in res.items():
                flag = "stable" if n == 1 else f"WOBBLES ({n} orderings)"
                print(f"    [{flag:>22}]  {q[:50]}")

    print("\n" + "=" * 78)
    print("QUALITY vs TEMPERATURE (means across eval set)")
    print("=" * 78)
    if quality_rows:
        qdf = pd.DataFrame(quality_rows)
        print(qdf.to_string(index=False))
        best = max(quality_rows, key=lambda r: (r["NDCG@10"], r["MRR"]))
        print(f"\n  Best NDCG@10 at temperature = {best['temperature']:.2f}")

    if stability_summary:
        print("\nSTABILITY vs TEMPERATURE (avg distinct top-10 orderings; 1 = best)")
        for temp in args.temps:
            if temp in stability_summary:
                vals = stability_summary[temp].values()
                print(f"  temp {temp:.2f}: {mean(vals):.2f}" if vals else f"  temp {temp:.2f}: n/a")

    print(
        "\nInterpretation: if temperature 0.0 ties for best quality AND shows "
        "1 ordering\nwhile higher temps wobble, then 0 is the right choice — "
        "not arbitrary.\n"
    )


if __name__ == "__main__":
    main()
