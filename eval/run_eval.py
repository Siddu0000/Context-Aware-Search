"""End-to-end retrieval eval: runs data/eval_queries.json and writes eval_results/*.csv."""

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

import app.config as cfg
import app.search as search_module
import app.translator as translator_module
from app import tracking
from app.config import EVAL_QUERIES_JSON, EVAL_RESULTS_DIR, FINAL_TOP_K, RETRIEVAL_TOP_K
from app.llm_client import active_model, param_downgrades, seed_sent, track_usage
from app.metrics import StageTimings, approx_tokens
from app.query_guard import sanitize_query
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
    """Resolve a query's relevance criteria into a set of matching Product_titles."""
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

    if "title_none_of" in criteria:
        # Excludes accessories that NAME the product: "roku remote", "airtag case",
        # "nightgown" for gown. Inferred-intent queries need it; keyword ones rarely do.
        for term in criteria["title_none_of"]:
            mask &= ~df["Product_title"].astype(str).str.lower().str.contains(
                re.escape(term.lower())
            )

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


USAGE_COLUMNS = ("llm_calls", "tokens_prompt", "tokens_completion",
                 "tokens_reasoning", "tokens_total")


def _stage_label(err: dict) -> str:
    """'rerank:partial' (as stress_test labels it): some entries were dropped, but the
    LLM's ranking WAS used, so it must not read like a full embedding-order fallback."""
    stage = str(err.get("stage"))
    return f"{stage}:partial" if err.get("code") == "partial" else stage


def is_degraded(fell_back) -> bool:
    """A fell_back value with any real fallback; ':partial' labels alone (rerank entries
    dropped, or the translator's query guard) still measure the LLM."""
    labels = str(fell_back or "").split(";")
    return any(lab and not lab.endswith(":partial") for lab in labels)


def _llm_facts(rerank_on: bool, translate_on: bool) -> dict:
    """What the backend ACTUALLY did: cfg can name a model or a seed the calls never had."""
    if not (translate_on or rerank_on):
        return {"llm_model_used": "off", "seed_sent": False}
    # None = the backend never built (e.g. no API key), so every LLM stage fell back
    return {"llm_model_used": active_model() or "unavailable", "seed_sent": seed_sent()}


def _downgrade_tags(model: str) -> dict:
    # Tags, not params: learned DURING the run. Per this run's model, because the
    # sets are process-wide and would otherwise blame a model compared earlier
    d = param_downgrades()
    return {k: str(model in models).lower() for k, models in d.items()}


def _print_downgrades(model: str) -> None:
    d = param_downgrades()
    if not any(d.values()):
        print("  downgrades   = none (JSON mode and temperature sent as requested)")
        return
    print(f"  downgrades   = json_mode_dropped={d['json_mode_dropped']} "
          f"temperature_dropped={d['temperature_dropped']}")
    hit = [k for k, models in d.items() if model in models]
    if hit:
        print(f"  !! {model} ran with {', '.join(hit)} -- not a failure (valid is "
              "unaffected), but not comparable with a run that had them.")


def _run_params(rerank_on: bool, fields: Tuple[str, ...], translate_on: bool) -> dict:
    """What produced these numbers: models, backend, modes. Read LIVE -- the
    comparison harnesses mutate cfg / translator_module between runs."""
    provider = cfg.LLM_PROVIDER
    backend = (cfg.SEARCH_BACKEND or "local").lower()
    return {
        "llm_provider": provider,
        "llm_model": {
            "openai": cfg.OPENAI_MODEL,
            "gemini": cfg.GEMINI_MODEL,
            "anthropic": cfg.ANTHROPIC_MODEL,
        }.get(provider, "?"),
        "search_backend": backend,
        "embedding_model": (
            cfg.VS_EMBEDDING_MODEL if backend == "databricks" else cfg.EMBEDDING_MODEL
        ),
        # vector = self-managed index + app-side query embedding (2026-10-09 on);
        # text = the old managed-embedding index. Runs of the two are not comparable.
        "vs_query_mode": cfg.VS_QUERY_MODE if backend == "databricks" else "n/a",
        "catalog": cfg.VS_TABLE if backend == "databricks" else cfg.PRODUCTS_CSV.name,
        # Content hash: the names above stay the same when the catalog text changes
        "catalog_fp": search_module.catalog_fingerprint(),
        "translator_mode": translator_module.TRANSLATOR_MODE if translate_on else "off",
        "rerank_on": rerank_on,
        "rerank_input_k": cfg.RERANK_INPUT_K,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        # The knob; seed_sent says whether it took effect (Databricks never gets a seed)
        "deterministic": cfg.DETERMINISTIC,
        **_llm_facts(rerank_on, translate_on),
        # What was EMBEDDED (the enriched tags join the default set), not just requested
        "fields": ",".join(search_module.embedded_fields() or fields),
        # Scores on different sets are NOT comparable (keyword set ~0.95 raw, context ~0.3)
        "eval_set": Path(EVAL_QUERIES_JSON).name,
        # The guard's output side can clip, screen or cap what the model wrote, so the
        # two arms of a guard on/off A/B must be told apart (the knob is in-process)
        "query_guard": cfg.QUERY_GUARD_ENABLED,
        "max_intents": cfg.MAX_INTENTS,
        # Same models + catalog + modes, different prompts: without these two a
        # pre/post prompt-edit pair differs only by timestamp in MLflow
        "prompts_sha": _prompts_sha(),
        "rerank_type_field": _rows_carry_type(),
    }


def _prompts_sha() -> str:
    """First 12 hex of sha256 over the six translator prompts, the rerank template and
    the assistant prompt, read LIVE (the harnesses swap prompts in-process)."""
    import hashlib

    import app.assistant as assistant_module
    import app.reranker as reranker_module

    texts = [getattr(translator_module, name) for name in (
        "QUERY_EXPANSION_PROMPT", "RECIPE_PROMPT", "HYDE_PROMPT", "HYBRID_PROMPT",
        "OUTFIT_PROMPT", "SETUP_PROMPT")]
    texts += [reranker_module.RERANK_PROMPT_TEMPLATE, assistant_module.ASSISTANT_PROMPT]
    # A separator no prompt contains: moving text across a boundary changes the digest
    return hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()[:12]


def _rows_carry_type() -> bool:
    """Whether retrieved rows carry product_type, i.e. reranker lines can show `type=`.

    Reads the loaded frame without loading one: called before load_index, it is False.
    """
    from app.product_type import TYPE_COLUMN

    df = getattr(search_module, "_df", None)
    return df is not None and TYPE_COLUMN in df.columns


def evaluate(
    rerank_on: bool,
    tag: str,
    fields: Tuple[str, ...] = DEFAULT_SEARCH_FIELDS,
    translate_on: bool = True,
) -> Path:
    """Run the eval set; CSV always, plus one MLflow run when tracking is on."""
    with tracking.eval_run(tag) as run:
        out_path, summary = _evaluate(rerank_on, tag, fields, translate_on)
        degraded = int(summary.get("n_degraded", 0))
        params = _run_params(rerank_on, fields, translate_on)
        run.log(
            params=params,
            metrics=summary,
            artifact=out_path if out_path.exists() else None,
            tags={"valid": str(degraded == 0).lower(), "tag": tag,
                  **_downgrade_tags(params["llm_model_used"])},
        )
    return out_path


def _evaluate(
    rerank_on: bool,
    tag: str,
    fields: Tuple[str, ...],
    translate_on: bool,
):
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
        # A silent LLM fallback makes a degraded run look like a clean one
        stage_errors: list = []

        # REAL provider-reported tokens; tokens_in_approx (len//4, input only)
        # badly understates a reasoning model and is kept only for old CSVs
        with track_usage() as usage:
            with timings.stage("translate"):
                if translate_on:
                    intents = translate_query(query, errors=stage_errors)
                    tokens_in += approx_tokens(query)
                else:
                    intents = [query]

            with timings.stage("retrieve"):
                candidates = search_products(intents, top_k=RETRIEVAL_TOP_K)

            # As app/main.py does: the reranker sees only what the query guard lets an
            # LLM see, and none of a query it refused ("" -> embedding order)
            llm_query = sanitize_query(query).clean
            if rerank_on and candidates and llm_query:
                with timings.stage("rerank"):
                    final = llm_rerank(
                        llm_query, candidates, top_k=FINAL_TOP_K, errors=stage_errors
                    )
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
                "llm_calls": usage["calls"],
                "tokens_prompt": usage["prompt_tokens"],
                "tokens_completion": usage["completion_tokens"],
                "tokens_reasoning": usage["reasoning_tokens"],
                "tokens_total": usage["total_tokens"],
                "fell_back": ";".join(_stage_label(e) for e in stage_errors),
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
    degraded = [r for r in rows if is_degraded(r["fell_back"])]
    partial = [r for r in rows if r["fell_back"] and not is_degraded(r["fell_back"])]
    if degraded:
        print(f"  !! {len(degraded)}/{len(rows)} QUERIES RAN DEGRADED "
              f"({ {r['fell_back'] for r in degraded} }) -- these metrics do NOT "
              "measure the full pipeline. Fix the cause before quoting them.")
    if partial:
        # Still the LLM's output (valid): rerank entries were malformed, or the query
        # guard trimmed the query / screened the model's phrases (translate:partial)
        stages = sorted({lab for r in partial for lab in r["fell_back"].split(";")
                         if lab.endswith(":partial")})
        print(f"  !  {len(partial)}/{len(rows)} queries had a PARTIAL stage "
              f"({', '.join(stages)}: rerank entries dropped, or the translator's query "
              "guard applied; the LLM's output was still used -- not a fallback)")
        if "translate:partial" in stages:
            # Eval queries are ordinary shopper text: the guard should never fire on them
            print("  !  translate:partial on an eval query is a guard false positive or a "
                  "screened reply (the 'Translator guard' log line names which): "
                  "investigate before quoting these numbers")
    summary = {"n_queries": len(rows), "n_degraded": len(degraded),
               "n_partial": len(partial)}
    if rows:
        for col in ("P@1", "P@5", "P@10", "R@10", "MRR", "NDCG@10", "ms_total"):
            summary[f"mean_{col}"] = mean(r[col] for r in rows)
            print(f"  mean_{col:9s} = {summary[f'mean_{col}']:.3f}")
        for col in USAGE_COLUMNS:
            summary[f"mean_{col}"] = mean(r[col] for r in rows)
            summary[f"sum_{col}"] = sum(r[col] for r in rows)
        print(f"  llm_calls    = {summary['sum_llm_calls']} "
              f"({summary['mean_llm_calls']:.1f}/query)")
        # Databricks FM APIs bill reasoning INSIDE completion_tokens without breaking
        # it out, so 0 there means "not reported", never "the model did not reason"
        reasoning = (f"of which reasoning {summary['sum_tokens_reasoning']:,}"
                     if summary["sum_tokens_reasoning"]
                     else "reasoning 0: not broken out, or a non-reasoning model")
        print(f"  tokens       = {summary['sum_tokens_total']:,} total "
              f"({summary['mean_tokens_total']:,.0f}/query; prompt "
              f"{summary['sum_tokens_prompt']:,}, completion "
              f"{summary['sum_tokens_completion']:,}, {reasoning})")
        if summary["sum_llm_calls"] and not summary["sum_tokens_total"]:
            print("  !! provider returned no usage data -- token counts are 0, not free")
    llm = _llm_facts(rerank_on, translate_on)
    print(f"  llm          = {llm['llm_model_used']} "
          f"(seed sent: {'yes' if llm['seed_sent'] else 'no'})")
    _print_downgrades(llm["llm_model_used"])
    print(f"  query_guard  = {cfg.QUERY_GUARD_ENABLED} (max_intents={cfg.MAX_INTENTS})")
    print(f"  rerank_on    = {rerank_on}")
    print(f"  fields       = {search_module.embedded_fields() or fields}")
    print(f"  saved to     = {out_path}")
    return out_path, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-rerank", action="store_true", help="Disable LLM rerank.")
    p.add_argument(
        "--exclude-description",
        action="store_true",
        help="Exclude prod_description from search text — diagnoses label leakage.",
    )
    p.add_argument("--tag", default=None, help="Filename tag for the output CSV.")
    p.add_argument("--queries", default=None,
                   help="Eval set JSON, e.g. data/eval_queries_context.json.")
    p.add_argument("--no-translate", action="store_true",
                   help="Search the raw query: the no-LLM control.")
    args = p.parse_args()
    if args.queries:
        global EVAL_QUERIES_JSON
        EVAL_QUERIES_JSON = cfg.eval_queries_path(args.queries)

    fields = DEFAULT_SEARCH_FIELDS
    if args.exclude_description:
        fields = tuple(f for f in DEFAULT_SEARCH_FIELDS if f != "prod_description")

    rerank_on = not args.no_rerank
    default_tag = f"{'rerank_on' if rerank_on else 'rerank_off'}"
    if args.exclude_description:
        default_tag += "_no_desc"
    tag = args.tag or default_tag
    if args.no_translate:
        tag += "_no_translate"
    evaluate(rerank_on=rerank_on, tag=tag, fields=fields, translate_on=not args.no_translate)


if __name__ == "__main__":
    main()
