"""Calibrate MIN_RESULT_RELEVANCE for whichever retrieval backend is active.

MIN_RESULT_RELEVANCE is an ABSOLUTE floor on the top retrieval score: below it,
/search answers "no matching products". It was tuned on gte-small cosine. A
different embedding model or Vector Search's scoring puts scores on another
scale, so a stale floor either hides every result or never fires.

Measures the top score for queries that SHOULD match (the eval set) and for
real-but-out-of-catalog queries that should NOT, then recommends a floor that
never blocks a real query. Blocking a real query is the expensive mistake; a
missed out-of-catalog query still gets reranked and usually looks fine.

Usage:
    python -m eval.calibrate_relevance            # raw query, no LLM calls
    python -m eval.calibrate_relevance --translate  # faithful: same intents /search uses
"""

import argparse
import json
import logging
import statistics

import app.config as cfg
from app.config import RETRIEVAL_TOP_K
from app.llm_client import configured_model
from app.search import best_score, search_products

logging.basicConfig(level=logging.WARNING)

# Real, shoppable-sounding requests for things this catalog does not carry
# (it is Amazon Fashion + Electronics + Grocery). Avoid anything plausibly sold
# in those three: reading glasses, pet food and phone cases would all be traps.
OUT_OF_CATALOG = [
    "car engine oil 5w-30",
    "lawn mower replacement blade",
    "motorcycle tyres",
    "wooden dining table for six",
    "bags of cement for construction",
    "kitchen sink faucet",
    "life insurance quote",
    "hotel booking in paris",
    "flight tickets to london",
    "piano lessons for beginners",
]

# Only meaningful in raw mode: the translator returns [] for these, which
# /search already treats as a clean no-match before any threshold applies
GIBBERISH = ["asdfghjkl", "zxqv wprt", "qqqqqq"]


def _top_score(query: str, translate: bool, errors: list | None = None):
    if translate:
        from app.translator import understand_query

        # A fallback to the raw query would score raw retrieval as if it were /search
        intents = understand_query(query, errors=errors)["intents"]
        if not intents:
            return None     # judged gibberish upstream; no threshold involved
    else:
        intents = [query]
    return best_score(search_products(intents, top_k=RETRIEVAL_TOP_K))


def _score_all(queries, translate: bool, degraded: dict) -> dict:
    """query -> top score for CLEAN runs only; fallbacks go to `degraded` instead."""
    scores = {}
    for q in queries:
        errs: list = []
        s = _top_score(q, translate, errs)
        # A 'partial' (query guard screened or clipped a phrase) still scored the LLM's
        # intents, as run_eval.is_degraded treats it; only a real fallback is excluded
        blocking = [e for e in errs if e.get("code") != "partial"]
        if blocking:
            degraded[q] = ";".join(str(e.get("stage")) for e in blocking)
        else:
            scores[q] = s
    return scores


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return (f"min {min(xs):.3f}  median {statistics.median(xs):.3f}  "
            f"max {max(xs):.3f}  (n={len(xs)})")


def run(translate: bool = False, margin: float = 0.02, queries: str | None = None) -> dict:
    # Resolve at CALL time: a notebook usually sets EVAL_QUERIES_JSON after importing config
    path = cfg.eval_queries_path(queries)
    with open(path, "r", encoding="utf-8") as f:
        in_domain = [q["query"] for q in json.load(f)]

    backend = (cfg.SEARCH_BACKEND or "local").lower()
    # The floor depends on the translator's wording, so the result is per-LLM
    llm = (configured_model() or cfg.LLM_PROVIDER) if translate else "off"
    print(f"\n=== MIN_RESULT_RELEVANCE calibration ===")
    print(f"backend={backend}  translate={translate}  llm={llm}  "
          f"current floor={cfg.MIN_RESULT_RELEVANCE}  set={path.name}\n")

    degraded: dict = {}
    ins = _score_all(in_domain, translate, degraded)
    outs = _score_all(OUT_OF_CATALOG, translate, degraded)
    gib = {} if translate else {q: _top_score(q, False) for q in GIBBERISH}

    if degraded:
        print(f"!! {len(degraded)}/{len(in_domain) + len(OUT_OF_CATALOG)} QUERIES RAN "
              "DEGRADED (an LLM stage fell back to the raw query). They are EXCLUDED "
              "from the stats and the recommendation -- fix the cause and re-run:")
        for q, stages in degraded.items():
            print(f"     [{stages}] {q}")
        print()

    print("SHOULD match (eval set):   ", _stats(ins.values()))
    print("should NOT (out-of-catalog):", _stats(outs.values()))
    if gib:
        print("gibberish (raw):            ", _stats(gib.values()))

    real = [s for s in ins.values() if s is not None]
    floor = round(min(real) - margin, 3) if real else None
    caught = sum(1 for s in outs.values() if s is not None and floor and s < floor)
    blocked_now = sum(1 for s in real if s < cfg.MIN_RESULT_RELEVANCE)

    print()
    print(f"current floor {cfg.MIN_RESULT_RELEVANCE}: blocks {blocked_now}/{len(real)} REAL queries"
          + ("   <-- would hide real results" if blocked_now else ""))
    if floor is not None:
        print(f"recommended   {floor}: blocks 0/{len(real)} real queries, "
              f"rejects {caught}/{len(outs)} out-of-catalog"
              + (f"   (from CLEAN queries only; {len(degraded)} degraded excluded)"
                 if degraded else ""))
        top_out = max((s for s in outs.values() if s is not None), default=None)
        if top_out is not None and top_out >= min(real):
            print("  note: score ranges OVERLAP -- no floor separates them. Out-of-catalog "
                  "rejection then rests on the translator and the reranker, not this knob.")
    else:
        print("no recommendation: no real query produced a clean score")

    worst = sorted(((s, q) for q, s in ins.items() if s is not None))[:3]
    print("\nlowest-scoring REAL queries (these set the floor):")
    for s, q in worst:
        print(f"  {s:.3f}  {q}")
    # in_domain / out_of_catalog hold clean scores only; degraded maps query -> stage
    return {"recommended": floor, "in_domain": ins, "out_of_catalog": outs,
            "gibberish": gib, "degraded": degraded, "llm": llm}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--translate", action="store_true",
                   help="Score the translator's intents (faithful to /search; costs LLM calls).")
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--queries", default=None,
                   help="Eval set JSON (default: EVAL_QUERIES_JSON env, else the keyword set).")
    a = p.parse_args()
    run(translate=a.translate, margin=a.margin, queries=a.queries)


if __name__ == "__main__":
    main()
