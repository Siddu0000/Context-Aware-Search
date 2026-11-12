"""Score a run of the complex query bank by group (2026-11-02, the Databricks stress test).

data/complex_queries.json holds 107 shopper-style queries (fashion, grocery, electronics,
13 multi-department baskets, 7 brand / noun collisions, 8 deliberate catalog gaps), in
eval/eval_page_types.py's own query format, so databricks/10_page_eval.py runs it as is:
    run(mode="on", queries="data/complex_queries.json")
That run prints ONE "Mean over all scored rows", which mixes rows that want opposite
things: on 12 accessory or owner-need rows the device itself is the WRONG answer
(core_top12 should be ~0), on 10 device asks it is the right one (>= 0.8). And its
vertical columns score each card alone, so a page of only popcorn passes a "snacks and a
streaming stick" basket. This reads the run's rows plus data/complex_queries_meta.json
(joined by query text) and reports, per arm:
  1. per-group means of the scored columns;
  2. core_top4 / core_top12 split by the meta's core_target, with pass counts;
  3. the dept_min rule for the multi-department rows, from `types_page` (a Fashion type
     or 'unknown' counts as Fashion, 'electronics' / 'grocery' as that vertical): a flat
     page needs N of its first 12 cards per department, a bundle M slots;
  4. validity: degraded rows, partial stages, misrouted bundles, memo hits (must be 0),
     and the weak-match banner on the gap rows.
Stdlib only and no app import, so it runs wherever the CSV is, a notebook included:
    csv_path, rows = run(mode="on", queries="data/complex_queries.json")
    from eval import score_complex_queries as scq
    print(scq.render(scq.score(rows)))
or from a shell:
    python -m eval.score_complex_queries <page_types_*.csv> [--meta PATH] [--json OUT]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

META_JSON = Path(__file__).resolve().parent.parent / "data" / "complex_queries_meta.json"
GROUPS = ("complex_fashion", "complex_grocery", "complex_electronics", "complex_cross",
          "complex_collision")
COLS = ("mismatch_top4", "mismatch_top12", "context_fit_top4", "context_fit_top12",
        "vertical_top4", "vertical_top12", "wrong_gender_top12", "P@4", "P@12",
        "core_top4", "core_top12", "match_weak", "items_bad", "items_good")
# eval_page_types.type_label: a Fashion row shows its product type, any other row its vertical
FASHION_TOKENS = {"apparel", "footwear", "jewelry", "bag", "headwear", "accessory", "unknown"}
PAGE = 12


def _num(value):
    """A CSV cell ('', 'True', '0.25') or an in-memory value (None, bool, float) as a
    float, or None when the column was not scored for that row."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if value in ("True", "False"):
        return 1.0 if value == "True" else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    values = [v for v in values if v is not None]
    return round(mean(values), 3) if values else None


def _is_false(value) -> bool:
    # bundle_ok is a bool in run()'s rows and the string 'False' in its CSV
    return value is False or value == "False"


def load_meta(path=None) -> dict:
    """{query text: meta entry}; the bank and its meta share the query text and order."""
    with open(path or META_JSON, encoding="utf-8") as f:
        return {m["query"]: m for m in json.load(f)}


def department(token: str) -> str:
    token = token.strip().casefold()
    if token in FASHION_TOKENS:
        return "Fashion"
    return {"electronics": "Electronics", "grocery": "Grocery"}.get(token, token or "?")


def dept_check(row: dict, rule: dict) -> dict:
    """Does the served page reach every named department? A page that serves one
    department only is the failure the multi-department rows exist to catch."""
    tokens = [t for t in str(row.get("types_page") or "").split(",") if t.strip()][:PAGE]
    counts = defaultdict(int)
    for token in tokens:
        counts[department(token)] += 1
    bundle = bool(row.get("bundle_type"))
    need = rule["bundle_min_slots_each"] if bundle else rule["flat_min_cards_each"]
    short = {d: counts.get(d, 0) for d in rule["departments"] if counts.get(d, 0) < need}
    return {"bundle": bundle, "need_each": need, "counts": dict(counts), "pass": not short,
            "short": short}


def _core_ok(target: str, value: float) -> bool:
    # '0': at most 1 device card of 12 (the expects say '0' or '~0'); '>=0.8': device asks
    return value <= 1 / PAGE + 1e-9 if target == "0" else value >= 0.8


def score(rows, meta=None) -> dict:
    """The report for one run: `rows` are eval_page_types rows (run()'s return value or
    its CSV read with csv.DictReader)."""
    meta = meta if meta is not None else load_meta()
    rows = list(rows)
    report = {"rows": len(rows),
              "unmatched_queries": sorted({r["query"] for r in rows if r["query"] not in meta}),
              "arms": {}}
    rows = [r for r in rows if r["query"] in meta]
    for arm in sorted({str(r.get("arm") or "") for r in rows}):
        mine = [r for r in rows if str(r.get("arm") or "") == arm]
        rep = {"runs": len(mine)}

        by_group = defaultdict(list)
        for r in mine:
            by_group[meta[r["query"]]["group"]].append(r)
        rep["by_group"] = {
            g: {"runs": len(rs), **{c: m for c in COLS
                                    if (m := _mean(_num(r.get(c)) for r in rs)) is not None}}
            for g, rs in sorted(by_group.items())}

        core = {}
        for target in ("0", ">=0.8"):
            rs = [r for r in mine if meta[r["query"]].get("core_target") == target
                  and _num(r.get("core_top12")) is not None]
            core[target] = {
                "runs": len(rs),
                "core_top4": _mean(_num(r.get("core_top4")) for r in rs),
                "core_top12": _mean(_num(r.get("core_top12")) for r in rs),
                "pass_core_top12": sum(_core_ok(target, _num(r["core_top12"])) for r in rs),
                "ids_failing": sorted({meta[r["query"]]["id"] for r in rs
                                       if not _core_ok(target, _num(r["core_top12"]))})}
        rep["core_by_target"] = core

        checks = {}
        for r in mine:
            rule = meta[r["query"]].get("dept_min")
            # a fallen-back page is a different pipeline: it is listed under validity
            if rule and not r.get("fell_back"):
                checks.setdefault(meta[r["query"]]["id"], []).append(dept_check(r, rule))
        rep["dept_min"] = {
            "rows_checked": len(checks),
            "runs": sum(len(cs) for cs in checks.values()),
            "runs_pass": sum(c["pass"] for cs in checks.values() for c in cs),
            "short": {qid: [c["short"] for c in cs if not c["pass"]]
                      for qid, cs in sorted(checks.items()) if not all(c["pass"] for c in cs)}}

        rep["validity"] = {
            "degraded": sorted({meta[r["query"]]["id"] for r in mine if r.get("fell_back")}),
            "partial": sorted({meta[r["query"]]["id"] for r in mine if r.get("partial")}),
            "misrouted": sorted({f"{meta[r['query']]['id']}->{r.get('bundle_type') or 'flat'}"
                                 for r in mine if _is_false(r.get("bundle_ok"))}),
            "memo_hits": sum(int(_num(r.get("memo_hits")) or 0) for r in mine),
            "gap_rows_weak_banner": {meta[r["query"]]["id"]: r.get("match_weak")
                                     for r in mine if meta[r["query"]].get("gap")}}
        report["arms"][arm] = rep
    return report


def _cell(value) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def render(report: dict) -> str:
    """The report as text: read the validity line first, as with every eval here."""
    out = [f"complex query bank: {report['rows']} rows"]
    if report["unmatched_queries"]:
        out.append(f"  !! {len(report['unmatched_queries'])} queries not in the meta "
                   "(another bank?): ignored")
    shown = ("context_fit_top12", "vertical_top12", "mismatch_top12", "wrong_gender_top12",
             "P@12", "match_weak")
    for arm, rep in report["arms"].items():
        v = rep["validity"]
        out.append(f"\narm {arm or '?'}: {rep['runs']} runs | degraded {len(v['degraded'])} "
                   f"{v['degraded'] or ''} | partial {len(v['partial'])} | misrouted "
                   f"{len(v['misrouted'])} {v['misrouted'] or ''} | memo hits {v['memo_hits']}")
        if v["degraded"] or v["memo_hits"]:
            out.append("  !! degraded rows or memo hits: those rows do NOT measure the pipeline")
        out.append(f"  {'group':20s} runs  " + "  ".join(f"{c[:12]:>12s}" for c in shown))
        for group, g in rep["by_group"].items():
            out.append(f"  {group:20s} {g['runs']:4d}  "
                       + "  ".join(f"{_cell(g.get(c)):>12s}" for c in shown))
        for target, c in rep["core_by_target"].items():
            out.append(f"  core_target {target:6s}: {c['runs']} runs, core4 {_cell(c['core_top4'])},"
                       f" core12 {_cell(c['core_top12'])}, pass {c['pass_core_top12']}/{c['runs']}"
                       + (f" (failing: {', '.join(c['ids_failing'])})" if c["ids_failing"] else ""))
        d = rep["dept_min"]
        out.append(f"  dept_min: {d['runs_pass']}/{d['runs']} runs pass on {d['rows_checked']} "
                   "multi-department rows"
                   + (f" (short: {json.dumps(d['short'])})" if d["short"] else ""))
        out.append(f"  gap rows, weak banner: {json.dumps(v['gap_rows_weak_banner'])}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("csv", help="a page_types_*.csv written by eval_page_types / 10_page_eval")
    ap.add_argument("--meta", default=None, help=f"default {META_JSON.name} in data/")
    ap.add_argument("--json", default=None, help="also write the full report here")
    args = ap.parse_args(argv)
    with open(args.csv, encoding="utf-8", newline="") as f:
        report = score(csv.DictReader(f), load_meta(args.meta))
    print(render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
