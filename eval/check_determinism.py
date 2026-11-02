"""Is the same search the same page? A determinism check over app.main.search (2026-10-27).

The Oct 27 meeting asked that the same search give the same results. Every deterministic
stage was measured stable with a fake LLM; the variation is the LLM's, which the reply
memo (app/llm_memo.py, LLM_MEMO) pins. This script checks that end to end, on the code
path the API serves: for each query, --runs cold runs (page_cache, the degraded store and
the /product cache cleared before each) of

    app.main.search  page 1 and page 2 (12 cards, rerank, ads and filters as the UI sends)
    app.main.product the page-1 top card, with its recommendations

and compares the catalog_index lists of both pages, the intents, constraints, bundle type,
facets, match quality, the sponsored labels and the product page's cross-sell / upsell.

--memo on  (default) asserts identical snapshots on every run and 0 provider calls on runs
           2..N (all replies replayed). Exit code 1 on any difference.
--memo off reports the model's raw stability instead: page-1 overlap@12, top-4 overlap,
           Kendall tau on the shared cards, identical intents, the fallback rate.
--subprocess runs run 1 in a FRESH Python process (another PYTHONHASHSEED; stands in for
           an API / App restart) and runs 2..N here, so only one catalog is in memory at
           a time.

The memo lives in --memo-dir: a new temp dir by default, never the app's LLM_MEMO_DIR and
never the Databricks App's memo volume (/Volumes/<catalog>/<schema>/app_state: a check must
not pre-warm, or read, what the API or the App serves); any other /Volumes/... path only
with --allow-volume. A query whose run 1 fell back (an LLM stage failed) is "not pinned":
retried up to --attempts times, never compared.

COST: real LLM calls. Run 1 makes ~3 per query (translate, rerank, cross-sell); with
--memo off every run does. Not run by the tests: tests/test_determinism.py drives these
functions with a fake LLM. Never logs or saves reply text; --out writes snapshots
(catalog indexes, intents, facets) for a later diff.

    python -m eval.check_determinism --runs 3 --subprocess
    python -m eval.check_determinism --memo off --runs 2 --query "black dress"
"""

import argparse
import json
import os
import posixpath
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import app.config as cfg

PAGE_SIZE = 12
HEAD_K = 4
# The meeting's queries (Oct 27: the watch gift, plus the two the determinism audit ran
# live), then the first DEFAULT_PAGE_TYPE_ROWS rows of data/page_type_queries.json
MEETING_QUERIES = ("gift for a vintage watch collector", "black dress", "power bank for camping")
DEFAULT_PAGE_TYPE_ROWS = 18
_SNAPSHOT_FIELDS = ("intents", "constraints", "bundle_type", "page1", "page2", "variants1",
                    "sponsored1", "facets", "match_quality", "total_results", "product")


def default_queries() -> list:
    out = list(MEETING_QUERIES)
    path = cfg.DATA_DIR / "page_type_queries.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        out += [r["query"] for r in rows[:DEFAULT_PAGE_TYPE_ROWS]]
    return list(dict.fromkeys(out))


def load_queries(path: Optional[str], extra: list) -> list:
    """--queries FILE (a JSON list of strings or of objects with "query") + --query TEXT."""
    out = []
    if path:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = rows.get("queries", rows) if isinstance(rows, dict) else rows
        out += [r["query"] if isinstance(r, dict) else str(r) for r in rows]
    out += list(extra or [])
    return list(dict.fromkeys(out)) or default_queries()


# ------------------------------------------------------------------ one run
def _ci(row) -> Optional[int]:
    value = row.get("catalog_index")
    return int(value) if value is not None else None


def _blocking(errors) -> list:
    # main._blocking_errors: a 'partial' stage still used the LLM's answer
    return [e for e in errors or [] if e.get("code") != "partial"]


def _labels(errors) -> str:
    return ";".join(sorted({f"{e.get('stage')}:{e.get('code')}" for e in errors or []}))


def _search_args() -> dict:
    """What /chat passes: search() is an endpoint function, and the filter / sort params
    it is not given read as unset (main._query_default), as the UI's first fetch sends."""
    return dict(top_k=PAGE_SIZE, rerank=cfg.RERANK_ENABLED, recommend=False,
                sponsored=cfg.SPONSORED_ENABLED, translator_mode=None, follow_up=False)


def cold(main_module) -> None:
    """A fresh start for the in-process caches (the memo is NOT cleared: that is the test)."""
    from app import page_cache

    page_cache.clear()
    main_module.clear_degraded_pools()


def snapshot(main_module, query: str, product: bool = True) -> dict:
    """One cold run of `query`: what a shopper sees, as comparable plain data."""
    from app.llm_client import track_usage

    cold(main_module)
    args = _search_args()
    with track_usage() as usage:
        p1 = main_module.search(query=query, page=1, **args)
        # page 2 the way the UI asks for it: a follow-up of results already shown
        p2 = main_module.search(query=query, page=2, **{**args, "follow_up": True})
        top = next((_ci(r) for r in p1.get("results") or [] if _ci(r) is not None), None)
        prod = None
        if product and top is not None:
            prod = main_module.product(catalog_index=top, query=query, recommend=True)
    errors = list(p1.get("errors") or []) + list((prod or {}).get("errors") or [])
    recs = (prod or {}).get("recommendations") or {}
    return {
        "query": query,
        "intents": list(p1.get("interpreted_as") or []),
        "constraints": list(p1.get("constraints") or []),
        "bundle_type": p1.get("bundle_type"),
        "page1": [_ci(r) for r in p1.get("results") or []],
        "page2": [_ci(r) for r in p2.get("results") or []],
        # collapsed colour / size variants and bundle-slot options, per card
        "variants1": [[_ci(v) for v in (r.get("variants") or r.get("alternatives") or [])]
                      for r in p1.get("results") or []],
        "sponsored1": [bool(r.get("is_sponsored")) for r in p1.get("results") or []],
        "facets": p1.get("facets"),
        "match_quality": p1.get("match_quality"),
        "total_results": p1.get("total_results"),
        "product": None if prod is None else {
            "catalog_index": top,
            "cross_sell": [_ci(r) for r in recs.get("cross_sell") or []],
            "upsell": [_ci(r) for r in recs.get("upsell") or []],
            "source": recs.get("cross_sell_source"),
        },
        "errors": _labels(errors),
        "fell_back": bool(_blocking(errors)),
        "llm_calls": usage["calls"],
        "memo_hits": usage.get("memo_hits", 0),
    }


def first_run(main_module, query: str, attempts: int, product: bool = True) -> dict:
    """Run 1, retried while an LLM stage fell back: a fallback is never pinned."""
    snap = snapshot(main_module, query, product)
    tries = 1
    while snap["fell_back"] and tries < attempts:
        snap = snapshot(main_module, query, product)
        tries += 1
    snap["attempts"] = tries
    return snap


# ------------------------------------------------------------------ comparison
def _canon(value):
    # JSON round trip: tuples vs lists and dict key order never count as a difference
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def differences(a: dict, b: dict) -> list:
    """The snapshot fields that differ, in _SNAPSHOT_FIELDS order."""
    return [f for f in _SNAPSHOT_FIELDS if _canon(a.get(f)) != _canon(b.get(f))]


def kendall_tau(a: list, b: list) -> Optional[float]:
    """Kendall tau-a between two orderings, over the items both contain (None if < 2)."""
    shared = [x for x in a if x in set(b)]
    n = len(shared)
    if n < 2:
        return None
    pos = {x: i for i, x in enumerate(b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos[shared[i]] < pos[shared[j]]:
                concordant += 1
            else:
                discordant += 1
    return round((concordant - discordant) / (n * (n - 1) / 2), 3)


def stability(a: dict, b: dict) -> dict:
    """How far two runs of one query agree on page 1 (the --memo off report)."""
    p1a, p1b = a.get("page1") or [], b.get("page1") or []
    return {
        "overlap12": len(set(p1a[:PAGE_SIZE]) & set(p1b[:PAGE_SIZE])),
        "top4": len(set(p1a[:HEAD_K]) & set(p1b[:HEAD_K])),
        "tau": kendall_tau(p1a[:PAGE_SIZE], p1b[:PAGE_SIZE]),
        "same_top1": bool(p1a) and bool(p1b) and p1a[0] == p1b[0],
        "same_intents": _canon(a.get("intents")) == _canon(b.get("intents")),
    }


def judge(runs: list, memo: bool) -> dict:
    """Verdict for one query's runs (run 1 first). With the memo on: every run equals run 1
    and runs 2..N made 0 provider calls. A run-1 fallback is not judged ("not pinned")."""
    base = runs[0]
    if base["fell_back"]:
        return {"query": base["query"], "status": "not pinned: fell back",
                "errors": base["errors"], "diffs": []}
    diffs = []
    calls_after = [r["llm_calls"] for r in runs[1:]]
    for i, r in enumerate(runs[1:], start=2):
        d = differences(base, r)
        if d:
            diffs.append((i, d))
    out = {"query": base["query"], "diffs": diffs, "calls_after_run1": calls_after,
           "pairs": [stability(base, r) for r in runs[1:]]}
    if memo:
        ok = not diffs and all(c == 0 for c in calls_after)
        out["status"] = "PASS" if ok else "FAIL"
    else:
        out["status"] = "identical" if not diffs else "differs"
    return out


# ------------------------------------------------------------------ the check
def _volume_key(path: str) -> Optional[tuple]:
    """A /Volumes path as comparable parts (catalog, schema, volume, *folders), or None for
    any other path. '.', '..' and repeated slashes are resolved, as the FUSE mount resolves
    them, and the three Unity Catalog names are casefolded: UC names are case-insensitive
    (databricks/12's scratch_root refuses 'APP_STATE' for that reason), so a plain string
    compare let /Volumes/dev/cas/App_State/... or .../evals/../app_state/... past the
    refusal. The folders below the volume keep their case."""
    text = str(path)
    if not text.startswith("/"):
        return None
    parts = posixpath.normpath("/" + text.lstrip("/")).strip("/").split("/")
    if parts[0].casefold() != "volumes":
        return None
    return tuple(p.casefold() for p in parts[1:4]) + tuple(parts[4:])


def _same_place(a: str, b: str) -> bool:
    va, vb = _volume_key(a), _volume_key(b)
    if va is not None or vb is not None:
        return va == vb

    def resolved(x):
        path = Path(x)
        return (path if path.is_absolute() else cfg.PROJECT_ROOT / path).resolve()

    return resolved(a) == resolved(b)


def app_memo_dir() -> str:
    """Where the Databricks App keeps its memo (databricks/app_bundle/app.yaml):
    /Volumes/<CAS_CATALOG>/<CAS_SCHEMA>/app_state/llm_memo. In a notebook the local
    LLM_MEMO_DIR is the default one, so this path is refused by name."""
    return f"/Volumes/{cfg.CAS_CATALOG}/{cfg.CAS_SCHEMA}/app_state/llm_memo"


def _refusal(chosen: str, allow_volume: bool) -> Optional[str]:
    """Why `chosen` may not hold the check's memo, or None."""
    from app import llm_memo

    if _same_place(chosen, llm_memo.location()):
        return f"--memo-dir is the app's own LLM_MEMO_DIR ({chosen})"
    app_state = app_memo_dir().rsplit("/", 1)[0]          # the whole app_state volume
    volume = _volume_key(chosen)
    if volume is not None and volume[:3] == _volume_key(app_state):
        return f"--memo-dir is in the App's memo volume ({app_state})"
    if volume is not None and not allow_volume:
        # A shared volume: another check, a notebook or an App may read it (pass
        # --allow-volume for a scratch volume of your own)
        return f"--memo-dir is on a UC Volume ({chosen}); pass --allow-volume"
    return None


def configure_memo(memo: bool, memo_dir: Optional[str],
                   allow_volume: bool = False) -> Optional[str]:
    """Point the app's memo at a scratch dir (never the app's own, never the App's memo
    volume, a /Volumes path only with allow_volume) and switch it."""
    from app import llm_memo

    if memo:
        chosen = memo_dir or tempfile.mkdtemp(prefix="cas_memo_check_")
        why = _refusal(str(chosen), allow_volume)
        if why:
            raise SystemExit(f"{why}: refused. A check must not read or pre-warm what the "
                             "API or the App serves.")
        cfg.LLM_MEMO_DIR = str(chosen)
    cfg.LLM_MEMO = memo
    llm_memo.reset()
    return cfg.LLM_MEMO_DIR if memo else None


def _child_run(queries: list, memo: bool, memo_dir: Optional[str], attempts: int,
               product: bool, allow_volume: bool = False) -> list:
    """Run 1 in a fresh interpreter (another hash seed): the 'restart' case."""
    with tempfile.TemporaryDirectory(prefix="cas_det_child_") as tmp:
        out = Path(tmp) / "run1.json"
        qfile = Path(tmp) / "queries.json"
        qfile.write_text(json.dumps(queries), encoding="utf-8")
        cmd = [sys.executable, "-X", "utf8", "-m", "eval.check_determinism", "--emit", str(out),
               "--queries", str(qfile), "--memo", "on" if memo else "off",
               "--attempts", str(attempts)]
        if memo_dir:
            cmd += ["--memo-dir", memo_dir]
        if not product:
            cmd.append("--no-product")
        if allow_volume:
            cmd.append("--allow-volume")
        env = dict(os.environ, PYTHONHASHSEED="4242")
        subprocess.run(cmd, check=True, cwd=str(cfg.PROJECT_ROOT), env=env)
        return json.loads(out.read_text(encoding="utf-8"))


def run_check(queries: list, runs: int = 3, memo: bool = True, memo_dir: Optional[str] = None,
              in_subprocess: bool = False, attempts: int = 2, product: bool = True,
              main_module=None, allow_volume: bool = False) -> list:
    """One verdict per query (judge() + its runs). `main_module` is for the tests.

    The memo knobs are restored afterwards: called in-process (a notebook, the tests), the
    check never leaves the app pointed at its scratch memo."""
    from app import llm_memo

    saved = (cfg.LLM_MEMO, cfg.LLM_MEMO_DIR)
    try:
        memo_dir = configure_memo(memo, memo_dir, allow_volume)
        if in_subprocess:
            firsts = _child_run(queries, memo, memo_dir, attempts, product, allow_volume)
        if main_module is None:
            import app.main as main_module
        if not in_subprocess:
            firsts = [first_run(main_module, q, attempts, product) for q in queries]
        verdicts = []
        for base in firsts:
            rest = [] if base["fell_back"] else [
                snapshot(main_module, base["query"], product) for _ in range(runs - 1)]
            verdicts.append({**judge([base, *rest], memo), "runs": [base, *rest],
                             "memo_dir": memo_dir})
        return verdicts
    finally:
        cfg.LLM_MEMO, cfg.LLM_MEMO_DIR = saved
        llm_memo.reset()


def print_report(verdicts: list, memo: bool) -> int:
    print(f"\n=== determinism check: {len(verdicts)} queries, memo "
          f"{'on (expect identical, 0 calls after run 1)' if memo else 'off (model stability)'}")
    failed = 0
    for v in verdicts:
        q = v["query"][:46]
        if v["status"].startswith("not pinned"):
            print(f"  {q:46s} NOT PINNED: run 1 fell back ({v['errors']})")
            continue
        runs = v["runs"]
        calls = ",".join(str(r["llm_calls"]) for r in runs)
        hits = ",".join(str(r.get("memo_hits", 0)) for r in runs)
        line = f"  {q:46s} {v['status']:9s} calls/run={calls} memo_hits/run={hits}"
        if v["diffs"]:
            line += "  first diff: " + "; ".join(f"run {i}: {', '.join(d)}" for i, d in v["diffs"])
        if not memo and v["pairs"]:
            pr = v["pairs"]
            taus = [p["tau"] for p in pr if p["tau"] is not None]
            line += (f"  overlap12={min(p['overlap12'] for p in pr)}/{PAGE_SIZE}"
                     f" top4={min(p['top4'] for p in pr)}/{HEAD_K}"
                     f" tau={min(taus) if taus else 'n/a'}"
                     f" same_intents={sum(p['same_intents'] for p in pr)}/{len(pr)}")
        print(line)
        failed += v["status"] == "FAIL"
    judged = [v for v in verdicts if not v["status"].startswith("not pinned")]
    n_fb = len(verdicts) - len(judged)
    if memo:
        print(f"\n  PASS {len(judged) - failed}/{len(judged)} judged, FAIL {failed}, "
              f"not pinned {n_fb}")
    else:
        same = sum(v["status"] == "identical" for v in judged)
        print(f"\n  identical {same}/{len(judged)} judged, fell back on run 1: {n_fb}")
    return 1 if failed else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--queries", default=None, help="JSON list of queries (or of {query: ...}).")
    p.add_argument("--query", action="append", default=[], help="One query; repeatable.")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--memo", choices=("on", "off"), default="on")
    p.add_argument("--memo-dir", default=None,
                   help="Memo store for the check (default: a new temp dir; never the app's, "
                        "never the App's app_state volume).")
    p.add_argument("--allow-volume", action="store_true",
                   help="Accept a /Volumes/... --memo-dir (a scratch volume of your own).")
    p.add_argument("--subprocess", action="store_true",
                   help="Run 1 in a fresh process (another PYTHONHASHSEED): the restart case.")
    p.add_argument("--attempts", type=int, default=2,
                   help="Retries of a run 1 that fell back before it is reported not pinned.")
    p.add_argument("--no-product", action="store_true", help="Skip /product of the top card.")
    p.add_argument("--out", default=None, help="Write every snapshot as JSON here.")
    p.add_argument("--emit", default=None, help=argparse.SUPPRESS)   # the --subprocess child
    args = p.parse_args(argv)
    if args.runs < 1:
        p.error("--runs must be >= 1")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    memo = args.memo == "on"
    queries = load_queries(args.queries, args.query)
    if args.emit:
        configure_memo(memo, args.memo_dir, args.allow_volume)
        import app.main as main_module

        firsts = [first_run(main_module, q, args.attempts, not args.no_product) for q in queries]
        Path(args.emit).write_text(json.dumps(firsts), encoding="utf-8")
        return 0
    t0 = time.perf_counter()
    verdicts = run_check(queries, runs=args.runs, memo=memo, memo_dir=args.memo_dir,
                         in_subprocess=args.subprocess, attempts=args.attempts,
                         product=not args.no_product, allow_volume=args.allow_volume)
    code = print_report(verdicts, memo)
    calls = sum(r["llm_calls"] for v in verdicts for r in v["runs"])
    used = verdicts[0].get("memo_dir") if verdicts else None
    print(f"  provider calls {calls} in all; {time.perf_counter() - t0:.0f}s; memo dir "
          f"{used if memo else '(off)'}")
    if args.out:
        Path(args.out).write_text(json.dumps(verdicts, indent=1, default=str), encoding="utf-8")
        print(f"  snapshots -> {args.out}")
    return code


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.WARNING)   # before app.main sets INFO
    sys.exit(main())
