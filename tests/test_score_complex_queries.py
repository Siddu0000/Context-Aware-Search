"""Offline tests for the complex query bank (data/complex_queries.json, 2026-11-02) and its
scorer, eval/score_complex_queries.py: no LLM, no catalog, no network.

The bank is only useful if databricks/10_page_eval.py can load it (eval_page_types'
prepare() rejects an unknown key or a bad type before any LLM spend) and if the scorer
splits what the eval's single mean mixes (core targets that point in opposite
directions, single-department pages on multi-department asks). Runs under pytest or
plainly: `python tests/test_score_complex_queries.py`.
"""

import csv
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import score_complex_queries as scq  # noqa: E402

BANK = ROOT / "data" / "complex_queries.json"


def _bank():
    with open(BANK, encoding="utf-8") as f:
        return json.load(f)


def _meta_by_id():
    return {m["id"]: m for m in scq.load_meta().values()}


def test_bank_loads_through_eval_page_types():
    from eval import eval_page_types as ept
    specs = ept.load_queries("data/complex_queries.json")
    assert len(specs) == 107
    assert len({s.query for s in specs}) == 107, "duplicate queries"
    assert {s.group for s in specs} == set(scq.GROUPS)


def test_meta_matches_bank():
    bank, meta = _bank(), list(scq.load_meta().values())
    assert [b["query"] for b in bank] == [m["query"] for m in meta]
    for b, m in zip(bank, meta):
        # the grader's pass bar lives in _context, prefixed with the meta id
        assert b["_context"].split(":", 1)[0] == m["id"], m["id"]
        assert b["group"] == m["group"], m["id"]
        assert (b.get("core") is None) == (m["core_target"] is None), m["id"]
        assert m["core_target"] in (None, "0", ">=0.8"), m["id"]
    assert len({m["id"] for m in meta}) == len(meta)
    assert sum(bool(m["dept_min"]) for m in meta) == 13
    assert sum(bool(m["gap"]) for m in meta) == 8
    assert sorted(m["core_target"] for m in meta if m["core_target"]) == ["0"] * 12 + [">=0.8"] * 10


def _row(entry, **cols):
    base = {"arm": "on", "query": entry["query"], "group": entry["group"], "fell_back": "",
            "partial": "", "bundle_type": "", "bundle_ok": True, "memo_hits": 0}
    base.update(cols)
    return base


def test_score_splits_core_targets_and_departments():
    meta = _meta_by_id()
    # gap rows also carry core targets (electronics-01, cross-19): keep the roles apart
    accessory = next(m for m in meta.values() if m["core_target"] == "0" and not m["gap"])
    device = next(m for m in meta.values() if m["core_target"] == ">=0.8" and not m["gap"])
    basket = next(m for m in meta.values() if m["dept_min"] and not m["gap"])
    gap = next(m for m in meta.values() if m["gap"] and not m["core_target"])
    first, second = basket["dept_min"]["departments"][:2]
    token = {"Fashion": "apparel", "Electronics": "electronics", "Grocery": "grocery"}
    rows = [
        # an accessory ask passes with at most one device card of 12, a device ask needs 0.8
        _row(accessory, core_top4=0.0, core_top12=1 / 12),
        _row(device, core_top4=1.0, core_top12=0.5),
        # 9 cards of one department and 3 of the other: the flat rule needs 3 each
        _row(basket, types_page=",".join([token[first]] * 9 + [token[second]] * 3)),
        _row(basket, arm="off", types_page=",".join([token[first]] * 12)),
        _row(gap, match_weak=True, fell_back="rerank", memo_hits=2, bundle_ok=False,
             bundle_type="recipe"),
        {"arm": "on", "query": "not in the bank"},
    ]
    report = scq.score(rows, scq.load_meta())
    assert report["unmatched_queries"] == ["not in the bank"]
    on = report["arms"]["on"]
    assert on["core_by_target"]["0"]["pass_core_top12"] == 1
    assert on["core_by_target"][">=0.8"]["ids_failing"] == [device["id"]]
    assert (on["dept_min"]["runs_pass"], on["dept_min"]["runs"]) == (1, 1)
    off = report["arms"]["off"]["dept_min"]
    assert off["runs_pass"] == 0 and off["short"] == {basket["id"]: [{second: 0}]}
    validity = on["validity"]
    assert validity["degraded"] == [gap["id"]] and validity["memo_hits"] == 2
    assert validity["misrouted"] == [f"{gap['id']}->recipe"]
    assert validity["gap_rows_weak_banner"] == {gap["id"]: True}
    text = scq.render(report)
    assert "degraded rows or memo hits" in text and "core_target" in text


def test_cli_reads_a_page_eval_csv():
    meta = _meta_by_id()
    device = next(m for m in meta.values() if m["core_target"] == ">=0.8")
    columns = ["arm", "query", "group", "types_page", "core_top4", "core_top12", "bundle_type",
               "bundle_ok", "fell_back", "partial", "memo_hits", "match_weak", "P@12"]
    with tempfile.TemporaryDirectory() as tmp:
        path, out = Path(tmp) / "page_types_on.csv", Path(tmp) / "report.json"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            # CSV cells are strings: booleans as 'True'/'False', unscored columns blank
            writer.writerow({"arm": "on", "query": device["query"], "group": device["group"],
                             "core_top4": "1.0", "core_top12": "0.917", "bundle_ok": "True",
                             "memo_hits": "0", "match_weak": "False", "P@12": ""})
        buf = io.StringIO()
        with redirect_stdout(buf):
            assert scq.main([str(path), "--json", str(out)]) == 0
        report = json.loads(out.read_text(encoding="utf-8"))
    core = report["arms"]["on"]["core_by_target"][">=0.8"]
    assert core["pass_core_top12"] == 1 and core["core_top12"] == 0.917
    assert report["arms"]["on"]["validity"]["misrouted"] == []
    assert "1/1" in buf.getvalue()


# ------------------------------------------------------------------ plain runner
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 -- report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:500]}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
