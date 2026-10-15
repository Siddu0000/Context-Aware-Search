"""Offline tests for eval/eval_page_types.py: no LLM, no catalog, no network.

The eval's own selftest does the work (it travels with the script, so it also runs in
a HEAD worktree: `python -m eval.eval_page_types --selftest`); this file runs it under
pytest or plainly (`python tests/test_eval_page_types.py`) and pins the query file
against the spec and against app/product_type.requested_types.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.product_type import requested_types  # noqa: E402
from eval import eval_page_types as ept  # noqa: E402

_BEACH_NONE_OF = ["bride", "bridal", "bridesmaid", "gown", "bikini", "swimsuit", "cover up",
                  "wool", "fleece", "thermal", "sweater", "coat", "costume"]


def test_page_metrics_both_row_shapes():
    ept._selftest_metrics()


def test_memo_replays_outputs_and_errors_by_window_and_constraints():
    ept._selftest_memo()


def test_app_path_run_with_fakes():
    ept._selftest_run()


def test_query_file_covers_the_spec():
    specs = {s.query: s for s in ept.load_queries()}
    assert len(specs) == len(ept.load_queries()), "duplicate queries"
    stakeholder = ["cloths for a beach wedding in summer", "summer beach wedding outfit dress",
                   "beach wedding outfit", "cozy desk setup ideas"]
    assert [q for q, s in specs.items() if s.group == "stakeholder"] == stakeholder
    collisions = {"organic green tea bags": "grocery", "ring light": "electronics",
                  "smart ring": "electronics", "apple watch": "electronics",
                  "laptop bag": "electronics", "camera bag": "electronics",
                  "desk fan": "electronics", "watch movies on my bedroom wall": "electronics"}
    for query, vertical in collisions.items():
        assert specs[query].vertical_in == {vertical}, query
    others = ["clothes for a beach wedding", "what should I wear to a beach wedding",
              "summer clothes", "linen outfit ideas", "breathable outfit for a humid day",
              "outfit for a job interview", "hiking outfit", "boho summer dress",
              "wedding guest dress", "something for her to wear to a black-tie gala",
              "cozy clothes for a lazy sunday at home", "running gear for winter",
              "rain gear for hiking", "what do I need for a beach wedding", "first aid kit",
              "kit kat", "gold drop earrings for a wedding", "waterproof hiking boots",
              "dress shoes for men", "gift for my wife", "home office setup",
              "gaming setup under $500"]
    assert set(stakeholder + others + list(collisions)) == set(specs)
    for query in ("cloths for a beach wedding in summer", "summer beach wedding outfit dress",
                  "beach wedding outfit", "clothes for a beach wedding"):
        assert specs[query].raw["context_none_of"] == _BEACH_NONE_OF, query
    cozy = specs["cozy desk setup ideas"]
    assert cozy.bad_re.search("USB hub") and not cozy.bad_re.search("hubcap")
    assert cozy.good_re.search("warm string light") and not cozy.good_re.search("lightning")


def test_expected_types_agree_with_requested_types():
    """A flat row that names its type expects what main.py will act on; a mismatch
    means the query file or product_type.requested_types is wrong."""
    for spec in ept.load_queries():
        if spec.types and spec.bundles == (None,):
            assert requested_types(spec.query) == spec.types, spec.query


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
