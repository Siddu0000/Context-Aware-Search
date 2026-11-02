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

import app.config as cfg  # noqa: E402
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
    # The guest-rule narrowing (2026-10-15): bridal asks the rubric must never cap
    bridal = ["wedding dress", "wedding costume", "bride costume",
              "Wedding dresses for a destination wedding in rajasthan"]
    # Device asks (2026-10-27, owner calls D1 / D2): the meeting's watch gift, the
    # electronics battery's failures and successes, an accessory ask, an owner need
    device = {"gift for a vintage watch collector": "watch", "laptop for college": "laptop",
              "airtag": None, "iphone": "phone",
              "noise cancelling headphones for travel": "headphones", "watch winder": None,
              "stop my laptop from overheating": "laptop", "women's watch": "watch",
              # the 2026-10-27 review round: a core-precision case and an add-on compound
              "ipad": "tablet", "headphone pads": "headphones"}
    assert set(stakeholder + others + bridal + list(collisions) + list(device)) == set(specs)
    for query, family in device.items():
        spec = specs[query]
        assert spec.group == "device" and spec.core == family and spec.bundles == (None,), query
    import app.device_role as dr
    assert dr.requested_device("gift for a vintage watch collector").gift
    assert dr.requested_device("watch winder").accessory_asked
    assert dr.requested_device("stop my laptop from overheating") is None
    assert dr.requested_device("airtag") is None
    assert dr.requested_device("ipad") == dr.DeviceAsk("tablet", False, False)
    assert dr.requested_device("headphone pads") is None
    gift = specs["gift for a vintage watch collector"]
    assert gift.none_re.search("SOONAN DIY Watch Repair Tool Kit")
    assert not gift.none_re.search("Men's Automatic Mechanical Watch")
    for query in ("cloths for a beach wedding in summer", "summer beach wedding outfit dress",
                  "beach wedding outfit", "clothes for a beach wedding"):
        assert specs[query].raw["context_none_of"] == _BEACH_NONE_OF, query
    for query in bridal:
        spec = specs[query]
        assert spec.group == "bridal" and spec.types == {"apparel"} and spec.bundles == (None,)
        assert "context_none_of" not in spec.raw, query  # bridal wear is the ask here
        assert spec.any_re.search("Women's Lace Bridal Gown"), query
    # Outfit bundles are off by default (OUTFIT_BUNDLES_ENABLED): no row expects one,
    # and every outfit-worded clothing row is a flat apparel page
    assert not [q for q, s in specs.items() if "outfit" in s.bundles]
    for query in ("summer beach wedding outfit dress", "beach wedding outfit",
                  "linen outfit ideas", "breathable outfit for a humid day",
                  "outfit for a job interview", "hiking outfit"):
        assert specs[query].types == {"apparel"} and specs[query].bundles == (None,), query
    cozy = specs["cozy desk setup ideas"]
    assert cozy.bad_re.search("USB hub") and not cozy.bad_re.search("hubcap")
    assert cozy.good_re.search("warm string light") and not cozy.good_re.search("lightning")


_OUTFIT_ROWS = ("summer beach wedding outfit dress", "beach wedding outfit",
                "linen outfit ideas", "breathable outfit for a humid day",
                "outfit for a job interview", "hiking outfit")


def test_outfit_rows_accept_an_outfit_with_the_knob_on():
    """Oct 15 review: with OUTFIT_BUNDLES_ENABLED=true (the documented revert) every
    outfit row scored bundle_ok=False and was listed as misrouted. The expectation now
    follows the knob; the JSON keeps the default's (flat)."""
    specs = {s.query: s for s in ept.load_queries()}

    def ok(spec, bundle_type):
        return ept.page_metrics(spec, {"bundle_type": bundle_type, "results": []})["bundle_ok"]

    old = cfg.OUTFIT_BUNDLES_ENABLED
    try:
        cfg.OUTFIT_BUNDLES_ENABLED = False
        for query in _OUTFIT_ROWS:
            assert ept.expected_bundles(specs[query]) == (None,), query
            assert ok(specs[query], None) and not ok(specs[query], "outfit"), query
        cfg.OUTFIT_BUNDLES_ENABLED = True
        for query in _OUTFIT_ROWS:
            assert ept.expected_bundles(specs[query]) == (None, "outfit"), query
            assert ok(specs[query], "outfit") and ok(specs[query], None), query
            assert not ok(specs[query], "setup"), query
        # no other row changes: none of them is a styling ask
        for query, spec in specs.items():
            if query not in _OUTFIT_ROWS:
                assert ept.expected_bundles(spec) == spec.bundles, query
    finally:
        cfg.OUTFIT_BUNDLES_ENABLED = old


def test_rows_record_the_display_knobs_and_the_prompts():
    """2026-10-19 review: REJECTS_AFTER_TAIL and VARIANT_COLLAPSE both change page 1 as
    served (band order; cards and total_results), and the QE and rerank prompts changed,
    yet a row recorded only arm / knob / outfit_bundles / row_text_case: a before/after
    pair, or an A/B of either knob, differed only by the file's timestamp."""
    import types

    import app.translator as tr

    spec = ept.load_queries()[0]
    served = types.SimpleNamespace(search=lambda **kw: {"results": [], "bundle_type": None})

    def row(**knobs):
        old = {name: getattr(cfg, name) for name in knobs}
        try:
            for name, value in knobs.items():
                setattr(cfg, name, value)
            return ept._run_one(served, spec, "on", None, 1, {},
                                types.SimpleNamespace(reused=set()), {})
        finally:
            for name, value in old.items():
                setattr(cfg, name, value)

    assert {"rejects_after_tail", "variant_collapse", "prompts_sha",
            "device_role_ranking", "core_top4", "core_top12", "first_core",
            "match_weak"} <= set(ept.COLUMNS)
    base = row(REJECTS_AFTER_TAIL=True, VARIANT_COLLAPSE=True, DEVICE_ROLE_RANKING=True)
    assert (base["rejects_after_tail"], base["variant_collapse"]) == (True, True), base
    assert row(REJECTS_AFTER_TAIL=False, VARIANT_COLLAPSE=True)["rejects_after_tail"] is False
    assert row(REJECTS_AFTER_TAIL=True, VARIANT_COLLAPSE=False)["variant_collapse"] is False
    # 2026-10-27: the device knob changes the window, display and reranker inputs
    assert base["device_role_ranking"] is True
    assert row(DEVICE_ROLE_RANKING=False)["device_role_ranking"] is False
    old = tr.QUERY_EXPANSION_PROMPT
    try:
        tr.QUERY_EXPANSION_PROMPT = old + " "
        assert row()["prompts_sha"] != base["prompts_sha"]
    finally:
        tr.QUERY_EXPANSION_PROMPT = old
    assert row()["prompts_sha"] == base["prompts_sha"]
    # the rerank lines gained subtype= / audience= with no catalog change
    import app.search as search
    assert base["row_text_case"] == search.ROW_TEXT_CASE != "original"


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
