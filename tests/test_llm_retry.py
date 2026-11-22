"""Offline tests for the retry on an unusable LLM reply (2026-11-19): the translator and the
reranker ask ONCE more when the model answered but nothing in its reply can be used
(`llm_client.UnusableReply`), and both read a few more reply shapes. Fake LLM, no catalog,
no network, the LLM memo off (tests/test_llm_memo.py covers the memo side).

Why: the Databricks App review (query bank, tab 2). "I need ingredients to prepare a quick,
healthy breakfast that takes less than 15 minutes" and "Find groceries that I can use to
prepare three different meals without wasting ingredients" showed "Query understanding is
temporarily unavailable (error LLMError)": the regex routes "ingredients" to RECIPE_PROMPT,
which wants a dish, and the model answered an empty list. "I need a smartphone with a great
camera and battery life for travelling, but I don't want a flagship-priced phone" showed
"AI ranking is temporarily unavailable (error LLMError)". Never retried: a provider error
(the openai SDK already retries those) and a reply the query guard screened out (an obeyed
injection gets no second try).

Runs under pytest (`python -m pytest tests/test_llm_retry.py`) or plainly
(`python tests/test_llm_retry.py`).
"""

import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_ENABLED", "false")

import app.config as cfg  # noqa: E402
from app import llm_client, query_guard  # noqa: E402
from app import reranker as rr  # noqa: E402
from app import translator as tr  # noqa: E402
from app.llm_client import LLMError, UnusableReply  # noqa: E402

BREAKFAST = ("I need ingredients to prepare a quick, healthy breakfast that takes less than "
             "15 minutes.")
MEALS = ("Find groceries that I can use to prepare three different meals without wasting "
         "ingredients.")
FLAT = "comfortable shoes for standing all day"


# ------------------------------------------------------------------ helpers
class FakeLLM:
    """Stands in for a stage's generate_json: records every prompt; `reply(prompt, n)`
    gives the PARSED reply (dict, list, str ...) or an exception to raise."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt, temperature=0.2, **kw):
        self.prompts.append(prompt)
        r = self.reply(prompt, len(self.prompts))
        if isinstance(r, BaseException):
            raise r
        return json.loads(json.dumps(r))   # a fresh object per call, as a real parse is


@contextlib.contextmanager
def fake_stage(module, reply):
    fake, real = FakeLLM(reply), module.generate_json
    module.generate_json = fake
    try:
        yield fake
    finally:
        module.generate_json = real


@contextlib.contextmanager
def patched(obj, attrs: dict):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def settings(**kw):
    """The knobs these tests read, pinned: .env must not decide what a test checks."""
    return patched(cfg, {"LLM_MEMO": False, "QUERY_GUARD_ENABLED": True,
                         "OUTFIT_BUNDLES_ENABLED": False, **kw})


@contextlib.contextmanager
def captured(name):
    records, logger = [], logging.getLogger(name)

    class H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h, old = H(level=logging.DEBUG), logger.level
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(h)
        logger.setLevel(old)


def understand(query, reply):
    """understand_query in query_expansion mode on a fake LLM: (out, errors, prompts, logs)."""
    errs: list = []
    with settings(), fake_stage(tr, reply) as llm, captured("app.translator") as logs:
        out = tr.understand_query(query, mode="query_expansion", errors=errs)
    return out, errs, llm.prompts, logs


GOOD = {"search_terms": ["memory foam work shoes", "slip resistant clogs"],
        "bundle_type": None, "constraints": []}


def second_is_good(first):
    """The first reply is `first`, the next one GOOD."""
    return lambda p, n: first if n == 1 else GOOD


# ------------------------------------------------------------------ translator
def test_a_routed_recipe_prompt_with_no_dish_asks_the_general_prompt():
    """The review's grocery asks: the recipe prompt's empty list is followed by ONE call of
    the general prompt, which may keep the recipe bundle or make the page flat."""
    for query in (BREAKFAST, MEALS):
        assert tr.detect_bundle_type(query) == "recipe"
        for bundle in ("recipe", None):
            def reply(p, n, bundle=bundle):
                if p.startswith(tr.RECIPE_PROMPT):
                    return {"search_terms": []}
                assert p.startswith(tr.general_prompt()) and tr._RETRY_NOTE not in p
                return {"search_terms": ["rolled oats", "greek yogurt", "eggs"],
                        "bundle_type": bundle, "constraints": []}

            out, errs, prompts, logs = understand(query, reply)
            assert out["intents"] == ["rolled oats", "greek yogurt", "eggs"], out
            assert out["bundle_type"] == bundle and out["is_recipe"] == (bundle == "recipe")
            assert errs == [] and len(prompts) == 2
            assert prompts[0].startswith(tr.RECIPE_PROMPT)
            assert any("recipe prompt reply unusable" in m for m in logs), logs
            # never the query or the reply in a log line (safety.md)
            assert not any("ingredients" in m or "oats" in m for m in logs), logs


def test_an_unusable_general_reply_is_asked_once_more_with_the_note():
    out, errs, prompts, logs = understand(FLAT, second_is_good({"search_terms": []}))
    assert out["intents"] == GOOD["search_terms"] and errs == []
    assert len(prompts) == 2 and tr._RETRY_NOTE not in prompts[0]
    # the note is part of the instructions; the query is still the prompt's last line
    assert prompts[1] == tr._with_query(tr.general_prompt() + tr._RETRY_NOTE, FLAT)
    assert any("general prompt reply unusable" in m for m in logs), logs
    assert not any("standing" in m for m in logs), logs


def test_every_unusable_kind_is_retried():
    unusable = [
        {"search_terms": []},                       # an empty list
        {"search_terms": 5},                        # the wrong type
        {"search_terms": {"q": "red shoes"}},       # a dict's bare strings are no list
        {},                                         # nothing at all
        ["red shoes", "boots"],                     # a bare array is no reply object
        "red shoes",                                # nor is a string
        UnusableReply("LLM returned invalid JSON: '{\"search_terms\": [\"a'"),
        UnusableReply("m reply truncated (hit max_tokens)."),
        UnusableReply("LLM returned no text content (reasoning-only reply?)"),
    ]
    for first in unusable:
        out, errs, prompts, _ = understand(FLAT, second_is_good(first))
        assert out["intents"] == GOOD["search_terms"] and errs == [], first
        assert len(prompts) == 2 and tr._RETRY_NOTE in prompts[1], first


def test_two_unusable_replies_fall_back_after_exactly_two_calls():
    # a plain ask: the general prompt, then the general prompt with the note
    out, errs, prompts, _ = understand(FLAT, lambda p, n: {"search_terms": []})
    assert out == {"intents": [FLAT], "bundle_type": None, "is_recipe": False,
                   "constraints": [], "query": FLAT}
    assert [e["code"] for e in errs] == ["LLMError"] and len(prompts) == 2
    # a routed ask: the recipe prompt, then the general prompt; never a third call
    out, errs, prompts, _ = understand(MEALS, lambda p, n: {"search_terms": []})
    assert out["intents"] == [MEALS] and out["bundle_type"] is None
    assert [e["code"] for e in errs] == ["LLMError"] and len(prompts) == 2
    assert prompts[0].startswith(tr.RECIPE_PROMPT) and tr._RETRY_NOTE not in prompts[1]


def test_a_clean_no_match_provider_errors_and_guard_refusals_are_not_retried():
    # gibberish is an answer: one call, no intents, no error
    out, errs, prompts, _ = understand("qwzxv plorp", lambda p, n: {"no_intent": True})
    assert out["intents"] == [] and errs == [] and len(prompts) == 1
    # an outage: the SDK has already retried; one call, the fallback, its HTTP code
    out, errs, prompts, _ = understand(
        FLAT, lambda p, n: RuntimeError("Error code: 429 - rate limit reached"))
    assert out["intents"] == [FLAT] and [e["code"] for e in errs] == ["429"]
    assert len(prompts) == 1
    # every phrase screened out by the guard: an obeyed injection gets no second try
    out, errs, prompts, _ = understand(
        FLAT, lambda p, n: {"search_terms": ["see https://evil.example/now"]})
    assert out["intents"] == [FLAT] and len(errs) == 1 and len(prompts) == 1


def test_phrase_objects_and_other_lists_are_read():
    read = tr._reply_terms
    assert read({"search_terms": ["red shoes", " ", 7, None, True]}) == ["red shoes"]
    # ingredient objects, a name-like field each (quantities are not phrases)
    assert read({"search_terms": [{"ingredient": "rolled oats", "quantity": "1 cup"},
                                  {"name": "eggs"}, {"item": "spinach"}]}) == \
        ["rolled oats", "eggs", "spinach"]
    # one list per dish
    assert read({"search_terms": {"overnight oats": ["oats", "milk"],
                                  "omelette": ["eggs"]}}) == ["oats", "milk", "eggs"]
    assert read({"search_terms": [["oats", "milk"], ["eggs"]]}) == ["oats", "milk", "eggs"]
    assert read({"search_terms": [{"dish": "omelette", "ingredients": ["eggs", "chives"]}]}) \
        == ["eggs", "chives"]
    # the phrases under another key, when search_terms has none
    assert read({"ingredients": ["oats", "milk"]}) == ["oats", "milk"]
    assert read({"search_terms": [], "items": [{"product": "almond butter"}]}) == \
        ["almond butter"]
    assert read({"search_terms": ["red shoes"], "ingredients": ["x"]}) == ["red shoes"]
    # never: a dict's bare strings, numbers, or a list nested past the depth limit
    assert read({"search_terms": {"q": "red shoes", "n": 3}}) == []
    assert read({"search_terms": [[[["too deep"]]]]}) == []
    assert read({"query": "red shoes"}) == []


def test_a_read_phrase_is_screened_like_a_plain_one():
    reply = {"search_terms": [{"name": "rolled oats"}, {"name": "see https://evil.example"}],
             "bundle_type": None}
    out, errs, prompts, _ = understand(FLAT, lambda p, n: reply)
    assert out["intents"] == ["rolled oats"] and len(prompts) == 1
    assert [e["code"] for e in errs] == ["partial"]


# ------------------------------------------------------------------ reranker
CANDS = [{"Product_title": t, "bsns_vrtcl_name": "Fashion", "categ_lvl2_name": "Women",
          "average_rating": 4.0, "rating_number": 100} for t in ("Dress A", "Dress B", "Dress C")]
RANKED = {"ranked": [{"idx": 1, "score": 90, "reason": "fits"}, {"idx": 0, "score": 60},
                     {"idx": 2, "score": 20}]}


def rerank(reply, query="summer dress"):
    errs: list = []
    with settings(), fake_stage(rr, reply) as llm, captured("app.reranker") as logs:
        out = rr.rerank(query, [dict(c) for c in CANDS], top_k=3, errors=errs)
    return [p["Product_title"] for p in out], out, errs, llm.prompts, logs


def test_an_empty_ranking_is_asked_once_more_with_every_candidate():
    titles, out, errs, prompts, logs = rerank(
        lambda p, n: {"ranked": []} if n == 1 else RANKED)
    assert titles == ["Dress B", "Dress A", "Dress C"] and errs == []
    assert out[0]["rerank_score"] == 90 and len(prompts) == 2
    assert rr._RETRY_NOTE.format(n=3) in prompts[1] and "all 3" in prompts[1]
    assert rr._RETRY_NOTE.format(n=3) not in prompts[0]
    assert any("Rerank: reply unusable" in m for m in logs), logs
    assert not any("summer" in m or "Dress" in m for m in logs), logs


def test_other_ranking_shapes_are_read_without_a_retry():
    shapes = [
        ({"results": [{"index": 1, "score": 90}, {"index": 0, "score": 50}]},
         ["Dress B", "Dress A"]),
        ({"ranked": [{"id": 2, "score": 80, "reason": "x"}, {"idx": 0, "score": 70}]},
         ["Dress C", "Dress A"]),
        ([{"idx": 1, "score": 60}], ["Dress B"]),          # a bare array IS the ranking
    ]
    for reply, want in shapes:
        titles, _, errs, prompts, _ = rerank(lambda p, n, r=reply: r)
        assert titles == want and errs == [] and len(prompts) == 1, (reply, titles)


def test_every_unusable_ranking_is_retried():
    unusable = [
        {"ranked": []},
        {"ranked": "Dress B first"},
        {},
        {"a": [{"idx": 0, "score": 1}], "b": [{"idx": 1, "score": 2}]},   # which list?
        {"ranked": [{"idx": 99, "score": 50}, {"idx": "x", "score": 40}]},  # no valid idx
        "Dress B is best",
        UnusableReply("m reply truncated (hit max_tokens)."),
        UnusableReply("LLM returned invalid JSON: '{\"ranked\": [{\"idx\": 0'"),
    ]
    for first in unusable:
        titles, _, errs, prompts, _ = rerank(lambda p, n, f=first: f if n == 1 else RANKED)
        assert titles == ["Dress B", "Dress A", "Dress C"] and errs == [], first
        assert len(prompts) == 2, first


def test_two_unusable_rankings_fall_back_after_exactly_two_calls():
    titles, out, errs, prompts, _ = rerank(lambda p, n: {"ranked": []})
    assert titles == ["Dress A", "Dress B", "Dress C"]          # embedding order
    assert all(p["rerank_score"] is None for p in out)
    assert [e["code"] for e in errs] == ["LLMError"] and len(prompts) == 2


def test_a_provider_error_is_never_retried_by_the_reranker():
    titles, out, errs, prompts, _ = rerank(
        lambda p, n: RuntimeError("Error code: 503 - service unavailable"))
    assert out[0]["rerank_score"] is None and [e["code"] for e in errs] == ["503"]
    assert len(prompts) == 1


def test_a_partial_ranking_is_used_not_retried():
    titles, _, errs, prompts, _ = rerank(
        lambda p, n: {"ranked": [{"idx": 2, "score": 80}, {"idx": 9, "score": 70}]})
    assert titles == ["Dress C"] and [e["code"] for e in errs] == ["partial"]
    assert len(prompts) == 1


# ------------------------------------------------------------------ llm_client
def test_an_unusable_reply_keeps_its_error_code_and_withholds_the_reply():
    assert issubclass(UnusableReply, LLMError)
    e = UnusableReply("LLM returned invalid JSON: '{\"ranked\": [{\"idx\": 0, \"score\": 500'")
    # "LLMError", never a status code read off a number in the model's text
    assert llm_client.error_code(e) == "LLMError"
    detail = query_guard.error_detail(e)
    assert "reply withheld" in detail and "500" not in detail and "ranked" not in detail
    assert llm_client.error_code(UnusableReply("x reply truncated (hit max_tokens).")) \
        == "LLMError"


def test_the_backend_marks_truncated_empty_and_invalid_replies_unusable():
    for raw in ("", "   "):
        try:
            llm_client._reply_text(raw)
            raise AssertionError("an empty reply must raise")
        except UnusableReply:
            pass
    try:
        llm_client._parse_reply("Sure! Here are some ideas: red shoes, boots")
        raise AssertionError("prose must raise")
    except UnusableReply as e:
        assert "reply withheld" in query_guard.error_detail(e)

    b = object.__new__(llm_client._OpenAIBackend)
    b.model, b.base_url = "m", "https://api.groq.com/openai/v1"

    def reply(finish, content):
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(usage=None,
                               choices=[SimpleNamespace(finish_reason=finish, message=msg)])

    groq = {"OPENAI_BASE_URL": b.base_url, "DETERMINISTIC": True,
            "TEMPERATURE_OVERRIDE": None, "GROQ_REASONING_FORMAT": "hidden"}
    with patched(cfg, groq):
        b.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: reply("length", '{"search_terms": ["a'))))
        try:
            b.generate_json("p", 0.2)
            raise AssertionError("a truncated reply must raise")
        except UnusableReply as e:
            assert "truncated" in str(e)
        b.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: reply("stop", '{"ok": true}'))))
        assert b.generate_json("p", 0.2) == '{"ok": true}'


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 -- report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:600]}")
    print(f"\n{len(tests) - failed} passed, {failed} failed, 0 skipped")
    sys.exit(1 if failed else 0)
