"""App-path page-type eval: what page 1 of /search SHOWS, by product type and context.

run_eval scores the reranker's own list and never runs app/main.py, so it cannot see
the ordering main.py applies around the reranker -- gender balancing, the rerank
window, the display order, slot order, sponsored boosts. That is where the demo
complaint lived ("cloths for a beach wedding in summer" served shoes, earrings and
bags, sometimes first; "cozy desk setup ideas" served cables and SSDs). This calls
app.main.search in-process the way /chat and /unified_search do -- page 1, 12 cards,
rerank on, sponsored on, no cross-sell -- and scores the page a shopper would see:

  flat page -> share of the top 4 / top 12 whose type is not the expected one, the
               first mismatch position, context fit (title words that betray the
               setting: 'bridal' or 'wool' at a beach wedding), vertical share,
               served-page P@4 / P@12 when an entry has relevance criteria
  bundle    -> routing (bundle_type), components that look right or wrong in the
               intents and in the slot primaries; for an outfit also garment slots
               before the first non-garment slot and a mixed-gender flag

The queries live in data/page_type_queries.json; each entry documents itself in
`_context`. Keys: query, group, types (expected FASHION_TYPES, or null = not scored),
bundle (expected bundle_type, or a list of acceptable ones), vertical_in /
vertical_not, context_none_of / context_any_of (title terms, word-bounded, case-
insensitive, a space also matches a hyphen, a trailing s/es plural allowed),
component_bad / component_good (word-bounded regexes), gender (women|men: share of
the opposite gender served), relevance (eval.run_eval.build_relevance_set criteria).

It ALSO runs on the pre-change code (a HEAD worktree with only this file, its JSON
and app/product_type.py copied in): rows without `product_type` are typed here with
classify_title, and cfg.TYPE_AWARE_RANKING is touched only when it exists (absent,
every mode collapses to one arm labelled 'head').

--mode both runs the knob OFF then ON for each query and repeat. Everything an LLM
does is outside the knob, so the ON arm replays the OFF arm's translation, and any
rerank call it makes with the same window (catalog_index in order) and the same
constraints -- errors included, so a fallback degrades both arms alike. A different
window or constraint list (the ON arm can add a product_type line) is a real call.
The two arms then differ by ordering policy, not by LLM sampling. The `memo` column
names what an arm replayed; its latency columns are left blank for those stages.

Every run starts from an EMPTY page_cache: its key ignores repeats (and, before the
change, the knob), so repeat 2 would otherwise serve repeat 1's pool.

Cost: 2 LLM calls per query run (translate + rerank); in `both` the second arm adds
only the rerank calls it cannot replay. 34 queries x 3 repeats ~ 200-300 calls.

Usage:
    python -m eval.eval_page_types                          # --mode both, 1 repeat
    python -m eval.eval_page_types --mode off --repeats 3
    python -m eval.eval_page_types --relevance data/eval_queries_context.json
    python -m eval.eval_page_types --selftest               # offline: fakes, no LLM
On Databricks: databricks/10_page_eval.py.
"""

import argparse
import contextlib
import copy
import csv
import inspect
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Optional

import app.config as cfg
from app import page_cache
from app.llm_client import active_model, seed_sent, track_usage
from app.product_type import (
    FASHION_TYPES,
    TYPE_COLUMN,
    classify_title,
    is_fashion,
    requested_types,
    slot_type,
)
from eval.metrics_ir import precision_at_k

PAGE_SIZE = 12
HEAD_K = 4                      # what is above the fold on the demo grid
QUERIES_JSON = cfg.DATA_DIR / "page_type_queries.json"
MODES = ("on", "off", "both")
_ABSENT = object()

COLUMNS = [
    "arm", "knob", "repeat", "group", "query", "requested",
    "bundle_expected", "bundle_type", "bundle_ok", "no_match", "n_results", "types_page",
    "mismatch_top4", "mismatch_top12", "first_mismatch", "unknown_top12",
    "context_fit_top4", "context_fit_top12", "vertical_top4", "vertical_top12",
    "wrong_gender_top4", "wrong_gender_top12", "P@4", "P@12",
    "garment_slots_first", "first_non_garment_slot", "mixed_gender", "slot_intent_types",
    "intents_bad", "intents_good", "items_bad", "items_good",
    "n_sponsored", "rerank_succeeded", "intents", "constraints",
    "fell_back", "partial", "memo",
    "llm_calls", "tokens_total", "ms_translate", "ms_retrieve", "ms_rerank", "ms_total",
]


# ------------------------------------------------------------------ query specs
@dataclass
class QuerySpec:
    query: str
    group: str = ""
    types: Optional[frozenset] = None
    bundles: tuple = (None,)
    vertical_in: frozenset = frozenset()
    vertical_not: frozenset = frozenset()
    none_re: Optional[re.Pattern] = None
    any_re: Optional[re.Pattern] = None
    bad_re: Optional[re.Pattern] = None
    good_re: Optional[re.Pattern] = None
    gender: Optional[str] = None
    relevance: Optional[dict] = None
    raw: dict = field(default_factory=dict)


_KEYS = {"query", "_context", "group", "types", "bundle", "vertical_in", "vertical_not",
         "context_none_of", "context_any_of", "component_bad", "component_good", "gender",
         "relevance"}
_BUNDLES = (None, "recipe", "outfit", "setup")


def _terms_re(terms) -> Optional[re.Pattern]:
    """One word-bounded alternation: 'gown' must not match 'nightgown', 'coat' not
    'raincoat'; 'cover up' also matches 'Cover-Up', and 'gown' matches 'gowns'."""
    if not terms:
        return None
    alts = sorted({re.escape(t.strip().casefold()).replace(r"\ ", r"[\s-]+")
                   for t in terms if t.strip()}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")(?:e?s)?\b", re.IGNORECASE)


def prepare(entry: dict) -> QuerySpec:
    """A validated spec: a typo in a key, type, bundle or regex fails HERE, before
    any LLM spend, not as a silently unscored row an hour into a run."""
    unknown = set(entry) - _KEYS
    if unknown or not str(entry.get("query") or "").strip():
        raise ValueError(f"bad entry {entry.get('query')!r}: unknown keys {sorted(unknown)}")
    types = entry.get("types")
    if types is not None and (not types or set(types) - set(FASHION_TYPES)):
        raise ValueError(f"{entry['query']!r}: types must be a non-empty subset of "
                         f"{FASHION_TYPES}, got {types!r}")
    bundle = entry.get("bundle")
    bundles = tuple(bundle) if isinstance(bundle, list) else (bundle,)
    if set(bundles) - set(_BUNDLES):
        raise ValueError(f"{entry['query']!r}: bundle must be among {_BUNDLES}")
    gender = entry.get("gender")
    if gender not in (None, "women", "men"):
        raise ValueError(f"{entry['query']!r}: gender must be women or men")
    regex = {k: re.compile(entry[k], re.IGNORECASE) if entry.get(k) else None
             for k in ("component_bad", "component_good")}
    return QuerySpec(
        query=entry["query"],
        group=entry.get("group", ""),
        types=frozenset(types) if types else None,
        bundles=bundles,
        vertical_in=frozenset(v.casefold() for v in entry.get("vertical_in") or ()),
        vertical_not=frozenset(v.casefold() for v in entry.get("vertical_not") or ()),
        none_re=_terms_re(entry.get("context_none_of")),
        any_re=_terms_re(entry.get("context_any_of")),
        bad_re=regex["component_bad"],
        good_re=regex["component_good"],
        gender=gender,
        relevance=entry.get("relevance"),
        raw=entry,
    )


def load_queries(path=None) -> list:
    """Specs from a page-type JSON or an eval-set JSON (query + relevance); a relative
    path is resolved against the repo root, as for EVAL_QUERIES_JSON."""
    path = cfg.eval_queries_path(str(path)) if path else QUERIES_JSON
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return [prepare(e) for e in entries]


def _specs(queries=None, relevance=None) -> list:
    """--queries alone, --relevance alone (that set's queries, scored by P@k), or both:
    the --queries list, with criteria matched from --relevance by query text."""
    if relevance and not queries:
        return load_queries(relevance)
    specs = load_queries(queries)
    if relevance:
        criteria = {s.query: s.relevance for s in load_queries(relevance)}
        for s in specs:
            s.relevance = s.relevance or criteria.get(s.query)
    return specs


# ------------------------------------------------------------------ row rulers
# The SAME rulers before and after the change: main.py's own _product_gender changes
# with it (C1), and HEAD rows carry no product_type.
_WOMEN_RE = re.compile(r"\b(women|women's|womens|woman|ladies|lady|girls?)\b")
_MEN_RE = re.compile(r"\b(men|men's|mens|man|boys?|gentlemen)\b")


def row_type(row: dict) -> Optional[str]:
    """The row's product type: its own product_type when annotated, else the title's
    (Fashion only), else None. Lowercase local rows and Title-case Databricks rows
    type the same (classify_title and is_fashion are case-insensitive)."""
    value = row.get(TYPE_COLUMN)
    if isinstance(value, str) and value:
        return value
    if is_fashion(row.get("bsns_vrtcl_name")):
        return classify_title(row.get("Product_title"))[0]
    return None


def type_label(row: dict) -> str:
    """A row's type for the page sequence; a non-Fashion row shows its vertical."""
    return row_type(row) or (str(row.get("bsns_vrtcl_name") or "?").casefold() or "?")


def row_gender(row: dict) -> str:
    """'women' | 'men' | 'neutral': the gender field when it is a non-empty string,
    else word-bounded category then title words, on Fashion rows only (local rows
    carry '' and Databricks rows None/NaN for a missing gender; 'replacement',
    'ramen', 'cowgirl' never gender a row)."""
    value = row.get("gender")
    if isinstance(value, str) and value.strip():
        value = value.strip().casefold()
        return value if value in ("women", "men") else "neutral"
    if not is_fashion(row.get("bsns_vrtcl_name")):
        return "neutral"
    for key in ("categ_lvl2_name", "Product_title"):
        text = str(row.get(key) or "").casefold().replace("’", "'")
        if _WOMEN_RE.search(text):
            return "women"
        if _MEN_RE.search(text):
            return "men"
    return "neutral"


def _vertical(row: dict) -> str:
    return str(row.get("bsns_vrtcl_name") or "").strip().casefold()


def _share(flags) -> Optional[float]:
    flags = list(flags)
    return round(sum(flags) / len(flags), 3) if flags else None


def _top(flags) -> tuple:
    """(share in the top 4, share in the top 12)."""
    flags = list(flags)
    return _share(flags[:HEAD_K]), _share(flags[:PAGE_SIZE])


def page_metrics(spec: QuerySpec, payload: dict, relevant=None) -> dict:
    """Score one served page. Flat-page type metrics only for a flat page with
    expected types; outfit metrics only for an outfit; the rest wherever their spec
    keys are set. None = not scored (blank in the CSV)."""
    results = list(payload.get("results") or [])[:PAGE_SIZE]
    titles = [str(r.get("Product_title") or "") for r in results]
    bundle = payload.get("bundle_type") or None
    intents = [str(i) for i in payload.get("interpreted_as") or []]
    m = {
        "bundle_type": bundle or "",
        "bundle_ok": bundle in spec.bundles,
        "no_match": bool(payload.get("no_match")),
        "n_results": len(results),
        "types_page": ",".join(type_label(r) for r in results),
        "n_sponsored": sum(bool(r.get("is_sponsored")) for r in results),
        "rerank_succeeded": payload.get("rerank_succeeded"),
        "intents": " | ".join(intents),
        "constraints": json.dumps(payload.get("constraints") or [], ensure_ascii=False,
                                  sort_keys=True),
    }
    if not bundle and spec.types:
        miss = [row_type(r) not in spec.types for r in results]
        m["mismatch_top4"], m["mismatch_top12"] = _top(miss)
        # 0 = no mismatch on the page
        m["first_mismatch"] = miss.index(True) + 1 if True in miss else 0
        m["unknown_top12"] = _share(row_type(r) == "unknown" for r in results)
    if spec.none_re or spec.any_re:
        fit = [(spec.none_re is None or not spec.none_re.search(t))
               and (spec.any_re is None or bool(spec.any_re.search(t))) for t in titles]
        m["context_fit_top4"], m["context_fit_top12"] = _top(fit)
    if spec.vertical_in or spec.vertical_not:
        ok = [(not spec.vertical_in or _vertical(r) in spec.vertical_in)
              and _vertical(r) not in spec.vertical_not for r in results]
        m["vertical_top4"], m["vertical_top12"] = _top(ok)
    if spec.gender:
        wrong = "men" if spec.gender == "women" else "women"
        m["wrong_gender_top4"], m["wrong_gender_top12"] = _top(
            row_gender(r) == wrong for r in results)
    if relevant:
        m["P@4"] = round(precision_at_k(titles, relevant, HEAD_K), 3)
        m["P@12"] = round(precision_at_k(titles, relevant, PAGE_SIZE), 3)
    if bundle == "outfit":
        kinds = [row_type(r) for r in results]
        lead = next((i for i, k in enumerate(kinds) if k != "apparel"), len(kinds))
        m["garment_slots_first"] = lead
        m["first_non_garment_slot"] = lead + 1 if lead < len(kinds) else 0
        genders = {row_gender(r) for r in results}
        m["mixed_gender"] = "women" in genders and "men" in genders
    if bundle:
        m["slot_intent_types"] = ",".join(
            slot_type(r.get("source_intent") or "") for r in results)
    for name, rx in (("bad", spec.bad_re), ("good", spec.good_re)):
        if rx is not None:
            m[f"intents_{name}"] = sum(bool(rx.search(i)) for i in intents)
            m[f"items_{name}"] = sum(bool(rx.search(t)) for t in titles)
    return m


def error_labels(errors) -> tuple:
    """(fell_back, partial): a 'partial' still used the LLM's output (cached by main,
    not degraded in run_eval); anything else means a stage fell back."""
    errors = errors or []
    fell = [str(e.get("stage")) for e in errors if e.get("code") != "partial"]
    part = [f"{e.get('stage')}:partial" for e in errors if e.get("code") == "partial"]
    return ";".join(fell), ";".join(part)


# ------------------------------------------------------------------ memoisation
_RERANK_FIELDS = ("rerank_score", "bayesian_rating", "final_score", "reason")


def _dump(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _row_key(row: dict):
    ci = row.get("catalog_index")
    return ("ci", int(ci)) if ci is not None else ("title", str(row.get("Product_title")))


def _rerank_input_k() -> int:
    """The reranker's window, read where the reranker reads it."""
    import app.reranker as reranker_module

    return getattr(reranker_module, "RERANK_INPUT_K", cfg.RERANK_INPUT_K)


class Memo:
    """The first arm's LLM outputs for one (query, repeat), replayed to the second.

    Keys carry everything the LLM saw that can differ between arms: the translator
    sees the query and mode either way; the reranker's prompt is fixed by the query,
    the window (catalog_index in order: the same rows render the same lines), the
    constraints and top_k. A rerank output is stored as each row's key plus the
    fields the reranker set, and replayed ONTO the current call's rows: the same row
    can reach the ON arm under another source_intent (the bundle keep filter), and
    slotting reads it."""

    def __init__(self):
        self.translate, self.rerank, self.reused = {}, {}, set()

    def reset(self):
        self.translate.clear()
        self.rerank.clear()
        self.reused = set()


def memo_translate(real, memo: Memo):
    """Wrap app.main.understand_query (the name main CALLS) with the memo."""
    def understand_query(user_query, mode=None, errors=None, **kwargs):
        key = (user_query, mode, _dump(kwargs))
        if key in memo.translate:
            out, errs = copy.deepcopy(memo.translate[key])
            memo.reused.add("translate")
        else:
            errs = []
            out = real(user_query, mode=mode, errors=errs, **kwargs)
            memo.translate[key] = copy.deepcopy((out, errs))
        if errors is not None:
            # Replayed too: else the reusing arm's fell_back undercounts
            errors.extend(errs)
        return out
    return understand_query


def memo_rerank(real, memo: Memo):
    """Wrap app.main.llm_rerank with the memo (see Memo for the key)."""
    def llm_rerank(query, candidates, top_k, errors=None, constraints=None, **kwargs):
        window = list(candidates[:_rerank_input_k()])
        key = (query, tuple(_row_key(c) for c in window), _dump(constraints), top_k,
               _dump(kwargs))
        if key in memo.rerank:
            stored, errs = memo.rerank[key]
            by_key = {_row_key(c): c for c in window}
            ranked = [{**dict(by_key.get(k, {})), **copy.deepcopy(f)} for k, f in stored]
            errs = copy.deepcopy(errs)
            memo.reused.add("rerank")
        else:
            errs = []
            ranked = real(query, candidates, top_k=top_k, errors=errs,
                          constraints=constraints, **kwargs)
            by_key = {_row_key(c): c for c in window}
            stored = []
            for out in ranked:
                src = by_key.get(_row_key(out), {})
                stored.append((_row_key(out), copy.deepcopy(
                    {k: v for k, v in out.items() if k in _RERANK_FIELDS or k not in src})))
            memo.rerank[key] = (stored, copy.deepcopy(errs))
        if errors is not None:
            errors.extend(errs)
        return ranked
    return llm_rerank


# ------------------------------------------------------------------ running
@contextlib.contextmanager
def _patched(obj, attrs: dict):
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def arms(mode: str) -> list:
    """[(label, knob value)]. Without the knob (pre-change code) one 'head' arm."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not hasattr(cfg, "TYPE_AWARE_RANKING"):
        return [("head", None)]
    return {"off": [("off", False)], "on": [("on", True)],
            "both": [("off", False), ("on", True)]}[mode]


def search_kwargs(search_fn, rerank: bool) -> dict:
    """EVERY parameter of the endpoint fn, as /chat's _run_search passes them: one
    left out keeps its Query() object as the value. A parameter added later is
    passed its own Query() default."""
    kwargs = dict(top_k=PAGE_SIZE, page=1, rerank=rerank, recommend=False,
                  sponsored=True, translator_mode=None)
    for name, p in inspect.signature(search_fn).parameters.items():
        if name != "query" and name not in kwargs and p.default is not p.empty:
            kwargs[name] = getattr(p.default, "default", p.default)
    return kwargs


def _relevance_sets(specs, catalog) -> dict:
    need = [s for s in specs if s.relevance]
    if not need:
        return {}
    # The served frame, never a fresh read_csv: the local backend lowercases titles
    # in place, and relevance is matched on the exact title (eval.md)
    from eval.run_eval import build_relevance_set

    if catalog is None:
        from app.search import get_dataframe

        catalog = get_dataframe()
    out = {}
    for s in need:
        rel = build_relevance_set(catalog, s.relevance)
        if not rel:
            print(f"[warn] no relevant products for {s.query!r}: P@k left blank")
        out[s.query] = rel or None
    return out


def _run_one(main, spec, arm, knob, rep, kwargs, memo, relevant) -> dict:
    if knob is not None:
        cfg.TYPE_AWARE_RANKING = knob
    page_cache.clear()
    memo.reused = set()
    t0 = time.perf_counter()
    with track_usage() as usage:
        try:
            payload, failure = main.search(query=spec.query, **kwargs), None
        except Exception as e:  # noqa: BLE001 -- one query failing must not end the run
            payload, failure = {}, e
    wall_ms = round((time.perf_counter() - t0) * 1000)
    errors = list(payload.get("errors") or [])
    if failure is not None:
        errors.append({"stage": "search", "code": type(failure).__name__})
    fell_back, partial = error_labels(errors)
    requested = requested_types(spec.query)
    row = {
        "arm": arm, "knob": "absent" if knob is None else knob, "repeat": rep,
        "group": spec.group, "query": spec.query,
        "requested": ",".join(sorted(requested)) if requested else "",
        "bundle_expected": "|".join(b or "flat" for b in spec.bundles),
        "fell_back": fell_back, "partial": partial,
        "memo": ";".join(sorted(memo.reused)),
        "llm_calls": usage["calls"], "tokens_total": usage["total_tokens"],
    }
    if failure is None:
        row.update(page_metrics(spec, payload, relevant))
    # Latency only for what this arm actually ran: a replayed stage took ~0 ms
    lat = payload.get("latency_ms") or {}
    row["ms_translate"] = "" if "translate" in memo.reused else lat.get("translate", "")
    row["ms_retrieve"] = lat.get("retrieve", "")
    row["ms_rerank"] = "" if "rerank" in memo.reused else lat.get("rerank", "")
    row["ms_total"] = "" if memo.reused else wall_ms
    return row


def _print_row(row: dict) -> None:
    bits = [f"{row['arm']:4s} r{row['repeat']} {row['query'][:44]!r:47s}"]
    for label, col in (("mm4", "mismatch_top4"), ("mm12", "mismatch_top12"),
                       ("ctx4", "context_fit_top4"), ("vert4", "vertical_top4"),
                       ("P@12", "P@12"), ("garm1st", "garment_slots_first"),
                       ("bad", "items_bad"), ("good", "items_good")):
        if row.get(col) not in (None, ""):
            bits.append(f"{label}={row[col]}")
    if row.get("bundle_type"):
        bits.append(f"bundle={row['bundle_type']}")
    if row["fell_back"]:
        bits.append(f"FELL_BACK={row['fell_back']}")
    if row["memo"]:
        bits.append(f"memo={row['memo']}")
    print(" ".join(bits))


def model_label() -> str:
    """The model the run used (else the one cfg selects), safe for a file name."""
    model = active_model()
    if not model:
        model = {"openai": cfg.OPENAI_MODEL, "gemini": cfg.GEMINI_MODEL,
                 "anthropic": cfg.ANTHROPIC_MODEL}.get(cfg.LLM_PROVIDER, cfg.LLM_PROVIDER)
    return model or "unknown"


def csv_path(out_dir, mode_label, tag="") -> Path:
    """eval_results/page_types_<mode>_<model>[_<tag>]_<ts>.csv"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_label())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"page_types_{mode_label}_{model}{'_' + tag if tag else ''}_{ts}.csv"


def _csv_row(row: dict) -> dict:
    return {k: ("" if v is None else v) for k, v in row.items()}


def run(mode="both", repeats=1, rerank=True, queries=None, relevance=None,
        out_dir=None, tag="", load=True, catalog=None):
    """Run every query `repeats` times per arm; write the CSV; print the summary.

    Returns (csv_path, rows). `load=False` and `catalog` are for the offline selftest
    (no catalog load); 10_page_eval.py passes out_dir (the UC Volume) and a tag."""
    import app.main as main

    specs = _specs(queries, relevance)
    plan = arms(mode)
    mode_label = "head" if plan[0][0] == "head" else mode
    if mode == "both" and len(plan) == 1:
        print("!! cfg.TYPE_AWARE_RANKING does not exist (pre-change code): one 'head' "
              "arm instead of off + on")
    if load:
        from app.search import load_index

        load_index()
    relevant = _relevance_sets(specs, catalog)
    # Short on purpose: a HEAD worktree under the scratch dir sits near Windows' 260-char
    # path limit ("rel_context" = data/eval_queries_context.json)
    rel = "rel_" + Path(relevance).stem.removeprefix("eval_queries_") if relevance else ""
    tag = "_".join(t for t in (tag, "" if rerank else "rerank_off", rel) if t)

    memo = Memo()
    # Only `both` replays: a single arm's repeats must each sample the LLM afresh
    patches = ({"understand_query": memo_translate(main.understand_query, memo),
                "llm_rerank": memo_rerank(main.llm_rerank, memo)} if len(plan) > 1 else {})
    kwargs = search_kwargs(main.search, rerank)
    knob_before = getattr(cfg, "TYPE_AWARE_RANKING", _ABSENT)
    out_path = csv_path(out_dir or cfg.EVAL_RESULTS_DIR, mode_label, tag)
    print(f"=== page-type eval: mode={mode_label}, {len(specs)} queries x {repeats} "
          f"repeat(s), arms={[a for a, _ in plan]}, rerank={'on' if rerank else 'off'} ===")
    rows = []
    # Opened BEFORE the first LLM call (an unwritable path fails now, not after the
    # spend) and flushed per row, so an interrupted run keeps what it measured
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="", extrasaction="ignore")
        writer.writeheader()
        try:
            with _patched(main, patches):
                # Repeats outermost: a provider hiccup then hits different queries,
                # not every repeat of one
                for rep in range(1, repeats + 1):
                    for spec in specs:
                        memo.reset()
                        for arm, knob in plan:
                            rows.append(_run_one(main, spec, arm, knob, rep, kwargs, memo,
                                                 relevant.get(spec.query)))
                            writer.writerow(_csv_row(rows[-1]))
                            f.flush()
                            _print_row(rows[-1])
        finally:
            if knob_before is not _ABSENT:
                cfg.TYPE_AWARE_RANKING = knob_before
            page_cache.clear()  # nothing from a forced knob value outlives the run

    summarise(rows, specs, rerank, out_path)
    return out_path, rows


# ------------------------------------------------------------------ summary
_SUMMARY_COLS = (
    ("mm4", "mismatch_top4"), ("mm12", "mismatch_top12"), ("unk12", "unknown_top12"),
    ("ctx4", "context_fit_top4"), ("ctx12", "context_fit_top12"),
    ("vert4", "vertical_top4"), ("vert12", "vertical_top12"),
    ("wrongG12", "wrong_gender_top12"), ("P@4", "P@4"), ("P@12", "P@12"),
    ("garm1st", "garment_slots_first"), ("int_bad", "intents_bad"),
    ("int_good", "intents_good"), ("item_bad", "items_bad"), ("item_good", "items_good"),
)


def _mean(rows, col) -> Optional[float]:
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    return round(mean(vals), 3) if vals else None


def summarise(rows, specs, rerank, out_path) -> dict:
    arm_names = list(dict.fromkeys(r["arm"] for r in rows))
    print("\n=== Per query, mean over repeats (blank = not scored) ===")
    for spec in specs:
        for arm in arm_names:
            mine = [r for r in rows if r["query"] == spec.query and r["arm"] == arm]
            bits = [f"{label}={v}" for label, col in _SUMMARY_COLS
                    if (v := _mean(mine, col)) is not None]
            firsts = [r.get("first_mismatch") for r in mine
                      if r.get("first_mismatch") not in (None, "")]
            if firsts:
                bits.append(f"1st_mm={firsts}")
            bundles = sorted({r.get("bundle_type") or "flat" for r in mine})
            mixed = [r for r in mine if r.get("mixed_gender") is True]
            if mixed:
                bits.append(f"mixed_gender={len(mixed)}/{len(mine)}")
            print(f"  {spec.query[:44]:44s} {arm:4s} [{'/'.join(bundles)}] {' '.join(bits)}")

    print("\n=== Page-1 types, stakeholder queries ===")
    for spec in specs:
        if spec.group != "stakeholder":
            continue
        print(f"  {spec.query}")
        for r in rows:
            if r["query"] == spec.query:
                print(f"    {r['arm']:4s} r{r['repeat']}: {r.get('types_page', '')}")

    print("\n=== Mean over all scored rows ===")
    summary = {}
    for label, col in _SUMMARY_COLS:
        means = {arm: _mean([r for r in rows if r["arm"] == arm], col) for arm in arm_names}
        if all(v is None for v in means.values()):
            continue
        summary[col] = means
        cells = "  ".join(f"{arm}={'-' if v is None else v}" for arm, v in means.items())
        delta = ""
        if arm_names == ["off", "on"] and None not in means.values():
            delta = f"  (on-off {means['on'] - means['off']:+.3f})"
        print(f"  {label:9s} {cells}{delta}")

    degraded = [r for r in rows if r["fell_back"]]
    partial = [r for r in rows if r["partial"]]
    misrouted = [r for r in rows if r.get("bundle_ok") is False]
    print("\n=== Validity ===")
    if degraded:
        print(f"  !! {len(degraded)}/{len(rows)} RUNS RAN DEGRADED "
              f"({sorted({r['fell_back'] for r in degraded})}) -- those rows do NOT "
              "measure the full pipeline. Fix the cause before quoting them.")
    else:
        print(f"  degraded     = 0/{len(rows)}")
    if partial:
        print(f"  !  {len(partial)}/{len(rows)} runs had a PARTIAL stage "
              f"({sorted({lab for r in partial for lab in r['partial'].split(';')})}): "
              "the LLM's output was still used -- not a fallback")
        if any("translate:partial" in r["partial"] for r in partial):
            print("  !  translate:partial on ordinary shopper text is a guard false positive "
                  "or a screened reply: investigate before quoting these numbers")
    if misrouted:
        print(f"  !  routing differs from expected on {len(misrouted)} run(s): "
              + "; ".join(sorted({f"{r['query']!r} -> {r.get('bundle_type') or 'flat'}"
                                  for r in misrouted})))
    reused = [r for r in rows if r["memo"]]
    if reused:
        print(f"  memo         = {len(reused)} run(s) replayed the first arm's LLM output "
              "(no latency recorded for those stages)")
    model = active_model()
    print(f"  llm          = {model or 'none built (no LLM call was made)'} "
          f"(seed sent: {'yes' if model and seed_sent() else 'no'})")
    knobs = {r["arm"]: r["knob"] for r in rows}
    print("  knob         = " + ("TYPE_AWARE_RANKING absent (pre-change code)"
                                 if "absent" in knobs.values() else
                                 ", ".join(f"{a}: TYPE_AWARE_RANKING={k}"
                                           for a, k in knobs.items())))
    print(f"  rerank_on    = {rerank}")
    print(f"  backend      = {(cfg.SEARCH_BACKEND or 'local').lower()}")
    print(f"  saved to     = {out_path}")
    summary["n_runs"], summary["n_degraded"] = len(rows), len(degraded)
    return summary


# ------------------------------------------------------------------ offline selftest
def _fake_row(ci, title, vertical, categ, gender=None, score=0.9, shape="databricks",
              typed=False):
    """A catalog row as each backend serves it: Databricks keeps the case and None
    for a missing gender; the local backend lowercases in place and fills ''."""
    row = {"catalog_index": ci, "Product_title": title, "bsns_vrtcl_name": vertical,
           "categ_lvl2_name": categ, "gender": gender, "score": score,
           "parent_asin": f"SELFTEST{ci}", "average_rating": 4.5, "rating_number": 120,
           "price": 25.0}
    if shape == "local":
        row.update(Product_title=title.lower(), bsns_vrtcl_name=vertical.lower(),
                   categ_lvl2_name=categ.lower(), gender=gender or "")
    if typed:
        row[TYPE_COLUMN] = classify_title(title)[0] if is_fashion(vertical) else None
    return row


_BEACH = [("Women's Linen Sundress", "Fashion", "Women's Clothing", None),
          ("Flat Sandals for Women", "Fashion", "Women's Clothing", None),
          ("Gold Drop Earrings", "Fashion", "Women's Clothing", None),
          ("Bridal Lace Gown", "Fashion", "Women's Clothing", None),
          ("Beach Cover-Up Kaftan", "Fashion", "Women's Clothing", None),
          ("Lens Cleaning Cloths", "Electronics", "Camera Accessories", None),
          ("Men's Linen Shirt", "Fashion", "Men's Clothing", "men")]


def _selftest_metrics():
    """Rulers and metrics on hand-built pages, in BOTH row shapes."""
    spec = prepare({"query": "clothes for a beach wedding", "types": ["apparel"],
                    "bundle": None, "vertical_in": ["Fashion"], "gender": "women",
                    "context_none_of": ["bridal", "gown", "cover up"],
                    "component_bad": r"\bsandals?\b"})
    expected = {"types_page": "apparel,footwear,jewelry,apparel,apparel,electronics,apparel",
                "mismatch_top4": 0.5, "mismatch_top12": 0.429, "first_mismatch": 2,
                "unknown_top12": 0.0, "context_fit_top4": 0.75, "context_fit_top12": 0.714,
                "vertical_top4": 1.0, "vertical_top12": 0.857, "wrong_gender_top4": 0.0,
                "wrong_gender_top12": 0.143, "items_bad": 1, "bundle_ok": True}
    for shape in ("databricks", "local"):
        for typed in (False, True):
            results = [_fake_row(i, *t, shape=shape, typed=typed) for i, t in enumerate(_BEACH)]
            m = page_metrics(spec, {"results": results, "interpreted_as": ["linen sundress"]})
            got = {k: m.get(k) for k in expected}
            assert got == expected, (shape, typed, got)
            assert "garment_slots_first" not in m and "P@12" not in m
    # An annotated row's own product_type wins over the title
    assert row_type({TYPE_COLUMN: "footwear", "Product_title": "Linen Dress",
                     "bsns_vrtcl_name": "Fashion"}) == "footwear"
    # NaN/None product_type on a non-Fashion row is no type (it shows its vertical)
    assert type_label({TYPE_COLUMN: float("nan"), "Product_title": "Ring Light",
                       "bsns_vrtcl_name": "electronics"}) == "electronics"
    # Gender traps: substrings never gender a row; non-Fashion rows are neutral
    for row, want in [
        ({"Product_title": "Replacement Band", "bsns_vrtcl_name": "Fashion",
          "categ_lvl2_name": "Watch Accessories", "gender": None}, "neutral"),
        ({"Product_title": "Cowgirl Boots", "bsns_vrtcl_name": "fashion", "gender": ""},
         "neutral"),
        ({"Product_title": "Ramen Noodles for Men", "bsns_vrtcl_name": "Grocery",
          "gender": ""}, "neutral"),
        ({"Product_title": "Unisex Hoodie", "bsns_vrtcl_name": "Fashion", "gender": "unisex"},
         "neutral"),
        ({"Product_title": "mens jogger", "bsns_vrtcl_name": "fashion", "gender": ""}, "men"),
        ({"Product_title": "Ankle Boots", "bsns_vrtcl_name": "Fashion",
          "categ_lvl2_name": "Women’s Shoes", "gender": float("nan")}, "women"),
    ]:
        assert row_gender(row) == want, (row, row_gender(row))
    # Word-bounded terms: no 'nightgown' for gown, 'Cover-Up' and plurals do match
    rx = _terms_re(["gown", "cover up", "coat"])
    assert rx.search("Beach Cover-Up") and rx.search("ball gowns") and rx.search("COATS")
    assert not rx.search("satin nightgown") and not rx.search("raincoat")
    # Outfit page: garment slots, mixed gender, components on intents AND primaries
    spec = prepare({"query": "beach wedding outfit", "bundle": "outfit",
                    "component_good": r"\b(sun)?dress(es)?\b", "component_bad": r"\bclutch\b"})
    slots = [dict(_fake_row(1, "Women's Linen Sundress", "Fashion", "Women's Clothing"),
                  source_intent="linen sundress"),
             dict(_fake_row(2, "Men's Leather Sandals", "Fashion", "Men's Clothing"),
                  source_intent="flat sandals"),
             dict(_fake_row(3, "Straw Clutch", "Fashion", "Women's Clothing"),
                  source_intent="straw clutch")]
    m = page_metrics(spec, {"results": slots, "bundle_type": "outfit",
                            "interpreted_as": ["linen sundress", "flat sandals", "straw clutch"]})
    assert (m["garment_slots_first"], m["first_non_garment_slot"], m["mixed_gender"]) == \
        (1, 2, True), m
    assert m["slot_intent_types"] == "apparel,footwear,bag", m
    assert (m["intents_good"], m["items_good"], m["intents_bad"], m["items_bad"]) == \
        (1, 1, 1, 1), m
    assert m["bundle_ok"] and "mismatch_top4" not in m
    assert error_labels([{"stage": "translate", "code": "partial"},
                         {"stage": "rerank", "code": "LLMError"}]) == \
        ("rerank", "translate:partial")
    # Validation fails fast
    for bad in ({"query": "x", "types": ["shoes"]}, {"query": "x", "bundle": "kit"},
                {"query": "x", "typo": 1}, {"query": "x", "component_bad": "("}):
        try:
            prepare(bad)
        except (ValueError, re.error):
            continue
        raise AssertionError(f"accepted {bad}")


def _selftest_memo():
    """Replay rules: same key -> one real call with its errors replayed; another
    constraint list or window -> a real call; replay lands on the CURRENT rows."""
    memo, calls = Memo(), []

    def fake_translate(user_query, mode=None, errors=None):
        calls.append(("t", user_query))
        errors.append({"stage": "translate", "code": "partial", "detail": "x"})
        return {"intents": ["a"], "constraints": [], "bundle_type": None}

    wrapped = memo_translate(fake_translate, memo)
    e1, e2 = [], []
    out1 = wrapped("q", mode=None, errors=e1)
    out1["intents"].append("mutated")
    out2 = wrapped("q", mode=None, errors=e2)
    assert calls == [("t", "q")] and memo.reused == {"translate"}
    assert e1 == e2 == [{"stage": "translate", "code": "partial", "detail": "x"}]
    assert out2["intents"] == ["a"], "a replay must not share objects with the first arm"

    def fake_rerank(query, candidates, top_k, errors=None, constraints=None):
        calls.append(("r", _dump(constraints)))
        errors.append({"stage": "rerank", "code": "partial"})
        return [dict(c, rerank_score=90 - i, bayesian_rating=4.0, final_score=88.0 - i,
                     reason="ok") for i, c in enumerate(candidates[:top_k])][::-1]

    rr = memo_rerank(fake_rerank, memo)
    window = [{"catalog_index": i, "Product_title": f"T{i}", "source_intent": "dress"}
              for i in range(3)]
    calls.clear()
    memo.reset()
    errs_a, errs_b = [], []
    first = rr("q", window, top_k=15, errors=errs_a, constraints=[])
    moved = [dict(c, source_intent="sandals") for c in window]   # same rows, other slot
    second = rr("q", moved, top_k=15, errors=errs_b, constraints=[])
    assert len(calls) == 1 and memo.reused == {"rerank"} and errs_a == errs_b
    assert [r["catalog_index"] for r in second] == [r["catalog_index"] for r in first]
    assert [r["rerank_score"] for r in second] == [r["rerank_score"] for r in first]
    assert {r["source_intent"] for r in second} == {"sandals"}, second
    rr("q", window, top_k=15, errors=[],
       constraints=[{"type": "product_type", "value": "clothing only"}])
    rr("q", window[::-1], top_k=15, errors=[], constraints=[])
    rr("q", window, top_k=12, errors=[], constraints=[])
    assert len(calls) == 4, "constraints, window order and top_k are all in the key"
    # Only the window is in the key: rows past RERANK_INPUT_K are never sent
    k = _rerank_input_k()
    long = [{"catalog_index": i, "Product_title": f"T{i}"} for i in range(k + 3)]
    rr("q", long, top_k=15, errors=[], constraints=[])
    rr("q", long[:k] + long[:k - 1:-1], top_k=15, errors=[], constraints=[])
    assert len(calls) == 5


def _selftest_run():
    """app.main.search end to end with every LLM, retrieval and catalog dependency
    faked: memo, error replay, page_cache clearing, the knob, P@k and the CSV."""
    import tempfile

    import pandas as pd

    import app.llm_client as llm_client
    import app.main as main
    import app.search as search_module

    beach_db = [_fake_row(i, *t, score=0.95 - i / 100, typed=True)
                for i, t in enumerate(_BEACH)]
    beach_local = [_fake_row(20 + i, *t, score=0.95 - i / 100, shape="local")
                   for i, t in enumerate(_BEACH)]
    kitkat = [_fake_row(40 + i, f"Kit Kat Wafer Bar {i}", "Grocery", "Snacks",
                        score=0.9 - i / 100) for i in range(3)]
    outfit = {"linen sundress": [_fake_row(60, "Women's Linen Sundress", "Fashion",
                                           "Women's Clothing", score=0.95, typed=True)],
              "flat sandals": [_fake_row(61, "Women's Flat Sandals", "Fashion",
                                         "Women's Clothing", score=0.90, typed=True)],
              "straw clutch": [_fake_row(62, "Straw Clutch Bag", "Fashion",
                                         "Women's Clothing", score=0.85, typed=True)]}
    # Intents are unique per query: the fake retrieval finds its pool by them
    pools = {"clothes for a beach wedding": {"linen sundress": beach_db},
             "cloths for a beach wedding": {"light linen sundress": beach_local},
             "summer clothes": {"summer dress": beach_local},
             "kit kat": {"kit kat": kitkat},
             "beach wedding outfit": outfit}
    by_intents = {tuple(p): p for p in pools.values()}
    queries = [
        {"query": "clothes for a beach wedding", "group": "stakeholder", "types": ["apparel"],
         "vertical_in": ["fashion"], "context_none_of": ["bridal", "gown", "cover up"],
         "relevance": {"title_any_of": ["sundress", "shirt"]}},
        {"query": "cloths for a beach wedding", "types": ["apparel"]},
        {"query": "summer clothes", "types": ["apparel"]},
        {"query": "kit kat", "vertical_in": ["grocery"]},
        {"query": "beach wedding outfit", "bundle": "outfit",
         "component_good": r"\b(sun)?dress(es)?\b"},
    ]
    translate_calls, rerank_calls, clears = [], [], []

    def fake_understand(user_query, mode=None, errors=None):
        translate_calls.append(user_query)
        if user_query.startswith("cloths"):
            errors.append({"stage": "translate", "code": "LLMError", "detail": "fake"})
        if user_query == "summer clothes":
            errors.append({"stage": "translate", "code": "partial", "detail": "fake"})
        bundle = "outfit" if user_query.endswith("outfit") else None
        return {"intents": list(pools[user_query]), "bundle_type": bundle,
                "is_recipe": False, "constraints": [], "query": user_query}

    def fake_search(intents, top_k=30, per_intent_quota=None, keep=None):
        pool = by_intents[tuple(intents)]
        rows = []
        for term in intents:
            for r in pool[term]:
                r = dict(r, source_intent=term)
                if keep is None or keep(r, term):
                    rows.append(r)
        return sorted(rows, key=lambda r: r["score"], reverse=True)

    def fake_rerank(query, candidates, top_k, errors=None, constraints=None):
        window = candidates[:_rerank_input_k()]
        rerank_calls.append((query, tuple(_row_key(c) for c in window), _dump(constraints)))
        if query == "summer clothes":
            errors.append({"stage": "rerank", "code": "partial", "detail": "fake"})
        return [dict(c, rerank_score=90 - i, bayesian_rating=4.0, final_score=89.0 - i,
                     reason="ok") for i, c in enumerate(window)][:top_k]

    def no_call(*_a, **_k):
        raise AssertionError("the selftest must not load the catalog or call an LLM")

    real_clear = page_cache.clear

    def counting_clear():
        clears.append(1)
        real_clear()

    catalog = pd.DataFrame(beach_db)
    knob_before = getattr(cfg, "TYPE_AWARE_RANKING", _ABSENT)
    tmp = Path(tempfile.mkdtemp(prefix="page_types_selftest_"))
    qfile = tmp / "queries.json"
    qfile.write_text(json.dumps(queries), encoding="utf-8")
    fakes = {"understand_query": fake_understand, "llm_rerank": fake_rerank,
             "search_products": fake_search, "build_recommendations": no_call,
             "sponsored_map": lambda: {"SELFTEST0": {"sponsor": "Selftest", "bid": 1.0}}}
    with _patched(main, fakes), _patched(search_module, {"load_index": no_call}), \
            _patched(llm_client, {"get_llm_client": no_call}), \
            _patched(page_cache, {"clear": counting_clear}):
        out_path, rows = run(mode="both", repeats=2, queries=str(qfile), out_dir=tmp,
                             load=False, catalog=catalog)
        assert main.understand_query is fake_understand and main.llm_rerank is fake_rerank
    plan = arms("both")
    n_runs = len(queries) * 2 * len(plan)
    # one clear before every run, one after the last
    assert len(rows) == n_runs and len(clears) == n_runs + 1, (len(rows), len(clears))
    # page_cache emptied before each run AND the memo reset per repeat: else repeat 2
    # would be served from the cache and never reach the translator
    assert len(translate_calls) == len(queries) * 2, translate_calls
    assert getattr(cfg, "TYPE_AWARE_RANKING", _ABSENT) == knob_before
    assert not any(r["fell_back"].startswith("search") for r in rows), \
        [r["fell_back"] for r in rows]
    by = {(r["query"], r["repeat"], r["arm"]): r for r in rows}
    for (query, rep, arm), r in by.items():
        if query.startswith("cloths"):
            assert r["fell_back"] == "translate", r          # replayed into both arms
        else:
            assert r["fell_back"] == "", r
        if query == "summer clothes":
            assert set(r["partial"].split(";")) == {"translate:partial", "rerank:partial"}, r
        assert r["knob"] == ("absent" if plan[0][0] == "head" else arm == "on"), r
    flat = by[("clothes for a beach wedding", 1, plan[0][0])]
    # The whole 7-row pool is served, so these do not depend on main's ordering
    assert sorted(flat["types_page"].split(",")) == sorted(
        "apparel,footwear,jewelry,apparel,apparel,electronics,apparel".split(",")), flat
    assert (flat["mismatch_top12"], flat["context_fit_top12"], flat["vertical_top12"],
            flat["P@12"]) == (0.429, 0.714, 0.857, 0.286), flat
    assert flat["n_sponsored"] == 1 and flat["memo"] == "" and flat["ms_total"] != ""
    local = by[("cloths for a beach wedding", 1, plan[0][0])]
    assert sorted(local["types_page"].split(",")) == sorted(flat["types_page"].split(","))
    out = by[("beach wedding outfit", 1, plan[0][0])]
    assert (out["bundle_type"], out["n_results"], out["garment_slots_first"],
            out["intents_good"], out["mixed_gender"]) == ("outfit", 3, 1, 1, False), out
    if len(plan) == 2:
        for rep in (1, 2):
            for q in queries:
                off, on = by[(q["query"], rep, "off")], by[(q["query"], rep, "on")]
                assert off["memo"] == "" and "translate" in on["memo"].split(";"), on
                assert on["ms_translate"] == "" and on["ms_total"] == "", on
                real = [c for c in rerank_calls if c[0] == q["query"]]
                # one real rerank per arm, minus the replays (a hit <=> no 2nd call)
                want = 2 * 2 - sum("rerank" in by[(q["query"], r, "on")]["memo"]
                                   for r in (1, 2))
                assert len(real) == want, (q["query"], len(real), want)
            # A grocery pool with one intent: same window, same constraints -> replayed
            assert "rerank" in by[("kit kat", rep, "on")]["memo"], by[("kit kat", rep, "on")]
    with open(out_path, encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == n_runs and list(written[0]) == COLUMNS
    assert out_path.name.startswith(f"page_types_{'head' if len(plan) == 1 else 'both'}_")
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()


def selftest() -> int:
    failed = 0
    for fn in (_selftest_metrics, _selftest_memo, _selftest_run):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001 -- report each, then exit non-zero
            failed += 1
            import traceback

            traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {str(e)[:500]}")
    print(f"\nselftest: {3 - failed} passed, {failed} failed "
          f"(knob {'present' if hasattr(cfg, 'TYPE_AWARE_RANKING') else 'absent'})")
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=MODES, default="both",
                   help="TYPE_AWARE_RANKING arm(s); 'both' replays the first arm's LLM output.")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--queries", default=None,
                   help="Page-type JSON (default data/page_type_queries.json).")
    p.add_argument("--relevance", default=None,
                   help="Eval-set JSON with relevance criteria, e.g. "
                        "data/eval_queries_context.json: served-page P@4 / P@12.")
    p.add_argument("--no-rerank", action="store_true", help="Disable the LLM rerank.")
    p.add_argument("--tag", default="", help="Extra CSV name tag.")
    p.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 -- a non-reconfigurable stream prints as it can
        pass
    logging.basicConfig(level=logging.WARNING)   # before app.main sets INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.selftest:
        sys.exit(selftest())
    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    run(mode=args.mode, repeats=args.repeats, rerank=not args.no_rerank,
        queries=args.queries, relevance=args.relevance, tag=args.tag)


if __name__ == "__main__":
    main()
