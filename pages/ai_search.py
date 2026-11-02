"""AI Search page — GET /search: LLM intents, embedding retrieval, LLM rerank."""

import os

import requests
import streamlit as st

from ui_common import (
    SCOPES,
    apply_scroll,
    bundle_ui,
    clear_filters_button,
    escape_md,
    filter_params,
    filter_rail,
    heading_html,
    inject_css,
    masthead,
    match_is_weak,
    ns,
    pagination,
    params_signature,
    render_detail_page,
    render_product_grid,
    request_scroll,
    reset_filters,
    results_bar,
    results_header_html,
    set_surface,
    sget,
    sinit,
    sort_note,
    sset,
    subnav_for,
    weak_match_html,
)

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
SEARCH_URL = f"{BACKEND}/search"

set_surface("ai")   # own namespace: independent of Keyword Search / Assistant
inject_css()
apply_scroll()      # one-shot: only after Next/Previous/a page number/View/Back/a filter

sinit({
    "query": "",
    "page": 1,
    "last_response": None,
    "view": "search",
    "detail_index": None,
    "detail_response": None,
    "settings": None,
    "filters": {},
    "sort": "relevance",
    "filter_seq": 0,
    "fetched": None,       # the params last_response answers (None = never fetched)
    "fetch_error": None,
    "filter_error": None,  # the API refused a follow-up's filters: the page is kept
    "return_card": None,
    "scope_gender": None,  # Women / Men / Kids picked in the sub-nav before a search
})

# Multiples of 4 fill whole rows of the 4-wide grid; all stay <= RERANK_POOL_K (30),
# so the API's rerank pool -- and its cache entry -- is the same for every choice
_PAGE_SIZES = (8, 12, 16, 20, 24, 28)


def _settings():
    top_k = st.selectbox("Results per page", _PAGE_SIZES, index=_PAGE_SIZES.index(12),
                         key=ns("set_top_k"))
    rerank = st.toggle(
        "AI ranking",
        value=True,
        key=ns("set_rerank"),
        help="Let the LLM re-rank results. Turn off to see raw embedding-similarity order.",
    )
    show_sponsored = st.toggle(
        "Sponsored results",
        value=True,
        key=ns("set_sponsored"),
        help="Paid placements. Always labelled, and only shown when they are "
        "genuinely relevant to the query.",
    )
    show_recs = st.toggle(
        "Product recommendations",
        value=True,
        key=ns("set_recs"),
        help="Show items that pair well with a product, and a better-rated "
        "alternative, on its page.",
    )
    show_debug = st.toggle(
        "Developer details",
        value=False,
        key=ns("set_dev"),
        help="Per-stage timings, the raw ranking scores (rerank / Bayesian rating / "
        "embedding similarity), the AI ranking's notes on each result and the filter "
        "counts. Off by default to keep demos clean.",
    )
    st.caption("Repeat searches are cached, so they return instantly.")
    return top_k, rerank, show_sponsored, show_recs, show_debug


submitted, (top_k, rerank, show_sponsored, show_recs, show_debug) = masthead(
    settings=_settings,
    label="What are you looking for?",
    placeholder="Describe what you need, e.g. something breathable for a humid day",
    scopes=SCOPES,
)
# ui_common reads this global key to decide whether cards show raw scores
st.session_state["show_dev_details"] = show_debug


def _params() -> dict:
    """Everything the current results depend on (recs live on the product page)."""
    return {
        "query": sget("query"),
        "top_k": top_k,
        "page": sget("page"),
        "rerank": str(rerank).lower(),
        "recommend": "false",
        "sponsored": str(show_sponsored).lower(),
        # Filters and sort are applied server-side to the CACHED pool: 0 LLM calls
        **filter_params(sget("filters"), sget("sort")),
    }


def _fetch(force: bool = False):
    """GET /search for the current state, unless last_response already answers it.

    Every change (a page, a filter, a setting) shows up as different params, so each
    one costs exactly one request, and an unrelated rerun (add to cart) costs none."""
    if not sget("query"):
        return
    params = _params()
    signature = params_signature(params)
    if not force and sget("fetched") == signature:
        return
    sset("fetched", signature)   # set on failure too: retry by pressing Search
    follow_up = not force and bool(sget("last_response"))
    if follow_up:
        # A page, filter, sort or page-size change of the results on screen: if their
        # AI stages fell back, the API serves that same pool for a short while instead
        # of re-running both LLM calls and moving the results under the shopper. A
        # Search submit (force) always retries. Not in the signature: it is how, not what
        params = {**params, "follow_up": "true"}
    sset("filter_error", None)
    try:
        r = requests.get(SEARCH_URL, params=params, timeout=90)
        r.raise_for_status()
        sset("last_response", r.json())
        sset("fetch_error", None)
    except requests.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if follow_up and status is not None and 400 <= status < 500:
            # The API refused these filters (a 422 names a limit): the service is up and
            # the pool on screen is still right, so keep it and say what happened. It
            # used to be wiped under "Could not reach the search service" (2026-10-19)
            sset("filter_error", "These filters could not be applied. Change or clear "
                                 "them to update the results.")
            return
        sset("last_response", None)
        # escaped: the exception text carries the request URL, i.e. the shopper's query
        sset("fetch_error", f"Could not reach the search service: {escape_md(e)}")


def _scope_filters() -> dict:
    """The department select as the new search's filters (ui_spec 3.2), applied to the
    matched pool with 0 LLM calls, plus a Women / Men / Kids shortcut picked before any
    search -- used by this one search only: the select shows no gender, so a gender
    kept for later searches would filter them unseen (2026-10-19 review)."""
    scope = st.session_state.get(ns("scope")) or "All"
    gender = sget("scope_gender")
    sset("scope_gender", None)
    if scope == "All":
        return {}
    if scope != "Fashion":
        gender = None
    return {"department": [scope], **({"gender": [gender]} if gender else {})}


if submitted:
    # The band is sticky, so a search can be sent from the bottom of a page: scroll in
    # THIS run, before the fetch. request_scroll() alone waits for the next rerun, i.e.
    # the next unrelated click. apply_scroll() above had nothing pending in a submit run
    # (callbacks' requests are consumed at the top of the run they trigger)
    request_scroll()
    apply_scroll()
    sset("query", submitted)
    sset("page", 1)
    sset("view", "search")
    sset("return_card", None)
    reset_filters()
    sset("filters", _scope_filters())
    with st.spinner("Understanding your request…"):
        _fetch(force=True)

_search_settings = (top_k, rerank, show_sponsored)
_settings_now = (*_search_settings, show_recs)
previous = sget("settings")
if previous != _settings_now:
    if previous is not None and tuple(previous[:3]) != _search_settings:
        sset("page", 1)   # another page size or ranking makes the page number meaningless
    if sget("view") == "detail":
        sset("detail_response", None)   # recommendations on/off changes the product page
    sset("settings", _settings_now)


def _render_intents(intents):
    """Plain text, never markdown: an intent is model-written (or, on a fallback, the
    shopper's literal words), and markdown would make a link in it clickable."""
    if not intents:
        return
    st.caption("What we searched for")
    with st.container(horizontal=True, gap="small", key="cas_intents"):
        for intent in intents:
            st.text(f"• {intent}")


def _render_dev_timings(data):
    timings = data.get("latency_ms", {})
    fell_back = data.get("rerank_succeeded") is False and data.get("rerank_requested")
    if not (show_debug and (timings or fell_back)):
        return
    with st.expander("Developer details — last search"):
        for stage, ms in timings.items():
            st.caption(f"{stage}: {ms} ms")
        if timings:
            st.caption(f"total: {sum(timings.values())} ms")
        if fell_back:
            st.caption("⚠️ AI ranking unavailable — similarity order used")
        mq = data.get("match_quality")
        if isinstance(mq, dict):
            st.caption(f"match quality: {escape_md(mq)}")


def _render_errors(data):
    ai_ranking_fell_back = (
        data.get("rerank_succeeded") is False and data.get("rerank_requested")
    )
    for err in data.get("errors", []):
        stage = err.get("stage")
        code = err.get("code", "?")
        if code == "partial" or (stage == "translate" and code == "guarded"):
            # Not an outage: a partial stage WAS applied (some rerank entries unusable, or
            # the query guard trimmed the query / screened the model's phrases), and
            # 'guarded' deliberately kept an instruction-shaped query away from the LLM.
            # The outage warnings below would show every sanitised query as broken
            if show_debug:
                label = {"rerank": "AI ranking", "translate": "Query understanding"}.get(
                    stage, stage)
                what = "skipped by the query guard" if code == "guarded" else "partly applied"
                st.caption(f"⚠️ {label} {what}: {err.get('detail', '')}")
        elif stage == "rerank":
            st.warning(
                f"⚠️ AI ranking is temporarily unavailable (error {code}) — "
                "showing results ranked by similarity instead."
            )
        elif stage == "translate":
            st.warning(
                f"⚠️ Query understanding is temporarily unavailable (error {code}) "
                "— searched using your words exactly as typed."
            )
        else:
            st.warning(f"⚠️ {stage} error {code}.")
    if not data.get("errors") and ai_ranking_fell_back:
        st.caption("⚠️ Showing results ranked by similarity (AI ranking unavailable).")


def _header(data) -> str:
    results = data.get("results", [])
    page = data.get("page", 1)
    page_size = data.get("page_size", top_k)
    start = (page - 1) * page_size + 1
    return results_header_html(
        start, start + len(results) - 1, data.get("total_results", len(results)),
        sget("query") or "", data.get("total_unfiltered"),
    )


def _render_results_column(data):
    results = data.get("results", [])
    # must come from the API: source_intent is set on every result, every query
    # bundle_type is recipe | outfit | setup | None (one card per component)
    bundle_type = data.get("bundle_type")
    bui = bundle_ui(bundle_type)

    st.html(heading_html(bui["heading"] if bui else "Results"))
    if bui:
        st.caption(bui["blurb"])
    _render_intents(data.get("interpreted_as", []))
    constraints = data.get("constraints") or []
    if constraints:
        # Model-written values: escaped, so a link or "$" in one renders as plain text
        chips = " · ".join(
            f"**{escape_md(c.get('value'))}** ({escape_md(c.get('type'))})"
            for c in constraints
        )
        st.caption(f"Filters applied from your wording: {chips}")

    if not results:
        if data.get("message"):
            st.info(data["message"])   # API text, e.g. "No products match these filters."
            clear_filters_button()
        else:
            st.warning("No products found.")
        return

    # "Related items" on every card only where the product page shows suggestions
    render_product_grid(results, bundle_type=bundle_type, related=show_recs)
    pagination(data.get("page", 1), data.get("total_pages", 1))


def _render_search_results():
    if sget("fetch_error"):
        st.error(sget("fetch_error"))
    if sget("filter_error"):
        st.warning(sget("filter_error"))   # fixed text
    data = sget("last_response")
    if not data:
        if not sget("query"):
            st.caption(
                "Describe what you need in your own words — the search understands the "
                "context, not just the keywords."
            )
        return

    if data.get("no_match"):
        _render_dev_timings(data)
        _render_errors(data)
        _render_intents(data.get("interpreted_as", []))
        st.info(
            data.get("message")
            or "No matching products found. Try different or more general words."
        )
        return

    # Old API (no "facets") and bundles (facets null) get the full-width grid, no rail
    # and no sort
    facets = data.get("facets")
    flat = isinstance(facets, dict) and not bundle_ui(data.get("bundle_type"))
    # The priced-items caption divides ROWS by rows: facets.price.n_priced counts pool
    # rows, while total_unfiltered counts cards after the variant collapse (an API before
    # 2026-10-19 sends no pool_rows, and its total_unfiltered was rows)
    pool_rows = data.get("pool_rows", data.get("total_unfiltered"))
    results_bar(_header(data), sortable=flat)
    if flat:
        note = sort_note(data.get("sort"), facets, pool_rows)
        if note:
            st.caption(note)   # fixed text and counts only
    _render_dev_timings(data)
    _render_errors(data)
    if match_is_weak(data):   # never over an empty page
        st.html(weak_match_html(sget("query") or ""))

    if flat:
        rail, main = st.columns([1, 4], gap="large")
        with rail:
            filter_rail(facets, pool_rows)
        with main:
            _render_results_column(data)
    else:
        _render_results_column(data)


_on_detail = sget("view") == "detail" and sget("detail_index") is not None
if not _on_detail and sget("query"):
    with st.spinner("Loading results…"):
        _fetch()
# After this run's fetches, so the shortcuts describe the page drawn below them: drawn
# before, a submit that returned a bundle still showed the previous page's shortcuts
subnav_for(sget("last_response"))
if _on_detail:
    render_detail_page(show_recs=show_recs)
else:
    _render_search_results()
