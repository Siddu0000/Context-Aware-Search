"""Keyword Search page — pure BM25 via GET /keyword_search, no LLM, no AI fallback."""

import os

import requests
import streamlit as st

from ui_common import (
    apply_scroll,
    escape_md,
    heading_html,
    inject_css,
    masthead,
    ns,
    pagination,
    params_signature,
    render_detail_page,
    render_product_grid,
    request_scroll,
    results_bar,
    results_header_html,
    set_surface,
    sget,
    sinit,
    sset,
)

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
KEYWORD_SEARCH_URL = f"{BACKEND}/keyword_search"

set_surface("keyword")   # own namespace, isolated from AI Search / Assistant
inject_css()
apply_scroll()           # one-shot: only after a page change, View or Back

sinit({
    "query": "",
    "page": 1,
    "view": "search",
    "detail_index": None,
    "detail_response": None,
    "classic_last_response": None,
    "classic_settings": None,
    "fetched": None,
    "fetch_error": None,
    "return_card": None,
})

_PAGE_SIZES = (8, 12, 16, 20, 24, 28)


def _settings():
    top_k = st.selectbox("Results per page", _PAGE_SIZES, index=_PAGE_SIZES.index(12),
                         key=ns("set_top_k"))
    show_sponsored = st.toggle(
        "Sponsored results",
        value=True,
        key=ns("set_sponsored"),
        help="Paid placements. Always labelled; only shown when they actually "
        "match the keywords.",
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
        help="Show the raw BM25 / Best-Match ranking scores on each card.",
    )
    st.caption("Keyword results use no AI, so they are instant. "
               "Product recommendations on a product's page do use AI.")
    return top_k, show_sponsored, show_recs, show_debug


# No department select or shortcuts here: /keyword_search takes no filters
submitted, (top_k, show_sponsored, show_recs, show_debug) = masthead(
    settings=_settings,
    label="Search by keyword",
    placeholder="Search by keyword, e.g. men's leather belt, wireless earbuds",
)
st.session_state["show_dev_details"] = show_debug


def _fetch(force: bool = False):
    """GET /keyword_search for the current query + page, unless already answered."""
    if not sget("query"):
        return
    params = {
        "query": sget("query"),
        "top_k": top_k,
        "page": sget("page"),   # an API without paging ignores it: page 1, no pager
        "sponsored": str(show_sponsored).lower(),
    }
    signature = params_signature(params)
    if not force and sget("fetched") == signature:
        return
    sset("fetched", signature)
    try:
        r = requests.get(KEYWORD_SEARCH_URL, params=params, timeout=20)
        r.raise_for_status()
        sset("classic_last_response", r.json())
        sset("fetch_error", None)
    except requests.RequestException as e:
        sset("classic_last_response", None)
        # escaped: the exception text carries the request URL, i.e. the shopper's query
        sset("fetch_error", f"Could not reach the search service: {escape_md(e)}")


if submitted:
    # Sticky band: a search sent from the bottom of a page lands at the top, this run
    request_scroll()
    apply_scroll()
    sset("query", submitted)
    sset("page", 1)
    sset("view", "search")
    sset("return_card", None)
    _fetch(force=True)

_settings_now = (top_k, show_sponsored, show_recs)
previous = sget("classic_settings")
if previous != _settings_now:
    if previous is not None and tuple(previous[:2]) != _settings_now[:2]:
        sset("page", 1)
    if sget("view") == "detail":
        sset("detail_response", None)
    sset("classic_settings", _settings_now)


def _render_results():
    if sget("query"):
        with st.spinner("Searching…"):
            _fetch()
    if sget("fetch_error"):
        st.error(sget("fetch_error"))
    data = sget("classic_last_response")
    if not data:
        if not sget("query"):
            st.caption(
                "Classic keyword matching (BM25) — no AI, no query rewriting. Fast and "
                "familiar, for when you know exactly what you want. To describe what "
                "you need in your own words, switch to **AI Search**."
            )
        return

    if data.get("no_match"):
        st.info(
            "No products matched those keywords. Try fewer or more general "
            "words — or switch to **AI Search** to describe what you need."
        )
        return

    results = data.get("results", [])
    matched = int(data.get("num_matched", 0) or 0)
    page = int(data.get("page", 1) or 1)
    page_size = int(data.get("page_size", top_k) or top_k)
    total = data.get("total_results")
    start = (page - 1) * page_size + 1
    results_bar(results_header_html(
        start, start + len(results) - 1,
        int(total) if total is not None else len(results), sget("query") or "",
    ))
    st.html(heading_html("Results"))
    if total is None or matched > int(total):
        st.caption(f"**{matched:,}** products contain these keywords.")
    if show_debug:
        st.caption(
            f"Developer details — top BM25 relevance "
            f"{data.get('top_relevance', 0):.2f} · top-result term coverage "
            f"{data.get('top_coverage', 0):.0%}"
        )
    if not results:
        st.warning("No products found.")
        return

    render_product_grid(results, related=show_recs)
    pagination(page, data.get("total_pages", 1))


if sget("view") == "detail" and sget("detail_index") is not None:
    render_detail_page(show_recs=show_recs)
else:
    _render_results()
