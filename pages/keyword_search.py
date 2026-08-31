"""Keyword Search page — pure BM25 via GET /keyword_search, no LLM, no AI fallback."""

import os

import requests
import streamlit as st

import ui_cart
from ui_common import (
    inject_css,
    render_detail_page,
    render_grid_card,
    set_surface,
    sget,
    sinit,
    sset,
)

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
KEYWORD_SEARCH_URL = f"{BACKEND}/keyword_search"

set_surface("keyword")   # own namespace, isolated from AI Search / Assistant
inject_css()

st.title("🔍 Keyword Search")
st.caption(
    "Classic keyword matching (BM25) — no AI, no query rewriting. Fast and "
    "familiar, for when you know exactly what you want. To describe what you "
    "need in your own words, switch to **AI Search**."
)

sinit({
    "query": "",
    "view": "search",
    "detail_index": None,
    "detail_response": None,
    "classic_last_response": None,
    "classic_settings": None,
})

with st.sidebar:
    st.header("Search settings")
    top_k = st.slider("Number of results", 3, 30, 12)
    show_sponsored = st.toggle(
        "Sponsored results",
        value=True,
        help="Paid placements. Always labelled; only shown when they actually "
        "match the keywords.",
    )
    show_recs = st.toggle(
        "Product recommendations",
        value=True,
        help="Show 'frequently bought together' and better-rated alternatives "
        "on a product's page.",
    )
    show_debug = st.toggle(
        "Developer details",
        value=False,
        help="Show the raw BM25 / Best-Match ranking scores on each card.",
    )
    st.session_state["show_dev_details"] = show_debug
    st.caption("No LLM calls on this page — results are instant.")
    st.divider()
    # st.page_link raises KeyError 'url_pathname' outside full page context
    st.markdown(f"### 🛒 Cart ({ui_cart.count()})")
    if ui_cart.count():
        st.caption("Open **Cart** in the nav to review or check out.")


def _fetch():
    q = sget("query")
    if not q:
        return
    try:
        r = requests.get(
            KEYWORD_SEARCH_URL,
            params={
                "query": q,
                "top_k": top_k,
                "sponsored": str(show_sponsored).lower(),
            },
            timeout=20,
        )
        r.raise_for_status()
        sset("classic_last_response", r.json())
    except requests.RequestException as e:
        st.error(f"Could not reach the search service: {e}")
        sset("classic_last_response", None)


query_input = st.text_input(
    "Search by keyword",
    placeholder="e.g. men's leather belt, wireless earbuds, chocolate chips",
)
if st.button("🔍 Search", type="primary") and query_input:
    sset("query", query_input)
    sset("view", "search")
    _fetch()

_settings = (top_k, show_sponsored, show_recs)
if sget("classic_settings") != _settings:
    if sget("view") == "detail":
        sset("detail_response", None)
    elif sget("query") and sget("classic_last_response") is not None:
        _fetch()
    sset("classic_settings", _settings)


def _render_results():
    data = sget("classic_last_response")
    if not data and sget("query"):
        with st.spinner("Searching…"):
            _fetch()
        data = sget("classic_last_response")
    if not data:
        return

    if data.get("no_match"):
        st.info(
            "No products matched those keywords. Try fewer or more general "
            "words — or switch to **AI Search** to describe what you need."
        )
        return

    matched = data.get("num_matched", 0)
    st.caption(f"**{matched:,}** products matched these keywords.")
    if show_debug:
        st.caption(
            f"Developer details — top BM25 relevance "
            f"{data.get('top_relevance', 0):.2f} · top-result term coverage "
            f"{data.get('top_coverage', 0):.0%}"
        )
    st.divider()
    st.subheader("🛍️ Results")
    results = data.get("results", [])
    if not results:
        st.warning("No products found.")
        return

    cols_per_row = 3
    for row_start in range(0, len(results), cols_per_row):
        row_items = results[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (offset, product) in zip(cols, enumerate(row_items)):
            with col:
                render_grid_card(product, rank=row_start + offset + 1)


if sget("view") == "detail" and sget("detail_index") is not None:
    render_detail_page(show_recs=show_recs)
else:
    _render_results()
