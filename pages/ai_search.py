"""AI Search page — GET /search: LLM intents, embedding retrieval, LLM rerank."""

import os

import requests
import streamlit as st

import ui_cart
from ui_common import (
    bundle_ui,
    inject_css,
    render_detail_page,
    render_grid_card,
    set_surface,
    sget,
    sinit,
    sset,
)

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
SEARCH_URL = f"{BACKEND}/search"

set_surface("ai")   # own namespace: independent of Keyword Search / Assistant
inject_css()

st.title("🧠 AI Search")
st.caption(
    "Describe what you need in your own words — the search understands the "
    "context, not just the keywords. Try *“something breathable for a humid "
    "day”* or *“ingredients to make pancakes”*."
)

sinit({
    "query": "",
    "page": 1,
    "last_response": None,
    "view": "search",
    "detail_index": None,
    "detail_response": None,
    "settings": None,
})

with st.sidebar:
    st.header("Search settings")
    top_k = st.slider("Results per page", 3, 30, 12)
    rerank = st.toggle(
        "AI ranking",
        value=True,
        help="Let the LLM re-rank results and explain why each one matched. "
        "Turn off to see raw embedding-similarity order.",
    )
    show_sponsored = st.toggle(
        "Sponsored results",
        value=True,
        help="Paid placements. Always labelled, and only shown when they are "
        "genuinely relevant to the query.",
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
        help="Per-stage timings and the raw ranking scores (rerank / Bayesian "
        "rating / embedding similarity). Off by default to keep demos clean.",
    )
    # ui_common reads this global key to decide whether cards show raw scores
    st.session_state["show_dev_details"] = show_debug
    st.caption("Repeat searches are cached, so they return instantly.")
    st.caption("Click any product to see its page and recommendations.")
    st.divider()
    # st.page_link raises KeyError 'url_pathname' outside full page context
    st.markdown(f"### 🛒 Cart ({ui_cart.count()})")
    if ui_cart.count():
        st.caption("Open **Cart** in the nav to review or check out.")


def _fetch():
    """Run a search for the current query + page (recs live on the product page)."""
    q = sget("query")
    if not q:
        return
    try:
        r = requests.get(
            SEARCH_URL,
            params={
                "query": q,
                "top_k": top_k,
                "page": sget("page"),
                "rerank": str(rerank).lower(),
                "recommend": "false",
                "sponsored": str(show_sponsored).lower(),
            },
            timeout=90,
        )
        r.raise_for_status()
        sset("last_response", r.json())
    except requests.RequestException as e:
        st.error(f"Could not reach the search service: {e}")
        sset("last_response", None)


query_input = st.text_input(
    "What are you looking for?",
    placeholder="e.g. breathable outfit for a humid day, ingredients for pancakes",
)

if st.button("🔍 Search", type="primary") and query_input:
    with st.spinner("Understanding your request…"):
        sset("query", query_input)
        sset("page", 1)
        sset("view", "search")
        _fetch()

_settings = (top_k, rerank, show_sponsored, show_recs)
if sget("settings") != _settings:
    if sget("view") == "detail":
        sset("detail_response", None)
        # invalidate the list too, else "Back to results" shows OLD-settings results
        sset("page", 1)
        sset("last_response", None)
    elif sget("query") and sget("last_response") is not None:
        sset("page", 1)
        _fetch()
    sset("settings", _settings)


def _render_search_results():
    data = sget("last_response")
    if not data and sget("query"):
        with st.spinner("Refreshing results…"):
            _fetch()
        data = sget("last_response")
    if not data:
        return

    timings = data.get("latency_ms", {})
    ai_ranking_fell_back = (
        data.get("rerank_succeeded") is False and data.get("rerank_requested")
    )
    if show_debug and (timings or ai_ranking_fell_back):
        with st.sidebar:
            st.divider()
            st.caption("**Developer details — last search**")
            for stage, ms in timings.items():
                st.caption(f"{stage}: {ms} ms")
            if timings:
                st.caption(f"total: {sum(timings.values())} ms")
            if ai_ranking_fell_back:
                st.caption("⚠️ AI ranking unavailable — similarity order used")

    for err in data.get("errors", []):
        stage = err.get("stage")
        code = err.get("code", "?")
        if stage == "rerank":
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

    if data.get("no_match"):
        if data.get("interpreted_as"):
            st.subheader("🧠 What we searched for")
            for intent in data.get("interpreted_as", []):
                st.markdown(f"- {intent}")
            st.divider()
        st.info(
            data.get("message")
            or "No matching products found. Try different or more general words."
        )
        return

    st.subheader("🧠 What we searched for")
    for intent in data.get("interpreted_as", []):
        st.markdown(f"- {intent}")
    constraints = data.get("constraints") or []
    if constraints:
        chips = " · ".join(
            f"**{c.get('value')}** ({c.get('type')})" for c in constraints
        )
        st.caption(f"Filters applied from your wording: {chips}")

    st.divider()
    results = data.get("results", [])
    page = data.get("page", 1)
    page_size = data.get("page_size", top_k)
    total_results = data.get("total_results", len(results))
    # must come from the API: source_intent is set on every result, every query
    # bundle_type is recipe | outfit | setup | None (one card per component)
    bundle_type = data.get("bundle_type")
    bui = bundle_ui(bundle_type)
    st.subheader(bui["heading"] if bui else "🛍️ Results")
    if bui:
        st.caption(bui["blurb"])
    if not results:
        st.warning("No products found.")
        return

    start_rank = (page - 1) * page_size + 1
    cols_per_row = 3
    for row_start in range(0, len(results), cols_per_row):
        row_items = results[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (offset, product) in zip(cols, enumerate(row_items)):
            with col:
                render_grid_card(
                    product,
                    rank=start_rank + row_start + offset,
                    bundle_type=bundle_type,
                )

    total_pages = data.get("total_pages", 1)
    prev_col, mid_col, next_col = st.columns([1, 3, 1])
    if prev_col.button("⬅ Previous", disabled=not data.get("has_prev"), key="pg_prev"):
        sset("page", page - 1)
        _fetch()
        st.rerun()
    mid_col.markdown(
        f"<div style='text-align:center'>Page <b>{page}</b> of <b>{total_pages}</b>"
        f" · {total_results} matches</div>",
        unsafe_allow_html=True,
    )
    if next_col.button("Next ➡", disabled=not data.get("has_next"), key="pg_next"):
        sset("page", page + 1)
        _fetch()
        st.rerun()


if sget("view") == "detail" and sget("detail_index") is not None:
    render_detail_page(show_recs=show_recs)
else:
    _render_search_results()
