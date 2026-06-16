"""Streamlit UI.

Over the bare results list this adds:
  1. Each product's reason (from the reranker).
  2. Per-stage latency for the request (debug toggle).
  3. Thumbs-up / thumbs-down feedback, posted to /feedback.
  4. Pagination (Prev/Next) over results beyond the first 10.
  5. A clearly-labelled "Sponsored" section, kept visually separate from
     organic results (paid placement must never look like earned placement).
  6. A "Frequently bought together / You might prefer" cross-sell + upsell
     panel under the results (page 1 only).
"""

import math
import os

import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
SEARCH_URL = f"{BACKEND}/search"
FEEDBACK_URL = f"{BACKEND}/feedback"


def _is_nan(x) -> bool:
    """True if x is float NaN. None and other types return False."""
    try:
        return isinstance(x, float) and math.isnan(x)
    except (TypeError, ValueError):
        return False


st.set_page_config(
    page_title="Context-Aware Agentic Search",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Context-Aware Agentic Search")
st.caption(
    "LLM query expansion + LLM rerank with reasoning. "
    "Searches what you *meant*, not what you typed."
)

# --- Session state ---------------------------------------------------------
for _k, _v in {"query": "", "page": 1, "last_response": None}.items():
    st.session_state.setdefault(_k, _v)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Results per page", 3, 30, 10)
    rerank = st.toggle("Enable LLM rerank", value=True)
    show_sponsored = st.toggle("Show sponsored", value=True)
    show_recs = st.toggle("Show recommendations", value=True)
    show_debug = st.toggle(
        "Show debug metrics",
        value=False,
        help="Per-stage latency and pipeline diagnostics. Off by default so "
        "they don't clutter the results during a demo.",
    )
    st.caption("Cache is disabled — every search hits the LLM directly.")
    st.caption("Toggle rerank off to see raw retrieval.")


def _fetch():
    """Run a search for the current session query + page; store the response."""
    q = st.session_state["query"]
    if not q:
        return
    try:
        r = requests.get(
            SEARCH_URL,
            params={
                "query": q,
                "top_k": top_k,
                "page": st.session_state["page"],
                "rerank": str(rerank).lower(),
                "recommend": str(show_recs).lower(),
                "sponsored": str(show_sponsored).lower(),
            },
            timeout=60,
        )
        r.raise_for_status()
        st.session_state["last_response"] = r.json()
    except requests.RequestException as e:
        st.error(f"Backend error: {e}")
        st.session_state["last_response"] = None


def _post_feedback(query: str, product_title: str, rating: int, rank: int, reason: str):
    try:
        requests.post(
            FEEDBACK_URL,
            json={
                "query": query,
                "product_title": product_title,
                "rating": rating,
                "rank": rank,
                "reason": reason,
            },
            timeout=5,
        )
    except requests.RequestException as e:
        st.warning(f"Could not record feedback: {e}")


def _rating_line(product: dict):
    """Render the NaN-safe rating line for a product, if present."""
    rating = product.get("average_rating")
    rating_n = product.get("rating_number")
    rating_valid = rating is not None and not _is_nan(rating)
    count_valid = rating_n is not None and not _is_nan(rating_n)
    if rating_valid:
        try:
            rating_val = float(rating)
            if 0 <= rating_val <= 5:
                stars = "⭐" * int(round(rating_val))
                line = f"{stars} **{rating_val:.1f}/5**"
                if count_valid:
                    line += f" — based on **{int(float(rating_n)):,}** ratings"
                else:
                    line += " — no rating count available"
                st.markdown(line)
        except (TypeError, ValueError):
            pass
    elif count_valid:
        st.markdown(f"_{int(float(rating_n)):,} ratings (no average)_")


def _render_card(product: dict, rank: int, *, feedback: bool = True, sponsored: bool = False):
    """Render one full-width product card."""
    with st.container(border=True):
        left, right = st.columns([2, 5])
        with left:
            if product.get("img_url"):
                st.image(product["img_url"], width=200)
        with right:
            # Defensive: badge if the caller says so OR the item is flagged
            # sponsored — paid placement must never render as organic.
            if sponsored or product.get("is_sponsored"):
                st.markdown(
                    f":orange[**⭐ SPONSORED**] · _by {product.get('sponsor', 'a partner')}_"
                )
            st.markdown(f"### {rank}. {product.get('Product_title', 'Untitled')}")
            meta = []
            if product.get("bsns_vrtcl_name"):
                meta.append(f"**{product['bsns_vrtcl_name']}**")
            if product.get("categ_lvl2_name"):
                meta.append(product["categ_lvl2_name"])
            if product.get("store"):
                meta.append(f"by {product['store']}")
            if product.get("color"):
                meta.append(f"Color: {product['color']}")
            if product.get("price") is not None and not _is_nan(product.get("price")):
                meta.append(f"**${product['price']}**")
            st.markdown(" · ".join(meta))

            _rating_line(product)

            # Scores (organic only — sponsored items aren't relevance-scored).
            if not sponsored:
                final = product.get("final_score")
                rerank_val = product.get("rerank_score")
                embed = product.get("score", 0)
                if final is not None and not _is_nan(final):
                    bayes = product.get("bayesian_rating")
                    bayes_str = f" · Bayes rating: {bayes}" if bayes else ""
                    st.caption(
                        f"**Final score: {final}/100**  "
                        f"(Rerank: {rerank_val}/100{bayes_str} · embed: {embed:.3f})"
                    )
                elif rerank_val is not None:
                    st.caption(f"Rerank: {rerank_val}/100 · embed: {embed:.3f}")
                else:
                    st.caption(f"Embedding sim: {embed:.3f}")

            if product.get("reason"):
                st.markdown(f"💡 _{product['reason']}_")

            if feedback:
                # Key on catalog_index (globally unique, stable across pages).
                # Truncated titles collide — thousands share a 20-char prefix.
                kid = product.get("catalog_index", rank)
                fb_cols = st.columns([1, 1, 8])
                if fb_cols[0].button("👍", key=f"fb_up_{kid}"):
                    _post_feedback(
                        st.session_state["query"],
                        product["Product_title"],
                        +1,
                        rank,
                        product.get("reason", ""),
                    )
                    st.toast("Thanks — feedback recorded.")
                if fb_cols[1].button("👎", key=f"fb_down_{kid}"):
                    _post_feedback(
                        st.session_state["query"],
                        product["Product_title"],
                        -1,
                        rank,
                        product.get("reason", ""),
                    )
                    st.toast("Thanks — feedback recorded.")


def _render_mini(item: dict, key: str):
    """Compact card for the cross-sell / upsell strip."""
    with st.container(border=True):
        if item.get("img_url"):
            st.image(item["img_url"], width=120)
        title = item.get("Product_title", "Untitled")
        st.markdown(f"**{title[:60]}{'…' if len(title) > 60 else ''}**")
        if item.get("price") is not None and not _is_nan(item.get("price")):
            st.caption(f"${item['price']}")
        if item.get("recommend_reason"):
            st.caption(f"_{item['recommend_reason']}_")
        if st.button("🔍 Search this", key=key):
            st.session_state["query"] = title
            st.session_state["page"] = 1
            _fetch()
            st.rerun()


# --- Search bar ------------------------------------------------------------
query_input = st.text_input(
    "What are you looking for?",
    placeholder="e.g. casual outfit, wireless earbuds, healthy breakfast cereal",
)

if st.button("🔍 Search", type="primary") and query_input:
    with st.spinner("Understanding your intent..."):
        st.session_state["query"] = query_input
        st.session_state["page"] = 1
        _fetch()

# If the user changed any sidebar setting while a search is active, re-run it
# from page 1 — otherwise the cached results (and their rank numbering) would
# no longer match the settings now shown in the sidebar. On first load (no
# query yet) we just record the settings without fetching.
_settings = (top_k, rerank, show_sponsored, show_recs)
if st.session_state.get("settings") != _settings:
    if st.session_state.get("query") and st.session_state.get("last_response") is not None:
        st.session_state["page"] = 1
        _fetch()
    st.session_state["settings"] = _settings

# --- Results ---------------------------------------------------------------
data = st.session_state.get("last_response")
if data:
    timings = data.get("latency_ms", {})
    if show_debug:
        with st.sidebar:
            st.divider()
            st.caption("**Debug — last request**")
            for stage, ms in timings.items():
                st.caption(f"{stage}: {ms} ms")
            if timings:
                st.caption(f"total: {sum(timings.values())} ms")
            if data.get("rerank_succeeded") is False and data.get("rerank_requested"):
                st.caption("⚠️ rerank fell back (LLM unavailable)")

    if data.get("rerank_succeeded") is False and data.get("rerank_requested"):
        st.caption("⚠️ Showing embedding-ranked results (reranker was unavailable).")

    st.subheader("🧠 Interpreted intents")
    for intent in data.get("interpreted_as", []):
        st.markdown(f"- {intent}")

    # --- Sponsored (page 1, visually separated) ---------------------------
    sponsored_items = data.get("sponsored", []) if show_sponsored else []
    if sponsored_items:
        st.divider()
        st.subheader("⭐ Sponsored")
        st.caption("Paid placements — ranked separately from organic results.")
        for i, product in enumerate(sponsored_items, start=1):
            _render_card(product, i, feedback=False, sponsored=True)

    # --- Organic results --------------------------------------------------
    st.divider()
    st.subheader("🛍️ Recommended Products")

    results = data.get("results", [])
    page = data.get("page", 1)
    page_size = data.get("page_size", top_k)
    if not results:
        st.warning("No products found.")
    else:
        start_rank = (page - 1) * page_size + 1
        for offset, product in enumerate(results):
            _render_card(product, start_rank + offset, feedback=True)

        # --- Pagination controls ------------------------------------------
        total_pages = data.get("total_pages", 1)
        total_results = data.get("total_results", len(results))
        prev_col, mid_col, next_col = st.columns([1, 3, 1])
        if prev_col.button("⬅ Prev", disabled=not data.get("has_prev"), key="pg_prev"):
            st.session_state["page"] = page - 1
            _fetch()
            st.rerun()
        mid_col.markdown(
            f"<div style='text-align:center'>Page <b>{page}</b> of <b>{total_pages}</b>"
            f" · {total_results} results</div>",
            unsafe_allow_html=True,
        )
        if next_col.button("Next ➡", disabled=not data.get("has_next"), key="pg_next"):
            st.session_state["page"] = page + 1
            _fetch()
            st.rerun()

    # --- Cross-sell / upsell (page 1 only) --------------------------------
    if show_recs and page == 1:
        recs = data.get("recommendations") or {}
        cross = recs.get("cross_sell", [])
        upsell = recs.get("upsell", [])
        if cross:
            st.divider()
            st.subheader("🧺 Frequently bought together")
            cols = st.columns(len(cross))
            for j, (col, item) in enumerate(zip(cols, cross)):
                with col:
                    _render_mini(item, key=f"cross_{item.get('catalog_index', j)}")
        if upsell:
            st.divider()
            st.subheader("⬆️ You might prefer")
            cols = st.columns(min(len(upsell), 3) or 1)
            for j, (col, item) in enumerate(zip(cols, upsell)):
                with col:
                    _render_mini(item, key=f"uprec_{item.get('catalog_index', j)}")
