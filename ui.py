"""Streamlit UI.

Adds three things over the original:
  1. Shows each product's reason (from the reranker).
  2. Shows per-stage latency for the request.
  3. Captures thumbs-up / thumbs-down feedback, posted to /feedback.
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
    "HyDE-style query expansion + LLM rerank with reasoning. "
    "Searches what you *meant*, not what you typed."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results", 3, 30, 10)
    rerank = st.toggle("Enable LLM rerank", value=True)
    show_debug = st.toggle(
        "Show debug metrics",
        value=False,
        help="Per-stage latency and pipeline diagnostics. Off by default so "
        "they don't clutter the results during a demo.",
    )
    st.caption("Cache is disabled — every search hits the LLM directly.")
    st.caption("Toggle rerank off to see raw retrieval.")


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


query = st.text_input(
    "What are you looking for?",
    placeholder="e.g. casual outfit, wireless earbuds, healthy breakfast cereal",
)

if st.button("🔍 Search", type="primary") and query:
    with st.spinner("Understanding your intent..."):
        try:
            r = requests.get(
                SEARCH_URL,
                params={
                    "query": query,
                    "top_k": top_k,
                    "rerank": str(rerank).lower(),
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            st.error(f"Backend error: {e}")
            st.stop()

    st.session_state["last_response"] = data

if "last_response" in st.session_state:
    data = st.session_state["last_response"]

    # Debug metrics moved OFF the results panel (Niharika 2026-06-11: "push
    # them to the left / discard them"). Now they only render in the sidebar
    # when the developer flips the debug toggle — invisible during a demo.
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

    # A single subtle warning stays in the main panel ONLY if rerank silently
    # fell back — the user should know results are embedding-only in that case.
    if data.get("rerank_succeeded") is False and data.get("rerank_requested"):
        st.caption("⚠️ Showing embedding-ranked results (reranker was unavailable).")

    st.subheader("🧠 Interpreted intents")
    for intent in data.get("interpreted_as", []):
        st.markdown(f"- {intent}")

    st.divider()
    st.subheader(
        "🛍️ Recommended Products"
        + (" — reranked" if data.get("rerank_enabled") else " — raw retrieval")
    )

    results = data.get("results", [])
    if not results:
        st.warning("No products found.")
    else:
        for rank, product in enumerate(results, start=1):
            with st.container(border=True):
                left, right = st.columns([2, 5])
                with left:
                    if product.get("img_url"):
                        st.image(product["img_url"], width=200)
                with right:
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

                    # --- Rating: dedicated line, NaN-safe -----------------
                    # Use pd.isna() so NaN doesn't slip past the None check.
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
                                    n = int(float(rating_n))
                                    line += f" — based on **{n:,}** ratings"
                                else:
                                    line += " — no rating count available"
                                st.markdown(line)
                        except (TypeError, ValueError):
                            pass
                    elif count_valid:
                        # Edge case: count present but average missing
                        st.markdown(f"_{int(float(rating_n)):,} ratings (no average)_")

                    # --- Scores -----------------------------------------------
                    final = product.get("final_score")
                    rerank = product.get("rerank_score")
                    embed = product.get("score", 0)
                    if final is not None and not _is_nan(final):
                        bayes = product.get("bayesian_rating")
                        bayes_str = f" · Bayes rating: {bayes}" if bayes else ""
                        st.caption(
                            f"**Final score: {final}/100**  "
                            f"(Rerank: {rerank}/100{bayes_str} · embed: {embed:.3f})"
                        )
                    elif rerank is not None:
                        st.caption(
                            f"Rerank: {rerank}/100 · embed: {embed:.3f}"
                        )
                    else:
                        st.caption(f"Embedding sim: {embed:.3f}")

                    if product.get("reason"):
                        st.markdown(f"💡 _{product['reason']}_")

                    # Feedback row
                    fb_cols = st.columns([1, 1, 8])
                    if fb_cols[0].button("👍", key=f"up_{rank}"):
                        _post_feedback(
                            data["user_query"],
                            product["Product_title"],
                            +1,
                            rank,
                            product.get("reason", ""),
                        )
                        st.toast("Thanks — feedback recorded.")
                    if fb_cols[1].button("👎", key=f"down_{rank}"):
                        _post_feedback(
                            data["user_query"],
                            product["Product_title"],
                            -1,
                            rank,
                            product.get("reason", ""),
                        )
                        st.toast("Thanks — feedback recorded.")
