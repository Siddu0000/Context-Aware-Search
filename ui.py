"""Streamlit UI.

Adds three things over the original:
  1. Shows each product's reason (from the reranker).
  2. Shows per-stage latency for the request.
  3. Captures thumbs-up / thumbs-down feedback, posted to /feedback.
"""

import os

import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
SEARCH_URL = f"{BACKEND}/search"
FEEDBACK_URL = f"{BACKEND}/feedback"

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
    st.divider()
    st.caption("Toggle rerank off to see raw HyDE retrieval.")


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
                params={"query": query, "top_k": top_k, "rerank": str(rerank).lower()},
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

    # Latency strip
    timings = data.get("latency_ms", {})
    if timings:
        cols = st.columns(len(timings))
        for c, (stage, ms) in zip(cols, timings.items()):
            c.metric(label=stage, value=f"{ms} ms")

    st.subheader("🧠 Interpreted Intents (HyDE)")
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
                    if product.get("color"):
                        meta.append(f"Color: {product['color']}")
                    if product.get("price") is not None:
                        meta.append(f"Price: ${product['price']}")
                    st.markdown(" · ".join(meta))

                    if product.get("rerank_score") is not None:
                        st.caption(
                            f"Rerank score: {product['rerank_score']}/100"
                            f" · embedding sim: {product.get('score', 0):.3f}"
                        )
                    else:
                        st.caption(f"Embedding sim: {product.get('score', 0):.3f}")

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
