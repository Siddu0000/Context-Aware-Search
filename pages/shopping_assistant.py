"""Shopping Assistant page — on-site helper bot over POST /chat."""

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
CHAT_URL = f"{BACKEND}/chat"

set_surface("assistant")   # own namespace, isolated from the other surfaces
inject_css()

st.title("💬 Shopping Assistant")
st.caption(
    "Chat the way you would with a shop assistant — it remembers the "
    "conversation, so you can refine your request as you go."
)

_DEFAULTS = {
    "chat_history": [],          # [{role, content}] — text transcript only
    "assistant_results": [],     # the ONE current results panel
    "assistant_query": "",       # the search behind the current panel
    "assistant_exclusions": [],  # active refine-exclusions on that search
    "assistant_meta": {},        # constraints / intents / bundle / errors
    "query": "",                 # this surface's product-page context
    "view": "search",
    "detail_index": None,
    "detail_response": None,
}
sinit(_DEFAULTS)

with st.sidebar:
    st.header("Assistant settings")
    top_k = st.slider("Products to show", 2, 12, 6)
    show_recs = st.toggle(
        "Product recommendations",
        value=True,
        help="Show 'frequently bought together' on a product's page.",
    )
    show_debug = st.toggle(
        "Developer details",
        value=False,
        help="Show the resolved search query, extracted constraints and raw "
        "ranking scores behind the current results.",
    )
    st.session_state["show_dev_details"] = show_debug
    if st.button("🗑️ Clear conversation"):
        for k in ("chat_history", "assistant_results", "assistant_query",
                  "assistant_exclusions", "assistant_meta"):
            sset(k, _DEFAULTS[k])
        st.rerun()
    st.caption("New searches replace the results; refinements update them in place.")
    st.divider()
    # st.page_link raises KeyError 'url_pathname' outside full page context
    st.markdown(f"### 🛒 Cart ({ui_cart.count()})")
    if ui_cart.count():
        st.caption("Open **Cart** in the nav to review or check out.")


def _send(message: str):
    """POST the turn (+ history + refinement context); update session state."""
    # snapshot history BEFORE appending: sending the turn twice muddies new_topic
    history = [
        {"role": t["role"], "content": t["content"]}
        for t in sget("chat_history")
    ]
    user_turn = {"role": "user", "content": message}
    sget("chat_history").append(user_turn)

    try:
        r = requests.post(
            CHAT_URL,
            json={
                "message": message,
                "history": history,
                "top_k": top_k,
                "last_search_query": sget("assistant_query"),
                "exclusions": sget("assistant_exclusions"),
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        sget("chat_history").append(
            {"role": "assistant",
             "content": f"Sorry — I couldn't reach the search service. ({e})"}
        )
        return

    reply_turn = {"role": "assistant", "content": data.get("reply", "")}
    action = data.get("action")

    if action in {"search", "refine"} and data.get("new_topic"):
        # unrelated goal: drop the finished topic's transcript, start a fresh page
        sset("chat_history", [user_turn, reply_turn])
    else:
        sget("chat_history").append(reply_turn)

    if action in {"search", "refine"}:
        # ONE results panel: replaced on a new search, updated on a refine
        sset("assistant_results", data.get("results", []))
        sset("assistant_query", data.get("search_query", ""))
        sset("assistant_exclusions", data.get("exclusions", []))
        sset("assistant_meta", {
            "constraints": data.get("constraints", []),
            "interpreted_as": data.get("interpreted_as", []),
            "errors": data.get("errors", []),
            "action": action,
            "new_topic": bool(data.get("new_topic")),
            "bundle_type": data.get("bundle_type"),
        })
        sset("query", data.get("search_query", "") or message)


def _render_transcript():
    if not sget("chat_history"):
        with st.chat_message("assistant"):
            st.markdown("Hi! What are you shopping for today?")
    for turn in sget("chat_history"):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])


def _render_results_panel():
    results = sget("assistant_results")
    if not results:
        return
    st.divider()
    n = len(results)
    excl = sget("assistant_exclusions")
    meta_now = sget("assistant_meta") or {}
    bundle_type = meta_now.get("bundle_type")
    bui = bundle_ui(bundle_type)
    # bundles render one card per component with option tabs, as on AI Search
    st.subheader(f"{bui['heading']} ({n})" if bui else f"🛍️ Current results ({n})")
    if bui:
        st.caption(bui["blurb"])
    bits = []
    if sget("assistant_query"):
        bits.append(f"showing: *{sget('assistant_query')}*")
    if excl:
        bits.append("excluding: " + ", ".join(f"*{e}*" for e in excl))
    if bits:
        st.caption(" · ".join(bits))

    meta = sget("assistant_meta") or {}
    if show_debug:
        dbg = []
        if meta.get("interpreted_as"):
            dbg.append("intents: " + " · ".join(meta["interpreted_as"]))
        if meta.get("constraints"):
            dbg.append("constraints: " + ", ".join(
                f"{c.get('value')} ({c.get('type')})" for c in meta["constraints"]
            ))
        if dbg:
            st.caption(" — ".join(dbg))
    for err in meta.get("errors") or []:
        st.caption(f"⚠️ {err.get('stage')} unavailable (error {err.get('code')})")

    cols_per_row = 3
    for row_start in range(0, len(results), cols_per_row):
        row = results[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (offset, product) in zip(cols, enumerate(row)):
            with col:
                render_grid_card(
                    product, rank=row_start + offset + 1, key_ns="chatpanel",
                    bundle_type=bundle_type,
                )


if sget("view") == "detail" and sget("detail_index") is not None:
    if st.button("⬅ Back to the conversation"):
        sset("view", "search")
        st.rerun()
    render_detail_page(show_recs=show_recs, show_back_button=False)
else:
    _render_transcript()
    _render_results_panel()
    if prompt := st.chat_input("Ask me anything about what you're shopping for…"):
        with st.spinner("Looking through the catalogue…"):
            _send(prompt)
        st.rerun()
