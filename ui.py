"""Streamlit UI.

Over the bare results list this adds:
  1. Each product's reason (from the reranker).
  2. Per-stage latency for the request (debug toggle).
  3. Thumbs-up / thumbs-down feedback, posted to /feedback.
  4. Pagination (Prev/Next) over results beyond the first page.
  5. A clearly-labelled "Sponsored" section, kept visually separate from
     organic results (paid placement must never look like earned placement).
  6. Clickable products: clicking any result opens a PRODUCT DETAIL page
     (Amazon-style) that shows that product plus its own cross-sell / upsell
     recommendations. Recommendations live on the detail page, not at the
     bottom of the results list.
"""

import math
import os

import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
SEARCH_URL = f"{BACKEND}/search"
UNIFIED_URL = f"{BACKEND}/unified_search"
PRODUCT_URL = f"{BACKEND}/product"
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

st.markdown(
    """
    <style>
    .cas-thumb {
        width: 100%; height: 180px; object-fit: cover;
        border-radius: 8px; display: block;
    }
    .cas-thumb-empty {
        width: 100%; height: 180px; border-radius: 8px;
        background: #f0f0f0; display: flex; align-items: center;
        justify-content: center; color: #999; font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_defaults = {
    "query": "",
    "page": 1,
    "last_response": None,
    "view": "search",
    "detail_index": None,
    "detail_response": None,
}
for _k, _v in _defaults.items():
    st.session_state.setdefault(_k, _v)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Results per page", 3, 30, 12)
    rerank = st.toggle("Enable LLM rerank", value=True)
    show_sponsored = st.toggle("Show sponsored", value=True)
    hybrid_mode = st.toggle(
        "🔀 Hybrid mode (keyword-first)",
        value=False,
        help="Swiggy model: run the keyword engine first; fall back to "
        "context-aware search only when keyword search misses. Shows which "
        "path served the query.",
    )
    show_recs = st.toggle("Show recommendations", value=True, help="On product pages.")
    show_debug = st.toggle(
        "Show debug metrics",
        value=False,
        help="Per-stage latency and pipeline diagnostics. Off by default so "
        "they don't clutter the results during a demo.",
    )
    st.caption("Repeat searches are cached for speed; failed runs are not cached.")
    st.caption("Click any product to open its page + recommendations.")


def _fetch():
    """Run a search for the current session query + page; store the response.

    Recommendations are NOT requested here — they live on the product detail
    page now, so the list view skips that LLM call.
    """
    q = st.session_state["query"]
    if not q:
        return
    if hybrid_mode:
        try:
            r = requests.get(
                UNIFIED_URL,
                params={
                    "query": q,
                    "top_k": top_k,
                    "rerank": str(rerank).lower(),
                    "sponsored": str(show_sponsored).lower(),
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            # Normalize the hybrid response into the shape the results view
            # expects. The intent path passes through the CAS context
            # (interpreted_as / errors / no_match); the keyword path has none.
            st.session_state["last_response"] = {
                "results": results,
                "no_match": data.get("no_match", not results),
                "message": data.get("message")
                or ("" if results else "No matching products found."),
                "page": 1,
                "page_size": top_k,
                "total_results": len(results),
                "total_pages": 1,
                "has_prev": False,
                "has_next": False,
                "sponsored": [r_ for r_ in results if r_.get("is_sponsored")],
                "recommendations": {"cross_sell": [], "upsell": []},
                "user_query": q,
                "interpreted_as": data.get("interpreted_as", []),
                "errors": data.get("errors", []),
                "rerank_requested": data.get("rerank_requested"),
                "rerank_succeeded": data.get("rerank_succeeded"),
                "unified_path": data.get("path"),
                "unified_reason": data.get("reason"),
                "keyword_coverage": data.get("keyword_coverage"),
                # Present on the intent path (CAS timings); {} on keyword hits,
                # which legitimately have no LLM stage timings.
                "latency_ms": data.get("latency_ms", {}),
            }
        except requests.RequestException as e:
            st.error(f"Backend error: {e}")
            st.session_state["last_response"] = None
        return
    try:
        r = requests.get(
            SEARCH_URL,
            params={
                "query": q,
                "top_k": top_k,
                "page": st.session_state["page"],
                "rerank": str(rerank).lower(),
                "recommend": "false",
                "sponsored": str(show_sponsored).lower(),
            },
            timeout=60,
        )
        r.raise_for_status()
        st.session_state["last_response"] = r.json()
    except requests.RequestException as e:
        st.error(f"Backend error: {e}")
        st.session_state["last_response"] = None


def _fetch_product(catalog_index: int):
    """Load a single product + its recommendations from /product."""
    try:
        r = requests.get(
            PRODUCT_URL,
            params={
                "catalog_index": catalog_index,
                "query": st.session_state.get("query", ""),
                "recommend": str(show_recs).lower(),
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Backend error: {e}")
        return None


def _post_feedback(query: str, product_title: str, rating: int, rank: int, reason: str):
    try:
        requests.post(
            FEEDBACK_URL,
            json={
                "query": query,
                "product_title": product_title,
                "rating": rating,
                "rank": rank or 1,
                "reason": reason,
            },
            timeout=5,
        )
    except requests.RequestException as e:
        st.warning(f"Could not record feedback: {e}")


def _open_detail(catalog_index: int):
    st.session_state["view"] = "detail"
    st.session_state["detail_index"] = int(catalog_index)
    st.session_state["detail_response"] = None
    st.rerun()


def _back_to_results():
    st.session_state["view"] = "search"
    st.rerun()


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


def _render_card(product: dict, rank=None, *, feedback: bool = True, sponsored: bool = False):
    """Render one full-width product card. rank=None omits the number prefix."""
    with st.container(border=True):
        left, right = st.columns([2, 5])
        with left:
            if product.get("img_url"):
                st.image(product["img_url"], width=200)
        with right:
            if sponsored or product.get("is_sponsored"):
                st.markdown(
                    f":orange[**⭐ SPONSORED**] · _by {product.get('sponsor', 'a partner')}_"
                )
            st.markdown(f"### {product.get('Product_title', 'Untitled')}")
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

            if sponsored or product.get("is_sponsored"):
                rel = product.get("sponsored_relevance")
                if rel is not None and not _is_nan(rel):
                    st.caption(f"**Relevance to query: {rel:.3f}** (cosine; not reranked)")
            else:
                final = product.get("final_score")
                rerank_val = product.get("rerank_score")
                embed = product.get("score")
                if final is not None and not _is_nan(final):
                    bayes = product.get("bayesian_rating")
                    bayes_str = f" · Bayes rating: {bayes}" if bayes else ""
                    st.caption(
                        f"**Final score: {final}/100**  "
                        f"(Rerank: {rerank_val}/100{bayes_str} · embed: {embed:.3f})"
                    )
                elif rerank_val is not None:
                    st.caption(f"Rerank: {rerank_val}/100 · embed: {embed:.3f}")
                elif embed is not None and not _is_nan(embed):
                    st.caption(f"Embedding sim: {embed:.3f}")

            if product.get("reason"):
                st.markdown(f"💡 _{product['reason']}_")

            cidx = product.get("catalog_index")
            if cidx is not None:
                tag = "sp" if (sponsored or product.get("is_sponsored")) else "or"
                if st.button("🔎 View details", key=f"view_{tag}_{cidx}"):
                    _open_detail(int(cidx))

            if feedback:
                kid = product.get("catalog_index", rank)
                fb_cols = st.columns([1, 1, 8])
                if fb_cols[0].button("👍", key=f"fb_up_{kid}"):
                    _post_feedback(
                        st.session_state["query"], product["Product_title"], +1, rank,
                        product.get("reason", ""),
                    )
                    st.toast("Thanks — feedback recorded.")
                if fb_cols[1].button("👎", key=f"fb_down_{kid}"):
                    _post_feedback(
                        st.session_state["query"], product["Product_title"], -1, rank,
                        product.get("reason", ""),
                    )
                    st.toast("Thanks — feedback recorded.")


def _grid_product_body(product: dict):
    """The per-product content of a grid card (image, title, price, rating,
    score, view button). Reused for each option in an ingredient carousel."""
    img = product.get("img_url")
    if img and not _is_nan(img):
        st.markdown(
            f'<img src="{img}" class="cas-thumb" loading="lazy" '
            f'onerror="this.style.display=\'none\'">',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="cas-thumb-empty">no image</div>', unsafe_allow_html=True)
    title = product.get("Product_title", "Untitled")
    st.markdown(f"**{title[:70]}{'…' if len(title) > 70 else ''}**")

    bits = []
    if product.get("categ_lvl2_name"):
        bits.append(product["categ_lvl2_name"])
    if product.get("color"):
        bits.append(product["color"].title())
    if bits:
        st.caption(" · ".join(bits))

    if product.get("price") is not None and not _is_nan(product.get("price")):
        st.markdown(f"**${product['price']}**")

    rating = product.get("average_rating")
    rating_n = product.get("rating_number")
    if rating is not None and not _is_nan(rating):
        try:
            rv = float(rating)
            cnt = f" ({int(float(rating_n)):,})" if (rating_n is not None and not _is_nan(rating_n)) else ""
            st.caption(f"{'⭐' * int(round(rv))} {rv:.1f}{cnt}")
        except (TypeError, ValueError):
            pass

    final = product.get("final_score")
    rerank_val = product.get("rerank_score")
    embed = product.get("score")
    if final is not None and not _is_nan(final):
        bayes = product.get("bayesian_rating")
        bayes_str = f" · Bayes {bayes}" if bayes else ""
        embed_str = f" · embed {embed:.3f}" if isinstance(embed, (int, float)) and not _is_nan(embed) else ""
        st.caption(f"**Score {final}/100** (rerank {rerank_val}{bayes_str}{embed_str})")
    elif rerank_val is not None:
        embed_str = f" · embed {embed:.3f}" if isinstance(embed, (int, float)) and not _is_nan(embed) else ""
        st.caption(f"Rerank {rerank_val}/100{embed_str}")
    elif embed is not None and not _is_nan(embed):
        st.caption(f"Embedding sim {embed:.3f}")
    elif product.get("best_match_score") is not None:
        # Keyword-path (hybrid) rows: show the lexical ranking signal so these
        # cards aren't the only ones with no relevance annotation.
        st.caption(
            f"Best match {product['best_match_score']:.3f}"
            f" · BM25 {product.get('keyword_score')}"
        )

    if product.get("reason"):
        r = product["reason"]
        st.caption(f"💡 _{r[:90]}{'…' if len(r) > 90 else ''}_")

    cidx = product.get("catalog_index")
    if cidx is not None:
        if st.button("🔎 View details", key=f"grid_{cidx}"):
            _open_detail(int(cidx))


def _render_grid_card(product: dict, rank: int):
    """Compact card for the multi-column results grid. For recipe results, an
    ingredient card carries `alternatives` (other brands for the same
    ingredient) shown as a swipeable carousel via tabs."""
    with st.container(border=True):
        if product.get("is_sponsored"):
            st.markdown(f":orange[**⭐ SPONSORED**] · _{product.get('sponsor', 'partner')}_")

        alternatives = product.get("alternatives") or []
        if alternatives:
            # Carousel: primary + alternatives as click-through options (brand A/B/C).
            options = [product] + list(alternatives)
            ingredient = product.get("source_intent")
            if ingredient:
                st.caption(f"🧺 {ingredient} · {len(options)} options")
            tabs = st.tabs([f"Option {i + 1}" for i in range(len(options))])
            for tab, opt in zip(tabs, options):
                with tab:
                    _grid_product_body(opt)
        else:
            _grid_product_body(product)


def _render_mini(item: dict, key_prefix: str):
    """Compact, clickable card for the cross-sell / upsell strip."""
    with st.container(border=True):
        if item.get("img_url"):
            st.image(item["img_url"], width=120)
        title = item.get("Product_title", "Untitled")
        st.markdown(f"**{title[:60]}{'…' if len(title) > 60 else ''}**")
        if item.get("price") is not None and not _is_nan(item.get("price")):
            st.caption(f"${item['price']}")
        if item.get("recommend_reason"):
            st.caption(f"_{item['recommend_reason']}_")
        cidx = item.get("catalog_index")
        if cidx is not None:
            if st.button("🔎 View details", key=f"{key_prefix}_{cidx}"):
                _open_detail(int(cidx))


query_input = st.text_input(
    "What are you looking for?",
    placeholder="e.g. casual outfit, wireless earbuds, healthy breakfast cereal",
)

if st.button("🔍 Search", type="primary") and query_input:
    with st.spinner("Understanding your intent..."):
        st.session_state["query"] = query_input
        st.session_state["page"] = 1
        st.session_state["view"] = "search"
        _fetch()

_settings = (top_k, rerank, show_sponsored, show_recs, hybrid_mode)
if st.session_state.get("settings") != _settings:
    if st.session_state.get("view") == "detail":
        st.session_state["detail_response"] = None
        # Also invalidate the results list: without this, "Back to results"
        # would show stale results fetched under the OLD settings (wrong badge,
        # wrong pagination state). The search view refetches when it sees
        # last_response is empty but a query is active.
        st.session_state["page"] = 1
        st.session_state["last_response"] = None
    elif st.session_state.get("query") and st.session_state.get("last_response") is not None:
        st.session_state["page"] = 1
        _fetch()
    st.session_state["settings"] = _settings


def _render_detail_page():
    idx = st.session_state.get("detail_index")
    if st.button("⬅ Back to results"):
        _back_to_results()

    resp = st.session_state.get("detail_response")
    if resp is None or resp.get("product", {}).get("catalog_index") != idx:
        with st.spinner("Loading product..."):
            resp = _fetch_product(idx)
            st.session_state["detail_response"] = resp
    if not resp:
        st.warning("Could not load this product.")
        return

    product = resp.get("product", {})
    st.divider()
    _render_card(product, rank=None, feedback=True, sponsored=False)

    recs = resp.get("recommendations") or {}
    cross = recs.get("cross_sell", [])
    upsell = recs.get("upsell", [])
    if not (cross or upsell):
        st.caption("No recommendations for this product.")
        return

    if cross:
        st.divider()
        st.subheader("🧺 Buy along with")
        cols = st.columns(len(cross))
        for col, item in zip(cols, cross):
            with col:
                _render_mini(item, key_prefix="d_cross")
    if upsell:
        st.divider()
        st.subheader("⬆️ You might prefer")
        cols = st.columns(min(len(upsell), 3) or 1)
        for col, item in zip(cols, upsell):
            with col:
                _render_mini(item, key_prefix="d_up")


def _render_search_results():
    data = st.session_state.get("last_response")
    if not data and st.session_state.get("query"):
        # Results were invalidated (e.g. settings changed while on the detail
        # page) — refetch under the current settings instead of showing nothing.
        with st.spinner("Refreshing results..."):
            _fetch()
        data = st.session_state.get("last_response")
    if not data:
        return

    # Hybrid-mode badge: which path served this query.
    path = data.get("unified_path")
    if path:
        if path == "keyword":
            st.success(f"⚡ Served by **keyword search** — {data.get('unified_reason', '')}")
        else:
            st.info(f"🧠 Served by **context-aware search** — {data.get('unified_reason', '')}")

    timings = data.get("latency_ms", {})
    rerank_fell_back = (
        data.get("rerank_succeeded") is False and data.get("rerank_requested")
    )
    # Only render the debug block when there is something to show — keyword-
    # path hybrid responses legitimately have no stage timings.
    if show_debug and (timings or rerank_fell_back):
        with st.sidebar:
            st.divider()
            st.caption("**Debug — last request**")
            for stage, ms in timings.items():
                st.caption(f"{stage}: {ms} ms")
            if timings:
                st.caption(f"total: {sum(timings.values())} ms")
            if rerank_fell_back:
                st.caption("⚠️ rerank fell back (LLM unavailable)")

    for err in data.get("errors", []):
        stage = err.get("stage")
        code = err.get("code", "?")
        if stage == "rerank":
            st.warning(f"⚠️ Reranker unavailable (error {code}) — showing embedding-ranked results.")
        elif stage == "translate":
            st.warning(f"⚠️ Query expansion unavailable (error {code}) — searched with your raw query.")
        else:
            st.warning(f"⚠️ {stage} error {code}.")
    if not data.get("errors") and data.get("rerank_succeeded") is False and data.get("rerank_requested"):
        st.caption("⚠️ Showing embedding-ranked results (reranker was unavailable).")

    if data.get("no_match"):
        if data.get("interpreted_as"):
            st.subheader("🧠 Interpreted intents")
            for intent in data.get("interpreted_as", []):
                st.markdown(f"- {intent}")
            st.divider()
        st.info(
            data.get("message")
            or "No matching products found. Try different or more general terms."
        )
        return

    # Keyword-path hybrid responses have no LLM intents — skip the empty header.
    if data.get("interpreted_as"):
        st.subheader("🧠 Interpreted intents")
        for intent in data.get("interpreted_as", []):
            st.markdown(f"- {intent}")

    st.divider()
    st.subheader("🛍️ Products")
    results = data.get("results", [])
    page = data.get("page", 1)
    page_size = data.get("page_size", top_k)
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
                _render_grid_card(product, rank=start_rank + row_start + offset)

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


if st.session_state.get("view") == "detail" and st.session_state.get("detail_index") is not None:
    _render_detail_page()
else:
    _render_search_results()
