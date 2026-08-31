"""Shared Streamlit rendering + fetch helpers, namespaced per search surface."""

import math
import os

import requests
import streamlit as st

import ui_cart

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
PRODUCT_URL = f"{BACKEND}/product"
FEEDBACK_URL = f"{BACKEND}/feedback"

_SURFACE_KEY = "_active_surface"


def set_surface(name: str) -> None:
    """Declare which surface is rendering; call once at the top of every page."""
    st.session_state[_SURFACE_KEY] = name


def surface() -> str:
    return st.session_state.get(_SURFACE_KEY, "ai")


def ns(key: str) -> str:
    """Namespaced key so two pages never fight over `view` / `detail_index` / `query`."""
    return f"{surface()}__{key}"


def sget(key: str, default=None):
    return st.session_state.get(ns(key), default)


def sset(key: str, value) -> None:
    st.session_state[ns(key)] = value


def sinit(defaults: dict) -> None:
    for k, v in defaults.items():
        st.session_state.setdefault(ns(k), v)


# Backend groups every bundle into one card per component; only wording differs
BUNDLE_UI = {
    "recipe": {
        "heading": "🧺 Shopping list",
        "blurb": "One card per ingredient — use the Option tabs to switch brands.",
        "slot_label": "🧺",
        "slot_word": "brands",
    },
    "outfit": {
        "heading": "👔 Complete the outfit",
        "blurb": "One card per piece — use the Option tabs to switch styles.",
        "slot_label": "👕",
        "slot_word": "options",
    },
    "setup": {
        "heading": "🖥️ Complete the setup",
        "blurb": "One card per component — use the Option tabs to switch models.",
        "slot_label": "🔌",
        "slot_word": "models",
    },
}


def bundle_ui(bundle_type):
    """Presentation strings for a bundle_type, or None for ordinary results."""
    return BUNDLE_UI.get(bundle_type or "")

_CARD_CSS = """
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
"""


def inject_css():
    st.markdown(_CARD_CSS, unsafe_allow_html=True)


def is_nan(x) -> bool:
    """True if x is float NaN. None and other types return False."""
    try:
        return isinstance(x, float) and math.isnan(x)
    except (TypeError, ValueError):
        return False


def _dev_details() -> bool:
    """Whether to show raw pipeline scores; each page's sidebar sets this."""
    return bool(st.session_state.get("show_dev_details"))


# Pipeline-state notes hidden from users; backend strings unchanged (evals match)
_REASON_TRANSLATIONS = {
    "(beyond reranked pool — embedding-similarity order)": "",
    "(rerank disabled)": "",
    "(rerank unavailable — embedding score only)": "",
}


def friendly_reason(reason: str) -> str:
    """User-facing version of a product's `reason`. Empty string = don't show."""
    if not reason:
        return ""
    return _REASON_TRANSLATIONS.get(reason.strip(), reason)


def match_caption(product: dict) -> str:
    """One plain-language relevance line for a product card."""
    final = product.get("final_score")
    rerank_val = product.get("rerank_score")
    embed = product.get("score")
    best_match = product.get("best_match_score")
    dev = _dev_details()

    if final is not None and not is_nan(final):
        line = f"**{final:.0f}% match**"
        if dev:
            bayes = product.get("bayesian_rating")
            bits = [f"rerank {rerank_val}"]
            if bayes:
                bits.append(f"Bayes {bayes}")
            if isinstance(embed, (int, float)) and not is_nan(embed):
                bits.append(f"embed {embed:.3f}")
            line += f"  ·  _{' · '.join(bits)}_"
        return line

    if rerank_val is not None:
        line = f"**{rerank_val}% match**"
        if dev and isinstance(embed, (int, float)) and not is_nan(embed):
            line += f"  ·  _embed {embed:.3f}_"
        return line

    if embed is not None and not is_nan(embed):
        # Cosine similarity, not an LLM judgement — label it as similarity
        line = f"≈ {float(embed) * 100:.0f}% similar"
        if dev:
            line += f"  ·  _embed {float(embed):.3f}_"
        return line

    if best_match is not None:
        line = "🔤 Keyword match"
        if dev:
            line += (
                f"  ·  _best-match {best_match:.3f} · "
                f"BM25 {product.get('keyword_score')}_"
            )
        return line

    return ""


def fetch_product(catalog_index: int, query: str, recommend: bool):
    """Load a single product + its recommendations from /product."""
    try:
        r = requests.get(
            PRODUCT_URL,
            params={
                "catalog_index": catalog_index,
                "query": query,
                "recommend": str(recommend).lower(),
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Backend error: {e}")
        return None


def post_feedback(query: str, product_title: str, rating: int, rank, reason: str):
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


def open_detail(catalog_index: int):
    """Open the product page for this surface only (namespaced)."""
    sset("view", "detail")
    sset("detail_index", int(catalog_index))
    sset("detail_response", None)
    st.rerun()


def back_to_results():
    sset("view", "search")
    st.rerun()


def add_to_cart_button(product: dict, key_ns: str = "", *, wide: bool = False):
    """Add-to-cart control; shows the current quantity once the item is in the cart."""
    cid = product.get("catalog_index")
    if cid is None:
        return
    qty = ui_cart.in_cart(cid)
    label = "🛒 Add to cart" if not qty else f"🛒 Add another ({qty} in cart)"
    if st.button(label, key=f"cart_{key_ns}_{cid}",
                 use_container_width=wide, type="secondary"):
        new_qty = ui_cart.add(product)
        st.toast(f"Added — {new_qty} in cart")
        st.rerun()


def rating_line(product: dict):
    """Render the NaN-safe rating line for a product, if present."""
    rating = product.get("average_rating")
    rating_n = product.get("rating_number")
    rating_valid = rating is not None and not is_nan(rating)
    count_valid = rating_n is not None and not is_nan(rating_n)
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


def render_card(
    product: dict, rank=None, *, query: str, feedback: bool = True, sponsored: bool = False
):
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
            if product.get("price") is not None and not is_nan(product.get("price")):
                meta.append(f"**${product['price']}**")
            st.markdown(" · ".join(meta))

            rating_line(product)

            if sponsored or product.get("is_sponsored"):
                rel = product.get("sponsored_relevance")
                if rel is not None and not is_nan(rel) and _dev_details():
                    st.caption(
                        f"_Relevance to query {rel:.3f} (cosine; not AI-ranked)_"
                    )
            else:
                caption = match_caption(product)
                if caption:
                    st.caption(caption)

            reason = friendly_reason(product.get("reason", ""))
            if reason:
                st.markdown(f"💡 _{reason}_")

            cidx = product.get("catalog_index")
            if cidx is not None:
                tag = "sp" if (sponsored or product.get("is_sponsored")) else "or"
                b1, b2, _sp = st.columns([2, 2, 3])
                with b1:
                    if st.button("🔎 View product", key=f"view_{tag}_{cidx}"):
                        open_detail(int(cidx))
                with b2:
                    add_to_cart_button(product, key_ns=f"card{tag}", wide=True)

            if feedback:
                kid = product.get("catalog_index", rank)
                fb_cols = st.columns([1, 1, 8])
                if fb_cols[0].button("👍", key=f"fb_up_{surface()}_{kid}"):
                    post_feedback(
                        query, product["Product_title"], +1, rank, product.get("reason", ""),
                    )
                    st.toast("Thanks — feedback recorded.")
                if fb_cols[1].button("👎", key=f"fb_down_{surface()}_{kid}"):
                    post_feedback(
                        query, product["Product_title"], -1, rank, product.get("reason", ""),
                    )
                    st.toast("Thanks — feedback recorded.")


def grid_product_body(product: dict, key_ns: str = ""):
    """Per-product content of a grid card; reused for each option in a carousel."""
    img = product.get("img_url")
    if img and not is_nan(img):
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

    if product.get("price") is not None and not is_nan(product.get("price")):
        st.markdown(f"**${product['price']}**")

    rating = product.get("average_rating")
    rating_n = product.get("rating_number")
    if rating is not None and not is_nan(rating):
        try:
            rv = float(rating)
            cnt = f" ({int(float(rating_n)):,})" if (rating_n is not None and not is_nan(rating_n)) else ""
            st.caption(f"{'⭐' * int(round(rv))} {rv:.1f}{cnt}")
        except (TypeError, ValueError):
            pass

    caption = match_caption(product)
    if caption:
        st.caption(caption)

    reason = friendly_reason(product.get("reason", ""))
    if reason:
        st.caption(f"💡 _{reason[:90]}{'…' if len(reason) > 90 else ''}_")

    cidx = product.get("catalog_index")
    if cidx is not None:
        if st.button("🔎 View product", key=f"grid_{key_ns}_{cidx}",
                     use_container_width=True):
            open_detail(int(cidx))
        add_to_cart_button(product, key_ns=f"grid{key_ns}", wide=True)


def render_grid_card(product: dict, rank: int, key_ns: str = "", bundle_type=None):
    """Compact grid card; a bundle card is one component + its `alternatives` as tabs."""
    with st.container(border=True):
        if product.get("is_sponsored"):
            st.markdown(f":orange[**⭐ SPONSORED**] · _{product.get('sponsor', 'partner')}_")

        alternatives = product.get("alternatives") or []
        if alternatives:
            options = [product] + list(alternatives)
            ingredient = product.get("source_intent")
            bui = bundle_ui(bundle_type) or BUNDLE_UI["recipe"]
            if ingredient:
                st.caption(
                    f"{bui['slot_label']} **{ingredient}** — "
                    f"{len(options)} {bui['slot_word']} to choose from"
                )
            tabs = st.tabs([f"Option {i + 1}" for i in range(len(options))])
            for tab, opt in zip(tabs, options):
                with tab:
                    grid_product_body(opt, key_ns=key_ns)
        else:
            grid_product_body(product, key_ns=key_ns)


def render_mini(item: dict, key_prefix: str):
    """Compact, clickable card for the cross-sell / upsell strip."""
    with st.container(border=True):
        if item.get("img_url"):
            st.image(item["img_url"], width=120)
        title = item.get("Product_title", "Untitled")
        st.markdown(f"**{title[:60]}{'…' if len(title) > 60 else ''}**")
        if item.get("price") is not None and not is_nan(item.get("price")):
            st.caption(f"${item['price']}")
        if item.get("recommend_reason"):
            st.caption(f"_{item['recommend_reason']}_")
        cidx = item.get("catalog_index")
        if cidx is not None:
            if st.button("🔎 View product", key=f"{key_prefix}_{cidx}",
                         use_container_width=True):
                open_detail(int(cidx))
            add_to_cart_button(item, key_ns=key_prefix, wide=True)


def render_detail_page(*, show_recs: bool, show_back_button: bool = True):
    """Product detail view, reading only this surface's namespaced state."""
    idx = sget("detail_index")
    if show_back_button and st.button("⬅ Back to results", key=f"detail_back_{surface()}"):
        back_to_results()

    resp = sget("detail_response")
    if resp is None or resp.get("product", {}).get("catalog_index") != idx:
        with st.spinner("Loading product..."):
            resp = fetch_product(idx, sget("query", ""), show_recs)
            sset("detail_response", resp)
    if not resp:
        st.warning("Could not load this product.")
        return

    product = resp.get("product", {})
    st.divider()
    render_card(product, rank=None, query=sget("query", ""),
                feedback=True, sponsored=False)

    recs = resp.get("recommendations") or {}
    cross = recs.get("cross_sell", [])
    upsell = recs.get("upsell", [])
    if not (cross or upsell):
        st.caption("No related products to suggest for this item.")
        return

    if cross:
        st.divider()
        st.subheader("🧺 Frequently bought together")
        cols = st.columns(len(cross))
        for col, item in zip(cols, cross):
            with col:
                render_mini(item, key_prefix="d_cross")
    if upsell:
        st.divider()
        st.subheader("⬆️ Better-rated alternative")
        cols = st.columns(min(len(upsell), 3) or 1)
        for col, item in zip(cols, upsell):
            with col:
                render_mini(item, key_prefix="d_up")
