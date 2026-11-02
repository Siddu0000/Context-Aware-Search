"""Cart page — session-only demo basket, deliberately shared across all surfaces."""

import streamlit as st

import ui_cart
from ui_common import (
    apply_scroll,
    cart_line_html,
    escape_md,
    inject_css,
    masthead,
    set_surface,
)

set_surface("cart")
inject_css()
apply_scroll()   # the footer's Back to top
masthead(title="Your cart")   # the same header band as every page; no settings here


def _minus(cid, qty: int) -> None:
    ui_cart.set_qty(cid, qty - 1)   # at 1 this removes the line


def _plus(cid, qty: int) -> None:
    ui_cart.set_qty(cid, qty + 1)


def _subtotal_html(total: float, n_priced: int, n_units: int) -> str:
    """'Subtotal (2 of 3 items): $46.92' -- listed prices only. st.html, never markdown:
    a '$' pair in a markdown line renders as LaTeX."""
    word = "item" if n_units == 1 else "items"
    if not n_priced:
        return (f'<div class="cas-subtotal">Subtotal ({n_units} {word}): '
                "<b>no listed price</b></div>")
    of = f"{n_priced} of {n_units}" if n_priced != n_units else f"{n_units}"
    return (f'<div class="cas-subtotal">Subtotal ({of} {word}): '
            f"<b>${total:,.2f}</b></div>")


def _render_line(entry: dict) -> None:
    p = entry["product"]
    qty = entry["qty"]
    cid = p.get("catalog_index")
    with st.container(key=f"card_cart_{cid}"):
        st.html(cart_line_html(p, qty))
        meta = []
        # Seller text (category, store) and model text (the bundle component an
        # intent named, or the shopper's literal words): escaped
        if p.get("categ_lvl2_name"):
            meta.append(escape_md(p["categ_lvl2_name"]))
        if p.get("source_intent"):
            meta.append(f"for *{escape_md(p['source_intent'])}*")
        if meta:
            st.caption(" · ".join(meta))
        with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
            with st.container(key=f"cas_qty_{cid}", horizontal=True,
                              vertical_alignment="center"):
                st.button("−", key=f"cartminus_{cid}", on_click=_minus, args=(cid, qty),
                          help="Remove one" if qty > 1 else "Remove from cart")
                st.html(f'<span class="cas-qty-n">{int(qty)}</span>', width="content")
                st.button("+", key=f"cartplus_{cid}", on_click=_plus, args=(cid, qty),
                          help="Add one")
            st.button("Delete", key=f"cartrm_{cid}", type="tertiary",
                      on_click=ui_cart.remove, args=(cid,))


def _render_cart(items: list) -> None:
    total, n_priced, n_unpriced = ui_cart.total_price()
    n_units = ui_cart.count()
    est_total, n_est = ui_cart.estimated_total()
    lines, side = st.columns([3, 1], gap="medium")
    with lines:
        with st.container(key="cas_cartlist"):
            st.html('<div class="cas-cart-headrow"><div class="cas-h2">Shopping Cart</div>'
                    '<div class="p">Price</div></div>')
            for entry in items:
                _render_line(entry)
            st.html(f'<div style="text-align:right">'
                    f"{_subtotal_html(total, n_priced, n_units)}</div>")
    with side:
        with st.container(key="cas_cartsub"):
            st.html(_subtotal_html(total, n_priced, n_units))
            if n_est:
                st.caption(
                    f"≈ ${est_total:,.2f} more for {n_est} item(s) with an estimated price "
                    "(from similar products) -- not included in the total."
                )
            if n_unpriced - n_est:
                st.caption(
                    f"⚠️ {n_unpriced - n_est} item(s) have no listed or estimated price, so "
                    "they are not included in the total."
                )
            # Grey, not the accent blue: a blue button reads as clickable
            st.button(
                "Proceed to checkout", width="stretch", disabled=True,
                help="Out of scope for the PoC: no payments or orders.",
            )
            st.caption(
                "Demo cart — no checkout, payment or order is created. Prices come "
                "from the catalog as-is."
            )
            st.button("Empty cart", key="cas_empty_cart", type="tertiary",
                      icon=":material/delete:", on_click=ui_cart.clear)


def _render_empty() -> None:
    with st.container(key="cas_cartlist"):
        st.html('<div class="cas-h2">Your cart is empty</div>')
        st.caption(
            "This is a demo cart — it lives in your session only, and there is no "
            "checkout or payment behind it."
        )
        # st.switch_page, never st.page_link (KeyError 'url_pathname')
        with st.container(horizontal=True, gap="small"):
            for label, key, path in (("Continue with AI Search", "ai", "pages/ai_search.py"),
                                     ("Keyword Search", "keyword", "pages/keyword_search.py"),
                                     ("Shopping Assistant", "assistant",
                                      "pages/shopping_assistant.py")):
                if st.button(label, key=f"cas_continue_{key}"):
                    st.switch_page(path)


items = ui_cart.items()
# if/else, not st.stop(): the licence footer in ui.py runs after the page
with st.container(key="cas_cartpage"):
    if not items:
        _render_empty()
    else:
        _render_cart(items)
