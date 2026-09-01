"""Cart page — session-only demo basket, deliberately shared across all surfaces."""

import streamlit as st

import ui_cart
from ui_common import inject_css, is_nan, set_surface

set_surface("cart")
inject_css()

st.title("🛒 Your cart")

items = ui_cart.items()

if not items:
    st.info(
        "Your cart is empty. Add products from **AI Search**, "
        "**Keyword Search** or the **Shopping Assistant**."
    )
    st.caption(
        "This is a demo cart — it lives in your session only, and there is no "
        "checkout or payment behind it."
    )
    st.stop()

total, n_priced, n_unpriced = ui_cart.total_price()

c1, c2, c3 = st.columns([2, 2, 2])
c1.metric("Items", ui_cart.count())
c2.metric("Distinct products", ui_cart.distinct_count())
c3.metric("Estimated total", f"${total:,.2f}")
if n_unpriced:
    st.caption(
        f"⚠️ {n_unpriced} item(s) have no listed price in the catalog, so they "
        "are not included in the total. (Many Amazon rows have a null price — "
        "we show the gap rather than quietly under-reporting the basket.)"
    )
st.divider()

for entry in items:
    p = entry["product"]
    qty = entry["qty"]
    cid = p.get("catalog_index")
    with st.container(border=True):
        img_col, mid, qty_col = st.columns([1, 4, 2])

        with img_col:
            img = p.get("img_url")
            if img and not is_nan(img):
                st.markdown(
                    f'<img src="{img}" class="cas-thumb" style="height:110px" '
                    f'loading="lazy" onerror="this.style.display=\'none\'">',
                    unsafe_allow_html=True,
                )

        with mid:
            st.markdown(f"**{p.get('Product_title', 'Untitled')}**")
            meta = []
            if p.get("categ_lvl2_name"):
                meta.append(p["categ_lvl2_name"])
            if p.get("store"):
                meta.append(f"by {p['store']}")
            if p.get("source_intent"):
                # which bundle component this item was chosen for
                meta.append(f"for *{p['source_intent']}*")
            if meta:
                st.caption(" · ".join(meta))
            price = p.get("price")
            if price is not None and not is_nan(price):
                line = float(price) * qty
                st.markdown(f"${float(price):,.2f} × {qty} = **${line:,.2f}**")
            else:
                st.markdown("_no listed price_")

        with qty_col:
            new_qty = st.number_input(
                "Qty", min_value=0, max_value=99, value=int(qty), step=1,
                key=f"cartqty_{cid}",
                help="Set to 0 to remove this item.",
            )
            if new_qty != qty:
                ui_cart.set_qty(cid, int(new_qty))
                st.rerun()
            if st.button("Remove", key=f"cartrm_{cid}", use_container_width=True):
                ui_cart.remove(cid)
                st.rerun()

st.divider()
left, right = st.columns([3, 2])
with left:
    st.caption(
        "Demo cart — no checkout, payment or order is created. Prices come "
        "from the catalog as-is."
    )
with right:
    if st.button("🗑️ Empty cart", use_container_width=True):
        ui_cart.clear()
        st.rerun()
    st.button(
        "Checkout (disabled in demo)", use_container_width=True, disabled=True,
        help="Out of scope for the PoC — payments/orders were never part of it.",
    )
