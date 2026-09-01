"""Session-only dummy cart, deliberately shared across surfaces (one basket)."""

import streamlit as st

CART_KEY = "cart"          # {catalog_index: {"product": {...}, "qty": int}}


def _cart() -> dict:
    return st.session_state.setdefault(CART_KEY, {})


def add(product: dict, qty: int = 1) -> int:
    """Add (or increment) a product. Returns the new quantity for that item."""
    cid = product.get("catalog_index")
    if cid is None:
        return 0
    cart = _cart()
    key = str(cid)
    if key in cart:
        cart[key]["qty"] += qty
    else:
        # Trimmed copy — full result rows would bloat the session with score fields
        cart[key] = {
            "qty": qty,
            "product": {
                k: product.get(k)
                for k in (
                    "catalog_index", "Product_title", "img_url", "price",
                    "categ_lvl2_name", "bsns_vrtcl_name", "store",
                    "average_rating", "rating_number", "source_intent",
                )
            },
        }
    return cart[key]["qty"]


def set_qty(catalog_index, qty: int) -> None:
    cart = _cart()
    key = str(catalog_index)
    if key not in cart:
        return
    if qty <= 0:
        cart.pop(key, None)
    else:
        cart[key]["qty"] = qty


def remove(catalog_index) -> None:
    _cart().pop(str(catalog_index), None)


def clear() -> None:
    st.session_state[CART_KEY] = {}


def items() -> list:
    """[{qty, product}] in insertion order."""
    return list(_cart().values())


def count() -> int:
    """Total units, not distinct lines."""
    return sum(i["qty"] for i in _cart().values())


def distinct_count() -> int:
    return len(_cart())


def total_price():
    """(total, n_priced, n_unpriced) — many catalog rows have a null price."""
    total, priced, unpriced = 0.0, 0, 0
    for i in _cart().values():
        p = i["product"].get("price")
        try:
            if p is None:
                unpriced += i["qty"]
                continue
            total += float(p) * i["qty"]
            priced += i["qty"]
        except (TypeError, ValueError):
            unpriced += i["qty"]
    return total, priced, unpriced


def in_cart(catalog_index) -> int:
    """Current qty of this product in the cart (0 if absent)."""
    entry = _cart().get(str(catalog_index))
    return entry["qty"] if entry else 0
