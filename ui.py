"""Streamlit entry point — defines the search surfaces and the top navigation."""

import streamlit as st

from ui_common import footer, wordmark_svg

st.set_page_config(
    page_title="Context-Aware Agentic Search",
    page_icon="🛒",
    layout="wide",
    # Settings live in each page's search band; nothing needs the sidebar
    initial_sidebar_state="collapsed",
)

# The app's own name and a magnifier as the wordmark: the marketplace LAYOUT is
# borrowed (2026-10-19), never its name, logo or marks. An inline SVG string: the
# Databricks App stages only the .py files, so a separate logo file would not reach it.
st.logo(wordmark_svg(), size="large")

# st.navigation, not pages/ auto-discovery: auto labels come from the filename. The
# Cart is hidden from the nav (the search band's Cart button and Buy Now open it, as on
# a marketplace); hidden internal pages stay reachable through st.switch_page
_PAGES = [
    st.Page(
        "pages/ai_search.py",
        title="AI Search",
        default=True,
    ),
    st.Page(
        "pages/keyword_search.py",
        title="Keyword Search",
    ),
    st.Page(
        "pages/shopping_assistant.py",
        title="Shopping Assistant",
    ),
    st.Page(
        "pages/cart.py",
        title="Cart",
        visibility="hidden",
    ),
]

st.navigation(_PAGES, position="top").run()

# Back to top, the page links and the licence strip. After the page, so it is a footer;
# no page calls st.stop(), which would skip it
footer()
