"""Streamlit entry point — defines the search surfaces and the sidebar nav."""

import streamlit as st

st.set_page_config(
    page_title="Context-Aware Agentic Search",
    page_icon="🛒",
    layout="wide",
)

# st.navigation, not pages/ auto-discovery: auto labels come from the filename
_PAGES = [
    st.Page(
        "pages/ai_search.py",
        title="AI Search",
        icon="🧠",
        default=True,
    ),
    st.Page(
        "pages/keyword_search.py",
        title="Keyword Search",
        icon="🔍",
    ),
    st.Page(
        "pages/shopping_assistant.py",
        title="Shopping Assistant",
        icon="💬",
    ),
    st.Page(
        "pages/cart.py",
        title="Cart",
        icon="🛒",
    ),
]

# Every shared build says so: the catalog's licence forbids client-facing use
st.sidebar.caption("Internal demo · catalog: Amazon Reviews 2023 (research licence)")

st.navigation(_PAGES).run()
