import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/search"

st.set_page_config(
    page_title="Intent-Aware Product Search",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Intent-Aware Product Search")
st.caption("We don’t search what you typed. We search what you meant.")

query = st.text_input(
    "What are you looking for?",
    placeholder="e.g. casual outfit, wireless earbuds, healthy breakfast cereal"
)

search_btn = st.button("🔍 Search")

if search_btn and query:
    with st.spinner("Understanding your intent..."):
        response = requests.get(BACKEND_URL, params={"query": query})

    if response.status_code != 200:
        st.error("Backend error. Please try again.")
    else:
        data = response.json()

        st.subheader("🧠 Interpreted Intent")
        for intent in data["interpreted_as"]:
            st.markdown(f"- **{intent}**")

        st.divider()

        st.subheader("🛍️ Recommended Products")

        if not data["results"]:
            st.warning("No products found.")
        else:
            cols = st.columns(3)

            for idx, product in enumerate(data["results"]):
                with cols[idx % 3]:
                    st.markdown(f"### {product['Prod_title']}")
                    st.markdown(f"**Category:** {product['bsns_vrtcl_name']}")
                    st.markdown(f"**Color:** {product['color']}")
                    st.markdown(f"**Price:** ${product['price']}")

                    if product.get("img_url"):
                        st.image(product["img_url"], width=300)
