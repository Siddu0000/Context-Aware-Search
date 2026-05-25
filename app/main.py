from fastapi import FastAPI
from app.translator import translate_query
from app.search import search_products

app = FastAPI(title="Intent-Aware Multi-Category Search")

@app.get("/")
def root():
    return {
        "status": "API running",
        "categories_supported": ["Fashion", "Electronics", "Food & Beverages"]
    }

@app.get("/search")
def search(query: str):
    search_terms = translate_query(query)
    products = search_products(search_terms)

    return {
        "user_query": query,
        "interpreted_as": search_terms,
        "results": products
    }
