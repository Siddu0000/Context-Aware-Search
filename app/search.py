import pandas as pd
from app.embeddings import embed_text, search_vectors

# Load dataset
df = pd.read_csv("data/products.csv")

# Normalize text columns
for col in [
    "bsns_vrtcl_name",
    "categ_lvl2_name",
    "Prod_title",
    "prod_description",
    "color"
]:
    df[col] = df[col].astype(str).str.lower()

# Create unified search text (ONLY VALID COLUMNS)
df["search_text"] = (
    df["bsns_vrtcl_name"] + " " +
    df["categ_lvl2_name"] + " " +
    df["Prod_title"] + " " +
    df["prod_description"] + " " +
    df["color"]
)

# Precompute embeddings
product_embeddings = embed_text(df["search_text"].tolist())

def search_products(search_terms, top_k=10):
    results = []

    for term in search_terms:
        query_embedding = embed_text([term])[0]
        indices, scores = search_vectors(query_embedding, product_embeddings, top_k=top_k)

        for idx, score in zip(indices, scores):
            item = df.iloc[idx].to_dict()
            item["score"] = float(score)
            results.append(item)

    if not results:
        return []

    final = (
        pd.DataFrame(results)
        .sort_values("score", ascending=False)
        .drop_duplicates(subset="Prod_title")
        .head(top_k)
    )

    return final.to_dict(orient="records")
