"""Step 3: end-to-end smoke test on Databricks — retrieval + one LLM call."""

import json
import os

from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI  # FM APIs speak the OpenAI protocol

ENDPOINT = "cas-search"
INDEX = "main.cas.products_index"

vs = VectorSearchClient()
idx = vs.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
# query_type="hybrid" (keyword+vector RRF) also exists — re-run router evals first
hits = idx.similarity_search(
    query_text="something breathable for a humid day",
    columns=["catalog_index", "Product_title", "categ_lvl2_name", "price"],
    num_results=5,
)
rows = hits.get("result", {}).get("data_array", [])
print(f"retrieval: {len(rows)} hits")
for r in rows:
    print("  -", str(r[1])[:60])

llm = OpenAI(
    api_key=os.environ["DATABRICKS_TOKEN"],
    base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
)
resp = llm.chat.completions.create(
    model="databricks-gpt-oss-120b",
    messages=[{
        "role": "user",
        "content": (
            "Convert this shopper query into exactly 3 concise product search "
            "intents as JSON {\"search_terms\": [...]}: "
            "\"warm wool sweater for winter\""
        ),
    }],
    temperature=0.2,
)
print("llm:", json.dumps(resp.choices[0].message.content)[:200])
print("\nSMOKE TEST PASSED — pipeline components live on Databricks.")
