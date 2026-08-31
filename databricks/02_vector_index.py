"""Step 2: vector search endpoint + delta-sync index (managed embeddings)."""

import sys

from databricks.vector_search.client import VectorSearchClient

ENDPOINT = "cas-search"
TABLE = "main.cas.products"
INDEX = "main.cas.products_index"

client = VectorSearchClient()  # picks up workspace auth / env vars

if "--teardown" in sys.argv:
    client.delete_index(endpoint_name=ENDPOINT, index_name=INDEX)
    print(f"Deleted {INDEX}. Endpoint billing stops ~24h after last index.")
    sys.exit(0)

try:
    client.create_endpoint(name=ENDPOINT, endpoint_type="STANDARD")
    print(f"Created endpoint {ENDPOINT}")
except Exception as e:  # already exists is fine
    print(f"Endpoint: {e}")

# Managed embeddings: the platform embeds search_text, replacing app/embeddings.py
index = client.create_delta_sync_index(
    endpoint_name=ENDPOINT,
    index_name=INDEX,
    source_table_name=TABLE,
    pipeline_type="CONTINUOUS",          # standard endpoints support continuous
    primary_key="catalog_index",
    embedding_source_column="search_text",
    embedding_model_endpoint_name="databricks-gte-large-en",
)
print(f"Index {INDEX} creating; initial sync of 300K rows runs on the platform "
      "(minutes, not the 25 hours the local CPU encode took).")
