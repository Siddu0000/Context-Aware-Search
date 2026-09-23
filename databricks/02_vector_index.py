"""Step 2: vector search endpoint + delta-sync index (managed embeddings)."""

import sys

from databricks.vector_search.client import VectorSearchClient

import app.config as cfg

import os

# The workspace catalog. `dev` is what we have write access to on the LatentView
# workspace; override with CAS_CATALOG / CAS_SCHEMA for any other workspace.
CATALOG = os.getenv("CAS_CATALOG", "dev")
SCHEMA = os.getenv("CAS_SCHEMA", "cas")
ENDPOINT = os.getenv("CAS_ENDPOINT", "cas-search")
TABLE = f"{CATALOG}.{SCHEMA}.products"
INDEX = f"{CATALOG}.{SCHEMA}.products_index"

print(f"endpoint: {ENDPOINT}\nindex: {INDEX}\nsource: {TABLE}")

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
    # Read from config so run_eval logs the model the index actually uses
    embedding_model_endpoint_name=cfg.VS_EMBEDDING_MODEL,
)
# Measured 2026-09-11: ~2,350 rows/min, so 300K rows = ~2 HOURS for the initial
# snapshot. The endpoint bills hourly from creation, so a rebuild is never cheap.
print(f"Index {INDEX} creating. Initial snapshot of 300K rows takes ~2 HOURS "
      "on a STANDARD endpoint (measured ~2,350 rows/min) -- far better than the "
      "25h local CPU encode, but it bills the whole time. Do not rebuild casually.")
