"""Step 1: the (enriched) catalog CSV -> Unity Catalog Delta table, CDF on.

Reads cfg.VS_CATALOG_CSV -- by default /Volumes/<catalog>/<schema>/raw/
products_enriched.csv, written locally by scripts/enrich_catalog.py (gitignored,
~230MB: upload it to the Volume first). The Apps fallback reads the same file,
so the table and the fallback can never disagree on which catalog is served.

search_text is built from the SAME field list app/search.py embeds locally
(DEFAULT_SEARCH_FIELDS + whichever ENRICHED_SEARCH_FIELDS the file has), and
text_hash = sha2(search_text) is what 02 keys stored embeddings on: re-running
this after a catalog change re-embeds only the products whose text changed.
"""

from pyspark.sql import Window, functions as F, types as T

import app.config as cfg
from app.search import DEFAULT_SEARCH_FIELDS, ENRICHED_SEARCH_FIELDS

TABLE = cfg.VS_TABLE
CSV_PATH = cfg.VS_CATALOG_CSV

print(f"target table: {TABLE}\nreading: {CSV_PATH}")

# Overwriting the table under a live index would leave the index serving vectors
# of the OLD catalog against the NEW row positions. Delete it first (02 pause()).
from databricks.vector_search.client import VectorSearchClient

try:
    _live = VectorSearchClient(disable_notice=True).index_exists(
        endpoint_name=cfg.VS_ENDPOINT, index_name=cfg.VS_INDEX)
except Exception as e:  # noqa: BLE001
    # No endpoint means no index; any other failure must not let us overwrite blind
    if getattr(e, "status_code", None) != 404 and "not found" not in str(e).lower():
        raise
    print(f"(no endpoint {cfg.VS_ENDPOINT}: {type(e).__name__})")
    _live = False
if _live:
    raise RuntimeError(
        f"{cfg.VS_INDEX} still exists. Run 02's pause() first, then this, then "
        "02's embed() and resume()."
    )

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.CAS_CATALOG}.{cfg.CAS_SCHEMA}")

df = (
    spark.read.option("header", True).option("multiLine", True)
    .option("escape", '"').csv(CSV_PATH)
)
fields = list(DEFAULT_SEARCH_FIELDS) + [c for c in ENRICHED_SEARCH_FIELDS if c in df.columns]
missing = [c for c in DEFAULT_SEARCH_FIELDS if c not in df.columns]
assert not missing, f"{CSV_PATH} lacks search fields {missing}"
print(f"search_text fields: {fields}")

for col in ("average_rating", "price", "price_estimate"):
    if col in df.columns:
        df = df.withColumn(col, F.col(col).cast(T.DoubleType()))
df = (
    df.withColumn("rating_number", F.col("rating_number").cast(T.LongType()))
      # The index holds vectors of this ONE column; mirrors app/search.py
      # _build_search_text (lowercased fields, space-joined; nulls skipped)
      .withColumn("search_text", F.lower(F.concat_ws(" ", *fields)))
      .withColumn("text_hash", F.sha2(F.col("search_text"), 256))
      # catalog_index MUST be dense 0..N-1 in FILE order: app/search.py uses it
      # as a ROW POSITION to hydrate results and serve GET /product, exactly as it
      # does locally. monotonically_increasing_id() alone is only dense when the
      # read happens to be one partition; row_number() over it makes that explicit.
      .withColumn("_file_order", F.monotonically_increasing_id())
      .withColumn(
          "catalog_index",
          (F.row_number().over(Window.orderBy("_file_order")) - 1).cast(T.LongType()),
      )
      .drop("_file_order")
)

# CDF must be enabled for a delta-sync vector index. overwriteSchema: the
# enriched catalog adds columns to the table the base catalog created.
(
    df.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(TABLE)
)
# The writer option only takes effect when the table is CREATED; make sure
spark.sql(f"ALTER TABLE {TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
stats = spark.table(TABLE).selectExpr(
    "count(*) AS n", "min(catalog_index) AS lo",
    "max(catalog_index) AS hi", "count(DISTINCT catalog_index) AS d",
    "count(DISTINCT text_hash) AS texts", "avg(length(search_text)) AS avg_chars",
).first()
assert stats.lo == 0 and stats.hi == stats.n - 1 and stats.d == stats.n, (
    f"catalog_index is not dense 0..N-1: {stats}"
)
print(
    f"Wrote {stats.n:,} rows to {TABLE} (CDF on); catalog_index dense 0..{stats.hi:,}; "
    f"{stats.texts:,} distinct texts, avg {stats.avg_chars:,.0f} chars. "
    "The enriched catalog of 2026-10-08 has 299,972 rows."
)
