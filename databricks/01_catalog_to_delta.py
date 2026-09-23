"""Step 1: products.csv -> Unity Catalog Delta table (with Change Data Feed)."""

import os

# The workspace catalog. `dev` is what we have write access to on the LatentView
# workspace; override with CAS_CATALOG / CAS_SCHEMA for any other workspace.
CATALOG = os.getenv("CAS_CATALOG", "dev")
SCHEMA = os.getenv("CAS_SCHEMA", "cas")
TABLE = f"{CATALOG}.{SCHEMA}.products"
# Upload products.csv to this volume first -- it is gitignored (192MB)
CSV_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw/products.csv"

print(f"target table: {TABLE}\nreading: {CSV_PATH}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

df = (
    spark.read.option("header", True).option("multiLine", True)
    .option("escape", '"').csv(CSV_PATH)
)

from pyspark.sql import Window, functions as F, types as T
df = (
    df.withColumn("average_rating", F.col("average_rating").cast(T.DoubleType()))
      .withColumn("rating_number", F.col("rating_number").cast(T.LongType()))
      .withColumn("price", F.col("price").cast(T.DoubleType()))
      # The index embeds ONE text column; mirrors app/search.py DEFAULT_SEARCH_FIELDS
      .withColumn(
          "search_text",
          F.lower(F.concat_ws(
              " ",
              "bsns_vrtcl_name", "categ_lvl2_name", "Product_title",
              "prod_description", "color", "material", "occasion",
          )),
      )
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

# CDF must be enabled for the delta-sync vector index
(
    df.write.format("delta").mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(TABLE)
)
stats = spark.table(TABLE).selectExpr(
    "count(*) AS n", "min(catalog_index) AS lo",
    "max(catalog_index) AS hi", "count(DISTINCT catalog_index) AS d",
).first()
assert stats.lo == 0 and stats.hi == stats.n - 1 and stats.d == stats.n, (
    f"catalog_index is not dense 0..N-1: {stats}"
)
print(f"Wrote {stats.n:,} rows to {TABLE} (CDF enabled); catalog_index dense 0..{stats.hi:,}")
