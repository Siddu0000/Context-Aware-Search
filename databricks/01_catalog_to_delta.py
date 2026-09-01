"""Step 1: products.csv -> Unity Catalog Delta table (with Change Data Feed)."""

CATALOG = "main"          # confirm the workspace catalog before running
SCHEMA = "cas"
TABLE = f"{CATALOG}.{SCHEMA}.products"
CSV_PATH = "/Volumes/main/cas/raw/products.csv"  # upload CSV to a UC Volume first

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

df = (
    spark.read.option("header", True).option("multiLine", True)
    .option("escape", '"').csv(CSV_PATH)
)

from pyspark.sql import functions as F, types as T
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
      # Stable primary key for the index and /product lookups
      .withColumn("catalog_index", F.monotonically_increasing_id())
)

# CDF must be enabled for the delta-sync vector index
(
    df.write.format("delta").mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(TABLE)
)
print(f"Wrote {df.count():,} rows to {TABLE} (CDF enabled)")
