"""Step 2: embed the catalog ONCE into Delta, then build the index from those vectors.

An AI Search (Vector Search) endpoint or index cannot be paused, and an index
nobody queries still bills its endpoint (one unit = 4 DBU/h on STANDARD). What
CAN be avoided is re-running the embedding model: embed() stores every vector in
Delta, and resume() builds a SELF-MANAGED index from them -- no embedding calls.
Between runs pause() deletes only the index; an endpoint with no index stops
billing 24h after its last index is deleted (since 2026-03-13):
https://learn.microsoft.com/en-us/azure/databricks/ai-search/cost-management

    exec(open("databricks/02_vector_index.py").read())
    status()              # what exists and what is billing
    embed()               # once, and after the catalog changes: new text only
    resume()              # index from stored vectors; polls until ONLINE
    ...                   # 03 - 08
    pause()               # delete the index; endpoint + vectors stay
    costs("2026-10-09")   # a day later: billed DBUs / $ (billing lags ~12h)

Functions, not sys.argv flags: flags do not reach a script run under exec().
Embedding goes through app.vector_search.embed_texts -- the call the app uses for
every query -- so catalog and query vectors cannot come from different models.
Vector Search is driven through its REST API (app.vector_search.rest, on the
preinstalled databricks-sdk), not the databricks-vectorsearch package: that one
pins protobuf<6 and broke the kernel on DBR 19.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pyspark.sql import Window, functions as F

import app.config as cfg
from app import vector_search

ENDPOINT = cfg.VS_ENDPOINT
INDEX = cfg.VS_INDEX
CATALOG_TABLE = cfg.VS_TABLE      # the catalog (01): catalog_index, search_text, text_hash
EMB = cfg.VS_EMBEDDINGS_TABLE     # store: (embedding_model, text_hash) -> embedding
VEC = cfg.VS_VECTORS_TABLE        # index source: catalog_index -> embedding
EMBED_MODEL = cfg.VS_EMBEDDING_MODEL
# Distinct names on purpose: 01 and 03 are exec()ed into the same notebook
# globals and define TABLE / MODEL of their own, which 02's functions would read
# List price of databricks-gte-large-en, DBU per 1M input tokens:
# https://www.databricks.com/product/pricing/foundation-model-serving
GTE_DBU_PER_M_TOKENS = 1.857

def _endpoint_state():
    try:
        ep = vector_search.endpoint_info(ENDPOINT)
    except Exception as e:  # noqa: BLE001 -- reported, not raised: status() must always print
        return f"ERROR ({type(e).__name__}: {str(e)[:80]})"
    if ep is None:
        return "MISSING"
    return (ep.get("endpoint_status") or {}).get("state", "?")


def _index_names():
    names, token = [], None
    while True:
        try:
            page = vector_search.rest("GET", "/indexes", query={
                "endpoint_name": ENDPOINT, **({"page_token": token} if token else {})})
        except Exception as e:  # noqa: BLE001
            if vector_search.is_not_found(e):  # no endpoint, no indexes
                return names
            raise
        names += [i.get("name") for i in page.get("vector_indexes") or [] if i.get("name")]
        token = page.get("next_page_token")
        if not token:
            return names


def _index_exists():
    return _index_info() is not None


def _index_info():
    """The index's REST description (status.detailed_state ...), or None if absent."""
    return vector_search.index_info(INDEX)


def _is_managed(info):
    """True for the pre-2026-10-09 index: the platform embedded search_text itself."""
    spec = (info or {}).get("delta_sync_index_spec") or {}
    return bool(spec.get("embedding_source_columns"))


def _table_exists(name):
    return spark.catalog.tableExists(name)


def _check_vectors():
    """Is VEC exactly one current vector per catalog row? Returns the counts."""
    if not _table_exists(VEC):
        return {"aligned": False, "why": f"{VEC} does not exist -- run embed()"}
    p = spark.table(CATALOG_TABLE).select("catalog_index", F.col("text_hash").alias("p_hash"))
    if "embedding_model" not in spark.table(VEC).columns:
        return {"aligned": False, "why": f"{VEC} predates model tracking -- run embed()"}
    v = spark.table(VEC).select(
        "catalog_index", F.col("text_hash").alias("v_hash"), F.size("embedding").alias("dim"),
        F.col("embedding_model").alias("v_model"))
    r = p.join(v, "catalog_index", "full_outer").agg(
        F.count("p_hash").alias("products"),
        F.count("v_hash").alias("vectors"),
        F.countDistinct(F.when(F.col("v_hash").isNotNull(), F.col("catalog_index"))).alias("distinct"),
        F.sum(F.when(F.col("p_hash").isNull() | F.col("v_hash").isNull(), 1).otherwise(0)).alias("unmatched"),
        F.sum(F.when(F.col("p_hash") != F.col("v_hash"), 1).otherwise(0)).alias("stale"),
        F.min("dim").alias("dim_min"), F.max("dim").alias("dim_max"),
        # Same hashes, other model = vectors the app's query embeddings cannot match
        F.sum(F.when(F.col("v_model") != F.lit(EMBED_MODEL), 1).otherwise(0)).alias("other_model"),
        F.max("v_model").alias("model"),
    ).first().asDict()
    r["aligned"] = (
        r["products"] > 0 and r["vectors"] == r["products"] == r["distinct"]
        and not r["unmatched"] and not r["stale"] and not r["other_model"]
        and r["dim_min"] == r["dim_max"] and (r["dim_min"] or 0) > 0
    )
    if not r["aligned"] and r["other_model"]:
        r["why"] = (f"{VEC} holds {r['model']} vectors but VS_EMBEDDING_MODEL is {EMBED_MODEL}"
                    " -- pause(), embed(), resume()")
    elif not r["aligned"]:
        r["why"] = (f"{VEC} does not match {CATALOG_TABLE} (catalog changed since embed()?) -- "
                    "run embed()")
    return r


def status():
    """What exists, and what is billing right now."""
    state = _endpoint_state()
    names = _index_names()
    print(f"endpoint  {ENDPOINT}: {state}")
    print(f"indexes   {names or 'none'}")
    info = _index_info()
    if info is not None:
        st = info.get("status") or {}
        kind = "MANAGED (old, base catalog)" if _is_managed(info) else "self-managed"
        print(f"index     {INDEX}: {st.get('detailed_state')} | "
              f"{st.get('indexed_row_count')} rows | {kind}")
    if _table_exists(EMB):
        n = spark.table(EMB).where(F.col("embedding_model") == EMBED_MODEL).count()
        print(f"store     {EMB}: {n:,} vectors for {EMBED_MODEL}")
    else:
        print(f"store     {EMB}: none yet -- run embed()")
    chk = _check_vectors()
    print(f"source    {VEC}: " + ("aligned with the catalog, "
          f"{chk['vectors']:,} x {chk['dim_min']} ({chk['model']})" if chk["aligned"] else chk["why"]))
    if names:
        print("billing   the endpoint hosts an index: ~4 DBU/h (one STANDARD unit) "
              "whether or not anyone queries it. pause() when done.")
    else:
        print("billing   no index: no serving charge from 24h after the last delete.")


# Retrying these cannot help: bad input, missing endpoint, no permission
_PERMANENT = {"BadRequest", "InvalidParameterValue", "NotFound", "ResourceDoesNotExist",
              "PermissionDenied", "Unauthenticated"}


def _embed_batch(texts, attempts=6):
    """embed_texts with backoff ON TOP of the SDK's own 429/503 retries (60s)."""
    for i in range(attempts):
        try:
            return vector_search.embed_texts(texts, batch=True)
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1 or type(e).__name__ in _PERMANENT:
                raise
            wait = min(60, 2 ** i * 2)
            print(f"    retry in {wait}s after {type(e).__name__}: {str(e)[:120]}")
            time.sleep(wait)


def embed(batch_size=64, workers=8, chunk_rows=4000, limit=None):
    """Embed every catalog text not yet stored for EMBED_MODEL, then rebuild VEC.

    Resumable: each chunk is committed to EMB as soon as it is embedded, so an
    interrupted run continues where it stopped. Re-running after 01 reloads a
    changed catalog embeds only the new or changed texts. `limit` = a pilot.
    """
    if "text_hash" not in spark.table(CATALOG_TABLE).columns:
        raise RuntimeError(f"{CATALOG_TABLE} has no text_hash: re-run databricks/01 first.")
    spark.sql(f"""CREATE TABLE IF NOT EXISTS {EMB} (
        text_hash STRING, embedding_model STRING, embedding ARRAY<FLOAT>,
        embedded_at TIMESTAMP)""")
    stored = spark.table(EMB).where(F.col("embedding_model") == EMBED_MODEL).select("text_hash")
    todo_df = (spark.table(CATALOG_TABLE).select("text_hash", "search_text")
               .dropDuplicates(["text_hash"]).join(stored, "text_hash", "left_anti"))
    total = todo_df.count()
    if limit is not None:
        todo_df = todo_df.limit(int(limit))
    todo = todo_df.toPandas()
    n = len(todo)
    print(f"{n:,} texts to embed with {EMBED_MODEL} (batch {batch_size}, {workers} workers)")

    t0, done, tokens = time.time(), 0, 0
    for start in range(0, n, chunk_rows):
        chunk = todo.iloc[start:start + chunk_rows]
        texts = chunk["search_text"].fillna("").tolist()
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_embed_batch, batches))
        vecs = [v for r in results for v in r[0]]
        tokens += sum(r[1] for r in results)
        rows = pd.DataFrame({"text_hash": chunk["text_hash"].tolist(),
                             "embedding_model": EMBED_MODEL, "embedding": vecs})
        (spark.createDataFrame(rows, "text_hash STRING, embedding_model STRING, embedding ARRAY<DOUBLE>")
         .withColumn("embedding", F.col("embedding").cast("array<float>"))
         .withColumn("embedded_at", F.current_timestamp())
         .write.mode("append").saveAsTable(EMB))
        done += len(chunk)
        rate = done / max(time.time() - t0, 1e-9)
        print(f"  {done:,}/{n:,} | {rate * 60:,.0f} texts/min | eta {(n - done) / rate / 60:,.1f} min"
              f" | {tokens / 1e6:,.2f}M tokens ~ {tokens / 1e6 * GTE_DBU_PER_M_TOKENS:,.1f} DBU (list)")
    if limit is not None:
        rest = total - done
        per = tokens / max(done, 1)
        mins = rest / max(done / max(time.time() - t0, 1e-9), 1e-9) / 60
        print(f"Pilot only: {done:,} of {total:,}. The other {rest:,} texts project to "
              f"~{rest * per / 1e6:,.1f}M tokens ~ {rest * per / 1e6 * GTE_DBU_PER_M_TOKENS:,.0f} "
              f"DBU (list) and ~{mins:,.0f} min. Run embed() for the rest.")
        return
    _materialize_vectors()


def _materialize_vectors():
    """VEC = one stored vector per catalog row, for the CURRENT catalog."""
    if _index_info() is not None:
        if _check_vectors()["aligned"]:
            print(f"{VEC} already current; {INDEX} exists, leaving both alone.")
            return
        raise RuntimeError(f"{INDEX} exists but {VEC} is out of date: pause(), embed(), resume().")
    latest = Window.partitionBy("text_hash").orderBy(F.col("embedded_at").desc())
    emb = (spark.table(EMB).where(F.col("embedding_model") == EMBED_MODEL)
           .withColumn("_rn", F.row_number().over(latest)).where("_rn = 1")
           .select("text_hash", "embedding_model", "embedding"))
    (spark.table(CATALOG_TABLE).select("catalog_index", "text_hash").join(emb, "text_hash", "inner")
     .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
     .option("delta.enableChangeDataFeed", "true").saveAsTable(VEC))
    # A STANDARD endpoint's delta-sync index needs Change Data Feed on its source
    spark.sql(f"ALTER TABLE {VEC} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    chk = _check_vectors()
    if not chk["aligned"]:
        raise RuntimeError(f"{VEC} came out misaligned: {chk}")
    print(f"{VEC}: {chk['vectors']:,} vectors x {chk['dim_min']} dims, aligned with {CATALOG_TABLE}.")


def _ensure_endpoint():
    # Wait for ONLINE before creating an index: the likely cause of the Sep 10
    # OFFLINE_FAILED build ($23.39 for nothing) was an index created on a
    # still-PROVISIONING endpoint. (Hypothesis -- never confirmed.)
    if _endpoint_state() == "MISSING":
        vector_search.rest("POST", "/endpoints", body={"name": ENDPOINT, "endpoint_type": "STANDARD"})
        print(f"Creating endpoint {ENDPOINT} (STANDARD) ...")
    t0 = time.time()
    while True:
        state = _endpoint_state()
        if state == "ONLINE":
            break
        if any(bad in state for bad in ("OFFLINE", "RED_STATE", "ERROR", "DELETED")):
            raise RuntimeError(f"endpoint {ENDPOINT}: {state}")
        if time.time() - t0 > 3600:
            raise TimeoutError(f"endpoint {ENDPOINT} not ONLINE after 60 min: {state}")
        print(f"  endpoint {ENDPOINT}: {state} ({(time.time() - t0) / 60:.1f} min)")
        time.sleep(30)
    print(f"endpoint {ENDPOINT}: ONLINE")


def resume(wait=True, poll_s=60):
    """Build INDEX from the stored vectors (no embedding calls); poll until ONLINE."""
    info = _index_info()
    if info is not None and _is_managed(info):
        raise RuntimeError(
            f"{INDEX} is the OLD managed-embedding index (base catalog, query_text). "
            "Run pause() to delete it, then resume().")
    chk = _check_vectors()
    if not chk["aligned"]:
        # With an index up, VEC can only drift via a model or catalog change (01 and
        # embed() refuse to rewrite under a live index): that index is stale too
        raise RuntimeError(chk["why"] if info is None else
                           f"{INDEX} exists but is stale: {chk['why']}")
    _ensure_endpoint()
    if info is None:
        _create_index(int(chk["dim_min"]))
        print(f"Creating {INDEX} from {VEC} ({chk['vectors']:,} vectors, no embedding calls).")
    else:
        print(f"{INDEX} already exists: not rebuilding.")
    if wait:
        _wait_online(chk.get("vectors") or 0, poll_s)


def _create_index(dim, attempts=40):
    """create_delta_sync_index, retried while a deleted index's name is still held.

    Deletion is two-phase: the index leaves GET first and frees its NAME later
    (up to ~10 min); a create in between fails "pending deletion" (Databricks CLI
    source, bundle/direct/dresources/vector_search_index.go, PR #6143).
    """
    for i in range(attempts):
        try:
            return vector_search.rest("POST", "/indexes", body={
                "name": INDEX,
                "endpoint_name": ENDPOINT,
                "primary_key": "catalog_index",
                "index_type": "DELTA_SYNC",
                "delta_sync_index_spec": {
                    "source_table": VEC,
                    # The catalog is static between runs: CONTINUOUS would keep a
                    # streaming pipeline billing while the index sits idle
                    "pipeline_type": "TRIGGERED",
                    "embedding_vector_columns": [
                        {"name": "embedding", "embedding_dimension": dim}],
                    # Default syncs EVERY source column; the app hydrates rows itself
                    "columns_to_sync": ["catalog_index", "embedding"],
                },
            })
        except Exception as e:  # noqa: BLE001
            if "pending deletion" not in str(e).lower() or i == attempts - 1:
                raise
            print(f"  {INDEX} name still pending deletion; retrying in 15s")
            time.sleep(15)


# status.detailed_state is in the REST reply but not the public schema. The old
# client's wait_until_ready(wait_for_updates=True) treated these as finished, and
# any state containing OFFLINE as failed (databricks-ai-search 0.78 source)
_DONE_STATES = {"ONLINE", "ONLINE_NO_PENDING_UPDATE", "ONLINE_DIRECT_ACCESS"}


def _wait_online(expected, poll_s, timeout_h=6):
    t0 = time.time()
    while True:
        st = (_index_info() or {}).get("status") or {}
        state = str(st.get("detailed_state", "?"))
        rows = int(st.get("indexed_row_count") or 0)
        mins = (time.time() - t0) / 60
        rate = f"{rows / mins:,.0f} rows/min" if mins > 0.5 and rows else ""
        print(f"  {mins:6.1f} min  {state:40s} {rows:>9,}/{expected:,}  {rate}")
        if "OFFLINE" in state or "FAILED" in state:
            raise RuntimeError(
                f"{INDEX} {state}: {st.get('message')}. A failed build bills like a good "
                "one: pause() now, then retry.")
        # Unknown ONLINE_* variant, or no detailed_state at all (it is not in the
        # documented schema): the documented `ready` + every row indexed also counts
        if state in _DONE_STATES or (
                ("ONLINE" in state or st.get("ready") is True) and expected and rows >= expected):
            if expected and rows != expected:
                print(f"WARNING: {rows:,} rows indexed, {expected:,} expected.")
            print(f"{INDEX} ONLINE after {mins:.1f} min. Endpoint bills ~4 DBU/h until pause().")
            return
        if mins > timeout_h * 60:
            raise TimeoutError(
                f"{INDEX} not ONLINE after {timeout_h}h ({state}, {rows:,} rows). It is still "
                "billing: inspect it in Catalog Explorer, or pause().")
        time.sleep(poll_s)


def pause():
    """Delete INDEX. The endpoint, EMB and VEC stay; nothing is re-embedded later."""
    if _index_info() is None:
        print(f"{INDEX} does not exist: nothing to delete.")
    else:
        vector_search.rest("DELETE", f"/indexes/{INDEX}")
        for _ in range(60):  # asynchronous; resume() also retries while the name is held
            if not _index_exists():
                break
            time.sleep(10)
        print(f"Deleted {INDEX} (its name can stay reserved for up to ~10 min).")
    others = [n for n in _index_names() if n != INDEX]
    if others:
        print(f"WARNING: {ENDPOINT} still hosts {others} -- it keeps billing until they go too.")
    else:
        print(f"{ENDPOINT} hosts no index: it stops billing 24h after this delete.")
    print(f"Kept: {EMB}, {VEC} -> resume() rebuilds with no embedding calls.")


def _cluster_id():
    """This notebook's classic cluster, or None on serverless."""
    try:
        cid = spark.conf.get("spark.databricks.clusterUsageTags.clusterId", None)
    except Exception:  # noqa: BLE001 -- serverless may refuse the key outright
        cid = None
    return cid or os.environ.get("DB_CLUSTER_ID") or None


def costs(since, until=None, mine_only=True, llms=None, cluster_id="auto"):
    """Billed DBUs and list-price $ per day/product/SKU since `since` (YYYY-MM-DD).

    system.billing.usage lags: "typically available within 12 hours" -- this is
    for the day after a run, not for watching one live (watch resume()'s log).
    mine_only keeps the endpoints this runbook touches. NOT exact: the pay-per-token
    endpoints (embedding model, LLMs) are WORKSPACE-WIDE, so their rows include
    every other caller that day, and index ingest has no endpoint name at all.
    run_as is no fix: this team runs as a shared account. The exact per-run numbers
    are embed()'s printed token count and each eval's MLflow token totals.
    On a classic cluster its DBUs are included (cluster_id="auto" = this notebook's
    cluster; the whole cluster, so a shared one includes its other users), but its
    Azure VM, disk and network cost is NOT: system.billing holds only what Databricks
    bills; Azure Cost Management shows the VMs (tagged ClusterId / ClusterName).
    """
    from databricks.sdk import WorkspaceClient

    ws = WorkspaceClient().get_workspace_id()
    mine = ""
    if mine_only:
        mine = """AND (u.usage_metadata.endpoint_name IN (:ep, :emb, LLMS)
                   OR (u.billing_origin_product IN ('VECTOR_SEARCH', 'AI_SEARCH')
                       AND u.usage_metadata.endpoint_name IS NULL)
                   CLUSTER)"""
    q = f"""
      SELECT u.usage_date, u.billing_origin_product AS product, u.sku_name,
             u.usage_metadata.endpoint_name AS endpoint,
             u.usage_metadata.dlt_pipeline_id AS pipeline,
             round(sum(u.usage_quantity), 2) AS dbu,
             round(sum(u.usage_quantity * lp.pricing.effective_list.default), 2) AS usd_list
      FROM system.billing.usage u
      JOIN system.billing.list_prices lp
        ON u.sku_name = lp.sku_name AND u.cloud = lp.cloud
       AND u.usage_end_time >= lp.price_start_time
       AND (lp.price_end_time IS NULL OR u.usage_end_time < lp.price_end_time)
      WHERE u.workspace_id = :ws AND u.usage_date BETWEEN :since AND :until {mine}
      GROUP BY ALL
      ORDER BY u.usage_date, usd_list DESC"""
    args = {"ws": str(ws), "since": since, "until": until or "2999-12-31"}
    if mine_only:
        args.update(ep=ENDPOINT, emb=EMBED_MODEL)
        # Every LLM the runbook calls (05 compares two), not only the configured one
        for i, m in enumerate(llms or (cfg.OPENAI_MODEL, "databricks-gpt-oss-120b")):
            args[f"llm{i}"] = m
    q = q.replace("LLMS", ", ".join(f":{k}" for k in args if k.startswith("llm")) or "NULL")
    cid = _cluster_id() if cluster_id == "auto" else cluster_id
    if mine_only and cid:
        args["cluster"] = cid
    q = q.replace("CLUSTER", "OR u.usage_metadata.cluster_id = :cluster" if "cluster" in args else "")
    out = spark.sql(q, args=args).toPandas()
    print(f"workspace {ws}, {since} .. {until or 'now'}: {out['dbu'].sum():,.2f} DBU, "
          f"${out['usd_list'].sum():,.2f} at list price"
          + (" -- MODEL_SERVING rows include other users of those shared endpoints" if mine_only else "")
          + (f"; cluster {cid} DBUs included, its Azure VM cost is not" if mine_only and cid else ""))
    return out


print(f"endpoint {ENDPOINT} | index {INDEX} | model {EMBED_MODEL}\n"
      f"catalog {CATALOG_TABLE} -> store {EMB} -> source {VEC}\n"
      "functions: status() embed() resume() pause() costs(since)")
