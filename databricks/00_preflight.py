"""Step 0: is this compute able to run the runbook? Run after Cells 1 and 2.

Written for classic clusters (e.g. Retail_DE_cluster), where access mode and
runtime are whatever the cluster was created with; it also passes on serverless.
Costs nothing but one ~2-token embedding call. One PASS / WARN / FAIL row per
check; any FAIL raises, so the cell turns red BEFORE 01/02 spend money.

    exec(open("databricks/00_preflight.py").read()); preflight()

Sources (checked 2026-10-10):
- Unity Catalog needs Standard or Dedicated access mode:
  https://learn.microsoft.com/en-us/azure/databricks/compute/access-mode-limitations
- Volumes need DBR 13.3 LTS+; 13.3 reached end of support 2026-08-22, so 14.3 LTS+:
  https://learn.microsoft.com/en-us/azure/databricks/volumes/
- Dedicated compute assigned to a GROUP cannot use model serving or AI Search:
  https://learn.microsoft.com/en-us/azure/databricks/compute/group-access
- Idle clusters keep billing DBUs + VMs until auto-termination:
  https://learn.microsoft.com/en-us/azure/databricks/compute/clusters-manage
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import app.config as cfg

_UC_MODES = {"DATA_SECURITY_MODE_STANDARD", "USER_ISOLATION",
             "DATA_SECURITY_MODE_DEDICATED", "SINGLE_USER"}


def _pf_row(rows, status, name, detail):
    rows.append((status, name, detail))
    print(f"{status:4s}  {name:34s} {detail}")


def _pf_check(rows, name, fn):
    """fn() -> (status, detail); an exception is a FAIL with its message."""
    try:
        status, detail = fn()
    except Exception as e:  # noqa: BLE001 -- a preflight reports, it never crashes
        status, detail = "FAIL", f"{type(e).__name__}: {str(e)[:160]}"
    _pf_row(rows, status, name, detail)


def _pf_cluster_id():
    try:
        cid = spark.conf.get("spark.databricks.clusterUsageTags.clusterId", None)
    except Exception:  # noqa: BLE001 -- serverless may refuse the key
        cid = None
    return cid or os.environ.get("DB_CLUSTER_ID")


def preflight():
    from databricks.sdk import WorkspaceClient
    import databricks.sdk.version as sdkv

    rows = []
    w = WorkspaceClient()
    dbr = os.environ.get("DATABRICKS_RUNTIME_VERSION", "")
    serverless = dbr.startswith("client.")

    def python():
        ok = sys.version_info >= (3, 10)
        return ("PASS" if ok else "FAIL",
                f"{sys.version.split()[0]} (the code uses 'X | None' annotations: 3.10+)")

    def runtime():
        if serverless:
            return "PASS", f"serverless ({dbr})"
        try:
            major, minor = (int(x) for x in dbr.split(".")[:2])
        except ValueError:
            return "WARN", f"unreadable DATABRICKS_RUNTIME_VERSION={dbr!r}: check the cluster page"
        if (major, minor) < (13, 3):
            return "FAIL", f"DBR {dbr}: UC Volumes need 13.3 LTS+ (use 15.4 / 16.4 / 17.3 LTS)"
        if (major, minor) < (14, 3):
            return "WARN", f"DBR {dbr} is end-of-support: use 15.4 / 16.4 / 17.3 LTS"
        if major >= 19:
            return "WARN", (f"DBR {dbr}: databricks-vectorsearch pins protobuf<6 and would "
                            "downgrade this runtime's 6.x; prefer an LTS (15.4-18)")
        return "PASS", f"DBR {dbr}"

    def sdk():
        v = tuple(int(x) for x in sdkv.__version__.split(".")[:2])
        # < 0.117 could raise CONTEXT_UNAVAILABLE_FOR_REMOTE_CLIENT on Spark Connect
        # (Standard) compute -- databricks-sdk-py issue #1463, fixed in 0.117.0
        return ("PASS" if v >= (0, 117) else "WARN",
                f"databricks-sdk {sdkv.__version__}" + ("" if v >= (0, 117) else
                " -- re-run Cell 1 (it installs --upgrade databricks-sdk>=0.117)"))

    def access_mode():
        if serverless:
            return "PASS", "serverless: Unity Catalog built in"
        cid = _pf_cluster_id()
        if not cid:
            return "WARN", "no cluster id found: check Access mode on the cluster page"
        c = w.clusters.get(cid)
        mode = getattr(c.data_security_mode, "value", c.data_security_mode)
        owner = c.single_user_name or ""
        name = f"{c.cluster_name} ({cid})"
        if mode in {"DATA_SECURITY_MODE_DEDICATED", "SINGLE_USER"} and owner and "@" not in owner:
            return "WARN", (f"{name}: Dedicated, assigned to {owner!r} -- if that is a GROUP, "
                            "model serving and AI Search are NOT reachable from it")
        if mode in _UC_MODES:
            return "PASS", f"{name}: {mode}"
        if mode == "DATA_SECURITY_MODE_AUTO":
            return "WARN", f"{name}: Auto -- confirm the page shows Standard or Dedicated"
        return "FAIL", (f"{name}: {mode or 'no access mode'} -- Unity Catalog needs Standard "
                        "or Dedicated access mode")

    def auto_terminate():
        if serverless:
            return "PASS", "serverless stops by itself"
        cid = _pf_cluster_id()
        if not cid:
            return "WARN", "unknown cluster"
        mins = w.clusters.get(cid).autotermination_minutes or 0
        if not mins:
            return "WARN", "auto-termination OFF: it bills DBUs + Azure VMs while idle"
        return ("PASS" if mins <= 120 else "WARN"), f"terminates after {mins} idle min"

    def unity_catalog():
        spark.sql(f"SHOW SCHEMAS IN {cfg.CAS_CATALOG}").collect()
        has = spark.catalog.tableExists(cfg.VS_TABLE)
        return "PASS", f"catalog {cfg.CAS_CATALOG} readable; {cfg.VS_TABLE} " + (
            "exists" if has else "not created yet (01 does it)")

    def catalog_file():
        ok = os.path.exists(cfg.VS_CATALOG_CSV)
        return ("PASS" if ok else "FAIL",
                cfg.VS_CATALOG_CSV + ("" if ok else " missing: upload products_enriched.csv"))

    def billing():
        # Optional: only 02 costs() needs it, so a missing grant must not block the run
        try:
            spark.sql("SELECT 1 FROM system.billing.usage LIMIT 1").collect()
        except Exception as e:  # noqa: BLE001
            return "WARN", (f"cannot read system.billing ({type(e).__name__}): costs() won't "
                            "work -- ask an admin for USE CATALOG system / SELECT on system.billing")
        return "PASS", "system.billing.usage readable (02 costs() works)"

    def endpoints():
        for name in (cfg.VS_EMBEDDING_MODEL, cfg.OPENAI_MODEL):
            w.serving_endpoints.get(name)
        return "PASS", f"{cfg.VS_EMBEDDING_MODEL}, {cfg.OPENAI_MODEL}"

    def embed_from_thread():
        # 02 embed() calls the endpoint from worker threads; prove auth works there
        from app import vector_search

        vecs, tokens = ThreadPoolExecutor(1).submit(
            vector_search.embed_texts, ["preflight"], True).result()
        return "PASS", f"{len(vecs[0])}-dim unit vector, {tokens} tokens"

    def vs_endpoint():
        from databricks.vector_search.client import VectorSearchClient

        try:
            ep = VectorSearchClient(disable_notice=True).get_endpoint(cfg.VS_ENDPOINT)
        except Exception as e:  # noqa: BLE001
            if "not found" in str(e).lower() or getattr(e, "status_code", None) == 404:
                return "PASS", f"{cfg.VS_ENDPOINT} absent: 02 resume() creates it"
            raise
        state = ((ep or {}).get("endpoint_status") or {}).get("state")
        return "PASS", f"{cfg.VS_ENDPOINT}: {state}"

    print(f"preflight: {cfg.CAS_CATALOG}.{cfg.CAS_SCHEMA} | embeddings {cfg.VS_EMBEDDING_MODEL}"
          f" | llm {cfg.OPENAI_MODEL}\n")
    for name, fn in [("python", python), ("runtime", runtime), ("databricks-sdk", sdk),
                     ("access mode", access_mode), ("auto-termination", auto_terminate),
                     ("unity catalog", unity_catalog), ("catalog file on the Volume", catalog_file),
                     ("billing tables", billing), ("serving endpoints", endpoints),
                     ("embedding from a worker thread", embed_from_thread),
                     ("vector search endpoint", vs_endpoint)]:
        _pf_check(rows, name, fn)

    fails = [n for s, n, _ in rows if s == "FAIL"]
    warns = [n for s, n, _ in rows if s == "WARN"]
    print(f"\n{len(rows) - len(fails) - len(warns)} PASS, {len(warns)} WARN, {len(fails)} FAIL")
    if fails:
        raise AssertionError(f"PREFLIGHT FAILED: {', '.join(fails)}")
    return rows
