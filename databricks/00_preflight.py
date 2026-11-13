"""Step 0: is this compute able to run the runbook? Run after Cells 1 and 2.

Written for classic clusters (e.g. Retail_DE_cluster), where access mode and
runtime are whatever the cluster was created with; it also passes on serverless.
Costs nothing but one ~2-token embedding call, one tiny LLM call (~50 tokens; added
2026-11-13, when Enforce Unity Gateway refused every query while an endpoint GET still
PASSed) and a tiny probe file written to and deleted from the App's memo volume, when
it exists. One PASS / WARN / FAIL row per
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
- The App keeps its LLM reply memo on a UC Volume (local disk is lost on restart):
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/key-concepts
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import app.config as cfg

if "databricks" not in sys.path:  # exec()ed again on every re-run of the cell
    sys.path.insert(0, "databricks")
import vs_shim  # noqa: E402 -- the App's memo volume and pins, shared with 09

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


def _pf_versus_app(installed, pins=None):
    """(status, detail): this notebook's pandas / numpy against the App's pins.

    The notebooks (10's page eval above all) run the app's code on the runtime's own
    pandas, the App on the pinned one: a different major can change NaN/str dtype
    semantics under the same code. A report, not a fix -- do NOT %pip install pandas 3
    on DBR (PySpark's toPandas support for it is unverified).
    `installed` maps a distribution name to its version, or None when absent.
    """
    if pins is None:
        try:
            pins = vs_shim.bundle_pins()
        except OSError as e:
            return "WARN", (f"{vs_shim.APP_REQUIREMENTS} unreadable ({type(e).__name__}): "
                            "run from the repo root (Cell 2's os.chdir)")
    rows, off = [], []
    for name in ("pandas", "numpy"):
        have, want = installed.get(name), pins.get(name)
        rows.append(f"{name} {have or '-'} (App {want or 'unpinned'})")
        if want and have != want:
            off.append(name)
    if not off:
        return "PASS", ", ".join(rows)
    return "WARN", (", ".join(rows) + f" -- {'/'.join(off)} differ: a page-eval number here may "
                    "not be what the App serves; do not %pip install pandas 3 on DBR")


def _pf_memo_probe(volume_path):
    """(status, detail) for the App's memo volume, by a dry write the notebook user makes.

    The probe goes to the volume ROOT (not the memo folder), so it never mixes with
    memo entries, and is deleted at once. It proves the volume exists and takes
    writes; the App's own WRITE VOLUME grant is 09 app_grant()'s job (its start-up
    check reports it). WARN, never FAIL: 01-08 and 10 do not need it.
    """
    import json
    import uuid

    if not os.path.isdir(volume_path):
        return "WARN", (f"{volume_path} missing (or not readable): 09 app_grant() creates it "
                        "(needs CREATE VOLUME on the schema); only the App needs it")
    probe = os.path.join(volume_path, f".preflight_{uuid.uuid4().hex}.json")
    body = json.dumps({"probe": "00_preflight"})
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write(body)
        with open(probe, encoding="utf-8") as f:
            back = f.read()
    except OSError as e:
        return "WARN", f"{volume_path} exists but a dry write failed ({type(e).__name__})"
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass
    if back != body:
        return "WARN", f"{volume_path}: the dry write read back different bytes"
    return "PASS", f"{volume_path} exists and took a dry write (removed)"


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
        return "PASS", f"DBR {dbr}" + (" (Beta, non-LTS)" if major >= 19 else "")

    def packages():
        # The 2026-10-10 failure: databricks-vectorsearch pinned protobuf<6, pip
        # downgraded DBR 19's 6.33.5 and the kernel died. The runbook no longer
        # installs it; a protobuf older than the one Spark Connect was generated
        # with means a stale notebook env -- detach and re-attach the notebook.
        import importlib.metadata as md

        try:
            import dotenv, openai, sklearn  # noqa: F401 -- imported by app/ at start-up
        except ImportError as e:
            # Seen 2026-10-12: Cell 1 had run, but the cluster had auto-terminated since
            return "FAIL", (f"{e} -- run Cell 1, then Cell 2, in THIS notebook. %pip installs "
                            "are notebook-scoped and vanish when the cluster restarts or the "
                            "notebook detaches")

        def ver(p):
            try:
                return md.version(p)
            except md.PackageNotFoundError:
                return "-"

        pb = ver("protobuf")
        spark_pb = ""
        try:
            from pyspark.sql.connect.proto import base_pb2  # noqa: F401 -- validates gencode
        except ImportError:
            pass  # no Spark Connect protos on this runtime
        except Exception as e:  # noqa: BLE001
            spark_pb = f"{type(e).__name__}: {str(e)[:120]}"
        if spark_pb:
            return "FAIL", f"protobuf {pb} breaks Spark Connect ({spark_pb}): detach + re-attach"
        oa = ver("openai")
        if oa != "-" and not oa.startswith("2."):
            return "WARN", (f"openai {oa}: app/llm_client.py is tested on 2.x (2.42.0) -- "
                            "re-run Cell 1, it pins openai==2.42.0")
        try:
            vs = md.version("databricks-vectorsearch")
            return "WARN", (f"databricks-vectorsearch {vs} installed but unused; it pins "
                            "protobuf<6 -- do not %pip install it")
        except md.PackageNotFoundError:
            pass
        return "PASS", f"protobuf {pb}, openai {ver('openai')}, databricks-sdk {ver('databricks-sdk')}"

    def sdk():
        v = tuple(int(x) for x in sdkv.__version__.split(".")[:2])
        # < 0.117 could raise CONTEXT_UNAVAILABLE_FOR_REMOTE_CLIENT on Spark Connect
        # (Standard) compute -- databricks-sdk-py issue #1463, fixed in 0.117.0
        return ("PASS" if v >= (0, 117) else "WARN",
                f"databricks-sdk {sdkv.__version__}" + ("" if v >= (0, 117) else
                " -- re-run Cell 1 (it installs databricks-sdk>=0.117)"))

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

    def versus_app():
        import importlib.metadata as md

        def have(p):
            try:
                return md.version(p)
            except md.PackageNotFoundError:
                return None

        return _pf_versus_app({p: have(p) for p in ("pandas", "numpy")})

    def memo_volume():
        return _pf_memo_probe(vs_shim.memo_volume_path())

    def billing():
        # Optional: only 02 costs() needs it, so a missing grant must not block the run
        try:
            spark.sql("SELECT 1 FROM system.billing.usage LIMIT 1").collect()
        except Exception as e:  # noqa: BLE001
            return "WARN", (f"cannot read system.billing ({type(e).__name__}): costs() won't "
                            "work -- ask an admin for USE CATALOG system / SELECT on system.billing")
        return "PASS", "system.billing.usage readable (02 costs() works)"

    def endpoints():
        # A Unity Gateway model service (system.ai.<model>) is no serving endpoint: a GET
        # would 404. The two call rows below prove both routes; a GET alone never did
        # (it PASSed while Enforce Unity Gateway refused every query, 2026-11-13).
        names = [cfg.VS_EMBEDDING_SERVICE or cfg.VS_EMBEDDING_MODEL, cfg.OPENAI_MODEL]
        for name in names:
            if name.count(".") != 2:
                w.serving_endpoints.get(name)
        return "PASS", ", ".join(n + (" (model service: see the call rows)" if n.count(".") == 2
                                      else "") for n in names)

    def llm_call():
        # One tiny call through the app's own client (memo off: a stored reply would
        # PASS over a refused route). The reply text is never printed.
        from app import llm_client

        llm_client._singleton = None  # built from Cell 2's values, not a stale client
        reply = llm_client.generate_json('Reply with the json object {"ok": true}',
                                         temperature=0.0, memo=False)
        ok = isinstance(reply, dict) and reply.get("ok") is True
        return ("PASS" if ok else "WARN"), (f"{cfg.OPENAI_MODEL} at {cfg.OPENAI_BASE_URL}: "
                                            + ("JSON reply as asked" if ok else "unexpected reply"))

    def embed_from_thread():
        # 02 embed() calls the endpoint from worker threads; prove auth works there
        from app import vector_search

        vecs, tokens = ThreadPoolExecutor(1).submit(
            vector_search.embed_texts, ["preflight"], True).result()
        return "PASS", f"{len(vecs[0])}-dim unit vector, {tokens} tokens"

    def vs_endpoint():
        from app import vector_search

        ep = vector_search.endpoint_info(cfg.VS_ENDPOINT)  # REST via the SDK; raises unless 404
        if ep is None:
            return "PASS", f"{cfg.VS_ENDPOINT} absent: 02 resume() creates it"
        return "PASS", f"{cfg.VS_ENDPOINT}: {(ep.get('endpoint_status') or {}).get('state')}"

    emb = (f"{cfg.VS_EMBEDDING_SERVICE} via Unity Gateway (vectors labelled "
           f"{cfg.VS_EMBEDDING_MODEL})" if cfg.VS_EMBEDDING_SERVICE else cfg.VS_EMBEDDING_MODEL)
    print(f"preflight: {cfg.CAS_CATALOG}.{cfg.CAS_SCHEMA} | embeddings {emb}"
          f" | llm {cfg.OPENAI_MODEL}\n")
    for name, fn in [("python", python), ("runtime", runtime), ("packages", packages),
                     ("databricks-sdk", sdk), ("pandas / numpy vs the App", versus_app),
                     ("access mode", access_mode), ("auto-termination", auto_terminate),
                     ("unity catalog", unity_catalog), ("catalog file on the Volume", catalog_file),
                     ("App memo volume (dry write)", memo_volume),
                     ("billing tables", billing), ("serving endpoints", endpoints),
                     ("embedding from a worker thread", embed_from_thread),
                     ("llm call (JSON mode)", llm_call),
                     ("vector search endpoint", vs_endpoint)]:
        _pf_check(rows, name, fn)

    fails = [n for s, n, _ in rows if s == "FAIL"]
    warns = [n for s, n, _ in rows if s == "WARN"]
    print(f"\n{len(rows) - len(fails) - len(warns)} PASS, {len(warns)} WARN, {len(fails)} FAIL")
    if fails:
        raise AssertionError(f"PREFLIGHT FAILED: {', '.join(fails)}")
    return rows
