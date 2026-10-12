"""Step 9: run CAS as a Databricks App (FastAPI backend + Streamlit UI) -- and stop it.

    exec(open("databricks/09_deploy_app.py").read())
    export_catalog()   # once, and after every 01 re-run: the App's catalog file
    stage()            # copy the app's files into APP_STAGE (no spaces, no notebooks)
    app_create()       # once: the App, its service principal, endpoint + Volume grants
    app_grant()        # after app_create() AND after every 02 resume(): index grant
    app_deploy()       # starts the App (billing starts) and deploys APP_STAGE
    app_share("someone@latentview.com")   # CAN USE; a group name works too
    app_status()
    app_stop()         # after the demo, then 02 pause()
    app_start()        # next demo (after 02 resume() + app_grant()): last deployment
    app_costs("2026-10-12")   # next day: the App's own DBUs

Functions are app_-prefixed because 02 is exec()ed into the same notebook and
defines status() / pause() / costs() of its own.

What the App needs, and why it is done this way (sources checked 2026-10-12):
- app.yaml at the ROOT of the deployed folder; a deployed folder cannot see files
  outside itself -- hence stage() instead of deploying the repo or databricks/:
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/deploy
- The App installs requirements.txt from that root on every deploy; the repo-root
  one pulls torch, so the bundle ships its own:
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/dependencies
- Apps cannot open /Volumes paths; app/vector_search.py downloads the catalog file
  through the Files API, which needs READ VOLUME (the Volume resource grants it):
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/uc-volumes
- The App runs as its own service principal. Resources grant it CAN QUERY on the two
  serving endpoints; the index needs USE CATALOG, USE SCHEMA and SELECT:
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/resources
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/vector-search
  The index is granted by SQL (app_grant), not as a resource: 02 pause()/resume()
  deletes and recreates it, and a recreated index is a new object without the grant.
- Billing: a RUNNING App bills (Medium = 0.5 DBU/h); a stopped one does not, and
  there is no idle auto-stop -- app_stop() after every demo:
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/compute-size
- Apps cannot be made public; users need an account identity + CAN USE:
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/permissions
  The catalog is Amazon Reviews 2023 (RESEARCH-ONLY licence): internal demos only.
"""

import io
import os
import time
from datetime import timedelta

import app.config as cfg
from app import vector_search

APP_NAME = cfg.APP_NAME
# The App reads THIS file (app.yaml VS_CATALOG_CSV): the Delta table's own rows and
# catalog_index, so the App hydrates exactly what the notebook evals hydrated
APP_CATALOG = f"/Volumes/{cfg.CAS_CATALOG}/{cfg.CAS_SCHEMA}/raw/app_catalog.csv"
APP_VOLUME = f"{cfg.CAS_CATALOG}.{cfg.CAS_SCHEMA}.raw"
APP_BUNDLE = "databricks/app_bundle"       # app.yaml, run_app.py, requirements.txt
APP_SOURCES = ["app", "pages", "ui.py", "ui_common.py", "ui_cart.py", "data/sponsored.json"]
_MARKER = ".cas_app_stage"                 # stage() only ever replaces its own folder


def _ws():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


def _app_stage():
    # Under the user's home, NOT the Git folder: its path has spaces, and the App's
    # service principal must be able to read the source folder (docs: deploy)
    return f"/Workspace/Users/{_ws().current_user.me().user_name}/{APP_NAME}"


APP_STAGE = os.environ.get("CAS_APP_STAGE") or _app_stage()


def _get_app():
    from databricks.sdk.errors import NotFound

    try:
        return _ws().apps.get(APP_NAME)
    except NotFound:
        return None


def _state(app):
    cs = getattr(app.compute_status, "state", None)
    st = getattr(app.app_status, "state", None)
    return (getattr(cs, "value", cs) or "?"), (getattr(st, "value", st) or "?")


def export_catalog(path=APP_CATALOG):
    """Write the App's catalog: dev.cas.products minus search_text, WITH catalog_index."""
    df = (spark.table(cfg.VS_TABLE).drop("search_text").orderBy("catalog_index")
          .toPandas().reset_index(drop=True))
    if not (df["catalog_index"].astype("int64").values == range(len(df))).all():
        raise RuntimeError(f"{cfg.VS_TABLE}.catalog_index is not dense 0..N-1: re-run 01.")
    df.to_csv(path, index=False)
    print(f"{path}: {len(df):,} rows, {os.path.getsize(path) / 1e6:,.0f} MB. "
          "The App reads it at every start (Files API download).")


def _bundle_env():
    """app.yaml's env as {name: value}: the values the App will actually run with."""
    import re

    text = open(os.path.join(APP_BUNDLE, "app.yaml"), encoding="utf-8").read()
    return dict(re.findall(r'-\s*name:\s*"?([A-Z0-9_]+)"?\s*\n\s*value:\s*"([^"]*)"', text))


def _check_bundle():
    """Refuse to ship an app.yaml that disagrees with what this notebook grants and checks."""
    env = _bundle_env()
    want = {"SEARCH_BACKEND": "databricks", "CAS_CATALOG": cfg.CAS_CATALOG,
            "CAS_SCHEMA": cfg.CAS_SCHEMA, "VS_INDEX": cfg.VS_INDEX, "VS_QUERY_MODE": "vector",
            "VS_EMBEDDING_MODEL": cfg.VS_EMBEDDING_MODEL, "VS_CATALOG_CSV": APP_CATALOG,
            "OPENAI_MODEL": cfg.OPENAI_MODEL, "OPENAI_AUTH": "databricks"}
    wrong = {k: (env.get(k), v) for k, v in want.items() if env.get(k) != v}
    if wrong:
        raise RuntimeError(f"{APP_BUNDLE}/app.yaml disagrees with this notebook "
                           f"(app.yaml, notebook): {wrong} -- the App would run a different "
                           "catalog/index/model than the one granted and checked here.")


def _app_sources():
    files = []
    for src in APP_SOURCES:
        if os.path.isdir(src):
            for root, dirs, fnames in os.walk(src):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                files += [os.path.join(root, f) for f in fnames if not f.endswith(".pyc")]
        else:
            files.append(src)
    pairs = [(f, f) for f in files] + [
        (os.path.join(APP_BUNDLE, f), f) for f in ("app.yaml", "run_app.py", "requirements.txt")]
    for local, _ in pairs:  # all checked BEFORE anything is deleted or uploaded
        if not os.path.isfile(local):
            raise RuntimeError(f"{local} missing: run 09 from the repo root (Cell 2's os.chdir)")
        if os.path.getsize(local) > 10 * 1024 * 1024:  # Apps refuse files over 10 MB
            raise RuntimeError(f"{local} is over 10 MB")
    return pairs


def stage(dest=None):
    """Copy the App's files (only those) into `dest`, replacing a previous stage()."""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.workspace import ImportFormat

    dest = dest or APP_STAGE
    _check_bundle()
    pairs = _app_sources()
    w = _ws()
    try:
        names = {os.path.basename(o.path) for o in w.workspace.list(dest)}
        if names and _MARKER not in names:
            raise RuntimeError(f"{dest} exists and was not made by stage(); not touching it.")
        w.workspace.delete(dest, recursive=True)
    except NotFound:
        pass
    w.workspace.mkdirs(dest)
    # Marker FIRST: a stage() that fails half-way is still recognised as ours next time
    w.workspace.upload(f"{dest}/{_MARKER}", io.BytesIO(b"made by databricks/09 stage()\n"),
                       format=ImportFormat.AUTO, overwrite=True)
    total = 0
    for local, rel in pairs:
        data = open(local, "rb").read()
        target = f"{dest}/{rel.replace(os.sep, '/')}"
        w.workspace.mkdirs(target.rsplit("/", 1)[0])
        # AUTO: a .py without the notebook header stays a FILE (the default, SOURCE,
        # would import every .py as a notebook and the App could not import it)
        w.workspace.upload(target, io.BytesIO(data), format=ImportFormat.AUTO, overwrite=True)
        total += len(data)
    print(f"staged {len(pairs)} files, {total / 1e6:.2f} MB -> {dest}")
    app = _get_app()
    if app is not None and app.service_principal_client_id and dest == APP_STAGE:
        # A recreated folder is a new object: the App's read grant did not survive
        _grant_stage_read(w, app.service_principal_client_id)


def _me():
    return _ws().current_user.me().user_name


def _own_app(force=False):
    """The App, refusing one another user created: names are unique per workspace."""
    app = _get_app()
    if app is not None and not force and app.creator and app.creator != _me():
        raise RuntimeError(f"App {APP_NAME} was created by {app.creator}, not you: set a different "
                           "CAS_APP_NAME before Cell 2 (or pass force=True if it IS yours).")
    return app


def _is_endpoint_refusal(e):
    from databricks.sdk.errors import PermissionDenied

    msg = str(e).lower()
    return (isinstance(e, PermissionDenied) or "permission" in msg) and (
        "endpoint" in msg or "serving" in msg)


def app_create(compute_size="MEDIUM", force=False):
    """Create the App (compute NOT started, so nothing bills yet) with its resources."""
    from databricks.sdk.service import apps as A

    existing = _own_app(force)
    if existing is not None:
        print(f"{APP_NAME} already exists ({existing.url}); compute/app: {_state(existing)}")
        return existing
    resources = [
        A.AppResource(name="llm-endpoint", serving_endpoint=A.AppResourceServingEndpoint(
            name=cfg.OPENAI_MODEL,
            permission=A.AppResourceServingEndpointServingEndpointPermission.CAN_QUERY)),
        A.AppResource(name="embedding-endpoint", serving_endpoint=A.AppResourceServingEndpoint(
            name=cfg.VS_EMBEDDING_MODEL,
            permission=A.AppResourceServingEndpointServingEndpointPermission.CAN_QUERY)),
        A.AppResource(name="catalog-volume", uc_securable=A.AppResourceUcSecurable(
            securable_full_name=APP_VOLUME,
            securable_type=A.AppResourceUcSecurableUcSecurableType.VOLUME,
            permission=A.AppResourceUcSecurableUcSecurablePermission.READ_VOLUME)),
    ]
    w = _ws()
    for attempt in (resources, resources[2:]):
        app = A.App(name=APP_NAME, compute_size=A.ComputeSize(compute_size), resources=attempt,
                    description="Context-Aware Search demo (internal; research-licensed catalog)")
        try:
            # create(), not create_and_wait(): the waiter waits for ACTIVE compute,
            # which no_compute=True deliberately never starts
            w.apps.create(app, no_compute=True)
            break
        except Exception as e:  # noqa: BLE001
            if attempt is not resources or not _is_endpoint_refusal(e):
                raise
            # Adding an endpoint resource needs CAN MANAGE on it, which a shared
            # account may not hold on the Databricks-hosted databricks-* endpoints
            print(f"create with endpoint resources refused ({type(e).__name__}: {str(e)[:200]})\n"
                  "retrying with the Volume only; app_deploy()'s start-up checks show whether the "
                  "App can still query the endpoints (if not: Edit > Resources in the App UI, "
                  "Serving endpoint, Can query -- by someone with CAN MANAGE on it)")
    created = _get_app()
    for _ in range(30):  # the service principal is provisioned asynchronously
        if created.service_principal_client_id:
            break
        time.sleep(2)
        created = _get_app()
    print(f"created {APP_NAME}: {created.url}\nservice principal {created.service_principal_client_id}"
          " -- now run app_grant()")
    return created


def _grant_stage_read(w, sp):
    """CAN_READ on the staged folder: the App's service principal must read its source."""
    from databricks.sdk.service.workspace import (WorkspaceObjectAccessControlRequest,
                                                  WorkspaceObjectPermissionLevel)

    oid = w.workspace.get_status(APP_STAGE).object_id
    w.workspace.update_permissions("directories", str(oid), access_control_list=[
        WorkspaceObjectAccessControlRequest(
            service_principal_name=sp, permission_level=WorkspaceObjectPermissionLevel.CAN_READ)])


def app_grant(force=False):
    """UC grants for the App's service principal. Re-run after EVERY 02 resume()."""
    app = _own_app(force)
    if app is None:
        raise RuntimeError("no App yet: run app_create() first")
    sp = app.service_principal_client_id
    stmts = [
        f"GRANT USE CATALOG ON CATALOG {cfg.CAS_CATALOG} TO `{sp}`",
        f"GRANT USE SCHEMA ON SCHEMA {cfg.CAS_CATALOG}.{cfg.CAS_SCHEMA} TO `{sp}`",
        f"GRANT READ VOLUME ON VOLUME {APP_VOLUME} TO `{sp}`",
        f"GRANT SELECT ON TABLE {cfg.VS_INDEX} TO `{sp}`",
    ]
    failed = 0
    for q in stmts:
        try:
            spark.sql(q)
            print(f"PASS  {q}")
        except Exception as e:  # noqa: BLE001 -- report every grant, then decide
            failed += 1
            print(f"FAIL  {q}\n      {type(e).__name__}: {str(e)[:200]}")
    try:
        _grant_stage_read(_ws(), sp)
        print(f"PASS  CAN_READ on {APP_STAGE}")
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"FAIL  CAN_READ on {APP_STAGE} (run stage() first?)\n"
              f"      {type(e).__name__}: {str(e)[:200]}")
    if failed:
        print("A FAIL usually means this account lacks MANAGE on that object: ask its owner "
              "to run the statement, or add the index in the App's UI under Edit > Resources "
              "(AI Search index, Can select).")


def _fingerprint(frame):
    import hashlib

    keys = frame.sort_values("catalog_index")["text_hash"].astype(str)
    return len(frame), hashlib.md5("\n".join(keys).encode()).hexdigest()


def _check_ready(force=False):
    """Everything the App needs that this notebook can see, checked BEFORE billing starts."""
    import pandas as pd
    from databricks.sdk.errors import NotFound

    table = spark.table(cfg.VS_TABLE)
    problems = []
    if not os.path.exists(APP_CATALOG):
        problems.append(f"{APP_CATALOG} missing: run export_catalog()")
    else:
        # Hydration is positional: an export from an OLDER 01 run with the same row
        # count would show the wrong product for every hit, and no start-up check sees it
        export = pd.read_csv(APP_CATALOG, usecols=["catalog_index", "text_hash"])
        if _fingerprint(export) != _fingerprint(table.select("catalog_index", "text_hash").toPandas()):
            problems.append(f"{APP_CATALOG} is not the current {cfg.VS_TABLE}: run export_catalog()")
    health = vector_search.index_health(table.count())
    if not health["ok"]:
        problems.append(f"index not usable ({health['detail']}): 02 resume(), then app_grant()")
    try:
        names = {os.path.basename(o.path) for o in _ws().workspace.list(APP_STAGE)}
        if _MARKER not in names:
            problems.append(f"{APP_STAGE} was not made by stage(): run stage()")
    except NotFound:
        problems.append(f"{APP_STAGE} missing: run stage()")
    if problems and not force:
        raise RuntimeError("not starting the App (it would bill and then fail):\n- "
                           + "\n- ".join(problems))
    for p in problems:
        print(f"WARNING (force=True): {p}")


def _settle(w, minutes=20):
    """Wait out a deployment already in flight -- start() redeploys the last one, and
    a new deployment created meanwhile fails (the Databricks CLI waits for the same)."""
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        app = w.apps.get(APP_NAME)
        dep = app.pending_deployment or app.active_deployment
        state = getattr(getattr(getattr(dep, "status", None), "state", None), "value", None)
        if app.pending_deployment is None and state != "IN_PROGRESS":
            return app
        time.sleep(10)
    raise TimeoutError(f"{APP_NAME}: a deployment is still in progress after {minutes} min")


def _watch(w, url, minutes=3):
    """After a start/deploy: the App process can still crash (e.g. a missing grant kills
    the backend's start-up), and compute keeps billing while it sits CRASHED."""
    deadline = time.time() + minutes * 60
    compute = status = "?"
    while time.time() < deadline:
        compute, status = _state(w.apps.get(APP_NAME))
        if status == "RUNNING":
            print(f"{APP_NAME} RUNNING: {url}\nThe backend needs ~1 minute more to load the catalog; "
                  f"{url}/logz should then show 'Start-up check index/embedding/llm: OK'.")
            return status
        time.sleep(15)
    print(f"WARNING: {APP_NAME} is {status} with compute {compute} -- STILL BILLING. "
          f"Read {url}/logz, then app_stop() or fix and app_deploy().")
    return status


def app_deploy(force=False):
    """Start the App's compute if needed (billing starts) and deploy APP_STAGE."""
    from databricks.sdk.service.apps import AppDeployment, AppDeploymentMode

    app = _own_app(force)
    if app is None:
        raise RuntimeError("no App yet: run app_create() and app_grant() first")
    _check_ready(force)
    w = _ws()
    _grant_stage_read(w, app.service_principal_client_id)  # cheap; survives a re-stage()
    started_here = False
    compute = _state(app)[0]
    if compute == "STARTING":
        w.apps.wait_get_app_active(APP_NAME, timeout=timedelta(minutes=20))
    elif compute != "ACTIVE":
        print(f"starting {APP_NAME} compute (billing starts now; a few minutes)...")
        w.apps.start_and_wait(APP_NAME, timeout=timedelta(minutes=20))
        started_here = True
    try:
        _settle(w)
        print(f"deploying {APP_STAGE} (installs requirements.txt; a few minutes)...")
        d = w.apps.deploy_and_wait(APP_NAME, AppDeployment(
            source_code_path=APP_STAGE, mode=AppDeploymentMode.SNAPSHOT),
            timeout=timedelta(minutes=30))
    except Exception as e:  # noqa: BLE001
        if started_here:
            print("deploy failed: stopping the compute this call started (it would keep billing)...")
            w.apps.stop_and_wait(APP_NAME, timeout=timedelta(minutes=20))
        raise RuntimeError(f"deploy failed: {e}\nRead the App's log: {app.url}/logz"
                           + ("" if started_here else " -- compute is still ACTIVE: app_stop()")) from e
    msg = getattr(d.status, "message", "") if d.status else ""
    print(f"deployed: {msg}")
    _watch(w, app.url)


def app_status():
    app = _get_app()
    if app is None:
        print(f"{APP_NAME}: does not exist")
        return None
    compute, status = _state(app)
    dep = app.active_deployment
    print(f"{APP_NAME}: compute {compute} (bills while ACTIVE), app {status}\n"
          f"url  {app.url}\nlogs {app.url}/logz\nsp   {app.service_principal_client_id}\n"
          f"by   {app.creator}\n"
          f"deployment {getattr(dep, 'deployment_id', None)} from "
          f"{getattr(dep, 'source_code_path', None)} at {getattr(dep, 'create_time', None)}")
    return app


def app_share(principal, level="CAN_USE", force=False):
    """Grant CAN USE to a user (contains @) or a group. Adds; never replaces existing grants."""
    from databricks.sdk.service.apps import AppAccessControlRequest, AppPermissionLevel

    if principal.strip().lower() in {"users", "account users", "all users"} and not force:
        # On a SHARED account that is everyone, each search spends LLM tokens, and the
        # catalog is research-licensed: name the demo group instead
        raise RuntimeError(f"{principal!r} is everyone on the account: name the demo "
                           "group or people instead (or force=True)")
    _own_app()
    who = {"user_name": principal} if "@" in principal else {"group_name": principal}
    _ws().apps.update_permissions(APP_NAME, access_control_list=[
        AppAccessControlRequest(permission_level=AppPermissionLevel(level), **who)])
    print(f"{principal}: {level} on {APP_NAME}")


def app_stop():
    """Stop the App's compute: it stops billing; config and deployment are kept."""
    _ws().apps.stop_and_wait(APP_NAME, timeout=timedelta(minutes=20))
    print(f"{APP_NAME} stopped. If you are done for the day: 02 pause() deletes the index.")


def app_start(force=False):
    """Start the App again; it redeploys its last deployment. Needs the index ONLINE."""
    app = _own_app(force)
    if app is None:
        raise RuntimeError("no App yet: app_create() ... app_deploy() first")
    _check_ready(force)
    w = _ws()
    _grant_stage_read(w, app.service_principal_client_id)
    w.apps.start_and_wait(APP_NAME, timeout=timedelta(minutes=20))
    _settle(w)
    _watch(w, app.url)


def app_costs(since, until=None):
    """The App's own billed DBUs (system.billing.usage, APPS rows) and list-price $."""
    app = _get_app()
    q = """
      SELECT u.usage_date, u.sku_name, round(sum(u.usage_quantity), 2) AS dbu,
             round(sum(u.usage_quantity * lp.pricing.effective_list.default), 2) AS usd_list
      FROM system.billing.usage u
      JOIN system.billing.list_prices lp
        ON u.sku_name = lp.sku_name AND u.cloud = lp.cloud
       AND u.usage_end_time >= lp.price_start_time
       AND (lp.price_end_time IS NULL OR u.usage_end_time < lp.price_end_time)
      WHERE u.billing_origin_product = 'APPS' AND u.usage_metadata.app_id = :app_id
        AND u.usage_date BETWEEN :since AND :until
      GROUP BY ALL ORDER BY u.usage_date"""
    out = spark.sql(q, args={"app_id": getattr(app, "id", "") or "", "since": since,
                             "until": until or "2999-12-31"}).toPandas()
    print(f"{APP_NAME} since {since}: {out['dbu'].sum():,.2f} DBU, ${out['usd_list'].sum():,.2f} "
          "at list price (billing lags ~12h; the index's endpoint is in 02 costs())")
    return out


print(f"app {APP_NAME} | stage {APP_STAGE} | catalog file {APP_CATALOG}\n"
      "functions: export_catalog() stage() app_create() app_grant() app_deploy() "
      "app_share(who) app_status() app_stop() app_start() app_costs(since)")
