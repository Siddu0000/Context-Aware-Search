"""Step 9: run CAS as a Databricks App (FastAPI backend + Streamlit UI) -- and stop it.

    exec(open("databricks/09_deploy_app.py").read())
    export_catalog()   # once, and after every 01 re-run: the App's catalog file
    stage()            # copy the app's files into APP_STAGE (no spaces, no notebooks)
    app_create()       # once: the App, its service principal, endpoint + Volume grants
    app_grant()        # after app_create() AND after every 02 resume(): index grant;
                       # also creates + grants the memo volume (an existing App too)
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
- The App's in-memory state and local disk are lost on every restart ("Apps don't
  preserve in-memory state after restarts"; the local file system "is lost when the
  app restarts"), and app_stop() after every demo restarts it, so the LLM reply memo
  that makes searches repeatable (LLM_MEMO, 2026-10-27) lives on a UC Volume of its
  own, APP_MEMO_VOLUME. The App writes it through the Files API, which needs WRITE
  VOLUME (resource "Can read and write") plus USE CATALOG / USE SCHEMA:
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/key-concepts
  https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/uc-volumes
  Memo files hold model replies that can echo a shopper's words (never the prompt):
  inside `dev`, `account users` hold broad privileges, so treat the volume as
  readable -- and, if they hold WRITE VOLUME, writable (the App replays what it finds
  there) -- account-wide; app_grant() prints a WARN when it sees either. A client
  deployment needs its own catalog/schema. With app.yaml LLM_MEMO="false" nothing
  creates or grants the volume.
"""

import io
import os
import sys
import time
from datetime import timedelta

import app.config as cfg
from app import vector_search

if "databricks" not in sys.path:  # exec()ed again on every re-run of the cell
    sys.path.insert(0, "databricks")
import vs_shim  # noqa: E402 -- the memo location and App pins, shared with 00 and 12

APP_NAME = cfg.APP_NAME
# The App reads THIS file (app.yaml VS_CATALOG_CSV): the Delta table's own rows and
# catalog_index, so the App hydrates exactly what the notebook evals hydrated
APP_CATALOG = f"/Volumes/{cfg.CAS_CATALOG}/{cfg.CAS_SCHEMA}/raw/app_catalog.csv"
APP_VOLUME = f"{cfg.CAS_CATALOG}.{cfg.CAS_SCHEMA}.raw"
# The App's LLM reply memo (app.yaml LLM_MEMO_DIR): READ + WRITE VOLUME, nothing else
APP_MEMO_VOLUME = vs_shim.memo_volume()
APP_MEMO_DIR = vs_shim.app_memo_dir()
APP_BUNDLE = "databricks/app_bundle"       # app.yaml, run_app.py, requirements.txt
# Directories are copied whole; "ui*.py" is a glob, so a new top-level ui_ module
# ships without an edit here. _app_sources() ALSO stages any other repo module the
# App imports (an AST walk), so an import added outside these cannot crash the App.
APP_SOURCES = ["app", "pages", "ui*.py", "data/sponsored.json"]
_MARKER = ".cas_app_stage"                 # stage() only ever replaces its own folder
# app/config.py reads LLM_MEMO as on for exactly _TRUE and off for anything else, so a
# word in neither set ("on", "enabled") would silently turn the memo off: refused
_TRUE, _FALSE = {"1", "true", "yes"}, {"0", "false", "no"}


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


def _memo_on(env):
    """app.yaml's LLM_MEMO as a bool; None when missing or not a plain yes/no word.

    Lower-cased but NOT stripped, exactly as app/config.py reads it (" true" is off there).
    """
    v = (env.get("LLM_MEMO") or "").lower()
    return True if v in _TRUE else False if v in _FALSE else None


def _check_bundle():
    """Refuse to ship an app.yaml that disagrees with what this notebook grants and checks."""
    env = _bundle_env()
    want = {"SEARCH_BACKEND": "databricks", "CAS_CATALOG": cfg.CAS_CATALOG,
            "CAS_SCHEMA": cfg.CAS_SCHEMA, "VS_INDEX": cfg.VS_INDEX, "VS_QUERY_MODE": "vector",
            "VS_EMBEDDING_MODEL": cfg.VS_EMBEDDING_MODEL, "VS_CATALOG_CSV": APP_CATALOG,
            "OPENAI_MODEL": cfg.OPENAI_MODEL, "OPENAI_AUTH": "databricks",
            # The ONLY memo path app_grant() grants WRITE VOLUME on: any other path is
            # the App's local disk (lost on restart) or a volume it cannot write
            "LLM_MEMO_DIR": APP_MEMO_DIR}
    wrong = {k: (env.get(k), v) for k, v in want.items() if env.get(k) != v}
    # The ROUTE too (Unity Gateway, 2026-11-13): absent from app.yaml = the pay-per-token
    # route, which must then be what this notebook tested
    if (env.get("VS_EMBEDDING_SERVICE") or "") != cfg.VS_EMBEDDING_SERVICE:
        wrong["VS_EMBEDDING_SERVICE"] = (env.get("VS_EMBEDDING_SERVICE"), cfg.VS_EMBEDDING_SERVICE)
    from urllib.parse import urlparse

    tested = "/" + (urlparse(cfg.OPENAI_BASE_URL or "").path.strip("/") or "serving-endpoints")
    shipped = "/" + (env.get("OPENAI_BASE_PATH") or "serving-endpoints").strip().strip("/")
    if shipped != tested:
        wrong["OPENAI_BASE_PATH"] = (env.get("OPENAI_BASE_PATH"), tested)
    if _memo_on(env) is None:
        # Spelled out, never left to the code default: app.yaml is the App's whole config
        wrong["LLM_MEMO"] = (env.get("LLM_MEMO"), "true")
    if wrong:
        raise RuntimeError(f"{APP_BUNDLE}/app.yaml disagrees with this notebook "
                           f"(app.yaml, notebook): {wrong} -- the App would run a different "
                           "catalog/index/model/memo than the one granted and checked here. "
                           "Every value must be a plain quoted `value:` (a `valueFrom:` is "
                           "invisible to this check).")
    if not _memo_on(env):
        print("NOTE: app.yaml sets LLM_MEMO=false -- the App re-asks the LLM on every new "
              "search and after every restart, so repeated searches can differ.")


def _repo_module(mod):
    """The repo file that `import mod` loads, or None (stdlib, third-party, an attribute).

    databricks/ never counts: it merges with the pip `databricks` namespace package
    (databricks.sdk), and nothing in it is part of the App.
    """
    path = mod.replace(".", "/")
    if not path or path.split("/")[0] == "databricks":
        return None
    for cand in (f"{path}.py", f"{path}/__init__.py"):
        if os.path.isfile(cand):
            return cand
    return None


def _imported_modules(roots):
    """Every repo .py file reachable from `roots` by import, lazy imports included.

    Static (AST) and conservative: an import inside a function still counts, since
    the App may reach it (a Gemini-only branch costs one small file, a missing one an
    ImportError mid-demo). Each module's parent package __init__.py comes along,
    because Python runs it before the module.
    """
    import ast

    seen, queue = set(), [r for r in roots if os.path.isfile(r)]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        parts = path.split("/")[:-1]
        queue += [p for p in ("/".join(parts[:i]) + "/__init__.py"
                              for i in range(1, len(parts) + 1)) if os.path.isfile(p)]
        tree = ast.parse(open(path, encoding="utf-8-sig").read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:  # relative: resolve against this file's package
                    pkg = parts[:len(parts) - node.level + 1]
                    base = ".".join(pkg + ([base] if base else []))
                mods = [base] + [f"{base}.{a.name}" for a in node.names]
            else:
                continue
            queue += [hit for hit in map(_repo_module, mods) if hit and hit not in seen]
    return seen


def _expand(src):
    """APP_SOURCES entry -> files ('/'-separated): a glob, a directory or one file."""
    import glob

    if any(ch in src for ch in "*?["):
        hits = sorted(glob.glob(src))
        if not hits:
            raise RuntimeError(f"nothing matches {src}: run 09 from the repo root (Cell 2's os.chdir)")
        return [h.replace(os.sep, "/") for h in hits]
    if os.path.isdir(src):
        out = []
        for root, dirs, fnames in os.walk(src):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            out += [os.path.join(root, f).replace(os.sep, "/")
                    for f in sorted(fnames) if not f.endswith(".pyc")]
        return out
    return [src]


def _app_sources():
    files = [f for src in APP_SOURCES for f in _expand(src)]
    listed = set(files)
    # uvicorn imports app.main, Streamlit runs ui.py and the pages: walk from every
    # staged .py file, so whatever they import from elsewhere in the repo ships too
    extra = sorted(_imported_modules([f for f in files if f.endswith(".py")]) - listed)
    if extra:
        print(f"also staging {len(extra)} module(s) the App imports: {', '.join(extra)}")
    files += extra
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
    return (isinstance(e, PermissionDenied) or "permission" in msg) and any(
        w in msg for w in ("endpoint", "serving", "model service", "model_service", "system.ai"))


def _model_resource(A, key, name):
    """The App resource that lets it call `name`, or None (then the start-up checks say).

    A Unity Gateway model service (a three-part name, "system.ai.gte-large-en") is a
    UC securable with "Can execute"; anything else a serving endpoint with "Can query".
    MODEL_SERVICE arrived in databricks-sdk 0.151.0 (Public Preview): an older SDK gets
    no resource, and the App relies on EXECUTE its service principal already holds
    (account users hold it on system.ai by default).
    """
    if name.count(".") != 2:
        return A.AppResource(name=f"{key}-endpoint", serving_endpoint=A.AppResourceServingEndpoint(
            name=name,
            permission=A.AppResourceServingEndpointServingEndpointPermission.CAN_QUERY))
    kind = getattr(A.AppResourceUcSecurableUcSecurableType, "MODEL_SERVICE", None)
    perm = getattr(A.AppResourceUcSecurableUcSecurablePermission, "EXECUTE", None)
    if kind is None or perm is None:
        print(f"this databricks-sdk has no MODEL_SERVICE app resource: {name} is not attached "
              "(App > Edit > Resources > Model > Unity Gateway, Can execute)")
        return None
    return A.AppResource(name=f"{key}-model", uc_securable=A.AppResourceUcSecurable(
        securable_full_name=name, securable_type=kind, permission=perm))


def _is_memo_volume_refusal(e):
    """A create() error about the memo volume resource (no MANAGE on it, or it is gone).

    Only when the message NAMES the memo volume and not the catalog one: a refused
    catalog volume must raise, not cost a second create() under a wrong diagnosis.
    Unity Catalog quotes names with backticks ('`dev`.`cas`.`raw`'), so those go first.
    """
    import re

    msg = str(e).lower().replace("`", "")
    memo = re.escape(vs_shim.MEMO_VOLUME_NAME.lower())
    raw = re.escape(APP_VOLUME.rsplit(".", 1)[-1].lower())
    return bool(re.search(rf"\b{memo}\b", msg)) and not re.search(rf"\b{raw}\b", msg)


def _volume_exists(full_name):
    try:
        spark.sql(f"DESCRIBE VOLUME {full_name}").collect()
        return True
    except Exception:  # noqa: BLE001 -- missing, or not allowed to see it
        return False


def _memo_wanted():
    """False only when app.yaml turns the memo OFF (the documented revert): then nothing
    creates the volume or grants WRITE VOLUME on it -- the App would not use it. An
    unreadable app.yaml (wrong cwd) or an unreadable value counts as on: stage()'s
    _check_bundle() refuses those anyway, and a grant too many is the safer mistake."""
    try:
        return _memo_on(_bundle_env()) is not False
    except OSError:
        return True


_MEMO_OFF_NOTE = ("NOTE  app.yaml sets LLM_MEMO=false: the memo volume {v} is not created or "
                  "granted (re-run app_grant() after turning it back on)")


def _memo_volume_ready():
    """Create the memo volume if missing; (PASS | OK | FAIL, report line). Never raises.

    CREATE VOLUME IF NOT EXISTS is a no-op on an existing volume, but still needs
    CREATE VOLUME on the schema; without it, an existing volume is fine (OK).
    """
    q = (f"CREATE VOLUME IF NOT EXISTS {APP_MEMO_VOLUME} "
         "COMMENT 'CAS App LLM reply memo (databricks/09); model replies, never prompts'")
    try:
        spark.sql(q)
        status, line = "PASS", q
    except Exception as e:  # noqa: BLE001 -- report, then decide from what exists
        if _volume_exists(APP_MEMO_VOLUME):
            status, line = "OK", (f"{q}\n      not run ({type(e).__name__}), but {APP_MEMO_VOLUME} "
                                  "exists")
        else:
            return "FAIL", (f"{q}\n      {type(e).__name__}: {str(e)[:200]}\n      ask an owner of "
                            f"{cfg.CAS_CATALOG}.{cfg.CAS_SCHEMA} to create it (CREATE VOLUME)")
    try:
        os.makedirs(APP_MEMO_DIR, exist_ok=True)  # /Volumes is mounted in a notebook
    except OSError as e:
        # Harmless: the App's first write creates the folder through the Files API
        line += f"\n      ({APP_MEMO_DIR} not created here: {type(e).__name__})"
    return status, line


def _memo_resource(A):
    """The memo volume as an App resource ("Can read and write"), or None if this SDK
    predates WRITE_VOLUME -- app_grant() then grants it by SQL, which is equivalent."""
    perm = getattr(A.AppResourceUcSecurableUcSecurablePermission, "WRITE_VOLUME", None)
    if perm is None:
        print("this databricks-sdk has no WRITE_VOLUME app resource: the memo volume is "
              "granted by app_grant() (SQL) instead")
        return None
    return A.AppResource(name="memo-volume", uc_securable=A.AppResourceUcSecurable(
        securable_full_name=APP_MEMO_VOLUME,
        securable_type=A.AppResourceUcSecurableUcSecurableType.VOLUME, permission=perm))


def app_create(compute_size="MEDIUM", force=False):
    """Create the App (compute NOT started, so nothing bills yet) with its resources."""
    from databricks.sdk.service import apps as A

    existing = _own_app(force)
    if existing is not None:
        # create() is the only place resources are set: an existing App gets the memo
        # volume (and any other grant) from app_grant(), which runs them as SQL
        print(f"{APP_NAME} already exists ({existing.url}); compute/app: {_state(existing)}\n"
              "resources unchanged here: run app_grant() (it also creates + grants the memo volume)")
        return existing
    models = [r for r in (_model_resource(A, "llm", cfg.OPENAI_MODEL),
                          _model_resource(A, "embedding",
                                          cfg.VS_EMBEDDING_SERVICE or cfg.VS_EMBEDDING_MODEL))
              if r is not None]
    resources = models + [
        A.AppResource(name="catalog-volume", uc_securable=A.AppResourceUcSecurable(
            securable_full_name=APP_VOLUME,
            securable_type=A.AppResourceUcSecurableUcSecurableType.VOLUME,
            permission=A.AppResourceUcSecurableUcSecurablePermission.READ_VOLUME)),
    ]
    # APPENDED after the catalog volume: the endpoint fallback below drops the model
    # resources only, so both volumes stay. A resource on a volume that does not exist
    # would fail the whole create(), so the volume is made (or found) first.
    memo = None
    if _memo_wanted():
        status, line = _memo_volume_ready()
        print(f"{status:4s}  {line}")
        memo = _memo_resource(A) if status != "FAIL" else None
        if memo is not None:
            resources.append(memo)
    else:
        print(_MEMO_OFF_NOTE.format(v=APP_MEMO_VOLUME))
    w = _ws()
    # Two independent fallbacks, in whichever order the server reports the refusals;
    # each fires at most once, so the loop ends (at most three create() calls)
    drop_endpoints = drop_memo = False
    while True:
        attempt = [r for r in resources
                   if not (drop_endpoints and any(r is m for m in models))
                   and not (drop_memo and r is memo)]
        app = A.App(name=APP_NAME, compute_size=A.ComputeSize(compute_size), resources=attempt,
                    description="Context-Aware Search demo (internal; research-licensed catalog)")
        try:
            # create(), not create_and_wait(): the waiter waits for ACTIVE compute,
            # which no_compute=True deliberately never starts
            w.apps.create(app, no_compute=True)
            break
        except Exception as e:  # noqa: BLE001
            if not drop_endpoints and _is_endpoint_refusal(e):
                # Adding an endpoint resource needs CAN MANAGE on it, which a shared
                # account may not hold on the Databricks-hosted databricks-* endpoints
                drop_endpoints = True
                print(f"create with endpoint resources refused ({type(e).__name__}: {str(e)[:200]})\n"
                      "retrying with the Volumes only; app_deploy()'s start-up checks show whether "
                      "the App can still call the models (if not: Edit > Resources in the App "
                      "UI, Model > Unity Gateway, Can execute -- or, for a databricks-* "
                      "endpoint, Serving endpoint, Can query by someone with CAN MANAGE on it)")
            elif memo is not None and not drop_memo and _is_memo_volume_refusal(e):
                # The user adding a resource needs CAN MANAGE on it: a volume someone
                # else created may refuse. app_grant() grants the same by SQL.
                drop_memo = True
                print(f"create with the memo volume resource refused ({type(e).__name__}: "
                      f"{str(e)[:200]})\nretrying without it; app_grant() grants READ + WRITE "
                      f"VOLUME on {APP_MEMO_VOLUME} by SQL")
            else:
                raise
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


_EVERYONE = {"account users", "users"}


def _held_by_everyone(privilege, securables):
    """Who already gives `privilege` to everyone on one of `securables`, else None.

    An App's service principal is an account identity, so a grant to `account users`
    reaches it -- on 2026-10-13 `dev` granted USE CATALOG that way while this account
    could not grant it (no MANAGE), which made a harmless FAIL look like a blocker.
    """
    for kind, name in securables:
        try:
            rows = spark.sql(f"SHOW GRANTS ON {kind} {name}").collect()
        except Exception:  # noqa: BLE001 -- not allowed to look: undecided, not covered
            continue
        for r in rows:
            d = r.asDict()
            if (str(d.get("Principal", "")).lower() in _EVERYONE
                    and str(d.get("ActionType", "")).upper() in {privilege, "ALL PRIVILEGES"}):
                return f"`{d.get('Principal')}` hold {privilege} on {kind.lower()} {name}"
    return None


def _warn_memo_exposure(securables):
    """WARN when everyone on the account can read or write the memo volume (never a FAIL).

    The App replays what it finds there to shoppers, and its replies can echo their
    words: everyone reading it is a privacy question, everyone writing it means an
    entry nobody's model wrote can be served (the guard re-screens it, but product
    words and assistant prose pass). `dev` grants `account users` broadly (2026-10-13),
    so on the PoC this is expected; it is printed so nobody ships it unnoticed.
    """
    held = [h for h in (_held_by_everyone(p, securables) for p in ("READ VOLUME", "WRITE VOLUME"))
            if h]
    if held:
        print(f"WARN  memo volume {APP_MEMO_VOLUME} is open to everyone: {'; '.join(held)}\n"
              "      every account user can read the App's stored replies (they can echo "
              "shopper words)" + (" and write entries the App would replay to shoppers"
                                  if any("WRITE VOLUME" in h for h in held) else "")
              + ".\n      Internal PoC on dev only: a client deployment needs its own schema "
              "where only the App holds WRITE VOLUME.")


def app_grant(force=False):
    """UC grants for the App's service principal. Re-run after EVERY 02 resume().

    Also the memo volume: created if missing, then READ + WRITE VOLUME. app_create()
    returns early for an App that already exists (cas-app was created 2026-10-13,
    before the memo), so this is how an existing App gets it.
    """
    app = _own_app(force)
    if app is None:
        raise RuntimeError("no App yet: run app_create() first")
    sp = app.service_principal_client_id
    cat = ("CATALOG", cfg.CAS_CATALOG)
    schema = ("SCHEMA", f"{cfg.CAS_CATALOG}.{cfg.CAS_SCHEMA}")
    memo = ("VOLUME", APP_MEMO_VOLUME)
    memo_wanted = _memo_wanted()
    # (privilege, object, its parents): a privilege on a parent is inherited
    grants = [
        ("USE CATALOG", cat, []),
        ("USE SCHEMA", schema, [cat]),
        ("READ VOLUME", ("VOLUME", APP_VOLUME), [schema, cat]),
        ("SELECT", ("TABLE", cfg.VS_INDEX), [schema, cat]),
    ]
    failed = 0
    memo_missing = False
    if memo_wanted:
        grants += [("READ VOLUME", memo, [schema, cat]), ("WRITE VOLUME", memo, [schema, cat])]
        status, line = _memo_volume_ready()
        print(f"{status:4s}  {line}")
        memo_missing = status == "FAIL"
        failed += memo_missing
    else:
        print(_MEMO_OFF_NOTE.format(v=APP_MEMO_VOLUME))
    for privilege, (kind, name), parents in grants:
        q = f"GRANT {privilege} ON {kind} {name} TO `{sp}`"
        if memo_missing and (kind, name) == memo:
            # An inherited privilege would read as OK here, yet the App's memo
            # start-up check fails without the volume itself
            failed += 1
            print(f"FAIL  {q}\n      not run: {APP_MEMO_VOLUME} does not exist (see above)")
            continue
        try:
            spark.sql(q)
            print(f"PASS  {q}")
        except Exception as e:  # noqa: BLE001 -- report every grant, then decide
            # No MANAGE here is normal on a shared catalog; what matters is whether
            # the App ends up holding the privilege anyway
            covered = _held_by_everyone(privilege, [(kind, name)] + parents)
            if covered:
                print(f"OK    {q}\n      not granted ({type(e).__name__}), but {covered}: the App's "
                      "service principal inherits it (app_deploy()'s start-up checks confirm)")
            else:
                failed += 1
                print(f"FAIL  {q}\n      {type(e).__name__}: {str(e)[:200]}")
    if memo_wanted and not memo_missing:
        _warn_memo_exposure([memo, schema, cat])
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
              "(AI Search index, Can select; the memo: Unity Catalog volume "
              f"{APP_MEMO_VOLUME}, Can read and write).")


def _fingerprint(frame):
    import hashlib

    keys = frame.sort_values("catalog_index")["text_hash"].astype(str)
    return len(frame), hashlib.md5("\n".join(keys).encode()).hexdigest()


def _memo_problem():
    """Why the App's memo may fail at start-up, as far as this notebook can see; else None.

    Without its volume the App still serves (a memo write never fails a search),
    but every restart re-samples the LLM: the determinism the memo exists for is gone.
    A proxy only: it reads the LOCAL app.yaml (app_start() redeploys whatever was
    deployed last) and this user's view of the volume, not the App's grant.
    """
    if not _memo_wanted():
        return None
    if not os.path.isdir(vs_shim.memo_volume_path()):
        return (f"memo volume {APP_MEMO_VOLUME} missing or not readable here (app.yaml "
                f"LLM_MEMO_DIR={APP_MEMO_DIR}): run app_grant()")
    return None


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
    memo = _memo_problem()
    if memo:
        # Never a blocker (and never needs force=True, which would also wave the real
        # blockers through): the App serves without its memo, it only stops repeating
        print(f"WARNING: {memo}\n         the App would still serve, but re-ask the LLM after "
              "every restart (repeated searches can differ); its /logz shows "
              "'Start-up check memo FAILED' if so.")
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
                  f"{url}/logz should then show 'Start-up check index/embedding/llm/memo: OK' "
                  "(memo FAILED = no WRITE VOLUME on the memo volume: app_grant(), then "
                  "app_stop() + app_start()). The memo reads and writes the volume inside "
                  "each new search (Files API calls): if /search is slow, LLM_MEMO=\"false\" "
                  "in app.yaml is the revert.")
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


print(f"app {APP_NAME} | stage {APP_STAGE} | catalog file {APP_CATALOG} | memo {APP_MEMO_DIR}\n"
      "functions: export_catalog() stage() app_create() app_grant() app_deploy() "
      "app_share(who) app_status() app_stop() app_start() app_costs(since)")
