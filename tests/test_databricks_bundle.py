"""Offline tests for the Databricks App bundle and runbook (databricks/09_deploy_app.py,
00_preflight.py, vs_shim.py, app_bundle/) and the Databricks retrieval client
(app/vector_search.py).

No network, no workspace, no LLM. databricks-sdk is FAKED (it is not installed in the
local .venv, and a test must never reach a workspace anyway), Spark is a fake that
records its SQL, and vector_search's SDK client is a fake. Runs under pytest
(`python -m pytest tests/test_databricks_bundle.py`) or plainly
(`python tests/test_databricks_bundle.py`).

Pinned here (Oct 27 spec, unit A4; the parity checklist's T1-T6):
- T1 stage() ships every repo module the App's two processes can import (lazy imports
  too) and every data file they read; every third-party package they import is pinned
  in app_bundle/requirements.txt, except a short list of packages only off-App
  branches import, and those only lazily. APP_SOURCES globs ui*.py and the AST walk
  stages any other imported repo module (a synthetic tree checks both).
- T2 the bundle pins equal the local .venv versions (a package not installed is
  skipped) and every pin admits Python 3.11 (the App's interpreter).
- T3 _check_bundle() passes the real app.yaml with the notebook's Cell 2 values and
  refuses a drifted value, a missing or unreadable LLM_MEMO, and an LLM_MEMO_DIR that
  is not the memo volume app_grant() grants. A missing memo volume is a WARNING in
  _check_ready(), never a blocker (app_start() of a healthy App goes ahead).
- T4 app_create() / app_grant() against a fake SDK and Spark: the memo volume is
  created, APPENDED as a WRITE_VOLUME resource (the endpoint fallback keeps both
  volumes, whichever refusal comes first), dropped only on a refusal that names it
  or an SDK without WRITE_VOLUME, and an existing App gets READ + WRITE VOLUME by SQL
  with PASS / OK / FAIL lines like the other grants, plus a WARN when everyone on the
  account can read or write the memo. app.yaml LLM_MEMO="false" creates and grants
  nothing.
- T5 run_app's child env and argv (private uvicorn port, no access log, one worker).
- T6 vector_search: load_catalog from the App's export through a fake Files API,
  _search's reply parsing, embed_texts' ordering / normalising / refusals.
- vs_shim.patch_eval_harness(memo=False) turns the memo off and prints it; 00's new
  rows (pandas / numpy vs the App, the memo volume's dry write).
- Phase C (unit D, 2026-10-27): 08_llm_smoke run end to end against a cooperative fake
  model behind the real llm_client.generate_json -- 43 provider calls, every one with
  the memo OFF, the notebook's LLM_MEMO restored, a memo hit FAILs its row (a broken
  switch is caught) -- its gift / laptop translator rows and its device-gift rerank row
  (the window /search sends: search._annotate_types, main._with_role_fields, the gate,
  device_role.constraint's own text); 10_page_eval runs with the memo off and counts
  data/page_type_queries.json; 12_determinism builds its memo folder only below a
  scratch volume (never app_state), passes --allow-volume only for it, runs the real
  check_determinism on a fake app.main, restores the memo knobs, writes snapshots and
  compares a restart with them.
"""

import contextlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Where the notebooks import vs_shim from (their Cell 2 + sys.path.insert(0, "databricks"))
if str(ROOT / "databricks") not in sys.path:
    sys.path.insert(0, str(ROOT / "databricks"))

import app.config as cfg  # noqa: E402
import vs_shim  # noqa: E402
from app import vector_search as VS  # noqa: E402

NB09 = ROOT / "databricks" / "09_deploy_app.py"
NB00 = ROOT / "databricks" / "00_preflight.py"
BUNDLE = ROOT / "databricks" / "app_bundle"

# The values Cell 2 (databricks/README.md) and app.yaml agree on
# (Unity Gateway since 2026-11-13: model services on the gateway route; the vectors keep
# their databricks-gte-large-en label)
CELL2 = {"CAS_CATALOG": "dev", "CAS_SCHEMA": "cas", "VS_INDEX": "dev.cas.products_index",
         "VS_TABLE": "dev.cas.products", "VS_EMBEDDING_MODEL": "databricks-gte-large-en",
         "VS_EMBEDDING_SERVICE": "system.ai.gte-large-en",
         "OPENAI_MODEL": "system.ai.qwen3-next-80b-a3b-instruct",
         "OPENAI_BASE_URL": "https://adb-1.azuredatabricks.net/ai-gateway/mlflow/v1",
         "APP_NAME": "cas-app"}
# The pay-per-token shape before it: the documented revert, which must still ship
LEGACY = {"VS_EMBEDDING_SERVICE": "", "OPENAI_MODEL": "databricks-qwen3-next-80b-a3b-instruct",
          "OPENAI_BASE_URL": "https://adb-1.azuredatabricks.net/serving-endpoints"}
MEMO_VOLUME = "dev.cas.app_state"
MEMO_DIR = "/Volumes/dev/cas/app_state/llm_memo"
SP = "sp-0000"


class _Skipped(Exception):
    """Raised by _skip() under the plain runner (it reports it), pytest or not."""


_PLAIN_RUNNER = False  # set by __main__: pytest.skip() there would abort the whole run


def _skip(reason):
    if _PLAIN_RUNNER:
        raise _Skipped(reason)
    try:
        import pytest
    except ImportError:
        raise _Skipped(reason)
    pytest.skip(reason)


# ------------------------------------------------------------------ small helpers
_MISSING = object()


@contextlib.contextmanager
def _patched(obj, **values):
    """Set attributes for the block, then restore them (delete the ones that were absent)."""
    old = {k: getattr(obj, k, _MISSING) for k in values}
    for k, v in values.items():
        setattr(obj, k, v)
    try:
        yield obj
    finally:
        for k, v in old.items():
            if v is _MISSING:
                with contextlib.suppress(AttributeError):
                    delattr(obj, k)
            else:
                setattr(obj, k, v)


@contextlib.contextmanager
def _env(**values):
    """os.environ for the block; None removes a key."""
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _quiet():
    """Capture what a notebook function prints (the test reads it)."""
    return contextlib.redirect_stdout(io.StringIO())


# ------------------------------------------------------------------ fake databricks-sdk
class _Rec:
    """A dataclass-like record: keyword fields, equality by fields."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"


class _NotFound(Exception):
    pass


class _PermissionDenied(Exception):
    pass


def _fake_sdk_modules(write_volume=True, model_service=True):
    """sys.modules entries for the databricks.sdk names 09 and 00 import.

    model_service=False is a databricks-sdk older than 0.151.0 (no MODEL_SERVICE)."""
    apps = types.ModuleType("databricks.sdk.service.apps")
    for name in ("App", "AppResource", "AppResourceServingEndpoint", "AppResourceUcSecurable",
                 "AppDeployment", "AppAccessControlRequest"):
        setattr(apps, name, type(name, (_Rec,), {}))
    apps.ComputeSize = lambda v: ("ComputeSize", v)
    apps.AppResourceServingEndpointServingEndpointPermission = SimpleNamespace(CAN_QUERY="CAN_QUERY")
    kinds, perms = {"VOLUME": "VOLUME"}, {"READ_VOLUME": "READ_VOLUME"}
    if model_service:
        kinds["MODEL_SERVICE"], perms["EXECUTE"] = "MODEL_SERVICE", "EXECUTE"
    apps.AppResourceUcSecurableUcSecurableType = SimpleNamespace(**kinds)
    if write_volume:
        perms["WRITE_VOLUME"] = "WRITE_VOLUME"
    apps.AppResourceUcSecurableUcSecurablePermission = SimpleNamespace(**perms)
    workspace = types.ModuleType("databricks.sdk.service.workspace")
    workspace.WorkspaceObjectAccessControlRequest = type(
        "WorkspaceObjectAccessControlRequest", (_Rec,), {})
    workspace.WorkspaceObjectPermissionLevel = SimpleNamespace(CAN_READ="CAN_READ")
    workspace.ImportFormat = SimpleNamespace(AUTO="AUTO")
    service = types.ModuleType("databricks.sdk.service")
    service.apps, service.workspace = apps, workspace
    errors = types.ModuleType("databricks.sdk.errors")
    errors.NotFound, errors.PermissionDenied = _NotFound, _PermissionDenied
    version = types.ModuleType("databricks.sdk.version")
    version.__version__ = "0.151.0"
    core = types.ModuleType("databricks.sdk.core")
    core.Config = lambda **kw: SimpleNamespace(**kw)
    sdk = types.ModuleType("databricks.sdk")
    sdk.service, sdk.errors, sdk.version, sdk.core = service, errors, version, core
    sdk.WorkspaceClient = lambda **kw: SimpleNamespace(config=kw.get("config"))
    return {"databricks.sdk": sdk, "databricks.sdk.service": service,
            "databricks.sdk.service.apps": apps, "databricks.sdk.service.workspace": workspace,
            "databricks.sdk.errors": errors, "databricks.sdk.version": version,
            "databricks.sdk.core": core}


@contextlib.contextmanager
def _fake_sdk(write_volume=True, model_service=True):
    mods = _fake_sdk_modules(write_volume, model_service)
    old = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        yield mods
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _Row:
    def __init__(self, d):
        self._d = d

    def asDict(self):  # noqa: N802 -- pyspark's name
        return dict(self._d)


class FakeSpark:
    """spark.sql that records each statement; `fail` = regexes of statements that raise."""

    def __init__(self, fail=(), volumes=(), grants=None):
        self.log = []
        self.fail = [re.compile(p) for p in fail]
        self.volumes = set(volumes)
        self.grants = grants or {}

    def sql(self, q, args=None):
        self.log.append(q)
        if q.startswith("DESCRIBE VOLUME "):
            if q.split()[-1] not in self.volumes:
                raise RuntimeError("[VOLUME_NOT_FOUND]")
            return SimpleNamespace(collect=lambda: [])
        if q.startswith("SHOW GRANTS ON "):
            rows = [_Row(r) for r in self.grants.get(q[len("SHOW GRANTS ON "):], [])]
            return SimpleNamespace(collect=lambda: rows)
        for p in self.fail:
            if p.search(q):
                raise RuntimeError("PERMISSION_DENIED: User does not have MANAGE")
        if q.startswith("CREATE VOLUME IF NOT EXISTS "):
            self.volumes.add(q.split()[5])
        return SimpleNamespace(collect=lambda: [])


class FakeWorkspace:
    """The slice of WorkspaceClient app_create() / app_grant() touch."""

    def __init__(self, existing=False, create_errors=()):
        self.created = []
        self.create_errors = list(create_errors)
        self.app = self._app([]) if existing else None
        self.permissions = []
        self.apps = SimpleNamespace(get=self._get, create=self._create)
        self.current_user = SimpleNamespace(me=lambda: SimpleNamespace(user_name="me@example.com"))
        self.workspace = SimpleNamespace(
            get_status=lambda path: SimpleNamespace(object_id=42),
            update_permissions=lambda kind, oid, access_control_list: self.permissions.append(
                (kind, oid, access_control_list)))

    @staticmethod
    def _app(resources):
        return SimpleNamespace(url="https://cas-app.example", creator="me@example.com",
                               service_principal_client_id=SP, compute_status=None,
                               app_status=None, resources=resources)

    def _get(self, name):
        if self.app is None:
            raise _NotFound(name)
        return self.app

    def _create(self, app, no_compute):
        assert no_compute is True  # create() must never start (bill) the compute
        self.created.append(app)
        if self.create_errors:
            raise self.create_errors.pop(0)
        self.app = self._app(app.resources)


@contextlib.contextmanager
def _cell2():
    """cfg as Cell 2 sets it (the local .env may name another model or catalog)."""
    with _patched(cfg, **CELL2):
        yield


def _load09(spark=None):
    """Exec 09 the way a notebook does (repo root cwd), with no workspace call.

    CAS_APP_STAGE skips the one module-level SDK call (the user's home folder).
    """
    ns = {"__name__": "nb09", "spark": spark or FakeSpark()}
    src = NB09.read_text(encoding="utf-8")
    with _chdir(ROOT), _env(CAS_APP_STAGE="/Workspace/Users/me@example.com/cas-app"), _quiet():
        exec(compile(src, str(NB09), "exec"), ns)  # noqa: S102 -- our own notebook
    return ns


def _nb09_with_fakes(spark, ws, memo_dir):
    """09 wired to a fake workspace; the memo dir is a temp folder, never /Volumes."""
    ns = _load09(spark)
    ns["_ws"] = lambda: ws
    ns["APP_MEMO_DIR"] = str(memo_dir)
    return ns


# ------------------------------------------------------------------ T1: what stage() ships
# Packages only off-App branches import (Gemini, Anthropic, the local embedder, MLflow,
# Spark in a notebook). The App never runs those branches, so they stay unpinned --
# but only LAZILY: a top-level import of one would crash the App at start-up.
_OFF_APP = {"anthropic", "google", "sentence_transformers", "mlflow", "pyspark", "torch",
            "transformers"}
# import name -> distribution name, where they differ
_DIST = {"dotenv": "python-dotenv", "sklearn": "scikit-learn", "databricks": "databricks-sdk",
         "yaml": "pyyaml", "google": "google-genai"}
# Data files the App never reads: the local catalog (the App reads VS_CATALOG_CSV from
# its Volume) and the eval query sets
_LOCAL_ONLY_DATA = {"data/products_enriched.csv", "data/products.csv", "data/eval_queries.json"}


def _resolve(mod):
    path = mod.replace(".", "/")
    if not path or path.split("/")[0] == "databricks":
        return None
    for cand in (f"{path}.py", f"{path}/__init__.py"):
        if (ROOT / cand).is_file():
            return cand
    return None


def _is_repo_top(top):
    return (ROOT / f"{top}.py").is_file() or (ROOT / top / "__init__.py").is_file()


def _lazy_nodes(tree):
    """Import nodes inside a function or lambda (they run only when it is called)."""
    import ast

    lazy = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for n in ast.walk(fn):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    lazy.add(id(n))
    return lazy


def _app_graph():
    """({repo file: eagerly imported?}, {third-party top name: [(where, eager?)]}, data files).

    Independent of 09's own walker. Roots: uvicorn's app.main, Streamlit's ui.py and
    every page (Streamlit runs them as scripts), run_app.py. A module is EAGER when a
    chain of module-level imports reaches it; anything only a function imports is lazy.
    """
    import ast

    roots = ["app/main.py", "ui.py", "databricks/app_bundle/run_app.py"] + sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "pages").glob("*.py"))
    files, third, data = {}, {}, set()
    queue = [(r, True) for r in roots]
    while queue:
        path, eager = queue.pop()
        if path in files and (files[path] or not eager):
            continue
        files[path] = eager
        parts = path.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            init = "/".join(parts[:i]) + "/__init__.py"
            if (ROOT / init).is_file():
                queue.append((init, eager))
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"), filename=path)
        lazy = _lazy_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(
                    r"data/[\w.-]+\.(json|csv|jsonl|parquet|txt)", node.value):
                data.add(node.value)
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                    and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                    and getattr(node.left, "id", getattr(node.left, "attr", None)) == "DATA_DIR"):
                data.add(f"data/{node.right.value}")
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                mods = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            child_eager = eager and id(node) not in lazy
            for mod in mods:
                hit = _resolve(mod)
                if hit:
                    queue.append((hit, child_eager))
                    continue
                top = mod.split(".")[0]
                if not top or top in sys.stdlib_module_names or (
                        top != "databricks" and _is_repo_top(top)):
                    continue
                third.setdefault(top, []).append((f"{path}:{node.lineno}", child_eager))
    return files, third, data


def _bundle_pins():
    return vs_shim.bundle_pins(str(BUNDLE / "requirements.txt"))


def _staged():
    with _cell2():
        ns = _load09()
    with _chdir(ROOT), _quiet():
        pairs = ns["_app_sources"]()
    return {rel.replace(os.sep, "/") for _, rel in pairs}


def test_stage_covers_runtime_imports():
    staged = _staged()
    files, _, _ = _app_graph()
    missing = sorted(p for p in files
                     if not p.startswith("databricks/app_bundle/") and p not in staged)
    assert not missing, f"the App imports these but stage() does not ship them: {missing}"
    assert {"app.yaml", "run_app.py", "requirements.txt"} <= staged
    # The App's two processes, by name
    assert {"app/main.py", "ui.py", "ui_common.py", "ui_cart.py", "app/vector_search.py"} <= staged
    assert not any(p.startswith("databricks/") for p in staged), "databricks/ must never ship"
    assert not any(p.endswith(".pyc") or "__pycache__" in p for p in staged)


def test_stage_ships_every_data_file_the_app_reads():
    staged = _staged()
    _, _, data = _app_graph()
    assert "data/sponsored.json" in data  # the scan works: cfg.SPONSORED_CONFIG
    missing = sorted(data - _LOCAL_ONLY_DATA - staged)
    assert not missing, f"read at runtime but not staged: {missing}"


def test_every_third_party_import_is_pinned_or_lazy_and_off_app():
    pins = _bundle_pins()
    _, third, _ = _app_graph()
    unpinned, eager_off_app = [], []
    for top, uses in sorted(third.items()):
        dist = _DIST.get(top, top.lower().replace("_", "-"))
        if dist in pins:
            continue
        if top not in _OFF_APP:
            unpinned.append(f"{top} ({uses[0][0]})")
        eager_off_app += [f"{top} at {where}" for where, eager in uses if eager]
    assert not unpinned, f"imported by the App, not pinned in app_bundle/requirements.txt: {unpinned}"
    assert not eager_off_app, f"an off-App package imported at start-up: {eager_off_app}"
    # run_app launches both servers as `python -m <module>`
    assert {"uvicorn", "streamlit", "databricks-sdk", "openai", "pydantic"} <= set(pins)
    # The walk saw the lazy SDK and LLM imports (else this test proves nothing)
    assert {"databricks", "openai", "fastapi", "pandas", "streamlit"} <= set(third)


def _write(root, rel, text=""):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _synthetic_repo(tmp):
    _write(tmp, "app/__init__.py")
    _write(tmp, "app/main.py", "import helper_top\nfrom pkg import mod\nimport databricks.foo\n"
                               "from databricks.sdk import WorkspaceClient\n"
                               "def later():\n    import lazy_top\n")
    _write(tmp, "app/__pycache__/main.cpython-313.pyc", "x")
    # A dotted import names only the leaf module: its two parent __init__.py files reach
    # the stage through the walker's parent loop alone (Python runs them first)
    _write(tmp, "pages/p.py", "import ui_common\nimport pkg2.sub.leaf\n")
    for f in ("ui.py", "ui_common.py", "ui_new.py", "helper_top.py", "lazy_top.py", "unused.py",
              "pkg/__init__.py", "pkg/inner.py", "databricks/foo.py",
              "pkg2/__init__.py", "pkg2/sub/__init__.py", "pkg2/sub/leaf.py"):
        _write(tmp, f)
    _write(tmp, "pkg/mod.py", "from .inner import x\n")
    _write(tmp, "data/sponsored.json", "{}")
    for f in ("app.yaml", "run_app.py", "requirements.txt"):
        _write(tmp, f"databricks/app_bundle/{f}")


def test_app_sources_globs_ui_and_stages_imported_modules():
    with _cell2():
        ns = _load09()
    with tempfile.TemporaryDirectory() as tmp:
        _synthetic_repo(tmp)
        out = io.StringIO()
        with _chdir(tmp), contextlib.redirect_stdout(out):
            pairs = ns["_app_sources"]()
        rels = sorted(rel.replace(os.sep, "/") for _, rel in pairs)
        assert rels == sorted([
            "app/__init__.py", "app/main.py", "pages/p.py", "ui.py", "ui_common.py",
            "ui_new.py",                                   # the glob: no edit needed
            "data/sponsored.json", "helper_top.py", "lazy_top.py",  # imported -> staged
            "pkg/__init__.py", "pkg/mod.py", "pkg/inner.py",        # incl. a relative import
            "pkg2/__init__.py", "pkg2/sub/__init__.py", "pkg2/sub/leaf.py",  # parent packages
            "app.yaml", "run_app.py", "requirements.txt"]), rels
        assert "unused.py" not in rels and "databricks/foo.py" not in rels
        assert "also staging 8 module(s)" in out.getvalue(), out.getvalue()


def test_app_sources_refuses_a_missing_glob_and_a_file_over_10_mb():
    with _cell2():
        ns = _load09()
    with tempfile.TemporaryDirectory() as tmp:
        _synthetic_repo(tmp)
        with open(Path(tmp) / "data" / "sponsored.json", "wb") as f:
            f.truncate(10 * 1024 * 1024 + 1)
        with _chdir(tmp), _quiet():
            try:
                ns["_app_sources"]()
                raise AssertionError("an 11 MB file was accepted")
            except RuntimeError as e:
                assert "over 10 MB" in str(e)
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "app/__init__.py")
        with _chdir(tmp), _quiet():
            try:
                ns["_app_sources"]()
                raise AssertionError("no ui*.py was accepted")
            except RuntimeError as e:
                assert "ui*.py" in str(e) and "repo root" in str(e)


# ------------------------------------------------------------------ T2: pins
def test_bundle_pins_equal_the_local_venv_and_admit_python_311():
    import importlib.metadata as md

    pins = _bundle_pins()
    assert {"pydantic", "starlette", "httpx"} <= set(pins), "transitive pins (parity 1.8)"
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        SpecifierSet = None
    checked, absent, wrong, too_new = 0, [], [], []
    for dist, want in sorted(pins.items()):
        try:
            d = md.distribution(dist)
        except md.PackageNotFoundError:
            absent.append(dist)  # e.g. databricks-sdk: notebook/App only
            continue
        checked += 1
        if d.version != want:
            wrong.append(f"{dist} pinned {want}, installed {d.version}")
        rp = d.metadata.get("Requires-Python")
        if SpecifierSet and rp and not SpecifierSet(rp).contains("3.11.0"):
            too_new.append(f"{dist} {d.version} requires Python {rp}")
    assert not wrong, f"the App would run other versions than the laptop: {wrong}"
    assert not too_new, f"the App's Python 3.11 cannot install: {too_new}"
    assert checked >= 10, (checked, absent)


def test_bundle_pins_parser():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write(tmp, "r.txt", "# c\nuvicorn[standard]==0.48.0\nscikit_learn == 1.9.0  # x\n"
                                  "\nopenai>=2\nPython.Dotenv==1.2.2\n")
        assert vs_shim.bundle_pins(str(p)) == {
            "uvicorn": "0.48.0", "scikit-learn": "1.9.0", "python-dotenv": "1.2.2"}


# ------------------------------------------------------------------ T3: app.yaml
def _yaml_with(tmp, edit):
    text = (BUNDLE / "app.yaml").read_text(encoding="utf-8")
    new = edit(text)
    assert new != text, "the edit did not apply"
    _write(tmp, "app.yaml", new)


def _check_bundle_raises(edit, *needles):
    with _cell2():
        ns = _load09()
        with tempfile.TemporaryDirectory() as tmp:
            _yaml_with(tmp, edit)
            ns["APP_BUNDLE"] = tmp
            try:
                with _quiet():
                    ns["_check_bundle"]()
            except RuntimeError as e:
                for n in needles:
                    assert n in str(e), (n, str(e))
                return
    raise AssertionError(f"_check_bundle accepted the edited app.yaml {needles}")


def _set_value(name, value):
    return lambda t: re.sub(rf'(name: {name}\n\s*value: )"[^"]*"', rf'\1"{value}"', t)


def test_check_bundle_passes_the_real_app_yaml():
    with _cell2():
        ns = _load09()
        with _chdir(ROOT), _quiet() as out:
            ns["_check_bundle"]()
            env = ns["_bundle_env"]()
    assert "NOTE" not in out.getvalue()
    assert env["LLM_MEMO"] == "true"
    assert env["LLM_MEMO_DIR"] == MEMO_DIR == ns["APP_MEMO_DIR"]
    assert ns["APP_MEMO_VOLUME"] == MEMO_VOLUME
    # The memo dir sits inside the very volume app_grant() grants WRITE VOLUME on
    assert env["LLM_MEMO_DIR"].startswith("/Volumes/" + MEMO_VOLUME.replace(".", "/") + "/")
    assert env["VS_CATALOG_CSV"] == ns["APP_CATALOG"]
    assert "/raw/" in env["VS_CATALOG_CSV"] and "app_state" not in env["VS_CATALOG_CSV"]


def test_check_bundle_refuses_drift():
    _check_bundle_raises(_set_value("VS_INDEX", "dev.cas.other_index"), "VS_INDEX")
    _check_bundle_raises(_set_value("OPENAI_MODEL", "databricks-gpt-oss-120b"), "OPENAI_MODEL")
    # The route as well as the model: an App on another route than the notebook tested
    _check_bundle_raises(_set_value("VS_EMBEDDING_SERVICE", ""), "VS_EMBEDDING_SERVICE")
    _check_bundle_raises(_set_value("OPENAI_BASE_PATH", "/serving-endpoints"), "OPENAI_BASE_PATH")
    _check_bundle_raises(lambda t: re.sub(r'  - name: OPENAI_BASE_PATH\n\s*value: "[^"]*"\n', "", t),
                         "OPENAI_BASE_PATH")


def test_check_bundle_accepts_the_legacy_route_when_the_notebook_runs_it():
    """The revert: pay-per-token names, no service, no base path (or /serving-endpoints)."""
    def legacy(t):
        t = re.sub(r'  - name: (VS_EMBEDDING_SERVICE|OPENAI_BASE_PATH)\n\s*value: "[^"]*".*\n', "", t)
        return _set_value("OPENAI_MODEL", LEGACY["OPENAI_MODEL"])(t)

    for edit in (legacy, lambda t: _set_value("OPENAI_BASE_PATH", "serving-endpoints/")(
            _set_value("VS_EMBEDDING_SERVICE", "")(
                _set_value("OPENAI_MODEL", LEGACY["OPENAI_MODEL"])(t)))):
        with _cell2(), _patched(cfg, **LEGACY):
            ns = _load09()
            with tempfile.TemporaryDirectory() as tmp:
                _yaml_with(tmp, edit)
                ns["APP_BUNDLE"] = tmp
                with _quiet():
                    ns["_check_bundle"]()  # raises on any disagreement


def test_check_bundle_refuses_a_wrong_or_missing_memo_dir():
    # The App's local disk (lost on restart), the catalog's read-only volume, another schema
    for bad in (".cache/llm_memo", "/Volumes/dev/cas/raw/llm_memo",
                "/Volumes/dev/other/app_state/llm_memo"):
        _check_bundle_raises(_set_value("LLM_MEMO_DIR", bad), "LLM_MEMO_DIR", bad)
    _check_bundle_raises(lambda t: re.sub(r'  - name: LLM_MEMO_DIR\n\s*value: "[^"]*"\n', "", t),
                         "LLM_MEMO_DIR")
    # valueFrom (a resource key) is legal app.yaml but invisible to the check: refused
    _check_bundle_raises(lambda t: re.sub(r'(name: LLM_MEMO_DIR\n\s*)value: "[^"]*"',
                                          r"\1valueFrom: memo-volume", t), "LLM_MEMO_DIR", "valueFrom")


def test_check_bundle_refuses_a_missing_or_unreadable_memo_switch():
    _check_bundle_raises(lambda t: re.sub(r'  - name: LLM_MEMO\n\s*value: "[^"]*"\n', "", t),
                         "LLM_MEMO")
    _check_bundle_raises(_set_value("LLM_MEMO", "maybe"), "LLM_MEMO", "maybe")
    # "on" reads as a yes here, but app/config.py would read it as OFF: refused
    _check_bundle_raises(_set_value("LLM_MEMO", "on"), "LLM_MEMO", "on")


def test_memo_switch_words_match_the_config_parser():
    """09's yes/no words for LLM_MEMO agree with what app/config.py actually turns on."""
    with _cell2():
        ns = _load09()
    path = ROOT / "app" / "config.py"
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    words = sorted(ns["_TRUE"] | ns["_FALSE"] | {"on", "off", "enabled", "TRUE", " yes ", "true "})
    for word in words:
        # A throwaway copy of config (its own namespace): app.config itself is untouched
        probe = {"__name__": "config_probe", "__file__": str(path)}
        with _env(LLM_MEMO=word):
            exec(code, probe)  # noqa: S102 -- our own config
        read = ns["_memo_on"]({"LLM_MEMO": word})
        assert read is None or read == probe["LLM_MEMO"], (word, read, probe["LLM_MEMO"])
    assert all(ns["_memo_on"]({"LLM_MEMO": w}) for w in ns["_TRUE"])  # the yes words ARE on


def test_check_bundle_accepts_memo_off_with_a_note():
    with _cell2():
        ns = _load09()
        with tempfile.TemporaryDirectory() as tmp:
            _yaml_with(tmp, _set_value("LLM_MEMO", "false"))
            ns["APP_BUNDLE"] = tmp
            with _quiet() as out:
                ns["_check_bundle"]()
    assert "LLM_MEMO=false" in out.getvalue()


def test_memo_problem_names_a_missing_volume():
    """_check_ready() prints this as a WARNING (next two tests), never a blocker."""
    with _cell2():
        ns = _load09()
    with tempfile.TemporaryDirectory() as tmp, _chdir(ROOT):
        vol = Path(tmp) / "app_state"
        ns["vs_shim"] = SimpleNamespace(memo_volume_path=lambda: str(vol))
        problem = ns["_memo_problem"]()
        assert problem and "app_grant()" in problem and MEMO_VOLUME in problem
        vol.mkdir()
        assert ns["_memo_problem"]() is None
        vol.rmdir()
        ns["_bundle_env"] = lambda: {"LLM_MEMO": "false"}
        assert ns["_memo_problem"]() is None  # memo off: nothing to check


def _ready_ns(catalog_present):
    """09 with a healthy index and stage; the export present (and current) or absent."""
    ns = _load09()
    ws = FakeWorkspace(existing=True)
    ws.workspace.list = lambda path: [SimpleNamespace(path=f"{path}/.cas_app_stage")]
    ns["_ws"] = lambda: ws
    ns["spark"].table = lambda name: SimpleNamespace(count=lambda: 3)
    ns["vector_search"] = SimpleNamespace(index_health=lambda n: {"ok": True, "detail": ""})
    if catalog_present:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        tmp.write("catalog_index,text_hash\n0,a\n")
        tmp.close()
        ns["APP_CATALOG"] = tmp.name
        ns["_fingerprint"] = lambda frame: (1, "same")  # the export IS the table
    else:
        ns["APP_CATALOG"] = "/Volumes/dev/cas/raw/app_catalog_cas_test.csv"  # absent here
    return ns


def test_check_ready_warns_but_never_blocks_on_the_memo_volume():
    """A missing memo volume costs repeatability, not the App: a WARNING, never a blocker,
    so force=True (which also waves the real blockers through) is never needed for it."""
    memo = "memo volume dev.cas.app_state missing: run app_grant()"
    with _fake_sdk(), _cell2():
        ns = _ready_ns(catalog_present=True)
        ns["spark"].table = lambda name: SimpleNamespace(
            count=lambda: 1, select=lambda *c: SimpleNamespace(toPandas=lambda: None))
        ns["_memo_problem"] = lambda: memo
        try:
            with _quiet() as out:
                ns["_check_ready"]()  # everything else healthy: starts, with a warning
        finally:
            os.remove(ns["APP_CATALOG"])
        assert f"WARNING: {memo}" in out.getvalue() and "still serve" in out.getvalue()
        ns = _ready_ns(catalog_present=False)
        ns["_memo_problem"] = lambda: memo
        try:
            with _quiet() as out:
                ns["_check_ready"]()
            raise AssertionError("started without the catalog export")
        except RuntimeError as e:
            assert "export_catalog()" in str(e) and "memo" not in str(e)  # a real blocker
        assert f"WARNING: {memo}" in out.getvalue()


def test_app_start_of_a_healthy_app_is_not_refused_over_the_memo():
    """The reviewer's case: an App deployed before the memo, the volume not visible here."""
    started = []
    with _fake_sdk(), _cell2():
        ns = _ready_ns(catalog_present=True)
        ns["spark"].table = lambda name: SimpleNamespace(
            count=lambda: 1, select=lambda *c: SimpleNamespace(toPandas=lambda: None))
        ws = ns["_ws"]()
        ws.apps.start_and_wait = lambda *a, **k: started.append(a)
        for name in ("_settle", "_watch", "_grant_stage_read"):
            ns[name] = lambda *a, **k: None
        ns["vs_shim"] = SimpleNamespace(memo_volume_path=lambda: "/Volumes/no/such/app_state")
        try:
            with _quiet() as out, _chdir(ROOT):
                ns["app_start"]()
        finally:
            os.remove(ns["APP_CATALOG"])
    assert started and "WARNING: memo volume dev.cas.app_state missing" in out.getvalue()


# ------------------------------------------------------------------ T4: create / grant
def _resources(app):
    return [(r.name, getattr(getattr(r, "uc_securable", None), "securable_full_name", None),
             getattr(getattr(r, "uc_securable", None), "permission", None)) for r in app.resources]


def _create(spark, ws, write_volume=True, bundle_env=None, model_service=True, **cfg_values):
    with _fake_sdk(write_volume, model_service), _cell2(), _patched(cfg, **cfg_values),             tempfile.TemporaryDirectory() as tmp:
        ns = _nb09_with_fakes(spark, ws, Path(tmp) / "llm_memo")
        if bundle_env is not None:  # app.yaml's env as the notebook would read it
            ns["_bundle_env"] = lambda: dict(bundle_env)
        with _quiet() as out:
            ns["app_create"]()
        made = (Path(tmp) / "llm_memo").is_dir()
    return out.getvalue(), made


def test_app_create_appends_the_memo_volume():
    spark, ws = FakeSpark(), FakeWorkspace()
    out, made = _create(spark, ws)
    assert len(ws.created) == 1
    assert _resources(ws.created[0]) == [
        ("llm-model", "system.ai.qwen3-next-80b-a3b-instruct", "EXECUTE"),
        ("embedding-model", "system.ai.gte-large-en", "EXECUTE"),
        ("catalog-volume", "dev.cas.raw", "READ_VOLUME"),
        ("memo-volume", MEMO_VOLUME, "WRITE_VOLUME")]
    # The volume exists before the resource that names it, and its folder is made
    assert spark.log[0].startswith(f"CREATE VOLUME IF NOT EXISTS {MEMO_VOLUME} ")
    assert made and f"PASS  CREATE VOLUME IF NOT EXISTS {MEMO_VOLUME}" in out


def test_app_create_legacy_names_get_serving_endpoint_resources():
    """The revert: databricks-* endpoints are attached as serving endpoints, Can query."""
    ws = FakeWorkspace()
    _create(FakeSpark(), ws, **LEGACY)
    got = ws.created[0].resources
    assert [(r.name, r.serving_endpoint.name, r.serving_endpoint.permission) for r in got[:2]] == [
        ("llm-endpoint", "databricks-qwen3-next-80b-a3b-instruct", "CAN_QUERY"),
        ("embedding-endpoint", "databricks-gte-large-en", "CAN_QUERY")]
    assert [r.name for r in got[2:]] == ["catalog-volume", "memo-volume"]


def test_app_create_without_model_service_attaches_no_model():
    # A databricks-sdk older than 0.151.0: no MODEL_SERVICE, so no model resource (the App
    # then relies on EXECUTE its principal holds), and both volumes still ship
    ws = FakeWorkspace()
    out, _ = _create(FakeSpark(), ws, model_service=False)
    assert [r[0] for r in _resources(ws.created[0])] == ["catalog-volume", "memo-volume"]
    assert "no MODEL_SERVICE app resource" in out


def test_app_create_model_service_refusal_falls_back_to_the_volumes():
    ws = FakeWorkspace(create_errors=[RuntimeError(
        "PERMISSION_DENIED: cannot add model service system.ai.gte-large-en")])
    out, _ = _create(FakeSpark(), ws)
    assert [[r[0] for r in _resources(a)] for a in ws.created] == [
        ["llm-model", "embedding-model", "catalog-volume", "memo-volume"],
        ["catalog-volume", "memo-volume"]]
    assert "Model > Unity Gateway, Can execute" in out


def test_app_create_endpoint_fallback_keeps_both_volumes():
    ws = FakeWorkspace(create_errors=[_PermissionDenied("no CAN MANAGE on serving endpoint")])
    out, _ = _create(FakeSpark(), ws)
    assert len(ws.created) == 2
    assert [r[0] for r in _resources(ws.created[1])] == ["catalog-volume", "memo-volume"]
    assert "retrying with the Volumes only" in out


def test_app_create_drops_a_refused_memo_volume_resource():
    ws = FakeWorkspace(create_errors=[
        _PermissionDenied("no CAN MANAGE on serving endpoint"),
        RuntimeError(f"PERMISSION_DENIED: User does not have MANAGE on Volume '{MEMO_VOLUME}'")])
    out, _ = _create(FakeSpark(), ws)
    assert [[r[0] for r in _resources(a)] for a in ws.created] == [
        ["llm-model", "embedding-model", "catalog-volume", "memo-volume"],
        ["catalog-volume", "memo-volume"], ["catalog-volume"]]
    assert "app_grant() grants READ + WRITE VOLUME" in out


def _memo_refusal():
    return RuntimeError(f"PERMISSION_DENIED: User does not have MANAGE on Volume '{MEMO_VOLUME}'")


def test_app_create_memo_refusal_then_endpoint_refusal():
    """The reverse order: the server's order is undocumented, and either must end in the
    Volumes-only create (the endpoint fallback used to be lost after a memo refusal)."""
    ws = FakeWorkspace(create_errors=[
        _memo_refusal(), _PermissionDenied("no CAN MANAGE on serving endpoint")])
    out, _ = _create(FakeSpark(), ws)
    assert [[r[0] for r in _resources(a)] for a in ws.created] == [
        ["llm-model", "embedding-model", "catalog-volume", "memo-volume"],
        ["llm-model", "embedding-model", "catalog-volume"], ["catalog-volume"]]
    assert "memo volume resource refused" in out and "retrying with the Volumes only" in out
    # Each fallback fires once: a third refusal is raised, not retried for ever
    ws = FakeWorkspace(create_errors=[
        _memo_refusal(), _PermissionDenied("no CAN MANAGE on serving endpoint"), _memo_refusal()])
    try:
        _create(FakeSpark(), ws)
        raise AssertionError("a repeated refusal was retried")
    except RuntimeError as e:
        assert MEMO_VOLUME in str(e)
    assert len(ws.created) == 3


def test_memo_volume_refusal_names_the_memo_volume_only():
    """Only an error that names the memo volume (and not the catalog one) drops it."""
    with _fake_sdk(), _cell2():
        ns = _load09()
    is_memo = ns["_is_memo_volume_refusal"]
    for msg, want in [
            (f"PERMISSION_DENIED: User does not have MANAGE on Volume '{MEMO_VOLUME}'", True),
            ("PERMISSION_DENIED: User does not have MANAGE on Volume '`dev`.`cas`.`app_state`'.",
             True),
            ("NOT_FOUND: Volume 'dev.cas.app_state' does not exist", True),
            # The catalog volume, however UC quotes it: raised, never a memo diagnosis
            ("PERMISSION_DENIED: User does not have MANAGE on Volume '`dev`.`cas`.`raw`'.", False),
            ("PERMISSION_DENIED: User does not have MANAGE on volume 'raw'", False),
            ("PERMISSION_DENIED: no MANAGE on dev.cas.raw or dev.cas.app_state", False),
            ("PERMISSION_DENIED: User does not have MANAGE on the volume", False),  # names neither
            ("no CAN MANAGE on serving endpoint", False),
            ("Volume dev.cas.app_statex is gone", False)]:  # a longer name is another volume
        assert is_memo(RuntimeError(msg)) is want, (msg, want)
    err = "PERMISSION_DENIED: User does not have MANAGE on Volume '`dev`.`cas`.`raw`'."
    ws = FakeWorkspace(create_errors=[RuntimeError(err), RuntimeError(err)])
    out = io.StringIO()
    with _fake_sdk(), _cell2(), tempfile.TemporaryDirectory() as tmp:
        ns = _nb09_with_fakes(FakeSpark(), ws, Path(tmp) / "llm_memo")
        try:
            with contextlib.redirect_stdout(out):
                ns["app_create"]()
            raise AssertionError("a refused catalog volume was retried")
        except RuntimeError as e:
            assert "`raw`" in str(e)
    # One create() call and no wrong diagnosis printed before the real error
    assert len(ws.created) == 1 and "memo volume resource refused" not in out.getvalue()
    assert f"PASS  CREATE VOLUME IF NOT EXISTS {MEMO_VOLUME}" in out.getvalue()  # captured


def test_app_create_with_the_memo_off_skips_the_volume():
    """app.yaml LLM_MEMO="false" (the documented revert): no volume, no WRITE resource."""
    spark, ws = FakeSpark(fail=[r"^CREATE VOLUME"]), FakeWorkspace()
    out, made = _create(spark, ws, bundle_env={"LLM_MEMO": "false"})
    assert [r[0] for r in _resources(ws.created[0])] == [
        "llm-model", "embedding-model", "catalog-volume"]
    assert not any("app_state" in q for q in spark.log) and not made
    assert "LLM_MEMO=false" in out and "FAIL" not in out


def test_memo_wanted_follows_app_yaml():
    with _cell2():
        ns = _load09()
    for env, want in [({"LLM_MEMO": "true"}, True), ({"LLM_MEMO": "false"}, False),
                      ({"LLM_MEMO": "0"}, False), ({}, True), ({"LLM_MEMO": "maybe"}, True)]:
        ns["_bundle_env"] = lambda env=env: dict(env)
        assert ns["_memo_wanted"]() is want, (env, want)

    def unreadable():
        raise FileNotFoundError("databricks/app_bundle/app.yaml")

    # Run from the wrong folder: grant (stage() refuses that app.yaml anyway)
    ns["_bundle_env"] = unreadable
    assert ns["_memo_wanted"]() is True


def test_app_create_raises_other_errors():
    ws = FakeWorkspace(create_errors=[RuntimeError("INVALID_PARAMETER_VALUE: bad compute size")])
    try:
        _create(FakeSpark(), ws)
        raise AssertionError("an unrelated create() error was swallowed")
    except RuntimeError as e:
        assert "bad compute size" in str(e)
    assert len(ws.created) == 1


def test_app_create_without_write_volume_or_volume_skips_the_resource():
    ws = FakeWorkspace()
    out, _ = _create(FakeSpark(), ws, write_volume=False)
    assert [r[0] for r in _resources(ws.created[0])][-1] == "catalog-volume"
    assert "no WRITE_VOLUME" in out
    # No CREATE VOLUME privilege and no volume: a resource on it would fail create()
    ws = FakeWorkspace()
    out, _ = _create(FakeSpark(fail=[r"^CREATE VOLUME"]), ws)
    assert [r[0] for r in _resources(ws.created[0])] == [
        "llm-model", "embedding-model", "catalog-volume"]
    assert f"FAIL  CREATE VOLUME IF NOT EXISTS {MEMO_VOLUME}" in out


def test_app_create_leaves_an_existing_app_to_app_grant():
    ws = FakeWorkspace(existing=True)
    spark = FakeSpark()
    out, _ = _create(spark, ws)
    assert ws.created == [] and spark.log == []
    assert "run app_grant()" in out


def _grant(spark, ws=None, bundle_env=None):
    ws = ws or FakeWorkspace(existing=True)
    with _fake_sdk(), _cell2(), tempfile.TemporaryDirectory() as tmp:
        ns = _nb09_with_fakes(spark, ws, Path(tmp) / "llm_memo")
        if bundle_env is not None:
            ns["_bundle_env"] = lambda: dict(bundle_env)
        with _quiet() as out:
            ns["app_grant"]()
    return out.getvalue()


def test_app_grant_creates_and_grants_the_memo_volume_for_an_existing_app():
    spark = FakeSpark()
    out = _grant(spark)
    grants = [q for q in spark.log if q.startswith("GRANT ")]
    assert grants == [
        "GRANT USE CATALOG ON CATALOG dev TO `sp-0000`",
        "GRANT USE SCHEMA ON SCHEMA dev.cas TO `sp-0000`",
        "GRANT READ VOLUME ON VOLUME dev.cas.raw TO `sp-0000`",
        "GRANT SELECT ON TABLE dev.cas.products_index TO `sp-0000`",
        f"GRANT READ VOLUME ON VOLUME {MEMO_VOLUME} TO `sp-0000`",
        f"GRANT WRITE VOLUME ON VOLUME {MEMO_VOLUME} TO `sp-0000`"]
    assert spark.log.index(next(q for q in spark.log if q.startswith("CREATE VOLUME"))) < \
        spark.log.index(grants[-1])
    assert out.count("PASS  ") == 8 and "FAIL" not in out  # volume + 6 grants + folder read
    # Who ELSE can read or write the memo is looked up on the clean path too: nobody here
    shown = [q for q in spark.log if q.startswith("SHOW GRANTS ON ")]
    assert {f"SHOW GRANTS ON VOLUME {MEMO_VOLUME}", "SHOW GRANTS ON SCHEMA dev.cas",
            "SHOW GRANTS ON CATALOG dev"} <= set(shown), shown
    assert "WARN" not in out


def test_app_grant_reports_ok_when_the_volume_exists_or_everyone_holds_the_grant():
    spark = FakeSpark(fail=[r"^CREATE VOLUME", rf"^GRANT WRITE VOLUME ON VOLUME {MEMO_VOLUME}"],
                      volumes=[MEMO_VOLUME],
                      grants={"SCHEMA dev.cas": [{"Principal": "account users",
                                                  "ActionType": "WRITE VOLUME"}]})
    out = _grant(spark)
    assert f"OK    CREATE VOLUME IF NOT EXISTS {MEMO_VOLUME}" in out
    assert f"OK    GRANT WRITE VOLUME ON VOLUME {MEMO_VOLUME}" in out
    assert "FAIL" not in out
    # OK for the App, but everyone else can write what the App replays: said out loud
    assert f"WARN  memo volume {MEMO_VOLUME} is open to everyone" in out
    assert "`account users` hold WRITE VOLUME on schema dev.cas" in out
    assert "write entries the App would replay" in out


def test_app_grant_warns_when_everyone_can_read_the_memo():
    # Read only (here via the catalog): a privacy note, without the replay warning
    spark = FakeSpark(grants={"CATALOG dev": [{"Principal": "account users",
                                               "ActionType": "READ VOLUME"}]})
    out = _grant(spark)
    assert f"WARN  memo volume {MEMO_VOLUME} is open to everyone" in out
    assert "can read the App's stored replies" in out
    assert "write entries" not in out and "FAIL" not in out
    # ALL PRIVILEGES covers both
    spark = FakeSpark(grants={f"VOLUME {MEMO_VOLUME}": [{"Principal": "users",
                                                         "ActionType": "ALL PRIVILEGES"}]})
    out = _grant(spark)
    assert "write entries the App would replay" in out


def test_app_grant_with_the_memo_off_neither_creates_nor_grants_the_volume():
    """The documented revert: no CREATE, no READ/WRITE VOLUME, no FAIL, no exposure check."""
    spark = FakeSpark(fail=[r"^CREATE VOLUME"])
    out = _grant(spark, bundle_env={"LLM_MEMO": "false"})
    assert not any("app_state" in q for q in spark.log), spark.log
    assert [q.split(" TO ")[0] for q in spark.log if q.startswith("GRANT ")] == [
        "GRANT USE CATALOG ON CATALOG dev", "GRANT USE SCHEMA ON SCHEMA dev.cas",
        "GRANT READ VOLUME ON VOLUME dev.cas.raw", "GRANT SELECT ON TABLE dev.cas.products_index"]
    assert "LLM_MEMO=false" in out and "FAIL" not in out and "WARN" not in out


def test_app_grant_fails_the_memo_grants_without_the_volume():
    # account users hold WRITE VOLUME on the schema, yet a missing volume is still a FAIL
    spark = FakeSpark(fail=[r"^CREATE VOLUME"],
                      grants={"SCHEMA dev.cas": [{"Principal": "account users",
                                                  "ActionType": "ALL PRIVILEGES"}]})
    out = _grant(spark)
    assert f"FAIL  CREATE VOLUME IF NOT EXISTS {MEMO_VOLUME}" in out
    assert f"FAIL  GRANT READ VOLUME ON VOLUME {MEMO_VOLUME}" in out
    assert f"FAIL  GRANT WRITE VOLUME ON VOLUME {MEMO_VOLUME}" in out
    assert not any(MEMO_VOLUME in q for q in spark.log if q.startswith("GRANT "))
    # The other grants still ran and passed
    assert "PASS  GRANT SELECT ON TABLE dev.cas.products_index" in out
    assert "Can read and write" in out


def test_app_grant_refuses_without_an_app():
    try:
        _grant(FakeSpark(), FakeWorkspace(existing=False))
        raise AssertionError("granted to an App that does not exist")
    except RuntimeError as e:
        assert "app_create()" in str(e)


# ------------------------------------------------------------------ T5: run_app
def _run_app():
    spec = importlib.util.spec_from_file_location("run_app_under_test", BUNDLE / "run_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_app_child_env():
    ra = _run_app()
    for host in ("adb-1.azuredatabricks.net", "https://adb-1.azuredatabricks.net/",
                 " https://adb-1.azuredatabricks.net "):
        with _env(DATABRICKS_HOST=host, OPENAI_BASE_URL=None, OPENAI_BASE_PATH=None,
                  BACKEND_URL="http://elsewhere"):
            env = ra._child_env()
        assert env["OPENAI_BASE_URL"] == "https://adb-1.azuredatabricks.net/serving-endpoints", host
        assert env["BACKEND_URL"] == "http://127.0.0.1:8001"
    # Unity Gateway: app.yaml's OPENAI_BASE_PATH, however its slashes are written
    for path in ("/ai-gateway/mlflow/v1", "ai-gateway/mlflow/v1/", " /ai-gateway/mlflow/v1 "):
        with _env(DATABRICKS_HOST="adb-1.azuredatabricks.net", OPENAI_BASE_URL=None,
                  OPENAI_BASE_PATH=path):
            assert (ra._child_env()["OPENAI_BASE_URL"]
                    == "https://adb-1.azuredatabricks.net/ai-gateway/mlflow/v1"), path
    # and the path app.yaml ships gives the URL the notebook tested (CELL2)
    with _cell2(), _chdir(ROOT):
        shipped = _load09()["_bundle_env"]()
    with _env(DATABRICKS_HOST="adb-1.azuredatabricks.net", OPENAI_BASE_URL=None,
              OPENAI_BASE_PATH=shipped["OPENAI_BASE_PATH"]):
        assert ra._child_env()["OPENAI_BASE_URL"] == CELL2["OPENAI_BASE_URL"]
    with _env(DATABRICKS_HOST="adb-1", OPENAI_BASE_URL="https://kept.example/serving-endpoints"):
        assert ra._child_env()["OPENAI_BASE_URL"] == "https://kept.example/serving-endpoints"
    with _env(DATABRICKS_HOST=None, OPENAI_BASE_URL=None):
        try:
            ra._child_env()
            raise AssertionError("started with no LLM base URL")
        except SystemExit as e:
            assert "DATABRICKS_HOST" in str(e.code)
    # app.yaml's BACKEND_URL agrees with the private port run_app gives uvicorn
    with _cell2():
        ns = _load09()
    with _chdir(ROOT):
        assert ns["_bundle_env"]()["BACKEND_URL"] == f"http://127.0.0.1:{ra.BACKEND_PORT}"


class _FakeProc:
    def __init__(self, args, env):
        self.args, self.env, self.returncode = args, env, 0

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def test_run_app_argv_keeps_the_api_private():
    ra = _run_app()
    procs = []

    def popen(args, env):
        procs.append(_FakeProc(args, env))
        return procs[-1]

    fake_signal = SimpleNamespace(signal=lambda *a: None, SIGTERM=15, SIGINT=2)
    fake_sp = SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired)
    with _patched(ra, signal=fake_signal, subprocess=fake_sp), _quiet(), _env(
            DATABRICKS_APP_PORT="8123", DATABRICKS_HOST="adb-1", OPENAI_BASE_URL=None):
        rc = ra.main()
    assert rc == 1  # a child that exits stops the App (the platform shows it crashed)
    api, ui = (p.args for p in procs)
    assert api == [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                   "--port", "8001", "--workers", "1", "--no-access-log"]
    head = [sys.executable, "-m", "streamlit", "run", "ui.py", "--server.port", "8123",
            "--server.address", "0.0.0.0", "--server.headless", "true"]
    assert ui[:len(head)] == head
    assert ui[len(head):] == ra.STREAMLIT_THEME_FLAGS
    flags = ra.STREAMLIT_THEME_FLAGS
    assert len(flags) % 2 == 0 and all(re.fullmatch(r"--(theme|client)\.[\w.]+", k)
                                       for k in flags[::2]), flags
    for p in procs:
        assert p.env["BACKEND_URL"] == "http://127.0.0.1:8001"
        assert p.env["OPENAI_BASE_URL"] == "https://adb-1/serving-endpoints"


# ------------------------------------------------------------------ T6: vector_search
@contextlib.contextmanager
def _vs_state(**cfg_values):
    """Restore vector_search's client cache, Spark and the cfg knobs it reads."""
    saved = dict(VS._workspace)
    with _patched(VS, _spark=None), _patched(cfg, **cfg_values):
        try:
            VS._workspace.clear()
            yield
        finally:
            VS._workspace.clear()
            VS._workspace.update(saved)


def _csv(rows, with_index=True):
    import pandas as pd

    df = pd.DataFrame(rows)
    if not with_index:
        df = df.drop(columns=["catalog_index"])
    return df.to_csv(index=False).encode()


def _files_api(payload, calls):
    def download(path):
        calls.append(path)
        return SimpleNamespace(contents=io.BytesIO(payload))

    return SimpleNamespace(files=SimpleNamespace(download=download))


def test_load_catalog_reads_the_app_export_through_the_files_api():
    path = "/Volumes/dev/cas/raw/app_catalog_cas_test.csv"
    assert not os.path.exists(path)
    rows = [{"catalog_index": i, "Product_title": f"t{i}", "price": None} for i in (2, 0, 3, 1)]
    calls = []
    with _vs_state(VS_CATALOG_CSV=path):
        VS._workspace[True] = _files_api(_csv(rows), calls)
        df = VS.load_catalog()
    assert calls == [path]
    assert df["catalog_index"].tolist() == [0, 1, 2, 3]
    assert df["Product_title"].tolist() == ["t0", "t1", "t2", "t3"]  # sorted by key, not file order
    assert list(df.index) == [0, 1, 2, 3]
    # The App's disk is small: the downloaded copy is deleted once parsed
    assert not os.path.exists(os.path.join(tempfile.gettempdir(), "cas_app_catalog_cas_test.csv"))


def test_load_catalog_refuses_a_sparse_key_and_numbers_a_raw_csv():
    path = "/Volumes/dev/cas/raw/app_catalog_cas_test.csv"
    sparse = [{"catalog_index": i, "Product_title": f"t{i}"} for i in (0, 1, 3)]
    with _vs_state(VS_CATALOG_CSV=path):
        VS._workspace[True] = _files_api(_csv(sparse), [])
        try:
            VS.load_catalog()
            raise AssertionError("a sparse catalog_index was accepted")
        except RuntimeError as e:
            assert "not dense" in str(e)
        raw = [{"catalog_index": 0, "Product_title": "a"}, {"catalog_index": 0, "Product_title": "b"}]
        VS._workspace[True] = _files_api(_csv(raw, with_index=False), [])
        df = VS.load_catalog()
    assert df["catalog_index"].tolist() == [0, 1] and df["Product_title"].tolist() == ["a", "b"]


def test_search_parses_keys_and_the_score_column():
    sent = []

    def rest(method, path, body=None, query=None):
        sent.append((method, path, body))
        return {"manifest": {"columns": [{"name": "catalog_index"}, {"name": "score"}]},
                "result": {"data_array": [["3", 0.9], [7.0, "0.5"], [], ["11.0", 0.1]]}}

    with _vs_state(VS_INDEX="dev.cas.products_index"), _patched(VS, rest=rest):
        out = VS._search(5, query_vector=[1.0, 0.0])
    assert out == [(3, 0.9), (7, 0.5), (11, 0.1)]
    assert sent == [("POST", "/indexes/dev.cas.products_index/query",
                     {"columns": ["catalog_index"], "num_results": 5, "query_vector": [1.0, 0.0]})]


def _embedder(data, tokens=7, calls=None):
    def query(name, input):  # noqa: A002 -- the SDK's keyword
        if calls is not None:
            calls.append((name, list(input)))
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=tokens))

    return SimpleNamespace(serving_endpoints=SimpleNamespace(query=query))


def _emb(i, vec):
    return SimpleNamespace(index=i, embedding=vec)


def test_embed_texts_restores_order_and_normalises():
    calls = []
    with _vs_state(VS_EMBEDDING_MODEL="databricks-gte-large-en", VS_EMBEDDING_SERVICE=""):
        VS._workspace[False] = _embedder([_emb(1, [0.0, 2.0]), _emb(0, [3.0, 4.0])], calls=calls)
        vecs, tokens = VS.embed_texts(["a", "b"])
        assert VS.embed_texts([]) == ([], 0)
    assert calls == [("databricks-gte-large-en", ["a", "b"])]  # one call, none for []
    assert tokens == 7
    assert vecs == [[0.6, 0.8], [0.0, 1.0]]
    assert all(isinstance(x, float) for v in vecs for x in v)  # json-safe, not numpy
    assert all(math.isclose(sum(x * x for x in v), 1.0) for v in vecs)


def test_embed_texts_refuses_short_ragged_empty_and_zero_replies():
    bad = {"short": [_emb(0, [1.0, 0.0])],
           "ragged": [_emb(0, [1.0, 0.0]), _emb(1, [1.0, 0.0, 0.0])],
           "empty": [_emb(0, [1.0, 0.0]), _emb(1, [])],
           "zero": [_emb(0, [1.0, 0.0]), _emb(1, [0.0, 0.0])]}
    for name, data in bad.items():
        with _vs_state(VS_EMBEDDING_MODEL="databricks-gte-large-en", VS_EMBEDDING_SERVICE=""):
            VS._workspace[False] = _embedder(data)
            try:
                VS.embed_texts(["a", "b"])
                raise AssertionError(f"{name} reply accepted")
            except RuntimeError:
                pass
        # the same refusals on the Unity Gateway route
        with _vs_state(VS_EMBEDDING_MODEL="databricks-gte-large-en",
                       VS_EMBEDDING_SERVICE="system.ai.gte-large-en", AI_GATEWAY_URL="https://g"), \
                _gateway_fake(data, []):
            try:
                VS.embed_texts(["a", "b"])
                raise AssertionError(f"{name} gateway reply accepted")
            except RuntimeError as e:
                assert "system.ai.gte-large-en" in str(e)


@contextlib.contextmanager
def _gateway_fake(data, calls, tokens=5):
    """The OpenAI client embed_texts builds for Unity Gateway, replaced by a recorder."""
    def create(model, input):  # noqa: A002 -- the openai keyword
        calls.append((model, list(input)))
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=tokens))

    fake = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with _patched(VS, _gateway_client=lambda batch=False: fake):
        yield


def test_embed_texts_through_unity_gateway():
    """VS_EMBEDDING_SERVICE set: the gateway is called by that name, never the endpoint."""
    calls = []

    def no_endpoint(**kw):
        raise AssertionError("serving_endpoints.query called on the gateway route")

    with _vs_state(VS_EMBEDDING_MODEL="databricks-gte-large-en",
                   VS_EMBEDDING_SERVICE="system.ai.gte-large-en", AI_GATEWAY_URL="https://g"), \
            _gateway_fake([_emb(1, [0.0, 2.0]), _emb(0, [3.0, 4.0])], calls):
        VS._workspace[False] = SimpleNamespace(serving_endpoints=SimpleNamespace(query=no_endpoint))
        vecs, tokens = VS.embed_texts(["a", "b"])
        assert VS.embed_texts([]) == ([], 0)
    assert calls == [("system.ai.gte-large-en", ["a", "b"])]
    assert tokens == 5 and vecs == [[0.6, 0.8], [0.0, 1.0]]


def test_gateway_url_and_client():
    """AI_GATEWAY_URL wins; else the SDK host. The token is asked of the SDK per request."""
    asked, made = [], []

    class Config:
        host = "https://adb-1.azuredatabricks.net/"

        def authenticate(self):
            asked.append(1)
            return {"Authorization": "Bearer tok-1"}

    class FakeOpenAI:
        def __init__(self, **kw):
            made.append(kw)

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = FakeOpenAI
    saved_mod, saved = sys.modules.get("openai"), dict(VS._gateway)
    try:
        sys.modules["openai"] = openai_mod
        VS._gateway.clear()
        with _vs_state(AI_GATEWAY_URL=""):
            VS._workspace[False] = SimpleNamespace(config=Config())
            assert VS.gateway_url() == "https://adb-1.azuredatabricks.net/ai-gateway/mlflow/v1"
            c1 = VS._gateway_client()
            assert VS._gateway_client() is c1 and len(made) == 1  # one client per (batch, URL)
            assert made[0]["base_url"] == VS.gateway_url() and made[0]["timeout"] == 15
            assert asked == [] and made[0]["api_key"]() == "tok-1" and asked == [1]
        with _vs_state(AI_GATEWAY_URL="https://adb-2.azuredatabricks.net/ai-gateway/mlflow/v1"):
            assert VS.gateway_url() == "https://adb-2.azuredatabricks.net/ai-gateway/mlflow/v1"
    finally:
        VS._gateway.clear()
        VS._gateway.update(saved)
        if saved_mod is None:
            sys.modules.pop("openai", None)
        else:
            sys.modules["openai"] = saved_mod


def test_query_many_embeds_once_and_honours_the_query_mode():
    calls, sent = [], []

    def rest(method, path, body=None, query=None):
        sent.append(body)
        return {"manifest": {"columns": [{"name": "catalog_index"}, {"name": "score"}]},
                "result": {"data_array": [[len(sent), 0.5]]}}

    with _vs_state(VS_QUERY_MODE="vector"), _patched(VS, rest=rest):
        VS._workspace[False] = _embedder([_emb(i, [1.0, float(i)]) for i in range(3)], calls=calls)
        assert VS.query_many(["x", "y", "z"], 4) == [[(1, 0.5)], [(2, 0.5)], [(3, 0.5)]]
    assert len(calls) == 1 and all("query_vector" in b for b in sent)
    sent.clear()
    with _vs_state(VS_QUERY_MODE="text"), _patched(VS, rest=rest):
        VS.query_many(["x"], 4)
        assert sent == [{"columns": ["catalog_index"], "num_results": 4, "query_text": "x"}]
    with _vs_state(VS_QUERY_MODE="hybrid"):
        try:
            VS.query_many(["x"], 4)
            raise AssertionError("an unknown VS_QUERY_MODE was accepted")
        except ValueError:
            pass


# ------------------------------------------------------------------ vs_shim and 00
def test_patch_eval_harness_turns_the_memo_off_and_says_so():
    import pathlib

    import app.llm_client as llm_client
    import eval.run_eval as run_eval
    from app import search, tracking

    made = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(_cell2())
        stack.enter_context(_patched(cfg, LLM_MEMO=True, SEARCH_BACKEND="local",
                                     LLM_MEMO_DIR=".cache/llm_memo"))
        stack.enter_context(_patched(llm_client, _singleton=llm_client._singleton))
        stack.enter_context(_patched(run_eval, EVAL_QUERIES_JSON=run_eval.EVAL_QUERIES_JSON,
                                     EVAL_RESULTS_DIR=run_eval.EVAL_RESULTS_DIR))
        stack.enter_context(_patched(search, load_index=lambda: None,
                                     get_dataframe=lambda: [1, 2, 3]))
        stack.enter_context(_patched(VS, set_spark=lambda s: None,
                                     index_info=lambda: {"name": "idx"}))
        stack.enter_context(_patched(tracking, enable_autolog=lambda: False))
        # Never create /Volumes/... on the laptop
        stack.enter_context(_patched(pathlib.Path, mkdir=lambda self, **kw: made.append(self)))
        out = stack.enter_context(_quiet())
        assert vs_shim.patch_eval_harness(object()) == {"name": "idx"}
        assert cfg.LLM_MEMO is False and cfg.SEARCH_BACKEND == "databricks"
        assert "memo    : off" in out.getvalue()
        vs_shim.patch_eval_harness(object(), memo=True)
        assert cfg.LLM_MEMO is True
        assert "memo    : ON -> .cache/llm_memo" in out.getvalue()
        assert "not a /Volumes path" in out.getvalue()
        # ON reaches direct app calls only: the header must not promise an eval replay
        assert "harnesses turn it off at their start" in out.getvalue()
        run_eval_src = (ROOT / "eval" / "run_eval.py").read_text(encoding="utf-8")
        pages_src = (ROOT / "eval" / "eval_page_types.py").read_text(encoding="utf-8")
        assert "llm_memo.disable_for_eval()" in run_eval_src  # what makes that line true
        assert "cfg.LLM_MEMO = False" in pages_src
    assert len(made) == 2 and all(p.as_posix().endswith("/Volumes/dev/cas/evals") for p in made)


def test_vs_shim_memo_location_follows_cfg():
    with _cell2():
        assert vs_shim.memo_volume() == MEMO_VOLUME
        assert vs_shim.app_memo_dir() == MEMO_DIR
        assert "the APP's memo" in vs_shim._memo_line(SimpleNamespace(LLM_MEMO=True,
                                                                      LLM_MEMO_DIR=MEMO_DIR))
    with _patched(cfg, CAS_CATALOG="client", CAS_SCHEMA="shop"):
        assert vs_shim.app_memo_dir() == "/Volumes/client/shop/app_state/llm_memo"


def _load00():
    ns = {"__name__": "nb00", "spark": FakeSpark()}
    with _chdir(ROOT):
        exec(compile(NB00.read_text(encoding="utf-8"), str(NB00), "exec"), ns)  # noqa: S102
    return ns


def test_preflight_compares_pandas_and_numpy_with_the_app_pins():
    ns = _load00()
    pins = {"pandas": "3.0.3", "numpy": "2.4.6"}
    assert ns["_pf_versus_app"]({"pandas": "3.0.3", "numpy": "2.4.6"}, pins)[0] == "PASS"
    status, detail = ns["_pf_versus_app"]({"pandas": "2.2.3", "numpy": "2.4.6"}, pins)
    assert status == "WARN" and "pandas 2.2.3 (App 3.0.3)" in detail and "%pip" in detail
    with tempfile.TemporaryDirectory() as tmp, _chdir(tmp):  # not the repo root
        assert ns["_pf_versus_app"]({"pandas": "3.0.3"})[0] == "WARN"
    with _chdir(ROOT):  # the real requirements file
        assert ns["_pf_versus_app"]({"pandas": "3.0.3", "numpy": "2.4.6"})[0] == "PASS"


def test_preflight_memo_volume_dry_write():
    ns = _load00()
    with tempfile.TemporaryDirectory() as tmp:
        vol = Path(tmp) / "app_state"
        status, detail = ns["_pf_memo_probe"](str(vol))
        assert status == "WARN" and "app_grant()" in detail  # missing: a WARN, never a FAIL
        vol.mkdir()
        assert ns["_pf_memo_probe"](str(vol))[0] == "PASS"
        assert list(vol.iterdir()) == []  # the probe is removed

        def refusing_open(path, mode="r", **kw):
            if "w" in mode:
                raise PermissionError(path)
            return open(path, mode, **kw)

        ns["open"] = refusing_open
        status, detail = ns["_pf_memo_probe"](str(vol))
        assert status == "WARN" and "dry write failed" in detail
        assert list(vol.iterdir()) == []

        def lying_open(path, mode="r", **kw):  # the write lands, the read-back differs
            if "w" in mode:
                return open(path, mode, **kw)
            return io.StringIO('{"probe": "someone else"}')

        ns["open"] = lying_open
        status, detail = ns["_pf_memo_probe"](str(vol))
        assert status == "WARN" and "read back different bytes" in detail
        assert list(vol.iterdir()) == []


def test_preflight_runs_the_new_rows():
    """preflight() itself, on fakes: the two rows print (other rows FAIL here, by design)."""
    with _fake_sdk(), tempfile.TemporaryDirectory() as tmp, _vs_state():
        ns = _load00()
        vol = Path(tmp) / "app_state"
        vol.mkdir()
        ns["vs_shim"] = SimpleNamespace(memo_volume_path=lambda: str(vol),
                                        bundle_pins=lambda: _bundle_pins(),
                                        APP_REQUIREMENTS="x")
        got = []

        def get(name):
            got.append(name)

        sys.modules["databricks.sdk"].WorkspaceClient = lambda **kw: SimpleNamespace(
            serving_endpoints=SimpleNamespace(get=get))
        from app import llm_client

        with _quiet() as out, _chdir(ROOT), _cell2(), _patched(
                llm_client, generate_json=lambda *a, **k: {"ok": True}):
            try:
                ns["preflight"]()
            except AssertionError:
                pass  # no workspace here: most rows FAIL
    text = out.getvalue()
    # Model services are proved by the call rows, never by a serving-endpoint GET
    assert got == [] and re.search(r"PASS  serving endpoints\s+system\.ai\.gte-large-en "
                                   r"\(model service", text), text
    assert re.search(r"PASS  llm call \(JSON mode\)\s+system\.ai\.qwen3", text), text
    assert re.search(r"PASS  App memo volume \(dry write\)\s+.*app_state exists", text), text
    assert re.search(r"(PASS|WARN)  pandas / numpy vs the App\s+pandas ", text), text


# ------------------------------------------------------------------ Phase C: 08, 10, 12
# The notebooks exec'd as a notebook runs them (repo-root cwd), on fakes only: a model
# behind the REAL llm_client.generate_json (it answers each prompt in shape and records
# usage like a provider), a fake pyspark, a fake app.main for check_determinism. The
# memo only ever points at a temp dir.
NB08 = ROOT / "databricks" / "08_llm_smoke.py"
NB10 = ROOT / "databricks" / "10_page_eval.py"
NB12 = ROOT / "databricks" / "12_determinism.py"
SMOKE_CALLS = 43   # 08's provider calls against a cooperative model: its docstring's Cost
DEVICE_ROW = "device gift (watch collector)"


def _exec_nb(path, name):
    ns = {"__name__": name}
    with _chdir(ROOT):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)  # noqa: S102
    return ns


# A cooperative model's intents for 08's translator queries (anything else: 3 sweaters)
_SMOKE_INTENTS = {
    "gift for a vintage watch collector": ["vintage mechanical wristwatch",
                                           "antique pocket watch",
                                           "vintage watch display case"],
    "laptop for college": ["lightweight laptop for college", "student laptop", "chromebook"],
    "genuine leather belt for men": ["men's genuine leather belt", "leather dress belt men",
                                     "men's casual leather belt"],
    "complete my look for a men's summer wedding": ["men's linen suit", "men's linen shirt",
                                                    "men's chino trousers"],
}


def _obeying_scores(line, query):
    """A rerank that keeps every cap 08 checks: the device and the asked type first,
    storage below the watches, tools and care, other types and the other gender at 30."""
    if "Men's Black Anime Graphic T-Shirt" in line:
        return 98
    if "role=device" in line:
        return 92
    if "Winder" in line or "Watch Box" in line:
        return 60
    if " Kit" in line or "Bridal" in line or re.search(r"type=(footwear|jewelry|bag)", line):
        return 30
    if "Women's" in line and query.startswith('"men'):
        return 30
    return 70


def _ignoring_scores(line, query):
    """What the recorded 2026-10-15 rerank did with this window: winders 85-88, the box
    72, kits 55-60 and the watch 40."""
    if "Winder" in line:
        return 88
    if "Watch Box" in line:
        return 72
    if " Kit" in line:
        return 58
    if "role=device" in line:
        return 40
    return _obeying_scores(line, query)


class _SmokeModel:
    """The provider behind llm_client.generate_json for 08. Records usage like a real
    one (08's "a real reply" check holds) and describes its request (memo_request), so
    the memo WOULD key it if it were on."""

    def __init__(self, intents=None, scores=_obeying_scores):
        import app.llm_client as llm_client

        self.model, self.base_url = cfg.OPENAI_MODEL, cfg.OPENAI_BASE_URL
        self.intents = dict(_SMOKE_INTENTS, **(intents or {}))
        self.scores = scores
        self.prompts, self.memo_seen = [], []
        self._record = llm_client._record_usage

    def memo_request(self, prompt, temperature):
        return {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "json_mode": True, "temperature": temperature}

    def generate_json(self, prompt, temperature):
        self.prompts.append(prompt)
        self.memo_seen.append(cfg.LLM_MEMO)
        reply = self.reply(prompt)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        self._record(len(prompt) // 4, len(text) // 4)
        return text

    def reply(self, prompt):
        from app import translator

        if "\nUser query: " in prompt:
            q = json.loads(prompt.rsplit("\nUser query: ", 1)[1])
            if prompt.startswith(translator.HYDE_PROMPT):
                return {"hypothetical_listing": "Women's breathable cotton linen blouse, "
                        "loose fit short sleeve top for hot humid summer days, light and airy."}
            if prompt.startswith(translator.HYBRID_PROMPT):
                return {"hypothetical_listing": "Breathable cotton blouse for humid weather.",
                        "search_terms": ["breathable cotton top", "linen blouse"]}
            if prompt.startswith(translator.RECIPE_PROMPT):
                return {"search_terms": ["all-purpose flour", "eggs", "whole milk",
                                         "baking powder", "unsalted butter"]}
            if prompt.startswith(translator.SETUP_PROMPT):
                return {"search_terms": ["27 inch monitor", "wireless keyboard and mouse",
                                         "ergonomic office chair", "standing desk", "webcam"]}
            if q == "asdfghjkl":
                return {"no_intent": True}
            reply = self.intents.get(q)
            if isinstance(reply, str):
                return reply                     # raw text: an unparseable reply
            if reply is not None:
                out = {"search_terms": reply}
                if "for men" in q:
                    out["constraints"] = [{"type": "gender", "value": "men"}]
                return out
            if "red shoes" in q:
                return {"search_terms": ["red shoes", "red sneakers", "red heels"]}
            return {"search_terms": ["warm wool sweater", "knit cardigan", "fleece pullover"]}
        if "The user searched for:" in prompt:
            query = prompt.split("The user searched for: ", 1)[1].split("\n", 1)[0]
            lines = re.findall(r"^  idx=(\d+) \|(.*)$", prompt, flags=re.MULTILINE)
            return {"ranked": [{"idx": int(i), "score": self.scores(rest, query),
                                "reason": "fits"} for i, rest in lines]}
        if "<<<MESSAGE" in prompt:
            msg = prompt.split("<<<MESSAGE\n", 1)[1].split("\nMESSAGE>>>", 1)[0]
            out = {"action": "search", "new_topic": True, "search_query": "warm sweater",
                   "reply": "Here you go.", "exclude_terms": []}
            if "remove the socks" in msg:
                out.update(action="refine", new_topic=False, search_query="",
                           exclude_terms=["socks"])
            elif "how are you" in msg:
                out.update(action="reply", new_topic=False, search_query="")
            elif "cheaper" in msg:
                out.update(new_topic=False, search_query="cheap warm wool sweater for men")
            elif "pancakes" in msg:
                out.update(search_query="pancake ingredients")
            return out
        if "cross-sell assistant" in prompt:
            return {"complementary": ["mouse pad", "usb hub"]}
        return {}


@contextlib.contextmanager
def _smoke(memo, memo_dir, **model_kw):
    """cfg as Cell 2 sets it for the Databricks LLM, the memo at `memo_dir` (a temp dir,
    never the repo's .cache), and a _SmokeModel as the LLM client. Yields the model."""
    import app.llm_client as llm_client
    from app import llm_memo

    with contextlib.ExitStack() as stack:
        stack.enter_context(_cell2())
        stack.enter_context(_patched(
            cfg, LLM_PROVIDER="openai", NUM_INTENTS=3, QUERY_GUARD_ENABLED=True,
            OUTFIT_BUNDLES_ENABLED=False, LLM_MEMO=memo, LLM_MEMO_DIR=str(memo_dir),
            OPENAI_BASE_URL="https://adb-1.azuredatabricks.net/serving-endpoints"))
        stack.enter_context(_env(OPENAI_MODEL=CELL2["OPENAI_MODEL"]))
        model = _SmokeModel(**model_kw)
        stack.enter_context(_patched(llm_client, get_llm_client=lambda: model,
                                     _singleton=llm_client._singleton))
        llm_memo.reset()
        try:
            yield model
        finally:
            llm_memo.reset()   # closes the SQLite store before the temp dir goes


def _row(rows, name):
    return next(r for r in rows if r["check"] == name)


def _problems(row, level=None):
    return [m for lvl, m in row["problems"] if level in (None, lvl)]


def _smoke_files(memo_dir):
    return sorted(p.name for p in Path(memo_dir).rglob("*") if p.is_file())


def test_smoke_08_runs_every_check_with_the_memo_off():
    """run() end to end on a cooperative model: the 43 calls its docstring counts, every
    one made with LLM_MEMO off although the notebook had it on, nothing stored, the
    params row and the result say so, and the notebook's value comes back -- also when
    a check raises."""
    ns = _exec_nb(NB08, "smoke_08")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
            _smoke(True, tmp) as model, _quiet() as out:
        result = ns["run"]()
        assert cfg.LLM_MEMO is True                       # the notebook's value is back
        assert _smoke_files(tmp) == []                    # nothing was memoised
    text = out.getvalue()
    assert len(model.prompts) == SMOKE_CALLS, len(model.prompts)
    assert set(model.memo_seen) == {False}
    assert result["usage"]["calls"] == SMOKE_CALLS and result["usage"]["memo_hits"] == 0
    assert result["passed"] and (result["fails"], result["warns"]) == (0, 0), \
        [(r["check"], r["problems"]) for r in result["checks"] if r["problems"]]
    assert result["llm_memo"] is False and result["llm_memo_before"] is True
    memo_row = _row(result["checks"], "LLM memo off")
    assert memo_row["area"] == "params" and memo_row["status"] == "PASS"
    assert "LLM_MEMO was on; restored after" in memo_row["note"]
    assert "LLM memo off, 0 memo hits" in text
    src = NB08.read_text(encoding="utf-8")
    assert f"Cost: {SMOKE_CALLS} LLM calls" in src       # the docstring follows the code
    # The new rows exist, under their areas
    for area, name in (("translator", "gift: vintage watch collector"),
                       ("translator", "device: laptop for college"), ("reranker", DEVICE_ROW)):
        assert _row(result["checks"], name)["area"] == area

    # A check that raises past _check still gives the notebook its memo back
    ns2 = _exec_nb(NB08, "smoke_08")

    def boom(rows):
        raise RuntimeError("x")

    ns2["_assistant_checks"] = boom
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
            _smoke(True, tmp), _quiet():
        try:
            ns2["run"]()
        except RuntimeError:
            pass
        else:
            raise AssertionError("run() swallowed the error")
        assert cfg.LLM_MEMO is True


def test_smoke_08_fails_rows_a_memo_answered():
    """The safety net behind the switch: with llm_memo.disable_for_eval broken, the memo
    stays on -- the params row FAILs on the first run already, the x3 injection probes
    replay their own first sample, and on a second run the translator rows are answered
    from the memo with 0 provider calls: every such row FAILs, never PASSes."""
    from app import llm_memo

    ns = _exec_nb(NB08, "smoke_08")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
            _patched(llm_memo, disable_for_eval=lambda: False), \
            _smoke(True, tmp) as model, _quiet():
        first = ns["run"]()
        n_first = len(model.prompts)
        assert _smoke_files(tmp)                          # it DID store replies
        second = ns["run"]()
    assert not first["passed"] and not second["passed"]
    assert _row(first["checks"], "LLM memo off")["status"] == "FAIL"
    pwned = _row(first["checks"], "translator: PWNED payload")
    assert pwned["status"] == "FAIL" and any("replayed from the LLM memo" in m
                                             for m in _problems(pwned, "FAIL")), pwned
    wool = _row(second["checks"], "expand: wool sweater")
    assert wool["calls"] == 0 and wool["status"] == "FAIL"
    assert any("replayed from the LLM memo" in m for m in _problems(wool, "FAIL")), wool
    assert second["usage"]["memo_hits"] > 0
    assert len(model.prompts) - n_first < n_first          # the second run mostly replayed


def _translator_rows(intents):
    ns = _exec_nb(NB08, "smoke_08")
    rows = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
            _smoke(False, tmp, intents=intents):
        ns["_translator_checks"](rows)
    return _row(rows, "gift: vintage watch collector"), _row(rows, "device: laptop for college")


def test_smoke_08_gift_and_laptop_translator_rows():
    """The watch gift WARNs unless its first intent is a watch itself (with at most one
    storage piece and no tool or care intent: owner call D1), FAILs on an unparseable
    reply; the laptop ask WARNs on an intent that is not a laptop itself."""
    gift, laptop = _translator_rows({})
    assert (gift["status"], laptop["status"]) == ("PASS", "PASS"), (gift, laptop)
    # The recorded 2026-10-27 failure: three accessory intents and no watch
    gift, laptop = _translator_rows({
        "gift for a vintage watch collector": ["vintage watch repair kit",
                                               "leather watch winder",
                                               "vintage watch display case"],
        "laptop for college": ["laptop sleeve", "laptop backpack", "student laptop"]})
    assert gift["status"] == "WARN" and not _problems(gift, "FAIL"), gift
    warns = " | ".join(_problems(gift, "WARN"))
    assert "first intent is not a watch itself: 'vintage watch repair kit'" in warns
    assert "2 storage / display intents" in warns
    assert "add-on intents the GIFT rule rules out: ['vintage watch repair kit']" in warns
    assert laptop["status"] == "WARN", laptop
    assert "not a laptop itself: ['laptop sleeve', 'laptop backpack']" in \
        " | ".join(_problems(laptop, "WARN"))
    # One display piece after the watch is what the GIFT bullet allows
    gift, _ = _translator_rows({"gift for a vintage watch collector": [
        "vintage automatic watch", "watch display box"]})
    assert gift["status"] == "PASS", gift
    # An unparseable reply is a translate error, so a FAIL, never a soft WARN
    gift, _ = _translator_rows({"gift for a vintage watch collector": "Sure! Watches."})
    assert gift["status"] == "FAIL" and any("translate errors" in m
                                            for m in _problems(gift, "FAIL")), gift


def test_smoke_08_device_window_is_what_search_sends():
    """The rerank row's rows, fields and constraint are /search's own for this query:
    the load-time columns by search._annotate_types, kinds that match app/device_role's
    reading, main's gate acting on this window, 08's role marking equal to
    main._with_role_fields, and no constraint but device_role.constraint's text -- the
    function main.py appends, imported, never retyped."""
    import app.main as main
    from app import device_role, product_type

    ns = _exec_nb(NB08, "smoke_08")
    query, specs = ns["_DEVICE_QUERY"], ns["_DEVICE_ROWS"]
    ask = device_role.requested_device(query)
    assert ask == device_role.DeviceAsk("watch", True, False)
    rows = ns["_device_window"]()
    assert [r["Product_title"] for r in rows] == [s[0] for s in specs]
    for row, (title, vertical, *_rest, kind) in zip(rows, specs):
        assert row["bsns_vrtcl_name"] == vertical
        assert row["watch_kind"] is not None, title       # the taxonomy columns are there
        assert all(not (isinstance(v, float) and math.isnan(v)) for v in row.values()), title
        if kind == "watch":
            assert device_role.is_core(row, "watch"), title
        else:
            assert device_role.role(row, "watch") == device_role.ACCESSORY, title
            assert device_role.addon_rank(row, "watch") == (0 if kind == "storage" else 2), title
    assert {s[1] for s in specs if s[5] == "watch"} == {"Fashion", "Electronics"}
    assert {s[5] for s in specs} == {"watch", "storage", "care"}
    # /search would act on this window: the gate passes, no type or guest constraint
    assert main._effective_device(query, rows, enabled=True) == ask
    assert not product_type.requested_types(query) and main._guest_constraint(query) is None
    main_src = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "rerank_constraints + [device_role.constraint(device)]" in main_src
    assert ns["_role_marked"](rows, ask) == main._with_role_fields(rows, ask)
    src = NB08.read_text(encoding="utf-8")
    assert "constraints = [device_role.constraint(ask)]" in src
    assert "the gift is a watch itself" not in src       # never a hand-copied constraint


def test_smoke_08_device_rerank_row_offline():
    """The row sends ` | role=` on every line and main.py's constraint, PASSes on a model
    that keeps the gift's order and caps, and WARNs (never FAILs) on the recorded one."""
    from app import device_role

    ask = device_role.requested_device("gift for a vintage watch collector")
    for scores, expect in ((_obeying_scores, set()),
                           (_ignoring_scores, {"gift cap", "40 or less", "storage piece"})):
        ns = _exec_nb(NB08, "smoke_08")
        rows = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
                _smoke(False, tmp, scores=scores) as model:
            ns["_rerank_checks"](rows)
        row = _row(rows, DEVICE_ROW)
        assert not _problems(row, "FAIL"), row
        warns = " | ".join(_problems(row, "WARN"))
        for needle in ("gift cap", "40 or less", "storage piece"):
            assert (needle in warns) is (needle in expect), (needle, warns)
        assert row["status"] == ("WARN" if expect else "PASS")
        prompt = next(p for p in model.prompts if '"gift for a vintage watch collector"' in p)
        assert prompt.count("| role=device ") == 4 and prompt.count("| role=accessory ") == 4
        assert "| type=accessory | audience=adult | role=device " in prompt
        assert f"     - [product_type] {device_role.constraint(ask)['value']}" in prompt
        assert prompt.count("     - [") == 1


# ------------------------------------------------------------------ 10 and 12: fakes
@contextlib.contextmanager
def _fake_pyspark():
    """pyspark.sql.SparkSession.builder.getOrCreate() -> a marker (no Spark locally)."""
    sql = types.ModuleType("pyspark.sql")
    sql.SparkSession = SimpleNamespace(builder=SimpleNamespace(getOrCreate=lambda: "spark"))
    pkg = types.ModuleType("pyspark")
    pkg.sql = sql
    mods = {"pyspark": pkg, "pyspark.sql": sql}
    old = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _fake_harness(calls, leave_memo_on=False):
    """vs_shim.patch_eval_harness without a workspace: records `memo` and sets it."""

    def patch(spark, verbose=True, llm=None, memo=False):
        calls.append({"spark": spark, "memo": memo})
        cfg.LLM_MEMO = True if leave_memo_on else bool(memo)
        print("memo    : " + ("ON" if cfg.LLM_MEMO else "off"))
        return {"name": "idx"}

    return patch


def test_page_eval_10_runs_with_the_memo_off_and_counts_the_queries():
    import eval.eval_page_types as page_eval

    rows = json.loads((ROOT / "data" / "page_type_queries.json").read_text(encoding="utf-8"))
    doc = NB10.read_text(encoding="utf-8")
    assert f"{len(rows)} queries ~" in doc and "38 queries" not in doc
    assert sum(r.get("group") == "device" for r in rows) == 10 and "the 10 'device' rows" in doc

    seen = []

    def fake_run(**kw):
        seen.append((cfg.LLM_MEMO, kw))
        return "x.csv", []

    for leave_on in (False, True):
        calls = []
        ns = _exec_nb(NB10, "nb10")
        with _fake_pyspark(), _patched(cfg, LLM_MEMO=True), _quiet(), \
                _patched(vs_shim, patch_eval_harness=_fake_harness(calls, leave_on)), \
                _patched(page_eval, run=fake_run):
            try:
                out = ns["run"](repeats=2)
            except RuntimeError as e:
                assert leave_on and "LLM memo is still on" in str(e), e
            else:
                assert not leave_on and out == ("x.csv", [])
        assert calls == [{"spark": "spark", "memo": False}]   # said out loud
    assert len(seen) == 1 and seen[0][0] is False and seen[0][1]["repeats"] == 2


def test_determinism_12_builds_its_memo_dir_below_a_scratch_volume_only():
    from eval import check_determinism as det

    ns = _exec_nb(NB12, "nb12")
    ns["_me"] = lambda: "me_example.com"
    with _cell2():
        root = ns["scratch_root"]()
        assert root == "/Volumes/dev/cas/evals/determinism"
        for bad in ("app_state", "APP_STATE", "../app_state", "evals/x", ""):
            try:
                ns["scratch_root"](bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted volume {bad!r}")
        fresh = ns["check_memo_dir"]()
        assert re.fullmatch(root + r"/me_example\.com/\d{8}-\d{6}/llm_memo", fresh), fresh
        again = f"{root}/me_example.com/20261030-120000/llm_memo"
        assert ns["check_memo_dir"](again) == again
        assert ns["check_memo_dir"](again + "/") == again
        for bad in (vs_shim.app_memo_dir(), f"{root}/../../app_state/llm_memo",
                    f"{root}/me/20261030-120000", "/Volumes/dev/cas/evals/other/llm_memo",
                    ".cache/llm_memo", "/Volumes/dev/cas/evals/determinism2/x/llm_memo"):
            try:
                ns["check_memo_dir"](bad)
            except ValueError:
                continue
            raise AssertionError(f"accepted memo_dir {bad!r}")
        # What 12 hands check_determinism passes ITS rules only with --allow-volume, and
        # the App's memo stays refused even then
        assert det._refusal(fresh, allow_volume=True) is None
        assert det._refusal(fresh, allow_volume=False) is not None
        assert det._refusal(vs_shim.app_memo_dir(), allow_volume=True) is not None
        for bad in ("/Volumes/dev/cas/app_state", f"{root}/../x", root, "/tmp/x"):
            try:
                ns["cleanup"](bad)
            except ValueError:
                continue
            raise AssertionError(f"cleanup accepted {bad!r}")
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp).as_posix() + "/determinism"
        ns["scratch_root"] = lambda volume="evals": scratch
        folder = Path(scratch) / "me" / "s1"
        (folder / "llm_memo").mkdir(parents=True)
        (folder / "llm_memo" / "x.json").write_text("{}")
        with _quiet():
            ns["cleanup"](f"{scratch}/me/s1")
        assert not folder.exists() and Path(scratch, "me").exists()


class _CheckMain:
    """app.main as check_determinism drives it: two fixed pages, a product page, no LLM.
    Records the memo knobs each search saw."""

    def __init__(self, page1=(1, 2, 3)):
        self.page1, self.seen = list(page1), []

    def clear_degraded_pools(self):
        pass

    def search(self, query, page, **kw):
        self.seen.append((cfg.LLM_MEMO, cfg.LLM_MEMO_DIR))
        ids = self.page1 if page == 1 else [7, 8]
        return {"results": [{"catalog_index": i} for i in ids], "interpreted_as": ["x"],
                "constraints": [], "bundle_type": None, "facets": {}, "match_quality": {},
                "total_results": 5, "errors": []}

    def product(self, catalog_index, query, recommend):
        return {"recommendations": {"cross_sell": [{"catalog_index": 9}], "upsell": [],
                                    "cross_sell_source": "llm"}, "errors": []}


def _run12(ns, main_module, **kw):
    """12's run() with the REAL check_determinism.run_check driving `main_module`."""
    from eval import check_determinism as det

    real, got = det.run_check, {}

    def run_check(*args, **kwargs):
        got.update(kwargs)
        return real(*args, main_module=main_module, **kwargs)

    calls = []
    with _fake_pyspark(), _patched(det, run_check=run_check), \
            _patched(vs_shim, patch_eval_harness=_fake_harness(calls)), _quiet() as out:
        verdicts = ns["run"](**kw)
    return verdicts, got, calls, out.getvalue()


def test_determinism_12_runs_the_check_on_a_scratch_memo_and_restores_the_knobs():
    ns = _exec_nb(NB12, "nb12")
    stamps = iter(f"2026103{i}-120000" for i in range(9))
    ns["_stamp"] = lambda: next(stamps)
    ns["_me"] = lambda: "me"
    # The notebook's memo ON (the default): patch_eval_harness and check_determinism both
    # leave it OFF, so only 12's own restore can bring it back
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, \
            _patched(cfg, LLM_MEMO=True, LLM_MEMO_DIR=Path(tmp, "app_memo").as_posix()):
        scratch = Path(tmp).as_posix() + "/determinism"
        ns["scratch_root"] = lambda volume="evals": scratch
        before = (cfg.LLM_MEMO, cfg.LLM_MEMO_DIR)
        qs = ["gift for a vintage watch collector", "black dress"]
        fake = _CheckMain()
        verdicts, got, calls, text = _run12(ns, fake, queries=qs, runs=3)
        chosen = f"{scratch}/me/20261030-120000/llm_memo"
        assert calls == [{"spark": "spark", "memo": False}]
        assert got["memo"] is True and got["memo_dir"] == chosen and got["allow_volume"] is True
        assert got["runs"] == 3
        # the check ran with ITS memo on, then the notebook's knobs came back
        assert set(fake.seen) == {(True, chosen)} and len(fake.seen) == 2 * 3 * 2
        assert (cfg.LLM_MEMO, cfg.LLM_MEMO_DIR) == before
        assert [v["status"] for v in verdicts] == ["PASS", "PASS"]
        assert "VERDICT: PASS (2 of 2 queries judged)" in text, text
        assert f"run(memo_dir={chosen!r})" in text and "provider calls 0" in text
        snaps = sorted(Path(scratch, "me", "20261030-120000").glob("snapshots-*.json"))
        assert len(snaps) == 1 and len(json.loads(snaps[0].read_text("utf-8"))) == 2

        # The restart case: run 1 of a new session against the earlier session's run 1
        _, got, _, text = _run12(ns, _CheckMain(), queries=qs, runs=1, memo_dir=chosen)
        assert got["memo_dir"] == chosen and got["allow_volume"] is True
        assert "identical with 0 provider calls: 2/2 compared" in text, text
        assert "VERDICT: PASS" in text
        _, _, _, text = _run12(ns, _CheckMain(page1=(3, 2, 1)), queries=qs, runs=1,
                               memo_dir=chosen)
        assert "identical with 0 provider calls: 0/2 compared" in text and "page1" in text
        assert "VERDICT: FAIL" in text, text

        # memo off: no folder, no --allow-volume, a stability report with no pass mark
        fake = _CheckMain()
        verdicts, got, _, text = _run12(ns, fake, queries=qs[:1], runs=2, memo=False)
        assert got["memo"] is False and got["memo_dir"] is None and got["allow_volume"] is False
        assert set(fake.seen) == {(False, before[1])}
        assert "VERDICT: reported (memo off" in text and "snapshots" not in text
        assert (cfg.LLM_MEMO, cfg.LLM_MEMO_DIR) == before
        for bad in ({"runs": 1}, {"memo_dir": vs_shim.app_memo_dir()}):
            try:
                _run12(ns, _CheckMain(), queries=qs, **bad)
            except ValueError:
                continue
            raise AssertionError(f"run() accepted {bad}")


# ------------------------------------------------------------------ plain runner
if __name__ == "__main__":
    _PLAIN_RUNNER = True
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging

    logging.basicConfig(level=logging.ERROR)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except _Skipped as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except Exception as e:  # noqa: BLE001 -- report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:500]}")
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
