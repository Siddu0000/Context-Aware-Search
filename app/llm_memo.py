"""Persistent memo of ACCEPTED LLM replies: the same request gets the same reply.

Why (Oct 27 meeting: "the same search must give the same results"): with a fake LLM the
pipeline served byte-identical pages run to run, in two processes with different hash
seeds (8 queries incl. a recipe, a setup, sorts, facets and the product page); all the
variation is the LLM's. Groq treats the seed as best effort -- the same translator prompt
gave a different reply on 3 of 3 cold runs (gpt-oss-20b, 2026-10-27; black dress page-1
overlap 0/12) -- Databricks gets no seed at all, and page_cache lives only as long as the
process. So the reply a stage ACCEPTED is kept, keyed by everything that was sent, and an
identical request replays it with 0 provider calls. The deterministic stages then hand
the next LLM stage the same bytes, so its key repeats too: the whole page repeats, after
a restart as well.

Rules, each a reason the memo cannot serve a page worse than a fresh call would:
- The RAW reply text is stored. A hit goes through today's parser and query guard, so a
  later parser or guard fix applies to old entries too.
- A reply is committed only at its stage's accept point (llm_client.accept_reply), never
  a fallback, which would otherwise be pinned for good. A hit the stage now refuses is
  deleted (llm_client.reject_reply), so it is asked again.
- The key is everything sent: MEMO_VERSION, provider, model, base URL, the messages as
  sent, JSON mode, effective temperature, seed (None when not sent) and reasoning format.
  A prompt edit, a model switch or another rerank window is another key; knobs that act
  after the LLM need not be in it.
- The first accepted reply wins (INSERT OR IGNORE; on a volume an exclusive create: a PUT
  with overwrite=False, a hard link on the FUSE mount, where a mount without hard links
  falls back to a check then a rename). commit() says whether the stored reply is OURS:
  when a concurrent request (another thread, App instance or process) stored another
  reply first, llm_client marks the request superseded and main.py caches nothing from
  it, so the next request is rebuilt from the stored winner instead of page_cache keeping
  the loser for the process lifetime.
- A store, read or write failure is a miss and a log line, never a failed search; after
  _TRIP_AFTER consecutive failures (or failed writes, whatever the reads did) the store
  is bypassed for _TRIP_COOL_S (the in-memory front keeps serving and keeping accepted
  replies), so a broken volume does not cost every search its round trips. A store that
  cannot be opened is not re-opened for that long either.
- LLM_MEMO_MAX_AGE_DAYS (0 = never) bounds how long an entry is kept: the replies echo
  shopper text. An older entry is a miss and is deleted when read; SQLite also drops
  them at every start (startup_check) and hourly on a commit; `purge --expired` (and
  `--to-limit`) deletes the ones never read again from a Volume.
- It pins the FIRST accepted sample, good or bad: `python -m app.llm_memo purge --query
  "..."` re-asks one query -- after the API (or App) restarts: a running process keeps the
  purged reply in its in-memory front and the page in page_cache.

Never logged or printed: keys, replies, queries (safety.md). Counts, stage names and
exception TYPES only.

Storage (LLM_MEMO_DIR): a local directory = SQLite (WAL) at <dir>/memo.sqlite; a
/Volumes/<catalog>/<schema>/<volume>/... path = one JSON file per key,
<dir>/<stage>/<k[:2]>/<k>.json, through the FUSE mount where the volume is mounted (a
notebook) and through the Files API where it is not (a Databricks App, whose local disk
and memory are lost on every restart). An in-memory LRU sits in front of either.

Evals opt out (disable_for_eval): they compare models, prompts and temperatures, and a
memo would replay the first sample and report it as the model's behaviour.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Iterator, Optional, Tuple

# Read as cfg.X at call time: evals and the determinism check flip LLM_MEMO in-process
import app.config as cfg

logger = logging.getLogger(__name__)

# Bump to orphan every stored entry at once: for a change in what a stage does with the
# raw reply that the prompt bytes do not show (a prompt edit needs no bump)
MEMO_VERSION = 1

# The stages with an accept point (llm_client.memo_stage); anything else is not memoised
STAGES = ("translate", "rerank", "cross_sell", "assistant")
_PROBE_STAGE = "_probe"
_FRONT_MAX = 2048           # in-memory LRU in front of the store (raw replies, ~1-3 KB)
_TOUCH_EVERY_S = 3600.0     # last_used is kept to the hour: at most 1 write per key per hour
_SWEEP_EVERY_S = 3600.0     # SQLite deletes expired entries at most hourly (on a commit)
_DEFAULT_DIR = ".cache/llm_memo"
# A store that keeps failing (a revoked WRITE VOLUME grant, a Files API outage) would add
# its round trips to every search; after this many failures in a row the store is
# bypassed for _TRIP_COOL_S and the searches ask the LLM, as with LLM_MEMO=false (the
# in-memory front still serves what it holds). The first call after the pause tries the
# store again; one more failure re-trips it. Writes keep their own streak, which only a
# stored entry ends: with READ but no WRITE grant every lookup "succeeds" (a 404 miss)
# and would otherwise reset the count before every failed commit, so it never tripped.
# A store that cannot even be opened is not re-opened for the same pause either
_TRIP_AFTER = 3
_TRIP_COOL_S = 60.0
_WRITE_OPS = frozenset({"write", "touch", "delete", "probe"})
# The start-up probe runs before uvicorn binds its port: a stalled store must not hold it
# (the LLM probe next to it gets 20 s and no retries for the same reason)
_PROBE_TIMEOUT_S = 15.0
_PURGE_NOTE = ("a running API or App keeps purged replies (its in-memory memo and "
               "page_cache) until it restarts")


def enabled() -> bool:
    """Whether a stage's LLM call may be served from (and saved to) the memo now."""
    return bool(getattr(cfg, "LLM_MEMO", False))


def max_age_days() -> float:
    """LLM_MEMO_MAX_AGE_DAYS as a number; 0 (the default) = entries never expire."""
    try:
        days = float(getattr(cfg, "LLM_MEMO_MAX_AGE_DAYS", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return days if days > 0 else 0.0


def _max_age_s() -> Optional[float]:
    """LLM_MEMO_MAX_AGE_DAYS in seconds; None = entries never expire."""
    days = max_age_days()
    return days * 86400 if days > 0 else None


def _expired(created, now: float) -> bool:
    # With an age limit, an entry with no readable creation time cannot be dated, so it
    # counts as expired
    limit = _max_age_s()
    if limit is None:
        return False
    try:
        created = float(created or 0.0)
    except (TypeError, ValueError):
        created = 0.0
    return now - created > limit


def disable_for_eval() -> bool:
    """Every eval harness calls this before its first LLM call and records the result.

    An eval compares models, prompts or temperatures, or measures run-to-run stability
    (compare_temperature): replaying the first accepted sample would report it as the
    model's behaviour on every repeat. Returns the value to record as `llm_memo`."""
    cfg.LLM_MEMO = False
    return False


def location() -> str:
    """LLM_MEMO_DIR as configured (a local directory or a /Volumes/... path)."""
    return str(getattr(cfg, "LLM_MEMO_DIR", "") or _DEFAULT_DIR)


def make_key(provider, base_url, request: dict) -> str:
    """sha256 of everything that decides the reply. `request` is the backend's own
    description of the call (llm_client backends' memo_request): model, messages as sent,
    json_mode, effective temperature, seed (None = not sent), reasoning_format and any
    other sent parameter under `extra` (Anthropic's max_tokens)."""
    fields = [
        MEMO_VERSION, provider, request.get("model"), base_url, request.get("messages"),
        bool(request.get("json_mode")), request.get("temperature"), request.get("seed"),
        request.get("reasoning_format"), request.get("extra"),
    ]
    blob = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def query_sha(query) -> Optional[str]:
    """What `purge --query` matches: sha256 of the case- and space-folded query, never the
    query itself. None when there is no query."""
    if not isinstance(query, str) or not query.strip():
        return None
    return hashlib.sha256(" ".join(query.casefold().split()).encode("utf-8")).hexdigest()


def prompts_digest() -> str:
    """First 12 hex of sha256 over every prompt an LLM stage sends, read LIVE (harnesses
    swap prompts in-process): the seven translator prompts (the legacy general prompt
    included), the rerank template, the assistant prompt and the cross-sell prompt.
    Logged at start-up and recorded by the evals as `prompts_sha`. 2026-10-27: moved here
    from eval/run_eval.py and COMPLEMENTARY_PROMPT added, so every digest changed once."""
    import app.assistant as assistant_module
    import app.recommendations as recommendations_module
    import app.reranker as reranker_module
    import app.translator as translator_module

    texts = [getattr(translator_module, name) for name in (
        "QUERY_EXPANSION_PROMPT", "RECIPE_PROMPT", "HYDE_PROMPT", "HYBRID_PROMPT",
        "OUTFIT_PROMPT", "SETUP_PROMPT", "QUERY_EXPANSION_OUTFIT_PROMPT")]
    texts += [reranker_module.RERANK_PROMPT_TEMPLATE, assistant_module.ASSISTANT_PROMPT,
              recommendations_module.COMPLEMENTARY_PROMPT]
    # A separator no prompt contains: moving text across a boundary changes the digest
    return hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ stores
class _Corrupt(Exception):
    """A stored entry that cannot be read back: a miss, and it is deleted."""


def _is_not_found(exc: Exception) -> bool:
    """A missing object on the Files API (404), as opposed to a refused or failed read."""
    try:
        from databricks.sdk.errors import NotFound
    except ImportError:
        NotFound = None
    if NotFound is not None and isinstance(exc, NotFound):
        return True
    return (isinstance(exc, FileNotFoundError)
            or type(exc).__name__ in {"NotFound", "ResourceDoesNotExist"}
            or getattr(exc, "status_code", None) == 404)


class _SqliteStore:
    """Local store: one SQLite file in WAL mode (the API and an eval process may share it).

    One connection behind a lock: writes are rare next to LLM latency (one per accepted
    reply), reads are a primary-key lookup."""

    kind = "sqlite"

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.location = str(directory / "memo.sqlite")
        self._lock = Lock()
        # autocommit (isolation_level=None): each statement is its own transaction
        self._db = sqlite3.connect(self.location, timeout=10.0, isolation_level=None,
                                   check_same_thread=False)
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS memo (key TEXT PRIMARY KEY, stage TEXT NOT NULL, "
                "raw TEXT NOT NULL, model TEXT, query_sha TEXT, created REAL NOT NULL, "
                "last_used REAL NOT NULL)")
            self._db.execute("CREATE INDEX IF NOT EXISTS memo_last_used ON memo (last_used)")
            # The age sweep deletes by creation time: indexed, so it is no table scan
            self._db.execute("CREATE INDEX IF NOT EXISTS memo_created ON memo (created)")
            # The start-up probe writes here, never into memo: it is no entry
            self._db.execute("CREATE TABLE IF NOT EXISTS probe (id INTEGER PRIMARY KEY, "
                             "token TEXT NOT NULL)")
        self._swept_at = 0.0

    def get_entry(self, stage: str, key: str) -> Optional[Tuple[str, float]]:
        """(raw, created) or None."""
        with self._lock:
            row = self._db.execute("SELECT raw, created FROM memo WHERE key = ?",
                                   (key,)).fetchone()
        if row is None:
            return None
        if not isinstance(row[0], str):
            raise _Corrupt()
        return row[0], row[1]

    def add(self, stage: str, key: str, raw: str, meta: dict) -> Tuple[str, float]:
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO memo (key, stage, raw, model, query_sha, created, "
                "last_used) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, stage, raw, meta.get("model"), meta.get("query_sha"), now, now))
            inserted = cur.rowcount == 1
            row = self._db.execute("SELECT raw, created FROM memo WHERE key = ?",
                                   (key,)).fetchone()
            if inserted:
                self._trim_locked(_max_entries())
                self._sweep_locked(now)
        if row is None:
            raise RuntimeError("memo write did not persist")
        return row[0], row[1]   # the first accepted reply, ours or a concurrent winner's

    def _sweep_locked(self, now: float, force: bool = False) -> int:
        # Expired entries that are never asked for again would otherwise stay for good
        limit = _max_age_s()
        if limit is None or (not force and now - self._swept_at < _SWEEP_EVERY_S):
            return 0
        self._swept_at = now
        cur = self._db.execute("DELETE FROM memo WHERE created < ?", (now - limit,))
        return max(0, cur.rowcount)

    def sweep_expired(self) -> int:
        """Delete every entry older than LLM_MEMO_MAX_AGE_DAYS (the start-up check)."""
        with self._lock:
            return self._sweep_locked(time.time(), force=True)

    def _trim_locked(self, limit: int) -> int:
        # LRU by last_used; the newest entries stay. COUNT first: the sorted DELETE is
        # the expensive part and is needed only once the bound is reached
        count = self._db.execute("SELECT COUNT(*) FROM memo").fetchone()[0]
        excess = count - max(0, int(limit))
        if excess <= 0:
            return 0
        # Oldest first through the last_used index: no sort of the whole table (a
        # (last_used, created) order sorted all 50K rows, ~160 ms per commit at the bound)
        cur = self._db.execute(
            "DELETE FROM memo WHERE key IN (SELECT key FROM memo ORDER BY last_used ASC "
            "LIMIT ?)", (excess,))
        return max(0, cur.rowcount)

    def trim(self, limit: int) -> int:
        with self._lock:
            return self._trim_locked(limit)

    def touch(self, stage: str, key: str, now: float) -> None:
        with self._lock:
            self._db.execute("UPDATE memo SET last_used = ? WHERE key = ?", (now, key))

    def delete(self, stage: str, key: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM memo WHERE key = ?", (key,))

    def records(self) -> Iterator[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT key, stage, model, query_sha, created, last_used, length(raw) "
                "FROM memo").fetchall()
        for key, stage, model, qsha, created, last_used, size in rows:
            yield {"key": key, "stage": stage, "model": model, "query_sha": qsha,
                   "created": created, "last_used": last_used, "bytes": size or 0}

    def probe(self) -> None:
        token = uuid.uuid4().hex
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO probe (id, token) VALUES (1, ?)", (token,))
            row = self._db.execute("SELECT token FROM probe WHERE id = 1").fetchone()
        if row is None or row[0] != token:
            raise RuntimeError("memo probe read back a different value")

    def close(self) -> None:
        with self._lock:
            self._db.close()


class _VolumeStore:
    """One JSON file per key on a UC Volume: <root>/<stage>/<k[:2]>/<k>.json.

    `files` is a databricks-sdk FilesAPI (an App: no /Volumes mount) or None for the FUSE
    mount (a notebook). A file appears whole -- a temp file hard-linked (or renamed) into
    place on FUSE, one PUT on the Files API -- so a partly written entry is never read.
    last_used is not kept here (a write per hit would double the traffic): `created`
    stands in for it in trims."""

    def __init__(self, root: str, files=None):
        self.location = root.rstrip("/")
        self._files = files
        self.kind = "volume-fuse" if files is None else "volume-files-api"
        self._dirs: set = set()          # directories created by this process (Files API)
        self._dirs_lock = Lock()

    def _path(self, stage: str, key: str) -> str:
        return f"{self.location}/{stage}/{key[:2]}/{key}.json"

    def _read(self, path: str) -> Optional[bytes]:
        if self._files is None:
            try:
                with open(path, "rb") as f:
                    return f.read()
            except FileNotFoundError:
                return None
        try:
            resp = self._files.download(path)
        except Exception as e:  # noqa: BLE001 -- a 404 is a miss; anything else is raised
            if _is_not_found(e):
                return None
            raise
        src = resp.contents
        try:
            return src.read()
        finally:
            close = getattr(src, "close", None)
            if close is not None:
                close()

    def _write(self, path: str, body: bytes, overwrite: bool) -> None:
        parent = path.rsplit("/", 1)[0]
        if self._files is None:
            os.makedirs(parent, exist_ok=True)
            tmp = f"{path}.{uuid.uuid4().hex}.tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(body)
                if overwrite:
                    os.replace(tmp, path)
                else:
                    self._create_exclusive(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            return
        with self._dirs_lock:
            known = parent in self._dirs
        if not known:
            # Idempotent (mkdir -p). Whether an upload creates parents is not documented
            # for every SDK version, so it is never relied on
            self._files.create_directory(parent)
            with self._dirs_lock:
                self._dirs.add(parent)
        self._files.upload(path, io.BytesIO(body), overwrite=overwrite)

    @staticmethod
    def _create_exclusive(tmp: str, path: str) -> None:
        """Put the whole `tmp` at `path` only if nothing is there yet (FUSE put-if-absent).

        A hard link is one step that fails when the path exists. An exists() check and
        then a rename are two remote round trips on a Volume: a writer that stored its
        entry between them was overwritten, and both were told the stored reply was
        theirs, so the first writer's process kept serving a reply the volume no longer
        held (Phase D review). Raises FileExistsError (no path: it names the key) when an
        entry is already there; add() then reads the winner back."""
        try:
            os.link(tmp, path)
            return
        except FileExistsError:
            pass
        except OSError as e:
            # A mount that refuses hard links (whether the UC Volume FUSE mount takes them
            # is not verified): the check-then-rename below, whose window stays open there.
            # Logged once, so a notebook run shows which of the two a volume got
            _warn_once("link", e, message=(
                f"LLM memo: the volume refused a hard link ({type(e).__name__}); a "
                "concurrent writer of the same entry can be overwritten (logged once)."))
            if not os.path.exists(path):
                os.replace(tmp, path)
                return
        raise FileExistsError("memo entry already written")

    @staticmethod
    def _decode(data: bytes, key: Optional[str] = None) -> dict:
        try:
            rec = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise _Corrupt() from e
        if not isinstance(rec, dict) or not isinstance(rec.get("raw"), str):
            raise _Corrupt()
        if key is not None and rec.get("key") != key:
            raise _Corrupt()
        return rec

    def get_entry(self, stage: str, key: str) -> Optional[Tuple[str, float]]:
        """(raw, created) or None."""
        data = self._read(self._path(stage, key))
        if data is None:
            return None
        rec = self._decode(data, key)
        return rec["raw"], rec.get("created") or 0.0

    def add(self, stage: str, key: str, raw: str, meta: dict) -> Tuple[str, float]:
        path = self._path(stage, key)
        now = time.time()
        body = json.dumps({"v": MEMO_VERSION, "key": key, "stage": stage, "raw": raw,
                           "model": meta.get("model"), "query_sha": meta.get("query_sha"),
                           "created": now}, ensure_ascii=False).encode("utf-8")
        try:
            # overwrite=False: an entry another request (or App instance) wrote first stays
            self._write(path, body, overwrite=False)
        except Exception as e:  # noqa: BLE001 -- already there (a concurrent winner) or
            refused = e         # failed: the read-back below tells them apart. Any error
            # is read back, not just a recognised conflict: what the SDK raises for an
            # existing file is not verified here, and a conflict taken for a failure would
            # serve our reply while the volume keeps the winner's (correctness 2)
        else:
            if self._files is not None:
                # The Files API refused nothing: the PUT with overwrite=False created the
                # file, so it holds ours. No read-back GET in the request path (an App
                # pays a round trip per call); FUSE reads back, as on a mount without hard
                # links its check-then-rename can race
                return raw, now
            refused = None
        stored = self.get_entry(stage, key)
        if stored is None:
            if refused is not None:
                # Nothing there: the write itself failed. Its own error, so the log names
                # the cause (PermissionError: no WRITE VOLUME grant), not "did not persist"
                raise refused
            raise RuntimeError("memo write did not persist")
        return stored

    def touch(self, stage: str, key: str, now: float) -> None:
        return None

    def delete(self, stage: str, key: str) -> None:
        self._remove(self._path(stage, key))

    def _remove(self, path: str) -> None:
        if self._files is None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            return
        try:
            self._files.delete(path)
        except Exception as e:  # noqa: BLE001
            if not _is_not_found(e):
                raise

    def _entry_paths(self) -> Iterator[str]:
        if self._files is None:
            for top, _dirs, names in os.walk(self.location):
                for name in names:
                    if name.endswith(".json"):
                        yield f"{top}/{name}".replace("\\", "/")
            return
        pending = [self.location]
        while pending:
            directory = pending.pop()
            try:
                entries = list(self._files.list_directory_contents(directory))
            except Exception as e:  # noqa: BLE001
                if _is_not_found(e):
                    continue
                raise
            for entry in entries:
                path = getattr(entry, "path", None) or f"{directory}/{entry.name}"
                if getattr(entry, "is_directory", False):
                    pending.append(path.rstrip("/"))
                elif path.endswith(".json"):
                    yield path

    def records(self) -> Iterator[dict]:
        for path in self._entry_paths():
            rel = path[len(self.location):].strip("/").split("/")
            if not rel or rel[0] == _PROBE_STAGE:
                continue
            data = self._read(path)
            if data is None:
                continue
            try:
                rec = self._decode(data)
            except _Corrupt:
                rec = {"stage": rel[0], "raw": "", "created": 0.0}
            # The directory is the stage (purge --stage deletes by it), whatever the body says
            yield {"key": path.rsplit("/", 1)[-1][:-len(".json")], "stage": rel[0],
                   "model": rec.get("model"), "query_sha": rec.get("query_sha"),
                   "created": rec.get("created") or 0.0, "last_used": rec.get("created") or 0.0,
                   "bytes": len(data), "_path": path}

    def trim(self, limit: int) -> int:
        recs = sorted(self.records(), key=lambda r: (r["last_used"], r["created"]), reverse=True)
        gone = 0
        for rec in recs[max(0, int(limit)):]:
            self._remove(rec["_path"])
            gone += 1
        return gone

    def probe(self) -> None:
        token = uuid.uuid4().hex
        path = f"{self.location}/{_PROBE_STAGE}/{token}.json"
        self._write(path, json.dumps({"token": token}).encode("utf-8"), overwrite=True)
        try:
            data = self._read(path)
            if data is None or json.loads(data.decode("utf-8")).get("token") != token:
                raise RuntimeError("memo probe read back a different value")
        finally:
            self._remove(path)

    def close(self) -> None:
        return None


def _max_entries() -> int:
    return int(getattr(cfg, "LLM_MEMO_MAX_ENTRIES", 50000) or 50000)


def _build_store(where: str):
    """The store LLM_MEMO_DIR names: a /Volumes/... path is a UC Volume (FUSE where it is
    mounted, i.e. a notebook; else the Files API, i.e. a Databricks App), anything else a
    local directory (relative = the project root) holding SQLite."""
    if where.startswith("/Volumes/"):
        parts = where.strip("/").split("/")
        if len(parts) < 4:
            raise ValueError("LLM_MEMO_DIR on a volume needs /Volumes/<catalog>/<schema>/<volume>")
        if os.path.isdir("/" + "/".join(parts[:4])):
            return _VolumeStore(where)
        from app.vector_search import _workspace_client  # one SDK client, one token cache

        return _VolumeStore(where, files=_workspace_client().files)
    path = Path(where)
    return _SqliteStore(path if path.is_absolute() else cfg.PROJECT_ROOT / path)


# ------------------------------------------------------------------ the memo
_warned: set = set()
_warned_lock = Lock()


def _warn_once(op: str, exc: Exception, message: Optional[str] = None) -> None:
    # The exception TYPE only: a message can carry a path or a reply fragment
    name = type(exc).__name__
    with _warned_lock:
        if (op, name) in _warned:
            return
        _warned.add((op, name))
    if message is not None:
        logger.warning(message)
        return
    logger.warning("LLM memo %s failed (%s); searching without it (logged once per type).",
                   op, name)


class Memo:
    """A store with an in-memory LRU in front, a breaker that bypasses a failing store,
    and per-process counters (/healthz)."""

    def __init__(self, store):
        self.store = store
        # key -> [raw, touched_at, created]: created dates the entry for the age limit
        self._front: "OrderedDict[str, list]" = OrderedDict()
        self._lock = Lock()
        self.counts = {"hits": 0, "misses": 0, "commits": 0, "discards": 0, "errors": 0,
                       "expired": 0, "bypassed": 0, "trips": 0}
        self._failures = 0           # consecutive store failures, any operation
        self._write_failures = 0     # consecutive failed writes; only a stored entry ends it
        self._open_until = 0.0       # time.monotonic() until which the store is bypassed

    def _count(self, name: str) -> None:
        with self._lock:
            self.counts[name] += 1

    def _remember(self, key: str, raw: str, touched: float, created: float) -> None:
        with self._lock:
            self._front[key] = [raw, touched, created]
            self._front.move_to_end(key)
            while len(self._front) > _FRONT_MAX:
                self._front.popitem(last=False)

    # ---------------------------------------------------------------- the breaker
    def _bypassed(self) -> bool:
        """True while the store is tripped; counted, so /healthz shows the skipped calls."""
        with self._lock:
            if time.monotonic() < self._open_until:
                self.counts["bypassed"] += 1
                return True
            return False

    def _ok(self, stored: bool = False) -> None:
        """A store call worked. `stored`: an entry was written (commit), the only success
        that proves the store takes writes -- a 404 read, a Volume's no-op touch or the
        delete of a missing file does not."""
        with self._lock:
            self._failures = 0
            if stored:
                self._write_failures = 0

    def _error(self, op: str, exc: Exception) -> None:
        _warn_once(op, exc)
        with self._lock:
            self.counts["errors"] += 1
            self._failures += 1
            if op in _WRITE_OPS:
                self._write_failures += 1
            trip = self._failures >= _TRIP_AFTER or self._write_failures >= _TRIP_AFTER
            if trip:
                self._open_until = time.monotonic() + _TRIP_COOL_S
                self.counts["trips"] += 1
        if trip:
            _warn_once("store", RuntimeError(), message=(
                f"LLM memo store bypassed for {_TRIP_COOL_S:.0f} s after {_TRIP_AFTER} "
                "consecutive errors or failed writes; retried after each pause (logged "
                "once)."))

    # ---------------------------------------------------------------- the calls
    def lookup(self, stage: str, key: str) -> Optional[str]:
        now = time.time()
        with self._lock:
            cached = self._front.get(key)
            if cached is not None and _expired(cached[2], now):
                del self._front[key]
                expired, cached = True, None
            else:
                expired = False
            if cached is not None:
                self._front.move_to_end(key)
                stale = now - cached[1] > _TOUCH_EVERY_S
                if stale:
                    cached[1] = now
                self.counts["hits"] += 1
                raw = cached[0]
        if expired:
            self._count("expired")
            self.discard(stage, key)
            self._count("misses")
            return None
        if cached is not None:
            if stale:
                self._touch(stage, key, now)
            return raw
        if self._bypassed():
            self._count("misses")
            return None
        try:
            entry = self.store.get_entry(stage, key)
        except _Corrupt:
            self._ok()
            self.discard(stage, key)
            entry = None
        except Exception as e:  # noqa: BLE001 -- an unreadable store is a miss
            self._error("read", e)
            entry = None
        else:
            self._ok()
        if entry is not None and _expired(entry[1], now):
            # Past LLM_MEMO_MAX_AGE_DAYS: asked again, and the old reply goes
            self._count("expired")
            self.discard(stage, key)
            entry = None
        if entry is None:
            self._count("misses")
            return None
        raw, created = entry
        self._remember(key, raw, now, created)
        self._touch(stage, key, now)
        self._count("hits")
        return raw

    def _touch(self, stage: str, key: str, now: float) -> None:
        if self._bypassed():
            return
        try:
            self.store.touch(stage, key, now)
        except Exception as e:  # noqa: BLE001
            self._error("touch", e)
        else:
            self._ok()

    def commit(self, stage: str, key: str, raw: str, model=None, query=None) -> bool:
        """Keep `raw`; True unless ANOTHER reply for this key was stored first (the first
        accepted reply wins, and that one is what every later request is served). A store
        that is failing or bypassed keeps the reply in the in-memory front only: nothing
        better is known, so True."""
        now = time.time()
        if self._bypassed():
            self._remember(key, raw, now, now)
            return True
        meta = {"model": model, "query_sha": query_sha(query)}
        try:
            stored, created = self.store.add(stage, key, raw, meta)
        except Exception as e:  # noqa: BLE001 -- never fails the search that produced it
            self._error("write", e)
            self._remember(key, raw, now, now)
            return True
        self._ok(stored=True)
        self._remember(key, stored, now, created)
        self._count("commits")
        return stored == raw

    def discard(self, stage: str, key: str) -> None:
        with self._lock:
            self._front.pop(key, None)
        if self._bypassed():
            return
        try:
            self.store.delete(stage, key)
        except Exception as e:  # noqa: BLE001
            self._error("delete", e)
            return
        self._ok()
        self._count("discards")

    def close(self) -> None:
        try:
            self.store.close()
        except Exception:  # noqa: BLE001
            pass


_memo: Optional[Memo] = None
_memo_for: Optional[str] = None
# (LLM_MEMO_DIR, time.monotonic() until, exception) of the last failed open: not retried
# for _TRIP_COOL_S, as a tripped store is not (opening a volume store builds an SDK client,
# whose credential lookup can be slow, on every lookup / commit / discard otherwise)
_open_failed: Optional[tuple] = None
_build_lock = Lock()


def get_memo() -> Memo:
    """The memo for the CURRENT LLM_MEMO_DIR; rebuilt when it changes (the determinism
    check and the tests point it at a temp dir in-process). Raises if the store cannot be
    opened (again, with no new attempt, for _TRIP_COOL_S after a failed open); the module
    functions below turn that into a miss."""
    global _memo, _memo_for, _open_failed
    where = location()
    with _build_lock:
        if _memo is None or _memo_for != where:
            if _memo is not None:
                _memo.close()
                _memo = None
            failed = _open_failed
            if failed is not None and failed[0] == where and time.monotonic() < failed[1]:
                raise failed[2].with_traceback(None)   # no traceback piling up per raise
            try:
                store = _build_store(where)
            except Exception as e:
                _open_failed = (where, time.monotonic() + _TRIP_COOL_S, e)
                raise
            _open_failed = None
            _memo, _memo_for = Memo(store), where
        return _memo


def reset() -> None:
    """Drop the open store, the in-memory front and a remembered failed open (tests, the
    determinism check)."""
    global _memo, _memo_for, _open_failed
    with _build_lock:
        if _memo is not None:
            _memo.close()
        _memo, _memo_for, _open_failed = None, None, None


def lookup(stage: str, key: str) -> Optional[str]:
    """The stored raw reply for `key`, or None (a miss, an unreadable or unopenable store)."""
    try:
        memo = get_memo()
    except Exception as e:  # noqa: BLE001
        _warn_once("open", e)
        return None
    return memo.lookup(stage, key)


def commit(stage: str, key: str, raw: str, *, model=None, query=None) -> bool:
    """Keep an ACCEPTED reply; silent on failure. `query` only feeds `query_sha`. False
    when another reply for the key was stored first (the caller's reply lost the race)."""
    try:
        memo = get_memo()
    except Exception as e:  # noqa: BLE001
        _warn_once("open", e)
        return True
    return memo.commit(stage, key, raw, model=model, query=query)


def discard(stage: str, key: str) -> None:
    """Forget an entry a stage refused or that no longer parses: the next call re-asks."""
    try:
        memo = get_memo()
    except Exception as e:  # noqa: BLE001
        _warn_once("open", e)
        return
    memo.discard(stage, key)


def counters() -> dict:
    """This process's hits / misses / commits / discards / errors (empty before first use)."""
    memo = _memo
    if memo is None:
        return {}
    with memo._lock:
        return dict(memo.counts)


def backend_kind() -> str:
    """'off', or the store kind LLM_MEMO_DIR selects (without opening it)."""
    if not enabled():
        return "off"
    where = location()
    if where.startswith("/Volumes/"):
        parts = where.strip("/").split("/")
        mounted = len(parts) >= 4 and os.path.isdir("/" + "/".join(parts[:4]))
        return "volume-fuse" if mounted else "volume-files-api"
    return "sqlite"


def startup_check(timeout: Optional[float] = None) -> dict:
    """Write and read back a probe object in the configured store. Never raises, and
    gives up after `timeout` seconds (default _PROBE_TIMEOUT_S): it runs before uvicorn
    binds its port, and a stalled Files API (the SDK's own timeouts and retries) would
    otherwise hold the App's start-up. A timed-out probe is FAILED; its worker thread is
    left to finish on its own (a daemon: it never blocks the exit).

    Without it a missing WRITE VOLUME grant or an unwritable .cache loses determinism
    silently: every search would just ask the LLM again. A failure also counts toward
    the store's breaker."""
    if not enabled():
        return {"ok": True, "enabled": False, "detail": "off (LLM_MEMO=false)"}
    limit = _PROBE_TIMEOUT_S if timeout is None else float(timeout)
    outcome: dict = {}

    def probe() -> None:
        try:
            memo = get_memo()
            outcome["memo"] = memo
            memo.store.probe()
            outcome["kind"] = memo.store.kind
            sweep = getattr(memo.store, "sweep_expired", None)
            if sweep is not None:
                # SQLite: entries past LLM_MEMO_MAX_AGE_DAYS go at every start (an indexed
                # DELETE); a Volume store has no index to do it cheaply: purge --expired
                sweep()
        except Exception as e:  # noqa: BLE001 -- reported in the log and /healthz
            outcome["error"] = e

    worker = threading.Thread(target=probe, name="llm-memo-probe", daemon=True)
    worker.start()
    worker.join(limit)
    memo = outcome.get("memo")
    if worker.is_alive():
        if memo is not None:
            memo._error("probe", TimeoutError())
        return {"ok": False, "enabled": True, "location": location(),
                "detail": f"TimeoutError: no answer from the memo store in {limit:.0f} s"}
    if "error" in outcome:
        e = outcome["error"]
        if memo is not None:
            memo._error("probe", e)
        return {"ok": False, "enabled": True, "location": location(),
                "detail": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"ok": True, "enabled": True, "backend": outcome["kind"], "location": location(),
            "detail": outcome["kind"]}


# ------------------------------------------------------------------ CLI
def _summary(records: list) -> dict:
    now = time.time()
    by_stage: dict = {}
    for r in records:
        by_stage[r.get("stage") or "?"] = by_stage.get(r.get("stage") or "?", 0) + 1
    created = [r["created"] for r in records if r.get("created")]
    return {
        "entries": len(records),
        "by_stage": dict(sorted(by_stage.items())),
        "kbytes": round(sum(r.get("bytes", 0) for r in records) / 1024, 1),
        "oldest_days": round((now - min(created)) / 86400, 2) if created else None,
        "newest_days": round((now - max(created)) / 86400, 2) if created else None,
        "max_age_days": max_age_days() or None,
        # Past LLM_MEMO_MAX_AGE_DAYS: never served again; `purge --expired` deletes them
        "expired": sum(1 for r in records if _expired(r.get("created"), now)),
    }


def _remove_records(store, records: list, done: dict) -> None:
    # Counted one by one: a purge the store stops halfway reports what it already removed
    for r in records:
        if "_path" in r:
            store._remove(r["_path"])
        else:
            store.delete(r.get("stage") or "", r["key"])
        done["removed"] += 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m app.llm_memo",
        description="Inspect or prune the LLM reply memo (LLM_MEMO_DIR). Prints counts "
                    "only: never keys, replies or queries.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats", help="entries per stage, size, age")
    purge = sub.add_parser(
        "purge", help="delete entries so their prompts are asked again",
        description="Delete entries so their prompts are asked again. A running API or "
                    "App keeps serving what it already holds (its in-memory memo and "
                    "page_cache): restart it for a purge to take effect there.")
    purge.add_argument("--stage", choices=STAGES)
    purge.add_argument("--older-than-days", type=float, default=None)
    purge.add_argument("--query", default=None,
                       help="every entry a search for this exact query committed "
                            "(case and spacing ignored)")
    purge.add_argument("--expired", action="store_true",
                       help="every entry older than LLM_MEMO_MAX_AGE_DAYS (none when 0)")
    purge.add_argument("--to-limit", action="store_true",
                       help="drop expired entries, then trim to LLM_MEMO_MAX_ENTRIES, least "
                            "recently used first (the size bound of a Volume store)")
    purge.add_argument("--all", action="store_true", help="every entry")
    args = p.parse_args(argv)

    store = _build_store(location())
    done = {"removed": 0}
    try:
        if args.cmd == "stats":
            records = list(store.records())
            out = {"backend": store.kind, "location": location(),
                   "max_entries": _max_entries(), **_summary(records)}
            print(json.dumps(out, indent=2))
            return 0
        filters = (args.stage, args.older_than_days, args.query)
        if (not args.all and all(f is None for f in filters) and not args.to_limit
                and not args.expired):
            p.error("purge needs --stage, --older-than-days, --query, --expired, --to-limit "
                    "or --all")
        if args.all or any(f is not None for f in filters):
            cutoff = (time.time() - args.older_than_days * 86400
                      if args.older_than_days is not None else None)
            wanted = query_sha(args.query) if args.query is not None else None
            doomed = [r for r in store.records()
                      if (args.stage is None or r.get("stage") == args.stage)
                      and (cutoff is None or (r.get("last_used") or 0) < cutoff)
                      and (args.query is None or r.get("query_sha") == wanted)]
            _remove_records(store, doomed, done)
        if (args.expired or args.to_limit) and _max_age_s() is not None:
            now = time.time()
            _remove_records(store, [r for r in store.records()
                                    if _expired(r.get("created"), now)], done)
        if args.to_limit:
            done["removed"] += store.trim(_max_entries())
        print(json.dumps({"backend": store.kind, "removed": done["removed"],
                          "note": _PURGE_NOTE}))
        return 0
    except Exception as e:  # noqa: BLE001 -- reported below, as its TYPE only
        # A refused read or delete (READ but no WRITE VOLUME, a FUSE I/O error) names the
        # entry file, <stage>/<k[:2]>/<k>.json, in its message, and an uncaught traceback
        # would print that key: the exception type only, as on every other memo error path
        out = {"backend": store.kind, "failed": args.cmd, "error": type(e).__name__}
        if args.cmd == "purge":
            out["removed"] = done["removed"]
            out["note"] = "entries removed before the error stay removed; " + _PURGE_NOTE
        print(json.dumps(out), file=sys.stderr)
        return 1
    finally:
        store.close()
        reset()   # an open memo in this process would hold the removed entries in memory


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
