"""One-time compaction of the embedding text for rows the local embedder would cut.

    python -m scripts.compact_search_text build [--dry-run]   # ~10 min, no model encode
    python -m scripts.compact_search_text encode              # re-encodes CHANGED rows only

`build` tokenizes every row's search_text with the embedder's own tokenizer, rewrites only
the rows over its limit (app/compact_text.py: extractive rules, no LLM), checks each fits,
prints retention numbers (counts only, never product text), and writes a sidecar next to
the catalog -- `<catalog>.compact_v<N>.csv` (catalog_index, parent_asin, embed_text) and
`.meta.json` (written last, with the sidecar's md5) -- so the catalog CSV, its hash and
the default embedding cache stay as they are. `encode` checks the sidecar against the
catalog, copies the default cache's vectors and re-encodes only the changed rows, in
resumable chunks named by the run (the sidecar's md5, the model, its length), into a
short-named compact cache plus a stamp recording that md5; leftovers of other runs are
deleted, with a line saying so. The API uses them only with SEARCH_TEXT_COMPACT=true
(app/config.py), every file present and the stamp matching; otherwise it serves the
default text and vectors, with a log line.

Local backend only: Databricks' gte-large-en reads 8,192 tokens, so nothing is cut there.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import compact_text as ct  # noqa: E402
from app import search as S  # noqa: E402
from app.config import CACHE_DIR, PRODUCTS_CSV  # noqa: E402
from app.embeddings import embed_text, get_default_embedder  # noqa: E402

CHUNK = 5000  # rows per resumable encode chunk


def _load():
    """The catalog and its default search_text exactly as search.load_index builds them."""
    df = pd.read_csv(PRODUCTS_CSV, low_memory=False)
    fields = S.effective_fields(df.columns)
    default = S._build_search_text(df, fields)          # fills missing fields with '' in place
    return df, fields, default, S._file_hash(PRODUCTS_CSV)


def _embedder():
    embedder = get_default_embedder()
    if embedder.is_openai:
        raise SystemExit("compact_search_text needs a sentence-transformers EMBEDDING_MODEL.")
    return embedder, embedder.max_seq_length()


def _counts(tok, texts, batch=4096):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], add_special_tokens=False, truncation=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return out


def _kept_words(tok, texts, budget, batch=2048):
    """Content-word sets of what the embedder reads of each text (its first `budget`
    tokens): the BEFORE side of the retention numbers."""
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], add_special_tokens=False, truncation=True,
                  max_length=budget)
        out.extend(set(ct.content_words(tok.decode(ids))) for ids in enc["input_ids"])
    return out


def _covered(value: str, words: set) -> bool:
    need = set(ct.content_words(value))
    return bool(need) and need <= words


def _share(value: str, words: set):
    need = set(ct.content_words(value))
    return len(need & words) / len(need) if need else None


def _replace(src: Path, dst: Path, tries: int = 20, wait: float = 0.5) -> None:
    """os.replace, retried for ~10 s: on Windows a file another process has open (an API
    boot hashing the sidecar, a notebook reading it) cannot be replaced until it lets go."""
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise SystemExit(f"Could not replace {dst.name}: another process has it open "
                                 f"(close it and re-run; the new file is {src.name}).")
            time.sleep(wait)


def _write_atomic(path: Path, write) -> None:
    """write(tmp_path), then rename onto `path`: a stopped build leaves no half file."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    _replace(tmp, path)


def build(dry_run: bool) -> None:
    t0 = time.perf_counter()
    df, fields, default, csv_hash = _load()
    embedder, max_len = _embedder()
    tok = embedder._st_model.tokenizer
    budget = max_len - 2                                # [CLS] and [SEP]
    print(f"catalog {PRODUCTS_CSV.name} ({csv_hash}), {len(df):,} rows; {embedder.model_name} "
          f"reads {max_len} tokens -> budget {budget}", flush=True)

    default_list = default.tolist()
    counts = _counts(tok, default_list)
    rows = ct.changed_rows(counts, budget)
    print(f"over budget (cut today): {len(rows):,} rows ({len(rows) / len(df):.1%})", flush=True)

    count = lru_cache(maxsize=1 << 18)(
        lambda s: len(tok(s, add_special_tokens=False, truncation=False)["input_ids"]))
    records = df.to_dict("records")
    compact = [ct.compact_text(records[i], count, budget) for i in rows]
    after = _counts(tok, compact)
    over = sum(n > budget for n in after)
    print(f"compacted in {time.perf_counter() - t0:.0f}s; rows still over budget: {over}",
          flush=True)

    # Retention on the changed rows: what the embedder reads today vs after
    before_words = _kept_words(tok, [default_list[i] for i in rows], budget)
    after_words = [set(ct.content_words(t)) for t in compact]
    print("\nfield retained in what the embedder reads (changed rows that have the field):")
    print(f"  {'field':<17}{'rows':>9}{'today':>9}{'compact':>9}")
    for field in ("Product_title",) + ct.TAG_FIELDS:
        have = [(i, str(records[r].get(field) or "")) for i, r in enumerate(rows)]
        have = [(i, v) for i, v in have if ct.content_words(v)]
        skipped = 0
        if field == "color":                    # 'default', '14': skipped on purpose
            real = [(i, v) for i, v in have if not ct.is_placeholder_colour(v)]
            skipped, have = len(have) - len(real), real
        if not have:
            continue
        b = sum(_covered(v, before_words[i]) for i, v in have) / len(have)
        a = sum(_covered(v, after_words[i]) for i, v in have) / len(have)
        note = f"  (+{skipped:,} placeholder values, skipped on purpose)" if skipped else ""
        print(f"  {field:<17}{len(have):>9,}{b:>9.1%}{a:>9.1%}{note}")
    for field in ("specs", "prod_description"):
        b_s, a_s = [], []
        for i, r in enumerate(rows):
            value = ct._blank(records[r].get(field))
            if field == "specs":                # the VALUES: keys are dropped on purpose
                value = " ".join(v.partition(":")[2] for v in value.split("; ") if ":" in v)
            sb, sa = _share(value, before_words[i]), _share(value, after_words[i])
            if sb is not None:
                b_s.append(sb)
                a_s.append(sa)
        if b_s:
            label = "spec values" if field == "specs" else "description"
            print(f"  {label:<17}{len(b_s):>9,}{np.mean(b_s):>9.1%}{np.mean(a_s):>9.1%}"
                  "  (mean share of its distinct words)")
    if rows:
        print(f"\nmean tokens of changed rows: today {np.mean([counts[i] for i in rows]):.0f} "
              f"(embedder reads {budget}), compact {np.mean(after):.0f}")

    if dry_run:
        print("\n--dry-run: nothing written")
        return
    if over:
        raise SystemExit(f"{over} compacted rows are still over budget: nothing written")
    side, meta_path = ct.sidecar_paths(PRODUCTS_CSV)
    old_meta = meta_path.read_bytes() if meta_path.exists() else None
    meta_path.unlink(missing_ok=True)       # no meta may name a sidecar being rewritten
    frame = pd.DataFrame({
        "catalog_index": rows,
        "parent_asin": S.asin_keys(df)[rows],
        "embed_text": compact,
    })
    try:
        _write_atomic(side, lambda p: frame.to_csv(p, index=False))
    except BaseException:
        if old_meta is not None:            # the old sidecar is still in place: so is its meta
            meta_path.write_bytes(old_meta)
        raise
    meta = {"version": ct.COMPACT_VERSION, "csv_hash": csv_hash, "n_rows": len(df),
            "model": embedder.model_name, "max_seq_length": max_len, "budget": budget,
            "fields": list(fields), "n_changed": len(rows),
            "sidecar_md5": ct.file_md5(side)}
    _write_atomic(meta_path, lambda p: p.write_text(json.dumps(meta, indent=1), encoding="utf-8"))
    print(f"\nwrote {side.name} ({len(rows):,} rows) and {meta_path.name}")


def _run_key(sidecar_md5: str, out_path: Path, max_len) -> str:
    """What the encoded vectors depend on: the sidecar's contents, the compact cache's
    name (embedding model, catalog hash, field list) and the embedder's length. Another
    model with the same tokenizer writes a byte-identical sidecar: its run must not
    resume from the first model's parts."""
    return hashlib.md5(f"{sidecar_md5}|{out_path.name}|{max_len}".encode()).hexdigest()[:12]


def _parts_dir(run_key: str) -> Path:
    """Resumable chunks, named by the run and the chunk size: a rebuilt sidecar, another
    model or another CHUNK never resumes from parts of a different run. Short on purpose
    (Windows' 260-character path limit)."""
    return CACHE_DIR / f"_enc_{run_key}_{CHUNK}"


def _staging(run_key: str) -> Path:
    return CACHE_DIR / f"_enc_{run_key}.w.npy"     # np.save needs the .npy suffix


def _remove_stale(keep=()) -> None:
    """Delete encode leftovers of other runs (parts folders, ~185 MB when nearly done; a
    staging file, ~460 MB), saying what went: nothing else ever reads them."""
    keep = {Path(p).name for p in keep}
    for p in sorted(CACHE_DIR.glob("_enc_*")):
        if p.name in keep:
            continue
        files = list(p.iterdir()) if p.is_dir() else [p]
        size = sum(f.stat().st_size for f in files if f.is_file())
        for f in files:
            f.unlink(missing_ok=True)
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass
        print(f"removed stale {p.name} ({size / 1e6:.0f} MB)")


def encode() -> None:
    df, fields, _default, csv_hash = _load()
    embedder, max_len = _embedder()
    side, meta_path = ct.sidecar_paths(PRODUCTS_CSV)
    if not side.exists() or not meta_path.exists():
        raise SystemExit("Run `build` first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    why = ct.sidecar_matches(meta, csv_hash, len(df), embedder.model_name, max_len, fields)
    side_md5 = ct.file_md5(side)
    if why is None and side_md5 != meta.get("sidecar_md5"):
        why = "the sidecar changed after its meta was written"
    side_df = pd.read_csv(side, keep_default_na=False,
                          dtype={"parent_asin": str, "embed_text": str})
    idx = side_df["catalog_index"].to_numpy(dtype=np.int64)
    why = why or ct.check_rows(meta, idx, side_df["parent_asin"].to_numpy(), S.asin_keys(df), len(df))
    if why:
        raise SystemExit(f"The sidecar does not fit this catalog / embedder ({why}): re-run `build`.")

    default_cache = CACHE_DIR / f"embeddings_{S._cache_key(fields, csv_hash)}.npy"
    out_path = ct.compact_cache_path(default_cache)
    stamp = ct.stamp_path(out_path)
    stamp_body = {"sidecar_md5": side_md5, "model": embedder.model_name,
                  "max_seq_length": max_len, "n_changed": len(idx)}
    if out_path.exists() and stamp.exists():
        if json.loads(stamp.read_text(encoding="utf-8")) == stamp_body:
            _remove_stale()                 # a run stopped inside its own cleanup
            print(f"Already complete: {out_path.name}")
            return
    stamp.unlink(missing_ok=True)           # vectors of another sidecar are never served
    if not default_cache.exists():
        raise SystemExit(f"Default cache missing ({default_cache.name}): start the API once "
                         "(or run scripts/encode_catalog.py) so the unchanged rows have vectors.")

    texts = side_df["embed_text"].tolist()
    n_chunks = -(-len(texts) // CHUNK)
    run = _run_key(side_md5, out_path, max_len)
    parts, staging = _parts_dir(run), _staging(run)
    _remove_stale(keep=(parts, staging))
    parts.mkdir(parents=True, exist_ok=True)
    print(f"Encoding {len(texts):,} changed rows of {len(df):,} with {embedder.model_name} "
          f"in {n_chunks} resumable chunks ...", flush=True)
    t0, fresh = time.perf_counter(), 0
    for i in range(n_chunks):
        lo, hi = i * CHUNK, min((i + 1) * CHUNK, len(texts))
        part = parts / f"p{i:05d}.npy"
        if part.exists() and np.load(part, mmap_mode="r").shape[0] == hi - lo:
            continue
        vecs = embed_text(texts[lo:hi])
        tmp = parts / f"p{i:05d}.w.npy"     # np.save needs the .npy suffix
        np.save(tmp, vecs)
        _replace(tmp, part)
        fresh += 1
        rate = fresh / (time.perf_counter() - t0)
        print(f"  chunk {i + 1}/{n_chunks} | elapsed {(time.perf_counter() - t0) / 60:.1f}m"
              f" | ETA {(n_chunks - i - 1) / rate / 3600:.1f}h", flush=True)

    vectors = np.load(default_cache)
    if vectors.shape[0] != len(df):
        raise SystemExit("The default cache does not have one vector per catalog row.")
    new = (np.concatenate([np.load(parts / f"p{i:05d}.npy") for i in range(n_chunks)])
           if n_chunks else vectors[idx])      # nothing over budget: the default vectors
    if new.shape[0] != len(idx):
        raise SystemExit("Encoded parts do not match the sidecar's rows: delete "
                         f"{parts.name} and re-run `encode`.")
    vectors[idx] = new
    np.save(staging, vectors)
    _replace(staging, out_path)
    _write_atomic(stamp, lambda p: p.write_text(json.dumps(stamp_body, indent=1), encoding="utf-8"))
    for i in range(n_chunks):
        (parts / f"p{i:05d}.npy").unlink(missing_ok=True)
    try:
        parts.rmdir()
    except OSError:
        pass
    print(f"DONE: {out_path.name} ({(time.perf_counter() - t0) / 3600:.1f}h). "
          "Set SEARCH_TEXT_COMPACT=true and restart the API to serve it.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="compact the rows the embedder cuts; write the sidecar")
    b.add_argument("--dry-run", action="store_true", help="print the numbers, write nothing")
    sub.add_parser("encode", help="re-encode the changed rows into the compact cache")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    build(args.dry_run) if args.cmd == "build" else encode()


if __name__ == "__main__":
    main()
