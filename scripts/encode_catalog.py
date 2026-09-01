"""Chunked, resumable catalog encode into the .cache/ path the server expects."""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from app.config import CACHE_DIR, PRODUCTS_CSV
from app.embeddings import embed_text, get_default_embedder
from app.search import DEFAULT_SEARCH_FIELDS, _build_search_text, _cache_key

# Note: don't set torch threads — cpu_count() oversubscribes; the default is faster
CHUNK = 5000  # rows per resumable chunk (~20 min of work each at 300K scale)


def main():
    fields = DEFAULT_SEARCH_FIELDS
    cache_path = CACHE_DIR / f"embeddings_{_cache_key(fields)}.npy"
    if cache_path.exists():
        print(f"Cache already complete: {cache_path.name}", flush=True)
        return

    print(f"Loading catalog from {PRODUCTS_CSV} ...", flush=True)
    df = pd.read_csv(PRODUCTS_CSV)
    texts = _build_search_text(df, fields).tolist()
    n = len(texts)
    n_chunks = -(-n // CHUNK)  # ceil
    tmp_dir = CACHE_DIR / f"_encode_tmp_{_cache_key(fields)}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_name = get_default_embedder().model_name
    print(f"Encoding {n:,} texts with {model_name} in {n_chunks} chunks of "
          f"{CHUNK} (resumable) ...", flush=True)

    t0 = time.perf_counter()
    for i in range(n_chunks):
        chunk_path = tmp_dir / f"chunk_{i:05d}.npy"
        if chunk_path.exists():
            continue  # resume: already encoded in a prior run
        lo, hi = i * CHUNK, min((i + 1) * CHUNK, n)
        vecs = embed_text(texts[lo:hi])
        # write-then-rename (kill-safe); temp name must end in .npy or np.save breaks it
        tmp = tmp_dir / f"chunk_{i:05d}.writing.npy"
        np.save(tmp, vecs)
        tmp.replace(chunk_path)
        done = i + 1
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed else 0
        eta_h = (n_chunks - done) / rate / 3600 if rate else 0
        print(f"  chunk {done}/{n_chunks} done ({hi:,}/{n:,} rows) "
              f"| elapsed {elapsed/60:.1f}m | ETA {eta_h:.1f}h", flush=True)

    print("All chunks encoded; concatenating -> final cache ...", flush=True)
    parts = [np.load(tmp_dir / f"chunk_{i:05d}.npy") for i in range(n_chunks)]
    full = np.concatenate(parts, axis=0)
    assert full.shape[0] == n, f"row mismatch: {full.shape[0]} != {n}"
    # temp name ends in .npy — np.save would otherwise append it and break the rename
    tmp_final = cache_path.parent / (cache_path.stem + ".writing.npy")
    np.save(tmp_final, full)
    tmp_final.replace(cache_path)
    for i in range(n_chunks):
        (tmp_dir / f"chunk_{i:05d}.npy").unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    print(f"\nDONE: {n:,} products, dim={full.shape[1]}, "
          f"cache={cache_path.name}, total {(time.perf_counter()-t0)/3600:.1f}h",
          flush=True)


if __name__ == "__main__":
    main()
