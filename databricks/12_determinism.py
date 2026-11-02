"""Step 12: the determinism check on Databricks -- the same search, the same page.

Why (the Oct 27 meeting: "the same search must give the same results"): Databricks
takes no seed, so on the App page_cache repeats a search only until a restart or an
eviction, and the LLM reply memo (app/llm_memo.py, LLM_MEMO; the App keeps it on the
app_state volume, 09) makes it repeat after that too. This runs
eval/check_determinism.py on the workspace -- SEARCH_BACKEND=databricks through
vs_shim, so gte-large-en Vector Search and the workspace LLM -- and checks it end to
end: per query, `runs` cold runs (page_cache, the degraded store and the /product cache
cleared before each) of app.main.search page 1 and page 2 and app.main.product of the
top card with its recommendations. With the memo on (the default) every run must equal
run 1 -- catalog_index lists, intents, constraints, bundle type, facets, match quality,
ad labels, cross-sell / upsell -- and runs 2..N must make 0 provider calls; the table
lists every field that differs, per run.

What only the workspace can show: the memo keys the rerank prompt, which holds the
retrieved rows, and every run embeds its intents again (databricks-gte-large-en; no
embedding memo this round, owner call D6). A query vector that is not bitwise stable,
or a near-tie Vector Search breaks another way, shows up as provider calls on run 2+
and a FAIL with page1 / page2 among the diffs: the one drift the memo cannot pin.
A rebuilt index (02 pause() / resume()) may legitimately move near-ties, so compare
runs within one build.

memo=False reports the model's raw stability instead (overlap@12, top-4, Kendall tau,
same intents), every run paying its LLM calls.

The check's memo lives in a NEW folder on a scratch volume of your own:
/Volumes/<CAS_CATALOG>/<CAS_SCHEMA>/<volume>/determinism/<you>/<UTC time>/llm_memo,
volume "evals" by default (04-07 and 10 already write their CSVs there; volume= picks
another). Never the App's app_state volume: refused here and by check_determinism,
which gets --allow-volume only for a path built below that scratch folder. The folder
also gets snapshots-<UTC time>.json (page ids, intents, facets per run). Both hold
model output that can echo the queries: cleanup(folder) deletes them.

The restart case: dbutils.library.restartPython(), Cell 2 again, then
run(memo_dir=<the folder printed at the end>): run 1 then replays the earlier session's
replies, so it must make 0 provider calls too, and its pages are compared with the
earlier session's run 1.

Cost, default 21 queries (the meeting's three + the first 18 page_type rows):
- memo on: run 1 ~3 LLM calls a query (translate, rerank, the product page's
  cross-sell) = ~63 calls and ~0.11M tokens, ESTIMATED (~4.9K for translate + rerank:
  the 2026-10-20 live qwen3.8-27b mean was 4.4K, before 2026-10-27's ~0.5K of prompt
  growth; ~0.4K for cross-sell); runs 2..N 0 LLM calls. A run 1 that fell back is
  retried (attempts=2), up to ~3 more calls each. ~5 s per Qwen call: run 1 ~5-6 min.
- memo off: every run pays them: ~190 calls / ~0.33M tokens for 3 runs.
- Every run, memo or not: 1 query-embedding call + 1 Vector Search query per search
  (page 2 comes from page_cache), and the product page up to RECOMMEND_MAX (4) + 1
  more retrievals. The printed totals are the real calls and tokens.

Usage in a notebook (after Cell 2; the index ONLINE; app.main needs FastAPI, see 10):
    exec(open("databricks/12_determinism.py").read())
    verdicts = run()                                    # 21 queries x 3 runs, memo on
    run(queries=["gift for a vintage watch collector"], runs=2)
    run(memo=False, runs=2)                             # the model's raw stability
    run(memo_dir="/Volumes/dev/cas/evals/determinism/<you>/<time>/llm_memo")  # restart
    cleanup("/Volumes/dev/cas/evals/determinism/<you>/<time>")
"""

import sys

sys.path.insert(0, "databricks")
import vs_shim  # noqa: E402

# Where check folders go: a volume the notebook user writes (04-07 and 10 write their
# CSVs to this one), never vs_shim.MEMO_VOLUME_NAME (the App's memo)
SCRATCH_VOLUME = "evals"
SCRATCH_FOLDER = "determinism"
MEMO_SUBDIR = "llm_memo"


def scratch_root(volume=SCRATCH_VOLUME):
    """/Volumes/<catalog>/<schema>/<volume>/determinism (catalog and schema read LIVE
    from cfg): every folder run() makes or reuses sits below it."""
    import re

    import app.config as cfg

    if not re.fullmatch(r"[A-Za-z0-9_]+", str(volume)):
        raise ValueError(f"volume must be a bare volume name, not {volume!r}")
    if str(volume).casefold() == vs_shim.MEMO_VOLUME_NAME:
        raise ValueError(f"{volume} is the App's memo volume: a check must never read or "
                         "pre-warm what the App serves. Use a scratch volume of your own.")
    return f"/Volumes/{cfg.CAS_CATALOG}/{cfg.CAS_SCHEMA}/{volume}/{SCRATCH_FOLDER}"


def _me():
    """The notebook user as a folder name ('me' when the SDK cannot say)."""
    import re

    try:
        from app.vector_search import _workspace_client

        name = _workspace_client().current_user.me().user_name or ""
    except Exception:  # noqa: BLE001 -- a folder name is no reason to fail the check
        name = ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "me"


def _stamp():
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _below(path, root):
    """`path` normalised ('..' resolved), if it lies strictly below `root`; else None."""
    import posixpath

    norm = posixpath.normpath(str(path))
    return norm if norm.startswith(root.rstrip("/") + "/") else None


def check_memo_dir(memo_dir=None, volume=SCRATCH_VOLUME):
    """The memo folder for a check: a new <scratch root>/<you>/<UTC time>/llm_memo, or an
    earlier check's (the restart case), which must sit below the scratch root -- the
    only /Volumes paths this notebook lets check_determinism use (--allow-volume)."""
    root = scratch_root(volume)
    if memo_dir is None:
        return f"{root}/{_me()}/{_stamp()}/{MEMO_SUBDIR}"
    norm = _below(memo_dir, root)
    if norm is None or norm.rsplit("/", 1)[-1] != MEMO_SUBDIR:
        raise ValueError(f"memo_dir must be an llm_memo folder run() made below {root}/ "
                         f"(it printed it), not {memo_dir!r}")
    return norm


def _snapshots(folder):
    """The snapshot files of earlier runs in `folder`, oldest first."""
    import os

    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(f"{folder}/{n}" for n in names
                  if n.startswith("snapshots-") and n.endswith(".json"))


def _restart_report(det, earlier, verdicts):
    """The restart case: this session's run 1 against the earlier session's run 1."""
    import json

    with open(earlier, encoding="utf-8") as f:
        before = {v["query"]: v["runs"][0] for v in json.load(f) if v.get("runs")}
    print(f"\n=== restart: run 1 vs the earlier session's run 1 ({earlier.rsplit('/', 1)[-1]})")
    same = compared = 0
    for v in verdicts:
        now, then = v["runs"][0], before.get(v["query"])
        if then is None or then.get("fell_back") or now.get("fell_back"):
            continue
        compared += 1
        diffs = det.differences(then, now)
        same += not diffs and now["llm_calls"] == 0
        if diffs or now["llm_calls"]:
            print(f"  {v['query'][:46]:46s} run 1 calls={now['llm_calls']} "
                  f"diffs: {', '.join(diffs) or 'none'}")
    print(f"  identical with 0 provider calls: {same}/{compared} compared")
    return {"compared": compared, "identical": same}


def run(queries=None, runs=3, memo=True, memo_dir=None, volume=SCRATCH_VOLUME, attempts=2,
        product=True):
    """eval/check_determinism on Vector Search; prints its table and the totals and
    returns its verdicts. The notebook's LLM_MEMO / LLM_MEMO_DIR come back afterwards."""
    try:
        import fastapi  # noqa: F401 -- app.main needs it
    except ImportError as e:
        raise RuntimeError(
            'FastAPI is missing: %pip install "fastapi==0.136.3" (no --upgrade), then '
            "dbutils.library.restartPython() and run Cell 2 again."
        ) from e
    import json
    import logging
    import os

    from pyspark.sql import SparkSession

    import app.config as cfg
    import app.llm_client as llm_client
    from app import llm_memo
    from eval import check_determinism as det

    if int(runs) < 2 and not memo_dir:
        raise ValueError("runs must be >= 2 (run 1 is what the others are compared with); "
                         "runs=1 only re-checks an earlier memo_dir")
    chosen = check_memo_dir(memo_dir, volume) if memo else None
    folder = chosen.rsplit("/", 1)[0] if chosen else None
    earlier = _snapshots(folder) if (folder and memo_dir) else []
    if isinstance(queries, str):
        queries = [queries]
    qs = det.load_queries(None, list(queries or []))

    # One INFO line per LLM request otherwise (the URL, never the query, but noise)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    saved = (cfg.LLM_MEMO, cfg.LLM_MEMO_DIR)
    try:
        # memo=False here: the header's memo line is the notebook's; the check switches
        # the memo itself, for its own folder only (check_determinism.configure_memo)
        vs_shim.patch_eval_harness(SparkSession.builder.getOrCreate(), memo=False)
        print(f"check   : {len(qs)} queries x {runs} runs, memo "
              + (f"ON -> {chosen} (this check only)" if memo else "off (raw stability)"))
        with llm_client.track_usage() as used:
            verdicts = det.run_check(qs, runs=int(runs), memo=memo, memo_dir=chosen,
                                     attempts=attempts, product=product,
                                     allow_volume=chosen is not None)
    finally:
        cfg.LLM_MEMO, cfg.LLM_MEMO_DIR = saved
        llm_memo.reset()

    failed = det.print_report(verdicts, memo)
    print(f"  provider calls {used['calls']} | tokens {used['total_tokens']:,} "
          f"(prompt {used['prompt_tokens']:,}, completion {used['completion_tokens']:,}) "
          f"| memo hits {used['memo_hits']}")
    if folder:
        if earlier:
            restart = _restart_report(det, earlier[0], verdicts)
            failed = failed or restart["identical"] < restart["compared"]
        os.makedirs(folder, exist_ok=True)
        out = f"{folder}/snapshots-{_stamp()}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(verdicts, f, indent=1, default=str)
        print(f"  snapshots -> {out}")
        print(f"  re-check after a restart: run(memo_dir={chosen!r})")
        print(f"  delete when done:         cleanup({folder!r})")
    judged = sum(not v["status"].startswith("not pinned") for v in verdicts)
    if not memo:
        verdict = "reported (memo off: stability, no pass mark)"
    elif failed:
        verdict = "FAIL"
    else:
        verdict = "PASS" if judged else "NONE (every run 1 fell back)"
    print(f"VERDICT: {verdict} ({judged} of {len(verdicts)} queries judged)")
    return verdicts


def cleanup(folder, volume=SCRATCH_VOLUME):
    """Delete a check folder run() made (its memo and snapshots: model output that can
    echo the queries). Refuses anything outside the scratch root."""
    import shutil

    root = scratch_root(volume)
    norm = _below(folder, root)
    if norm is None:
        raise ValueError(f"cleanup() deletes only folders below {root}/, not {folder!r}")
    shutil.rmtree(norm)
    print(f"deleted {norm}")
