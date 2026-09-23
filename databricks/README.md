# Databricks migration — runbook

Decided on the 2026-08-13 call (Databricks = non-negotiable priority) and
sequenced on the 2026-08-24 call: migrate the FOUNDATIONAL, already-tested
solution, while enhancements keep being developed locally. Component mapping
verified against Databricks docs on 2026-07-29 — see `docs/RESEARCH.md` §3.
Measured results are recorded in CLAUDE.md ("Databricks migration — VERIFIED
RESULTS").

## How the code runs here

`SEARCH_BACKEND=databricks` switches `app/search.py` to Mosaic AI Vector
Search. Every consumer — `main.py`, recommendations, the keyword engine and the
whole eval suite — goes through that module, so **the eval exercises exactly
the code the deployed app runs**. The index returns only `catalog_index` +
score; full rows are hydrated from the catalog, so both backends return
identical row shapes (images, `parent_asin` for sponsored ads, etc.).

| Script | Does | LLM calls | Billing |
|---|---|---|---|
| `01_catalog_to_delta.py` | CSV → Delta table, CDF on, **dense** `catalog_index` | 0 | compute minutes |
| `02_vector_index.py` | endpoint + delta-sync index (`databricks-gte-large-en`) | 0 | **endpoint bills hourly from here** |
| `03_smoke_test.py` | one similarity query + one JSON-mode LLM call, both asserted | 1 | — |
| `04_run_eval.py` | retrieval eval; `--rerank` for the full pipeline | 18 / 36 | — |
| `05_compare_llms.py` | `list_models()`, `run([...])` — intent gen + rerank; `control()` = no-LLM baseline | 36 / 32 per model (keyword / context); control 0 | — |
| `06_compare_translators.py` | query_expansion vs hyde vs hybrid, rerank off | 18 / 16 per mode (keyword / context) | — |
| `07_calibrate_threshold.py` | calibrates `MIN_RESULT_RELEVANCE` for the VS score scale | ~28 keyword / ~26 context | — |
| `08_llm_smoke.py` | LLM-only acceptance check: every LLM call site once with the configured model (translator + bundles, rerank, chat assistant, cross-sell, injection x3), asserting no fallback, no JSON-mode/temperature downgrade | 25 | — (**no index needed**) |
| `vs_shim.py` | shared setup: switch backend, results → UC Volume, MLflow on, prints the live `llm :` | — | — |

## The one-window run (do it all in one sitting)

The initial index snapshot takes **~2 hours for 300K rows** (~2,350 rows/min)
and the endpoint bills the whole time. Batch every eval into that window.

**Notebook setup** (serverless). Cell 1:
```python
%pip install databricks-vectorsearch openai python-dotenv scikit-learn mlflow
dbutils.library.restartPython()
```
Cell 2 — env, including MLflow:
```python
import os, sys
REPO = "/Workspace/Users/<you>/CAS - Sai/Context-Aware-Search"
sys.path.insert(0, REPO); os.chdir(REPO)
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
TOKEN = ctx.apiToken().get()
HOST = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ.update({
    "CAS_CATALOG": "dev", "CAS_SCHEMA": "cas",
    "DATABRICKS_TOKEN": TOKEN, "DATABRICKS_HOST": HOST,
    "LLM_PROVIDER": "openai", "OPENAI_API_KEY": TOKEN,
    "OPENAI_BASE_URL": HOST + "/serving-endpoints",
    # Chosen 2026-09-24 (latency; quality tied on the keyword set). Revert: "databricks-gpt-oss-120b"
    "OPENAI_MODEL": "databricks-qwen3-next-80b-a3b-instruct",
    "MLFLOW_ENABLED": "true",
    "MLFLOW_EXPERIMENT": f"/Users/{ctx.userName().get()}/cas-evals",
})
```
`app.config` reads this env ONCE, at the first `import app...`. Changed Cell 2
later? `dbutils.library.restartPython()` and re-run it. Before reading any number,
check the model the run ACTUALLY used: run_eval's summary line `llm = <model>`
(`off` for a no-LLM control), or 07's `llm=` header. The setup header's `llm :` is
cfg's LIVE model; 05's `run()` prints the list it will run instead, and `control()`
prints `off`.

**Order:**
1. `08` — the LLM acceptance check. It needs no index, so run it BEFORE paying for
   one: every row PASS (read the WARNs) or stop here. Re-run it after any model change.
2. `dev.cas.products` is already verified dense (2026-09-23): **skip `01`** unless
   the catalog changes. If you do re-run it, delete the index first.
3. Check whether the index still exists; only build it if it does not.
4. `02` — then poll until `ONLINE` with 299,972 rows (~2h).
5. `03` smoke test.
6. `04 --rerank` → the full-pipeline number vs the recorded Groq NDCG 0.950.
7. `05` → CONFIRM Qwen on the CONTEXT set (the keyword set saturates: a no-LLM
   control scored the same there), against gpt-oss-120b and the no-LLM control.
   Then `06` on the same set → confirm or overturn query_expansion with Qwen
   intents. Compare its modes with EACH OTHER only: the recorded 0.904 / 0.716
   came from an older eval set.
   ```python
   os.environ["EVAL_QUERIES_JSON"] = "data/eval_queries_context.json"
   exec(open("databricks/05_compare_llms.py").read())
   run(["databricks-qwen3-next-80b-a3b-instruct", "databricks-gpt-oss-120b"])
   control()
   exec(open("databricks/06_compare_translators.py").read())
   run()                                 # now 06's run(): all three modes
   os.environ.pop("EVAL_QUERIES_JSON", None)  # read on EVERY call: left set, 07 below runs it twice
   ```
8. `07` → set `MIN_RESULT_RELEVANCE` for the deployed app. Re-run it with Qwen on
   BOTH sets — the 2026-09-23 value (0.638) came from gpt-oss intents — naming each
   set explicitly, and take the LOWER recommendation:
   ```python
   exec(open("databricks/07_calibrate_threshold.py").read())
   kw = run(translate=True, queries="data/eval_queries.json")
   ctx = run(translate=True, queries="data/eval_queries_context.json")
   ```
   The two `set=` header lines must name different files.
9. Tear down (below).

Every eval writes a CSV to `/Volumes/dev/cas/evals/` **and** an MLflow run
(params = models/modes, metrics = quality + real token totals, the CSV as an
artifact, tag `valid=true|false`). Each LLM call is also an MLflow trace
linked to its run. **Never compare a run tagged `valid=false`** — an LLM stage
fell back and the numbers measure something else.

## Cost guardrails (treat as rules)

- **Measured (list prices):** one clean index build ≈ **$20** (~79% sync/embedding
  at $0.45/DBU, ~21% endpoint); the endpoint alone ≈ **$5-8/day** while it exists
  (billing days were partial, so $/calendar-day understated it).
  A failed build costs the same as a good one (Sep 10's OFFLINE_FAILED: $23.39),
  so delete a FAILED index immediately. Keep the index up if (days until next
  use × ~$7) < ~$20 -- break-even ~3 days; otherwise delete and rebuild.
- `02` waits for the endpoint to be ONLINE before creating the index (likely cause
  of the Sep 10 OFFLINE_FAILED build).
- `02` uses TRIGGERED sync: the catalog never changes, and CONTINUOUS would keep a
  streaming pipeline billing while the index sits idle.
- Teardown (the `--teardown` flag does not work under `exec`; use this):
  ```python
  from databricks.vector_search.client import VectorSearchClient
  VectorSearchClient(disable_notice=True).delete_index(
      endpoint_name="cas-search", index_name="dev.cas.products_index")
  ```
- Coming back within a day or two? Leaving it up may cost less than a rebuild.
- Stop the App when not demoing (it bills per hour while running).
- Report incurred token/DBU costs upfront — the MLflow runs now carry real
  token totals per eval, so there is no excuse not to.

## What does NOT move yet

Local-only until agreed: latest enhancements under test, the fine-tune
pipeline (becomes a GPU job here later). Amazon Reviews 2023 is research-only —
fine for this internal migration, must be flagged before any client-facing use.
