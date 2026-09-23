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
| `03_smoke_test.py` | one similarity query + one LLM call | 1 | — |
| `04_run_eval.py` | retrieval eval; `--rerank` for the full pipeline | 18 / 36 | — |
| `05_compare_llms.py` | `list_models()` then `run([...])` — intent gen + rerank | 36 per model | — |
| `06_compare_translators.py` | query_expansion vs hyde vs hybrid, rerank off | 18 per mode | — |
| `07_calibrate_threshold.py` | calibrates `MIN_RESULT_RELEVANCE` for the VS score scale | ~28 | — |
| `vs_shim.py` | shared setup: switch backend, results → UC Volume, MLflow on | — | — |

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
    "OPENAI_MODEL": "databricks-gpt-oss-120b",
    "MLFLOW_ENABLED": "true",
    "MLFLOW_EXPERIMENT": f"/Users/{ctx.userName().get()}/cas-evals",
})
```

**Order:**
1. Delete any existing index (a rewrite of the table re-embeds everything anyway).
2. `01` — must print `catalog_index dense 0..299,971`.
3. `02` — then poll until `ONLINE` with 299,972 rows (~2h).
4. `03` smoke test.
5. `04 --rerank` → the full-pipeline number vs the recorded Groq NDCG 0.950.
6. `05` → pick the LLM. `06` → confirm or overturn query_expansion.
7. `07` → set `MIN_RESULT_RELEVANCE` for the deployed app.
8. Tear down (below).

Every eval writes a CSV to `/Volumes/dev/cas/evals/` **and** an MLflow run
(params = models/modes, metrics = quality + real token totals, the CSV as an
artifact, tag `valid=true|false`). Each LLM call is also an MLflow trace
linked to its run. **Never compare a run tagged `valid=false`** — an LLM stage
fell back and the numbers measure something else.

## Cost guardrails (treat as rules)

- The vector endpoint bills from creation and only stops **~24h after the last
  index is deleted** — idle is not free. A rebuild costs ~2 endpoint-hours
  before the first query, so it is never a casual action.
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
