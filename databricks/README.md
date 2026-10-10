# Databricks migration — runbook

Decided on the 2026-08-13 call (Databricks = non-negotiable priority) and
sequenced on the 2026-08-24 call: migrate the FOUNDATIONAL, already-tested
solution, while enhancements keep being developed locally. Component mapping
verified against Databricks docs on 2026-07-29 — see `docs/RESEARCH.md` §3.
Measured results are recorded in CLAUDE.md ("Databricks migration — VERIFIED
RESULTS").

## How the code runs here

`SEARCH_BACKEND=databricks` switches `app/search.py` to Mosaic AI Vector
Search (now "AI Search"). Every consumer — `main.py`, recommendations, the
keyword engine and the whole eval suite — goes through that module, so **the
eval exercises exactly the code the deployed app runs**. The index returns only
`catalog_index` + score; full rows are hydrated from the catalog, so both
backends return identical row shapes (images, `parent_asin` for sponsored ads, etc.).

**Embeddings are computed once and stored** (since 2026-10-09). An AI Search
endpoint or index cannot be paused, and an index nobody queries still bills
its endpoint. So `02 embed()` embeds the catalog with `databricks-gte-large-en`
into Delta, `02 resume()` builds a *self-managed* index from those stored
vectors (no embedding calls), and `02 pause()` deletes only the index. The
endpoint stops billing 24h after its last index is deleted ([cost
management](https://learn.microsoft.com/en-us/azure/databricks/ai-search/cost-management)).
The app embeds each search's intents with the same endpoint (one call) and
queries with `query_vector` (`VS_QUERY_MODE=vector`).

```
products_enriched.csv --01--> dev.cas.products (search_text, text_hash, catalog_index)
  --02 embed()-->  dev.cas.products_embeddings  (model, text_hash) -> vector   [kept forever]
               ->  dev.cas.products_vectors     catalog_index -> vector    [index source]
  --02 resume()--> dev.cas.products_index (self-managed)                    [delete between runs]
```

| Script | Does | LLM calls | Billing |
|---|---|---|---|
| `01_catalog_to_delta.py` | enriched CSV → Delta, CDF on, **dense** `catalog_index`, `text_hash`; refuses while the index exists | 0 | compute minutes |
| `02_vector_index.py` | `status()` `embed()` `resume()` `pause()` `costs()` | 0 (embedding tokens only in `embed()`) | **index on the endpoint ≈ 4 DBU/h** from `resume()` to `pause()` |
| `03_smoke_test.py` | app-path retrieval + own-text self-retrieval + one JSON-mode LLM call, all asserted | 1 | — |
| `04_run_eval.py` | retrieval eval; `--rerank` for the full pipeline | 18 / 36 | — |
| `05_compare_llms.py` | `list_models()`, `run([...])` — intent gen + rerank; `control()` = no-LLM baseline | 36 / 32 per model (keyword / context); control 0 | — |
| `06_compare_translators.py` | query_expansion vs hyde vs hybrid, rerank off | 18 / 16 per mode (keyword / context) | — |
| `07_calibrate_threshold.py` | calibrates `MIN_RESULT_RELEVANCE` for the VS score scale | ~28 keyword / ~26 context | — |
| `08_llm_smoke.py` | LLM-only acceptance check: every LLM call site once with the configured model (translator + bundles, rerank, chat assistant, cross-sell, injection x3), asserting no fallback, no JSON-mode/temperature downgrade | 25 | — (**no index needed**) |
| `vs_shim.py` | shared setup: switch backend, results → UC Volume, MLflow on, prints the live `llm :` | — | — |

## Before the run (laptop)

1. Push the branch, then pull it into the workspace Git folder.
2. Upload the enriched catalog (gitignored, ~230MB) to the Volume `01` reads:
   `databricks fs cp --overwrite data/products_enriched.csv dbfs:/Volumes/dev/cas/raw/products_enriched.csv`
   (or Catalog Explorer → `dev.cas.raw` → Upload).

## The run

**Notebook setup** (serverless). Cell 1:
```python
%pip install databricks-vectorsearch databricks-sdk openai python-dotenv scikit-learn mlflow
dbutils.library.restartPython()
```
Cell 2 — env, including MLflow. Both LLMs named below are open-weight
(Apache-2.0) models served by Databricks pay-per-token; nothing leaves the workspace:
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
    # Query expansion + rerank. Chosen 2026-09-24 (latency). Alternative: "databricks-gpt-oss-120b"
    "OPENAI_MODEL": "databricks-qwen3-next-80b-a3b-instruct",
    "VS_EMBEDDING_MODEL": "databricks-gte-large-en",
    "VS_QUERY_MODE": "vector",
    "MLFLOW_ENABLED": "true",
    "MLFLOW_EXPERIMENT": f"/Users/{ctx.userName().get()}/cas-evals",
})
```
`app.config` reads this env ONCE, at the first `import app...`. Changed Cell 2
later? `dbutils.library.restartPython()` and re-run it. Before reading any number,
check the model the run ACTUALLY used: run_eval's summary line `llm = <model>`
(`off` for a no-LLM control), or 07's `llm=` header.

**Order** (one notebook cell per numbered step):
```python
# 1. LLM acceptance check -- needs no index, so run it BEFORE paying for one
exec(open("databricks/08_llm_smoke.py").read()); run()

# 2. What exists and what bills
exec(open("databricks/02_vector_index.py").read()); status()

# 3. Only if status() shows the OLD managed index: delete it (base catalog, query_text)
pause()

# 4. Load the enriched catalog (refuses while an index exists)
exec(open("databricks/01_catalog_to_delta.py").read())

# 5a. Embed a pilot: prints texts/min and PROJECTS the full run's tokens / DBU / minutes
exec(open("databricks/02_vector_index.py").read())
embed(limit=2000)

# 5b. Read 5a's projection first, then embed the rest. Resumable: just re-run after an error
embed()

# 6. Build the index from stored vectors; prints progress every minute until ONLINE
resume()

# 7. Smoke test: app-path retrieval, self-retrieval, one LLM call
exec(open("databricks/03_smoke_test.py").read())

# 8. Retrieval eval, then the full pipeline
sys.argv = ["04"]; exec(open("databricks/04_run_eval.py").read())
sys.argv = ["04", "--rerank"]; exec(open("databricks/04_run_eval.py").read())

# 9. Context set: open LLMs vs the no-LLM control, then translator modes
os.environ["EVAL_QUERIES_JSON"] = "data/eval_queries_context.json"
exec(open("databricks/05_compare_llms.py").read())
run(["databricks-qwen3-next-80b-a3b-instruct", "databricks-gpt-oss-120b"])
control()
exec(open("databricks/06_compare_translators.py").read())
run()
os.environ.pop("EVAL_QUERIES_JSON", None)

# 10. "No products found" floor for this score scale -- take the LOWER of the two
exec(open("databricks/07_calibrate_threshold.py").read())
kw = run(translate=True, queries="data/eval_queries.json")
ctx_set = run(translate=True, queries="data/eval_queries_context.json")

# 11. Done: delete the index (stored vectors + endpoint stay)
exec(open("databricks/02_vector_index.py").read()); pause(); status()

# 12. NEXT DAY (billing lags ~12h): what the run cost
exec(open("databricks/02_vector_index.py").read()); costs("2026-10-09")
```
Next run: steps 1, 2, 6-11 only. `embed()` is needed again only after `01`
reloads a CHANGED catalog (it then embeds only the changed text) or after
`VS_EMBEDDING_MODEL` changes (then `pause()`, `embed()`, `resume()`: the vectors
table records its model, and `resume()` refuses vectors from another model).
`costs()` is approximate: the pay-per-token endpoints are shared by the whole
workspace. The exact per-run numbers are `embed()`'s token count and each eval's
MLflow token totals.

Every eval writes a CSV to `/Volumes/dev/cas/evals/` **and** an MLflow run
(params = models/modes/catalog/query mode, metrics = quality + real token
totals, the CSV as an artifact, tag `valid=true|false`). **Never compare a run
tagged `valid=false`** — an LLM stage fell back. Runs with
`vs_query_mode=vector` (stored vectors, enriched catalog) are NOT comparable
with the September numbers (managed index, base catalog).

## Cost guardrails (treat as rules)

- **No pause exists.** An AI Search endpoint/index cannot be stopped or scaled to
  zero; an index that nobody queries still bills one STANDARD unit, 4 DBU/h
  (≈ $6.72/day at the $0.07/DBU list rate;
  [pricing](https://www.databricks.com/product/pricing/vector-search)). An
  endpoint with NO index stops billing 24h after its last index is deleted
  ([cost management](https://learn.microsoft.com/en-us/azure/databricks/ai-search/cost-management)).
  So: `pause()` after every run; keep the index only if you are back within ~4 days.
- **Measured (list prices), managed builds:** ≈ **$20** per build incl. a session
  (Sep 11: $20.27); a failed build costs the same (Sep 10 OFFLINE_FAILED: $23.39),
  so `pause()` a FAILED index immediately (`resume()` tells you).
- **Not yet measured:** `embed()` (estimate: ~190M chars ≈ 48M tokens × 1.857 DBU/1M
  ≈ 88 DBU, once — `embed()` prints the running token count) and a stored-vector
  `resume()`. Run `costs()` the day after and record both here.
- `costs()` is not live: billing is "typically available within 12 hours". During
  a run, watch the `embed()` / `resume()` progress lines instead.
- `02` waits for the endpoint to be ONLINE before creating the index (likely cause
  of the Sep 10 OFFLINE_FAILED build) and uses TRIGGERED sync (CONTINUOUS keeps a
  streaming pipeline billing while the index sits idle).
- Stop the App when not demoing — a stopped App costs nothing.
- Report incurred token/DBU costs upfront — the MLflow runs carry real token
  totals per eval.

## What does NOT move yet

Local-only until agreed: latest enhancements under test, the fine-tune
pipeline (becomes a GPU job here later). Amazon Reviews 2023 is research-only —
fine for this internal migration, must be flagged before any client-facing use.
