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
| `00_preflight.py` | `preflight()`: access mode, runtime, packages/protobuf, SDK, auto-termination, UC, Volume file, billing access, endpoints, embedding auth from a worker thread | 0 (one ~2-token embedding) | — |
| `01_catalog_to_delta.py` | enriched CSV → Delta, CDF on, **dense** `catalog_index`, `text_hash`; refuses while the index exists | 0 | compute minutes |
| `02_vector_index.py` | `status()` `embed()` `resume()` `pause()` `costs()` | 0 (embedding tokens only in `embed()`) | **index on the endpoint ≈ 4 DBU/h** from `resume()` to `pause()` |
| `03_smoke_test.py` | app-path retrieval + own-text self-retrieval + one JSON-mode LLM call, all asserted | 1 | — |
| `04_run_eval.py` | retrieval eval; `--rerank` for the full pipeline | 18 / 36 | — |
| `05_compare_llms.py` | `list_models()`, `run([...])` — intent gen + rerank; `control()` = no-LLM baseline | 36 / 32 per model (keyword / context); control 0 | — |
| `06_compare_translators.py` | query_expansion vs hyde vs hybrid, rerank off | 18 / 16 per mode (keyword / context) | — |
| `07_calibrate_threshold.py` | calibrates `MIN_RESULT_RELEVANCE` for the VS score scale | ~28 keyword / ~26 context | — |
| `08_llm_smoke.py` | LLM-only acceptance check: every LLM call site once with the configured model (translator + bundles, rerank incl. a typed clothing window under the `[product_type]` constraint, chat assistant, cross-sell, injection x3), asserting no fallback, no JSON-mode/temperature downgrade | 40 | — (**no index needed**) |
| `09_deploy_app.py` | the Databricks App: `export_catalog()` `stage()` `app_create()` `app_grant()` `app_deploy()` `app_share()` `app_status()` `app_stop()` `app_start()` `app_costs()` | 1 tiny check per App start (+ each shopper search) | **App ≈ 0.5 DBU/h (Medium) while RUNNING**, plus the index |
| `10_page_eval.py` | `run()`: the APP-PATH page-type eval (`eval/eval_page_types.py` through `app.main.search`, page 1 as the App shows it): product types in the top 4 / 12 of clothing asks, outfit slot order, setup components, routing, collisions keep their vertical; `mode="both"` (default) = `TYPE_AWARE_RANKING` off then on, the LLM output replayed into the on arm. Needs FastAPI (not in Cell 1) | ~70-100 per repeat (34 queries) | — (index ONLINE) |
| `app_bundle/` | what `stage()` puts at the App's root: `app.yaml` (env), `run_app.py` (starts FastAPI + Streamlit), `requirements.txt` (exact pins, no torch) | — | — |
| `vs_shim.py` | shared setup: switch backend, results → UC Volume, MLflow on, prints the live `llm :` | — | — |

## Before the run (laptop)

1. Push the branch, then pull it into the workspace Git folder.
2. Upload the enriched catalog (gitignored, ~230MB) to the Volume `01` reads:
   `databricks fs cp --overwrite data/products_enriched.csv dbfs:/Volumes/dev/cas/raw/products_enriched.csv`
   (or Catalog Explorer → `dev.cas.raw` → Upload).

## The run

**Compute.** Serverless, or a classic all-purpose cluster (we use `Retail_DE_cluster`).
A cluster must have:
- **Access mode Standard or Dedicated.** "No isolation shared" cannot reach Unity
  Catalog (tables, Volumes, system tables). Dedicated must be assigned to a USER,
  not a group: group-assigned compute cannot reach model serving or AI Search.
- **DBR 14.3 LTS or newer** (15.4 / 16.4 / 17.3 LTS preferred; `Retail_DE_cluster` runs
  DBR 19, which works with the Cell 1 below). 13.3 is end-of-support.
- **Auto-termination on** (e.g. 30-60 min). A running cluster bills DBUs + Azure VMs
  even when idle, and `02 pause()` does not stop it. If it is a shared team cluster,
  leave its lifecycle to its owner; `costs()` then includes other users' DBUs on it.
- Step 0 below checks all of this and costs ~2 tokens.

Cell 1 installs ONLY what the runtime lacks, and never with `--upgrade`:
```python
%pip install "databricks-sdk>=0.117.0" "openai==2.42.0" python-dotenv
dbutils.library.restartPython()
```
- `databricks-sdk>=0.117.0`: runtimes bundle older SDKs, and before 0.117 it can fail on
  Standard/Spark Connect compute. Its own deps (protobuf>=4.25.8) leave the runtime's alone.
- `openai==2.42.0`: the version app/llm_client.py is tested with. Unpinned, pip took
  openai 3.28.0 on 2026-10-10 -- a new MAJOR version (on httpx2) the code never ran on.
- **Do NOT install `databricks-vectorsearch`.** It is a deprecated shim that pins
  protobuf<6; on DBR 19 (protobuf 6.33.5) that downgrade killed the Python kernel
  ("gencode 6.33.5 runtime 5.29.6", 2026-10-10). The code drives Vector Search through
  its REST API with the preinstalled SDK instead.
- Do not re-install `mlflow` / `scikit-learn` either: every runtime ships them, and
  `--upgrade` over them moved core packages (wcwidth) out from under the kernel.
- **Re-run Cells 1 and 2 at the start of EVERY session**, and after any cluster restart
  or detach: `%pip` installs are notebook-scoped and vanish then. `Retail_DE_cluster`
  auto-terminates after 15 idle minutes, so a coffee break is enough. Variables from
  `exec(...)` (e.g. `status()`, `resume()`) are lost too: re-run that step's `exec` line.
- If a bad install already broke the notebook: **detach and re-attach it** (that resets
  notebook-scoped libraries), then run Cell 1 again.
Cell 2 — env, including MLflow. Both LLMs named below are open-weight
(Apache-2.0) models served by Databricks pay-per-token; nothing leaves the workspace:
```python
import os, sys
from databricks.sdk import WorkspaceClient
REPO = "/Workspace/Users/<you>/CAS - Sai/Context-Aware-Search"
sys.path.insert(0, REPO); os.chdir(REPO)
# The SDK's notebook auth works on serverless AND on Standard / Dedicated clusters;
# the older getContext().apiToken() call is not documented for Standard access mode
_w = WorkspaceClient()
HOST = _w.config.host.rstrip("/")
TOKEN = _w.config.authenticate()["Authorization"].split(" ", 1)[1]
ME = _w.current_user.me().user_name
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
    "MLFLOW_EXPERIMENT": f"/Users/{ME}/cas-evals",
})
```
`app.config` reads this env ONCE, at the first `import app...`. Changed Cell 2
later? `dbutils.library.restartPython()` and re-run it. Before reading any number,
check the model the run ACTUALLY used: run_eval's summary line `llm = <model>`
(`off` for a no-LLM control), or 07's `llm=` header.

**Order** (one notebook cell per numbered step):
```python
# 0. Pre-flight: compute, runtime, Unity Catalog, Volume file, endpoints, auth from threads
exec(open("databricks/00_preflight.py").read()); preflight()

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
Next run: steps 0, 1, 2, 6-11 only. `embed()` is needed again only after `01`
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

## Measuring the type-aware ranking change (2026-10-14)

The change (CLAUDE.md, "Type-aware ranking") was measured only LOCALLY (gte-small, Groq); the
demo complaint came from the App (gte-large-en + Qwen). It needs no `01` / `02` re-run and no
re-embed: `product_type` is derived from the titles when the catalog loads. With the index ONLINE:
```python
# BEFORE pulling the change (old code in the Git folder, Cells 1-2 run): the context set,
# full pipeline, 3 times (Qwen gets no seed, so one run is noise). KEEP the three CSVs
os.environ["EVAL_QUERIES_JSON"] = "data/eval_queries_context.json"
for _ in range(3):
    sys.argv = ["04", "--rerank"]; exec(open("databricks/04_run_eval.py").read())
os.environ.pop("EVAL_QUERIES_JSON", None)

# Pull the branch, dbutils.library.restartPython(), Cells 1-2 again (Cell 1 + "fastapi==0.136.3"
# for 10). Then the same 04 x3 -- the after runs carry MLflow param prompts_sha, the before runs don't
os.environ["EVAL_QUERIES_JSON"] = "data/eval_queries_context.json"
for _ in range(3):
    sys.argv = ["04", "--rerank"]; exec(open("databricks/04_run_eval.py").read())
os.environ.pop("EVAL_QUERIES_JSON", None)

# App-path page eval: knob off then on, the LLM output replayed (~70-100 calls)
exec(open("databricks/10_page_eval.py").read()); run(mode="both")

# LLM acceptance (40 calls): read the new row "reranker / type constraint (clothes ask)" --
# WARN = Qwen scored a non-apparel row above 40 under the [product_type] constraint
exec(open("databricks/08_llm_smoke.py").read()); run()
```
What to read: `fell_back` first (a degraded run measures nothing); the context-set P@10 / MRR
means before vs after (gaps under ~0.05 P@10 on 16 queries are a tie, .claude/rules/eval.md); on
the page eval, `mismatch_top4` / `mismatch_top12` of the clothing asks (0 / 0 locally with the
knob on, Groq qwen3.8-27b, also on the shipped prompts) and the collisions ("tea bags", "apple watch") staying in their vertical. "off" is NOT
the pre-change code (prompts, routing, the gender fix and `type=` are outside the knob); for a
true before, see 10's docstring (a Git folder at the previous commit with three files copied in).

## Deploy the App (09)

The App runs the same code as the notebooks — FastAPI (`app/main.py`) on a private
port and the Streamlit UI on the App's port — as its OWN service principal, against
the same index and LLM. Nothing in it uses a personal token.

Before the first deploy: the branch with `09` and `databricks/app_bundle/` is pulled into
the workspace Git folder, the index is ONLINE (`02 resume()`), and Cells 1-2 have run.
Creating an App and granting on `dev.cas` may need someone with MANAGE on those objects:
`app_grant()` prints PASS/FAIL per grant and says what to ask for.

```python
exec(open("databricks/09_deploy_app.py").read())

# A. Once (and after every 01 re-run): the App's catalog = dev.cas.products WITH its
#    catalog_index, so the App shows exactly the rows the evals scored
export_catalog()

# B. Copy only the App's files to /Workspace/Users/<you>/cas-app (no spaces, no torch)
stage()

# C. Once: the App + resources (CAN QUERY on both endpoints, READ on the Volume).
#    Created STOPPED: nothing bills yet
app_create()

# D. Grants for its service principal: USE CATALOG/SCHEMA, READ VOLUME, SELECT on the
#    index, read on the staged folder. RE-RUN after EVERY 02 resume() (new index object)
app_grant()

# E. Start + deploy (billing starts). Prints the URL. The backend loads ~1 min more:
#    <url>/logz should show "Start-up check index/embedding/llm: OK" and "service ready"
app_deploy()

# F. Who may open it (account users only -- Apps cannot be public)
app_share("someone@latentview.com")

# G. After the demo: stop the App, then delete the index
app_stop()
exec(open("databricks/02_vector_index.py").read()); pause()
```
Next demo: `02 resume()` → `app_grant()` → `app_start()` (no re-deploy needed). After a
code change: `stage()` → `app_deploy()`. `app_costs("YYYY-MM-DD")` the day after.
After pulling the type-aware ranking change (2026-10-14): `stage()` → `app_deploy()` only -- no
`export_catalog()`, `01` or `02` re-run. `product_type` is derived from the titles at load (adds
~6-10 s measured on the laptop; not yet measured on the App's 2 vCPU), never stored or embedded;
`TYPE_AWARE_RANKING` defaults to true.

`app_deploy()` / `app_start()` check BEFORE starting (billed) compute: the export matches
`dev.cas.products` row for row (else `export_catalog()`), the index is ONLINE with the same
row count, and `stage()` ran. A deploy that fails stops the compute it started; one that
starts but crashes prints `STILL BILLING`. They refuse an App another user created
(`CAS_APP_NAME` picks another name), and `app_share()` refuses "account users".

What can go wrong, and where it shows:
- `/logz` `Start-up check index FAILED` → index paused/deleted or SELECT missing: `resume()`,
  `app_grant()`, then `app_stop()` + `app_start()`.
- `Start-up check embedding/llm FAILED ... PERMISSION` → the App lacks CAN QUERY: in the
  App's page, Edit → Resources → Serving endpoint → Can query (needs CAN MANAGE on it).
- The UI says "could not reach the search service" for the first minute after a start:
  the backend is still loading the catalog and keyword index (~30-60 s).
- The App is internal-only: the catalog is Amazon Reviews 2023 (research licence).

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
- **Measured 2026-10-12 (Retail_DE_cluster):** `embed()` 44.73M tokens ≈ 83 DBU at the
  script's list-rate estimate, ~38 min, once; a stored-vector `resume()` took 47.8 min
  (no embedding calls). Their billed $ come from `costs("2026-10-12")` the day after.
- `costs()` is not live: billing is "typically available within 12 hours". During
  a run, watch the `embed()` / `resume()` progress lines instead.
- `02` waits for the endpoint to be ONLINE before creating the index (likely cause
  of the Sep 10 OFFLINE_FAILED build) and uses TRIGGERED sync (CONTINUOUS keeps a
  streaming pipeline billing while the index sits idle).
- **Stop the App when not demoing** (`app_stop()`) — a stopped App costs nothing, a
  running one bills even with no visitors (there is no idle auto-stop). `pause()` warns
  if the App is still running, since without the index it is useless anyway.
- Report incurred token/DBU costs upfront — the MLflow runs carry real token
  totals per eval.

## What does NOT move yet

Local-only until agreed: latest enhancements under test, the fine-tune
pipeline (becomes a GPU job here later). Amazon Reviews 2023 is research-only —
fine for this internal migration, must be flagged before any client-facing use.
