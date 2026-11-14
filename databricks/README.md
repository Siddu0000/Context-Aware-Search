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
| `00_preflight.py` | `preflight()`: access mode, runtime, packages/protobuf, SDK, auto-termination, UC, Volume file, billing access, endpoints, embedding auth from a worker thread; since 2026-10-27 also pandas / numpy against the App's pins and the App's memo volume (a dry write at its root, deleted at once), both WARN, never FAIL | 0 (one ~2-token embedding) | — |
| `01_catalog_to_delta.py` | enriched CSV → Delta, CDF on, **dense** `catalog_index`, `text_hash`; refuses while the index exists | 0 | compute minutes |
| `02_vector_index.py` | `status()` `embed()` `resume()` `pause()` `delete_endpoint()` `costs()` | 0 (embedding tokens only in `embed()`) | **index on the endpoint ≈ 4 DBU/h** from `resume()` to `pause()` |
| `03_smoke_test.py` | app-path retrieval + own-text self-retrieval + one JSON-mode LLM call, all asserted | 1 | — |
| `04_run_eval.py` | retrieval eval; `--rerank` for the full pipeline | 18 / 36 | — |
| `05_compare_llms.py` | `list_models()`, `run([...])` — intent gen + rerank; `control()` = no-LLM baseline | 36 / 32 per model (keyword / context); control 0 | — |
| `06_compare_translators.py` | query_expansion vs hyde vs hybrid, rerank off | 18 / 16 per mode (keyword / context) | — |
| `07_calibrate_threshold.py` | calibrates `MIN_RESULT_RELEVANCE` for the VS score scale | ~28 keyword / ~26 context | — |
| `08_llm_smoke.py` | LLM-only acceptance check: every LLM call site once with the configured model (translator + bundles, rerank incl. a typed clothing window under the `[product_type]` constraint -- since 2026-10-19 9 annotated rows with `subtype=` / `audience=` and the guest `[occasion]` constraint too, WARN on a type or guest cap above 40 -- chat assistant, cross-sell, injection x3; since 2026-10-27 the watch-gift and laptop translator rows and the device-gift rerank window with its ` \| role=` marks), asserting no fallback, no JSON-mode/temperature downgrade. Runs with the LLM memo OFF (2026-10-27): a row a memo hit answered FAILs. Rows listed in CLAUDE.md ("Running the eval on Databricks") | 43 (~55K tokens, estimated) | — (**no index needed**) |
| `09_deploy_app.py` | the Databricks App: `export_catalog()` `stage()` `app_create()` `app_grant()` `app_deploy()` `app_share()` `app_status()` `app_stop()` `app_start()` `app_costs()`; since 2026-10-27 `app_create()` / `app_grant()` also create and grant the memo volume `dev.cas.app_state` | 1 tiny check per App start (+ each new shopper search; a repeat replays from the memo) | **App ≈ 0.5 DBU/h (Medium) while RUNNING**, plus the index |
| `10_page_eval.py` | `run()`: the APP-PATH page-type eval (`eval/eval_page_types.py` through `app.main.search`, page 1 as the App shows it): product types in the top 4 / 12 of clothing asks, bridal wear on a bridal ask, outfit slot order (outfit knob on only), setup components, routing, collisions keep their vertical; `mode="both"` (default) = `TYPE_AWARE_RANKING` off then on, the LLM output replayed into the on arm; the CSV records `rejects_after_tail`, `variant_collapse`, `prompts_sha` (2026-10-19); since 2026-10-27 the 10 'device' rows (`core_top4` / `core_top12` / `first_core`), and it switches the LLM memo off (refusing to start if it stays on). Needs FastAPI (not in Cell 1) | ~100-145 per repeat (48 queries; ~0.24-0.38M tokens, estimated) | — (index ONLINE) |
| `11_idle_guard.py` | `check()` (dry run: reports what it would do) / `check(dry_run=False)`: after 3 idle days (no `queryIndex` in `system.access.audit`, no rebuild) stops the App and runs `02 pause()`. Assumptions UNVERIFIED (see "Idle teardown") | 0 | the job's compute while it runs |
| `12_determinism.py` | `run()`: `eval/check_determinism.py` on Vector Search (2026-10-27): per query, cold runs of pages 1 + 2 and the top card's product page; with the memo ON (in a scratch folder, never the App's) runs 2..N must equal run 1 with 0 provider calls; `run(memo=False)` reports the model's raw stability; `cleanup(folder)`. See "Determinism check (12)". Needs FastAPI (as 10) | memo on: run 1 ~63 (21 queries; ~0.11M tokens, estimated), runs 2+ 0 | — (index ONLINE) |
| `app_bundle/` | what `stage()` puts at the App's root: `app.yaml` (env; since 2026-10-27 `LLM_MEMO` + `LLM_MEMO_DIR` on the memo volume), `run_app.py` (starts FastAPI + Streamlit; carries the UI theme, see "Deploy the App"), `requirements.txt` (exact pins, no torch) | — | — |
| `vs_shim.py` | shared setup: switch backend, results → UC Volume, MLflow on, the LLM memo off unless `memo=True`, prints the live `llm :` and `memo :` lines; the App's memo volume (00, 09, 12) and pins (00) | — | — |

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
(Apache-2.0) models served by Databricks; nothing leaves the workspace.

**Unity Gateway (2026-11-13).** The workspace now has "Enforce Unity Gateway" on: every
`databricks-*` pay-per-token endpoint refuses calls ("Please use Unity Gateway"), and the
same models are model services (`system.ai.*`) at `<workspace URL>/ai-gateway/mlflow/v1`.
That route answers ONLY on the workspace URL (`adb-....azuredatabricks.net`, from
`spark.databricks.workspaceUrl`); the SDK's notebook host (`https://eastus-c3...`, the
regional control plane) answers 404 there. `seed` is refused (400), so the app never sends
it to `/ai-gateway/`. `VS_EMBEDDING_MODEL` stays the LABEL of the stored vectors (no
re-embed); `VS_EMBEDDING_SERVICE` is what is called. REVERT (enforcement off): the
commented legacy lines below instead of the gateway ones.
```python
import os, sys
from databricks.sdk import WorkspaceClient
REPO = "/Workspace/Users/<you>/CAS - Sai/Context-Aware-Search"
sys.path.insert(0, REPO); os.chdir(REPO)
# The SDK's notebook auth works on serverless AND on Standard / Dedicated clusters;
# the older getContext().apiToken() call is not documented for Standard access mode
_w = WorkspaceClient()
HOST = _w.config.host.rstrip("/")
WS = "https://" + spark.conf.get("spark.databricks.workspaceUrl")   # the gateway's host
TOKEN = _w.config.authenticate()["Authorization"].split(" ", 1)[1]
ME = _w.current_user.me().user_name
os.environ.update({
    "CAS_CATALOG": "dev", "CAS_SCHEMA": "cas",
    "DATABRICKS_TOKEN": TOKEN, "DATABRICKS_HOST": HOST,
    "LLM_PROVIDER": "openai", "OPENAI_API_KEY": TOKEN,
    # Unity Gateway: LLM and embeddings on the workspace URL
    "OPENAI_BASE_URL": WS + "/ai-gateway/mlflow/v1",
    "AI_GATEWAY_URL": WS + "/ai-gateway/mlflow/v1",
    # Query expansion + rerank. Chosen 2026-09-24 (latency)
    "OPENAI_MODEL": "system.ai.qwen3-next-80b-a3b-instruct",
    "VS_EMBEDDING_SERVICE": "system.ai.gte-large-en",
    # Legacy pay-per-token route (enforcement off), instead of the four lines above:
    # "OPENAI_BASE_URL": HOST + "/serving-endpoints",
    # "OPENAI_MODEL": "databricks-qwen3-next-80b-a3b-instruct",
    "VS_EMBEDDING_MODEL": "databricks-gte-large-en",   # the stored vectors' label
    "VS_QUERY_MODE": "vector",
    "MLFLOW_ENABLED": "true",
    "MLFLOW_EXPERIMENT": f"/Users/{ME}/cas-evals",
})
```
The App's `app.yaml` carries the same route (`OPENAI_BASE_PATH`, `VS_EMBEDDING_SERVICE`,
the `system.ai` model name), and `09 stage()` refuses it if it disagrees with Cell 2. A new
App gets "Unity Gateway model service" resources (Can execute); the existing `cas-app` keeps
its old endpoint resources, so its service principal needs EXECUTE on both `system.ai`
services (account users hold it by default) or the resources added in the App UI (Edit >
Resources > Model). `02 costs()` does not see gateway charges (they are recorded under
`usage_metadata.ai_gateway`).
`app.config` reads this env ONCE, at the first `import app...`. Changed Cell 2
later? `dbutils.library.restartPython()` and re-run it. Before reading any number,
check the model the run ACTUALLY used: run_eval's summary line `llm = <model>`
(`off` for a no-LLM control), or 07's `llm=` header.

Cell 2 sets no `LLM_MEMO` (2026-10-27), so app calls you make yourself in a notebook
(`app.main.search`, the translator...) run with the code default: the LLM reply memo ON, an
SQLite file under the Git folder (`.cache/llm_memo`), replaying the first reply each stage
accepted. The runbook's steps do not: 04-07, 10 and 12 switch it through
`vs_shim.patch_eval_harness` (off; 12 then points it at its own scratch folder), and 08 and
every `eval/` harness but `check_determinism` (12's) turn it off themselves (the shim's
`memo :` header line and 08's params row "LLM memo off" show it). To keep it off for your
own calls too, add `"LLM_MEMO": "false"` to Cell 2; never point `LLM_MEMO_DIR` at the App's
`app_state` volume.

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

# 11. Done: delete the index (stored vectors + endpoint stay). Optionally the endpoint too:
#     delete_endpoint() ends its 24h billing tail; resume() must then recreate the endpoint first
#     (time unmeasured; _ensure_endpoint waits up to 60 min)
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

## Measuring on Qwen (not run yet)

Four rounds of ranking changes are measured only LOCALLY so far (gte-small + Groq; the demo
complaint of 2026-10-14 came from the App: gte-large-en + Qwen): the type-aware ranking
(2026-10-14), the 2026-10-15 changes (outfit bundles OFF, guest rule and SETUP_PROMPT narrowed,
type-aware amendments, shop filters, display casing), the 2026-10-19 apparel round (rejects
after the tail, variant collapse, fit / age window tiers, rubric, QE prompt, guest rule in code,
marketplace UI) and the 2026-10-27 round (device-role ranking for device and gift asks, the
Electronics / Grocery taxonomy and left rail, the LLM reply memo, the "Related items" button);
CLAUDE.md, Decisions, has each. None needs `01`, `02`, `export_catalog()` or a re-embed:
`product_type.annotate` (in `search.load_index`, both backends) derives `product_type` and,
since 2026-10-19, `fine_subtype`, `shopper_gender`, `audience` and `color_family` at load; since
2026-10-27 `catalog_taxonomy.annotate_catalog` adds `item_category`, `item_role`, `watch_kind`,
`diet_tags`, `works_with`, `item_condition` and `brand` the same way; search_text is
byte-identical. Pulling the current branch measures 2026-10-14, -15, -19 and -27 together. For
the type-aware change alone, check out commit f75b3a6 (its before: 1174089; 34 page-eval
queries there); the 2026-10-15, 2026-10-19 and 2026-10-27 rounds have no commit of their own
yet (uncommitted when this was written): record their commits here when they land.

With the index ONLINE:
```python
# OPTIONAL BEFORE (old code in the Git folder, Cells 1-2 run): the context set, full pipeline,
# 3 times (Qwen gets no seed, so one run is noise). KEEP the three CSVs
os.environ["EVAL_QUERIES_JSON"] = "data/eval_queries_context.json"
for _ in range(3):
    sys.argv = ["04", "--rerank"]; exec(open("databricks/04_run_eval.py").read())
os.environ.pop("EVAL_QUERIES_JSON", None)

# Pull the branch, dbutils.library.restartPython(), Cells 1-2 again (Cell 1 + "fastapi==0.136.3"
# for 10 and 12). Then, in this order:
# LLM acceptance (43 calls, no index needed; runs with the LLM memo off)
exec(open("databricks/08_llm_smoke.py").read()); run()

# App-path page eval: knob off then on, the LLM output replayed (~100-145 calls, 48 queries)
exec(open("databricks/10_page_eval.py").read()); run(mode="both")

# Determinism (2026-10-27): memo ON in a scratch folder, run 1 ~63 calls, runs 2-3 none
exec(open("databricks/12_determinism.py").read()); verdicts = run()

# Stress queries (2026-11-02): the 107-query complex bank, one arm (~214 calls, ~0.36-0.55M
# tokens, ~20 min, all ESTIMATES), then its per-group report (eval/score_complex_queries.py)
exec(open("databricks/10_page_eval.py").read())
csv_path, rows = run(mode="on", queries="data/complex_queries.json")
from eval import score_complex_queries as scq; print(scq.render(scq.score(rows)))

# The same 04 x3 -- the after runs carry MLflow param prompts_sha, pre-2026-10-14 runs don't
os.environ["EVAL_QUERIES_JSON"] = "data/eval_queries_context.json"
for _ in range(3):
    sys.argv = ["04", "--rerank"]; exec(open("databricks/04_run_eval.py").read())
os.environ.pop("EVAL_QUERIES_JSON", None)
```
What to read: `fell_back` first in every run (a degraded run measures nothing); the context-set
P@10 / MRR means before vs after (gaps under ~0.05 P@10 on 16 queries are a tie,
.claude/rules/eval.md). Then, per round:
- **2026-10-14 (type-aware)**: on the page eval, `mismatch_top4` / `mismatch_top12` of the
  clothing asks (0 / 0 locally with the knob on, Groq qwen3.8-27b, also on the 2026-10-14
  prompts (QE 167a8685, SETUP 5b20a07c; superseded 2026-10-15)) and the collisions ("tea bags",
  "apple watch") staying in their vertical; in 08, "reranker / type constraint (clothes ask)":
  WARN = Qwen scored a non-apparel row above 40 under the `[product_type]` constraint. "off" is
  NOT the pre-change code (prompts, routing, the gender fix and `type=` are outside the knob);
  for a true before, see 10's docstring (a Git folder at the previous commit with three files
  copied in).
- **2026-10-15**: in 08, "outfit ask -> flat garments" (must be flat; WARN = not NUM_INTENTS
  men's garments) and "setup bundle" (WARN = no monitor / keyboard-mouse, or a mood item nobody
  asked for); in 10, the 4 'bridal' rows count the bridal wear served (context_any_of; want it on
  page 1) and "clothes for a beach wedding" must serve none (context_none_of) -- the live check of
  the guest-rule narrowing, unverified so far (.claude/rules/eval.md, gpt-oss-20b), including the
  query behind the meeting's "wedding costume" complaint (exact wording unconfirmed). (The
  fixed-intent guest-rule A/B used on 2026-10-15 was a laptop scratchpad script, not in the repo:
  it pins Groq gpt-oss-20b and local paths and would need porting before it could run here.)
- **2026-10-19 (apparel round)**: in 08, the clothes-ask rerank row now also sends the guest
  `[occasion]` constraint and `subtype=` / `audience=` rows (WARN = a type or guest cap above 40);
  in 10, 'something for her to wear to a black-tie gala' now gets the formal-subtype partition and
  the guest constraint. The new columns read the catalog's `gender`, `categ_lvl2_name` and `color`
  (dev.cas.products and the App export carry them); check the boot log's `Annotations derived:
  {...}` value counts: a missing `color` shows as an all-"none" color_family; a missing `gender`
  falls back to `categ_lvl2_name` (invisible in the counts), and with both missing shopper_gender
  comes from title words alone (fewer "unisex", more women / men than the tag gives). Rerank-on
  numbers do not compare across 2026-10-19 (rubric, row fields, QE prompt: `prompts_sha`,
  `row_text_case` 'original+fine_fields'); +15.6% tokens per live rerank-on query locally (QE
  prompt ~245, rerank call ~375 tokens per query: .claude/rules/eval.md). The graded apparel
  battery is a laptop harness pinned to Groq qwen3.8-27b and local paths (.claude/rules/eval.md,
  "Apparel battery"): grading the App's pages needs it ported.
- **2026-10-27 (device asks, determinism, rail)**: 08 runs with the memo OFF and restores the
  notebook's value afterwards; a row a memo hit answered FAILs. Its new rows: translator "gift:
  vintage watch collector" (WARN unless the first intent is a watch itself, on more than one
  storage / display intent, on any tool, care or other add-on intent: owner call D1) and
  "device: laptop for college" (WARN on an intent that is not a laptop), reranker "device gift
  (watch collector)" (the window /search sends under `DEVICE_ROLE_RANKING`: rows marked
  ` | role=device` / ` | role=accessory`, judged under `device_role.constraint`'s own text;
  WARN when a care or tool row scores above 40, a watch 40 or less, or a storage piece at or
  above the lowest watch), and params "LLM memo off". In 10, the 10 'device' rows: read
  `core_top4` / `core_top12` / `first_core`. Watches sit in Fashion AND Electronics, so whether
  the device gate acts can differ between gte-large-en and the local gte-small; a gate that
  dropped the ask logs "Device-role ranking: device ask dropped (minority_pool | no_core_rows)"
  (no CSV column). `DEVICE_ROLE_RANKING` acts only inside `TYPE_AWARE_RANKING`, so 10's off
  arm has no device ordering; a true before also needs `app/device_role.py` and
  `app/catalog_taxonomy.py` copied in for the core columns (10's docstring). 12: "Determinism
  check (12)" below. Rerank-on numbers do not compare across 2026-10-27: QE prompt 94fffb8d ->
  d3e184c6, rubric 8e7b7f67 -> 089fb621, `row_text_case` 'original+fine_fields+role', and
  `prompts_sha` (0af957a6337a now) also hashes the cross-sell prompt, so it differs from every
  earlier run's even where no prompt changed. Locally (Groq qwen3.8-27b, 1 sample, memo off)
  "gift for a vintage watch collector" went from 0 to 12 of 12 watches on page 1 and 9 plain
  device asks from 0.86 to 0.94 core share in the top 12 (the 2026-10-27 session archive, an
  `oct27/` folder in that session's TEMP scratchpad on the dev laptop); nothing yet on
  gte-large-en + Qwen3-Next.

**`RERANK_INPUT_K` stays 15 (owner call 8, 2026-10-19), here too.** 24-30 on Databricks is an
open owner item (CLAUDE.md, Open work; its estimated quality and token cost:
.claude/rules/eval.md, "Apparel battery", BEFORE: rerank prompt ~2.5K at 15 rows with the
2026-10-19 rubric and row fields). To try it, set `RERANK_INPUT_K` in Cell 2's env (app.yaml for
the App) BEFORE the first `import app...` (it is read by value at import: CLAUDE.md, "llm_client
reads config dynamically"). Context set first, rerank on; read `fell_back`.

## Determinism check (12, 2026-10-27; NOT yet run on the workspace)

The Oct 27 meeting asked that the same search give the same results. Databricks takes no
seed and page_cache lives in memory (lost on every restart), so on the App only the LLM reply
memo (`app/llm_memo.py`, `LLM_MEMO`, on the `app_state` volume) makes a repeated search repeat
once page_cache no longer holds it. 12 runs `eval/check_determinism.py`
through `vs_shim` (gte-large-en Vector Search + the workspace LLM): per query, `runs` cold runs
(page_cache, the degraded store and the /product cache cleared before each) of page 1, page 2
and the top card's product page with its recommendations. With the memo on, every run must
equal run 1 (catalog_index lists, intents, constraints, bundle type, facets, match quality, ad
labels, cross-sell / upsell) and runs 2..N must make 0 provider calls; the table names the
fields that differ, per run. Needs the index ONLINE and FastAPI (as 10).
```python
exec(open("databricks/12_determinism.py").read())
verdicts = run()                       # the 21 default queries x 3 runs, memo ON (scratch folder)
run(queries=["gift for a vintage watch collector"], runs=2)
run(memo=False, runs=2)                # the model's raw stability: no pass mark, every run pays

# The restart case: dbutils.library.restartPython(), Cell 2, the exec line again, then the
# llm_memo folder run() printed: run 1 must replay the earlier session with 0 provider calls
run(memo_dir="/Volumes/dev/cas/evals/determinism/<you>/<time>/llm_memo")

# When done: the folder holds model output that can echo the queries
cleanup("/Volumes/dev/cas/evals/determinism/<you>/<time>")
```
- **Where**: a NEW folder per run, `/Volumes/<CAS_CATALOG>/<CAS_SCHEMA>/evals/determinism/<you>/<UTC
  time>/llm_memo` (`volume=` picks another scratch volume of your own), next to
  `snapshots-<UTC time>.json` (page ids, intents, facets per run). Never the App's `app_state`
  volume, in any letter case: 12 and `check_determinism` both refuse it, because a check must
  never read or pre-warm what the App serves.
- **Reading a FAIL**: the memo keys the rerank prompt, which holds the retrieved rows, and every
  run embeds its intents again (no query-embedding memo: owner call D6, open). So a FAIL that
  names `page1` / `page2` with provider calls on runs 2+ is retrieval drift -- a query vector
  that is not bitwise stable, or a Vector Search near-tie broken another way -- the one drift the
  memo cannot pin. A rebuilt index (`02 pause()` / `resume()`) may legitimately move near-ties:
  compare runs within one build.
- **Not `--subprocess`**: do not run `python -m eval.check_determinism --subprocess` from a
  notebook. Its run 1 goes to a fresh child process, which gets neither vs_shim's in-process
  `SEARCH_BACKEND=databricks` nor a Spark session. 12 runs the check in-process; its restart
  case above stands in for `--subprocess`.
- **Cost, ESTIMATED**: memo on, run 1 ~63 LLM calls and ~0.11M tokens for the 21 default
  queries (translate, rerank and the product page's cross-sell per query), runs 2..N 0 LLM calls;
  memo off, every run pays run 1's calls. Every run, memo or not, still embeds each search's
  intents and queries Vector Search. The printed totals are the real calls and tokens. Locally
  (Groq qwen3.8-27b, 6 queries, `--memo on --runs 3 --subprocess`, 2026-10-30) the check PASSed
  6 of 6: run 1 made 18 calls and 30,613 tokens, runs 2-3 were identical with 0 provider calls.

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

# C. Once: the App + resources (CAN QUERY on both endpoints, READ on the catalog Volume,
#    since 2026-10-27 READ + WRITE on the memo volume dev.cas.app_state, created first).
#    Created STOPPED: nothing bills yet
app_create()

# D. Grants for its service principal: USE CATALOG/SCHEMA, READ VOLUME, SELECT on the
#    index, read on the staged folder; since 2026-10-27 also creates the memo volume if
#    missing and grants READ + WRITE VOLUME on it.
#    RE-RUN after EVERY 02 resume() (new index object)
app_grant()

# E. Start + deploy (billing starts). Prints the URL. The backend loads ~1 min more:
#    <url>/logz should show "Start-up check index/embedding/llm/memo: OK" and "service ready"
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
`export_catalog()`, `01` or `02` re-run. `product_type` is derived from the titles at load (timing:
the 2026-10-19 note below; not yet measured on the App's 2 vCPU), never stored or embedded;
`TYPE_AWARE_RANKING` defaults to true.
After pulling the 2026-10-15 changes: `stage()` → `app_deploy()` only. `OUTFIT_BUNDLES_ENABLED`
defaults to false. The shop UI's light theme reaches the App ONLY through `STREAMLIT_THEME_FLAGS`
in `app_bundle/run_app.py`: `stage()` does not copy `.streamlit/`, so keep those flags in step
with `.streamlit/config.toml` (`tests/test_ui.py::test_light_theme_on_both_launch_paths` fails if
they drift). Without `base=light` the theme follows the viewer's OS and the white cards sit on a
dark page. Streamlit pin: CLAUDE.md, Gotchas (1.58). Title dedup on Databricks rows is now
case-insensitive: CLAUDE.md, Decisions (display casing).
After pulling the 2026-10-19 changes: `stage()` → `app_deploy()` only. No new staged files (the
skin lives in `ui_common.py`; 09 APP_SOURCES stages `ui*.py` and `pages/`). The theme flags
changed with the marketplace skin (`ui_common.SKIN`): primary / link `#007185`, text `#0F1111`,
border `#D5D9D9`, white backgrounds, plus `--theme.linkUnderline false`, `--theme.orangeColor
#FFA41C` (the rail's review stars) and `--theme.font "Arial, Helvetica, sans-serif"` (system
fonts only); `test_skin_is_the_theme` and
`test_light_theme_on_both_launch_paths` keep `SKIN`, `.streamlit/config.toml` and these flags
equal. After pulling the 2026-10-26 palette change (eBay-style colours, nothing else; CLAUDE.md,
"Marketplace-style UI"): `stage()` → `app_deploy()` only; the flags are now primary / link
`#0968F6`, text `#191919`, border `#8F8F8F`, `--theme.orangeColor #191919`,
`--theme.secondaryBackgroundColor #F7F7F7` (the rest unchanged). Load cost on the laptop (CLAUDE.md, Gotchas: `product_type`): `annotate` ~2.0-3.3 s before
the 2026-10-19 columns and ~3.6-6.4 s after, +2-3 s in paired runs; the four new columns add ~16
MB RSS (one local measurement, next to the App's 1,486 MB catalog + keyword index; CLAUDE.md,
Databricks App deployment); both unmeasured on the App's 2 vCPU / Medium. New
knobs: `REJECTS_AFTER_TAIL` and `VARIANT_COLLAPSE` default true, `WEAK_MATCH_MIN_STRONG` 0;
app.yaml sets none of them, so the App runs the defaults.

**After pulling 2026-10-27** (watch-gift / device ranking, deterministic results, "Related
items", the Electronics and Grocery rail; NOT yet run on the workspace): `stage()` →
`app_grant()` → `app_deploy()`.
- **Nothing to re-run on the data side**: no `export_catalog()`, `01`, `02` or re-embed. The
  taxonomy (`item_category`, `item_role`, `watch_kind`, `diet_tags`, `works_with`,
  `item_condition`, `brand`; `app/catalog_taxonomy.py`) is derived at every load, as
  `product_type` is, from columns `dev.cas.products` and the App export already carry (the
  vertical, the title, `dietary`, `store`, `specs`); never stored or embedded, and search_text
  is unchanged. Databricks has no CSV hash, so none of the laptop's `.cache/` applies: every
  App start and every notebook `load_index` derives it again, ~+13 s on the laptop, unmeasured
  on the App's 2 vCPU.
- **`app_grant()` is the new step.** `cas-app` predates the memo, and `app_create()` returns
  early for an existing App, so `app_grant()` is how it gets the memo volume: `CREATE VOLUME IF
  NOT EXISTS dev.cas.app_state` (needs CREATE VOLUME on the schema; an existing volume is OK
  without it), then READ VOLUME and WRITE VOLUME on it next to the old grants, each printed
  PASS / OK (inherited) / FAIL; with the volume missing, its two grants are not run. It also
  prints a WARN when `account users` can read or write the volume. Inside `dev` that is expected
  (they hold broad privileges there), but the volume holds model replies that can echo a
  shopper's words, and a writer can plant a reply the App replays: a client deployment needs its
  own schema where only the App holds WRITE VOLUME. A NEW App also gets the volume as an App
  resource from `app_create()` (`memo-volume`, "Can read and write", after the catalog volume);
  if that resource is refused, the create retries without it and `app_grant()`'s SQL grant
  covers it.
- **app.yaml**: `LLM_MEMO` "true", `LLM_MEMO_DIR` "/Volumes/dev/cas/app_state/llm_memo".
  `stage()` refuses any other `LLM_MEMO_DIR` and an `LLM_MEMO` other than 1/true/yes or
  0/false/no. REVERT = `LLM_MEMO: "false"`: 09 then neither creates nor grants the memo volume,
  and every search asks the LLM again. A missing memo volume is a WARNING in `app_deploy()` /
  `app_start()`, never a refusal: the App serves, but re-asks the LLM after every restart.
- **Check after the start**: `/logz` shows `Start-up check memo: OK`, and its `Effective config:`
  line (also `/healthz` → `config`) reads `prompts_sha=0af957a6337a` and
  `row_text_case=original+fine_fields+role`, as the laptop's does; a different sha means other
  prompts. The `X-LLM-Memo-Hits` response header counts the replies a request replayed.
- **Also in this round**: the bundle pins `pydantic==2.13.4`, `starlette==1.2.1` and
  `httpx==0.28.1`; `STREAMLIT_THEME_FLAGS` gained `--theme.sidebar.backgroundColor #FFFFFF`;
  `APP_SOURCES` is a glob plus an AST walk that stages any other repo module the App imports (0
  extras today). New knobs: `DEVICE_ROLE_RANKING` (default true) and `LLM_MEMO_MAX_ENTRIES`
  (50000) / `LLM_MEMO_MAX_AGE_DAYS` (0 = never); app.yaml sets none of them. A Volume store
  is trimmed to `LLM_MEMO_MAX_ENTRIES` only by `python -m app.llm_memo purge --to-limit` run
  against it, never on its own, and a running App keeps purged replies (in memory and in
  page_cache) until it restarts.
- **Never run on the workspace**: the SDK's `WRITE_VOLUME` App resource, CREATE VOLUME rights on
  `dev.cas`, the memo's put-if-absent there (hard links on the UC FUSE mount, the Files API's
  `overwrite=False`), the memo's latency on `/search`, the Python 3.11 runtime, and 08's new
  rows, 10's device rows and 12 on Qwen3-Next.

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
- `/logz` `Start-up check memo FAILED` → the memo volume is missing or the App lacks WRITE
  VOLUME on it: `app_grant()`, then `app_stop()` + `app_start()`. The App still serves; it only
  stops repeating searches across restarts. Each new search reads and writes that volume through
  the Files API (latency unmeasured on the App): if `/search` is slow, `LLM_MEMO: "false"` in
  app.yaml is the revert.
- A bad reply stays pinned: the memo keeps the FIRST reply a stage accepted, good or bad, until
  `python -m app.llm_memo purge` (e.g. `--query`) removes it and the App restarts.
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
  So: `pause()` after every run; keep the index only if you are back within ~4 days
  (`11_idle_guard.py` applies a 3-idle-day variant (Sai, 2026-10-15; assumes a ~$15-20
  stored-vector rebuild, unmeasured until `02 costs()`) -- see "Idle teardown").
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
- **2026-10-27 notebook steps, ESTIMATED (none run on Qwen yet; read the printed totals):** 08
  = 43 LLM calls (was 40), ~3.6 min, ~55K tokens; 10 = ~100-145 calls and ~0.24-0.38M tokens per
  repeat (48 queries); 12 = run 1 ~63 calls and ~0.11M tokens (21 queries, memo on), runs 2+ 0
  calls. The prompts grew: the QE prompt by ~256 prompt tokens per translate call (measured,
  gpt-oss-20b, context set), the rubric by ~165 per rerank call (size estimate); locally (Groq
  qwen3.8-27b, pages 1 + 2, 1 sample each) the 14 electronics queries both runs share averaged
  5,113 tokens a query against 4,649 on the pre-round code (2026-10-27).
- **The App with the memo (2026-10-27):** a repeated search (any spacing or letter case)
  replays from the memo with 0 LLM calls (if Vector Search returns the same rows: "Determinism
  check (12)"), and the product page's recommendations are cached per
  product and query, so a demo that repeats searches spends no new tokens on them. A NEW search
  adds Files-API reads and writes on the memo volume, and every start ~+13 s of taxonomy
  derivation (laptop figure): both unmeasured on the App. Measure `/search` p50 / p95 with
  `LLM_MEMO` true vs false before a demo.

### Idle teardown (Sai, 2026-10-15; NOT yet run on the workspace)
- `02 delete_endpoint()`, after `pause()`: deletes the endpoint too, ending the 24h it
  still bills after its last index goes. Refuses while the endpoint hosts ANY index, so it
  cannot take one down. `resume()` recreates it, adding endpoint creation time (unmeasured; the
  measured 47.8 min build already spent ~10 min in PROVISIONING_ENDPOINT with the endpoint up;
  `_ensure_endpoint` waits up to 60 min); the stored vectors stay, so no embedding calls.
- `11_idle_guard.py`: `check()` (dry run: prints what it would do) / `check(dry_run=False)`.
  Idle = no `queryIndex` audit event for `cfg.VS_INDEX` AND no (re)build for `IDLE_DAYS` = 3
  (Sai: past ~3 idle days at ~$7/day an estimated ~$15-20, ~48 min rebuild is cheaper; the stored-vector
  rebuild's billed $ is still unmeasured, the 2026-10-09 rule above says ~4 days for a managed
  one). Then `09 app_stop()` and `02 pause()` (index only; endpoint and vectors stay).
  Returns no-index / unknown / keep / would-delete / deleted. Safe on missing data: an
  unreadable audit table or no timestamps at all -> "unknown", nothing deleted.
- **UNVERIFIED assumptions, check before scheduling:** (a) the audit filter --
  `service_name = 'vectorSearch'`, `action_name IN ('queryIndex', 'queryVectorIndex')`,
  `request_params['name'|'index_name']` = the index -- was never matched against real rows;
  (b) the index info's `creation_timestamp` (top level or under `status`, epoch ms) is
  assumed; (c) `system.access.audit` must be enabled and readable. Failure mode of (a): the
  filter finds nothing, the BUILD time alone decides, and an index queried every day is torn
  down 3 days after `resume()`. So: query the index (e.g. `03`), wait out the audit lag
  (minutes to hours), run `check()` and confirm the printed "query" time is that query's.
- Scheduling (only after that): Workflows -> Create job -> a notebook that runs Cells 1-2
  (11 opens `databricks/...` by relative path, so it needs Cell 2's `os.chdir(REPO)`),
  `exec(open("databricks/11_idle_guard.py").read())` and `check(dry_run=False)`, daily, on
  the runbook's compute. The run-as identity needs SELECT on `system.access.audit` plus what
  02 and 09 need. The job's compute bills too: on `Retail_DE_cluster` (all-purpose) each daily
  run starts the cluster until auto-termination.

## What does NOT move yet

Local-only until agreed: latest enhancements under test, the fine-tune
pipeline (becomes a GPU job here later). Amazon Reviews 2023 is research-only —
fine for this internal migration, must be flagged before any client-facing use.
