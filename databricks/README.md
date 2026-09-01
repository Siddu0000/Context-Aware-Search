# Databricks migration — runbook

Decided on the 2026-08-13 call (Databricks = non-negotiable priority) and
sequenced on the 2026-08-24 call: migrate the FOUNDATIONAL, already-tested
solution now, while enhancements keep being developed/tested locally. New
features move over only once agreed. Component mapping was verified against
Databricks docs on 2026-07-29 — see `docs/RESEARCH.md` §3.

## Dependency checklist (Tarun / "Tanu")

- [ ] Workspace URL + PAT or OAuth credentials (entity account, shared budget)
- [ ] Unity Catalog: which catalog + schema we may create tables in
- [ ] Permission to create: one vector search endpoint, one serving query path,
      one Databricks App
- [ ] Confirm nobody has an active demo (our AI usage is heavier than other
      projects on this account — flagged to Niharika 2026-08-24)

## Step order (scripts in this folder)

1. `01_catalog_to_delta.py` — upload `data/products.csv` (300K rows) to a UC
   Delta table with Change Data Feed enabled (required for delta-sync).
2. `02_vector_index.py` — create a STANDARD vector search endpoint (GA;
   storage-optimized is overkill at 300K) + a delta-sync index with
   Databricks-managed embeddings (`databricks-gte-large-en`, 1024-dim).
   NOTE: prior gte-small eval numbers do NOT transfer — re-run
   `eval/run_eval.py --no-rerank` against the new index before quoting anything.
3. `03_smoke_test.py` — similarity query + one `databricks-gpt-oss-120b`
   pay-per-token call (same model family we run via Groq today, so
   translator/reranker prompts port unchanged; kills key_rotator + the Groq
   8000-TPM workaround).
4. `app.yaml` — deploy the Streamlit UI as a Databricks App (GA; Medium
   compute = 2 vCPU/6GB at 0.5 DBU/hr is enough once vector search replaces
   the in-process encode).

## Cost guardrails (from the calls — treat as rules)

- The vector endpoint bills from creation and only stops **24h after the last
  index is deleted** — idle ≠ free. Between demo windows: delete the index
  (`02_vector_index.py --teardown`) and recreate before the next demo
  (delta-sync re-syncs automatically).
- Stop the App when not demoing (it bills per hour while running).
- Compute auto-terminates at the account's 10-min idle setting — leave that.
- Report incurred token/DBU costs upfront; pricing of the offering itself is
  senior folks' problem, cost transparency is ours.

## What does NOT move yet

Local-only until agreed: latest enhancements under test, the fine-tune
pipeline (becomes a GPU job here later), eval reruns. Amazon Reviews 2023 is
research-only — fine for this internal migration, must be flagged before any
client-facing use regardless of platform.
