# CAS feasibility study: connectors, scalability, architecture and cloud fit

**Status.** Prepared 2026-11-02 for Niharika and Sai. Internal, not for clients. Evidence: the research of 2026-11-02 (eight research lanes and four gap follow-ups). Every research-lane finding was checked by an adversarial verifier, and where a verifier corrected a finding this study uses the corrected fact. The four gap follow-ups are newer and narrower and were not separately verified; where one conflicts with a lane, this study uses the gap file. Sections 3-8 were each checked against the evidence, and Sections 1-2 and 9-11 against Sections 3-8. Nothing here was measured on Databricks under load, and no load test exists anywhere in the repo.

### How to read this

- Each section opens with its answer, then gives the reasons.
- Every number and product fact carries a source in brackets. A source is one of three things:
  - a full URL, with the page's own date (or "fetched 2026-11-02" when the page shows none);
  - a repo path:line;
  - a scratch file, written SCR/... . SCR is the 2026-11-02 session archive: the scratchpad folder of the Claude Code session that wrote this study, in TEMP on the dev laptop and outside the repo. Its machine path is kept out, as for the oct19/ and oct27/ archives (.claude/rules/eval.md); TEMP gets cleaned, so persisting it is owed.
- CLAUDE.md and databricks/README.md line numbers refer to the committed files (git HEAD fe206d8). On 2026-11-02 the working copy of CLAUDE.md gained 8 lines (in the working copy, add 2 to lines 36-956 and 8 to later lines) and the working copy of databricks/README.md gained 6 lines after line 246 (add 6 to later lines). .claude/rules/eval.md gained lines only at its end, so its citations hold.
- "Phase 0-3" are the rollout phases of Section 9 (Phase 0 is the offline proof). The catalog compiler's own phases are written "compiler P0-P5".
- Each fact also says how it is known:
  - **measured**: we ran it and recorded the number;
  - **code**: read in the repository;
  - **documented**: a vendor page says so ("documented (repo)" means the project's own notes);
  - **estimate**: our arithmetic on measured or documented numbers, or a sizing in developer-days; never measured;
  - **inference**: our reading across sources, which no source states.
- Dev-days are days of one developer who knows the repo, tests included. They are developer time, not calendar time, and none has been validated.
- The recent quality numbers were measured on Groq's qwen3.8-27b preview model with the local embedding model, and were graded by two LLM grader prompts, not by people. None was measured on a configuration a client would run [SCR/gtm/research/claims.json CLA-69].

Terms used throughout:

| Term | Meaning |
|---|---|
| CAS | Context-Aware Agentic Search, this proof of concept (PoC) |
| LLM | Large language model: the AI model that rewrites the query and re-ranks results |
| API | Application programming interface: the routes other systems call |
| P0 accounts | The priority-zero target retailers: H&M, C&A, Kroger, Albertsons, H-E-B |
| SKU | One sellable item (one size and colour of a product) |
| vCPU | A virtual CPU core on a cloud machine |
| p50 / p90 / p95 | The value that 50% / 90% / 95% of measurements fall under (p50 = the median) |
| Intent | One search phrase the LLM writes from the shopper's words ("travel mug" from "keep my coffee hot") |
| Head / tail query | A short query that names a product ("black dress") / a long query that describes a need ("something breathable for a humid day") |
| UC | Unity Catalog, Databricks' data catalog |
| AI Search | Databricks' vector search service (formerly "Vector Search") |
| FM APIs | Databricks Foundation Model APIs: models hosted by Databricks |
| DBU | Databricks Unit, Databricks' billing unit |
| PPT / PT | Pay-per-token / provisioned throughput (reserved LLM capacity, billed by the hour) |
| ITPM / OTPM | Input / output tokens per minute, an LLM rate limit |
| QPS | Queries per second |
| BM25 | The standard keyword-ranking formula behind most text search engines |
| ANN | Approximate nearest-neighbour search, the way vector stores find similar vectors |
| RRF | Reciprocal rank fusion: merging two ranked lists by each item's position in them |
| CDF | Change data feed, Delta's log of changed rows |
| P@10 | The share of the first 10 results that are relevant |
| Strict / lenient | The share of result cards graded fully relevant / at least partly relevant |
| D-lite | Our recommended overlay (Section 7): head queries stay on the client's engine; tail queries get CAS's rewritten query and filters, still run on the client's engine |
| Patterns A-E | Section 7's options: A full replacement; B query-understanding layer; C rerank layer; D router (D-lite = router + B; D-full = router to CAS's own catalog path); E rank fusion |

## 1. Executive summary

| Question | Short answer | Confidence | Section |
|---|---|---|---|
| 1. Connectors: can CAS read client tables in Hive, BigQuery, Snowflake, Parquet and external databases? | **Not today.** CAS reads one file shape, its own comma-separated CSV, so every client table must first be hand-converted. That takes about a day per client (estimate), and a wrongly shaped file can fail quietly: of 11 client-shaped test catalogs, 4 loaded but served wrong or degraded results (measured). A designed adapter layer, not yet built, would read files and Unity Catalog tables directly, and BigQuery, Snowflake and SQL databases through Databricks Lakehouse Federation (read-only links from Unity Catalog) or through native readers. It is about 8.5-12 dev-days (estimate), plus admin lead time. Hive tables whose files sit in cloud storage can be federated (documented). Hive on on-premises HDFS (Hadoop's file system) needs an export first (documented), and so do transactional Hive tables until a client sample is tested (inference). The overlay we recommend needs no catalog copy in its first two phases (inference). | High that it does not work today (code, and a probe of 11 catalogs). Medium on effort (estimates). Low on per-source behaviour: no federation, Auto Loader or connector read has been run. | 3 |
| 2. Scalability: how far does it go? | **Shown only at demo size:** ~300K products, one process, one query at a time. No load test exists, and nothing was measured on Databricks under load. The first limits sit outside CAS: our Databricks LLM allows at most ~2.4 new searches per second per workspace on pay-per-token, and ~0.7 per second on gpt-oss-120b, the model an Azure-EU workspace would most likely use (estimates and upper bounds). Inside CAS, every API worker holds the whole catalog and its keyword index in memory: ~1.7 GB at 300K products and ~5.2 GB at 1M. From ~5M products the design must change (estimates). | Medium on where the limits are (documented limits plus arithmetic). Low on real throughput (no load test). | 6 (also 4, 5) |
| 3. Architecture: replace the client's search, or sit on top of it? | **Sit on top.** All five P0 accounts already run site search (documented; the engine is named for Kroger and H&M, Albertsons' AI layer is named, and H-E-B's and C&A's text engines are not). Start with D-lite: head queries stay on the client's engine untouched. Tail queries get CAS's rewritten query and filters, still run on that engine; this is 12-20 dev-days for the first engine (estimate), provided the engine accepts injected filters, which has not yet been confirmed for any target's engine. Add CAS reranking of the tail second: 17-28 dev-days in all (estimate). Today CAS can only replace a search: it returns rows from its own catalog copy, and it has never been connected to a client's live engine (code). | Medium. An offline simulation on a stand-in engine (16 queries) supports it. The lift on a real client engine is unmeasured. | 7 |
| 4. Cloud fit: is CAS cloud agnostic? | **Not as a fact today.** The Python core uses no cloud SDK and is portable (code). But the only cloud deployment is Databricks, run on one Azure workspace. Databricks itself runs on AWS, Azure and Google Cloud (documented), so that shape could follow a client, subject to model regions; only Azure has been run (inference). A native AWS, Azure, Google Cloud, Snowflake or Kubernetes deployment needs a container, API authentication and adapters. That is ~6-10.5 dev-days for a portable package and ~20-28 for the first cloud-native target (estimates). Two limits weigh more: our Databricks LLM, Qwen3-Next, is listed in no Azure EU region, and every word list, tokenizer, prompt, embedding model and eval set in CAS is English (code, documented). | High on what runs today. Medium on effort. The EU model question needs Databricks' written answer. | 8 (also 5) |

### Key numbers

- **One file shape loads today.** That shape is a UTF-8, comma-separated CSV with our 13 core column names. Of 11 client-shaped test catalogs, 6 crashed at boot, 4 loaded but served wrong or degraded results, and 1 ran but lost every department-specific feature. A Shopify export showed one blank product for every query, with HTTP 200, a success code (measured 2026-10-30; local backend, fake LLM) [SCR/oct27/byo/REPORT.md:11-33; app/search.py:348].
- **2 LLM calls per new search; 0 for page turns, filters, sorts and exact repeats** (code). Page 2 came from cache with 0 LLM calls on 29 of 29 flat held-out queries, median 2 ms (measured 2026-11-02, local) [app/main.py:1928-2022; app/main.py:2263-2369; SCR/nov02/REPORT.md:56; SCR/gtm/research/llm.json latency table].
- **~10 s per new search on Databricks** (measured). That is 10.2 s with Qwen3-Next and AI Search, rerank on, 18 keyword queries, in the eval harness, not the App (2026-10-12) [CLAUDE.md:860-864]. Locally on Groq qwen3.8-27b: median (p50) 3.6 s, p90 4.6 s (31 queries, 2026-11-02, rate-limit pauses removed) [SCR/gtm/research/latency_from_nov02.txt]. No search that uses the LLM has been measured below ~3 s.
- **Quality on 31 held-out queries.** On page 1, an average of 87% of cards per query were judged at least partly relevant, and 61% fully relevant. The same catalog searched by embeddings without an LLM scored 59% and 38%. Paired per query, the strict lift was +0.23 (25 wins, 4 losses, 2 ties). Measured 2026-11-02, with caveats:
  - Groq qwen3.8-27b, local setup;
  - two LLM grader prompts, blind, one live sample per query;
  - the 60 held-out queries were LLM-written; 34 were hand-picked for the live run with catalog supply checked, and 31 ran before the token cap.
  This is not a comparison with any retailer's engine [SCR/nov02/REPORT.md:50-83, 126-129; SCR/gtm/research/claims.json CLA-06, CLA-10]. The largest strict lift was electronics (+0.31, n=6), then fashion (+0.27, n=13) [SCR/gtm/research/claims.verify.json CLA-07].
- **Overlay simulation.** A plain word-match engine fed CAS's rewritten queries went from P@10 0.206 to 0.369-0.394 on 16 inferred-intent queries. CAS's own retrieval, reranker off, scored 0.463, so the rewritten queries closed about 63-73% of the gap (measured offline, 0 LLM calls; the stand-in engine was CAS's own BM25 keyword engine, not a client's; one gpt-oss-20b sample per prompt) [SCR/gtm/research/arch_sim/sim_patterns_out.json; SCR/gtm/research/architecture.verify.json ARC-12].
- **LLM ceiling: at most ~2.4 new searches per second (~145 per minute) per workspace** on pay-per-token Qwen3-Next. The limit is 100,000 output tokens per minute in the Enterprise-tier table; on gpt-oss-120b it is ~0.7 per second. Both are estimates and upper bounds: the per-search token counts were measured on Groq's qwen3.8-27b, a different model and tokenizer, and an undocumented default output reservation per request may lower the ceiling [https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits, ms.date 2026-10-30; SCR/gtm/research/gap_1.json G1-14; SCR/gtm/research/llm.verify.json LLM-24].
- **Memory: ~1.7 GB per API worker at 300K products, ~5.2 GB at 1M, ~25 GB at 5M.** These are estimates, extrapolated from a laptop; nothing was measured above 300K real rows. A Medium Databricks App has 6 GB [SCR/gtm/research/probe_scalability.json; https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/compute-size, ms.date 2026-09-25].
- **Cost: a fixed floor of ~$551 a month plus ~$1.46 per 1,000 new searches** (estimates, East US list prices):
  - the floor is one standard AI Search unit ($204) plus one Medium App ($347); it is $898 with the two App instances Databricks recommends;
  - the per-search figure is Qwen3-Next pay-per-token tokens;
  - at 1 new search per second around the clock with no cache hits, the total is ~$4,392 a month;
  - these price Qwen3-Next at East US list prices, with tokens counted on qwen3.8-27b; no Azure EU region lists Qwen3-Next, and on gpt-oss-120b at West Europe list prices the token line is ~$2.40 per 1,000 searches (Section 5).
  [SCR/gtm/research/gap_1.json G1-25, G1-26, G1-28, G1-33 and its unit price table; https://www.databricks.com/product/pricing/vector-search, fetched 2026-11-02; https://www.databricks.com/product/pricing/foundation-model-serving, fetched 2026-11-02]
- **Region** (documented):
  - Qwen3-Next is listed in 0 Azure EU or UK regions;
  - gpt-oss-120b is listed in the six Azure EU Data Boundary regions;
  - Qwen3-Next is a Public Preview model offered under provisioned throughput in no region, and Private Link to Databricks-hosted models needs provisioned throughput, so a client that requires private connectivity to the LLM cannot use Qwen3-Next today (the last step is an inference);
  - in our one Databricks comparison, gpt-oss-120b took 30.3 s per query against Qwen3-Next's 10.4 s (keyword set, 2026-09-23, quality tie; measured).
  [https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability, ms.date 2026-10-30; https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/compliance, ms.date 2026-10-28; SCR/gtm/research/gap_4.json G4-01, G4-03, G4-08, G4-09; CLAUDE.md:777-780]
- **All 5 of 5 P0 accounts already run site search** (documented; H-E-B's and C&A's text engines are unnamed):
  - Kroger: Empathy.co, since 2020;
  - H&M: Voyado, rebuilt in 2026;
  - Albertsons: Google's Conversational Commerce agent plus an OpenAI-model assistant, keyword engine unnamed;
  - H-E-B: built in-house, engine unknown;
  - C&A: Syte for visual search, text engine unknown.
  Public cloud signals (2019-2026 sources) show Azure and Google Cloud for three accounts each, Databricks for two (Kroger's 84.51° and H&M) and Snowflake for one (C&A) [SCR/gtm/research/gap_2.json G2-06, G2-15, G2-25; SCR/gtm/research/connectors.verify.json CON-40..45]. These are announcements and job ads; current stacks may differ (inference).

### What we recommend

1. **Sit on top, in two steps.** First D-lite (a router plus query understanding into the client's own engine), then CAS reranking of the tail. Do not offer full replacement or rank fusion to the P0 accounts (Section 7).
2. **Prove it before building it.** For each client, run a 2-3 week offline proof on its own query logs against a read-only sandbox of its engine (estimate) [SCR/gtm/research/architecture.json recommendations[1]]. Start with Kroger's Empathy.co, the one named engine a target runs itself [SCR/gtm/research/gap_2.json G2-24].
3. **Settle the outside limits early.** Confirm each client's cloud and region in discovery. Get Databricks' written answer on Qwen3-Next in the EU and on provisioned throughput. Run a workspace load test before any throughput or SLA (service-level agreement) statement. Before any client-facing service, add LLM timeouts and output caps and make the seed setting explicit (Sections 5, 6, 8).
4. **Build connectors only when a client needs CAS to hold its catalog**, for D-full or a CAS-hosted demo: compiler P0 plus a first wave of readers (one federation proof on Databricks, then native Snowflake, BigQuery and SQL readers) is ~8.5-12 dev-days, plus admin lead time (estimate). Until then, a hand-shaped CSV covers a demo (~1 day per client, estimate) (Section 3).
5. **Describe it as it is.** Say "a cloud-portable core with a Databricks-native reference deployment", "English catalogs and queries today" and "connectors designed, not built". If a client will not run Databricks, the overlay is the easiest pattern to move: the portable package (container, API authentication, settings, seed handling), token-provider authentication and an acceptance smoke test come to ~8-11.5 dev-days, with no vector store needed (estimate and inference; Sections 7, 8, 9).
6. **Clear the legal items before packaging anything:**
   - the network-use clause (section 13) of the AGPL v3, the GNU Affero General Public License, on any code handover [LICENSE:1-2];
   - the research-only Amazon Reviews 2023 demo data [.claude/rules/safety.md:190-195].

## 2. What CAS is today

**In one paragraph.** CAS is a two-stage AI search layer, a proof of concept on a public research catalog of 299,972 products (fashion, electronics, grocery) [CLAUDE.md:851-853; SCR/gtm/research/claims.json CLA-60]. An LLM turns the shopper's words into search phrases and constraints. Vector search finds candidates for each phrase. A second LLM call re-scores the best 15, and ordering code arranges the page (code) [app/translator.py:580-677; app/search.py:494-548; app/reranker.py:275-380; app/main.py:1921]. It is meant to sit inside a client's existing search stack, not to be a standalone product [CLAUDE.md:1297-1303].

### The pipeline, step by step

1. **Guard.** Before any LLM sees the query, code cuts it at its first instruction-like marker. A query that opens with an instruction gets a plain embedding search and no LLM call (code) [app/translator.py:604-614; app/config.py:94].
2. **Translator (LLM call 1).** One call writes the search phrases ("intents"), typed constraints and a bundle type:
   - phrases: usually 3; 30 of 31 held-out queries had 3, one had 9 (measured) [SCR/gtm/research/gap_1.json G1-01];
   - constraints: gender, dietary, material, budget or other [app/translator.py:427-440];
   - bundle type: recipe or setup; outfit bundles have been off since 2026-10-15 [app/config.py:203].
3. **Retrieval.** Each phrase is turned into a vector and matched against the product vectors. The local backend uses gte-small vectors scanned in memory; the Databricks backend uses gte-large-en and an AI Search index. The store returns only (row number, score), and CAS fills in each full product row ("hydrates" it) from its own in-memory copy of the catalog (code) [app/search.py:519, 537-548; app/vector_search.py:240-274].
4. **Reranker (LLM call 2).** The top 15 rows (RERANK_INPUT_K) are scored 0-100, each with a one-line reason. A prompt rule tells the model to cap rows that break a stated attribute (colour, type, gender, size, diet; for clothing and shoes, age) at 40. The model applies the cap; code does not enforce it. A budget is not capped, and capped rows are demoted, not removed. The average rating then breaks near-ties (weight 0.05) (code) [app/reranker.py:275-380; app/config.py:120, 129; SCR/gtm/research/claims.verify.json CLA-21].
5. **Ordering** (app/main.py). This step does five things (code) [app/main.py:930, 1009, 1280, 1617, 1825, 1921]:
   - product-type, gender, age and device tiers;
   - rejected rows after the unjudged ones;
   - sponsored items gated on the rerank score and always labelled;
   - colour and size variants collapsed into one card;
   - facets for the filter rail.
6. **API and UI.** FastAPI routes serve the pipeline: /search, /keyword_search, /unified_search, /product, /chat, /feedback, /healthz and /stats. A Streamlit UI serves the demo (code) [app/main.py:228, 240, 2064, 2423, 2459, 2530, 2623, 2738; ui.py].

**Calls per action** (code):
- A new search makes 2 LLM calls; on Databricks it also makes 1 embedding call and ~3.2 AI Search queries.
- A product page adds 1 cross-sell call.
- Each Shopping Assistant chat turn adds 1 call.
- An API caller that omits `recommend=false` pays a third call on page 1 of every new search, because the default adds cross-sell (on Databricks, also about 5 more embedding calls and 5 more AI Search queries).
[app/main.py:1928-2022, 2070-2072; app/config.py:162; app/assistant.py:15; SCR/gtm/research/gap_1.json G1-01, G1-02]

**Caches** (code):
- **page_cache** keeps each query's ranked pool, 64 pools per process. Page 2, filters and sorts then need 0 LLM calls [app/config.py:131; app/page_cache.py:59-93].
- **The LLM reply memo** stores the first accepted reply to each exact request and replays it. An identical search then makes 0 provider calls. Measured on 2026-10-30 (Groq qwen3.8-27b, local): 6 of 6 queries were identical across 3 runs, with 0 calls on runs 2-3. The memo pins the first answer, good or bad [app/llm_memo.py; app/config.py:143-158; CLAUDE.md:591-616; .claude/rules/eval.md:500-509].
- **Single flight:** concurrent identical searches share one pipeline run [app/main.py:1415, 1453].
- All three live in one process. Only the memo's store (SQLite locally; a UC volume on the App, built but not yet deployed) is shared [SCR/gtm/research/scalability.json SCA-09, SCA-10].

```
 Shopper query
      |
      v
 [Guard] -- opens with an instruction? --> plain embedding search, no LLM
      |
      v
 [1. Translator]  LLM call 1 -> ~3 search phrases + constraints + bundle type
      |
      v
 [2. Retrieval]   per phrase: embed -> vector index  (local numpy | Databricks AI Search)
      |           -> (row, score) -> hydrate full rows from the in-memory catalog copy
      v
 [3. Reranker]    LLM call 2 -> top 15 scored 0-100 with a reason; small rating blend
      |
      v
 [4. Ordering]    type / gender / age / device tiers, rejects last,
      |           ads gated + labelled, variant collapse, facets
      v
 [page_cache]     ranked pool kept per query + settings -> pages, filters, sorts: 0 LLM calls
      |
      v
 [5. FastAPI]  -> Streamlit demo UI, or a client system

 LLM reply memo: every accepted LLM reply is stored; an identical request replays it
 (0 provider calls). Retrieval still runs on a memo hit; only a page_cache hit skips it.
```

### The two deployment shapes

| Shape | Retrieval | LLM | Hosting | Where it has run |
|---|---|---|---|---|
| Local (default) | gte-small vectors scanned in memory; full catalog and keyword index in the process | An OpenAI-compatible endpoint with a fixed key (a host that refuses `seed` needs a code change: Section 5). The default base URL is Groq (US); Gemini and Anthropic backends also exist | uvicorn + Streamlit | One Windows dev laptop (measured: every recent quality number) [app/config.py:66, 82, 262; CLAUDE.md:791-796] |
| Databricks | gte-large-en endpoint + an AI Search index built from stored vectors; the catalog is still held in each process | Qwen3-Next 80B (Public Preview) through the FM APIs | A Databricks App: FastAPI on 127.0.0.1:8001 plus Streamlit on the public port; one API worker; workspace users only | One Azure workspace. An earlier version was created there on 2026-10-13 and served the 2026-10-14 demo. The current version (with the memo) is not deployed, and its memory and start-up on 2 vCPUs are unmeasured (documented (repo)) [databricks/app_bundle/run_app.py:88-91; databricks/app_bundle/app.yaml:21-60; CLAUDE.md:1686, 1785, 1796, 1805] |

### The seams that make it adaptable

| Seam | What it allows | Where (code) | Limit today |
|---|---|---|---|
| Catalog loader | One loader per backend produces the in-memory catalog; a designed (not built) catalog compiler would write the same shape, so ranking code would not change | app/search.py:306-348; app/vector_search.py:106-141; SCR/oct27/byo/REPORT.md:47-58 | Reads only our 13-column CSV (Section 3) |
| Retrieval seam `_raw_hits` | Per phrase, a best-first list of (row number, score); a new vector store is an adapter here | app/search.py:537-548 | Two backends; `SEARCH_BACKEND` is not validated; the key must be a dense row number (Section 4) |
| OpenAI-compatible LLM client | Three settings (base URL, key, model) switch host; Gemini and Anthropic backends exist; a self-refreshing token for Databricks | app/llm_client.py:232, 269-292, 295, 381; app/config.py:65-73 | Only Databricks gets refreshing tokens; a fixed seed goes to every non-Databricks host; no timeout (Section 5) |
| Reranker as a function | Takes candidate rows and constraints, returns scored rows; could reorder another engine's top 15 | app/reranker.py:275-380 | Reads CAS column names; returns at most 15 rows (Section 7) |
| Translator that needs no catalog | Query in, intents and constraints out; the core of the overlay | app/translator.py:580-677 | No endpoint exposes it alone yet (Section 7) |
| Keyword-first router | /unified_search runs BM25 first and calls CAS when the top hit covers under 75% of the query's words | app/hybrid_router.py:11-50; app/main.py:2459-2527 | No tests, no UI page calls it, never measured live (Section 7) |

**What CAS does not do** (out of scope, on hold or not built):
- online A/B testing; visual search; multilingual search; SLA or SOC 2 (a security audit report);
- purchase-history personalisation; store-level inventory;
- a merchandiser console; an analytics dashboard; autocomplete; agentic checkout.
[CLAUDE.md:1332-1340; SCR/gtm/research/claims.json CLA-67]

The offline test suite (476 passing tests on fakes, 2026-10-27 count) guards the code, not quality or scale [CLAUDE.md:36-41].

## 3. Data connectors: can CAS read the client's tables?

Answer: not today. CAS reads one file shape, our own comma-separated CSV, so every client table has to be hand-converted into it first (code, and a probe on 2026-10-30). The platform pieces for reading these sources already exist on Databricks, though we have run none of them. Lakehouse Federation (read-only links from Unity Catalog, Databricks' data catalog, to outside systems) reaches BigQuery, Snowflake, the main relational databases and Hive metastores whose files sit in cloud storage, and Auto Loader loads Parquet, ORC, Avro, JSON and CSV files (documented). What CAS lacks is one adapter layer that reads those sources and writes our catalog shape. It is designed, not built: about 8.5-12 developer-days for files, Unity Catalog tables, BigQuery, Snowflake and SQL databases (estimate; the low end assumes the compiler's optimistic 4 days, and the byo report advises planning 5-7), plus waiting time for an administrator to create the connections, which is in no estimate. Hive tables stored on on-premises Hadoop (HDFS) need an export first: federation does not support HDFS (documented), and a JDBC route exists but is untested. Transactional (ACID) Hive tables should be treated the same way until a client sample is tested (inference). [app/search.py:348; SCR/oct27/byo/REPORT.md:11-33, 400-412; https://learn.microsoft.com/en-us/azure/databricks/query-federation/, ms.date 2026-09-29; https://learn.microsoft.com/en-us/azure/databricks/ingestion/cloud-object-storage/auto-loader/, ms.date 2026-10-15; SCR/gtm/research/connectors.verify.json CON-35; SCR/gtm/research/gap_3.json G3-03, G3-07, G3-21]

### What CAS reads today

- Locally: one bare pandas read of the file at PRODUCTS_CSV. It must be comma-separated UTF-8 (gzipped .csv.gz also works) with our column names: 13 Amazon-era core columns such as Product_title, bsns_vrtcl_name and parent_asin; the enriched demo file carries 26 (code; the 26 counted by the 2026-11-02 probe) [app/search.py:348; CLAUDE.md:886-887; SCR/oct27/byo/REPORT.md:21; SCR/gtm/research/probe_scalability.json "columns_raw": 26]. Our own demo file is served whenever it exists, unless PRODUCTS_CSV is set explicitly (code) [app/config.py:29-32].
- On Databricks: notebook 01 reads one CSV path (a file, or a folder of CSV parts sharing one header) on a Unity Catalog volume with Spark, comma-delimited UTF-8 only, asserts our search columns, casts rating_number without checking it exists, and overwrites the products table (code) [databricks/01_catalog_to_delta.py:20, 37-43, 50, 69-76; SCR/oct27/byo/REPORT.md:22-25]. The deployed App has no Spark: at start-up it downloads a CSV export of that table through the Files API (code) [app/vector_search.py:107-122, 144; databricks/09_deploy_app.py:119-127].
- No connector library is installed. Neither requirements file lists SQLAlchemy, Snowflake, BigQuery, Hive, Trino or Excel packages (code; pip list of the local environment, 2026-11-02) [requirements.txt:1-14; databricks/app_bundle/requirements.txt:12-29; SCR/gtm/research/connectors.verify.json CON-05]. The App bundle does pin pyarrow 24.0.0, so the Parquet crash below comes from the loader calling read_csv on it, not from a missing library (code) [databricks/app_bundle/requirements.txt:16; app/search.py:348; SCR/oct27/byo/design_RECOMMENDED.md:327].
- The bring-your-own-catalog (byo) probe (2026-10-30) fed 11 client-shaped catalogs to a copy of the code (a fake large language model, LLM, in place of the real one; local backend; no Spark). 6 crashed at boot: a Google Merchant Center TSV (tab-separated) feed, Parquet, a SQLite file, a sqlite URL, JSON Lines and a semicolon CSV. 4 loaded but served wrong or degraded results, and 1 ran but lost every department-specific feature. The worst, a Shopify export, showed one blank product for every query, with HTTP 200 (measured) [SCR/oct27/byo/REPORT.md:30-33, 163-187].
- A client table already in Unity Catalog (UC) has no direct path in: it must be exported to a CSV in our shape first (code reading) [SCR/oct27/byo/REPORT.md:129].
- So a pilot on today's code means about one day of hand reshaping per client: a SQL view that renames columns and normalises prices, gender and ratings. Every silent failure from the probe stays possible (estimate) [SCR/oct27/byo/REPORT.md:35-39, 131-140].

### Source by source

How to read the table. "Today" is code plus the measured probe. The two path columns and the refresh column are documented vendor behaviour unless marked (inference). Effort is developer-days on top of the designed compiler P0, the catalog compiler's first phase (next subsections), for one developer who knows the repo, tests included: all estimates, none measured [SCR/gtm/research/connectors.verify.json CON-35; SCR/gtm/research/gap_3.json G3-31]. "0 on Databricks" means our code needs only the generic uc:// reader; someone must still create the connection (see Privileges). No federation, Auto Loader or JDBC (Java Database Connectivity, a standard database-driver interface) read has been run, and no off-Databricks reader has queried a live server; the only runs are the offline pandas 3 tests further down. Source keys in brackets are listed under the table.

| Source | What CAS reads today | Path on Databricks | Path off Databricks | Incremental refresh | Effort (dev-days, estimate): on / off Databricks | Main risk |
|---|---|---|---|---|---|---|
| Hive, files on cloud storage, plain (non-transactional) tables | Nothing [probe] | External Hive metastore (HMS) federation: metastore 0.13 / 2.3 / 3.1 on MySQL, SQL Server or PostgreSQL; Delta, CSV, JSON, Avro, Parquet, ORC, text, SerDe (custom-format) tables, views, partitions; Iceberg in Public Preview; then uc:// [hms] [hms-ext] [ext] | impyla or Trino client over HiveServer2 [hive-py] | updated_at watermark; no Hive change capture checked | 0 (plus the heaviest admin setup) / 1.5 [est] | Needs a storage credential and external location on the client's storage; partitions or an Avro schema outside the table folder cannot be read [hms]. From our Azure workspace, S3 is read-only (documented) and HMS federation over S3 is not documented, so plan an AWS workspace or a copy (inference from its absence) [s3az] [hms] |
| Hive on on-premises HDFS | Nothing [probe] | Not supported: on-prem HDFS and remote (Thrift) metastore mode are listed as unsupported [hms-aws]. Either the client copies tables to cloud storage as Parquet / ORC, or a JDBC UC connection to HiveServer2 (generally available on Databricks Runtime 19+, no foreign catalog) [jdbc] | Same HiveServer2 clients, with network into the client's data centre (inference) | updated_at watermark | Export is client work, not estimated; JDBC path not estimated, untested | Open Spark bug: Hive JDBC returns column names in place of row data [jdbc]; Kerberos / LDAP logins |
| Hive transactional (ACID) tables | Nothing [probe] | Not listed as supported or unsupported for federation; Databricks lists "ACID fine-grained updates" as unsupported and a Spark ticket closed in 2021 says Spark does not support Hive transactional tables [acid]. Treat as export first (inference) | HiveServer2, where Hive resolves the change files (inference) | updated_at watermark | As HDFS | Untested on any table |
| Google BigQuery | Nothing [probe] | Query federation: Databricks Runtime (DBR) 16.1+ or a pro / serverless SQL warehouse, a Google service-account JSON key, reads through the BigQuery Storage API; then uc:// [bq] | google-cloud-bigquery + db-dtypes, Storage Read API to Arrow [bq-py]; resolves with our pandas 3 pin and passes an offline import test, no live query run [pd3] | CHANGES (needs enable_change_history = TRUE) or APPENDS; both cost BigQuery compute [cdc] | 0 / 1.0 [est] | Views are always materialized at extra BigQuery cost; INTERVAL columns make a table unreadable; BIGNUMERIC defaults to a type that holds only values below 1 [bq]; cross-cloud from our Azure workspace (inference) |
| Snowflake | Nothing [probe] | Query federation for every table, over Snowflake's driver on the client's Snowflake compute; catalog federation only for Snowflake-managed Iceberg tables (DBR 16.4 LTS, long-term support, or later); then uc:// [sf] | snowflake-connector-python with its pyarrow extra and fetch_arrow_batches (from code reading of the 5.1.0 package; no live query); its pandas extra needs pandas below 3 [sf-py] [pd3] | Streams: advance only inside a write transaction, go stale after the table's retention [cdc] | 0 / 1.5 [est] | Whole-number NUMBER columns with NULLs arrive as floating-point numbers, 123 as 123.0 (select ids as text) [sf-py]; client warehouse cost |
| Parquet, ORC, Avro, JSON, JSON Lines files | Parquet and JSON Lines crash at boot [probe] | Auto Loader or COPY INTO (JSON, CSV, XML, Parquet, Avro, ORC, text) into Delta, each file loaded once; then uc:// [files] | compiler P0 readers for Parquet and JSON Lines [p0]; pyarrow.dataset for ORC [oss]; Avro needs a library not yet chosen | Auto Loader picks up new files; text_hash diff | Parquet / JSON Lines in compiler P0; ORC / Avro 0.25; Auto Loader landing 1.0 (compiler P2) [est] [p0] | Nested fields need flattening (inference) |
| Delta or Iceberg tables in object storage | Nothing [probe] | UC external table, or catalog federation; then uc:// [fed] [ext] | deltalake (delta-rs), pyiceberg; both work with pandas 3 in an offline test [oss] [pd3] | Delta change data feed within its retention window [cdc]; Iceberg not checked | 0 / 1.0 readers + 1.0 for s3://, abfss://, gs:// addresses [est] | Per-cloud storage credentials; integers with NULLs become floating-point numbers [pd3] |
| Client's own Databricks UC table or view | No direct path: export to CSV first [probe] | uc:// (spark.table, every column cast to text), designed in compiler P0, checked by code reading only [p0] | Databricks SQL Connector, fetchall_arrow, service-principal OAuth [uc-py] | Delta change data feed (opt-in per table) [cdc] | 0 (in compiler P0) + 0.5-1 workspace proof [est] | 01 overwrites a same-named products table; reading to pandas above ~1M rows unmeasured [p0] |
| Delta Sharing (now OpenSharing) | Nothing [probe] | Databricks-to-Databricks share mounted as a read-only catalog; then uc:// [share] | delta-sharing package; latest release needs pandas below 3, so pin 1.4.1 [pd3] | Change feed only if the provider shares the table with history [share] | 0 / 0.5 + 0.25 for the pin [est] | Provider must share with history |
| PostgreSQL, MySQL, SQL Server | Nothing; compiler P0 adds only a SQLite file reader [probe] [p0] | Query federation [fed]; Lakeflow Connect (managed ingestion) change capture at volume, release state per connector not checked [lfc] | One SQLAlchemy "sql:" reader [sqla] | Lakeflow change capture, or updated_at watermark | 0 / 1.5 [est] | Snapshot load on the client's live operational database (inference) |
| Oracle, Teradata, Amazon Redshift | Nothing [probe] | Query federation [fed] | SQLAlchemy: Oracle built in; teradatasqlalchemy, sqlalchemy-redshift [sqla] | updated_at watermark | 0 / +0.25-0.5 per extra driver [est] | Driver install and licensing (unverified) |
| Azure Synapse | Nothing [probe] | Query federation [fed] | SQL Server dialect (inference) | updated_at watermark | 0 / 0.5 [est] | Entra ID login, firewall rules (inference; no Synapse page fetched) |
| Microsoft Fabric (OneLake) | Nothing [probe] | OneLake catalog federation: DBR 18.0+, standard access mode only, read-only; no arrays, maps, structs, views [fabric] | SQL endpoint on TCP 1433, Entra ID users or service principals only [fabric] | Watermark; no change capture checked | 0 (plus a Fabric admin) / 1.0 [est] | Fabric admin must enable three tenant settings and a workspace setting [fabric] |
| AWS Glue catalog | Nothing [probe] | Glue federation is documented for Databricks on AWS only [glue] | AWS SDK for pandas, pyiceberg Glue catalog [oss] | Not checked | 0 on an AWS workspace / 1.0 [est] | No path from our Azure workspace (inference) |
| CSV, TSV, ";" or "|" files, .gz | Comma UTF-8 only; TSV and ";" crash, or load as one column [probe] | Volume file read by 01 today (comma UTF-8 only); Auto Loader later [files] [probe] | compiler P0 all-text reader with delimiter and encoding checks [p0] | New files; text_hash diff | In compiler P0 [p0] | Silent misreads of prices, encodings, delimiters [probe] |
| Excel (.xlsx, .xls) | No Excel engine installed (pip list, 2026-11-02) [SCR/gtm/research/connectors.verify.json CON-05, CON-20] | Native on DBR 17.1+ (Auto Loader, read_files, COPY INTO) [xlsx] | pandas.read_excel plus openpyxl or calamine, a new dependency [xlsx] | File snapshot; text_hash diff | 0 / 0.5 [est] | One header row only; merged cells fill only the top-left cell; no password-protected files [xlsx] |
| Google Merchant Center (GMC) feed | TSV feed crashes at boot [probe] | Feed file to a volume, then the compiler P0 TSV preset; XML through a planned parser [p0] | Same file; the Content API for Shopping reached sunset on 2026-08-18 (requests without an extension fail intermittently from 2026-09-01; final shutdown planned for early 2027), so any API puller must use the Merchant API [gmc] | Full feed snapshot; text_hash diff | TSV in compiler P0; XML 1.0 (not in the byo total); Merchant API puller 2.0 [est] | Feeds carry no ratings, and 01 fails without a rating_number column (code reading; Spark not run) [probe] |
| Shopify | CSV export serves one blank product, HTTP 200 [probe] | Lakeflow Connect for Shopify, in Beta, enabled by a workspace admin on the Previews page; then uc:// + a preset [shop] | compiler P0 Shopify CSV preset; Admin GraphQL bulk export as JSON Lines [shop] [p0] | Lakeflow captures product and variant deletions; a column rename needs a full refresh [shop] | Preset in compiler P0 (dropped in its 3-day cut); Lakeflow preset 1.0; bulk API puller 2.0 [est] [p0] | Beta; old and new CSV header names; variants on following rows sharing the URL handle (implied by Shopify's help page) [shop] |
| Salesforce Commerce Cloud (SFCC), SAP Commerce Cloud, commercetools | Nothing [probe] | No managed connector named on the fetched Lakeflow page; export file to storage, then Auto Loader [lfc] [suites] | Export file + a new preset, or the platform's product API [suites] | SFCC delta exports need Salesforce Support to enable them [suites] | SFCC 3.0, SAP 2.5, commercetools 2.0: one figure each, on or off Databricks [est] | SFCC export format not inspected; commercetools "staged" drafts vs "current" live data [suites] |

Source keys for the table:
- [probe] SCR/oct27/byo/REPORT.md:30-33, 163-187 (measured 2026-10-30: local backend, fake LLM, no Spark) and REPORT.md:22-26, 129, 241 (code reading; Spark not run).
- [p0] SCR/oct27/byo/REPORT.md:400-412 and SCR/oct27/byo/design_RECOMMENDED.md:322-341, 625-635 (design, not built).
- [est] SCR/gtm/research/connectors.json CON-35 (as corrected in SCR/gtm/research/connectors.verify.json CON-35) and its source table (connectors.json tables[0]), and SCR/gtm/research/gap_3.json G3-31 (estimates).
- [fed] https://learn.microsoft.com/en-us/azure/databricks/query-federation/ (ms.date 2026-09-29).
- [hms] https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-concepts (ms.date 2026-09-11).
- [hms-aws] https://docs.databricks.com/aws/en/query-federation/hms-federation-concepts (Last updated Sep 11, 2026).
- [hms-ext] https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-external (ms.date 2026-10-06).
- [ext] https://learn.microsoft.com/en-us/azure/databricks/tables/external (ms.date 2026-10-06).
- [acid] https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-hive-compatibility (Last updated Sep 11, 2026); https://issues.apache.org/jira/browse/SPARK-15348 (resolved 2021-05-25).
- [jdbc] https://learn.microsoft.com/en-us/azure/databricks/connect/jdbc-connection (ms.date 2026-10-02); https://issues.apache.org/jira/browse/SPARK-45403 (created 2023-10-03, open).
- [s3az] https://learn.microsoft.com/en-us/azure/databricks/connect/unity-catalog/cloud-storage/s3/s3-external-location-manual (ms.date 2026-10-06).
- [hive-py] https://github.com/cloudera/impyla ; https://github.com/trinodb/trino-python-client ; https://github.com/dropbox/PyHive (fetched 2026-11-02).
- [bq] https://learn.microsoft.com/en-us/azure/databricks/query-federation/bigquery (ms.date 2026-10-09).
- [bq-py] https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.table.RowIterator ; https://docs.cloud.google.com/bigquery/docs/reference/storage (fetched 2026-11-02).
- [sf] https://learn.microsoft.com/en-us/azure/databricks/query-federation/snowflake (ms.date 2026-10-13); https://learn.microsoft.com/en-us/azure/databricks/query-federation/snowflake-catalog-federation (ms.date 2026-10-06).
- [sf-py] https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-pandas ; https://docs.snowflake.com/en/user-guide/key-pair-auth (fetched 2026-11-02).
- [pd3] SCR/gtm/research/gap_3.json G3-24, G3-25, G3-29, G3-30 (offline resolve and smoke test, measured 2026-11-02, no live server); G3-26, G3-27 (PyPI metadata, documented); G3-28 (code reading of the Snowflake 5.1.0 package).
- [files] https://learn.microsoft.com/en-us/azure/databricks/ingestion/cloud-object-storage/auto-loader/ (ms.date 2026-10-15); https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/copy-into/ (Last updated Oct 6, 2026).
- [oss] https://arrow.apache.org/docs/python/dataset.html (Arrow v26.0.0 docs); https://py.iceberg.apache.org/ ; https://delta-io.github.io/delta-rs/latest/ ; https://aws-sdk-pandas.readthedocs.io/en/stable/ (fetched 2026-11-02).
- [xlsx] https://docs.databricks.com/aws/en/query/formats/excel (Last updated Sep 11, 2026); https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html (fetched 2026-11-02).
- [uc-py] https://docs.databricks.com/aws/en/dev-tools/python-sql-connector (Last updated Sep 29, 2026).
- [share] https://docs.databricks.com/aws/en/delta-sharing/ (Last updated Oct 27, 2026); https://docs.databricks.com/aws/en/delta-sharing/read-data-databricks (Last updated Sep 24, 2026).
- [sqla] https://docs.sqlalchemy.org/en/20/dialects/ (2.0 docs, fetched 2026-11-02).
- [lfc] https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/ (Last updated Oct 30, 2026).
- [fabric] https://learn.microsoft.com/en-us/azure/databricks/query-federation/onelake (ms.date 2026-10-28); https://learn.microsoft.com/en-us/fabric/data-warehouse/connectivity (ms.date 2026-09-09).
- [glue] https://docs.databricks.com/aws/en/query-federation/hms-federation-glue (Last updated Oct 13, 2026).
- [cdc] https://docs.databricks.com/aws/en/delta/delta-change-data-feed (Last updated Oct 9, 2026); https://docs.snowflake.com/en/user-guide/streams-intro ; https://docs.cloud.google.com/bigquery/docs/change-history (fetched 2026-11-02).
- [gmc] https://developers.google.com/shopping-content/guides/deprecation-and-sunset (fetched 2026-11-02).
- [shop] https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/shopify (Last updated Oct 9, 2026); https://help.shopify.com/en/manual/products/import-export/using-csv ; https://shopify.dev/docs/api/usage/bulk-operations/queries (fetched 2026-11-02).
- [suites] https://salesforcecommercecloud.github.io/b2c-dev-doc/docs/current/jobstepapi/html/api/jobstep.ExportCatalog.html ; https://salesforcecommercecloud.github.io/b2c-dev-doc/docs/current/jobstepapi/html/api/jobstep.CatalogDeltaExport.html ; https://learning.sap.com/courses/mastering-the-technical-essentials-of-sap-commerce-cloud/omni-commerce-connect-restful-web-services_a60bebeb-75bc-4a6d-972b-96c17363b909 ; https://docs.commercetools.com/api/projects/productProjections (fetched 2026-11-02).

### Hive: three different answers

Hive is not one case. Before scoping, ask each Hive client three things: where the table files live (HDFS or cloud object storage), whether the tables are transactional, and whether the metastore's backing database can be reached from Databricks (recommendation) [SCR/gtm/research/gap_3.json recommendations].

1. Files on cloud storage, plain tables: federate. HMS federation reads Hive managed and external tables in every format Databricks lists for external tables (Delta, CSV, JSON, Avro, Parquet, ORC, text), plus SerDe tables, views and partitions (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-concepts, ms.date 2026-09-11; https://learn.microsoft.com/en-us/azure/databricks/tables/external, ms.date 2026-10-06]. The connection logs in to the metastore's backing database (MySQL, SQL Server or PostgreSQL), not to the Hive Thrift service, so the client must expose that database to Databricks over the network (documented; the network step is an inference) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-external, ms.date 2026-10-06].
2. Files on on-premises HDFS (Hadoop Distributed File System): export first. Databricks lists on-premises HDFS, and remote Thrift metastore mode, as not supported for HMS federation (documented) [https://docs.databricks.com/aws/en/query-federation/hms-federation-concepts, Last updated Sep 11, 2026]. The client can write a plain Parquet or ORC copy to cloud storage, which we then federate or load. The alternative, a JDBC connection to HiveServer2 with the client's own driver, is generally available on DBR 19 and later (our cluster runs DBR 19), but it has no foreign catalog, a 400 MiB driver memory limit, and the target cannot sit in the workspace's own virtual network (documented) [https://learn.microsoft.com/en-us/azure/databricks/connect/jdbc-connection, ms.date 2026-10-02; CLAUDE.md:1641-1643]. An open Spark bug reports the Hive JDBC driver returning column names in place of data, so that path must be tested before it is promised (documented) [https://issues.apache.org/jira/browse/SPARK-45403, created 2023-10-03].
3. Transactional (ACID) tables: export first until a client sample proves otherwise. No fetched page lists them as supported or unsupported for federation. The nearby statements point the same way: Databricks lists "ACID fine-grained updates" as an unsupported Hive feature, and a Spark ticket closed in 2021 says Spark does not support Hive transactional tables and has problems reading them (documented). The recommendation is an inference [https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-hive-compatibility, Last updated Sep 11, 2026; https://issues.apache.org/jira/browse/SPARK-15348; SCR/gtm/research/gap_3.json G3-06, G3-07].

### BigQuery and Snowflake

- BigQuery query federation needs DBR 16.1+ (or a pro or serverless SQL warehouse), a Google service-account JSON key, and outbound HTTPS to bigquery.googleapis.com, bigquerystorage.googleapis.com, oauth2.googleapis.com and accounts.google.com (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/bigquery, ms.date 2026-10-09]. Catches on the same page: views and external tables are always materialized, which adds BigQuery compute charges; a table with an INTERVAL column cannot be described or queried; BIGNUMERIC maps by default to DecimalType(38,38), which holds only values below 1 in magnitude, so a BIGNUMERIC price column needs its default scale changed; DATETIME arrived as text on DBR 16.4 to 17.x (documented) [same URL; SCR/gtm/research/connectors.verify.json CON-36].
- Snowflake: every table is reachable by query federation, which runs on the client's Snowflake compute; catalog federation (reading files directly) covers only Snowflake-managed Iceberg tables (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/snowflake-catalog-federation, ms.date 2026-10-06]. Query federation runs on DBR 13.3 LTS or later, catalog federation on DBR 16.4 LTS or later (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/snowflake, ms.date 2026-10-13; same catalog-federation page]. Login options: see Risks below.

### Where the target accounts keep product data

Public sources answer this only partly; they are 2019-2026 press releases, filings, vendor case studies and job ads, and current stacks may differ. The last column is our inference.

| Account | Public evidence on where product data lives | Connector to expect first (inference) |
|---|---|---|
| H&M | In-house "Product Data Hub, built on microservices" on a MACH (microservices, API-first, cloud-native, headless) architecture [https://www.ecommerceexpo.co.uk/news/hms-global-product-information-lead-enhancing-customer-experience-ai-data, 27 Aug 2024]; Google Cloud data backbone partnership [https://www.googlecloudpresscorner.com/2022-06-30-Google-Cloud-Announces-New-Partnership-with-Global-Fashion-Retailer, 2022-06-30]; named early adopter of Databricks' Lakehouse for Retail, cloud not stated [https://www.databricks.com/company/newsroom/press-releases/databricks-launches-data-lakehouse-for-retail-and-consumer-goods-customers, 2022-01-13]; the H&M brand sells on its own brand.com platform [https://webbjobb.io/jobb/208490/software-engineer-solution-architect, H&M job ad, 1 Apr 2022], and COS, ARKET, & Other Stories and Weekday moved to Centra in 2024 [https://centra.com/news/hm-invests-in-centra, 7 May 2025] | A UC table or share if they run Databricks; otherwise BigQuery, a feed export, or a Centra export for the sister brands |
| C&A | Product data pass through a product information management (PIM) system, vendor unnamed [https://www.c-and-a.com/si/sl/corporate/company/careers/jobs/working-student-mfd-onsite-merchandise-1-1-66568-en-gb, undated, fetched 2026-11-02]; Snowflake is its central data warehouse, cloud not stated [https://www.snowflake.com/en/blog/c-and-a-build-data-driven-fashion-empire/, 2023-08-17] | Snowflake query federation, or the native Snowflake reader |
| Kroger | All site search runs on Empathy.co, which Kroger operates itself (since 2020); the catalog is fed into Empathy's index builders, upstream system not named [https://empathy.co/assets/media/uploads/resources/kroger-building-and-evolving-ai-search-with-empathy.pdf, undated, data to June 2025]; Azure "preferred cloud platform for Retail as a Service" (scoped to that programme) [https://news.microsoft.com/source/2019/01/07/kroger-and-microsoft-partner-to-redefine-customer-experience-introduce-digital-solutions-for-retail-industry/, 2019-01-07]; Google Cloud analytics tools [https://www.googlecloudpresscorner.com/2023-01-20-Google-Cloud-and-Deloitte-Boost-Grocery-Associate-Productivity-and-Improve-the-Customer-Experience, 2023-01-20]; 84.51° (Kroger) a Databricks early adopter [https://www.databricks.com/company/newsroom/press-releases/databricks-launches-data-lakehouse-for-retail-and-consumer-goods-customers, 2022-01-13] | The feed Kroger already builds for its search engine, or a UC table through 84.51° |
| Albertsons | eCommerce sites moved to Azure [https://news.microsoft.com/source/2019/02/22/albertsons-companies-to-transform-experiences-for-shoppers-with-microsoft-cloud-and-ai/, 2019-02-22]; commerce platform vendor not named in the part of the 10-K read [https://www.sec.gov/Archives/edgar/data/0001646972/000164697226000032/aci-20260228.htm, FY ended 2026-02-28] | Unknown: ask in discovery |
| H-E-B (US) | Built in-house; store assortment comes from PALS, an in-house service updated as shelf locations change, replacing a nightly batch [https://careers.heb.com/blog/simplifying-the-assortment-riddle, 2025-03-07]; no platform vendor named | Unknown; likely an export or API from in-house services |

### Recommended design: one adapter layer, run offline

- One function, read_source(uri), returns every value as text plus a fingerprint of what was read. Readers are registered by address scheme: file paths, sqlite:, uc://catalog.schema.table, sql:<SQLAlchemy URL>#<query>, bq://, sf://, hive:// or trino://, iceberg://, delta:, share:. Each is imported only when used, with its own optional requirements file (design; the extra schemes extend the byo design, which refuses s3://, abfss:// and https:// in compiler P0) [SCR/oct27/byo/design_RECOMMENDED.md:322-341; SCR/gtm/research/connectors.json CON-51; SCR/gtm/research/connectors.verify.json CON-51].
- It runs in an offline "compile" step: a command-line tool locally, a job on Databricks. The step proposes a column mapping, a named person confirms it once, a report says which features each gap switches off, and it writes today's catalog.csv plus a manifest. Notebooks 01 (load), 02 (embed and index) and 09 (deploy), the API and the App stay unchanged (design, not built) [SCR/oct27/byo/REPORT.md:317-357].
- On Databricks, every source that federation, sharing or Lakeflow Connect can mount goes through the single uc:// reader. Client credentials stay in UC connections and secret scopes, never in our repo or the mapping file (recommendation) [SCR/gtm/research/connectors.json CON-37, CON-51].
- Running connectors offline is a design choice, not a code limit. The App could read a UC table directly with the Databricks SQL Connector, but that is unmeasured (corrected finding) [SCR/gtm/research/connectors.verify.json CON-03; https://docs.databricks.com/aws/en/dev-tools/python-sql-connector, Last updated Sep 29, 2026].
- Connectors only read. Federation, OneLake federation and Delta Sharing are read-only (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/, ms.date 2026-09-29; https://learn.microsoft.com/en-us/azure/databricks/query-federation/onelake, ms.date 2026-10-28; https://docs.databricks.com/aws/en/delta-sharing/, Last updated Oct 27, 2026]. CAS writes only inside its own catalog and schema (products, products_embeddings, products_vectors, the index, and its raw, app_state and evals volumes), provided CAS_CATALOG and CAS_SCHEMA point at a schema CAS owns. Until the planned ownership guard exists, 01 would overwrite a table named products in whatever schema it is pointed at (code) [databricks/01_catalog_to_delta.py:69-76; app/config.py:263-276; SCR/gtm/research/connectors.verify.json CON-52]. So each client gets its own catalog and schema [SCR/oct27/byo/REPORT.md:117-119].

### What a federated table can and cannot be

- It can be read by Spark like any table, read-only, and it can feed Lakeflow pipelines' materialized views and streaming tables (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-concepts, ms.date 2026-09-11].
- It cannot be a Vector Search index source. Sources must be Delta tables, streaming tables, materialized views or managed Iceberg v3+ tables, and a standard endpoint needs change data feed (CDF, Delta's built-in log of row changes) on the source (documented) [https://docs.databricks.com/aws/en/vector-search/create-vector-search, Last updated Oct 29, 2026; https://learn.microsoft.com/en-us/azure/databricks/ai-search/ai-search, ms.date 2026-10-29]. The HMS federation page also lists AI Search (the Azure docs' name for Vector Search) as not supported on foreign tables (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-concepts, ms.date 2026-09-11].
- That does not block CAS, because CAS already copies. 01 writes the catalog into its own Delta table with CDF on; 02 joins it with stored embeddings into products_vectors; the index syncs from that table (code) [databricks/01_catalog_to_delta.py:69-76; databricks/02_vector_index.py:238-246, 305-318]. A federated source changes only 01's read: a table read in place of a CSV read (inference) [SCR/gtm/research/gap_3.json G3-23].
- The copy has a cost nobody has measured: 01's Spark write, plus source compute (BigQuery materialization for views, the client's Snowflake warehouse for query federation) [SCR/gtm/research/gap_3.json G3-23].
- Databricks positions query federation for ad hoc reporting, business intelligence and proof-of-concept use, and recommends Lakeflow Connect where both exist and volume or latency matter (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/, ms.date 2026-09-29]. A nightly read of a ~300K-row catalog is light, so query federation should fit a nightly compile (inference; no federation run measured).

### Privileges a client or metastore admin must grant

Every federation type needs privileges at the metastore level, the top of Unity Catalog. Catalog federation (Hive, Snowflake Iceberg, OneLake) also needs storage objects and a grant on the client's own storage.

| Federation type | Objects to create | Databricks privileges | Client-side asks | Difficulty (inference) |
|---|---|---|---|---|
| BigQuery, query federation | Connection (service-account key) + foreign catalog | CREATE CONNECTION on the metastore; CREATE CATALOG on the metastore plus connection ownership or CREATE FOREIGN CATALOG [bq] | Service account with BigQuery User and BigQuery Data Viewer [bq] | Low: lightest proof |
| Snowflake, query federation | Same | Same; workspace admins hold both by default if the workspace was enabled for UC automatically [sf] | OAuth security integration or key pair; a role with USAGE and SELECT [sf] | Low |
| Snowflake, catalog federation (Iceberg only) | + storage credential, external location, authorized paths | + CREATE STORAGE CREDENTIAL, CREATE EXTERNAL LOCATION, CREATE FOREIGN SECURABLE; readers need MODIFY as well as SELECT [sf] | Access to the Iceberg storage | Medium |
| External Hive metastore | Connection to the metastore database + storage credential + external location(s) + foreign catalog | CREATE CONNECTION; CREATE STORAGE CREDENTIAL (account or metastore admins by default); CREATE EXTERNAL LOCATION; CREATE CATALOG; CREATE FOREIGN CATALOG; CREATE FOREIGN SECURABLE [hms-ext] [priv] | Metastore database reachable; on Azure, the storage account's Owner or User Access Administrator grants the connector's managed identity Storage Blob Data Contributor (the only role the page names; read-only is a Unity Catalog flag); hierarchical namespace on; no write-once (WORM) containers [adls] | High |
| Hive over JDBC (HiveServer2) | JDBC connection + driver file in a UC volume | CREATE CONNECTION; CREATE or MANAGE on the volume [jdbc] | HiveServer2 reachable, credentials | High, untested |
| Microsoft Fabric (OneLake) | Storage credential + connection + foreign catalog | Metastore admin, or CREATE CONNECTION + CREATE STORAGE CREDENTIAL; a service-principal credential only by an account admin through the API [fabric] | Fabric admin: three tenant settings and a workspace setting; identity granted Contributor [fabric] | Medium to high |
| Delta Sharing (Databricks to Databricks) | Catalog from the share | Metastore admin, or CREATE CATALOG + USE PROVIDER [share] | Provider creates the share and recipient | Low, if the client runs Databricks |

Extra keys: [priv] https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/access-control/privileges-reference (ms.date 2026-10-12); [adls] https://learn.microsoft.com/en-us/azure/databricks/connect/unity-catalog/cloud-storage/external-locations-adls (ms.date 2026-10-14). Other keys as under the source table.

Our own position. This account has no MANAGE right on the dev catalog: a grant was refused on 2026-10-13 (recorded in the repo after a workspace run) [CLAUDE.md:1798-1800]. Whether it holds CREATE CONNECTION or CREATE CATALOG on the metastore is not recorded, and neither is the access mode of Retail_DE_cluster [SCR/gtm/research/gap_3.json G3-20; CLAUDE.md:1641-1643]. On dedicated access mode compute the querying user must own the connection; CREATE FOREIGN CATALOG is not enough (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/foreign-catalogs, ms.date 2026-09-11]. So the 0.5-1 day workspace proof waits on an administrator. The lightest proof is BigQuery or Snowflake query federation: no storage objects and nothing granted on client storage. Admin lead time is unknown and is not inside any estimate here (inference) [SCR/gtm/research/gap_3.json G3-21].

### Off Databricks: do the readers work with our pandas 3 pin?

The App pins pandas 3.0.3 and pyarrow 24.0.0 (code) [databricks/app_bundle/requirements.txt:15-16]. An offline pip dry-run resolve (wheels only, nothing installed) against the App's pins on Python 3.13 Windows and Python 3.11 Linux (assumed to match the App), plus an offline smoke test on Python 3.13 Windows only, with no server and no credentials, gave these results (measured, 2026-11-02) [SCR/gtm/research/gap_3.json G3-24, G3-25, G3-27, G3-29]:

| Reader | Resolves with the App pins | Offline smoke on pandas 3.0.3 | Use |
|---|---|---|---|
| google-cloud-bigquery 3.47.0 + db-dtypes 1.9.0 | Yes | Import and empty to_dataframe / to_arrow OK (a deprecation warning for DataFrames) | Yes; skip the google-cloud-bigquery-storage pandas extra, capped below pandas 3 since 2.41.0 |
| snowflake-connector-python 5.1.0, pandas extra | No (pandas below 3) | - | No |
| snowflake-connector-python 5.1.0, pyarrow extra | Yes | Import OK | Yes: fetch_arrow_batches, convert in CAS (code reading of the wheel) [G3-28] |
| deltalake 1.6.6 | Yes | Local write and to_pandas OK | Yes |
| pyiceberg 0.12.0 | Yes | Append and scan OK (needed FsspecFileIO on Windows) | Yes |
| SQLAlchemy 2.1.6 + snowflake-sqlalchemy + sqlalchemy-bigquery | Yes | pd.read_sql on SQLite OK | Yes |
| databricks-sql-connector 4.8.0 | Yes | Import OK | Yes |
| impyla 0.24.0, trino 0.340.0 | Yes (no pandas dependency) | Not run | Yes |
| awswrangler 3.18.0 | Yes | Not run | Yes |
| delta-sharing 1.4.3 (latest) | No (pandas below 3, pyarrow up to 23.0.1, requests up to 2.33.0) | - | No |
| delta-sharing 1.4.1 | Yes | Import OK; load_as_pandas not run | Yes, pinned |
| Every compatible reader plus the whole App bundle | Yes, 131 packages | - | One environment is possible |

What this means: pandas 3 does not change the estimates. Snowflake probably needs no separate environment, and open Delta Sharing needs a version pin (estimate) [SCR/gtm/research/gap_3.json G3-31]. Not tested: any live query against BigQuery, Snowflake, Databricks SQL, Hive or a sharing server. A compile that runs on a Databricks cluster uses the cluster's own pandas, and DBR 19's version was not checked (inference) [SCR/gtm/research/gap_3.json G3-32]. Version caps appeared in patch releases (delta-sharing 1.4.2 on 2026-08-10, bigquery-storage 2.41.0 on 2026-08-24; documented on PyPI), so the resolve should be re-run before each connector release (recommendation) [SCR/gtm/research/gap_3.json G3-27, recommendations].

### Risks: type drift, logins, network

Type drift (a column's type changing on the way in):
- Integers with missing values turn into floating-point numbers. Snowflake whole-number NUMBER columns with a NULL come back as float64: 123 becomes 123.0 and ids above 2^53 lose precision, so ids must be selected as text (documented; corrected finding) [https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-pandas, fetched 2026-11-02; SCR/gtm/research/connectors.verify.json CON-25]. Our smoke test saw the same in deltalake and pyiceberg: an int64 column [12, None, 3] came back as float64 (measured) [SCR/gtm/research/gap_3.json G3-30].
- BigQuery INTERVAL, BIGNUMERIC and DATETIME, as above (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/bigquery, ms.date 2026-10-09].
- OneLake rejects arrays, maps and structs, and tables whose column names differ only in case; in Snowflake catalog federation, tables with the same name ignoring case conflict (documented) [https://learn.microsoft.com/en-us/azure/databricks/query-federation/onelake, ms.date 2026-10-28; https://learn.microsoft.com/en-us/azure/databricks/query-federation/snowflake-catalog-federation, ms.date 2026-10-06].
- Lakeflow's Shopify connector cannot follow column renames or type changes; a rename needs a full refresh (documented) [https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/shopify, Last updated Oct 9, 2026].
- 01 stops the whole run on one malformed numeric cell, because ANSI mode is on in DBR 17+ (code reading; Spark not run) [SCR/oct27/byo/REPORT.md:241].
- The answer in the design: read every value as text, type it only in the normalisers, and diff the mapping on every refresh (design) [SCR/oct27/byo/design_RECOMMENDED.md:322-333; SCR/gtm/research/connectors.json CON-36].

Logins differ per source (documented): a service-account key for BigQuery (Databricks recommends storing it as a secret); Snowflake OAuth, Entra ID or Okta OAuth, an access token, a PEM private key (key pair of at least 2048-bit RSA) or username and password for Snowflake [also https://docs.snowflake.com/en/user-guide/key-pair-auth, fetched 2026-11-02]; Entra ID users or service principals only for Fabric, no SQL logins; a username and password on the metastore database for external Hive metastores; Kerberos, LDAP, JWT or OAuth2 for Hive and Trino clients [https://learn.microsoft.com/en-us/azure/databricks/query-federation/bigquery, ms.date 2026-10-09; https://learn.microsoft.com/en-us/azure/databricks/query-federation/snowflake-catalog-federation, ms.date 2026-10-06; https://learn.microsoft.com/en-us/fabric/data-warehouse/connectivity, ms.date 2026-09-09; https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-external, ms.date 2026-10-06; https://github.com/cloudera/impyla; https://github.com/trinodb/trino-python-client]. On Databricks these sit in UC connections, so CAS code never holds them. Off Databricks the compile tool must read them from the environment, never from the mapping file (recommendation) [SCR/gtm/research/connectors.json CON-37].

Network (documented unless marked): BigQuery needs outbound HTTPS to the four Google endpoints above, and serverless compute uses a separate egress path; Snowflake may need its Online Certificate Status Protocol (OCSP) responder; Fabric needs TCP 1433 treated as SQL Server traffic; an external Hive metastore needs network reach to its database [same sources as the logins paragraph]. Our workspace runs on Azure [CLAUDE.md:1524], so BigQuery (Google Cloud) or AWS-hosted sources are cross-cloud reads; egress charges and latency were not checked (inference) [SCR/gtm/research/connectors.verify.json CON-38].

### Refresh design

- Today a refresh is a rebuild: pause the index, run 01, embed, rebuild the index (code) [databricks/01_catalog_to_delta.py:29-33; SCR/oct27/byo/REPORT.md:115-127]. Embedding is already incremental at the text level: 02 embeds only text_hash values it has not stored (code) [databricks/02_vector_index.py:186-194]. But the index key, catalog_index, is a row position, so one inserted row shifts every later key and the whole index is rebuilt (code) [databricks/01_catalog_to_delta.py:55-63; databricks/02_vector_index.py:305-318]. Measured once on 2026-10-12 on our Azure workspace: ~38 min to embed 296,483 texts (~83 DBU, Databricks Units, the platform's billing unit, at list price; the script's own estimate) and 47.8 min to build the index for 299,972 rows [CLAUDE.md:1537-1543].
- Locally the embedding cache is keyed on the md5 of the whole CSV, so any byte change re-encodes everything: ~24 h per 300K rows on the laptop CPU (code; documented in the repo) [app/search.py:106-133; CLAUDE.md:12].
- First design (recommendation): a nightly full snapshot plus the text_hash diff, for every source. The manifest records a source fingerprint: Delta version, BigQuery snapshot time, Snowflake query id or file md5 [SCR/gtm/research/connectors.json CON-33 and recommendations]. A full snapshot is cheap at our scale: the 299,972-row catalog exported to 249 MB of CSV on 2026-10-13 and reads with pandas in 6.76 s on the laptop (measured) [CLAUDE.md:1796; SCR/gtm/research/probe_scalability.json "read_csv_s"]. The read from a client warehouse, and its source-side compute, is unmeasured.
- Change data capture (CDC, reading only the rows that changed at the source) exists for the main warehouses (documented): Delta CDF is opt-in per table (automatic on DBR 19+ for Unity Catalog managed tables with row tracking) and kept only for a retention window; Snowflake streams advance only inside a write transaction and go stale after the table's retention period; BigQuery CHANGES needs enable_change_history = TRUE and both CHANGES and APPENDS cost BigQuery compute [https://docs.databricks.com/aws/en/delta/delta-change-data-feed, Last updated Oct 9, 2026; https://docs.snowflake.com/en/user-guide/streams-intro; https://docs.cloud.google.com/bigquery/docs/change-history, fetched 2026-11-02]. CDC buys little until the index key is the client's product id with a merge into products_vectors: compiler P3, 5.5 dev-days (estimate). After that, each CDC reader is about 1.0 dev-day (estimate) [SCR/oct27/byo/design_RECOMMENDED.md:741; SCR/gtm/research/connectors.verify.json CON-35].
- Source CDC mainly matters for sub-daily refreshes. Near or above ~1M rows, the App's in-memory catalog becomes the limit before refresh cost does: an estimated ~5.2 GB per process at 1M rows, against 6 GB on a Medium App (estimate, unmeasured) [SCR/gtm/research/probe_scalability.json per_catalog_size "1M"; SCR/oct27/byo/REPORT.md:239; SCR/gtm/research/connectors.verify.json CON-34].
- Grocery note: H-E-B updates store assortment as shelf locations change [https://careers.heb.com/blog/simplifying-the-assortment-riddle, 2025-03-07], and Kroger's search vendor reports cutting index updates from 24 h to 5 min in 2021 [https://empathy.co/assets/media/uploads/resources/kroger-building-and-evolving-ai-search-with-empathy.pdf, undated, data to June 2025] (both documented; the second is a vendor case study). A nightly CAS copy is not built for that, and store-level inventory is on HOLD in CAS scope [CLAUDE.md:1333-1334]. Stock and availability should stay in the client's own engine (inference).

### Data residency

- A connector copies client catalog rows into the CAS workspace's region (inference); our demo workspace's region is not recorded [SCR/gtm/research/gap_4.json open_questions]. CAS then sends seller text (titles, descriptions) into the rerank and cross-sell prompts (code, as recorded in the repo) [CLAUDE.md:775-776]. That text is not screened for injected instructions: the indirect-injection risk is open (code) [.claude/rules/safety.md:139-141].
- On the Databricks path both models are Databricks-hosted serving endpoints; the repo records that "nothing leaves the workspace", but where the call is processed depends on the model's region (next bullet) [CLAUDE.md:1527, 1552]. The local default sends prompts to Groq, an outside service (code) [app/config.py:51, 65-66].
- For EU retailers the model's region matters. Databricks' model-region pages list databricks-qwen3-next-80b-a3b-instruct in no Azure EU or UK region; on AWS it is listed pay-per-token in Frankfurt, Ireland, London and Paris (documented) [https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability, ms.date 2026-10-30; https://docs.databricks.com/aws/en/unity-gateway/model-region-availability, Last updated Oct 30, 2026; SCR/gtm/research/gap_4.json G4-01, G4-04]. The embedding model gte-large-en and the fallback LLM gpt-oss-120b are listed in six Azure EU regions without the cross-geography mark (documented) [SCR/gtm/research/gap_4.json G4-03]. The cloud section covers the model options. For connectors, the rule is: before an EU pilot, confirm the workspace region, the model region, and whether catalog text may reach the LLM at all (recommendation).

### Build order and effort

All figures are estimates for one developer who knows the repo, tests included; none is measured. None includes owner and legal sign-offs or admin lead time, and the byo report says its phase totals also exclude Databricks workspace runs; step 2 below is a workspace proof estimated on its own [SCR/oct27/byo/REPORT.md:400-421; SCR/gtm/research/connectors.verify.json CON-35]. Connectors are on the critical path only for the patterns that hold a catalog copy (Section 7's A, D-full and E) or for a CAS-hosted demo; the overlay Section 7 recommends (B, then C) needs no catalog copy in its first two phases (inference).

| Step | What it delivers | Dev-days (estimate) | Depends on | Source |
|---|---|---|---|---|
| 0. Interim pilot on today's code | Hand-shaped CSV per client; silent failures stay live | ~1 per client | Nothing | SCR/oct27/byo/REPORT.md:35, 131-140 |
| 1. Compiler P0 | read_source for the CSV family, Parquet, JSON Lines, SQLite and uc://; detection; GMC / Shopify / legacy presets; mapping.json; coverage report; ownership guard | 4.0 optimistic, 5-7 safer | Nothing | SCR/oct27/byo/REPORT.md:404, 412 |
| 2. uc:// workspace proof | One federated BigQuery or Snowflake table plus one Databricks-to-Databricks share, read as text, compiled, accepted by 01 | 0.5-1 | An admin creates the connection and foreign catalog (lead time unknown) | SCR/gtm/research/connectors.verify.json CON-35; SCR/gtm/research/gap_3.json G3-21 |
| 3. Native Snowflake reader | Key-pair login, fetch_arrow_batches, in the CAS environment | 1.5 (may trim 0.25-0.5; adds a live check) | Step 1 | SCR/gtm/research/connectors.verify.json CON-35; SCR/gtm/research/gap_3.json G3-31 |
| 4. Native BigQuery reader | Storage Read API to Arrow | 1.0 | Step 1 | SCR/gtm/research/connectors.verify.json CON-35 |
| 5. Generic sql: reader | SQLAlchemy for PostgreSQL, MySQL, SQL Server, Oracle and others | 1.5, +0.25-0.5 per extra driver (not in byo's ~36-day total) | Step 1 | SCR/gtm/research/connectors.verify.json CON-35 |
| First wave (steps 2-5) | | 4.5-5 | | SCR/gtm/research/connectors.verify.json CON-35 |
| Later, when a named client asks | GMC XML 1.0 (not in byo's total); Excel off Databricks 0.5; Delta + Iceberg off Databricks 1.0; s3:// / abfss:// / gs:// addresses 1.0; open Delta Sharing 0.5 + 0.25 pin; Hive / Trino off Databricks 1.5; Merchant API 2.0; Shopify bulk 2.0; Lakeflow Shopify preset 1.0; SFCC 3.0; SAP Commerce 2.5; commercetools 2.0 | As listed | Step 1 | SCR/gtm/research/connectors.verify.json CON-35; SCR/gtm/research/gap_3.json G3-31 |
| Compiler P2, Databricks native | 01 reads uc:// and the mapping itself; staging, validate, promote; Auto Loader landing | 7.0 | Step 1 | SCR/oct27/byo/REPORT.md:406 |
| Compiler P3, stable ids | Index key = client product id; merge instead of rebuild | 5.5 | Step 1 | SCR/oct27/byo/REPORT.md:407 |
| CDC readers | Delta CDF, Snowflake streams, BigQuery CHANGES, watermarks | ~1.0 each | Compiler P3 | SCR/gtm/research/connectors.verify.json CON-35 |

Reading the table: compiler P0 plus the first wave is about 8.5-12 dev-days (our arithmetic on the estimates above; 8.5 assumes compiler P0's optimistic 4 days) for files, UC tables, BigQuery, Snowflake and SQL databases, plus admin lead time. The full byo design, which adds serving changes, Databricks-native loading, stable ids and feed-first quality work, is ~36 dev-days, and the sql: reader and GMC XML parser come on top of it [SCR/oct27/byo/REPORT.md:410; SCR/gtm/research/connectors.verify.json CON-35]. Per client, outside all totals: ~1-2 days to review the mapping and run the first build [SCR/oct27/byo/REPORT.md:443].

How to word it for a client. Today: a pilot needs a one-time mapping of the client's product data into our format, about a day of work per client (estimate), and the probe's silent failures stay possible; automated onboarding is designed, not built [SCR/gtm/research/claims.json CLA-52, CLA-53]. After compiler P0, the first wave and a passing workspace proof: CAS reads the catalog read-only from where it lives, through Unity Catalog or from a Merchant Center export (and a Shopify export if its preset survives the compiler P0 cut), keeps its own copy, and a person confirms the field mapping once [SCR/gtm/research/connectors.json recommendations; SCR/gtm/research/connectors.verify.json CON-45]. Name as working only the sources that had a workspace run (at first, the one federated BigQuery or Snowflake table and the share from step 2). Call the rest of the federation list (a Hive metastore on cloud storage, SQL Server, Oracle, Teradata, Redshift, Synapse, Fabric) "supported by the Databricks platform, not yet tested with CAS" (recommendation). Do not call onboarding automatic before the byo gate is met: auto-accept precision of at least 0.99 on at least 200 labelled columns from at least 8 real schemas [SCR/oct27/byo/REPORT.md:390]. Do not promise unattended setup or a fit for every schema or source [SCR/gtm/research/claims.json CLA-54].

### Open questions for this area

- Which source does each first pilot actually use? H&M's product data hub, C&A's PIM or Snowflake and Kroger's search feed are only partly documented; Albertsons and H-E-B are unknown.
- Will a client grant a CAS workspace a federation connection (service-account key, Snowflake OAuth or key pair, network allow-lists), or must CAS run inside the client's own workspace and cloud?
- Who is the metastore admin for dev, does our account hold CREATE CONNECTION or CREATE CATALOG, and how long does a connection plus foreign catalog take to get?
- Is Retail_DE_cluster in standard or dedicated access mode? Dedicated mode needs connection ownership, and OneLake federation refuses it.
- Can HMS federation read a Hive ACID table at all? Only a test on a client sample will tell.
- Is a read-only Azure role enough for the connector identity on a client's Azure Data Lake Storage (ADLS), and is a storage account in another Entra tenant supported? The docs name only a contributor-level role.
- What does a nightly federated snapshot cost (Databricks compute, BigQuery materialization, the client's Snowflake warehouse), and what do cross-cloud reads from our Azure workspace cost in egress and latency? Nothing is measured.
- How much memory and time does the uc:// read to pandas take above ~1M rows? Unmeasured.
- Which region is our demo workspace in, and which region would a client's EU workspace use? Neither is recorded [SCR/gtm/research/gap_4.json open_questions].
- What shape is Lakeflow Connect's Shopify output (Beta), and what format is an SFCC catalog export? Neither was inspected.
- How often do retailers need a refresh (daily prices, stock)? The answer decides whether CDC is ever worth building.
- Live behaviour is untested: Snowflake fetch_arrow_batches, delta-sharing 1.4.1 load_as_pandas, google-cloud-bigquery to_dataframe on a real table, HiveServer2 over JDBC. Which pandas version does DBR 19 ship?
- Who signs the mapping file (a merchandiser for departments, an engineer for id and price), and where does it live? It must never live in our repo [SCR/oct27/byo/REPORT.md:458].

## 4. Search backends: where the vectors can live

Answer. CAS asks its vector store one narrow question: for each search phrase, which catalog rows match best, and with what score. Two answers are built today: a search inside the server's own memory ("local"), and Databricks AI Search (formerly Databricks Vector Search). Databricks documents AI Search on AWS, Azure and Google Cloud; CAS's Databricks backend has run only on one Azure workspace (evals, no load test), and per-region availability was not checked [CLAUDE.md:1524-1525; SCR/gtm/research/retrieval.json RET-20]. Of the 14 platforms we researched (Snowflake counted once), all document metadata filters, and all but Amazon S3 Vectors document some combined keyword-plus-vector ("hybrid") search: on Snowflake only through Cortex Search, not the plain VECTOR type, on Vertex AI you supply the keyword vectors yourself, and on pgvector the application merges the two (documented, 4.3). Because the seam is narrow, a new vector backend is an adapter estimated at roughly 3-7 developer-days for vector-only retrieval (an unvalidated estimate), not a redesign. None of the other 13 has been built or tested. On catalog size, the larger limit is not where the vectors live: it is the full catalog copy and keyword index that CAS keeps inside every server process (measured on a laptop, extrapolated above 300K rows, 4.8).

How each fact is known is marked: measured, code (read in the repository), documented (a vendor page), estimate, or inference. Nothing in this section was measured on Databricks under load: no load test exists anywhere in the repository.

Terms used below. ANN = approximate nearest-neighbour search, the fast, slightly inexact way vector stores find similar vectors; HNSW is the most common ANN index structure. BM25 = the standard keyword-relevance formula behind most text search engines. RRF = Reciprocal Rank Fusion, which merges a keyword ranking and a vector ranking by position in each list, not by raw score. QPS = queries per second. DBU = Databricks Unit, Databricks' billing unit. SKU = one sellable item. LLM = large language model, the AI model that rewrites the query and re-ranks results. API = the programming interface other systems call; REST = the plain web-request style of API. RAM = a server's working memory. SQL = the standard database query language. p50 / p90 / p95 = the value that 50% / 90% / 95% of measurements fall under (p50 = the median). HTTP 429 = the "too many requests" error a service returns when a rate limit is hit.

### 4.1 The seam: how CAS uses a vector store today

- One function picks the backend. `_raw_hits(search_terms, top_k)` returns, for each search phrase ("intent"), a best-first list of (catalog_index, score). With SEARCH_BACKEND=databricks it calls Databricks; otherwise it embeds the phrases on the server and scans an in-memory vector matrix (code) [app/search.py:537-548].
- The store returns only a row key and a score. CAS then fills in ("hydrates") the full product row from its own in-memory copy of the catalog, by row position (code) [app/search.py:519]. So the key must be a dense row number 0..N-1, not a SKU id; the Databricks loader refuses anything else (code) [app/vector_search.py:131-139].
- Why this matters: everything after retrieval (title de-duplication, the AI re-ranking, ordering, filters, ads, recommendations) sees the same columns whatever the store (blank values differ, 4.4). Columns CAS derives when it loads the catalog (product type, the electronics and grocery taxonomy) reach every backend without being stored in it (code) [app/search.py:306-331; app/search.py:494-534].
- On Databricks, one call embeds all phrases of a search with the databricks-gte-large-en model. CAS then sends one REST query per phrase, one after another, and asks only for catalog_index (code) [app/vector_search.py:240-274].
- Load per shopper search: about 3.2 vector queries and 1 embedding call per search from the shipped UI (30 of 31 held-out queries had 3 phrases, one had 9), and about 8.2 vector queries and 6-7 embedding calls when an API caller leaves recommendations on, which is the default; a product-page open adds about 5-6 more of each (code plus a count of recorded queries) [SCR/gtm/research/gap_1.json G1-01, G1-02, summary]. A platform's QPS limit therefore binds about 3-8 times sooner than shopper searches per second suggest (inference).

### 4.2 What a new backend involves

The contract a third backend must meet (code) [SCR/gtm/research/retrieval.json RET-38, confirmed with additions in retrieval.verify.json]:

| Piece | What the new backend must do | Where it lives today |
|---|---|---|
| Name and settings | A SEARCH_BACKEND value plus endpoint, index, credentials and embedding-model settings, validated at start-up | app/config.py:258-291 |
| Catalog load | Load the catalog with a dense catalog_index (= row position), then run the same load-time annotation | app/search.py:306-331 |
| Readiness | Treat the new name as ready without local vectors | app/search.py:300-303 |
| Retrieval | Per phrase, best first, up to 60 (catalog_index, score) pairs; embed queries with the SAME model that embedded the catalog | app/search.py:537-548 |
| Catalog fingerprint | Report a catalog fingerprint for eval provenance | app/search.py:87-103 |
| Index build and refresh | Bulk-load key + vector (plus text and filter columns if hybrid or in-index filters are wanted); re-embed only changed text | databricks/02_vector_index.py |
| Health checks | Index exists, row count matches the catalog, one embedding works; /healthz names the right embedding model | app/main.py:70-128, 173 |
| Relevance floor | Calibrate MIN_RESULT_RELEVANCE on the new score scale (4.4) | app/config.py:121-125 |
| Evals and tests | Backend recorded in eval output; a fake-client test; keyword set, context set and the no-LLM control | eval/run_eval.py:139-154 |
| Unchanged | search_products, hydration, de-duplication, main.py ordering, recommendations, keyword search | - |

Effort, all estimates (none validated: the verifier marked the per-backend total "unverifiable", while confirming its arithmetic):

| Work | Dev-days | Source |
|---|---|---|
| Turn the two if-branches into a validated backend registry (once) | 0.5-1 | SCR/gtm/research/retrieval.json RET-39 |
| Query adapter: load, retrieval, id mapping, health check, settings | 1-2 per backend | RET-39 |
| Index build and refresh job | 1-3 per backend (top of range with hybrid text and filter columns) | RET-39 |
| Score calibration, eval runs, fake-client tests | 1-2 per backend | RET-39 |
| Total per new vector backend | about 3-7 | RET-39; SCR/gtm/research/retrieval.verify.json RET-39 |
| Hand /keyword_search to the platform's own BM25 (4.9) | +2-4 | RET-39 |
| Stable product ids and incremental refresh (needed to sit on a client's engine; 4.9) | 5.5 | SCR/oct27/byo/REPORT.md:407 |

The 3-7 figure assumes stored vectors are reused and that hydration and keyword search stay in process. For scale, today's Databricks path is 274 + 488 + 153 lines (code) [app/vector_search.py; databricks/02_vector_index.py; databricks/vs_shim.py, wc -l].

### 4.3 Platforms researched

What each vendor page says. All rows except "CAS local" are documented, from pages fetched 2026-11-02 (page dates given where the page shows one). Only the first two rows run in CAS; none of the others has been run against CAS.

| Platform | Metadata filters | Keyword + vector hybrid (how delivered) | Notes | Source |
|---|---|---|---|---|
| CAS local (today) | None inside retrieval; the shop filters run in process on the cached result pool (code) | None. A separate in-process BM25 engine serves Keyword Search (code) | Exact brute-force scan of 299,972 x 384 vectors in RAM; deterministic (4.10) | app/search.py:537-548; app/keyword_search.py:150-204; CLAUDE.md:679-681 |
| Databricks AI Search, standard endpoint (CAS's built backend) | Yes, on any indexed column | Server-side: BM25 over all text columns + ANN, merged by RRF (rrf_param 60), scores normalised so the highest possible is 1, at most 200 results; about 4x the resources of ANN. Full-text-only queries are Beta | ~320M vectors at 768 dimensions per endpoint (Delta Sync indexes; ~2M for Direct Vector Access); 2M (768-d) per unit at 4 DBU/h, $0.28/h effective US East list price; 20-50 ms and 30-200+ QPS (768-d figures); beyond one unit QPS plateaus near 30 (ANN). Same capacity text in the AWS, Azure and Google Cloud doc sets. CAS's index syncs only key + vector (4.5) | https://learn.microsoft.com/en-us/azure/databricks/ai-search/ai-search (ms.date 2026-10-29); https://learn.microsoft.com/en-us/azure/databricks/ai-search/query-ai-search (ms.date 2026-10-29); https://learn.microsoft.com/en-us/azure/databricks/ai-search/best-practices (ms.date 2026-10-29); https://www.databricks.com/product/pricing/vector-search (no date shown, fetched 2026-11-02); https://docs.databricks.com/aws/en/vector-search/vector-search and https://docs.databricks.com/gcp/en/vector-search/vector-search (last updated Oct 29, 2026) |
| Databricks AI Search, storage-optimized endpoint | Yes, SQL-like filter strings; it over-fetches then filters, so it can return nothing even when matches exist | As standard, plus a dedicated full-text index (Beta) | ~1B vectors at 768-d; 18.29 DBU/h per unit, 64M vectors, $1.28/h list; latency "about 250 ms" on one page, 300-500 ms and 30-50 QPS on another, no target QPS option; scheduled (triggered) refresh only, each sync partly rebuilds the index; dimensions divisible by 16 | ai-search and best-practices pages above; pricing page above |
| Snowflake Cortex Search | Yes: @eq, @contains, @gte, @lte, @primarykey, with and / or / not | Server-side: vector + keyword retrieval, then a semantic rerank; weights configurable | Max 1,000 results per query; under 400M rows for best serving; beyond 20 QPS per service needs extra capacity (HTTP 429 otherwise); billed per GB-month indexed + embedding tokens + the refresh warehouse; accepts your own vectors (Multi-index); AWS, Azure and Google Cloud regions, some only through cross-region inference | https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview; https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/query-cortex-search-service (fetched 2026-11-02) |
| Snowflake VECTOR type + VECTOR_COSINE_SIMILARITY | SQL WHERE | None: similarity functions only | Up to 4,096 dimensions; no vector index named on the fetched pages, so a full scan is an inference; not supported in Iceberg tables, Snowpipe, search optimisation or clustering keys | https://docs.snowflake.com/en/sql-reference/data-types-vector; https://docs.snowflake.com/en/sql-reference/functions/vector_cosine_similarity (fetched 2026-11-02) |
| BigQuery VECTOR_SEARCH | Yes. Pre-filter pruning is optimised only on partitioned or stored columns; a pre-filter on non-stored clustered columns causes a full table scan | A hybrid combining semantic and lexical (keyword) search is documented; the fetched page does not say how it is fused | Brute force without a vector index; IVF or TreeAH index (two kinds of ANN index; TreeAH for up to 200M rows; an index on a table under 10 MB is not populated; refresh is asynchronous); no index on Standard edition; billed by bytes scanned or slots; no latency figure | https://docs.cloud.google.com/bigquery/docs/vector-search-intro; https://docs.cloud.google.com/bigquery/docs/vector-index (fetched 2026-11-02) |
| Vertex AI Vector Search (the page brands it "Gemini Enterprise Agent Platform") | Numeric and text-attribute "restricts" | You generate the sparse (keyword) vectors yourself (TF-IDF, BM25 or SPLADE: keyword-weighting methods); the service merges by RRF | ScaNN index (Google's ANN library); "a minimal setup (under $100 per month)"; quotas page not reachable (HTTP 404) | https://docs.cloud.google.com/vertex-ai/docs/vector-search/overview; https://docs.cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search (fetched 2026-11-02) |
| Azure AI Search | Filters, facets, sorting, scoring profiles | Server-side, one request: BM25 + HNSW or exhaustive vector search in parallel, merged by RRF; optional semantic ranker reranks the top 50 | 4,096 dimensions max; vector quota per partition for newer services (35 GB on S1, 300 GB on S3); Serverless Developer tier is preview, billed since Sep 13, 2026, throttled at 50 queries/s per index | https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview (ms.date 2026-08-31); https://learn.microsoft.com/en-us/azure/search/semantic-search-overview (ms.date 2026-08-05); https://learn.microsoft.com/en-us/azure/search/search-limits-quotas-capacity (ms.date 2026-09-16) |
| Amazon OpenSearch Service | Filters; post_filter can return fewer than k results | Hybrid query (OpenSearch 2.11+) through a search pipeline: score normalisation, or an RRF score-ranker | Vectors up to 10,000 floats; k up to 10,000; by default vector graphs get 25% of instance RAM; _msearch sends several phrase queries in one request | https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html; https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/ (fetched 2026-11-02) |
| Amazon S3 Vectors | Yes, up to 2 KB of filterable metadata per vector | None. AWS points hybrid, faceting and aggregations to OpenSearch | Generally available Dec 2, 2025 in 14 regions; 2B vectors per index; top-K 10,000 (100 per results page); latency under one second for infrequent queries and as low as 100 milliseconds for more frequent ones (vendor's figures); "ideal for workloads where queries are less frequent" | https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-s3-vectors-generally-available/ (posted Dec 2, 2025); https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html; https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html (fetched 2026-11-02) |
| Elasticsearch | Yes | Server-side RRF retriever (generally available, rank constant 60) over a BM25 retriever and a vector retriever. ELSER, Elastic's learned keyword model, is recommended for English only, encodes the first 512 tokens and needs "the appropriate subscription level" | Up to 4,096 dimensions; default float index bbq_disk on 9.4+ only where the licence allows, bbq_hnsw on 9.1-9.3 for 384+ dimensions | https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector; https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion; https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-elser (fetched 2026-11-02) |
| pgvector (Postgres 13+) | SQL; with approximate indexes the filter runs after the index scan (iterative scans since 0.8.0 help) | In the application: Postgres full-text search plus RRF or a cross-encoder; the page does not say Postgres ranking is BM25 | HNSW or IVFFlat; indexes up to 2,000 dimensions (4,000 as halfvec). Google AlloyDB adds a ScaNN index; no hybrid statement was fetched for it | https://github.com/pgvector/pgvector; https://docs.cloud.google.com/alloydb/docs/ai/perform-vector-search (fetched 2026-11-02) |
| MongoDB Atlas Vector Search | Pre-filters on indexed boolean, date, objectId, number, string and UUID fields | Server-side: $rankFusion (RRF, constant fixed at 60, MongoDB 8.0+) or $scoreFusion (8.3+); the two sub-searches run one after the other | Up to 8,192 dimensions; HNSW plus exact search; also self-managed or local via the Atlas CLI; no scale or latency figures | https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/; https://www.mongodb.com/docs/atlas/atlas-vector-search/hybrid-search/ (fetched 2026-11-02) |
| Pinecone | Yes, up to 40 KB of filterable metadata per document | Dense + sparse: two linked indexes merged by the client, or one index with client-weighted query vectors | Dense up to 20,000 dimensions; top_k 10,000. Starter plan AWS only (us-east-1); Builder ($20/month flat), Standard ($50/month minimum) and Enterprise list AWS, Azure and Google Cloud. Managed service only | https://docs.pinecone.io/reference/api/database-limits/operation-limits; https://docs.pinecone.io/guides/search/hybrid-search; https://www.pinecone.io/pricing/ (no date shown, fetched 2026-11-02) |
| Qdrant | Rich payload filters | Server-side: prefetch + RRF (v1.10+) or DBSF (Distribution-Based Score Fusion, v1.11+); BM25 on full-text-indexed payload fields | Open source (Apache-2.0 per the research lane, not re-verified) or Qdrant Cloud | https://qdrant.tech/documentation/concepts/hybrid-queries/; https://github.com/qdrant/qdrant (fetched 2026-11-02) |
| Weaviate | Yes | Server-side BM25F (BM25 over several fields) + vector with an alpha weight; relative-score fusion is the default since v1.24 | Most of the code is BSD-3; Enterprise features need a licence key; Docker, Kubernetes or Weaviate Cloud | https://docs.weaviate.io/weaviate/search/hybrid; https://github.com/weaviate/weaviate (fetched 2026-11-02) |
| Milvus / Zilliz | Yes (not detailed on the fetched pages) | A BM25 function turns text into sparse vectors, which combine with dense search through its hybrid search | Up to 32,768 dimensions; topk up to 16,384; Lite, Standalone, Distributed or Zilliz Cloud; Apache-2.0 and "billions of vectors" not re-verified | https://milvus.io/docs/full-text-search.md; https://milvus.io/docs/limitations.md (fetched 2026-11-02) |

Reading the table:
- Filters: every platform documents them. Where they apply matters: storage-optimized Databricks and pgvector filter after the approximate search, and OpenSearch's post_filter can return fewer than k results, so a narrow filter can return fewer results, or none; on BigQuery a pre-filter on non-stored clustered columns forces a full table scan (documented, rows above).
- Hybrid comes in three shapes (documented, rows above). Server-side fusion in one request: Databricks, Azure, Elasticsearch, OpenSearch, MongoDB, Qdrant, Weaviate, Milvus, Snowflake Cortex Search. Bring your own keyword vectors: Vertex, and Pinecone (merged or weighted by the client). In the application: pgvector. BigQuery documents a hybrid without saying, on the fetched page, how it is fused. No hybrid: Amazon S3 Vectors and plain Snowflake VECTOR.
- Gaps in the evidence: most pages give no latency or QPS figure, and prices were not fetched for OpenSearch, Elastic, MongoDB, Qdrant, Weaviate, Milvus or Azure AI Search tiers, so no cost-per-1,000-searches comparison exists [SCR/gtm/research/retrieval.verify.json, missing items].
- The engines the target retailers are documented to run for site search are commerce search products, not vector stores, so they are not rows here, and none was examined as a retrieval backend (documented for Kroger, H&M and H-E-B Mexico) [SCR/gtm/research/retrieval.verify.json RET-41]. Kroger runs Empathy.co for all web and app search since 2020, operated by Kroger itself, and Empathy's case study lists a hybrid index and vector search [https://empathy.co/assets/media/uploads/resources/kroger-building-and-evolving-ai-search-with-empathy.pdf, undated, data to June 2025, fetched 2026-11-02; SCR/gtm/research/gap_2.json G2-01, G2-02]. Kroger's link to Elastic is cluster monitoring, not site search [https://www.elastic.co/industries/retail-ecommerce; https://www.elastic.co/elasticon/tour/2020/columbus/protecting-your-cluster-from-your-humans, fetched 2026-11-02]. H&M overhauled its onsite search across more than 60 markets with Voyado [https://retailtechinnovationhub.com/home/2026/9/27/handm-overhauls-its-onsite-search-across-more-than-60-markets-in-partnership-with-voyado, byline 28 Sep 2026, fetched 2026-11-02]. H-E-B Mexico runs Algolia, since 2023 [https://www.algolia.com/customers/h-e-b-mexico, fetched 2026-11-02]. The H-E-B US engine is unknown (the only evidence is a 2019 talk on Elasticsearch for its Central Market banner), C&A's text search engine is unknown (its only named discovery vendor is Syte, visual search), and Albertsons' keyword engine is unnamed while its conversational layer is Google's Conversational Commerce agent [SCR/gtm/research/gap_2.json G2-06, G2-07, G2-10, G2-17]. C&A's central data warehouse is Snowflake, which makes Cortex Search a natural candidate there (inference) [https://www.snowflake.com/en/blog/c-and-a-build-data-driven-fashion-empire/, Aug 17, 2023]. Plugging CAS onto such an engine is covered in 4.9 and in the architecture section of this study.

### 4.4 Fix these in the code first

1. SEARCH_BACKEND is not validated (code). The setting is only lower-cased [app/config.py:262], and every branch tests for exactly "databricks" [app/search.py:296-303, 310, 539]. A typo such as "databrick" silently serves the local backend, while /healthz reports search_backend=<the typo> with the local embedding model [app/main.py:119-128], and the Databricks start-up checks are skipped [app/main.py:173]. This was traced by reading the code, not by running it. The Databricks App deploy notebook already refuses an app.yaml whose SEARCH_BACKEND is not "databricks" [databricks/09_deploy_app.py:147-166; databricks/app_bundle/app.yaml:21-22], so the hazard is for laptops, notebooks and any other deployment (code). The registry fix in 4.2 removes it (0.5-1 dev-day, estimate).
2. The "no products found" floor is a raw score (code). MIN_RESULT_RELEVANCE is compared with the best raw retrieval score [app/config.py:121-125; app/main.py:1981; app/search.py:551-557], and its value already differs by machine: 0.28 in the local settings file, the default 0.5 elsewhere [CLAUDE.md:1081-1083]. Each backend scores on its own scale:
   - Local cosine similarity: gte-small scores every query 0.83-0.94 and even gibberish about 0.85, so locally the floor has never fired (measured) [CLAUDE.md:1081-1086].
   - Databricks ANN returns 1/(1 + squared distance) (documented) [https://learn.microsoft.com/en-us/azure/databricks/ai-search/ai-search, ms.date 2026-10-29]. Calibration (2026-09-23, keyword set only, gte-large-en with gpt-oss-120b intents) found real queries at 0.658-0.817 and out-of-catalog queries at 0.524-0.665: the ranges overlap. A floor of 0.638 blocked 0 of 18 real queries and 7 of 10 out-of-catalog ones, but it is not set (measured) [CLAUDE.md:1714-1720].
   - Databricks hybrid scores are normalised so the highest possible score is 1 [same page]. Azure's documented hybrid example scores 0.0333 [https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview, ms.date 2026-08-31]. BM25 scores have no fixed range (inference).
   So every new backend, or a switch to hybrid, needs eval/calibrate_relevance re-run before the floor means anything (inference). CAS should not be described as reliably detecting "no good match" [SCR/gtm/research/claims.json CLA-63].
3. The result cache key leaves out the backend and the embedding model (code) [app/main.py:2253-2259]. In a process that switches backend (Databricks notebooks do, through vs_shim), a page cached from the other backend can be served. Both belong in the key.
4. Blank values differ by backend (code). The local loader turns missing text into '' [app/search.py:148-152; app/main.py:354]; Databricks rows keep None or NaN [CLAUDE.md:1748]. Today's readers treat both as absent, but a new loader must choose one on purpose (inference).
5. Smaller items (code). A non-Databricks deployment gets no index probe at start-up [app/main.py:173]. The embedding-comparison evals (eval/compare_embeddings.py, eval/compare_search_text.py) reach into local internals and stay local-only [SCR/gtm/research/retrieval.verify.json RET-38].

### 4.5 Why today's Databricks index cannot do hybrid search or filtering

- What the index holds (code). Its source table, products_vectors, has catalog_index, text_hash, embedding_model and embedding, and the index syncs only catalog_index and embedding [databricks/02_vector_index.py:237-243, 305-319].
- What the docs allow (documented). Only indexed columns can be returned or used as filters, and the index schema is fixed at creation, so any change needs a new index [https://learn.microsoft.com/en-us/azure/databricks/ai-search/create-ai-search, ms.date 2026-10-29].
- So hybrid search or in-index filters on Databricks need a new source table carrying the product text and filter columns, a new index, a re-calibrated floor, and an eval run with the no-LLM control (inference). Today's index took 47.8 minutes to build for 299,972 rows (measured, 2026-10-12) [CLAUDE.md:1541-1542]. No dev-day estimate exists for this change.
- Hybrid specifics (documented): BM25 over all text columns, RRF, at most 200 results (enough for CAS's 60 per phrase), and about 4x the resources of an ANN query; full-text queries, facets and sorting are Beta [https://learn.microsoft.com/en-us/azure/databricks/ai-search/query-ai-search, ms.date 2026-10-29; https://learn.microsoft.com/en-us/azure/databricks/ai-search/best-practices, ms.date 2026-10-29]. Whether hybrid beats plain ANN on CAS's phrases is unmeasured.
- Today's UI does not need in-index filters. The shop filters, sorts and facets run in process on the cached result pool, with 0 LLM calls (code) [CLAUDE.md:679-681]. In-index filtering would matter only if filtering moved into retrieval (inference).
- Source tables (documented). An index can be built only from a Delta table, a streaming table, a materialized view or a managed Iceberg v3+ table, and a standard endpoint needs change data feed (CDF: Delta's log of changed rows, which the refresh reads). A federated (foreign) Hive, BigQuery or Snowflake table cannot be the source; the Hive federation page lists AI Search among features not supported on foreign tables, though a pipeline's materialized view or streaming table built from one can be. CAS already copies the catalog into its own Delta tables, so federation changes only how notebook 01 reads it [SCR/gtm/research/gap_3.json G3-22, G3-23; https://learn.microsoft.com/en-us/azure/databricks/ai-search/ai-search, ms.date 2026-10-29; https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-concepts, fetched 2026-11-02].
- Capacity is sized from the source table when the index is created, and a small start "can result in capacity-exhausted errors", so a pilot index should be built on the full-size catalog (documented) [same page].
- Throughput (documented, never measured for CAS). Standard endpoints: 20-50 ms and 30-200+ QPS, figures given for 768-dimension vectors; beyond one unit QPS plateaus near 30 (ANN), and asking for 10x more results "can double query latency and reduce QPS capacity by 3x". CAS's 1024-d vectors and 30-60 results per phrase therefore lower those figures (inference) [best-practices page above; SCR/gtm/research/gap_1.json G1-12]. CAS queries through the workspace query URL [app/vector_search.py:258-259], which the docs cap at "a few tens of QPS" and say not to use for high-QPS production [https://learn.microsoft.com/en-us/azure/databricks/ai-search/high-qps, ms.date 2026-10-27]. The scalability lane puts that at roughly 5-10 uncached searches per second before HTTP 429 errors, assuming about 5 vector queries per search (estimate) [SCR/gtm/research/scalability.json SCA-11]; at the API default's ~8.2 queries per search it is lower (inference). Extra capacity (target_qps) is billed whether or not traffic arrives, is best effort, and has no published price; the priceable route the docs give is replicating the index across endpoints, at the unit price per replica [high-qps page; SCR/gtm/research/gap_1.json G1-11]. The LLM's token limit binds even earlier: about 2.4 uncached searches per second per workspace on pay-per-token Qwen3-Next (estimate: Enterprise-tier workspace limits, token counts per search taken from a different model, qwen3.8-27b on Groq; the undocumented default output reservation may lower it) [SCR/gtm/research/gap_1.json G1-06, G1-13, G1-14].
- Cost (documented, arithmetic). A standard unit holds 2M vectors at 768-d for 4 DBU/h ($0.28/h US East list price, about $204 a month); at CAS's 1024-d that is about 1.5M vectors per unit, so 300K-1M rows fit one unit (estimate) [https://www.databricks.com/product/pricing/vector-search, no date shown, fetched 2026-11-02; SCR/gtm/research/gap_1.json G1-10]. Full costs are in the cost section of this study.

### 4.6 Embeddings are not portable between models

A vector is only comparable with vectors from the same model, so the catalog and every query must use one model (code comment) [app/config.py:277-279]. Moving a client to its platform's own embedding model (Snowflake, Azure OpenAI, Vertex and so on) means re-embedding the catalog, rebuilding the index, re-calibrating the floor and re-running the evals (inference).

What that has cost so far (measured): about 24 hours on a laptop CPU for 300K rows locally [CLAUDE.md:12]; on Databricks (2026-10-12), about 38 minutes and 44.73M tokens for 296,483 distinct texts, plus the 47.8-minute index build [CLAUDE.md:1537-1542]. The "~83 DBU" for that embedding run is the script's own estimate, never confirmed from billing (about $5.81 at the $0.07/DBU list rate, arithmetic) [SCR/gtm/research/retrieval.verify.json RET-37; SCR/gtm/research/gap_1.json G1-24]. On a Databricks refresh only changed text is re-embedded, keyed by model and text hash (code) [SCR/oct27/byo/REPORT.md:74].

Quality does not carry over between models either. On one rerank-off run of the 18-query keyword set (2026-09-11, base catalog before enrichment), gte-large-en scored NDCG@10 0.955 against gte-small's 0.927 (measured; NDCG@10 rewards relevant items near the top of the first 10) [CLAUDE.md:1554-1562; SCR/gtm/research/retrieval.verify.json RET-37]. That set is saturated (raw retrieval with no LLM already scores near the full pipeline), so it shows the direction more than the size of the gap [.claude/rules/eval.md:24].

| Model | Where CAS uses it | Licence | Dimensions / input limit | Language | Source |
|---|---|---|---|---|---|
| thenlper/gte-small | Local backend | MIT | 384 / 512 tokens (CAS caps input at 128) | English only | https://huggingface.co/thenlper/gte-small (fetched 2026-11-02); app/embeddings.py:47-48 |
| databricks-gte-large-en = Alibaba-NLP/gte-large-en-v1.5 | Databricks backend | Apache-2.0 (model card) | 1024 / 8,192 tokens; not normalised by the endpoint (CAS normalises) | "the English version of the model" | https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/supported-models (ms.date 2026-10-30); https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5 (fetched 2026-11-02) |
| Qwen3-Embedding-0.6B (databricks-qwen3-embedding-0-6b) | Not used; Databricks' recommended model for production on standard endpoints, served with provisioned throughput (the model is in Public Preview) | Apache-2.0 | Up to 1024 / 32k tokens | 100+ languages (documented) | https://learn.microsoft.com/en-us/azure/databricks/ai-search/create-ai-search (ms.date 2026-10-29); https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/supported-models (ms.date 2026-10-30); https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (fetched 2026-11-02) |

- Both models in use have open weights, so they can be run outside Databricks (documented licences; the large model loads with trust_remote_code=True). gte-large-en-v1.5 is a later generation of the Alibaba GTE family, not the "large" sibling of gte-small, so local and Databricks vectors were never comparable [SCR/gtm/research/retrieval.verify.json RET-36].
- Whether self-hosted gte-large-en-v1.5 gives the same vectors as the Databricks endpoint is undocumented and untested. If it does, the stored vectors could move to another platform without re-embedding (inference).
- Language (measured on a small sample). Both models are English models. An illustrative offline gte-small probe (6 hand-labelled queries against 15 titles per language) put the right title first for 6 of 6 English, 3 of 6 German and 4 of 6 Swedish queries; gte-large-en was not probed [SCR/gtm/research/gap_4.json G4-20]. The multilingual Qwen3 model is untested by CAS, and switching to it means re-embedding and re-measuring everything [SCR/gtm/research/gap_4.json G4-21].
- Region (documented). databricks-gte-large-en is listed pay-per-token, with no cross-geography mark, in the six Azure EU Data Boundary regions that host Foundation Model APIs (in uksouth it carries the mark) and in AWS eu-central-1 and eu-west-1/2/3. On Google Cloud the region table and the Designated Services page disagree, and europe-west1 reads "Not supported" [SCR/gtm/research/gap_4.json G4-03, G4-04, G4-05, G4-06; https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability, ms.date 2026-10-30; https://docs.databricks.com/aws/en/unity-gateway/model-region-availability, Oct 30, 2026; https://docs.databricks.com/gcp/en/resources/designated-services, Oct 26, 2026]. Whether AI Search itself is offered in each client region was not checked [SCR/gtm/research/retrieval.json RET-20].

### 4.7 The local backend: a measured slowdown with a small fix

The first six rows were measured on one laptop (Intel Core Ultra 7 165H, 22 threads, 64 GB), single sessions, no LLM, read-only [SCR/gtm/research/probe_scalability.json; SCR/gtm/research/retrieval_probe/probe_out.json]. The last row is the retrieve stage timed inside the 2026-11-02 held-out run (local backend; that run also made LLM calls, timed separately) [SCR/gtm/research/latency_from_nov02.txt].

| Measurement | Median per phrase | Notes |
|---|---|---|
| Code as written, 299,972 x 384, top 60 (n=12) | 366 ms (326-415) | scikit-learn cosine_similarity + full sort, the path CAS runs per phrase [app/embeddings.py:117-121]. Verifier re-run: 368 ms (n=20) |
| Code as written, same matrix, top 50 (n=20) | 460 ms (p95 595) | Second probe, same file |
| The cosine_similarity call alone (profiling run) | 482 ms | A plain matrix-vector product on the same matrix: 7.4 ms; full sort 20.9 ms vs partial sort 3.2 ms |
| Fixed: plain product + partial sort | 10.1 ms (n=12); 12.1 ms (n=20) | Query vectors were catalog rows: both methods ranked the query's own row first on 12 of 12. Verifier re-run: 10.4 ms |
| 300,000 x 1024 synthetic (Databricks-size vectors) | 1,374 ms as written; 25.7 ms fixed | |
| 1,000,000 x 384 synthetic, fixed | 36.8 ms (22 threads); 96.7 ms (2 threads) | As written: not run (about 3 GiB peak); about 1.5 s per phrase (linear estimate) and about 1.5 GB of temporary memory per in-flight call (estimate scaled from a 0.44 GB measurement at 300K; the probe's own derived estimate was 2.3 GB) [SCR/gtm/research/scalability.verify.json SCA-18] |
| Live retrieve stage, 31 held-out queries (encode + scan + hydration) | p50 847 ms, mean 1,280 ms, p90 2,365 ms | Per search, not per phrase; local gte-small |

- Why it is slow (measured by profiling): the scikit-learn cosine_similarity call took 482 ms of the roughly 500 ms step, against 7.4 ms for a plain matrix-vector product, because it normalises a copy of the whole matrix on every call. That copy is about 0.44 GB of temporary memory per in-flight call at 300K (tracemalloc measurement in the verifier's re-run; the probe's ~0.66 GB peak-memory difference likely also holds one-off buffers) [SCR/gtm/research/probe_scalability.json; SCR/gtm/research/scalability.verify.json SCA-04, SCA-18]. The stored vectors are already unit length (norms 0.9994-1.0006), so a plain matrix-vector product plus a partial sort gives near-identical top-60 lists (mean overlap 0.999, minimum 0.983, same order on 13 of 20 queries, in the verifier's re-run) [SCR/gtm/research/scalability.verify.json SCA-04, SCA-24].
- Do not normalise into a copy: in the probe that raised memory from 593 MB to 1,033 MB [SCR/gtm/research/probe_scalability.json].
- The change is a few lines in app/embeddings.py (inference) and has not been made. It should come with a test that the top-60 lists stay the same. It affects only SEARCH_BACKEND=local; Databricks is unaffected.
- With the fix, the local scan stays under about 100 ms per phrase up to 1M x 384 vectors even on 2 threads (measured, synthetic). Catalogs of several million rows, or 1024-d vectors at millions of rows, need an ANN service (inference).
- This corrects an internal line: docs/RESEARCH.md:116-117 says "At 60K products, no vector DB needed". The served catalog has 299,972 rows, and the line holds there only with the fix (inference). Do not reuse it [SCR/gtm/research/claims.json CLA-66].

### 4.8 The real ceiling: the catalog and keyword index inside each process

- Every backend keeps the whole catalog and the BM25 keyword index in each API process (code) [SCR/gtm/research/scalability.json SCA-01]. Moving vectors to a managed store does not change that.
- Size. 1,486 MB of process memory was measured for catalog + keyword index on 2026-10-12/13, before later columns were added [CLAUDE.md:1779-1782]. The 2026-11-02 probe measured the keyword index at about 1.0 GB and the served catalog table at about 0.6 GB for 299,972 rows (laptop). It estimates about 1.7 GB per API process at 300K rows and about 5.2 GB at 1M on the Databricks App path, and about 2.3 GB at 300K on the local path, which also holds the vectors (linear steady-state estimates, boot-time peaks not included, nothing measured above 300K real rows) [SCR/gtm/research/probe_scalability.json; SCR/gtm/research/retrieval.verify.json RET-08]. A Medium Databricks App has 6 GB, shared with the UI: by that estimate it holds one or two API processes at 300K and none at 1M [SCR/gtm/research/scalability.json SCA-16; SCR/gtm/research/scalability.verify.json SCA-16].
- Speed (measured, laptop). A keyword query takes a median 320.5 ms at 300K rows and grows roughly in line with rows (101.7 ms at 100K), because it loops in Python over every document. Hydrating 300 hits takes 140.8 ms, and 4 threads ran it at 0.89x of one thread (the keyword search at 1.15x): Python's global interpreter lock (GIL) lets only one thread run Python code at a time [SCR/gtm/research/probe_scalability.json; SCR/gtm/research/scalability.json SCA-03, SCA-06].
- Fetching full rows from the store instead of from memory would drop the columns CAS derives at load time, unless they were written into the store (inference) [SCR/gtm/research/retrieval.verify.json RET-08]. Scaling options are in the scalability section of this study.

### 4.9 Handing keyword search to the platform, or sitting on the client's engine

- Today's Keyword Search does not use SEARCH_BACKEND (code). It is an in-process BM25 engine whose search() returns four values: results, number matched, top relevance and top coverage (the "4-tuple") [app/keyword_search.py:150-204]. It ranks by 0.7 BM25 + 0.2 log-popularity + 0.1 rating [app/keyword_search.py:39-41]. The keyword-first router counts a query as a keyword hit when the top result covers at least 75% of its terms [app/hybrid_router.py:18-25]. That router has never been measured or covered by a test, and no UI page calls it [SCR/gtm/research/architecture.json ARC-01].
- Most platforms in 4.3 have their own BM25 or hybrid search. Handing Keyword Search to one means rebuilding the 4-tuple, the popularity/rating blend and the coverage signal from the platform's top hit: about 2-4 extra dev-days (estimate) [SCR/gtm/research/retrieval.json RET-39]. It would remove the ~1 GB per 300K rows and the per-document loop (inference; unmeasured on this catalog) [SCR/gtm/research/scalability.json SCA-23].
- CAS's tokeniser keeps only a-z and 0-9, so German or Swedish words split into fragments ("Kopfhörer" -> "kopfh", "rer") (code, measured) [app/keyword_search.py:12; SCR/gtm/research/gap_4.json G4-15]. A platform's language analysers could help, but no non-English evaluation exists.
- A client's existing search engine (keyword or vector) could be CAS's retrieval backend through the same seam, on three conditions (inference): its document ids map to CAS's catalog rows; CAS still loads a full copy of the catalog for hydration and derived columns; and the relevance floor is replaced or re-calibrated, since BM25 scores have no fixed range [SCR/gtm/research/retrieval.verify.json RET-41; SCR/gtm/research/architecture.json ARC-06]. It also needs stable product ids instead of today's re-assigned row numbers (5.5 dev-days with incremental refresh, estimate) [SCR/oct27/byo/REPORT.md:407]. Commerce search products (Algolia, Bloomreach, Constructor, Coveo, Google's commerce search) were not examined as backends [SCR/gtm/research/retrieval.verify.json RET-41].
- An offline proxy gives a first idea (measured, small). CAS's own BM25 engine stood in for a client keyword engine and received CAS's recorded phrases, on 16 inferred-intent queries with one gpt-oss-20b sample per prompt. P@10 (the share of the top 10 results that are relevant) was 0.206 for the raw query and 0.369-0.394 for CAS's phrases through the keyword engine on the shipped prompt (0.41-0.46 on the previous prompt), against 0.463 for CAS's own vector retrieval with re-ranking off. On the shipped prompt that is about 63% of the gap between the raw query and CAS's own retrieval, and the RRF-merged variant lost to CAS's own retrieval on 9 of 16 queries (3 wins, 4 ties) [SCR/gtm/research/architecture.verify.json ARC-12]. Gaps under ~0.05 P@10 on 16 queries count as a tie [.claude/rules/eval.md:65]. Keyword-only retrieval of CAS phrases has not been measured live.

### 4.10 Determinism

- The local scan is exact and repeatable (measured): with no LLM, a re-run in a fresh process with another hash seed reproduced 34 of 34 ordered 24-row lists [SCR/nov02/REPORT.md:37].
- For ANN indexes (Databricks' ANN index and every other ANN row in 4.3), whether the same query returns the same top 60 run to run is undocumented and unmeasured. The LLM reply memo pins the LLM's answers, not retrieval [SCR/gtm/research/retrieval.verify.json, missing items]. The Databricks determinism notebook (databricks/12_determinism.py) has never been run on the workspace [CLAUDE.md:1693-1698]. An index rebuild may also change ANN results (inference).

### Open questions for this area

- Which engine and data platform does the first pilot client run (Kroger: Empathy.co; H&M: Voyado; C&A: Snowflake warehouse, text search engine unknown; H-E-B US and Albertsons' keyword engine unknown)? That decides which adapter to build first, not the feature table. Owner: Sai / Niharika.
- Does self-hosted Alibaba-NLP/gte-large-en-v1.5 produce the same vectors as the Databricks endpoint? Test: 100 catalog texts, cosine of both outputs. If yes, stored vectors can move platforms without re-embedding.
- Does Databricks AI Search return identical top-60 lists run to run? Run databricks/12_determinism.py or a direct repeat on the workspace.
- Does hybrid beat plain ANN on CAS's phrases? Needs a new index with text columns, a re-calibrated floor and run_eval with the no-LLM control. No dev-day estimate exists yet.
- How many searches per second can CAS's Databricks path really serve with 1024-d vectors, 60 results per phrase and the workspace query URL, and should it move to the route the docs recommend for high QPS? No load test exists.
- Is Snowflake Cortex Search's documented 20 QPS per service enough for a pilot at 3-8 queries per shopper search (about 6-7 three-phrase searches per second, estimate) [SCR/gtm/research/retrieval.verify.json RET-40]? Is BigQuery VECTOR_SEARCH (bytes-scanned billing, no latency figure) usable for online search at all?
- Which floor should each backend use, and should MIN_RESULT_RELEVANCE become a per-backend setting? The Databricks value 0.638 came out of calibration but is not set, and only the keyword set was calibrated.
- Is AI Search, and each alternative platform, offered in the client's region? Not mapped (S3 Vectors launched in 14 regions; Cortex Search uses cross-region inference in some).
- Not fetched: Vertex AI Vector Search quotas (HTTP 404), prices for OpenSearch, Elastic, MongoDB, Qdrant, Weaviate, Milvus and Azure AI Search tiers, the clouds offered by Elastic Cloud and MongoDB Atlas, and whether Postgres full-text ranking is BM25.

## 5. Language models: portability, cost and throughput

Answer. A new search makes 2 calls to a large language model (LLM); page turns, filters, sorts and exact repeat searches make none [code: app/main.py:1928-2022, app/main.py:2263-2369]. On today's prompts a new search uses about 4,200 input and 690 output tokens, roughly $1.46 per 1,000 new searches on our Databricks model at list price; the tokens were counted on Groq's qwen3.8-27b preview model, not on the Databricks model, and the price uses Databricks' $0.07/DBU example rate [tokens measured, cost an estimate: SCR/gtm/research/llm.verify.json LLM-09, LLM-40, LLM-42]. The code already reaches hosts that expose the OpenAI chat protocol with a fixed key; production use on Azure (Entra ID), AWS Bedrock or Google Vertex needs an estimated 1.5-2.5 dev-days of auth and request work plus a smoke test per host; Azure OpenAI, Bedrock, Vertex, Snowflake and self-hosted vLLM have never been called by CAS [estimate and code: SCR/gtm/research/cloud_ops.verify.json CLO-27]. The real limits are capacity, latency and region: one Databricks pay-per-token endpoint allows at most about 145 new searches a minute on the published Enterprise-tier limits (an upper-bound estimate), a search took about 10 s there in our eval harness (measured, Sep 23 and Oct 12; never measured on the deployed App), and our chosen model, Qwen3-Next (Public Preview), is listed in no Azure EU region and for provisioned throughput in no region. On Azure EU the only listed model CAS has run is gpt-oss-120b (or 20b); gpt-oss-120b was about 3x slower in our one comparison (30.3 s vs 10.4 s per query, keyword set, Sep 23) [https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits, ms.date 2026-10-30; CLAUDE.md:777-780; SCR/gtm/research/gap_4.json G4-01, G4-03, G4-08, G4-11].

How to read the labels: measured = we ran it; code = read from the repo; documented = a vendor page says so; estimate = arithmetic on measured or documented numbers; inference = our reading, not stated anywhere. No load test exists in the repo, and nothing about scale has been measured on Databricks.

### 5.1 How many LLM calls each action makes

CAS calls the LLM twice per new search: once to turn the shopper's words into search phrases and constraints ("translate"), and once to re-score the top 15 retrieved products ("rerank"). Everything after that reuses the cached result pool.

| action | LLM calls | other calls (Databricks path) | kind | source |
|---|---|---|---|---|
| New search from the AI Search page | 2 (translate, then rerank of up to 15 rows) | 1 embedding call, ~3.2 Vector Search queries (one per search phrase) | code; measured: all 31 held-out queries made exactly 2 calls | app/main.py:1928-1933; app/main.py:2015-2022; app/config.py:129; pages/ai_search.py:120; SCR/gtm/research/gap_1.json G1-01; SCR/nov02/live/results.json |
| Page 2+, page size up to 30, filter, sort, facets (while the pool is in this process's 64-entry page cache) | 0 | 0 | code; measured: page 2 from cache with 0 calls on 29 of 29 flat queries, median 2 ms (Groq qwen3.8-27b, local) | app/main.py:2263-2369; app/config.py:127, 131; SCR/nov02/REPORT.md:56 |
| Exact repeat of a search (case and spacing ignored), after a restart or cache eviction | 0 provider calls (the LLM reply memo replays the stored reply) | still 1 embedding call and ~3.2 Vector Search queries | measured Oct 30 (Groq qwen3.8-27b, local, 6 queries): identical over 3 runs, 0 calls on runs 2-3; code | .claude/rules/eval.md:502-509; SCR/gtm/research/gap_1.json G1-05 |
| Two identical searches at the same moment, in the same process | 1 pipeline run; the second waits up to 60 s for it, then runs its own | - | code | app/main.py:1415; app/main.py:1453 |
| Gibberish (the translator finds no intent) or a pool below the relevance floor | 1 (translate only; no rerank) | at most 1 embedding call | code | app/translator.py:496-497; app/main.py:1981-1982 |
| Product page open | +1 cross-sell call | 5-6 embedding calls and 5-6 Vector Search queries, one after another; results cached per process (256 entries, keyed by product and query) | code | app/main.py:2537-2568; app/page_cache.py:21; SCR/gtm/research/gap_1.json G1-04 |
| /search called by an API client that omits `recommend` | 3 (the default adds a cross-sell call on page 1) | 6-7 embedding calls, ~8.2-9.2 Vector Search queries | code | app/main.py:2070-2072; app/config.py:162; SCR/gtm/research/gap_1.json G1-02 |
| Shopping Assistant chat turn | +1 assistant call; a "search" turn then adds the 2 above (0 if cached); a "refine" turn adds 0 while the pool is still in this process's 64-entry cache or the memo, else 2 | as the search | code (corrected verdict) | app/assistant.py:15; app/assistant.py:158-159; SCR/gtm/research/llm.verify.json LLM-05 |
| Query that opens with an instruction ("ignore the rules...") | 0 (the injection guard sends a literal embedding search) | 1 embedding call | code | app/translator.py:604-614 |
| A stage failed and fell back (rate limit, bad JSON) | the result is never cached, so the next fresh submit pays again; page and filter clicks reuse the degraded pool for 120 s | - | code | app/main.py:1349; app/main.py:1418-1450; SCR/gtm/research/llm.verify.json LLM-02 |

One integration rule follows. A client system that calls `/search` must pass `recommend=false`. The default (`RECOMMEND_ENABLED=true`) adds 1 LLM call, about 5 embedding calls and about 5 Vector Search queries to every new search [code: app/main.py:2070-2072, app/config.py:162; SCR/gtm/research/gap_1.json G1-32]. Our own UI and the `/unified_search` and `/chat` routes already send false [code: pages/ai_search.py:120, app/main.py:2490, app/main.py:2653].

### 5.2 Tokens per search

| item | input tokens | output tokens | kind | source |
|---|---|---|---|---|
| New search, today's prompts (31 held-out queries, Nov 2, memo off, Groq qwen3.8-27b, local) | 4,213 mean | 689 mean (total 4,902; median 4,959; range 3,276-5,237) | measured | SCR/nov02/live/results.json; SCR/gtm/research/llm.verify.json LLM-09 |
| of which translate | 1,461 | 66 | measured | same |
| of which rerank (15 rows, one reason sentence per row) | 2,751 | 623 | measured | same; app/reranker.py:104 |
| Qwen3-Next on Databricks, older and shorter prompts (Sep 23) | 2,709 | 658 | measured | CLAUDE.md:1729 |
| gpt-oss-120b on Databricks, same run (a reasoning model) | 2,806 | 2,177 | measured | CLAUDE.md:1730 |
| gpt-oss-120b scaled to today's prompts | 4,213 | ~2,280 | estimate (the Sep 23 output ratio, 2,177 / 658 = 3.3x, applied to today's 689; never measured on today's prompts) | SCR/gtm/research/llm.json (cost table); SCR/gtm/research/llm.verify.json, missing items |
| Product-page cross-sell call (6 calls, Oct 30) | 238 | 48 | measured | SCR/oct27/live_c/determinism/ledger.json; SCR/gtm/research/llm.verify.json LLM-10 |
| Chat turn (assistant call only) | ~800-4,100 | not measured | estimate (prompt 3,180 characters + up to 6 history turns of 2,000 characters + a 1,000-character message) | app/assistant.py:15; app/main.py:2582-2588; SCR/gtm/research/llm.verify.json LLM-13 |

Three caveats. First, the Nov 2 tokens were counted on Groq's qwen3.8-27b, a different model and tokenizer from Qwen3-Next, and Groq lists it as a Preview model "intended for evaluation purposes only" [documented: https://console.groq.com/docs/models, fetched 2026-11-02]. Second, token counts are not comparable across model families [CLAUDE.md:1733]. Third, the 31 queries include one setup bundle, one recipe and one partial rerank [SCR/gtm/research/llm.verify.json LLM-09]. Prompt size is also a lever: vertical-specific bullets are about 45% of the prompt text, so a grocery-only client pays about 475 tokens of fashion rules per rerank, about 17% of the 2,751-token rerank prompt (estimates) [SCR/oct27/byo/REPORT.md:281; SCR/gtm/research/llm.verify.json LLM-14].

### 5.3 Cost per 1,000 new searches, by model and host

Every figure below is an estimate: measured tokens multiplied by the host's list price. It counts the 2 LLM calls only. The query embedding adds about $0.003 per 1,000 (~0.2%) [estimate: SCR/gtm/research/gap_1.json G1-27]; the Vector Search endpoint and App compute are fixed costs outside this table (5.4). Databricks bills in DBUs (Databricks Units); its page quotes $0.07 per DBU as an example rate [https://www.databricks.com/product/pricing/foundation-model-serving, no date shown, fetched 2026-11-02], which equals the Azure East US Premium list meter for serverless model serving; West Europe is $0.084 (+20%) [https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Azure Databricks' and armRegionName eq 'eastus' and https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Azure Databricks' and armRegionName eq 'westeurope', both fetched 2026-11-02; SCR/gtm/research/gap_1.json G1-08].

| model | host | list $ per 1M tokens in / out | tokens in / out used | $ per 1,000 new searches (estimate) | source |
|---|---|---|---|---|---|
| Qwen3-Next 80B A3B (Public Preview) | Databricks pay-per-token, $0.07/DBU (US) | $0.15 / $1.20 | 4,213 / 689 | $1.46 | https://www.databricks.com/product/pricing/foundation-model-serving (fetched 2026-11-02) |
| Qwen3-Next 80B A3B | same, with the tokens Databricks itself measured (older prompts) | $0.15 / $1.20 | 2,709 / 658 | $1.20 | CLAUDE.md:1729; same page |
| gpt-oss-120b (reasoning) | Databricks pay-per-token (US); Azure AI Foundry Global; Groq | $0.15 / $0.60 | 4,213 / ~2,280 (est.) | $2.00 | same page; https://prices.azure.com/api/retail/prices?$filter=armRegionName eq 'eastus' and productName eq 'Azure OpenAI OSS Models' (effective 2026-03-01); https://console.groq.com/docs/models |
| gpt-oss-120b | Databricks pay-per-token, West Europe ($0.084/DBU) | $0.18 / $0.72 | 4,213 / ~2,280 (est.) | $2.40 | SCR/gtm/research/gap_1.json (unit price table) |
| gpt-oss-20b (reasoning) | Databricks pay-per-token (US); Groq about the same | $0.07 / $0.30 | 4,213 / ~2,280 (est.; 120b's output ratio assumed, 20b's own unmeasured) | $0.98 | https://www.databricks.com/product/pricing/foundation-model-serving; SCR/gtm/research/llm.verify.json LLM-42 |
| Qwen3 Next 80B A3B | AWS Bedrock on demand, US East / Ohio / Oregon | $0.15 / $1.20 | 4,213 / 689 | $1.46 | https://aws.amazon.com/bedrock/pricing/ (no date shown, fetched 2026-11-02) |
| Qwen3 Next 80B A3B | AWS Bedrock, Europe (Ireland, Milan) | $0.18 / $1.41 | 4,213 / 689 | $1.73 | same; SCR/gtm/research/gap_1.json G1-21 |
| gpt-oss-120b | AWS Bedrock Standard (price shown for Sydney only) | $0.1545 / $0.618 | 4,213 / ~2,280 (est.) | $2.06 | https://aws.amazon.com/bedrock/pricing/ |
| gpt-4.1-mini | Azure OpenAI Global / OpenAI API | $0.40 / $1.60 | 4,213 / 689 | $2.79 | https://prices.azure.com/api/retail/prices?$filter=armRegionName eq 'eastus' and contains(skuName,'4.1 mini'); https://developers.openai.com/api/docs/pricing (fetched 2026-11-02) |
| gemini-2.5-flash-lite | Gemini Developer API, paid tier (not Vertex; Vertex prices did not load) | $0.10 / $0.40 | 4,213 / 689 | $0.70 | https://ai.google.dev/gemini-api/docs/pricing (fetched 2026-11-02) |
| qwen3-next-80b-a3b (Preview) | Snowflake AI_COMPLETE, a SQL function our client cannot call | $0.18 / $1.44 | 4,213 / 689 | $1.75 | https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf (effective Oct 29, 2026) |
| openai-gpt-5-mini (a reasoning model, so a lower bound; Preview footnote in the table) | Snowflake Cortex REST | $0.25 / $2.00 | 4,213 / 689 | at least $2.43 | same PDF; SCR/gtm/research/llm.verify.json LLM-37 |
| claude-haiku-4-5 | Snowflake Cortex REST | $1.00 / $5.00 | ~4,790 / ~752 (est.) | $8.55 | same PDF |
| qwen/qwen3.8-27b (Preview; the model behind our Oct 30 / Nov 2 numbers) | Groq | $0.80 / $4.00 | 4,213 / 689 | $6.13 | https://console.groq.com/docs/models |
| Qwen3-Next or gpt-oss-120b, open weights | self-hosted vLLM on the client's GPUs | GPU-hours, not tokens | same | not priced | https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/ (page dated October 31, 2026) |

Add-ons, all estimates: a product-page open adds about $0.10 per 1,000 opens (tokens plus 5 embedding calls), and an API caller that leaves `recommend` on adds about $0.10 per 1,000 searches (+7%) [SCR/gtm/research/gap_1.json G1-27]. If 30% or 50% of new searches were exact repeats, the Qwen3-Next figure would fall to about $1.02 or $0.73; the real repeat share needs a client query log [SCR/gtm/research/llm.verify.json LLM-43]. The Groq preview model costs about 4.2x Qwen3-Next and should not appear in a client cost model [SCR/gtm/research/llm.verify.json LLM-45]. The rows price tokens, not quality: gpt-4.1-mini and gpt-5-mini appear nowhere in the repo, so CAS has never run them, and Azure, Bedrock, Vertex, Snowflake and self-hosted vLLM have never been called (5.5) [code: repo search outside .venv, 2026-11-02; SCR/gtm/research/cloud_ops.verify.json CLO-27].

### 5.4 Is token cost the deciding factor?

Per search, no. At retailer volume, it becomes the largest line. Capacity and latency decide feasibility first.

- The always-on floor is about $551 per month at East US list prices: one standard Vector Search unit ($204) plus one Medium Databricks App ($347). With the 2 App instances Databricks recommends for availability it is about $898 (estimate from documented prices) [SCR/gtm/research/gap_1.json G1-25].
- LLM tokens reach that floor at about 378,000 new (uncached) searches a month, about 0.15 per second on average (estimate) [SCR/gtm/research/gap_1.json G1-26].
- Above that, tokens dominate. At 1 search per second around the clock (2.6M a month) on a 300K-product catalog, tokens are about $3,834 of a $4,392 monthly bill (87%); if 75% of searches were answered from the memo or page cache, the bill would be about $1,517 (estimates) [SCR/gtm/research/gap_1.json G1-26, G1-28].
- So tokens are not "small next to the platform cost" at retailer volume: 1M new searches a month on Qwen3-Next is about $1,459, about 7x one standard Vector Search unit ($204) and about 2.6x the $551 floor (estimates) [SCR/gtm/research/llm.verify.json LLM-42 (corrected); SCR/gtm/research/gap_1.json G1-25].
- What moves the token bill: the share of searches served from cache (each point of hit share removes about 1% of LLM spend), `recommend=false` for integrations, and a rerank that returns scores without a reason sentence per row (estimated -36% cost; needs a quality test) [estimates: SCR/gtm/research/gap_1.json G1-32].
- For scale, Instacart reports that behind a cache of frequent ("head") queries only 2% of its queries needed the model at request time. That is Instacart's traffic, not a CAS measurement [documented: https://company.instacart.com/tech-innovation/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms, Nov 21, 2025].

### 5.5 Portability: which LLM hosts the code can reach

`app/llm_client.py` has three backends [code: app/llm_client.py:232-266, 295-378, 381-419]:
1. OpenAI-compatible: the OpenAI SDK pointed at any base URL, with a fixed key or (on Databricks only) a token that refreshes itself. Most hosts below use this path, switched by three settings: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` [code: app/config.py:51-77].
2. Gemini: the Gemini Developer API with an API key, not Google Vertex [code: app/key_rotator.py:24].
3. Anthropic: the Anthropic API directly, `max_tokens` 1024 [code: app/llm_client.py:399].

The pipeline has run on Gemini, on Groq (gpt-oss-120b, gpt-oss-20b, qwen3.8-27b) and on Databricks (gpt-oss-120b, Qwen3-Next, and Claude Haiku 4.5 in the Sep 23 comparison). Provider quirks found on the way (Databricks rejects `seed`, returns typed reply blocks; Claude endpoints refuse JSON mode; one Claude model refuses `temperature`) are handled in code [measured and code: CLAUDE.md:1576-1600, CLAUDE.md:1724-1732; SCR/gtm/research/claims.json CLA-70].

| host | reachable by configuration today? | production authentication | JSON mode / `seed` (as documented) | our models there | source |
|---|---|---|---|---|---|
| Databricks Foundation Model (FM) API, pay-per-token | Yes, in use | Workspace or App service-principal token, refreshed per request (built) | JSON mode yes; `seed` rejected with a 400, so the code skips it | Qwen3-Next, gpt-oss-120b / 20b | https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference (Oct 2, 2026); app/llm_client.py:201-203, 269-292 |
| Azure OpenAI / AI Foundry (v1 API) | Yes, with a resource key: base URL `https://<resource>.openai.azure.com/openai/v1/`, model = deployment name | Entra ID needs a token-provider option (not built) | JSON mode yes for the models the page lists (the list did not render; "JSON" must appear in the messages; ours do); `seed` Preview, best effort; neither checked for gpt-oss on Foundry | gpt-oss-120b, gpt-4.1-mini | https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle (ms.date 2026-05-13); https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/json-mode (ms.date 2026-05-13); https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reproducible-output (ms.date 2025-11-26) |
| AWS Bedrock (`bedrock-runtime .../openai/v1`) | Yes, with a long-term API key, which AWS recommends "only for exploration" | Short-term keys (up to 12 h, AWS's production advice) or SigV4 request signing: needs the token-provider option | not documented on the fetched page: smoke test needed (our code sends `seed`) | gpt-oss-120b (AWS's own example); Qwen3 Next priced, on this route unconfirmed | https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html; https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html (fetched 2026-11-02) |
| Google Vertex AI (`.../endpoints/openapi`) | Protocol yes, but the key is an OAuth access token that lasts 1 hour | Needs the token-provider option | not documented for these models | gpt-oss 120B managed; Qwen3-Next not listed | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate/openai/auth-and-credentials; https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/use-open-models (fetched 2026-11-02) |
| Gemini Developer API | Yes, own backend (API key) | API key | structured output yes; OpenAI-library compatibility "still in beta" | Gemini models | https://ai.google.dev/gemini-api/docs/openai (fetched 2026-11-02) |
| Snowflake Cortex REST (OpenAI SDK route) | Yes, with a PAT (programmatic access token): base URL `https://<account>.snowflakecomputing.com/api/v2/cortex/v1` | PAT, JWT or OAuth | `response_format` honoured for OpenAI models, JSON-schema only for Claude, ignored for others; `seed` not listed; `max_tokens` errors (use `max_completion_tokens`) | no gpt-oss or Qwen on this route; gpt-5-mini, claude-haiku-4-5 | https://docs.snowflake.com/user-guide/snowflake-cortex/open_ai_sdk (fetched 2026-11-02) |
| Snowflake AI_COMPLETE (SQL function) | No: needs a new backend | Snowflake session | - | qwen3-next-80b-a3b (Preview) | https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf |
| Self-hosted vLLM (the client's cloud or on-premises) | Yes: OpenAI-compatible server with `--api-key` (which protects only the /v1, /v2 and /inference paths) | Fixed key behind the client's network | page silent on `seed` / `response_format` for Chat: smoke test needed | Qwen3-Next, gpt-oss (Apache-2.0 weights) | https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/ (Oct 31, 2026); SCR/gtm/research/llm.verify.json LLM-39 |
| Groq (our development host) | Yes, in use | Fixed key | JSON mode yes (gpt-oss-20b returned json_validate_failed errors); `seed` accepted, best effort | gpt-oss-120b / 20b (Production), qwen3.8-27b (Preview) | https://console.groq.com/docs/models (fetched 2026-11-02); .claude/rules/eval.md:179; SCR/nov02/REPORT.md:13-16 |

None of Azure OpenAI, Bedrock, Vertex, Snowflake or vLLM has ever been called by CAS. Before any claim about a host, the Databricks acceptance smoke test (notebook 08: 43 LLM calls, about 55K tokens, estimated) should pass on it, including the injection rows [SCR/gtm/research/cloud_ops.verify.json CLO-27; CLAUDE.md:1663].

### 5.6 Three code gaps before another host goes to production

| gap | what happens today | fix | effort | source |
|---|---|---|---|---|
| 1. Refreshing credentials | Only `OPENAI_AUTH=databricks` passes a self-refreshing token. Azure Entra ID, Bedrock short-term keys (up to 12 h) and Vertex tokens (1 h) would expire mid-session, and every call would then fail and fall back (inference). | A token-provider option in the OpenAI backend, the same pattern the Databricks path uses (not a new backend). | ~1-1.5 dev-days (estimate) | code: app/llm_client.py:269-313; SCR/gtm/research/cloud_ops.verify.json CLO-27 |
| 2. `seed` sent to every non-Databricks host | With `DETERMINISTIC` on (the default), the code sends `seed=42` unless the base URL contains "serving-endpoints". JSON mode and temperature refusals are learned and retried; a `seed` refusal is not, so every call would 400 and every stage would silently fall back (translate to the raw query, rerank to embedding order). That is how Databricks first failed. | An explicit setting for sending `seed`, on only for hosts known to accept it; also check `response_format` per host. | ~0.5-1 dev-day (estimate) | code: app/llm_client.py:201-203, 331-332, 166-173, 353-372; app/config.py:107-108; CLAUDE.md:1576-1580; SCR/gtm/research/cloud_ops.verify.json CLO-08, CLO-27 |
| 3. No timeout, retry limit or output cap | The OpenAI client is built with no timeout, `max_retries` or `max_tokens`, so the SDK defaults apply (600 s read timeout, 2 retries) while the UI gives up at 90 s (/search), 60 s (/product) and 120 s (/chat). Only the Databricks start-up probe sets 20 s and 0 retries. On Databricks, a request without `max_tokens` reserves an undocumented default output amount against the per-minute output limit. | Set a client timeout, `max_retries` 0-1 and per-stage `max_tokens` (proposed translate ~150, rerank ~800, against measured maxima of 110 and 715 on qwen3.8-27b). These caps fit a non-reasoning model only: gpt-oss counts its reasoning inside the output (~2,280 tokens per search, estimated), so a rerank cap of 800 would cut its replies short, and a cut reply raises "truncated" and falls back (inference). Snowflake's route needs `max_completion_tokens` instead of `max_tokens`. | a few lines in app/llm_client.py (no dev-day estimate recorded) | code: app/llm_client.py:310-313, 375-376; app/main.py:85-86; CLAUDE.md:1833-1835; https://docs.snowflake.com/user-guide/snowflake-cortex/open_ai_sdk; pages/ai_search.py:148; ui_common.py:998; pages/shopping_assistant.py:97; https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits (ms.date 2026-10-30); SCR/gtm/research/gap_1.json (recommendations) |

The seed matters less than it looks. It does not make the model repeatable: on Nov 2 the same seeded translator request gave byte-identical replies on 15 of 45 pairs with gpt-oss-20b and about 9 in 10 with qwen3.8-27b, and Databricks accepts no seed at all [measured: SCR/nov02/REPORT.md:11-48]. Repeatable pages come from the LLM reply memo, which stores the first accepted reply and replays it [measured: .claude/rules/eval.md:500-509].

### 5.7 Model availability

| model | Databricks pay-per-token | Databricks provisioned throughput (PT, dedicated capacity) | other managed hosts | self-host | source |
|---|---|---|---|---|---|
| Qwen3-Next 80B A3B Instruct (our Databricks model) | Yes; "Public Preview" on all three clouds' model pages, "(Beta)" in the limits table, max 10,000 output tokens | Priced (104.762 DBU/h per 50 model units, ~$5,353 per always-on block-month at $0.07/DBU), but not listed under PT in any region of the Azure, AWS or GCP region tables | Bedrock (priced; OpenAI route unconfirmed); Snowflake only as the AI_COMPLETE SQL function (Preview); not in Vertex's managed list | Apache-2.0, 80B total / 3B active; the model card's vLLM example splits it across 4 GPUs (`--tensor-parallel-size 4`) | https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/supported-models (ms.date 2026-10-30); https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability (ms.date 2026-10-30); https://www.databricks.com/product/pricing/foundation-model-serving; https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct; SCR/gtm/research/gap_4.json G4-08, G4-09 |
| gpt-oss-120b | Yes, no preview label | Yes (71.429 DBU/h per 50 model units, ~$3,650 per block-month); listed in US and EU rows | Azure AI Foundry, Bedrock, Vertex (managed), Groq (Production) | Apache-2.0; fits one 80 GB GPU | same pages; https://huggingface.co/openai/gpt-oss-120b; SCR/gtm/research/gap_1.json G1-15 |
| gpt-oss-20b | Yes | Yes (53.571 DBU/h per 50 model units) | Bedrock (Sydney price shown), Groq (Production); Azure East US lists only fine-tuned versions | runs within 16 GB | same; SCR/gtm/research/llm.verify.json LLM-29 |
| Qwen3.5 122B A10B (Public Preview) | Listed in EU rows on all three clouds | PT preview on AWS EU | - | - | SCR/gtm/research/gap_4.json G4-12 (never evaluated by CAS) |

What this means:
- gpt-oss-120b is the most widely hosted comparable model: Databricks (including PT), Azure, Bedrock, Vertex, Groq and self-hosting [documented: SCR/gtm/research/llm.json hosting table]. It is a reasoning model: on Databricks it wrote about 3.3x the output tokens of Qwen3-Next and took 30.3 s per query against 10.4 s, at a tied quality score on a saturated query set [measured: CLAUDE.md:777-780, CLAUDE.md:1729-1730].
- gpt-oss-20b is cheaper but was an unreliable reranker on Groq: its rerank fell back on 5 of 16 context-set queries (Oct 14) and on 5 of 9 live searches (Oct 15) [measured: .claude/rules/eval.md:179-181].
- "Priority pay-per-token", Databricks' option for extra capacity without a dedicated endpoint, does not list Qwen3-Next or gpt-oss [documented: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/priority-mode, ms.date 2026-10-26; SCR/gtm/research/gap_1.json G1-20].
- Every recent app-path quality, token and latency number (Oct 20, Oct 30, Nov 2) is on Groq's qwen3.8-27b preview model with the local embedding model, not on the demo model gpt-oss-120b or on Databricks; the Oct 30 prompt A/B and the Nov 2 seed test used gpt-oss-20b [SCR/gtm/research/llm.verify.json LLM-40]. Quality must be re-measured on whichever model a client's region allows.

### 5.8 Throughput caps

Databricks publishes three pay-per-token limits for Qwen3-Next and gpt-oss on the Enterprise workspace tier: 1,000,000 input tokens per minute (ITPM), 100,000 output tokens per minute (OTPM) and 360,000 queries per hour (QPH). Separately, all Foundation Model endpoints in a workspace share 200 queries per second (QPS), and the GTE Large embedding model allows 540,000 QPH. Databricks recommends provisioned throughput for "sustained usage patterns" [documented: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits, ms.date 2026-10-30].

| limit (Databricks, Qwen3-Next pay-per-token) | per new search | ceiling, new searches | kind | source |
|---|---|---|---|---|
| OTPM 100,000 | 689 output tokens (today's prompts) | ~145 per minute = ~2.42 per second (binding) | estimate | limits page; SCR/gtm/research/gap_1.json G1-14 |
| OTPM 100,000 | 658 output tokens (Databricks-measured, older prompts) | ~152 per minute = ~2.53 per second | estimate | limits page; SCR/gtm/research/scalability.json SCA-13 |
| OTPM 100,000, API default (cross-sell on) | 737 output tokens | ~136 per minute = ~2.26 per second | estimate | SCR/gtm/research/gap_1.json G1-14 |
| ITPM 1,000,000 | 4,213 input tokens | ~237 per minute = ~3.96 per second | estimate | limits page |
| QPH 360,000 | 2 calls | ~3,000 per minute = 50 per second | estimate | limits page |
| Workspace FM QPS 200 (including embedding calls: our reading) | 3 calls (UI) / 9 (API default) | ~66.7 / ~22.2 per second | estimate; inference on what counts | limits page; SCR/gtm/research/gap_1.json G1-31 |
| GTE Large QPH 540,000 (the query-embedding model) | 1 embedding call (UI) / 6 (API default); memo hits still embed | 150 / 25 per second, whatever the memo hit rate | estimate | limits page; SCR/gtm/research/gap_1.json G1-31 |
| OTPM 100,000 on gpt-oss-120b | ~2,280 output tokens (reasoning, est.) | ~44 per minute = ~0.73 per second | estimate | limits page; SCR/gtm/research/llm.verify.json LLM-24 |
| OTPM 100,000, rerank returning scores only | ~246 output tokens (est.) | ~6.8 per second | estimate (lever, untested) | SCR/gtm/research/gap_1.json G1-32 |

Reconciling the figures. The ~145 per minute and the ~2.4-2.5 per second quoted elsewhere are the same ceiling: 145 per minute equals 2.42 per second on today's prompts, and the scalability lane's 2.5 per second used the older, shorter Databricks token count. Four assumptions sit under them:
1. They count only searches that reach the LLM. Memo and page-cache hits use no LLM tokens; a memo hit still makes 1 embedding call and ~3.2 Vector Search queries, and only a page-cache hit skips everything [code: SCR/gtm/research/gap_1.json G1-05]. 10 searches per second would need more than 75% of them answered from cache [estimate: SCR/gtm/research/gap_1.json G1-14].
2. They are upper bounds. The app sends no `max_tokens`, so Databricks reserves an undocumented default output amount per request and can refuse with a 429 (rate-limit error) before the arithmetic says it should [documented: limits page; SCR/gtm/research/llm.verify.json LLM-24].
3. The page states the limits per pay-per-token endpoint, for the Enterprise workspace tier, and says they vary by tier. Every app in a workspace that calls the same model shares that endpoint (inference). Databricks' pricing pages say Azure Premium corresponds to the Enterprise tier, but the limits page does not say its figures apply to Azure Premium (so: inference) [SCR/gtm/research/gap_1.json G1-34], and LatentView's workspace tier is not recorded [SCR/gtm/research/scalability.verify.json SCA-13].
4. The app process itself holds one thread per uncached search for its whole duration; with FastAPI's 40-thread pool and ~10.4 s per search that is about 3.85 uncached searches per second per process, and with no client timeout a stalled provider pins threads (estimate) [SCR/gtm/research/scalability.json SCA-07]. As shipped, each App instance runs one such process [code: databricks/app_bundle/run_app.py:91].

Other hosts publish higher defaults (estimates from their documented limits and our token profile):

| host and model | published default limit | ceiling, new searches per minute | source |
|---|---|---|---|
| Azure AI Foundry gpt-oss-120b | 5M tokens per minute (TPM) / 5K requests per minute (RPM) | ~770 | https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quotas-limits (ms.date 2026-08-20) |
| Azure OpenAI gpt-4.1-mini, Tier 1 Global | 5M TPM / 5K RPM | ~1,020 | same |
| Azure OpenAI gpt-4.1-mini, Tier 0 Global (low-tier subscriptions) | 200K TPM / 200 RPM | ~41 | same; SCR/gtm/research/llm.verify.json LLM-30 |
| Snowflake Cortex REST gpt-5-mini | 2M TPM / 2K RPM per account per model | ~408 at most (a reasoning model writes more output than our profile) | https://docs.snowflake.com/user-guide/snowflake-cortex/open_ai_sdk |
| Snowflake Cortex REST claude-haiku-4-5 | 5M TPM / 10K RPM | ~902 | same |
| Groq qwen3.8-27b, Developer-plan row | 250K TPM / 1K RPM | ~51 | https://console.groq.com/docs/models |
| Groq, the limits the project runs under (free tier as used; the page does not settle which plan the row is) | 8K TPM, 200K tokens per day per model | ~1.6 (about 40 a day) | https://console.groq.com/docs/rate-limits (no date shown, fetched 2026-11-02); SCR/gtm/research/scalability.verify.json SCA-19 |
| AWS Bedrock | quotas not fetched | unknown | open question |

Provisioned throughput removes the per-minute token limits ("No TPM restrictions"), allows up to 200 QPS per workspace, and the page says its dedicated resources "ensure consistent latency" [documented: limits page]. Three things stop us pricing it for CAS. Qwen3-Next is not listed under PT in any region (section 5.7). Tokens per second per model unit are not documented for our models; the only worked example is Llama 4 Maverick [documented: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/model-units, ms.date 2026-09-23]. And a block only beats pay-per-token when it is busy: one always-on block costs the same as pay-per-token tokens at about 0.69 sustained searches per second on gpt-oss-120b, and a block sized for a peak but running at a third of it costs about 2.7x pay-per-token (estimates) [SCR/gtm/research/gap_1.json G1-18, G1-35]. A workspace load test, or the account team's model-unit sizes, would settle this; neither exists yet.

### 5.9 Latency

| setup | what | value | kind | source |
|---|---|---|---|---|
| Groq qwen3.8-27b, local embeddings, 31 held-out queries (Nov 2) | translate call | median 429 ms, p90 641 ms | measured | SCR/gtm/research/latency_from_nov02.txt |
| same | rerank call (15 rows) | median 1,734 ms, p90 1,992 ms | measured | same |
| same | retrieval stage (local, 300K rows) | median 847 ms, p90 2,365 ms | measured | same |
| same | page 1 of a new search, rate-limit pacing removed | median 3,611 ms, p90 4,635 ms, max 9,433 ms | measured | same |
| same | page 1 including the harness's pacing for the account's token limit (and 28 rate-limit errors) | median 60,159 ms | measured (an account limit, not model speed) | same; SCR/gtm/research/llm.verify.json LLM-46 |
| same | page 2 (from cache) | median 2 ms | measured | SCR/gtm/research/llm.json (latency table) |
| Databricks Qwen3-Next + Vector Search (eval harness, not the App) | per query, rerank on / off | 10.2 s / 4.9 s (Oct 12); 10.4 s (Sep 23) | measured | CLAUDE.md:777-780; CLAUDE.md:860-864 |
| Databricks gpt-oss-120b (Sep 23) | per query, rerank on | 30.3 s | measured | CLAUDE.md:1730 |
| Databricks claude-haiku-4-5 (Sep 23) | per query, rerank on | 9.7 s | measured | CLAUDE.md:1728 |
| Groq gpt-oss-120b (the local demo model) | per query | ~20-25 s | measured (undated note; no run recorded) | CLAUDE.md:794 |
| LLM memo, local SQLite | hit | 0.85 ms | measured (bench; the App's Volume store is unmeasured) | CLAUDE.md:614; SCR/gtm/research/llm.verify.json LLM-52 |

How to read this. The rerank is the slow call because it writes about 620 output tokens, a reason sentence for each of 15 rows, against about 66 for translate; its latency is about 4x translate's [measured: SCR/gtm/research/llm.verify.json LLM-48]. On Databricks pay-per-token, rerank added about 5.3 s per query, a difference of two run means [inference: CLAUDE.md:860-864]. That is consistent with serving speed being the biggest lever, but model, prompt and harness all differ between the Groq and Databricks runs, so it is not proof [SCR/gtm/research/llm.verify.json LLM-48]. The Databricks App's own p50 and p95 latency, with the memo on and off, have never been measured [CLAUDE.md:1168]. The steps run in strict order, because retrieval needs the translator's phrases [code: app/main.py:1928-2022].

For comparison, one documented case: Instacart reached a 300 ms target for its own query-understanding model only after merging adapter weights into the model and moving to H100 GPUs (from about 700 ms on A100), and it serves frequent queries from cache [documented: https://company.instacart.com/tech-innovation/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms, Nov 21, 2025]. That is one model call at Instacart, not a whole results page. No CAS search that uses the LLM has been measured below about 3 s; the fastest was 3.3 s (Groq, local, pacing removed) [measured: SCR/gtm/research/latency_from_nov02.txt].

| lever | expected effect | kind | status | source |
|---|---|---|---|---|
| Cache: page cache + LLM memo + single flight | page turns, filters and repeats skip both calls (page 2 median 2 ms) | measured | built, on by default; memo latency on the App's Volume store unmeasured | SCR/nov02/REPORT.md:56; SCR/gtm/research/llm.verify.json LLM-52 |
| Warm the memo nightly with the client's top queries | frequent queries answered with no LLM wait | inference | not built | SCR/gtm/research/llm.json (recommendations) |
| A non-reasoning model | Qwen3-Next 10.4 s vs gpt-oss-120b 30.3 s on Databricks | measured | done on Databricks; the local demo still runs gpt-oss-120b | CLAUDE.md:777-780; CLAUDE.md:794 |
| Faster serving (another host or PT) | Qwen-family rerank ~1.7 s on Groq vs ~5.3 s added on Databricks | measured, confounded | Qwen3-Next not listed for PT | SCR/gtm/research/llm.verify.json LLM-48 |
| Rerank returns scores without reasons | most of ~620 output tokens removed; reasons show only in Developer details | estimate | needs a quality A/B | app/reranker.py:104; SCR/gtm/research/gap_1.json G1-32 |
| Skip rerank for frequent or confident queries | ~5.3 s saved on Databricks | inference; quality cost unmeasured | not built | CLAUDE.md:860-864 |
| Small or distilled model for translate | Instacart's pattern; translate writes ~66 tokens | documented (Instacart) | not started | docs/RESEARCH.md:110-112; Instacart post above |
| Bound the tail: timeout, retries, `max_tokens` | stops 600 s stalls and OTPM reservations | code | gap 3 above | app/llm_client.py:310-313 |

### 5.10 EU regions (H&M, C&A)

The Databricks model-region pages for Azure, AWS and GCP (all dated 2026-10-30, fetched 2026-11-02) give this picture [documented: https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability; https://docs.databricks.com/aws/en/unity-gateway/model-region-availability; https://docs.databricks.com/gcp/en/unity-gateway/model-region-availability; SCR/gtm/research/gap_4.json G4-01 to G4-06]:

| cloud | EU / UK regions with FM APIs | Qwen3-Next (pay-per-token) | gpt-oss-120b / 20b and gte-large-en | notes |
|---|---|---|---|---|
| Azure | francecentral, germanywestcentral, northeurope, westeurope, swedencentral, switzerlandnorth, uksouth | Not listed in any of them, with or without the cross-geography mark. Listed only in 8 US regions (unmarked) and 8 Asia-Pacific / Americas regions (marked "requires cross geography routing") | Listed pay-per-token and PT, unmarked, in the six EU-boundary regions; marked in uksouth | norwayeast, switzerlandwest and ukwest: "Not supported" |
| AWS | eu-central-1, eu-west-1, eu-west-2, eu-west-3 | Listed, unmarked; no PT | Listed pay-per-token and PT, unmarked | eu-north-1 (Stockholm) is not a row |
| GCP | europe-west2, europe-west3 | Listed, unmarked; no PT | Listed pay-per-token; PT entries marked | GCP's Designated Services page (Oct 26, 2026) marks all Europe FM APIs as needing cross-Geo processing, which contradicts the table; europe-west1 (Belgium) "Not supported"; europe-west4 not a row |

Turning on cross-Geo processing gives up in-Geo processing (storage stays in-Geo), and it is on by default only outside the US and EU [documented: https://learn.microsoft.com/en-us/azure/databricks/resources/databricks-geos, ms.date 2026-09-11; SCR/gtm/research/gap_4.json G4-07]. On Azure EU it would not help anyway: the model is not listed there at all, with or without the mark [SCR/gtm/research/gap_4.json G4-01].

What follows (we do not know which cloud or region H&M or C&A use, or whether they use Databricks at all) [SCR/gtm/research/gap_4.json, open questions]:
- An H&M or C&A workspace on Azure EU cannot use Qwen3-Next today. Of the models CAS has run, only gpt-oss-120b and 20b are listed there; the EU rows also list Qwen3.5 122B (preview), Llama and Gemma models, which CAS has never run [documented: SCR/gtm/research/gap_4.json G4-01, G4-03, G4-12].
- The switch itself is one setting (`OPENAI_MODEL=databricks-gpt-oss-120b`) [code: CLAUDE.md:790; databricks/app_bundle/app.yaml:39]. The evidence does not carry over: gpt-oss-120b took 30.3 s against 10.4 s per query on Databricks (quality tied), costs about $2.40 per 1,000 new searches at West Europe list prices, and caps near 44 new searches a minute on pay-per-token (estimates; see 5.3 and 5.8) [measured: CLAUDE.md:777-780]. The acceptance (08), page eval (10) and determinism (12) notebooks would need re-running on it [CLAUDE.md:1663; SCR/gtm/research/gap_4.json G4-11]. gpt-oss-20b is not a safe fallback: its reranks failed often on Groq (5.7).
- On Databricks on AWS EU, Qwen3-Next is listed in-region on pay-per-token (no PT) [SCR/gtm/research/gap_4.json G4-04]. Separately, AWS's own Bedrock service sells it in Ireland and Milan at $0.18 / $1.41 per 1M tokens, about $1.73 per 1,000 new searches (estimate); Bedrock's quotas and its OpenAI-route support for this model are unchecked [https://aws.amazon.com/bedrock/pricing/, fetched 2026-11-02; SCR/gtm/research/gap_1.json G1-21].
- Qwen3.5 122B A10B (Public Preview) is listed in EU rows on all three clouds and is the only same-family candidate; CAS has never tested it [SCR/gtm/research/gap_4.json G4-12].
- Language is a separate limit. All 10 prompts are English and name no output language, and both embedding models are English models; how Qwen3-Next or gpt-oss would answer a German or Swedish query is unmeasured. CAS should be described as working on English catalogs and queries today [code and documented: SCR/gtm/research/gap_4.json G4-20, G4-22, G4-25].

### 5.11 Private connectivity

On Databricks, private connectivity (such as Azure Private Link) "is only supported for model serving endpoints that use provisioned throughput or endpoints that serve custom models" [documented: https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/compliance, ms.date 2026-10-28]. The same page recommends PT for workloads that need compliance certifications beyond HIPAA [documented: same page; SCR/gtm/research/llm.verify.json, missing items]. Qwen3-Next is not listed under PT in any region, so a client that requires Private Link to the LLM cannot use it on Databricks today (inference from those two documented facts) [SCR/gtm/research/gap_4.json G4-08]. Of the PT families the EU rows list, the ones closest to CAS are GPT OSS 120B / 20B (language model) and GTE v1.5 (English) (embeddings), plus Qwen3.5 122B (preview) on AWS EU; the rows also list Llama and Gemma families, which CAS has not run [documented: SCR/gtm/research/gap_4.json G4-03, G4-08]. Whether the per-search query embedding on pay-per-token gte-large-en also falls under a client's private-connectivity rule is an open question [SCR/gtm/research/gap_4.json, recommendations].

### 5.12 Data-use terms per host

| host | what the vendor documents | source |
|---|---|---|
| Databricks FM API | Data may be processed outside the originating region but not outside its geography (non-US/EU workspaces need cross-Geo processing). The fetched pages contain no statement on whether prompts or outputs train models: ask the account team in writing. | https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/compliance (ms.date 2026-10-28); https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/ (Oct 30, 2026) |
| Azure OpenAI / Foundry | Prompts and completions are not available to OpenAI or other model providers and are not used to train foundation models without permission. Global deployments may process in any geography; DataZone stays in the US or EU zone. Abuse monitoring may store flagged prompts for human review in-geo unless modified abuse monitoring is approved. | https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/openai/data-privacy (ms.date 2026-05-18); SCR/gtm/research/llm.verify.json LLM-30 |
| AWS Bedrock | Model providers have no access to Bedrock logs, prompts or completions; a zero-data-retention mode can be set per Region or project. | https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html; https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html (fetched 2026-11-02) |
| Google Vertex AI | No training or fine-tuning on customer data without permission; the global endpoint may serve from any region, so a residency-bound client must use a regional endpoint. | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/data-governance (no date shown, fetched 2026-11-02); SCR/gtm/research/cloud_ops.verify.json |
| Gemini Developer API | Paid tier: "Used to improve our products": No. | https://ai.google.dev/gemini-api/docs/pricing (fetched 2026-11-02) |
| Snowflake Cortex | Customer data is never used to train models offered to other customers; models run inside Snowflake's perimeter unless the customer elects otherwise. | https://docs.snowflake.com/en/guides-overview-ai-features (fetched 2026-11-02) |
| Self-hosted vLLM | Data stays in the client's account (no third party). | inference from the deployment model |
| Groq | Not reviewed. Development host only. | SCR/gtm/research/llm.json (hosting table) |

CAS itself also stores data: the LLM memo keeps accepted replies, which can echo shopper words, in a store the deployment owns (on Databricks, a Unity Catalog volume in the client's workspace) [code and documented: .claude/rules/safety.md, "Stored LLM replies"].

### Open questions for this area

- Which cloud and region does each target run Databricks in, is cross-Geo processing allowed, and is Private Link to model endpoints required? These decide whether Qwen3-Next is usable at all.
- Databricks account team, in writing: is Qwen3-Next reachable from an Azure EU workspace; is PT for it planned; what tokens per second does a model unit deliver for a ~4.2K-in / ~0.7K-out request on gpt-oss-120b; what $/DBU applies in the client's region and tier; and is there a no-training commitment for FM API prompts and outputs?
- GCP Europe: the region table lists the models unmarked in europe-west2/3, while the Designated Services page says Europe FM APIs need cross-Geo processing. Which governs?
- Which workspace tier would a client use? The published limits are for the Enterprise tier, and LatentView's own tier is not recorded.
- How large is Databricks' default `max_tokens` reservation, and so what is the real admitted ceiling below ~145 new searches a minute?
- Do Bedrock, Snowflake REST, vLLM and Vertex accept `seed` and `response_format: json_object`? Is Qwen3 Next served on Bedrock's OpenAI-compatible route, and what are Bedrock's quotas? Vertex's price for gpt-oss 120B?
- Quality, tokens and latency on the model a client's region allows (gpt-oss-120b; possibly Qwen3.5 122B) and on Databricks with today's prompts: none has been measured. The App's p50 / p95 latency with the memo on and off is also unmeasured.
- What share of a target's searches are exact repeats? This sizes memo savings and needs a client query log.
- Does a rerank without per-row reasons keep quality? It would cut cost by an estimated 36% and raise the pay-per-token ceiling about 2.8x.
- Assistant (chat) tokens per turn are unmeasured.

## 6. Scalability: catalog size, traffic and cost

Answer. CAS today is a single-process demo. Its memory and CPU costs were measured only offline, on one laptop, with 299,972 parent products. Its speed was measured only one query at a time (about 10 s per new search in Databricks evaluation runs). Nothing has been measured on Databricks under load, and no load test exists. By our estimates (laptop extrapolations), as built one API worker needs ~1.7 GB at 300K rows, which fits the default Medium Databricks App, and ~5.2 GB at 1M rows, which needs a Large App. At about 5M rows only one worker fits the largest App size, so from about 5M rows the design has to change; at 20M the App cannot start. Traffic hits outside limits before CPU does. Databricks pay-per-token Qwen3-Next allows at most roughly 2.4 new searches per second on a workspace's endpoint (estimate, Enterprise-tier limits). No Azure EU region lists Qwen3-Next, and the in-region gpt-oss-120b allows at most ~0.7 per second (estimate). So 10 or 50 searches per second need high cache-hit rates or LLM capacity bought outside pay-per-token. Cost is a fixed floor of about $551 a month plus about $1.46 per 1,000 new searches (East US list prices, Qwen3-Next pay-per-token, demo UI path). That is list-price arithmetic, not a bill.

### How to read this section

Each fact carries a label. Measured: we ran it and timed it. Code: read from the repo. Documented: a vendor page says so. Estimate: arithmetic on the above. Inference: our reading. SCR means the 2026-11-02 session archive (see "How to read this").

- Measured: offline probes on 2026-11-02 on the dev laptop (Intel Core Ultra 7 165H, 64 GB RAM, Python 3.13.7, pandas 3.0.3), on the 299,972-row enriched catalog, with no LLM and no network [SCR/gtm/research/probe_scalability.json]. A second reviewer re-ran part of it and reproduced the raw catalog read exactly (266.7 MB) [SCR/gtm/research/scalability.verify.json, SCA-02].
- Not measured: Databricks at any load; more than 300K real rows (synthetic vector tests at 1M x 384 and 300K x 1024 are the only exceptions); more than one user at a time. No load test exists anywhere in the repo. Every memory-at-scale, searches-per-second and dollar figure below is an estimate.
- What a row is (measured): one row is one parent product with a single colour field, not one row per size or colour SKU. The 299,972 rows are 299,972 distinct parent product ids [SCR/gtm/research/gap_2.json G2-21; scripts/load_amazon_data.py:228-242]. A client feed with one row per variant would multiply the row count.

### 6.1 Catalog size

How the app holds a catalog (code). Every API process loads the whole catalog into memory as one pandas table (pandas is Python's table library) [app/search.py:348 for the laptop; app/vector_search.py:106-141 for the Databricks App]. At start-up it also builds its own keyword index on that table [app/main.py:192-193]. That index uses BM25, the standard keyword-ranking formula. On the laptop path the process also holds the embedding matrix (the numeric vectors behind semantic search). Each search result is filled in ("hydrated") from the in-memory table by row position, on the laptop and on Databricks alike. Databricks AI Search (the new name of Vector Search) returns only a row key and a score [app/search.py:519; app/vector_search.py:258-260]. So catalog size sets each process's memory and start-up time.

The Databricks App runs one API worker next to the Streamlit demo UI, in one container [databricks/app_bundle/run_app.py:88-96]. Uvicorn, the web server, starts extra workers as fresh processes [https://uvicorn.dev/deployment/, fetched 2026-11-02]. So each extra worker repeats the start-up and holds its own copy of everything (inference from app/main.py:163-195).

Measured at 300K rows:

| What | Measured value | Note | Source |
|---|---|---|---|
| Product table in memory, laptop path | 594.6 MB = 2,078 bytes per row (40 columns) | four long-text columns are 56% of it | SCR/gtm/research/probe_scalability.json |
| Product table, Databricks App path | ~1,429 bytes per row (estimate) | the App's export drops the search_text column | same; databricks/09_deploy_app.py:117-122 |
| Keyword (BM25) index | 1,006 MB (~3.1-3.5 KB per row); built in 36.0 s; query median 320 ms, p95 445 ms | query time grows with row count: a Python loop over every product | same; app/keyword_search.py:113-129 |
| Hydration per search (300 hits) | 140.8 ms median (0.306 ms per row) | 300 hits assumes 5 intents x 60 results; a UI search averages ~3.2 intents. Grows with hits and columns, not with catalog size | same; SCR/gtm/research/scalability.verify.json SCA-06, SCA-21 |
| Laptop vector scan, per search intent | 460 ms median (reviewer's re-run: 368 ms) | sklearn copies and re-normalises the whole matrix on every call | same; app/embeddings.py:117-121 |
| Same scan, matrix normalised once, partial sort | 12.1 ms (reviewer: 10.4 ms) | near-identical results: top 60, mean overlap 0.999, minimum 0.983 over 20 queries | SCR/gtm/research/scalability.verify.json SCA-04 |
| Peak memory while building the table, laptop path | 1,998 MB (722 MB once built) | includes building search_text, which the App skips; start-up peaks are not in the per-row figures | SCR/gtm/research/probe_scalability.json |
| One cached result pool | ~0.74 MB per 180-row entry; 64 entries ~45 MB (estimate) | pools are typically ~150-180 rows but can reach ~1,440 (24 intents x 60); cache memory is small; how long entries survive is the problem (6.2) | same; SCR/gtm/research/scalability.verify.json SCA-21 |

Threads do not speed up the Python parts (measured). Standard Python (CPython) has a global lock, the GIL, that lets only one thread run Python code at a time. With 4 threads doing the same total work, hydration ran at 0.89x (no gain), the BM25 search (100K rows) at 1.15x, and a numpy matrix product at 2.47x [SCR/gtm/research/probe_scalability.json, concurrency]. So inside one process, hydration and keyword search run one at a time. Only numeric-library work and network waits overlap. The App's Python (3.11 per CLAUDE.md, not verified) also has a GIL [SCR/gtm/research/scalability.verify.json SCA-06].

The laptop vector scan is slow for a fixable reason (measured). Almost all of its 460 ms is a per-call copy and re-normalisation of the 300K x 384 matrix, although the stored vectors are already unit length [SCR/gtm/research/probe_scalability.json, cosine_decompose]. Normalising once at load and taking a partial sort cuts it to 10-12 ms per intent, about 35-38x faster. It also removes a ~0.44 GB temporary copy per call in flight [SCR/gtm/research/scalability.verify.json SCA-04, SCA-18]. On a synthetic 1M x 384 matrix the fixed method took 37 ms per intent on 22 threads and 97 ms on 2 threads [SCR/gtm/research/probe_scalability.json, syn1m384]. This matters only for laptop demos. On Databricks the vector scan runs inside AI Search.

Estimated memory and start-up time for larger catalogs. These are linear extrapolations of the 300K measurements, on laptop CPU. They exclude start-up peaks (CSV read, annotation, BM25 build) and Streamlit's own memory [SCR/gtm/research/scalability.verify.json SCA-16]. On the App path each API worker needs ~5.0 KB per row (1.43 KB table, ~0.07 KB text hash, 3.5 KB BM25) plus ~0.2 GB. BM25 is about 70% of that [SCR/gtm/research/scalability.json SCA-16].

| Catalog rows | RAM per API worker, App path (est.) | RAM per process, laptop path (est.) | Start-up per worker, App path (est.) | One BM25 query (est.) | Databricks App size that fits (est.) |
|---|---|---|---|---|---|
| 300K | ~1.7 GB | ~2.3 GB | ~61 s | 320 ms (measured) | Medium (6 GB, 2 vCPU; the default size): 1 worker; a 2nd worker is unproven once start-up peaks are counted |
| 1M | ~5.2 GB | ~7.3 GB | ~3.4 min | ~1.1 s | Large (12 GB, 4 vCPU); Medium no longer fits next to Streamlit |
| 5M | ~25 GB | ~36 GB | ~17 min | ~5.3 s | XLarge (48 GB, 12 vCPU), one worker only |
| 20M | ~100 GB | ~143 GB | ~68 min | ~21 s | none: the App cannot start |

Sources: SCR/gtm/research/probe_scalability.json (derived.per_catalog_size); App sizes from https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/compute-size (ms.date 2026-09-25): Medium up to 2 vCPUs, 6 GB, 0.5 DBU per hour; Large 4 vCPUs, 12 GB, 1 DBU; XLarge 12 vCPUs, 48 GB, 3 DBU. A DBU (Databricks Unit) is Databricks' billing unit.

Start-up time (measured parts, estimated totals). At 300K the steps took: CSV read 6.8 s, product typing 6.2 s, catalog taxonomy 12.5 s (the App derives it at every start) and BM25 build 36.0 s [SCR/gtm/research/probe_scalability.json]. Step times varied 30-50% in one re-run (CSV read 4.5 s against 6.8 s). They exclude the App's download of the ~249 MB catalog file and its start-up checks [SCR/gtm/research/scalability.verify.json SCA-17]. Every worker and every extra App instance pays this, so restarts and deploys get slower as the catalog grows. Separately, a catalog refresh is a full rebuild today: the Databricks index build was measured at 47.8 min for 299,972 rows on 2026-10-12, plus a redeploy [CLAUDE.md:1541-1545; SCR/oct27/byo/REPORT.md:238].

AI Search at scale (documented). A standard endpoint "unit" holds 2M vectors of 768 dimensions "or the equivalent", so by proportion ~1.5M of CAS's 1024-dimension vectors (inference) [https://learn.microsoft.com/en-us/azure/databricks/ai-search/cost-management, ms.date 2026-10-20]. Past one unit, throughput "plateaus at approximately 30 QPS" (queries per second) for approximate nearest-neighbour (ANN) search. Storage-optimized endpoints hold far more, but answer in 300-500 ms at 30-50 QPS [https://learn.microsoft.com/en-us/azure/databricks/ai-search/best-practices, ms.date 2026-10-29]. So a 5M-row catalog (about 4 units) also lowers AI Search throughput (inference).

Where the target catalogs likely fall. No target retailer publishes a total online SKU count in any source we fetched [SCR/gtm/research/gap_2.json G2-05, G2-09, G2-13, G2-16, G2-19]. The public anchors are:

| Case | Size and basis | Band | Fits today's design? | Kind | Source |
|---|---|---|---|---|---|
| CAS demo | 299,972 parent products | 300K | yes: ~1.7 GB per worker, one Medium App (est.) | rows measured, RAM estimated | SCR/gtm/research/gap_2.json G2-21; SCR/gtm/research/scalability.json SCA-16 |
| One grocery store or fulfilment centre, online | 15K-50K items: FMI average US supermarket 33,248 (2025); Kroger fulfilment centre 35,000 (Oct 12, 2021); Central Market 15,000+ (Jun 1, 2021); heb.com ship-to-home 50,000 (Nov 12, 2015) | well under 300K | yes, 6-20x headroom (estimate) | documented inputs | https://fmi.org/our-research/supermarket-facts (fetched 2026-11-02); https://ycharts.com/news/story/PRN-CL36718-20211012 (fetched 2026-11-02); https://www.supermarketnews.com/retailers/central-market-expands-curbside-home-delivery (fetched 2026-11-02); https://ktxs.com/news/texas/50000-products-now-available-through-new-h-e-b-online-store (fetched 2026-11-02) |
| Grocery chain, all banners in one catalog | not published; 10x the FMI store (~332K) is an assumed illustration | 300K-1M | marginal on Medium; a Large App or thin workers | estimate | SCR/gtm/research/gap_2.json G2-22 |
| One fashion market, one row per style-colour | not published for H&M or C&A | probably under 300K | likely yes | inference | SCR/gtm/research/gap_2.json G2-16, G2-23 |
| One fashion market, one row per SKU | only anchor: C&A Brazil, 150,000 SKUs (2020; a different business, marketplace included) | under 300K at that size | yes at that size | documented anchor | https://www.casestudies.com/company/vtex/case-study/ca-fashion-retail-ecommerce-at-its-finest (undated; data Jul-Sep 2020; fetched 2026-11-02) |
| All H&M online markets in one catalog | 150K x 62 markets = 9.3M (the per-market size is assumed) | 5M-20M | no: needs the re-architecture in 6.4 | estimate | SCR/gtm/research/gap_2.json G2-23; https://hmgroup.com/?p=311 (figures as of 31 Aug 2026) |

Two limits shape this more than row count. First, for fashion the multiplier is markets. CAS holds one currency per catalog [SCR/oct27/byo/REPORT.md:460], multilingual search is out of scope [CLAUDE.md:1332], and every query reader, title rule, keyword tokenizer and embedding model in CAS is English-only: a German or Swedish catalog loses most of the context logic and some parts break outright, and no non-English evaluation exists [SCR/gtm/research/gap_4.json G4-15 to G4-25]. So the working shape today is one English catalog per market (inference); one index across 62 markets needs the thin-worker design [SCR/gtm/research/gap_2.json G2-23]. Second, for grocery the harder limit is per-store availability. H-E-B updates per-store assortment as shelf locations change, and Kroger's engine updates its index every 5 minutes per a vendor case study (since 2021) [SCR/gtm/research/gap_2.json G2-11, G2-02]. CAS has no store or inventory model (on hold) [CLAUDE.md:1333-1334]. That availability should stay in the client's own engine (inference) [SCR/gtm/research/gap_2.json G2-22].

### 6.2 Traffic

What one action costs in outside calls (code). The shipped UI turns cross-sell off. An API caller of /search that does not send recommend gets the default, true, which adds a cross-sell LLM call and up to five more searches (six on fallback) on page 1 [pages/ai_search.py:120; app/config.py:162; app/main.py:2070-2072; SCR/gtm/research/gap_1.json G1-02]. /chat and /unified_search always send recommend=false [SCR/gtm/research/gap_1.json G1-03].

| Action | LLM calls | Embedding calls | AI Search queries | Tokens in / out | Source |
|---|---|---|---|---|---|
| New search from the shipped UI | 2 (translate, rerank) | 1 | ~3.2 (one per intent) | 4,213 / 689 | app/main.py:1955-1980; app/vector_search.py:240-250; SCR/gtm/research/gap_1.json G1-01, G1-06 |
| New search through the API with defaults | 3 | 6 (7 on fallback) | ~8.2 (9.2) | 4,451 / 737 | app/recommendations.py:134-138, 219; SCR/gtm/research/gap_1.json G1-02 |
| Product page open | 1 | 5 (6) | 5 (6), one after another | 238 / 48 | app/main.py:2537-2568; SCR/gtm/research/gap_1.json G1-04, G1-06 |
| Page 2, filter, sort, or a repeat while still cached | 0 | 0 | 0 | 0 | SCR/gtm/research/gap_1.json G1-03; SCR/nov02/REPORT.md:56 |
| Repeat after the result cache dropped it (LLM memo hit) | 0 | 1 | ~3.2 | 0 | SCR/gtm/research/gap_1.json G1-05 |

The search token counts and the ~3.2 intents were measured on 2026-11-02 on 31 held-out queries with Groq qwen3.8-27b, which is a different model and tokenizer from Databricks' Qwen3-Next; the product-page counts come from 6 cross-sell calls on 2026-10-30 on the same model [SCR/gtm/research/gap_1.json G1-01, G1-06]. The only Qwen3-Next figure on Databricks is 2,709 in / 658 out per search, from 2026-09-23, on older and shorter prompts [CLAUDE.md:1729; SCR/gtm/research/scalability.verify.json SCA-13]. A memo hit skips the LLM but still runs retrieval; only a result-cache (page_cache) hit skips everything.

Time per new search (measured, one query at a time). On Databricks a new search took about 10 s in evaluation runs on the keyword query set (the eval harness, not the App): 10.4 s on 2026-09-23 and 10.2 s with rerank on, on 2026-10-12 [CLAUDE.md:1729; CLAUDE.md:863]. On the laptop (local gte-small retrieval, Groq qwen3.8-27b, memo off, 31 queries) page 1 took p50 3.6 s, p90 4.6 s, with the test harness's own rate-limit pauses removed; the fastest was 3.3 s [SCR/gtm/research/latency_from_nov02.txt]. No AI search path has been measured under ~3 s [SCR/gtm/research/claims.json CLA-46].

The ceilings on new (uncached) searches. All are estimates from documented limits and the call counts above. OTPM and ITPM mean output and input tokens per minute; QPH means queries per hour. The limits page defines the token limits per endpoint; the figures below are for a workspace's one Qwen3-Next pay-per-token endpoint.

| Limit | Documented value | Ceiling on new searches | Source |
|---|---|---|---|
| Qwen3-Next pay-per-token, OTPM | 100,000 | at most ~2.4/s (UI path); ~2.3/s with API defaults | https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits (ms.date 2026-10-30); SCR/gtm/research/gap_1.json G1-14; SCR/gtm/research/llm.verify.json LLM-24 |
| Same model, ITPM | 1,000,000 | ~4.0/s | same |
| Same model, QPH | 360,000 | ~50/s (2 LLM calls each) | same |
| All model endpoints together, QPS per workspace | 200 | ~67/s on the UI path; ~22/s with API defaults | same; SCR/gtm/research/gap_1.json G1-31 (that embedding calls count toward it is an inference) |
| Embedding model (GTE Large), QPH | 540,000 (150/s) | 150/s UI; 25/s with API defaults; memo hits still embed | same |
| AI Search through the workspace query URL (how CAS calls it today) | "capped at a few tens of QPS" | ~5-10/s, assuming ~5 vector queries per search (the UI path's ~3.2 raises it; the API default's ~8.2 lowers it) | https://learn.microsoft.com/en-us/azure/databricks/ai-search/high-qps (ms.date 2026-10-27); app/vector_search.py:258-260; SCR/gtm/research/scalability.json SCA-11 |
| AI Search standard endpoint, through its own index URL | 20-200 QPS by index size (high-qps page); 30-200+ (best-practices page); ~30 past one unit | ~6-60/s per endpoint copy | https://learn.microsoft.com/en-us/azure/databricks/ai-search/high-qps (ms.date 2026-10-27); https://learn.microsoft.com/en-us/azure/databricks/ai-search/best-practices (ms.date 2026-10-29); SCR/gtm/research/gap_1.json G1-11, G1-12 |
| API worker thread pool | 40 threads | ~3.85/s per process at 10.4 s per search | https://anyio.readthedocs.io/en/stable/threads.html and https://fastapi.tiangolo.com/async/ (fetched 2026-11-02); app/main.py:2064-2065 |
| Hydration CPU under the GIL | 140.8 ms per 300 hits | at most ~7/s per process (an upper bound: other per-request Python work is unmeasured) | SCR/gtm/research/probe_scalability.json; SCR/gtm/research/scalability.verify.json missing items |
| Groq free tier, as the project uses it for laptop demos | 8,000 tokens/min, 200,000/day per model | ~1.6 searches per minute, ~39 per day | https://console.groq.com/docs/rate-limits (no date shown, fetched 2026-11-02); .claude/rules/eval.md:413-418 |

The thread-pool figure uses Little's law: requests in flight = arrival rate x time each takes, so 40 threads / 10.4 s = ~3.85 per second.

Caveats that can lower these ceilings:
- The model limits are the Enterprise-tier table. Our workspace's tier is not recorded. Azure Premium probably maps to it, per a pricing-page footnote (inference) [SCR/gtm/research/gap_1.json G1-34; SCR/gtm/research/scalability.verify.json SCA-13].
- The app sends no max_tokens. Without it, Databricks reserves a default output budget per request against OTPM at admission, and the page does not give its size. So the admitted rate is probably below ~2.4/s (inference, unmeasured) [https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits, ms.date 2026-10-30; app/llm_client.py:317-335; SCR/gtm/research/llm.verify.json LLM-24].
- Shared pay-per-token capacity can answer 503 when it runs out [https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits, ms.date 2026-10-30; SCR/gtm/research/llm.json LLM-23].
- Qwen3-Next is a Public Preview ("Beta" in the limits table) model. In 8 non-US Azure regions it is served only "based on GPU availability" with cross-geography routing turned on [SCR/gtm/research/gap_4.json G4-02, G4-09; https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability, ms.date 2026-10-30].
- The AI Search figures are for 768-dimension vectors. CAS sends 1024-dimension vectors and asks for 30-60 results per intent, and both lower throughput (documented direction, size unmeasured) [SCR/gtm/research/scalability.verify.json missing items; https://learn.microsoft.com/en-us/azure/databricks/ai-search/best-practices, ms.date 2026-10-29].
- The LLM client sets no timeout. The SDK defaults (600 s read timeout, 2 retries) apply, so a stalled provider holds threads [app/llm_client.py:310-313; CLAUDE.md:1867].

The caches are per process, except the memo's store (code). The result cache (page_cache) holds 64 result pools [app/config.py:131]. The product-page cache holds 256 [app/page_cache.py:21]. The LLM memo keeps a 2,048-entry front in memory [app/llm_memo.py:85]. These, and the "one run per search at a time" guard, are not shared across workers or App instances, so two workers can each pay for the same new search. The memo's store behind that front is the one piece of state already shared: a SQLite file on one machine locally, and one file per entry on a Unity Catalog (UC) volume on the App [SCR/gtm/research/scalability.json SCA-09, SCA-10; app/llm_memo.py:335-399]. At R distinct new searches per second a cached pool survives about 64/R seconds: ~6 s at 10/s and ~1.3 s at 50/s (estimate) [SCR/gtm/research/scalability.json SCA-09]. After that, page 2 or a filter click re-runs retrieval and hydration (the LLM replies replay from the memo). On the App, memo entries are read and written over the network on the request thread. Their latency is unmeasured [app/llm_memo.py:335-399; CLAUDE.md:1055].

What each traffic level needs (estimates, 300K rows, UI path; "searches" are new submissions, before any cache or memo hits):

| Sustained new searches/s | LLM | AI Search | API workers | Caches and UI |
|---|---|---|---|---|
| 1 | fits Qwen3-Next pay-per-token (~2.4/s ceiling) | fits today's route (~3 QPS) | 1 worker, one Medium App | per-process caches are fine; the Groq laptop setup cannot serve this |
| 10 | at least ~76% memo or result-cache hits, or capacity outside pay-per-token | ~32 QPS: at or over the "few tens" cap; needs the index URL, perhaps 2 endpoint copies | ~2.6 processes by threads; sized as 2 Medium instances x 2 workers (2 workers per Medium App is unproven) | a cached pool lives ~6 s: needs a shared cache |
| 50 | ~95% hits, or provisioned capacity; QPH alone caps at ~50/s | ~160 QPS (~410 with API defaults): ~6 endpoint copies at 30 QPS each | ~13 processes by threads, ~7 CPUs of hydration; sized as 4 Large instances | shared cache and memo; the retailer's front end instead of Streamlit; model QPS reaches 450 if every search also opens one product page, over the 200 cap |

Sources: SCR/gtm/research/scalability.json tables[0] with the corrections in SCR/gtm/research/scalability.verify.json; SCR/gtm/research/gap_1.json G1-14, G1-29, G1-30, G1-31. The instance sizing assumes more than one worker per instance; the App as shipped runs one [databricks/app_bundle/run_app.py:91]. A memo hit does not reduce AI Search or embedding load; only a result-cache hit does [SCR/gtm/research/gap_1.json G1-05]. Databricks Apps allow at most 5 instances per app by default. That limit can be raised through the Databricks representative, and a workspace allows 20 such apps [https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/horizontal-scaling, ms.date 2026-09-25]. Streamlit computes for every user on its server and needs sticky sessions behind a load balancer, so it is a demo UI, not a scaling path [https://docs.streamlit.io/develop/concepts/architecture/architecture, fetched 2026-11-02].

Europe changes the LLM line. No Azure EU or UK region lists Qwen3-Next. The model-region pages for Azure, AWS and GCP are all dated 2026-10-30 [SCR/gtm/research/gap_4.json G4-01; https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability, ms.date 2026-10-30]. On Azure EU the in-region models CAS could switch to are gpt-oss-120b and gpt-oss-20b; the documented one-line revert is gpt-oss-120b, a reasoning model. Qwen3.5 122B is also listed there, as a preview CAS has never evaluated [SCR/gtm/research/gap_4.json G4-03, G4-11, G4-12]. With ~2,280 output tokens per search (estimate), gpt-oss-120b pay-per-token allows at most ~0.7 searches per second (~44 per minute) under the same 100,000 OTPM [SCR/gtm/research/llm.verify.json LLM-24]. It also took 30.3 s against Qwen3-Next's 10.4 s per query in the 2026-09-23 keyword-set evaluation [CLAUDE.md:1729-1730]. By the same Little's-law arithmetic, that cuts the thread-pool ceiling to ~1.3 searches per second per process (estimate). gpt-oss-20b has been unreliable as a reranker in our evaluations [SCR/gtm/research/gap_4.json G4-11]. AWS does list Qwen3-Next pay-per-token in eu-central-1 and eu-west-1, -2 and -3 [SCR/gtm/research/gap_4.json G4-04].

### 6.3 Cost

All figures below are list-price arithmetic, not measured bills. They use East US Azure Premium list prices fetched on 2026-11-02, 730 hours a month, and traffic held constant 24 hours a day [SCR/gtm/research/gap_1.json; model in SCR/gtm/gap_1/cost_model.py; https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Azure Databricks' and armRegionName eq 'eastus' (fetched 2026-11-02)].

Fixed floor (estimate). With zero traffic, the floor is $551.15 a month. That is one standard AI Search unit at $204.40 (4 DBU/h x $0.07 x 730 h) plus one Medium App at $346.75 (0.5 DBU/h x $0.95 x 730 h). It is $897.90 with the 2 App instances Databricks recommends for availability [SCR/gtm/research/gap_1.json G1-25; https://www.databricks.com/product/pricing/vector-search (no date shown, fetched 2026-11-02); https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/horizontal-scaling, ms.date 2026-09-25]. CLAUDE.md's "~$150-240/month" covers the AI Search endpoint alone. It leaves out the App and all token spend [CLAUDE.md:1621; SCR/gtm/research/claims.verify.json CLA-48].

Variable cost (estimate). Per 1,000 new UI searches on Qwen3-Next pay-per-token ($0.15 per 1M input tokens, $1.20 per 1M output), LLM tokens cost $1.46. Output tokens are 57% of that. Query embeddings add $0.003. These use the qwen3.8-27b token counts above (another tokenizer); on the older Databricks-measured token counts it is $1.20. API defaults add $0.10 (+7%). Each product page open costs $0.10 per 1,000 opens. On gpt-oss-120b the same search would cost ~$2.00 per 1,000, and on gpt-oss-20b ~$0.98 (their reasoning tokens estimated) [SCR/gtm/research/gap_1.json G1-07, G1-27; https://www.databricks.com/product/pricing/foundation-model-serving (no date shown, fetched 2026-11-02)]. Cache and memo hits cut the token line in proportion.

Where tokens take over (estimate). Tokens equal the one-instance floor at ~378,000 new searches a month, about 0.15 per second on average. Above that, tokens dominate: at 1 search per second sustained they are 87% of the bill [SCR/gtm/research/gap_1.json G1-26].

All-in monthly cost by scale point (estimates; UI path; East US list; tokens priced at Qwen3-Next pay-per-token):

| Catalog rows | New searches/s, 24x7 | Memo / cache hits | All-in, tokens at pay-per-token price | Fits Databricks pay-per-token? | All-in, Databricks only with gpt-oss-120b provisioned (bracket) |
|---|---|---|---|---|---|
| 300K | 1 | 0% | $4,392 | yes | same |
| 300K | 1 | 50% | $2,475 | yes | same |
| 300K | 1 | 75% | $1,517 | yes | same |
| 300K | 10 | 0% | $39,513 | no (OTPM) | $48,624-$74,174 |
| 300K | 10 | 75% | $10,412 | no (3% over) | $15,428-$19,078 |
| 300K | 50 | 0% | $196,055 | no | $237,962-$369,363 |
| 300K | 50 | 75% | $50,875 | no | $61,351-$94,202 |
| 1M | 1 | 0% | $4,473 (thin workers) | yes | same |
| 1M | 10 | 75% | $10,574 (thin workers) | no | $15,589-$19,239 |
| 5M | 1 | 0% | $5,329 (thin workers; $6,739 as built) | yes | same |
| 5M | 10 | 0% | $41,387 (thin workers) | no | $50,498-$76,048 |
| 5M | 50 | 0% | $201,028 (thin workers only; as built cannot run) | no | $242,935-$374,336 |

Source: SCR/gtm/research/gap_1.json tables[0] (G1-28). "Thin workers" is a remedy that is not built (6.4). The last column is a bracket between two assumptions about throughput per model unit, which Databricks does not publish for these models (low confidence, G1-19), and it needs a model switch. Where the answer is "no", the pay-per-token column shows what that capacity would cost at that price if it existed. One example is AWS Bedrock, which sells Qwen3 Next 80B at the same $0.15 / $1.20 per 1M tokens in US East; its quotas were not checked [https://aws.amazon.com/bedrock/pricing/ (no date shown, fetched 2026-11-02); SCR/gtm/research/gap_1.json G1-21]. Catalog size moves only the fixed lines; traffic and hit rate move the bill.

Provisioned throughput (PT): dedicated LLM capacity, billed by the hour in blocks of "model units" (documented and estimated). An always-on 50-unit block costs $5,353 a month for Qwen3-Next, $3,650 for gpt-oss-120b and $2,737 for gpt-oss-20b at list price [https://www.databricks.com/product/pricing/foundation-model-serving (no date shown, fetched 2026-11-02); SCR/gtm/research/gap_1.json G1-15]. Three problems follow.
- Qwen3-Next is priced for PT but no region lists it as available for PT. A reviewer read the whole region page (dated 2026-10-30), and the gap check found the same on Azure, AWS and GCP [SCR/gtm/research/scalability.verify.json SCA-15; SCR/gtm/research/gap_4.json G4-08]. On Databricks, PT therefore means switching to a PT-listed model, such as gpt-oss-120b (Qwen3.5 122B is PT-listed only as a preview in some regions and is unevaluated), and re-running the model acceptance checks [SCR/gtm/research/scalability.verify.json SCA-15; SCR/gtm/research/gap_4.json G4-12].
- Databricks does not publish tokens per second per model unit for these models. The only example is Llama 4 Maverick: 50 units give about 3,250 tokens/s at 3,500 in / 300 out [https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/model-units, ms.date 2026-09-23]. So one gpt-oss-120b block carries somewhere around 0.5-0.8 searches per second (estimate, low confidence) [SCR/gtm/research/gap_1.json G1-19].
- PT is not cheaper per search. A block costs the same as pay-per-token only when it carries ~1.4 (Qwen3-Next) or ~0.7 (gpt-oss-120b) searches per second around the clock. It buys capacity above the caps, not a lower unit price [SCR/gtm/research/gap_1.json G1-18]. Sized for a 3:1 daily peak and run at ~33% on average, a block costs ~2.7x pay-per-token (estimate) [SCR/gtm/research/gap_1.json G1-35].

Priority pay-per-token is not an option for CAS's models. Its supported list (GLM 5.x, Kimi K3, DeepSeek V4.1 Flash, a Qwen3.5 122B preview in West Europe) includes neither Qwen3-Next nor gpt-oss [https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/priority-mode, ms.date 2026-10-26; SCR/gtm/research/gap_1.json G1-20].

AI Search capacity (documented and estimated). One standard unit, $204.40 a month, covers 300K or 1M rows; 5M rows need about 4 units ($818) [SCR/gtm/research/gap_1.json G1-10]. "Target QPS", the documented way to buy more throughput on a standard endpoint, is billed "regardless of actual query traffic" and has no published price [https://learn.microsoft.com/en-us/azure/databricks/ai-search/high-qps, ms.date 2026-10-27]. The priceable route is copying the index to more endpoints. At a conservative 30 QPS per copy, 10 searches/s needs 2 copies ($409 a month at 300K-1M rows, $1,635 at 5M) and 50/s needs 6 ($1,226 at 300K-1M rows, $4,906 at 5M) [SCR/gtm/research/gap_1.json G1-30, tables[0]]. Standard endpoints beat storage-optimized at every CAS scale point ($204 against $935 per unit) [same].

App and Lakebase (estimate, low confidence). App compute is never the main cost. With thin workers, every scenario fits in at most 4 Large instances ($2,774 a month). Lakebase (Databricks' managed Postgres database) adds $81-$1,294 at ~$40 per compute unit per month; the per-unit-hour reading of its price is an inference [SCR/gtm/research/gap_1.json G1-22, G1-29]. As built, 5M rows need one XLarge App ($2,080 a month) per worker and cannot reach 50/s within 5 instances [same, G1-29]. This sizing assumes more than one worker per instance and an assumed 1.0 s per memo-hit search [same, G1-29].

One-time catalog embedding (measured tokens, estimated price): 44.73M tokens and ~83 DBU for 299,972 rows, about $5.81. A linear estimate gives ~$19 at 1M rows and ~$97 at 5M [CLAUDE.md:1538-1540; SCR/gtm/research/gap_1.json G1-24].

Europe (documented prices, estimated totals). West Europe list prices are +20% on model serving, +5% on Apps and +23% on Lakebase against East US. The same scenario costs ~19% more: $5,220 against $4,392 at 300K rows, 1/s, no hits [https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Azure Databricks' and armRegionName eq 'westeurope' (fetched 2026-11-02); SCR/gtm/research/gap_1.json G1-33]. That figure still prices Qwen3-Next, which no Azure EU region lists (6.2). On gpt-oss-120b the token line is ~$2.00 per 1,000 searches in East US and ~$2.40 in West Europe (arithmetic: West Europe unit prices are 1.2x). So an Azure-EU client's LLM line is roughly 1.6x the East US Qwen3-Next figure, before its lower capacity ceiling (estimate).

Cost levers, cheapest first (code and estimates) [SCR/gtm/research/gap_1.json G1-32]:
1. API callers send recommend=false, or the default becomes false. This saves 1 LLM, 5 embedding and 5 AI Search calls per search [app/config.py:162].
2. Batch the product-page grounding phrases into one embedding call. query_many already takes a list, but each phrase is a separate search today [app/recommendations.py:138; app/vector_search.py:240-250].
3. Ask the reranker for scores only, without a reason sentence per row. Shoppers see reasons only with Developer details on. Estimated effect (inference, untested): 689 to ~246 output tokens, $1.46 to $0.93 per 1,000 searches (-36%), and the pay-per-token ceiling from ~2.4 to ~6.8 searches/s. It needs a quality A/B first [SCR/gtm/research/gap_1.json G1-32; SCR/gtm/research/llm.json LLM-50; SCR/gtm/research/llm.verify.json LLM-50].
4. Set max_tokens per stage: translate ~150 and rerank ~800, against measured maxima of 110 and 715 (31 calls each, Groq qwen3.8-27b, 2026-11-02) [SCR/gtm/research/gap_1.json recommendations; SCR/nov02/live/results.json].
5. Warm the memo with frequent queries. Each point of hit share removes about 1% of LLM spend.

### 6.4 Remedies, in order

| # | Remedy | What it removes | Evidence | Effort and status |
|---|---|---|---|---|
| 1 | Normalise the laptop embedding matrix once; use a matrix product and a partial sort | laptop vector scan 460 to ~12 ms per intent; ~0.44 GB temporary copy per call | measured: SCR/gtm/research/scalability.verify.json SCA-04, SCA-24 | a few lines in app/embeddings.py and search.load_index (inference); no dev-day estimate; not done. Laptop path only |
| 2 | Query AI Search through its index URL with the App service principal's OAuth token; run per-intent queries in parallel; batch product-page embeddings | the "few tens of QPS" cap on the workspace URL; intents run one after another | documented: https://learn.microsoft.com/en-us/azure/databricks/ai-search/high-qps (ms.date 2026-10-27); code: app/vector_search.py:240-260 | described by the research lane as a small code change in app/vector_search.py (inference) [SCR/gtm/research/scalability.json tables[3]]; no dev-day estimate. Target QPS is unpriced |
| 3 | Buy LLM capacity and cut tokens: provisioned throughput (on Databricks this means switching to a PT-listed model such as gpt-oss-120b and re-running the acceptance checks) or another provider; max_tokens per stage; recommend=false for API callers; scores-only rerank after an A/B | the ~2.4/s pay-per-token ceiling; admission reservations | documented and estimated: SCR/gtm/research/gap_1.json G1-14 to G1-20, G1-32 | no dev-day estimate; PT throughput per unit has to come from the Databricks account team |
| 4 | Thin workers: product-id keys and incremental refresh (catalog-onboarding phase P3); rows fetched by id from a Lakebase synced table; product type and taxonomy stored at ingestion; BM25 moved to AI Search hybrid or full-text search, or to the retailer's own keyword engine | ~1.5 KB per row of table plus ~3.5 KB per row of BM25 in every worker; ~19 s of annotation per 300K rows at every App start; the row-by-row keyword loop | code, measured, documented: SCR/gtm/research/scalability.json SCA-01, SCA-02, SCA-03, SCA-22, SCA-23; https://learn.microsoft.com/en-us/azure/databricks/oltp/projects/sync-tables (ms.date 2026-10-30) | P3: 5.5 dev-days (estimate for one developer, not measured; the fourth phase of a ~36 dev-day onboarding plan) [SCR/oct27/byo/REPORT.md:400-410]. The rest is not estimated. Lakebase lookup latency for 150-300 rows per search is unmeasured; a synced table loads up to 2,000 rows/s per compute unit as a snapshot, ~150 rows/s per unit in continuous mode (documented) [SCR/gtm/research/gap_1.json G1-23]. Hybrid search merges vector and keyword results by Reciprocal Rank Fusion (RRF). It needs text columns in the index, which means a new index (~48 min build), uses ~4x the resources of plain vector queries and returns at most 200 results by default. Full-text search is in Beta [https://learn.microsoft.com/en-us/azure/databricks/ai-search/query-ai-search, ms.date 2026-10-29; SCR/gtm/research/scalability.verify.json SCA-23] |
| 5 | Scale out with shared state: more workers or App instances; result cache and memo in a shared store (Lakebase or Redis); async endpoints with LLM timeouts; the retailer's front end instead of Streamlit, through an API-only deployment | 40 threads and one GIL per process; per-process caches; Streamlit sharing the container | code and documented: SCR/gtm/research/scalability.json SCA-06 to SCA-10, SCA-20; SCR/gtm/research/architecture.json ARC-08, ARC-09 | only after step 4, since RAM multiplies by workers. The API listens on 127.0.0.1 inside the App today and has no authentication of its own [databricks/app_bundle/run_app.py:89-91; SCR/gtm/research/architecture.json ARC-08], and Databricks Apps cannot be made public [https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/permissions, ms.date 2026-09-11; SCR/gtm/research/architecture.json ARC-09], so step 5 includes the API-only deployment described in the architecture section. No dev-day estimate |

Before any searches-per-second figure reaches a client, a workspace load test is owed. It should cover: App start-up time and memory on 2 vCPU at 300K rows; memo read and write latency on the App's volume; a stepped load test (1, 5 and 10 searches per second) that records which limit refuses requests (LLM or AI Search); and the footprint without BM25. Then repeat it on a synthetic 1M-row catalog. Databricks links a load-test notebook for AI Search endpoints [https://learn.microsoft.com/en-us/azure/databricks/ai-search/best-practices, ms.date 2026-10-29]. No effort estimate exists for this work [SCR/gtm/research/scalability.json recommendations].

### Open questions for this area

- Which Databricks workspace tier and region would each client use? The model limits used here are the Enterprise-tier table; Azure Premium probably maps to it (inference). In Azure EU regions Qwen3-Next is not listed at all, and the listed Qwen3.5 122B preview is unevaluated.
- How large is Databricks' default output reservation when max_tokens is omitted? It decides how far below ~2.4 searches per second requests start being refused.
- How many tokens per second does one provisioned-throughput block of gpt-oss-120b deliver for CAS's request shape (~4.2K in, ~0.7K out)? Only the Databricks account team or the model-optimization API can say.
- What does target QPS cost per QPS, and what throughput does one 1024-dimension endpoint copy sustain at 30-60 results per intent?
- What share of a target retailer's searches are exact repeats, how many page turns and filter clicks follow each search, and what are peak and average searches per second? Without a query log, the 0 / 50 / 75% columns are scenarios, not forecasts.
- What is the serving unit for fashion: one card per SKU, per style-colour or per title? And one catalog per market, or one across markets? This sets the row count by up to an order of magnitude.
- How large is a grocery chain's union catalog across banners? No target publishes it.
- What are Lakebase lookup latency for 150-300 rows per search and the memo's file read/write latency on the App?
- What are the App's real start-up time and memory on 2 vCPU, and do 2 workers fit on a Medium App at 300K once start-up peaks are counted? (The laptop probe implies ~109 s for a local start on a taxonomy-cache miss, and ~61 s on the App path, against CLAUDE.md's ~50 s for a boot from the caches; not fully reconciled.)
- Can AI Search hybrid or full-text search reproduce BM25's popularity and rating blend well enough to replace it?
- What are Bedrock's quotas for Qwen3 Next 80B? They decide whether pay-per-token pricing can carry 10-50 searches per second off Databricks.
- Is raising the 5-instance App limit practical, or does high-traffic serving belong outside Databricks Apps (see the cloud section)?

## 7. Replace the client's search, or sit on top of it?

Answer. Sit on top of the client's search; do not replace it (our recommendation, from the evidence below). None of the five P0 accounts is starting without search (documented). Kroger's search runs on Empathy.co, and H&M rebuilt its onsite search with Voyado in 2026. Albertsons runs Google's and an OpenAI-model assistant alongside an unnamed keyword engine. H-E-B builds in-house with an unnamed engine, and C&A's text engine is unknown (Syte supplies its visual search) [GAP2 G2-01, G2-06, G2-07, G2-11, G2-14, G2-17, G2-25]. Start with "D-lite": keyword-like queries stay on the client's engine. Only the queries it handles badly (long, natural-language ones) get CAS's rewritten query and filters, still run on that engine. Add CAS reranking of that tail second. Today CAS can only replace a search, and it has never been connected to any client's live engine. /search returns rows of CAS's own catalog copy, and on Databricks the API cannot be reached from outside the App (code, documented) [ARCH ARC-03, ARC-08, ARC-09; SCR/gtm/research/claims.json CLA-57]. Phase 1 (an API-only service, a first engine adapter and the router) is about 12-20 developer-days. That is an estimate, and it is our reading of an inconsistent figure in the lane (see Effort).

### How to read this section

- Labels: measured = numbers from a run; code = read in the repo; documented = a fetched page says it; estimate = our own sizing; inference = our reasoning across sources.
- Per-stage latency has never been measured on Databricks. Only whole-pipeline run_eval times exist there: 4.9 s per query with rerank off, 10.2 s with rerank on (keyword set, n=18, 2026-10-12) [CLAUDE.md:862-863]. Nothing has been measured on the Databricks App itself, and no load test exists anywhere in the repo.
- Source keys. SCR is the session scratchpad folder:
  - [SIM] = SCR/gtm/research/arch_sim/sim_patterns_out.json. Its script is SCR/gtm/research/arch_sim/sim_patterns.py. The verifier re-ran it with 0 differences (SCR/gtm/verify_architecture/sim_rerun_out.json).
  - [LAT] = SCR/gtm/research/latency_from_nov02.txt, recomputed from SCR/nov02/live/results.json. Groq qwen3.8-27b, local gte-small, 31 held-out queries, 2026-11-02, with the harness's rate-limit pauses removed.
  - [ARCH] = SCR/gtm/research/architecture.json (finding ids ARC-nn). Its verdicts are in [ARCH-V] = SCR/gtm/research/architecture.verify.json.
  - [GAP2] = SCR/gtm/research/gap_2.json (G2-nn).
  - [GAP4] = SCR/gtm/research/gap_4.json (G4-nn).
  - [BYO] = SCR/oct27/byo/REPORT.md.
  - [NOV2] = SCR/nov02/analyse/score.txt.

### The five patterns

"Head" queries are short and name a product ("black dress"). "Tail" queries are long and describe a need ("something breathable for a humid day").

- A. Full replacement. The client's search box calls CAS. CAS holds the whole catalog, its own index and all ranking. The client's engine is retired.
- B. Query-understanding layer. CAS turns the shopper's words into search intents and typed constraints (for example "under $60" becomes a budget). An adapter compiles them into the client engine's own query and filter syntax. The client's engine retrieves and ranks. CAS needs no copy of the catalog.
- C. Rerank layer. The client's engine retrieves as it does today. CAS's LLM (large language model) reranker reorders the engine's top results (CAS sends 15 rows) and returns a score and a reason per item.
- D. Router. A rule decides per query: head queries stay on the client's engine, and the rest go to CAS.
  - D-lite = router + B. The tail still runs on the client's engine, with CAS's rewritten query and filters.
  - D-full = router to CAS's own indexed path (CAS's catalog copy, retrieval and ordering).
- E. Rank fusion. Both systems run on every query, and their lists are merged by RRF (reciprocal rank fusion: an item scores the sum of 1/(60 + its rank) over the lists).

### What each pattern costs and what the simulation shows

How the simulation worked (measured, 0 LLM calls, 2026-11-02):
- The client's engine was played by CAS's own BM25 keyword engine (BM25 is a classic word-match ranking) over the 299,972-row catalog.
- It was fed the translator intents recorded in the 2026-10-30 prompt A/B: Groq gpt-oss-20b, translator only, one sample per prompt.
- The score is P@10, the share of the first 10 results that are relevant.
- The queries are the 16 "context" queries, where the product type never appears in the query ("keep my coffee hot" means a travel mug) [SIM; .claude/rules/eval.md:22-31].
- Figures are for the shipped query-expansion prompt (d3e184c6). The previous prompt (94fffb8d) is in brackets.
- "CAS's own path" below is the recorded CAS list with the reranker off, so neither the reranker nor main.py's ordering is in these numbers.

| Pattern | What the client keeps | What CAS needs | Simulation, context set, P@10 (measured) | Added latency per query | Effort, dev-days (estimate) | Fit: fashion / grocery |
|---|---|---|---|---|---|---|
| A full replacement | Nothing of its search. Engine, merchandising rules and stock filters move to CAS or are lost (inference) | The whole catalog compiled, embedded and held in memory: ~1.5 GB measured for catalog + keyword index [CLAUDE.md:1780], ~1.7 GB per API worker estimated at 300K rows [SCR/gtm/research/probe_scalability.json "derived"]. Also /api/ routes and auth, and keys by the client's product id | CAS's own path 0.463 (0.450) vs stand-in engine 0.206 [SIM] | The whole page is CAS. Groq qwen3.8-27b, local: p50 3.6 s / p90 4.6 s [LAT]. Databricks Qwen3-Next: ~10.2 s per query, whole pipeline, rerank on (run_eval, keyword set, n=18, 2026-10-12; not the App) [CLAUDE.md:862-863] | ~40+, before the operating layer incumbents already ship (merchandiser console, A/B tests, analytics, stock) [ARCH tables[0]] | Neither segment: none of the five accounts is greenfield [GAP2 G2-25] |
| B query understanding | Engine, index, filters, facets, merchandising, stock, ads | An LLM endpoint and a small API; a constraint-to-filter compiler per engine; no catalog | 0.206 -> 0.369 (RRF merge), 0.394 (round-robin merge), 0.369 (one combined query); (0.419 / 0.406 / 0.456). Keyword set 0.929 -> 0.993 (14 queries) [SIM] | +1 LLM call. Translate p50 429 ms / p90 641 ms (Groq qwen3.8-27b) [LAT]. Plus the engine's own intent searches, one round trip where the engine batches them [ARCH ARC-31]. Not measured on Databricks: the nearest figure, 4.9 s for translate + retrieval with rerank off, is an upper bound (inference) [CLAUDE.md:862-863] | 9-15 for the first engine, +3-6 for each further engine [ARCH tables[0]] | Grocery first (inference): the engine keeps store-level stock and out-of-stock substitution, which CAS lacks [docs/COMPETITIVE_SUMMARY.md:42-43], and its own diet facets (Kroger, since 2021) [GAP2 G2-02, G2-11]. Fashion gets the translator's clothing rules only |
| C rerank | Engine and its retrieval; CAS reorders the top 15 | Top-N rows' text per request; a field mapper; a wrapper that re-appends rows 16+ and any row the model dropped. Seller text enters the rerank prompt unscreened: indirect injection is still OPEN [.claude/rules/safety.md:139-140] | Ceiling only: a perfect reranker over the engine's top 15 reaches 0.263 (top 30: 0.381); over B's top 15, 0.500 (0.556) [SIM] | +1 LLM call over 15 rows. p50 1,734 ms / p90 1,992 ms (Groq) [LAT]. ~5.3 s on Databricks Qwen3-Next, by difference of two separate runs (estimate) [CLAUDE.md:862-863] | 5-8, plus a main.py refactor if the type-aware ordering is wanted [ARCH tables[0]] | Fashion, on top of B (the rubric carries clothing rules [app/reranker.py:20-60]; inference). Weak on its own for inferred-intent queries |
| D-lite (router + B) | Everything; head queries untouched | A router adapter on the engine's top hit, plus B | 0.369 (0.425); keyword set 0.929 on the 14 A/B queries, the engine's own score [derived here from [SIM]'s per-query rows; re-derived by this section's check, not in the lane or verifier files] | Head: 0. Tail: the router's first engine call, then as B: the translate call ~0.4-0.6 s on Groq, plus the engine's intent searches. With C, the two LLM calls sum to ~2.2-2.6 s (sum of the stage p50s / p90s; estimate) [LAT] | 12-20 (this section's reading; see Effort) | Both segments (inference) |
| D-full (router to CAS) | Engine, for the head only | As A, plus the router | Routed 0.456 (0.444); keyword set 0.907 vs engine 0.929 (14 queries); 5 of 30 queries misrouted [SIM; ARCH-V ARC-15] | Head: 0. Tail: a full CAS page, 3.6 s p50 on Groq [LAT] / ~10 s on Databricks run_eval [CLAUDE.md:862-863] | 30-45 [ARCH tables[0]] | Fashion, if a pilot shows the full path earns its cost (inference); one catalog per market [BYO:460] |
| E fusion (RRF) | Engine, but CAS runs on every query | As A, plus an id crosswalk between the two systems | 0.331 vs CAS alone 0.463: CAS better on 8 queries, worse on 3, tied on 5. It does beat the plain engine (0.206; 10 wins, 2 losses). Keyword set 0.964 vs engine 0.929 [SIM] | CAS's full latency on every query | 30-45 [ARCH tables[0]] | Not recommended |

Reading the numbers:

- On head queries CAS adds little (measured, 18 keyword queries on our catalog).
  - On the 18 keyword queries the stand-in engine already scores 0.928, and plain embedding search with no LLM scores 0.944. The set is saturated [SIM; .claude/rules/eval.md:24].
  - B nudged the engine from 0.929 to 0.993 on the 14 of those queries the A/B covered (1 win, 13 ties) [SIM].
  - The Nov 2 held-out run did not test head queries, so it is not cited here [ARCH-V ARC-16].
- B nearly doubles a word-match engine on inferred-intent queries, but stops short of CAS's own retrieval (measured on the stand-in, 16 queries).
  - On the shipped prompt it moves 0.206 to 0.369-0.394 (x1.8-1.9), against CAS's 0.463.
  - That closes about 63% (RRF merge) to 73% (round-robin) of the gap. B with the RRF merge loses to CAS on 9 of 16 paired queries (3 wins, 4 ties) [SIM; ARCH-V ARC-12].
  - The previous prompt scored higher (0.406-0.456). The RRF and round-robin gaps between the prompts (0.050 and 0.012) count as ties under the repo's 0.05 rule. The one-call gap (0.087) does not. But each prompt had one gpt-oss-20b sample, and single-query swings of +-0.4 are noise on that model [.claude/rules/eval.md:65, :191].
- Reranking alone cannot rescue a weak list (measured ceiling).
  - Even a perfect reranker over the engine's top 15 reaches only 0.263 [SIM].
  - So C belongs on top of B, not instead of it.
  - The real reranker's gain on a client engine's list has not been measured.
- Fusion dilutes CAS on the queries CAS exists for (measured).
  - It scores 0.331 against CAS's 0.463, and it costs a full CAS run on every query [SIM].
  - Caveat: the fused CAS list was only 10 deep and not reranked, so ranks 11-12 could only come from the engine's list. An equal-depth or weighted fusion was not tried [ARCH-V ARC-14].
- The existing router rule splits the two sets almost cleanly (measured). The rule: the engine's best-matching document must contain at least 75% of the query's words, stopwords excluded [app/hybrid_router.py:18-25; app/keyword_search.py:150-187].
  - 17 of 18 keyword queries stay on the engine, and 15 of 16 context queries go to CAS [SIM].
  - It misroutes 5 of the 30 queries with both lists recorded:
    - "warm wool sweater for cold weather" (coverage 0.80) stays on the engine at 0.0, where CAS scores 1.0;
    - "never lose my keys again" (coverage 1.00) stays on the engine at 0.1, where CAS scores 0.2;
    - "power bank to charge my phone on the go" and "baking for a friend with celiac disease" go to CAS and do worse;
    - so does one more context query, which differs by prompt: "cut sugar from my coffee without losing the sweetness" on the previous prompt, "breakfast I can make in two minutes before work" on the shipped one [SIM; ARCH-V ARC-15].
  - A perfect router would score 0.512-0.525 on the context set and 1.000 on the keyword set [SIM]. So the rule must be calibrated on the client's own query logs.
- D-lite (router + B) scores 0.369 on the context set, the same as B, because the router sends 15 of 16 of those queries to the tail. On the 14 keyword queries it scores 0.929, the engine's own score, because the head stays untouched. (Measured. This section derived it from [SIM]'s per-query rows, and its check re-derived it; neither the lane nor the verifier reported it.)
- Fit by vertical, from the Nov 2 held-out run (measured; Groq qwen3.8-27b, local gte-small, 31 AI-written held-out queries, one run, 2026-11-02). That run graded the first page of results. "Strict" = the share of cards graded fully relevant, averaged over two LLM grader prompts (grader model not recorded). The control is the same catalog searched by embeddings with no LLM.
  - Fashion has the weakest plain baseline: page-1 strict 0.23 with no LLM, against 0.50 for the full CAS path (+0.27, n=13; 11 wins, 1 loss, 1 tie).
  - Grocery's plain search is already stronger: 0.52 -> 0.71 (+0.19, n=8; its 95% interval includes 0).
  - Electronics: 0.51 -> 0.82 (+0.31, n=6; 5 wins, 0 losses, 1 tie), the largest strict lift. On lenient scores fashion and electronics tie (+0.34 each) [NOV2:16-18, :78-80, :84-86].
  - These lifts are for the whole pipeline against an embedding control. They cannot be credited to main.py's ordering alone, and part of the apparel tuning sits in the translator, so B carries some of it [ARCH-V ARC-29].
  - "Grocery fits B" is an inference, not a measurement.

### What the simulation cannot tell us

- The engine is a stand-in, not a client engine. It has no typo tolerance, synonyms, field tuning or merchandising rules [SCR/gtm/research/arch_sim/sim_patterns.py:6-8].
  - Kroger's Empathy engine already lists a hybrid index and vector search [GAP2 G2-02].
  - Against such an engine the starting point is probably higher (inference). For comparison, CAS's own no-LLM embedding search scores 0.319 on these queries [SIM].
  - So B's gain there is unmeasured, and probably smaller (inference).
- These numbers are not "X% better than your current search", or than any retailer's engine. That comparison has never been measured [SCR/gtm/research/claims.json CLA-10].
- The intents are one gpt-oss-20b sample per prompt, with the reranker off. Single-query swings of +-0.4 are noise on that model [.claude/rules/eval.md:191].
- Relevance comes from rule-based title and attribute criteria, not human labels.
  - The data are English queries on the Amazon catalog: 16 context and 18 keyword queries (14 in the keyword arms).
  - The queries are not weighted by traffic [SIM; .claude/rules/eval.md:5-31].
- C is a ceiling, not a run of the real reranker. D-full and E used CAS's recorded top 10 with the reranker off and without main.py's ordering.
- The stand-in's speed (pure Python, p50 257 ms / p90 475 ms on a laptop, 282 queries) says nothing about a production engine [SIM meta].
- The verifier found that duplicate titles do not inflate the stand-in's score [ARCH-V ARC-12].

### What CAS can and cannot do today

1. /search can only replace (code).
   - Every result is a whole row of CAS's catalog. It is keyed by catalog_index, a row position in that catalog, and GET /product takes that position [app/search.py:519-521; app/main.py:2064, :2530].
   - An id compiled into the catalog rides along with the row (today parent_asin). But /search cannot annotate ids it was never given [ARCH-V ARC-03].
   - A client engine could also be added as a third retrieval backend next to "local" and "databricks", which would keep main.py's ordering [app/search.py:537-548]. That still needs the whole catalog in CAS's memory, mapped to the client's ids. It also needs score normalisation: main.py drops the page when the best score is under MIN_RESULT_RELEVANCE, a cosine threshold (default 0.5) that BM25 or Algolia scores are not on [app/main.py:1981; app/config.py:125; ARCH ARC-06].
   - No integration with any client's live search engine exists [SCR/gtm/research/claims.json CLA-57].
2. /unified_search is already a keyword-first router, pattern D (code).
   - It runs CAS's own BM25 engine first. It calls the full CAS search (with recommendations off) when the top hit covers less than 75% of the query's terms, or when nothing matches [app/hybrid_router.py:11-50; app/main.py:2459-2527].
   - No UI page calls it and no test exercises it. It has never been measured live; the simulation above is its first offline measurement [pages/ai_search.py:39; pages/keyword_search.py:29; pages/shopping_assistant.py:25; ARCH-V ARC-01].
   - Its keyword engine is passed in from outside, so an adapter for the client's engine can stand in. The adapter must recompute "coverage" from the client's top hit; CAS computes it from title, category, colour and material [app/hybrid_router.py:1-2, :23; app/keyword_search.py:43-45].
   - Its tokenizer reads only ASCII letters and digits, so a German or Swedish word is split into fragments ("Kopfhörer" -> "kopfh", "rer") [app/keyword_search.py:12; GAP4 G4-15]. The coverage score is then unreliable (inference).
3. The translator needs no catalog, but nothing exposes it alone (code).
   - One LLM call returns the intents, a bundle type and constraints. On failure the cleaned query becomes the only intent [app/translator.py:580-677].
   - Constraints are typed free text (gender, dietary, material, budget or other; up to 60 characters, e.g. "under $60"), not engine filters [app/translator.py:427-440]. So each engine needs its own constraint-to-filter compiler.
   - No endpoint exposes the translator on its own today. The API's routes are /healthz, /stats, /search, /keyword_search, /unified_search, /product, /chat and /feedback [app/main.py:228, :240, :2064, :2423, :2459, :2530, :2623, :2738; ARCH ARC-04].
4. The reranker is a function over candidate rows (code).
   - It reads CAS's column names (title, category, vertical, price, colour, material, occasion, rating, the first 100 characters of the description) and sends the first 15 rows (RERANK_INPUT_K) [app/reranker.py:126-162, :286; app/config.py:129].
   - It returns at most those 15. On success it returns only the rows the model returned; on failure, the input order with no score [app/reranker.py:362, :364-380].
   - So a C wrapper must re-append the client's rows 16+ and any row the model dropped. Only then does a failure leave the client's order intact [ARCH-V ARC-05]. It also reads and writes the LLM memo.
5. The type-aware ordering does not travel with the reranker (code).
   - The ordering covers type, gender, age, fit and device tiers, display bands and ad boosting. It lives in _ranked_pool, next to CAS's own retrieval.
   - Facets and variant collapse run afterwards, in search(), on the cached pool. All of them read columns derived when the catalog loads [app/main.py:930, :1009, :1921, :2358-2365; app/product_type.py:1123; app/catalog_taxonomy.py:2092].
   - Plain C gets the reranker's rubric, not this ordering. Getting the ordering means annotating the client's rows and refactoring _ranked_pool and search() [ARCH-V ARC-07].
6. Nothing outside the App can call the API today.
   - The Databricks App starts FastAPI on 127.0.0.1:8001 with one worker, and Streamlit takes the public port (code) [databricks/app_bundle/run_app.py:9-12, :88-91]. The API has no authentication of its own (code) [ARCH-V ARC-08].
   - Databricks Apps cannot be made public. Token access works only for apps that expose /api/ routes. External applications should use service-principal M2M (machine-to-machine) credentials, and callers need CAN USE on the app (documented) [https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/permissions and https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/connect-local, ms.date 2026-09-11].
   - So patterns B-E all need an API-only deployment with /api/ routes. The retailer's backend would call it, never the shopper's browser (inference).
7. CAS sets no LLM timeout of its own (code).
   - The openai SDK defaults apply: a 600 s read timeout and 2 retries [app/llm_client.py:310-313; CLAUDE.md:1867-1869].
   - So any overlay needs a client-side deadline that falls back to the engine's own results.
   - Under A, CAS down means search down.
8. Throughput limits the tail, not only latency (estimate).
   - Databricks publishes an output limit for Qwen3-Next (listed as Beta) on pay-per-token endpoints: 100,000 output tokens per minute (OTPM). The figure is from its Enterprise-tier table, and our workspace's tier is not recorded (documented) [https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits, ms.date 2026-10-30; SCR/gtm/research/scalability.verify.json SCA-13].
   - At that limit a workspace can serve roughly 2.4 uncached full searches per second on today's prompts (2.5 on the older Databricks-measured token count). At 10 distinct searches per second, more than about 76% would have to come from cache or the reply memo (estimate) [SCR/gtm/research/gap_1.json G1-14; SCR/gtm/research/probe_scalability.json "derived"].
   - The 2.4 uses today's 689 output tokens per search, counted on Groq's qwen3.8-27b (a different model and tokenizer); the 2.5 uses 2,709 prompt / 658 completion tokens from a 2026-09-23 Databricks run on older, shorter prompts. Both assume 2 LLM calls per search. An API caller that leaves recommend on (the API default) adds a cross-sell call on every page 1, which lowers the ceiling to about 2.3 per second [SCR/gtm/research/gap_1.json G1-14; SCR/gtm/research/scalability.verify.json SCA-13, missing; app/config.py:162; pages/ai_search.py:120].
   - Databricks recommends provisioned throughput for sustained production use. No region offers Qwen3-Next that way, so raising this cap likely means a model switch (documented + inference) [scalability.verify.json SCA-13; GAP4 G4-08].
   - Under D only the routed tail counts against the limit. Per routed query, B adds one LLM call (~1,500 prompt tokens on qwen3.8-27b) and C one more (~2,630-3,690 prompt tokens). These are measured per-call counts (Groq, 2026-10-30) [.claude/rules/eval.md:418; ARCH ARC-30]. How many output tokens a B-only call writes on Qwen3-Next is unmeasured.
   - The API runs one worker. On the laptop, 4 threads made row loading slower, not faster (speedup 0.89, Python 3.13 with the GIL; measured) [SCR/gtm/research/probe_scalability.json "measured.concurrency"].
   - Databricks Apps run at most 5 instances per app (a default limit, raisable on request; documented). CAS's page cache is in-process, so pages 2+ come from cache with 0 LLM calls only when they reach the same instance (code) [ARCH ARC-10; SCR/gtm/research/scalability.verify.json TABLE-bottleneck].

### Which CAS outputs survive when the client's engine owns the list

| CAS output | Where it is computed | Under B | Under C |
|---|---|---|---|
| Intents and constraints | translator.understand_query [app/translator.py:580-677] | Yes: they are B's output, compiled into engine filters | Only if B also runs. The reranker takes constraints as input [app/reranker.py:275-281] |
| Rerank score and per-item reason | reranker [app/reranker.py:275-380] | No | Yes, for at most 15 rows. Reasons pass the output screen. Today's UI shows them only under Developer details [CLAUDE.md:1476-1478] |
| Facets | _facets in search(), over CAS's pool [app/main.py:1617, :2358] | No: the client engine's facets serve | No |
| match_quality ("No exact matches" banner) | _match_quality in search(), over CAS's pool; reads rerank scores and type columns [app/main.py:1875-1900, :2359] | No: nothing is scored | Only after annotating the client's rows and refactoring (inference). Unproven anyway: it flagged 0 of the 8 worst live pages [CLAUDE.md:1230] |
| Variant collapse (one card per style) | _collapse_variants in search(); Fashion rows; search.style_key with English colour and size words [app/main.py:1825, :2362; app/search.py:465; GAP4 G4-18] | No | No as built; a refactor could apply it to the client's list (inference) |
| Sponsored gating | _boost_sponsored: ads keyed by parent_asin, shown only above rerank score 40 [app/main.py:1274-1280; app/sponsored.py:1-16; app/config.py:175] | No: the client's own ad stack stays (e.g. Criteo inside Albertsons' search [GAP2 G2-06]) | No as built. The rerank score could feed the client's own ad gate (inference) |
| Type, gender, age, fit and device ordering | _flat_window / _flat_display in _ranked_pool, reading load-time columns [app/main.py:930, :1009, :1921] | No | No for plain C. It needs the annotators on the client's rows and a refactor [ARCH-V ARC-07] |
| Recipe and setup bundles | The translator sets bundle_type [app/translator.py:580-591]; main.py builds slots over CAS's pool [app/main.py:322, :1226] | Partly: B can return the bundle type and one intent per slot, and the engine runs one query per slot (inference) | No |
| Cross-sell and upsell | Cross-sell: LLM complement phrases [app/recommendations.py:60], then a grounding search [:127-138]. Upsell: CAS embedding neighbours plus ratings [:189-219] | Cross-sell phrases yes, if the client sends the anchor product's text and grounds the phrases on its engine (inference). Upsell no | Same as B |
| Injection screen and reply memo | query_guard and llm_client | Yes. The guard's input layer is English-only [GAP4 G4-19] | Yes, but the screen does not cover the candidate rows: seller text enters the rerank prompt unscreened [.claude/rules/safety.md:139-140] |

### What the client engines let us plug into

All rows below are documented. Pages were fetched 2026-11-02, with the page's own date where it shows one.

| Engine | B: rewritten query + structured filter | C: rerank hook | D / E: routing or fusion | Source |
|---|---|---|---|---|
| Elasticsearch | Bool-query filters, which do not score; _msearch batches several searches into one request | text_similarity_reranker retriever (GA, generally available; window default 10). A custom inference service accepts the "rerank" task (GA, 8.19.0), so CAS could in principle sit behind it. It carries scores only, not reasons (inference) | RRF retriever (GA, rank constant 60) | https://www.elastic.co/docs/explore-analyze/query-filter/languages/querydsl ; https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-msearch ; https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/text-similarity-reranker-retriever ; https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-inference-put-custom ; https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion (no page dates) |
| OpenSearch | Not fetched (standard query filters) | Rerank processor (2.12; by_field 2.18); tutorials for externally hosted cross-encoders | Score-ranker processor applies RRF to hybrid queries (2.19) | https://docs.opensearch.org/latest/search-plugins/search-pipelines/rerank-processor/ ; https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/ |
| Solr 10.1 | fq filters without changing scores | Rerank / learning-to-rank parsers use Solr-side models. No external hook on the page, so CAS would sit in the application layer (inference) | - | https://solr.apache.org/guide/solr/latest/query-guide/common-query-parameters.html ; https://solr.apache.org/guide/solr/latest/query-guide/query-re-ranking.html |
| Algolia | filters (attributesForFaceting, numeric, AND / OR / NOT); one multi-query request carries N intents | None found on the pages fetched; up to 1,000 hits per page to rerank outside | NeuralSearch "conservative" preset: semantic search only when keyword search finds nothing (not on every plan) | https://www.algolia.com/doc/api-reference/api-parameters/filters/ ; https://www.algolia.com/doc/rest-api/search/search.md ; https://www.algolia.com/doc/api-reference/api-parameters/hitsPerPage/ ; https://www.algolia.com/doc/guides/ai-relevance/neuralsearch/get-started |
| Bloomreach Discovery | fq, including price ranges (price:[200 TO 300]); efq for boolean filters, but no numeric ranges | Not checked | - | https://documentation.bloomreach.com/discovery/reference/faceting-and-filtering (updated 2026-07-27) |
| Constructor | pre_filter_expression scopes Search and Browse (ranges, and / or / not; depth <= 5; <= 150 filters) | Not checked | - | https://docs.constructor.com/reference/shared-pre-filter-expressions ; https://docs.constructor.com/reference/shared-filter-expressions (updated 2025-11-26) |
| Google Vertex AI Search for commerce | Filter expressions (price: IN(*, 100.0e), NOT), orderBy | Not checked | Conversational Commerce agent: "defaults to traditional search results for simple queries" | https://docs.cloud.google.com/retail/docs/filter-and-order ; https://cloud.google.com/blog/products/ai-machine-learning/introducing-conversational-commerce-agent-on-vertex-ai (2025-09-11) |
| Empathy.co | Extension API not read | Not checked | AI Mode button for queries of two or more keywords; it "does not replace" keyword search and resolves back into search queries | https://docs.empathy.co/blog/welcome-ai-mode.html (updated 2026-10-30) |
| commercetools | Filters and facets unchanged across modes | - | Merchant-chosen mode: lexical / semantic / hybrid (RRF) | https://commercetools.com/blog/b2b-product-spotlight-storefront-search (2026-05-19) |
| Coveo | Partial only: the legacy JavaScript Search UI documents a basic query (q) plus a code-generated advanced expression (aq). Main docs returned HTTP 503 four times | Unverified | Unverified | https://coveo.github.io/search-ui/interfaces/iquery.html (no date; fetched 2026-11-02 by the verifier); https://docs.coveo.com/en/1445/ (503) |
| Voyado | Not checked | Not checked | Not checked | [GAP2 open questions] |

- Every engine with a fetched page takes free text plus a structured filter, each with its own price-range syntax. So B needs a compiler per engine (inference) [ARCH ARC-31].
- "Not checked" means no page that would document a rerank hook was fetched. It does not mean the hook is absent [ARCH-V missing].
- The router or toggle shape is established: Algolia, Google, Empathy and commercetools all document one (documented, sources above).
- That also means some incumbents already sell the feature CAS would add (inference).

### The P0 incumbents, and which adapter to build first

| Account | Search incumbent today (documented, fetched 2026-11-02) | What it means for an overlay |
|---|---|---|
| Kroger | Empathy.co serves all search, navigation and keyword traffic on Kroger's web and apps, since 2020. Kroger operates it itself as a PaaS (platform-as-a-service). Its component map lists a hybrid index and vector search, diet facets (2021) and 5-minute index updates [https://empathy.co/assets/media/uploads/resources/kroger-building-and-evolving-ai-search-with-empathy.pdf, vendor case study, undated, data to June 2025]. Kroger's Elastic link is monitoring, not search [https://www.elastic.co/industries/retail-ecommerce] | The only named engine that the retailer operates itself: the first adapter [GAP2 G2-24, G2-25]. CAS adds query understanding into typed filters (and reranking), not vector retrieval. Not yet known: whether Empathy accepts injected filters or an external reranker, and whether Kroger uses Empathy's own AI Mode |
| H&M | Voyado: onsite search rebuilt across 60+ markets [https://retailtechinnovationhub.com/home/2026/9/27/handm-overhauls-its-onsite-search-across-more-than-60-markets-in-partnership-with-voyado, 28 Sep 2026]. H&M holds a minority stake in Voyado [https://www.retail-week.com/technology/handm-has-taken-a-minority-stake-in-a-swedish-retail-technology-platform/7047788.article, 2024-12-18] | CAS would compete with an intent-driven engine H&M rebuilt its search on this year (inference). Voyado's API is not checked. Do not budget an adapter until a buyer asks [GAP2 recommendations] |
| Albertsons | Google's Conversational Commerce agent runs in its Ask AI tool [https://www.googlecloudpresscorner.com/2025-09-10-Google-Cloud-Launches-Conversational-Commerce-Agent,-Delivering-AI-Enabled,-Personalized-Shopping-Experiences-for-Customers, 2025-09-10]. An AI shopping assistant on OpenAI models [https://www.albertsonscompanies.com/newsroom/press-releases/news-details/2025/Albertsons-Companies-Accelerates-Digital-Transformation-with-the-Albertsons-AI-Shopping-Assistant-Redefining-the-Grocery-Shopping-Experience/default.aspx, 2025-12-03]. Criteo ads inside its conversational search [https://www.grocerydive.com/news/albertsons-media-collective-criteo-ai-search-brand-ads-ecommerce-omnichannel/823557/, 2026-06-24]. Keyword engine unnamed: Bloomreach appears only on an undated aggregator page [GAP2 G2-07] | No named engine to target. The layer CAS would add has already been bought, twice (inference) |
| H-E-B (US) | Engine unknown. Built in-house by H-E-B Digital; per-store assortment comes from an in-house service that updates as shelf locations change [https://careers.heb.com/blog/simplifying-the-assortment-riddle, 2025-03-07]. Elasticsearch powered Central Market search in a talk whose URL carries 2019 (page undated) [https://www.elastic.co/elasticon/tour/2019/austin/elastic-at-h-e-b-providing-a-better-shopping-experience-at-central-market]. H-E-B Mexico, a separate business, runs Algolia [https://www.algolia.com/customers/h-e-b-mexico, since 2023] | An in-house team is a plausible D-lite partner (inference). If heb.com still runs Elasticsearch (the last public sign is from 2019), an Elasticsearch / OpenSearch adapter would be the second build (inference) |
| C&A (Europe) | Syte visual AI logo on Syte's fashion page [https://www.syte.ai/ecommerce-fashion/, page (c) 2024]. Text engine and commerce platform unknown [GAP2 G2-17, G2-18] | Ask first. Visual search is out of CAS scope, so Syte is complementary [CLAUDE.md:1332] |

What this means for the first adapter:
- Build for Empathy (Kroger) first. It is the one named engine the retailer runs itself [GAP2 G2-24].
- Before writing code, read Empathy's extension API or ask Kroger. The case study shows a self-operated platform, but no extension API was read [GAP2 open questions].
- Effort, by arithmetic on the lane's per-engine estimates (estimate): Kroger + H&M = 12-21 dev-days; all five accounts on five different engines = 21-39. The second figure is hypothetical: three of the five keyword engines are unnamed, and Albertsons' AI layer is a competing assistant, not an engine CAS would compile filters into [GAP2 G2-24].

Language and region limit B and C at H&M and C&A:
- CAS is English-only today (measured, offline probe on hand-written parallel English / German / Swedish text, no LLM; a small, illustrative sample).
  - CAS's query readers reacted to 10 of 10 English queries but 0 of 10 German and 0 of 10 Swedish ones.
  - The injection guard's input layer cut 2 of 2 English overrides but 0 of 2 German and 0 of 2 Swedish ones; those reach the LLM verbatim (the output screen still applies).
  - Variant collapse merged 2 of 2 English colour / size pairs, but 0 of 2 German and 0 of 2 Swedish [SCR/gtm/gap_4/out/lang_probe.json; GAP4 G4-16, G4-18, G4-19].
- Qwen3-Next, the Databricks LLM CAS uses today, is not listed in any Azure EU or UK region (documented) [https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability, ms.date 2026-10-30; GAP4 G4-01].
  - It is listed pay-per-token in AWS Frankfurt, Ireland, London and Paris. On Google Cloud the region table and the Designated Services page disagree [GAP4 G4-04, G4-06].
  - It is in Public Preview. No region offers it under provisioned throughput, and Private Link covers only provisioned-throughput endpoints. So a client that needs private connectivity to the LLM cannot use Qwen3-Next today (documented + inference) [GAP4 G4-08, G4-09].
  - On Azure EU the documented in-region options are gpt-oss-120b and gpt-oss-20b [GAP4 G4-03]. gpt-oss-120b took 30.3 s per query against Qwen's 10.4 s, at tied quality (Databricks, whole pipeline, rerank on, keyword set, 2026-09-23) [CLAUDE.md:778-779]. gpt-oss-20b is unusable for rerank comparisons [.claude/rules/eval.md:179].
  - The other in-region candidate, Qwen3.5 122B, is a model CAS has never tested [GAP4 G4-12].
- Multilingual search is out of scope [CLAUDE.md:1332]. So pilots there would be English-only (inference).

### Where each pattern can run

This is inference unless marked; the cloud section of this study covers it in full.
- B and C need an LLM endpoint the client allows and a small API. The LLM client speaks Gemini, OpenAI, Anthropic and OpenAI-compatible endpoints; Gemini (earlier), Groq and Databricks are the hosts used so far (code; documented (repo)) [CLAUDE.md:203, 791-793; SCR/gtm/research/claims.json CLA-56, CLA-70]. So B and C could run as a container in the client's own cloud.
- Two limits today (code). The API loads the whole catalog and keyword index at start-up even when only B or C would be used, so a catalog-free service is a build item, not a setting [app/main.py:161-193]. And CAS has only ever run on a Windows laptop and one Azure Databricks workspace, never on AWS, Google Cloud or a client's cloud [SCR/gtm/research/claims.json CLA-56].
- A, D-full and E need the compiled catalog plus an index: Databricks Vector Search, or the in-process local index [app/search.py:537-548]. That ties them to where the catalog copy lives.
- The code is licensed AGPL v3. Its network-use clause (section 13) matters if code is handed over to run in a client's cloud; that is a legal question, not answered here [LICENSE:1-2, :540; SCR/gtm/research/claims.json CLA-59].

### Effort, and which D-lite figure this section uses

The lane's effort table is inconsistent, and the verifier flagged it [ARCH-V missing].
- The table defines D-lite as "B + C" at 12-20 dev-days. Its own rows put B (9-15) + C (5-8) at 14-23 before any router work.
- Its Phase 1 recommendation defines D-lite as router + B, with C arriving in Phase 2.

How this section resolves it:
- This section uses D-lite = router + B.
- It reads the 12-20 figure as B's 9-15 plus ~3-5 for the router part: the adapter that recomputes coverage from the engine's top hit, router tests (none exist) and threshold calibration. This split is our reading; the lane does not break the figure down.
- Phase 2 adds C, so the tail work totals 17-28.
- The lane also lists ~3-5 days of items common to B-D: an API-only service with /api/ routes and service-principal auth, responses keyed by the client's product id, a per-engine constraint-to-filter compiler, and router tests. This section counts them inside B's 9-15, whose build list names the API, hosting, auth and compiler, and puts the router tests in the router part [ARCH tables[0], recommendations].
- All figures are developer time, not calendar time. The catalog figures exclude Databricks workspace runs and owner and legal sign-offs [BYO:417-421].

| Item | Dev-days (estimate; one developer who knows the repo) | Source |
|---|---|---|
| B, first engine: /api/understand, filter compiler, intent executor + merge, client prompt tuning and eval set, hosting and auth | 9-15 | [ARCH tables[0]] |
| B, each further engine | +3-6 | [ARCH tables[0]] |
| D-lite = router + B (Phase 1) | 12-20 (router part ~3-5, our reading) | [ARCH tables[0]], re-read here |
| C on the tail: /api/rerank by client id, field mapper, tail re-append (the re-append comes from the verifier, not the lane's list) | +5-8 (Phase 2 total 17-28) | [ARCH tables[0]; ARCH-V ARC-05] |
| D-full: router + compiled and indexed catalog | 30-45. Of this, the catalog-compiler phases P0-P3 are ~24 (4.0 + 7.75 + 7.0 + 5.5), or 25-27 with P0 at the safer 5-7 | [ARCH tables[0]; BYO:404-407, :412] |
| A full replacement | ~40+ (the lane's figure: the full catalog design ~36 + hosting 2-3), before any operating layer | [ARCH tables[0]; BYO:410] |
| E fusion | 30-45 | [ARCH tables[0]] |
| B adapters for Kroger + H&M / all five accounts | 12-21 / 21-39 (arithmetic; the second is hypothetical) | [GAP2 G2-24] |
| Per client, A / D-full / E only: review the catalog mapping and run the first build | ~1-2 days, plus the embedding run (~38 min per 300K texts on Databricks) | [BYO:443-444] |
| Phase 0 offline proof | 2-3 weeks of calendar time | [ARCH recommendations] |

### Recommended phases

0. Offline proof, about 2-3 weeks (estimate), no production traffic.
   - Take 30-50+ real head and tail queries from the client's logs.
   - Run three arms against a read-only sandbox of the client's engine API: (i) the engine on the raw query; (ii) the engine on CAS's intents and filters (B); (iii) B plus CAS rerank (C).
   - Score them with a context eval set the client's merchandisers write.
   - Replay the router offline to size the tail share, latency and token bill.
   - Prerequisites: the client agrees that query text (and, for C, product text) may go to the chosen LLM host.
   - Exit: B beats the raw query on the tail and leaves the head unchanged.
1. D-lite behind a client flag.
   - Head queries stay on the engine, untouched.
   - Low-coverage, zero-result or long natural-language queries go to a new /api/understand endpoint (to be built), then to the client's own engine.
   - The client sets a deadline. On a timeout or error it serves its own raw results (fail-open), because CAS sets no LLM timeout of its own.
   - Expected added LLM time on the tail: ~0.4-0.6 s for the translate call on Groq qwen3.8-27b (p50-p90) [LAT]. The router's engine call and the engine's intent searches come on top, and no overlay has been timed end to end. It is unmeasured on Databricks, so do not quote it for a Databricks deployment.
2. Add C on the tail's top 15.
   - Return scores and reasons keyed by the client's product id, and re-append everything the reranker did not return.
   - Agree first who wins when CAS's order conflicts with the client's pins, boosts or sponsored slots.
3. D-full only where it earns its cost.
   - Compile and index the catalog only where recipe / setup bundles, the type-aware ordering, cross-sell and upsell show a measured gain online. That is most likely fashion (inference).
   - Plan one catalog per market: CAS holds one currency per catalog [BYO:460], and H&M sells online in 62 markets [https://hmgroup.com/?p=311, figures as of 31 Aug 2026].

Not recommended for the P0 accounts: A (full replacement) and E (fusion).

### Open questions for this area

- What share of a client's real traffic would the router send to CAS? Our query sets are not traffic-weighted. The answer sets the latency budget, the token bill and whether pay-per-token limits suffice.
- How much does B lift a real engine (Empathy, Algolia, Elasticsearch, Bloomreach) that already has synonyms, typo tolerance, merchandising rules and, at Kroger, vector search? Our stand-in is plain BM25.
- How much does the real LLM reranker gain on a client engine's list? Only a perfect-rerank ceiling is known.
- Who wins when CAS's rerank order conflicts with the client's pins, boosts or sponsored slots?
- May query text (B) and product text (C) go to an LLM, and on which host? Qwen3-Next is not listed in any Azure EU region. On Azure EU the tested in-region option is gpt-oss-120b, about 3x slower on the keyword set [CLAUDE.md:778-779; GAP4 G4-01].
- Engine hooks still unread: does Empathy accept injected filters or an external reranker, and does Kroger already use Empathy's AI Mode? Is Voyado's API open to either? Coveo's hooks are unverified (its docs returned 503).
- Can Elasticsearch's custom-inference rerank path reach a Databricks-hosted CAS? An App's service-principal token lasts 1 hour [CLAUDE.md:1766], while an inference endpoint is configured with static headers (inference, unverified) [ARCH-V missing]. Would that path keep the per-item reasons?
- Which router signal works on client logs: 75% term coverage (5 of 30 misrouted here), zero results, word count, or an opt-in button as in Empathy's AI Mode?
- How many workers or App instances does an API-only B / C service need per client query rate? No load test exists, and the API runs one worker today.
- Are the prompt's Amazon-tuned clothing, device and gift rules right for each client's assortment, or does each client need its own prompt variant and eval set?
- Should results be served per SKU, per style-colour or per title (an open owner question that also affects D-full sizing) [BYO:461]?

## 8. Cloud agnosticism, security, region and language fit

Answer. Not today. CAS is a portable Python core (the API, the ranking logic and the UI) with one working cloud deployment, on Databricks (one Azure workspace), plus a laptop mode that needs Python and an OpenAI-compatible LLM endpoint (Groq in the US by default; run on one Windows laptop). Databricks itself runs on AWS, Azure and Google Cloud, so the Databricks version can follow a client to any of the three (only Azure has been run), but that is a dependency on Databricks, not independence from it. Running natively on AWS, Azure, Google Cloud, Snowflake or a client's Kubernetes needs a container image (none exists), API authentication and per-cloud adapters: about 6-10.5 dev-days for a portable package and about 20-28 for the first cloud-native target (estimates). Two further limits matter for the P0 accounts: the LLM we chose on Databricks (Qwen3-Next, a Public Preview model) is not listed in any Azure EU or UK region, and every word list, tokenizer, prompt, embedding model and eval set in CAS is English.

How to read the sources. Each fact carries a tag: measured (we ran it), code (read in the repo), documented (a vendor page, or the repo's own notes where marked "repo"), estimate, or inference. Repo paths are relative to the Context-Aware-Search repo. SCR = the 2026-11-02 session archive (see "How to read this"). Nothing about scale has been measured on Databricks, and no load test exists anywhere in the repo.

### 8.1 What runs where today

The portable core. The API (FastAPI + uvicorn), data handling (pandas, numpy, scikit-learn), the local embedder (sentence-transformers with gte-small), the LLM client (the OpenAI SDK pointed at an OpenAI-compatible endpoint), the keyword engine (BM25, a standard keyword-ranking formula, written in numpy/pandas), the result cache, the SQLite reply memo, optional MLflow tracing and the Streamlit UI use no cloud SDK [code; requirements.txt:1-14; app/keyword_search.py:4-10; app/page_cache.py:4-11; app/llm_memo.py:216-232; app/tracking.py:25-34]. The root requirements also install Google's google-genai and the Anthropic SDK, used only by the optional Gemini and Anthropic LLM backends [code; requirements.txt:8, 13; SCR/gtm/research/cloud_ops.json CLO-10].

The six Databricks-specific pieces [code; verified 2026-11-02, SCR/gtm/research/cloud_ops.verify.json CLO-01..07]:

| # | Databricks piece | What it does for CAS | Where in code |
|---|---|---|---|
| 1 | Vector Search, called over its REST API through databricks-sdk | Holds the product vectors; answers each search phrase with (row number, score) | app/vector_search.py:36, 45-55, 240-274 |
| 2 | Embedding endpoint databricks-gte-large-en | Turns every query, and once the whole catalog, into a 1,024-number vector | app/vector_search.py:204-237; app/config.py:279 |
| 3 | Unity Catalog (UC, Databricks' data catalog) Volumes and Spark | Store the catalog file and the LLM reply memo; an App reads them through the Files API | app/vector_search.py:106-161; app/llm_memo.py:335-399, 567-578 |
| 4 | OPENAI_AUTH=databricks | Asks databricks-sdk for a workspace token before every LLM call (the SDK caches it and refreshes it before expiry; an App's token lasts 1 hour) | app/config.py:67-73; app/llm_client.py:269-298 |
| 5 | Databricks Apps launcher and bundle | Starts API + UI on the App's port; sets the backend, models and memo folder | databricks/app_bundle/run_app.py:51-96; databricks/app_bundle/app.yaml:21-60 |
| 6 | Operations notebooks 01, 02, 09, 11, 12 and UC grants | Load the catalog, embed it, build the index, deploy, grant access, stop when idle | databricks/09_deploy_app.py:394-468, 526-580, 741; databricks/11_idle_guard.py:1-19 |

One setting, SEARCH_BACKEND ('local' by default, or 'databricks'), switches retrieval and the catalog load [code; app/config.py:262; app/search.py:308-322, 537-548]. The memo's Volume store (an LLM_MEMO_DIR under /Volumes) and the token callable (OPENAI_AUTH=databricks) have their own settings and also import the Databricks module, and the start-up health checks of index, embeddings and LLM run only on the Databricks backend [code; SCR/gtm/research/cloud_ops.verify.json CLO-01; app/main.py:173-176]. The LLM client sends a fixed seed (42) whenever DETERMINISTIC is on (the default) and no temperature sweep is running, unless the base URL contains 'serving-endpoints' (Databricks) [code; app/config.py:107-108; app/llm_client.py:196-203]. A new provider that rejects the seed would make every LLM stage fail and fall back to plain embedding search, because there is no learn-once fallback for the seed as there is for JSON mode and temperature [code; app/llm_client.py:167-174, 363-372].

Two shapes run today:

| Shape | What it uses | Where it has run | Status |
|---|---|---|---|
| Local backend (default) | gte-small embeddings in the process, a numpy vector index in memory, SQLite memo, an OpenAI-compatible LLM whose default base URL is Groq (US) | The dev laptop (Windows, Python 3.13.7) | Measured: every recent quality number (Groq models). Estimated at 300K rows on the laptop: ~2.3 GB memory per process and ~109 s start-up when the taxonomy cache misses (~50 s on a warm cache per CLAUDE.md) [estimate; SCR/gtm/research/probe_scalability.json, derived.per_catalog_size.300K; CLAUDE.md:13]. A new catalog took ~24 h to encode on CPU for 300K rows [measured once; CLAUDE.md:12-13] |
| Databricks backend | Vector Search, gte-large-en, Qwen3-Next through the Foundation Model APIs (FM APIs, Databricks-hosted models), a Databricks App | One Azure Databricks workspace (catalog `dev`); region not recorded | The LLM acceptance smoke ran 2026-10-12; the App was created 2026-10-13 and showed the 2026-10-14 demo; the current version (with the memo) is not yet deployed; the App's memory and start-up on 2 vCPUs and its Python 3.11 runtime are untested [measured / documented (repo); CLAUDE.md:783-787, 1686, 1785, 1796, 1805; SCR/gtm/research/cloud_ops.verify.json CLO-13]. Databricks on AWS or Google Cloud never run |

Databricks across clouds. In Databricks' own words, "Databricks is designed to work seamlessly across AWS, Microsoft Azure, and Google Cloud" [documented; https://www.databricks.com/en/company/partners/cloud-partners, fetched 2026-11-02]. So the Databricks shape can be offered to a client on any of the three, subject to model availability in their region (8.7). It still needs Databricks.

No container. There is no Dockerfile, compose file, Containerfile, Helm chart, Terraform, or CI workflow in the repo (search on 2026-11-02, 0 matches; no .github folder) [code; SCR/gtm/research/cloud_ops.verify.json CLO-12; repo listing 2026-11-02]. The only deployable package is the Databricks App bundle. A container image is an estimated 2-3 dev-days; its size is unmeasured, roughly 0.6-0.8 GB for a 'remote' image (models and vectors in a managed service) and 1.5-2.5 GB for a 'local' image with torch and gte-small, plus 0.46 GB if the 300K-row vector file is baked in [estimate; SCR/gtm/research/cloud_ops.verify.json CLO-14].

Suggested collateral wording: "a cloud-portable core with a Databricks-native reference deployment (Databricks runs on AWS, Azure and Google Cloud)". Do not write "cloud agnostic" or "runs on any cloud" until the container (C1), parameter negotiation (C3) and API authentication (C10) ship and one non-Databricks LLM passes the acceptance smoke (C4) [inference; SCR/gtm/research/cloud_ops.json recommendations]; even then, name the platforms actually run rather than claim "any cloud".

### 8.2 Portability matrix

Legend. Works today = built and run by us; in the On-prem / Kubernetes column that means on the dev laptop only, never in a container or on Kubernetes. Config = a documented endpoint fits existing settings; CAS has never called it. Code change (Cn) = an edit to existing modules. New adapter (Cn) = a new module behind an interface. Effort per Cn is in 8.3. Every non-Databricks cell is untested.

| Component | Databricks (on AWS, Azure or Google Cloud) | AWS native | Azure native | Google Cloud native | Snowflake | On-prem / Kubernetes | Sources |
|---|---|---|---|---|---|---|---|
| LLM (translate, rerank, cross-sell, assistant) | Works today on Azure Databricks: Qwen3-Next FM API, 08 smoke 2026-10-12. Region limits: 8.7 | Config only with a long-term Bedrock API key (AWS: "exploration only"); production short-term keys (up to 12 h) or SigV4 (AWS request signing) need C2; seed / JSON mode unknown: C3 | Config with an Azure OpenAI key; Entra ID (Microsoft sign-in) tokens need C2 | Code change C2: Vertex AI's OpenAI-compatible endpoint takes service-account tokens that last 1 hour by default | Config: Cortex chat completions with a programmatic access token; JSON mode handling for non-OpenAI models: C3; a non-Cortex LLM needs an External Access Integration | Config: vLLM's OpenAI-compatible server (Groq works today on the laptop) | CLAUDE.md:783-787; https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html; https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html; https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle (ms.date 2026-05-13); https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate/openai/auth-and-credentials; https://docs.snowflake.com/user-guide/snowflake-cortex/open_ai_sdk; https://docs.vllm.ai/en/latest/serving/online_serving/ (all fetched 2026-11-02) |
| Embeddings (query and catalog) | Works today: databricks-gte-large-en | C6: Titan Text Embeddings v2 (1,024-d); full re-embed | C6: Azure OpenAI embeddings; re-embed (no Azure embeddings page was fetched) | C6: text-embedding-005 / gemini-embedding; re-embed | C6: EMBED_TEXT_1024 / AI_EMBED, or Cortex Search embeds itself | Works today: gte-small on CPU; or a Text Embeddings Inference server via C6 | app/vector_search.py:204-237; https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html; https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings; https://docs.snowflake.com/en/user-guide/snowflake-cortex/vector-embeddings; https://huggingface.co/docs/text-embeddings-inference/index |
| Vector store | Works today: self-managed Vector Search index | New adapter C5: OpenSearch k-NN or pgvector (RDS / Aurora) | C5: Azure AI Search or pgvector | C5: Vertex Vector Search, BigQuery vector search or pgvector | C5: Cortex Search, or VECTOR type + VECTOR_COSINE_SIMILARITY | Works today: in-memory numpy index at 300K rows; C5 pgvector beyond that | app/search.py:297-328; https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html; https://learn.microsoft.com/en-us/azure/search/vector-search-overview (ms.date 2026-08-05); https://docs.cloud.google.com/bigquery/docs/vector-search-intro; https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview; https://github.com/pgvector/pgvector |
| Catalog storage and build | Works today for OUR 13-column CSV only: Volume -> 01 (Delta) -> 02 (embed, index) -> 09 (export). Client tables need the catalog compiler | C9 (s3:// paths) + C7 build CLI | C9 (abfss:// / Blob) + C7 | C9 (gs://, BigQuery export) + C7 | C9 (stage file) + C7, or a Cortex Search service | Works today: local CSV + scripts/encode_catalog.py | databricks/README.md:40-47; SCR/oct27/byo/REPORT.md (sections 1, 6); app/config.py:11-14, 167 |
| Hosting | Built as a Databricks App; an earlier version was created 2026-10-13 and showed the 2026-10-14 demo, the current one is not deployed; workspace users only | C1 container on ECS Fargate | C1 on Azure Container Apps | C1 on Cloud Run (keep 1 warm instance) | C1 on Snowpark Container Services (SPCS) | C1 + Helm chart (C13) | databricks/app_bundle/run_app.py; CLAUDE.md:1686, 1785, 1796; https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html; https://learn.microsoft.com/en-us/azure/container-apps/overview; https://docs.cloud.google.com/run/docs/overview/what-is-cloud-run; https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview |
| Sign-in (people) / API auth (systems) | People: works today via workspace single sign-on (SSO); no anonymous access. API: not exposed (bound to 127.0.0.1 inside the App); exposing it needs C10 | People: config, ALB OpenID Connect (OIDC) or Cognito. API: C10 | People: config, Container Apps "Easy Auth". API: C10 | People: config, Identity-Aware Proxy. API: C10 | People: Snowflake identities only. API: C10 | People: config, oauth2-proxy. API: C10 | https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth (updated Oct 9, 2026); databricks/app_bundle/run_app.py:89-91; https://docs.aws.amazon.com/elasticloadbalancing/latest/application/listener-authenticate-users.html; https://learn.microsoft.com/en-us/azure/container-apps/authentication; https://docs.cloud.google.com/iap/docs/enabling-cloud-run; https://docs.snowflake.com/en/release-notes/bcr-bundles/2024_04/bcr-1611; https://oauth2-proxy.github.io/oauth2-proxy/ |
| LLM reply memo (repeatable results) | Built (UC Volume, one JSON file per entry); not yet run on the workspace App; grants unverified | C8: S3, or Redis / Postgres | C8: Blob, or Redis / PostgreSQL | C8: GCS, or Memorystore / Cloud SQL | C8: stage or table | Works today: SQLite on the laptop; on Kubernetes it would need a persistent volume and one replica (untested) | app/llm_memo.py:335-399, 567-578; databricks/app_bundle/app.yaml:57-60; CLAUDE.md:1785-1790 |

Notes on the matrix:
- LLM. None of the endpoints named in this row (Bedrock, Azure OpenAI, Vertex AI, Snowflake Cortex, vLLM) has been called by CAS. Besides Databricks, CAS has run on Groq and, earlier, on Google's Gemini API through its own SDK [documented (repo); CLAUDE.md:791-793, 1023]. Each new endpoint needs a ported acceptance smoke (C4: 43 calls, ~55K tokens per run, estimated) including the injection rows before we name it in collateral [estimate; CLAUDE.md:1663-1664; SCR/gtm/research/cloud_ops.verify.json CLO-27]. Bedrock is not config-only in production: AWS calls long-term API keys "Recommended only for exploration" and short-term keys last up to 12 hours, so AWS needs the same token-provider change (C2) as Google Cloud and Entra ID [documented; https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html, fetched 2026-11-02]. Snowflake's page says Claude models accept only a JSON-schema response format and other non-OpenAI families ignore it; CAS sends the simpler JSON-object format [documented; https://docs.snowflake.com/user-guide/snowflake-cortex/open_ai_sdk, fetched 2026-11-02], and its JSON-mode fallback only recognises an error that names 'json_object' and 'not supported', so whether a Cortex refusal triggers it is untested [code; app/llm_client.py:167-169; SCR/gtm/research/cloud_ops.verify.json missing items].
- Embeddings. The local embedder's OpenAI path triggers only for model names starting 'text-embedding-' and has no base-URL setting of its own, so the OpenAI SDK sends it to OPENAI_BASE_URL, the chat endpoint (Groq locally). A managed embedding endpoint is not cleanly selectable today, and a name like Vertex's 'text-embedding-005' would be misrouted [code; app/embeddings.py:22-23, 35-40; SCR/gtm/research/cloud_ops.verify.json CLO-11]. Any model switch means re-embedding the catalog.
- Vector store. That each cloud's managed Postgres offers pgvector was not fetched (inference) [SCR/gtm/research/cloud_ops.verify.json CLO-34].
- Hosting. Scale-to-zero hosts (Cloud Run, Container Apps) clash with CAS's start-up: ~50 s per CLAUDE.md, ~61 s (Databricks path) to ~109 s (local path, taxonomy-cache miss) per the 2026-11-02 laptop probe; the runbook says the App's backend needs ~30-60 s to load; none of this is measured on 2-vCPU hosts. Keep one warm instance (inference) [CLAUDE.md:13; SCR/gtm/research/probe_scalability.json; databricks/README.md:499-500]. Each process holds ~1.7 GB (Databricks path) or ~2.3 GB (local path) at 300K rows, plus ~0.7 GB transient per in-flight local search (estimates, same probe file). Caches and the single-flight lock live in one process, so replicas do not share them [code; app/page_cache.py:71-93; app/main.py:1453].
- Snowflake. SPCS endpoints are private by default; a public one needs the BIND SERVICE ENDPOINT privilege and admits only users of that Snowflake account whose role holds USAGE on the endpoint. Containers have no internet access without an External Access Integration. Ingress closes connections idle for 90 s; the UI's calls stay inside the container, so only an external API caller on a slow request is exposed (inference, untested) [documented; https://docs.snowflake.com/en/developer-guide/snowpark-container-services/service-network-communications and https://docs.snowflake.com/en/release-notes/bcr-bundles/2024_04/bcr-1611, fetched 2026-11-02]. Streamlit in Snowflake states no supported Streamlit version (CAS pins 1.58), and that it cannot also host the FastAPI half is an inference [documented/inference; https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit; requirements.txt:9].

### 8.3 Effort to make CAS portable (estimates, not measured)

One developer who knows the repo, tests included [estimate; SCR/gtm/research/cloud_ops.json, table "Code changes"; sums checked in SCR/gtm/research/cloud_ops.verify.json].

| Id | Change | Dev-days | Needed for |
|---|---|---|---|
| C1 | Container image in two flavours (remote / local), generic entrypoint, non-root user, health probe, .dockerignore | 2-3 (a dependency lock step and an image vulnerability scan are not costed) | every non-Databricks target |
| C2 | Token providers beyond 'databricks' (Entra ID, Google credentials, Bedrock short-term keys or SigV4) | 1-1.5 | Azure Entra, Google Cloud, AWS production |
| C3 | Learn-once handling of a refused seed and of JSON-object vs JSON-schema formats | 0.5-1 | any new LLM provider |
| C4 | Provider-neutral acceptance smoke (port of databricks/08, injection rows included) | 1 (+ ~55K tokens per provider run) | before naming any provider |
| C5 | Vector-store interface + first adapter (pgvector); then each managed store | 3-4 first; 2-3 each further | cloud-native retrieval at scale |
| C6 | Embedding client with its own base URL / key, batch and query, normalisation | 1.5-2 | managed embeddings off Databricks |
| C7 | Build command off Databricks: encode -> vector store -> index load -> catalog snapshot | 3-4 | catalog refreshes off Databricks |
| C8 | Memo backend on object storage or Redis / Postgres, shared by replicas | 2-3 | repeatable results on serverless containers |
| C9 | Object-store paths and env settings for the data, cache, log and sponsored-ads paths | 1-1.5 | read-only container images |
| C10 | API authentication (API key or OIDC token check), CORS, rate limiting | 2-3 | the "API inside the client stack" delivery |
| C11 | Backend-neutral start-up checks and health probes | 0.5 | operations on every target |
| C12 | Shared result caches across replicas (optional) | 2 | horizontal scale |
| C13 | Infrastructure as code: Helm chart; Terraform / Bicep per cloud; SPCS spec | Helm 2; 2-4 per cloud; SPCS ~2 | repeatable installs |
| C14 | Per-cloud cost guard (instance limits, budgets, index teardown) | 0.5-1 per cloud | cost control off Databricks |
| C15 | OpenTelemetry / log shipping (optional) | 1 | client observability stacks |
| Sum A | Portable package: C1 + C3 + C9 + C10 + C11 (+ C2) | 6-9 (7-10.5 with C2) | a standard container host, in-memory index, an OpenAI-compatible LLM |
| Sum B | First cloud-native target: Sum A + C4 + C5 + C6 + C7 + C8 + one C13 | ~20-28 (the AWS example needs C2, so its upper end is ~28.5) | e.g. AWS with pgvector or OpenSearch + Bedrock |
| Sum C | Each further cloud: one C5 adapter + C13 + a C4 run + C14 | ~5-9 | |

Excluded from every sum: the client catalog compiler (compiler P0 is 4 dev-days optimistic, 5-7 safer; the full design ~36) [estimate; SCR/oct27/byo/REPORT.md:47-51, 404-412], a security review, legal work on licences, LLM token budgets for evals, and runs on client workspaces.

### 8.4 Where the target accounts are known to run

Only one account has a fetched source tying its search to a cloud: Albertsons' conversational search uses Google's agent built on Vertex AI. For the other four, no fetched source names the cloud under search. What is public [documented; SCR/gtm/research/connectors.json CON-40..44 and SCR/gtm/research/gap_2.json, fetched 2026-11-02]:

| Account | Public cloud / data platform signals | Source (date) |
|---|---|---|
| Kroger | Azure "preferred cloud platform for Retail as a Service" (scoped to that programme, not Kroger's cloud overall); Google Cloud analytics and AI tools; subsidiary 84.51° an early Databricks Lakehouse for Retail adopter | https://news.microsoft.com/source/2019/01/07/kroger-and-microsoft-partner-to-redefine-customer-experience-introduce-digital-solutions-for-retail-industry/ (Jan 7, 2019); https://www.googlecloudpresscorner.com/2023-01-20-Google-Cloud-and-Deloitte-Boost-Grocery-Associate-Productivity-and-Improve-the-Customer-Experience (Jan 20, 2023); https://www.databricks.com/company/newsroom/press-releases/databricks-launches-data-lakehouse-for-retail-and-consumer-goods-customers (Jan 13, 2022) |
| Albertsons | Azure named its preferred cloud, eCommerce sites moved to Azure; its Ask AI conversational search runs Google's Conversational Commerce agent, built on Vertex AI, and its own release says its AI shopping assistant is powered by OpenAI models | https://news.microsoft.com/source/2019/02/22/albertsons-companies-to-transform-experiences-for-shoppers-with-microsoft-cloud-and-ai/ (Feb 22, 2019); https://www.googlecloudpresscorner.com/2025-09-10-Google-Cloud-Launches-Conversational-Commerce-Agent,-Delivering-AI-Enabled,-Personalized-Shopping-Experiences-for-Customers (Sep 10, 2025); SCR/gtm/research/gap_2.json G2-06 (Albertsons release, Dec 3, 2025) |
| H-E-B | Unknown; it builds its digital stack in-house (H-E-B Digital) | SCR/gtm/research/connectors.json CON-44; SCR/gtm/research/gap_2.json G2-11 |
| H&M | A Google Cloud partnership to build an "enterprise data backbone including a core data platform"; named among early adopters of Databricks' Lakehouse for Retail, cloud not stated; a 2022 H&M job ad for its brand.com platform asked for "cloud solutions (preferably Azure)" | https://www.googlecloudpresscorner.com/2022-06-30-Google-Cloud-Announces-New-Partnership-with-Global-Fashion-Retailer (Jun 30, 2022); https://www.databricks.com/company/newsroom/press-releases/databricks-launches-data-lakehouse-for-retail-and-consumer-goods-customers (Jan 13, 2022); https://webbjobb.io/jobb/208490/software-engineer-solution-architect (job ad Apr 1, 2022; SCR/gtm/research/gap_2.json G2-15); SCR/gtm/research/connectors.verify.json CON-40 |
| C&A | Snowflake is its central data warehouse; cloud underneath not stated | https://www.snowflake.com/en/blog/c-and-a-build-data-driven-fashion-empire/ (Aug 17, 2023) |

Inference: Azure and Google Cloud each appear for three accounts (Azure: Kroger, scoped to one programme; Albertsons; an H&M job ad. Google Cloud: Kroger, Albertsons, H&M), Databricks for two (Kroger's 84.51° and H&M; the clouds under them not stated) and Snowflake for one (C&A). So the Azure Databricks shape is the likeliest ask, Google Cloud is at least as plausible as Snowflake, and an SPCS + Cortex shape fits C&A. This must be confirmed in discovery.

### 8.5 Security and enterprise checklist

In place:

| Item | Status | How known | Source |
|---|---|---|---|
| No personal data in logs | No query, reply, title or memo key in any log line; request logs carry path and counts; the App runs uvicorn with no access log | code | .claude/rules/safety.md:11-16; app/main.py:216-224; databricks/app_bundle/run_app.py:89-92 |
| Prompt-injection guard (direct, via the query) | Cut before and screen after every LLM call; on Qwen3-Next every injection row of 08 passed (2026-10-12), with one warning on the marker-free class; rows added since 2026-10-15 not re-run on Qwen; the input cut is English-only (German and Swedish overrides pass it, 8.8) | measured | app/config.py:94; CLAUDE.md:758-776, 1681; SCR/gtm/research/gap_4.json G4-19 |
| Token accounting and health | Every response carries X-LLM-Calls, -Prompt-Tokens, -Completion-Tokens, -Memo-Hits; /healthz reports start-up checks, config and memo counters | code | app/main.py:210-215, 228 |
| Secrets | Read from the environment; the App's service principal is injected, and a workspace token is fetched from databricks-sdk before each LLM call (no DATABRICKS_TOKEN in app.yaml) | code | app/config.py:67-73; app/llm_client.py:269-292; databricks/app_bundle/app.yaml:13-18 |
| Sign-in for people on Databricks | Workspace SSO; the App acts as its own service principal (no per-user access control) | documented | https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth (updated Oct 9, 2026) |
| Cost controls | Reply memo (a repeat search made 0 LLM calls: 6 of 6 queries, 3 runs, 2026-10-30, local with Groq qwen3.8-27b), result caches, idle teardown notebook | measured / code | .claude/rules/eval.md (Determinism check); app/page_cache.py:71-114; databricks/11_idle_guard.py:1-19 |

Missing or open:

| Gap | Why it matters | How known | Source |
|---|---|---|---|
| The API has no authentication (no API key, OAuth or CORS code) | The agreed delivery is an API inside the client's stack; C10 before any pilot | code (0 matches, 2026-11-02) | app/main.py:198-237 |
| Indirect injection is open | Seller titles and descriptions enter the rerank and cross-sell prompts unscreened; non-English and marker-free overrides are residual | documented (repo) | CLAUDE.md:770-776; .claude/rules/safety.md:139 |
| Memo keys are plain sha256 hashes; stored values echo shopper text | Anyone who can list the store can confirm a guessed query; replies are shopper data | code | app/llm_memo.py:167, 175; .claude/rules/safety.md:168-181 |
| Our `dev` catalog is not least-privilege | All account users hold broad rights there (whether they can read or write the memo volume is unverified); a client needs its own catalog / schema with write access only for the app identity | documented (repo) | CLAUDE.md:1796-1804; SCR/gtm/research/cloud_ops.verify.json CLO-43 |
| Demo data is research-only | Amazon Reviews 2023 has no licence assigned; 299,932 of 299,972 rows hotlink Amazon images and sponsored.json (Amazon ids and titles) sits in git history; client deployments use the client's catalog | documented | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1 (Mar 8, 2024); .claude/rules/safety.md:191; SCR/gtm/research/cloud_ops.json CLO-49 |
| Code licence is AGPL v3 (GNU Affero General Public License) | Handing a client an image or source is distribution, and section 13 covers network use: a legal question before packaging | code / inference | LICENSE:1-2 |
| The local default LLM is Groq in the US | By default Groq retains no inference data; what it keeps (logs up to 30 days when troubleshooting or investigating abuse) sits in US Google Cloud buckets; client data must never use the default | code / documented | app/config.py:66; https://console.groq.com/docs/your-data (fetched 2026-11-02) |
| No CI (automated test runs), no lockfile, no SBOM (software bill of materials), no image scan | Only the App bundle pins exact versions; the root requirements and requirements-databricks.txt use minimum versions, and the root file omits databricks-sdk and mlflow; tests run by hand | code | requirements.txt:1-14; requirements-databricks.txt:15-19; databricks/app_bundle/requirements.txt:12-29 |
| No restricted-category blocklist | Adult or weapon items in a catalog are not filtered | documented (repo) | .claude/rules/safety.md:37-39 |
| No erasure or retention design per platform | The memo can be purged by query, stage or age; nothing maps data-subject requests or the feedback log to a client's storage | code / inference | CLAUDE.md:54; SCR/gtm/research/cloud_ops.verify.json (missing items) |
| No SLA or SOC 2 (security audit report) | Out of scope for the PoC (2026-08-24); competitors list them | documented (repo) | CLAUDE.md:1332; docs/COMPETITIVE_SUMMARY.md:53 |

### 8.6 Shopper traffic, sign-in and private networking

Shoppers cannot reach CAS anonymously on either managed platform. Databricks: "You can't make Databricks apps public" and anonymous access is not supported [documented; https://docs.databricks.com/aws/en/dev-tools/databricks-apps/permissions, updated Sep 11, 2026]. Snowflake SPCS: only users of the Snowflake account whose role holds USAGE on the endpoint can use a public endpoint [documented; https://docs.snowflake.com/en/release-notes/bcr-bundles/2024_04/bcr-1611, fetched 2026-11-02]. So on both, the retailer's own website backend must call CAS with a token, and CAS answers it; the shopper never talks to CAS directly (inference). Today the Databricks App exposes only the Streamlit UI; the API listens on 127.0.0.1 inside it [code; databricks/app_bundle/run_app.py:89-91]. Exposing the API to a retailer's backend is a change (C10 plus an App change) and has not been tried.

Private networking. On Databricks, "Private connectivity (such as Azure Private Link) is only supported for model serving endpoints that use provisioned throughput or endpoints that serve custom models" [documented; https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/compliance, ms.date 2026-10-28]. Provisioned throughput (PT) means reserved model capacity, priced in DBU (Databricks' billing unit) per hour per 50 model units; CAS uses pay-per-token (PPT) [documented; https://www.databricks.com/product/pricing/foundation-model-serving, fetched 2026-11-02, per SCR/gtm/research/gap_1.json G1-15]. No region on any of the three clouds lists Qwen3-Next under PT [documented; the three model-region pages in 8.7, dated 2026-10-30]. So a client that requires Private Link to the LLM cannot use Qwen3-Next on Databricks today (inference from those two facts). Among the PT families the EU rows list (they list others too, such as Llama, Gemma and BGE), the ones that matter for CAS are GPT OSS 120B / 20B and GTE v1.5 (English), plus Qwen3.5 122B (preview) on AWS EU; on Google Cloud's europe-west2/3 those PT entries carry the cross-Geo mark [documented; SCR/gtm/research/gap_4.json G4-03, G4-05, G4-08]. Whether the per-search query embedding on pay-per-token gte-large-en also falls under a client's private-connectivity rule is open. Private networking for Vector Search, Apps and the non-Databricks targets was not researched.

Production tier. Databricks' FM API overview recommends priority pay-per-token or provisioned throughput for production [documented; quoted in SCR/gtm/research/claims.verify.json, missing items, ms.date 2026-10-30]; the priority tier's model list includes neither Qwen3-Next nor gpt-oss (its only Qwen entry is Qwen3.5 122B, a West Europe-only preview) [documented; https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/priority-mode, ms.date 2026-10-26; SCR/gtm/research/gap_1.json G1-20]. CAS runs on standard pay-per-token. Inference: production, Private Link and Azure EU all point the same way, to gpt-oss on PT. CAS has never run on PT, and has not run gpt-oss on Databricks since 2026-09-23. One always-on 50-unit gpt-oss-120b block lists at $3,650/month (East US, $0.07/DBU, 730 h), and tokens per second per model unit is not documented for CAS's models, so how many blocks CAS would need is unknown [documented price + arithmetic; SCR/gtm/research/gap_1.json G1-15, G1-17].

### 8.7 Region fit

Where the models are listed (Databricks model-region pages, dated 2026-10-30, fetched 2026-11-02: https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability, https://docs.databricks.com/aws/en/unity-gateway/model-region-availability, https://docs.databricks.com/gcp/en/unity-gateway/model-region-availability; region-by-region extracts in SCR/gtm/gap_4/out/ and SCR/gtm/research/gap_4.json G4-01..06). "Cross-Geo" = the workspace may send model requests to another geography; storage stays home.

| Cloud | Regions | Qwen3-Next (what CAS uses) | gpt-oss-120b / 20b and gte-large-en | Databricks Geo page (Oct 26, 2026) |
|---|---|---|---|---|
| Azure | francecentral, germanywestcentral, northeurope, westeurope, swedencentral, switzerlandnorth | Not listed (pay-per-token or PT) | Listed pay-per-token and PT, no cross-Geo mark | EU Data Boundary: available |
| Azure | uksouth | Not listed | Listed, cross-Geo mark | United Kingdom: available only with cross-Geo |
| Azure | norwayeast, switzerlandwest, ukwest | No FM APIs at all | No FM APIs | - |
| Azure | 8 US regions (centralus, eastus, eastus2, northcentralus, southcentralus, westus, westus2, westus3) | Listed, no mark; no PT | Listed pay-per-token and PT in 7 of the 8; westus3 lists neither gte-large-en nor gpt-oss (no PT there) [SCR/gtm/gap_4/out/azure_matrix_all.txt; SCR/gtm/research/gap_1.json G1-16] | Americas: processed in the US |
| AWS | eu-central-1 (Frankfurt), eu-west-1 (Ireland), eu-west-2 (London), eu-west-3 (Paris) | Listed pay-per-token, no mark; no PT | Listed pay-per-token and PT, no mark | Europe: available |
| AWS | eu-north-1 (Stockholm) | Not a row of the table | Not a row | - |
| Google Cloud | europe-west1 (Belgium) | Not supported | Not supported | Europe: only with cross-Geo |
| Google Cloud | europe-west2 (London), europe-west3 (Frankfurt) | Listed pay-per-token, no mark; no PT | Pay-per-token no mark; PT (and batch AI Functions) with cross-Geo mark | Europe: only with cross-Geo (conflicts with the region table) |
| Google Cloud | europe-west4 (Netherlands) | Not a row | Not a row | - |

Geo-page sources: https://learn.microsoft.com/en-us/azure/databricks/resources/designated-services, https://docs.databricks.com/aws/en/resources/designated-services, https://docs.databricks.com/gcp/en/resources/designated-services (all Oct 26, 2026). Cross-Geo processing is on by default only for workspaces outside the US and the EU that have no compliance security profile [documented; https://learn.microsoft.com/en-us/azure/databricks/resources/databricks-geos, ms.date 2026-09-11], and EU Data Boundary workspaces rely on EU data centers for processing [documented; https://learn.microsoft.com/en-us/azure/databricks/resources/designated-services, Oct 26, 2026]. An EU retailer that requires in-Geo processing should keep it off (inference). Qwen3-Next 80B A3B Instruct is itself in Public Preview on all three clouds [documented; https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/supported-models, 2026-10-30].

What this means per account (inference from the table; the targets' clouds and regions are not known, 8.4):
- US grocery targets (Kroger, Albertsons, H-E-B): the current model is listed in-region on all three clouds (8 Azure US regions; AWS us-east-1, us-east-2, us-west-2; Google Cloud us-central1, us-east1, us-east4, us-west1) [documented; SCR/gtm/research/gap_4.json, table 2]. Two exceptions: Azure westus3 lists Qwen3-Next but not the gte-large-en embedding CAS needs for every query, and AWS us-west-1 lists no FM APIs [documented; SCR/gtm/gap_4/out/azure_matrix_all.txt, SCR/gtm/gap_4/out/aws_matrix_all.txt].
- An Azure-EU retailer (H&M in Sweden or C&A in Germany / the Netherlands, if they use Azure): Qwen3-Next is not available in-region, marked or not. The documented one-line switch is OPENAI_MODEL=databricks-gpt-oss-120b [documented (repo); CLAUDE.md:790]. On our Databricks run of 2026-09-23 (05, keyword set of 18 queries, rerank on, one run), gpt-oss-120b took 30.3 s per query against Qwen3-Next's 10.4 s, with NDCG 0.943 vs 0.953, a quality tie on a saturated set (the no-LLM control scored 0.954) [measured; CLAUDE.md:777-780, 1724-1730]. gpt-oss-20b failed too many reranks to use (on Groq: 5 of 16 context-set reranks fell back on 2026-10-14, 5 of 9 live reranks on 2026-10-15) [measured; .claude/rules/eval.md, App-path eval]. Nothing measured on Qwen3-Next carries over (and the recent graded numbers ran on Groq's qwen3.8-27b, the 2026-10-30 QE A/B on Groq's gpt-oss-20b: neither is a Databricks model); 08 (acceptance), 05 with its no-LLM control, 10 (page eval) and 12 (determinism) would need to be re-run on gpt-oss-120b in an EU workspace. Qwen3.5 122B A10B (preview) is listed in the EU rows of all three clouds (not Google Cloud's europe-west1) but CAS has never run it [documented; SCR/gtm/research/gap_4.json G4-12]. The embedding model gte-large-en is listed in-region, so retrieval needs no change.
- An AWS-EU retailer: Qwen3-Next is listed pay-per-token in Frankfurt, Ireland, London and Paris (not Stockholm). Outside Databricks, Bedrock also sells Qwen3 Next 80B A3B in Europe (Ireland, Milan) at $0.18 / $1.41 per 1M input / output tokens; its quotas were not checked and CAS has never called Bedrock [documented; https://aws.amazon.com/bedrock/pricing/, fetched 2026-11-02; SCR/gtm/research/gap_1.json G1-21].
- A Google Cloud-EU retailer: Belgium has no FM APIs; London and Frankfurt list Qwen3-Next, but the Geo page says Europe needs cross-Geo processing. Which page governs is a question for Databricks [documented; SCR/gtm/research/gap_4.json G4-06].
- Price: West Europe list prices for Databricks model serving are +20% over East US ($0.084 vs $0.07 per DBU), and +5% on Apps [documented; https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Azure Databricks' and armRegionName eq 'westeurope', fetched 2026-11-02; SCR/gtm/research/gap_1.json G1-33].
- Elsewhere: Asia Geo traffic can be processed in Japan, South Korea or Singapore; Australia / New Zealand and India only with cross-Geo; Mainland China and Middle East / Africa not available [documented; https://docs.databricks.com/aws/en/resources/designated-services, Oct 26, 2026, per SCR/gtm/research/cloud_ops.verify.json missing items]. On Azure, Qwen3-Next outside the US is listed only with the cross-Geo routing mark (Australia East, Brazil South, Canada Central, Central India, East Asia, Japan East, Korea Central, Southeast Asia) [documented; SCR/gtm/research/gap_4.json G4-02]. Our own workspace's region is not recorded.

LLM data terms by provider (fetched 2026-11-02):

| Provider | Training on customer data | Retention | Where processed | Source |
|---|---|---|---|---|
| Databricks FM APIs | No (paid accounts) | Up to 30 days in the workspace region, for abuse detection | Workspace Geo; Americas in the US; cross-Geo processing is an admin setting, on by default outside the US and the EU | https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving (ms.date 2026-10-06); https://docs.databricks.com/aws/en/resources/designated-services (Oct 26, 2026); https://learn.microsoft.com/en-us/azure/databricks/resources/databricks-geos (ms.date 2026-09-11) |
| Azure OpenAI (models sold by Azure) | No | Abuse-monitoring store in the resource geography | Standard: in geography; DataZone: US or EU; Global: anywhere | https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy (ms.date 2026-05-18) |
| Amazon Bedrock | Model providers have no access to prompts or completions | Per-Region mode; 'none' = zero data retention (some models require review) | The chosen Region | https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html; https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html |
| Google Vertex AI | Not without permission | Gemini caching on by default, can be turned off | Chosen region; the global endpoint may serve from any region | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/data-governance |
| Snowflake Cortex | Not stated on the page read | Not stated on the page read | Inside the Snowflake perimeter | https://docs.snowflake.com/user-guide/snowflake-cortex/open_ai_sdk |
| Groq (local default) | Not addressed on the page | None by default; logs up to 30 days when troubleshooting; zero retention available | US Google Cloud buckets | https://console.groq.com/docs/your-data |

### 8.8 Language fit

Short answer: English only. Multilingual search was put out of scope on 2026-08-24 [documented; CLAUDE.md:1332], and the code shows why it would not work by switching a setting.

What is English-only (static count, measured 2026-11-02) [SCR/gtm/research/gap_4.json G4-14, G4-15, G4-20, G4-22; SCR/gtm/gap_4/out/vocab_union.json]:
- 295 English regex and word-list constants holding ~4,452 distinct word forms, across 10 modules (catalog_taxonomy ~2,704 forms, product_type ~1,728, device_role ~588, the enrichment script ~391, main ~237 and five smaller ones). The counts are approximate.
- 10 English prompts (24,139 characters); none states an output language.
- 4 tokenizers that accept only ASCII letters and digits, so 'für' becomes 'f' + 'r' and 'Kopfhörer' becomes 'kopfh' + 'rer' [code; app/keyword_search.py:12; app/catalog_taxonomy.py:117; app/query_guard.py:358; app/main.py:1128].
- 2 English embedding models: gte-small "exclusively caters to English texts" and Databricks' gte-large-en "serves the English version of the model" [documented; https://huggingface.co/thenlper/gte-small, no date; https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/supported-models, 2026-10-30].
- Every eval set (18 keyword, 16 context, 48 page-type queries, 5 recipes, 53 stress queries, the 60 held-out queries of 2026-11-02) and both graded batteries are English: no German or Swedish quality, latency or determinism number exists [code; data/*.json; .claude/rules/eval.md; SCR/nov02/REPORT.md].

Offline probe, hand-written parallel English / German / Swedish text (10 queries and 15 titles per language; 6 queries against those titles for the embedding rows), run through the repo's own code with no LLM call. It is an illustrative sample, not a benchmark [measured 2026-11-02; SCR/gtm/gap_4/out/lang_probe.json, SCR/gtm/gap_4/out/embed_probe.json]:

| Check | English | German | Swedish |
|---|---|---|---|
| Queries where any query reader fires (type, gender, bundle, device) | 10/10 | 0/10 | 0/10 |
| Fashion titles typed (DE / SV hits are loanwords only: 'Parka', 'sneakers') | 7/7 | 1/7 | 2/7 |
| Electronics titles categorised | 5/5 | 2/5 (1 wrong: a phone case read as a phone) | 1/5 |
| Grocery titles categorised | 3/3 | 0/3 | 0/3 |
| Colour words mapped | 6/6 | 0/6 | 0/6 |
| Colour / size variant pairs merged into one card | 2/2 | 0/2 | 0/2 |
| Injection overrides cut by the guard's input layer | 2/2 | 0/2 | 0/2 |
| gte-small: right title ranked first, query and catalog in the same language | 6/6 | 3/6 | 4/6 |
| gte-small: English search phrases against this language's catalog | 6/6 | 4/6 | 2/6 |

What fails closed and what breaks [inference from the probe; SCR/gtm/research/gap_4.json G4-25]:
- Fails closed (does nothing): product-type, gender and age reading, the wedding-guest rule, device-versus-accessory ranking and bundle routing all stay off for every German and Swedish query. The page falls back to embedding retrieval plus the LLM rerank, with weaker English-only embeddings.
- Breaks: the ASCII tokenizers make keyword search (BM25) match fragments such as 'f' and 'r'; the Electronics taxonomy misreads a German phone case as a phone; variant merging stops, so duplicate cards return; and German or Swedish injection overrides pass the guard's input layer and reach the LLM verbatim (the output screen still applies).
- Unmeasured: whether Qwen3-Next or gpt-oss write German or English search phrases for a German query, and how gte-large-en scores German or Swedish text (no LLM or workspace calls were allowed).

A multilingual embedding exists on Databricks: databricks-qwen3-embedding-0-6b (Public Preview, "supports 100+ languages"), listed in the Azure EU-boundary rows. Switching means re-embedding the catalog (the gte-large-en embed of 299,972 rows took ~38 min on 2026-10-12; a new model's time is unmeasured), rebuilding the index (a stored-vector rebuild took 47.8 min) and re-measuring every retrieval number; CAS has never tested it [documented / measured; SCR/gtm/research/gap_4.json G4-21; CLAUDE.md:1537-1542]. If German or Swedish enters scope, the work list is Unicode tokenizers, a multilingual embedding with a full re-embed, per-language versions of the ~4,452 word forms (with German compound splitting: 'Sommerkleid', 'Laufschuhe'), per-language guard markers, UI strings and a German and a Swedish eval set. No dev-day estimate exists for it.

Line for collateral: "CAS works on English-language catalogs and queries today. German or Swedish search is a scoped localisation project (tokenizers, word lists, embedding model, prompts, eval set), not a setting." Do not promise H&M or C&A native-language search. The one configuration the evidence covers is an English catalog feed with English queries [SCR/gtm/research/gap_4.json recommendations].

### 8.9 Costs measured on our workspace

All at Databricks list prices, on our Azure PoC workspace. The client's contract, region and catalog size will change them.

| Item | Value | Date | How known | Source |
|---|---|---|---|---|
| Vector Search endpoint hosting an index, idle | ~$5-8/day; 4 DBU/h = $6.72/day at $0.07/DBU (US East) | Billing Sep 10-11, 2026, recorded 2026-09-23 | derived from billing + pricing page | CLAUDE.md:1610-1621; databricks/README.md:505-510; https://www.databricks.com/product/pricing/vector-search (fetched 2026-11-02, no date shown) |
| One clean managed index build | ~$20 ($20.27 on Sep 11, the build plus an eval session); a failed build cost $23.39 (Sep 10) | 2026-09-11 | measured | CLAUDE.md:1610-1613; databricks/README.md:514-516 |
| One-time catalog embed, 299,972 rows | ~38 min, 44.73M tokens, ~83 DBU (the script's estimate) | 2026-10-12 | measured time; estimated DBU | CLAUDE.md:1537-1540 |
| Index rebuild from stored vectors (now used) | 47.8 min; dollars unmeasured | 2026-10-12 | measured time | CLAUDE.md:1541-1542 |
| Endpoint kept up for a month | ~$150-240/month for the idle endpoint alone; excludes App compute and per-token LLM and embedding charges | - | estimate (extrapolation) | CLAUDE.md:1621; SCR/gtm/research/cloud_ops.verify.json CLO-53 |
| Medium Databricks App (up to 2 vCPUs, 6 GB) | 0.5 DBU/h, about $347/month per instance at East US list ($0.95/DBU x 730 h) | - | documented rates + arithmetic (not measured on our bill) | https://docs.databricks.com/aws/en/dev-tools/databricks-apps/compute-size (updated Sep 25, 2026); https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Azure Databricks' and armRegionName eq 'eastus' (fetched 2026-11-02); SCR/gtm/research/gap_1.json G1-09 |

There is no pause: an endpoint bills while it hosts an index and stops billing 24 hours after its last index is deleted [documented; https://learn.microsoft.com/en-us/azure/databricks/ai-search/cost-management, ms.date 2026-10-20]. The notebook 11_idle_guard.py stops the App and deletes the index after 3 idle days; its audit field names are unverified, so it should be dry-run first [code; databricks/11_idle_guard.py:1-19; CLAUDE.md:1624-1625]. The full monthly model (App, Vector Search and tokens by traffic level) is in the cost section; at East US list its fixed floor is $551/month for one Vector Search unit plus one Medium App, before tokens, or $898 with the 2 App instances Databricks recommends for availability [estimate; SCR/gtm/research/gap_1.json summary, G1-25]. Larger sizes exist if needed: Large Apps (up to 4 vCPUs, 12 GB, 1 DBU/h), XLarge (up to 12 vCPUs, 48 GB, 3 DBU/h), and a storage-optimized Vector Search unit (18.29 DBU/h for 64M vectors of 768 dimensions) [documented; SCR/gtm/research/gap_1.json G1-09, G1-10]. On scale-to-zero container hosts elsewhere, idle compute can cost nothing, but CAS's ~50-110 s start-up argues for one warm instance, and a managed vector index bills whether queried or not (inference).

### Open questions for this area

- Licence: can LatentView deliver CAS images or source to retailers under AGPL v3, or must it relicense its own code? (legal)
- Which cloud and region do H&M, C&A and the grocery targets run Databricks or Snowflake in, if at all? Is cross-Geo processing allowed? Is Private Link to model endpoints required? Our own workspace's region is not recorded either.
- Databricks account team, in writing: is Qwen3-Next reachable from an Azure EU workspace with cross-Geo on (the page lists no entry)? Which Google Cloud page governs Europe? Is PT planned for Qwen3-Next?
- If the answer for Azure EU, Private Link or production is gpt-oss-120b on PT: re-run 08, 05 with its control, 10 and 12 on it, budget the ~3x per-query LLM time measured on 2026-09-23, and size the PT blocks (throughput per model unit is undocumented for CAS's models).
- Do Bedrock, Vertex AI and Snowflake Cortex accept `seed` and the JSON-object format? Untested; CAS has no seed fallback today.
- Can a retailer's backend call a Databricks App's API with a service-principal token at shopper volume, or should production CAS run as a container in the retailer's own cloud? Not tested.
- How do Qwen3-Next and gpt-oss respond to German or Swedish queries with English prompts, and how does gte-large-en score German or Swedish text? Unmeasured. Would an English catalog feed with English queries be acceptable to H&M or C&A for a pilot?
- Memo hardening for a client: HMAC-keyed (secret-keyed) hashes, a schema only the App identity can write, a maximum age, keeping the assistant stage out of a shared volume?
- SPCS's 90 s idle limit on direct API calls; Streamlit in Snowflake's supported versions against CAS's 1.58 pin.
- Container image size, start-up time and memory on a 2-vCPU host: never built or measured; is one always-warm instance acceptable cost-wise?
- Who owns the trace store (MLflow) for a client deployment, and how do erasure requests and retention map to the client's storage?

## 9. Recommendation and phased plan

**Answer.**
- Run CAS as a query-understanding layer on top of each client's own search engine.
- Prove it offline on that client's queries before building anything.
- Add the features that need CAS to hold the catalog only where a pilot shows they pay.

The phases follow Section 7's recommendation; pattern letters are Section 7's (A full replacement, B query understanding, C rerank, D router, E rank fusion). Section 3's connectors and Section 8's cloud packaging come in only when a phase needs them. All effort figures are estimates for one developer who knows the repo; none is measured.

### 9.1 Why this order

- **It keeps what the client already has.** The overlay leaves the client's engine, merchandising rules, stock filters, facets and ads in place (Section 7). That matches the agreed positioning: CAS sits inside the client's stack [CLAUDE.md:1297-1303].
- **It needs no catalog copy.** Patterns B and C send queries or top results, not the catalog, so neither the connector gap (Section 3) nor the catalog-in-memory limit (Section 6) blocks them (inference). The API still loads a catalog at start-up today, so a catalog-free service is a build item (code) [app/main.py:161-193].
- **It keeps LLM load low.** Only tail queries reach the LLM. That keeps traffic under the ~2.4 new searches per second per-workspace ceiling for longer, and cuts the token bill (inference from Sections 5 and 6).
- **It is the easiest pattern to move off Databricks.** It needs an LLM endpoint and a small API, with no vector store or embeddings (inference; Sections 7 and 8).

### 9.2 Phases

**Phase 0: offline proof (per client).**
- *Goal:* show on the client's own queries that CAS's rewritten queries and filters improve its engine's tail results.
- *Entry:*
  - a named client contact;
  - 30-50+ real head and tail queries from its logs (more is better);
  - a read-only sandbox of its engine's API;
  - merchandisers willing to write a context eval set;
  - written approval that query text (and, for the C arm, product text) may go to the chosen LLM host;
  - the client's cloud and region.
- *Work:*
  - Run three arms against the sandbox: the engine on the raw query; the engine on CAS's intents and filters (B); B plus CAS rerank (C).
  - Replay the router offline to size the tail share, the added latency and the token bill.
  - Confirm which LLM the client's region allows. On Azure EU the listed models CAS has run are gpt-oss-120b and gpt-oss-20b, and 20b failed too many reranks on Groq, so in practice it is gpt-oss-120b (Qwen3.5 122B, a preview, is listed but untested); re-run 08 (acceptance), 05 with its no-LLM control, 10 (page eval) and 12 (determinism) on it.
  [SCR/gtm/research/architecture.json recommendations[1]; CLAUDE.md:1663-1698]
- *Exit:*
  - B beats the raw query on the tail and leaves the head unchanged.
  - The tail's LLM load fits the pay-per-token ceiling, or a capacity plan is agreed.
  - The data-use approval is in writing.
- *Delivers to the client:* an evidence report on its own queries; no production change.
- *Effort:* about 2-3 weeks of calendar time (estimate). The developer work overlaps the Phase 1 build and is not estimated separately.

**Track R: readiness (internal, in parallel with Phase 0, before any client-facing service).**
- *Work, code:*
  - an LLM client timeout, max_retries of 0-1 and per-stage max_tokens (not estimated: a few lines) [app/llm_client.py:310-313];
  - an explicit seed setting and per-host response-format checks (C3, 0.5-1 dev-days);
  - page_cache keys that include the backend and embedding model (not estimated) [app/main.py:2253-2259];
  - SEARCH_BACKEND validation through the backend registry (0.5-1): today a typo silently serves local search on a laptop or in a notebook, while the App's deploy notebook already refuses one [app/config.py:262; app/search.py:296-303; databricks/09_deploy_app.py:147-166].
- *Work, deployment:*
  - deploy the current App version and verify the memo-volume grants [CLAUDE.md:1785];
  - a stepped workspace load test (Section 6.4; not estimated).
- *Work, outside answers:*
  - Databricks' written answers on Qwen3-Next in the EU, provisioned throughput, and tokens per second per model unit;
  - legal answers on AGPL v3 and the demo data.
- *Exit:*
  - the load test has recorded which limit refuses requests first at 1, 5 and 10 searches per second;
  - the model the client's region allows passes 08;
  - legal has answered.

**Phase 1: D-lite behind a client flag.**
- *Entry:*
  - Phase 0 exit met;
  - the engine's filter-injection hook confirmed in writing;
  - hosting decided: a Databricks App that exposes /api/ routes with service-principal (machine-to-machine) authentication, or a container in the client's cloud [https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/permissions, ms.date 2026-09-11];
  - Track R's code items done.
- *Work:*
  - a /api/understand endpoint;
  - a constraint-to-filter compiler for the engine, and an intent executor with merge;
  - a router adapter that recomputes coverage from the engine's top hit, with router tests and a threshold calibrated on the client's logs;
  - a fail-open deadline: on a timeout or error the client serves its own results;
  - a client prompt variant and eval set;
  - hosting and authentication, and a catalog-free start-up. We assume the start-up change sits inside B's hosting item (inference).
- *Exit:*
  - in the client's own online test, the tail improves and the head is unchanged; CAS has no A/B tooling (out of scope) [CLAUDE.md:1332];
  - timeouts fall back to the engine's results;
  - tail traffic stays within the LLM ceiling.
- *Delivers to the client:* if the Phase 0 proof holds, better results for long, natural-language queries, inside its own engine, with its merchandising, stock, facets and ads intact.
- *Effort:*
  - 12-20 dev-days for the first engine;
  - +3-6 for each further engine;
  - +8-11.5 if it must run in the client's cloud rather than on Databricks (Sum A with C2, plus C4; the overlap with B's hosting work is not netted, so the top of that range is high).

**Phase 2: CAS rerank on the tail.**
- *Entry:*
  - Phase 1 stable;
  - an agreed policy for when CAS's order conflicts with the client's pins, boosts or sponsored slots;
  - a decision on screening seller text before it reaches the rerank prompt, since indirect injection is open [.claude/rules/safety.md:139-141];
  - client approval for product text to reach the LLM.
- *Work:*
  - a /api/rerank endpoint keyed by the client's product id;
  - a field mapper;
  - re-appending rows 16+ and any row the model dropped, so a failure leaves the client's order intact.
- *Exit:* a measured gain of the real reranker over the engine's own order on the client's tail. Only a perfect-rerank ceiling is known today [SCR/gtm/research/arch_sim/sim_patterns_out.json].
- *Delivers to the client:* if the exit test holds, reordered top results for tail queries, with optional per-item reasons for merchandisers.
- *Effort:* +5-8 dev-days (17-28 cumulative).

**Phase 3 (optional): D-full where it pays.**
- *Entry:*
  - a measured gain for features that need CAS's own catalog: recipe and setup bundles, type-aware ordering, cross-sell and upsell;
  - the serving unit agreed (SKU, style-colour or title), with one catalog per market;
  - the model and region settled;
  - a load test passed at the target rate.
- *Work:*
  - catalog compiler phases P0-P3;
  - connectors through uc:// on Databricks, with a workspace proof; native readers only off Databricks;
  - routing to CAS's own path;
  - above ~1M products, thin workers (not estimated).
- *Exit:* the routed queries beat the engine in the client's online test, within the latency budget.
- *Delivers to the client:* CAS-ranked pages for routed queries, with CAS's bundles and recommendations. Fashion is the likeliest fit (inference).
- *Effort:*
  - 30-45 dev-days on its own, or about 25-42 after D-lite, because the router already exists (inference);
  - per client, ~1-2 days to review the mapping, plus the embedding run (~38 min per 300K texts on Databricks).

**Side path: a CAS-hosted demo on a client's catalog** (a sales step, possible at any time).
- On today's code: a hand-shaped CSV, ~1 day per client, with silent failures still possible.
- Robust: compiler P0 plus the first wave of readers, 8.5-12 dev-days plus admin lead time.

**Not recommended for the P0 accounts:** A (full replacement) and E (rank fusion) (Section 7).

### 9.3 Effort roll-up

Every row is an estimate for one developer who knows the repo, tests included, unless marked otherwise. "Not estimated" means no section gives a figure. Where several sections give the same item, it appears once.

| Work item | Dev-days | Kind | Section | Source |
|---|---|---|---|---|
| **Overlay.** Phase 0 offline proof, per client | 2-3 weeks of calendar time, not dev-days | estimate | 7 | SCR/gtm/research/architecture.json recommendations[1] |
| **Overlay.** B: query-understanding layer, first engine (/api/understand, filter compiler, intent executor + merge, client prompt tuning + eval set, hosting and auth; includes the ~3-5 days of plumbing shared by B-D) | 9-15 | estimate | 7 | SCR/gtm/research/architecture.json tables[0], recommendations[5] |
| **Overlay.** Router part of D-lite (coverage from the engine's top hit, router tests, threshold calibration) | ~3-5 | inference (D-lite minus B) | 7 | SCR/gtm/research/architecture.json tables[0] |
| **Overlay.** D-lite = B + router (Phase 1) | 12-20 | estimate | 7 | SCR/gtm/research/architecture.json tables[0]; SCR/gtm/research/architecture.verify.json missing[5] |
| **Overlay.** B, each further engine | +3-6 | estimate | 7 | SCR/gtm/research/architecture.json tables[0] |
| **Overlay.** B adapters for Kroger + H&M / for all five accounts (arithmetic on the two rows above; not additional) | 12-21 / 21-39 | estimate | 7 | SCR/gtm/research/gap_2.json G2-24 |
| **Overlay.** C: rerank layer on the tail (/api/rerank, field mapper, tail re-append) | 5-8 | estimate | 7 | SCR/gtm/research/architecture.json tables[0]; SCR/gtm/research/architecture.verify.json ARC-05 |
| **Overlay.** Phase 1 + 2 cumulative | 17-28 | estimate | 7 | arithmetic on the rows above |
| **Overlay.** D-full: router + compiled and indexed catalog | 30-45 | estimate | 7 | SCR/gtm/research/architecture.json tables[0] |
| **Overlay.** Pattern A, full replacement (not recommended) | ~40+ | estimate | 7 | SCR/gtm/research/architecture.json tables[0]; SCR/oct27/byo/REPORT.md:410 |
| **Overlay.** Pattern E, rank fusion (not recommended) | 30-45 | estimate | 7 | SCR/gtm/research/architecture.json tables[0] |
| **Catalog.** Interim pilot: hand-shape one client catalog into our CSV | ~1 per client | estimate | 3 | SCR/oct27/byo/REPORT.md:35, 131-140 |
| **Catalog.** Compiler P0: offline catalog compiler (CSV family, Parquet, JSON Lines, SQLite and uc:// readers; presets; mapping.json; coverage report; ownership guard) | 4.0 optimistic, 5-7 safer | estimate | 3, 7, 8 | SCR/oct27/byo/REPORT.md:404, 412 |
| **Catalog.** uc:// workspace proof on one federated BigQuery or Snowflake table plus a Databricks-to-Databricks share | 0.5-1 (admin lead time excluded, unknown) | estimate | 3 | SCR/gtm/research/connectors.verify.json CON-35; SCR/gtm/research/gap_3.json G3-21 |
| **Catalog.** Native Snowflake reader (key pair, fetch_arrow_batches) | 1.5 (may trim 0.25-0.5; adds a live check) | estimate | 3 | SCR/gtm/research/connectors.verify.json CON-35; SCR/gtm/research/gap_3.json G3-31 |
| **Catalog.** Native BigQuery reader (Storage Read API to Arrow) | 1.0 | estimate | 3 | SCR/gtm/research/connectors.verify.json CON-35 |
| **Catalog.** Generic sql: reader (SQLAlchemy; not in the compiler's ~36) | 1.5, +0.25-0.5 per extra driver | estimate | 3 | SCR/gtm/research/connectors.verify.json CON-35 |
| **Catalog.** First wave of readers (the four rows above) | 4.5-5 | estimate | 3 | SCR/gtm/research/connectors.verify.json CON-35 |
| **Catalog.** Compiler P0 + first wave | 8.5-12 | estimate | 3 | our arithmetic on the rows above |
| **Catalog.** On-demand readers off Databricks: Hive/Trino 1.5; Excel 0.5; Delta + Iceberg 1.0; s3:// / abfss:// / gs:// addresses 1.0; open Delta Sharing 0.5 + 0.25 version pin; ORC/Avro 0.25; Azure Synapse 0.5; Microsoft Fabric 1.0; AWS Glue 1.0 | 0.25-1.5 each; 7.5 if all were built | estimate | 3 | SCR/gtm/research/connectors.verify.json CON-35; SCR/gtm/research/connectors.json tables[0]; SCR/gtm/research/gap_3.json G3-31 |
| **Catalog.** On-demand feeds and commerce suites: Google Merchant Center (GMC) XML feed parser 1.0; Google Merchant API puller 2.0; Shopify bulk puller 2.0; Lakeflow Shopify preset 1.0; Salesforce Commerce Cloud preset 3.0; SAP Commerce puller 2.5; commercetools puller 2.0 | 1-3 each; 13.5 if all were built | estimate | 3 | SCR/gtm/research/connectors.verify.json CON-35 |
| **Catalog.** Compiler P1, serving generalisation (client department names, currency module, dedup by id) | 7.75 | estimate | 7 (inside D-full) | SCR/oct27/byo/REPORT.md:405 |
| **Catalog.** Compiler P2, Databricks-native loading (01 reads uc:// and the mapping; staging, validate, promote; includes the 1.0-day Auto Loader landing) | 7.0 | estimate | 3 | SCR/oct27/byo/REPORT.md:406; SCR/oct27/byo/design_RECOMMENDED.md:740 |
| **Catalog.** Compiler P3, stable product ids + incremental index refresh (counted once; also in Sections 4, 6, 7) | 5.5 | estimate | 3, 4, 6, 7 | SCR/oct27/byo/REPORT.md:407 |
| **Catalog.** Each CDC (change data capture) reader after P3 (Delta CDF, Snowflake streams, BigQuery CHANGES, watermarks) | ~1.0 each | estimate | 3 | SCR/gtm/research/connectors.verify.json CON-35 |
| **Catalog.** Compiler P4 (feed-first domain and quality, client eval kit) + P5 (residue, handover) | 10.0 + 2.0 | estimate | 3 (inside ~36) | SCR/oct27/byo/REPORT.md:408-409 |
| **Catalog.** Full catalog design P0-P5 | ~36 | estimate | 3, 7, 8 | SCR/oct27/byo/REPORT.md:410 |
| **Catalog.** Per client: review the mapping and run the first build | ~1-2 (+ ~38 min of embedding per 300K texts on Databricks) | estimate | 3, 7 | SCR/oct27/byo/REPORT.md:443-444 |
| **Retrieval.** Validated backend registry (once; the same refactor as C5's interface) | 0.5-1 | estimate | 4 | SCR/gtm/research/retrieval.json RET-39 |
| **Retrieval.** New vector backend inside today's shape: query adapter 1-2, index build/refresh 1-3, score calibration + evals + tests 1-2 | 3-7 per backend | estimate (verifier: unverifiable, arithmetic consistent) | 4 | SCR/gtm/research/retrieval.json RET-39; SCR/gtm/research/retrieval.verify.json RET-39 |
| **Retrieval.** Hand /keyword_search to a platform's own BM25 or hybrid search | +2-4 | estimate | 4 | SCR/gtm/research/retrieval.json RET-39 |
| **LLM.** Token-provider authentication (Entra ID, Microsoft's sign-in service; Bedrock short-term keys or SigV4 request signing; Vertex tokens) = C2 | 1-1.5 | estimate | 5, 8 | SCR/gtm/research/cloud_ops.verify.json CLO-27; SCR/gtm/research/cloud_ops.json C2 |
| **LLM.** Seed and response-format negotiation = C3 | 0.5-1 | estimate | 5, 8 | SCR/gtm/research/cloud_ops.verify.json CLO-27, CLO-08; SCR/gtm/research/cloud_ops.json C3 |
| **LLM.** Provider-neutral acceptance smoke (port of databricks/08, injection rows included) = C4 | 1 (+ ~55K tokens per provider run) | estimate | 5 (no figure), 8 | SCR/gtm/research/cloud_ops.json C4; CLAUDE.md:1663-1664 |
| **LLM.** Client timeout, max_retries, per-stage max_tokens | not estimated ("a few lines") | code change | 5, 6 | SCR/gtm/research/llm.json latency levers; SCR/gtm/research/gap_1.json recommendations |
| **Scale.** Laptop vector-scan fix plus a test pinning the top-60 lists | not estimated (a few lines, inference) | code change | 4, 6 | SCR/gtm/research/scalability.verify.json SCA-24 |
| **Scale.** Query AI Search through its index URL with OAuth; parallel per-intent queries; batched product-page embeddings | not estimated ("small code change") | code change | 6 | SCR/gtm/research/scalability.json tables[3]; SCR/gtm/research/gap_1.json G1-32 |
| **Scale.** LLM capacity and token discipline (provisioned throughput or a model switch with re-runs of 05/08/10; recommend=false for API callers; scores-only rerank A/B) | not estimated | code + evaluation + procurement | 5, 6 | SCR/gtm/research/gap_1.json G1-14 to G1-20, G1-32 |
| **Scale.** Thin workers beyond P3 (rows from a Lakebase synced table, taxonomy stored at ingestion, BM25 replaced) | not estimated | re-architecture | 6 | SCR/gtm/research/scalability.json SCA-22, SCA-23 |
| **Scale.** Scale out: more workers or instances, a shared cache and memo, async endpoints, an API-only deployment | not estimated | re-architecture | 6 | SCR/gtm/research/scalability.json tables[3]; SCR/gtm/research/architecture.json ARC-08, ARC-09 |
| **Scale.** Workspace load test (start-up and memory on 2 vCPUs at 300K; memo latency; stepped 1/5/10 searches per second; then a synthetic 1M-row catalog) | not estimated | measurement | 6 | SCR/gtm/research/scalability.json recommendations |
| **Cloud.** C1 container image, two flavours (dependency lock and image scan not costed) | 2-3 | estimate | 8 | SCR/gtm/research/cloud_ops.json C1; SCR/gtm/research/cloud_ops.verify.json CLO-14 |
| **Cloud.** C5 vector-store interface + first adapter (pgvector) / each further managed store | 3-4 / 2-3 each | estimate | 8 | SCR/gtm/research/cloud_ops.json C5 |
| **Cloud.** C6 embedding client with its own base URL and key | 1.5-2 | estimate | 8 | SCR/gtm/research/cloud_ops.json C6 |
| **Cloud.** C7 build command off Databricks (encode, vector store, index load, catalog snapshot) | 3-4 | estimate | 8 | SCR/gtm/research/cloud_ops.json C7 |
| **Cloud.** C8 memo backend shared by replicas | 2-3 | estimate | 8 | SCR/gtm/research/cloud_ops.json C8 |
| **Cloud.** C9 object-store paths and settings | 1-1.5 | estimate | 8 | SCR/gtm/research/cloud_ops.json C9 |
| **Cloud.** C10 API authentication, CORS (cross-origin browser rules), rate limiting | 2-3 | estimate | 8 | SCR/gtm/research/cloud_ops.json C10 |
| **Cloud.** C11 backend-neutral start-up checks and health probes | 0.5 | estimate | 8 | SCR/gtm/research/cloud_ops.json C11 |
| **Cloud.** C12 shared result caches across replicas (optional) | 2 | estimate | 8 | SCR/gtm/research/cloud_ops.json C12 |
| **Cloud.** C13 infrastructure as code: Helm chart / Terraform or Bicep per cloud / Snowpark Container Services spec | 2 / 2-4 per cloud / ~2 | estimate | 8 | SCR/gtm/research/cloud_ops.json C13 |
| **Cloud.** C14 per-cloud cost guard | 0.5-1 per cloud | estimate | 8 | SCR/gtm/research/cloud_ops.json C14 |
| **Cloud.** C15 OpenTelemetry / log shipping (optional) | 1 | estimate | 8 | SCR/gtm/research/cloud_ops.json C15 |
| **Cloud.** Sum A, portable package (C1 + C3 + C9 + C10 + C11, optionally + C2) | 6-9 (7-10.5 with C2) | estimate | 8 | SCR/gtm/research/cloud_ops.json SUM A; sums checked in SCR/gtm/research/cloud_ops.verify.json |
| **Cloud.** Sum B, first cloud-native target off Databricks (Sum A + C4 + C5 + C6 + C7 + C8 + one C13) | ~20-28 (the AWS example needs C2: ~28.5 at the top) | estimate | 8 | SCR/gtm/research/cloud_ops.json SUM B |
| **Cloud.** Sum C, each further cloud | ~5-9 | estimate | 8 | SCR/gtm/research/cloud_ops.json SUM C |

**Totals by path** (estimates; ranges; each path stands alone unless it says "after"):

| Path | What it adds up | Dev-days |
|---|---|---|
| Overlay Phase 1 on Databricks (recommended first) | D-lite, first engine | 12-20 |
| Overlay Phases 1 + 2 on Databricks | + C on the tail | 17-28 |
| ... per further engine | B adapter (C's per-engine work is not estimated) | +3-6 |
| Overlay in the client's own cloud instead | + Sum A with C2 (7-10.5) + C4 (1); no vector store, embeddings or catalog needed (inference) | Phase 1: 20-31.5; Phases 1 + 2: 25-39.5 (overlap with B's hosting and auth not netted) |
| CAS-hosted demo on today's code | hand-shaped CSV | ~1 per client |
| Connectors for a CAS-held catalog, on Databricks | compiler P0 + first wave | 8.5-12 + admin lead time |
| D-full after D-lite | 30-45 minus the router already built (~3-5); includes compiler P0-P3 (~24-27) | ~25-42 (inference) |
| Full catalog design | compiler P0-P5 | ~36 |
| First cloud-native target off Databricks, with CAS's own retrieval | Sum B + relevance-floor calibration (1-2, from Section 4); add compiler P0 (4-7) and native readers (~4: Snowflake 1.5, BigQuery 1.0, sql: 1.5) if the catalog comes from a warehouse | ~21-30 before the catalog work |
| Each further cloud | Sum C | ~5-9 |
| Must be scheduled but has no estimate | LLM timeouts and caps; page_cache key fix; laptop scan fix; AI Search index-URL queries; thin workers; scale-out; workspace load test; re-runs of 05/08/10/12 per model; seller-text screen; memo hardening; restricted-category blocklist; localisation | - |

**Where sections overlap, and which figure we used:**
- *Sections 5 and 8.* Token-provider authentication (1-1.5) and seed / response-format negotiation (0.5-1) are the same items as C2 and C3, so they are counted once. Section 5 gives no figure for the acceptance smoke; we used Section 8's C4, 1 dev-day plus ~55K tokens per provider.
- *Sections 4 and 8 on vector backends:* different scopes, different numbers.
  - Section 4's 3-7 per backend assumes stored vectors are reused and hydration stays in process. We use it for a backend added within today's deployment shape, for example a client engine as the retrieval backend.
  - Section 8's C5 + C6 + C7 (7.5-10) also covers re-encoding the catalog off Databricks, a build command and a memo store. We use Sum B for a first non-Databricks target.
  - Sum B does not name relevance-floor calibration, so we add Section 4's 1-2 days to it (our addition).
  - Section 4's backend registry (0.5-1) is the same refactor as C5's interface, so it is not added to Sum B again.
- *Compiler phases cited by several sections.* Compiler P0 (Sections 3, 7, 8) and P3 (Sections 3, 4, 6, 7) are counted once. Both sit inside D-full's 30-45 and inside the full ~36. The Auto Loader landing (1.0) is part of P2 (7.0), not extra.
- *D-lite.* The lane labelled its 12-20 "B + C", but its own rows put B + C at 14-23 before any router work. We use Section 7's reading: D-lite = B (9-15) + router (~3-5). C arrives in Phase 2 (17-28 cumulative) [SCR/gtm/research/architecture.verify.json missing[5]].
- *Hosting and authentication.* B's 9-15 includes ~3-5 of shared plumbing: an API-only service, service-principal auth and responses keyed by client id. If the overlay runs in the client's cloud, C1 (container) and C10 (API auth) overlap that plumbing. We did not net the overlap, so the top of those totals is conservative.
- *Phase terms.* "Phase 0-3" are the rollout phases of Sections 7 and 9 (Phase 0 is the offline proof). The catalog compiler's phases are "compiler P0-P5" here and in Sections 3 and 8 (Section 7 writes "catalog-compiler phases P0-P3").

### 9.4 Prerequisites outside our control

| Prerequisite | Who | Needed for | Lead time |
|---|---|---|---|
| Metastore-level grants: CREATE CONNECTION, CREATE CATALOG (or connection ownership) and, for Hive or OneLake catalog federation, storage credentials and external locations; network reach to the client's sources | Metastore or account admin (ours for dev; the client's for its workspace) | uc:// proof, every federated source | Unknown; in no estimate [SCR/gtm/research/gap_3.json G3-20, G3-21] |
| Real query logs (head and tail; ideally enough to see the repeat share and the peak rate) | Client | Phase 0, router calibration, memo and cost sizing | Unknown |
| A read-only sandbox of the client's engine API, and its extension-hook documentation (filter injection, external rerank) | Client and its engine vendor | Phase 0, Phase 1 adapter | Unknown; Empathy, Voyado and Coveo hooks unread [SCR/gtm/research/gap_2.json open_questions] |
| Written approval for query text (B) and product text (C) to reach a named LLM host and region; whether Private Link is required | Client (security, legal) | Phases 0-2 | Unknown |
| A context eval set written by the client's merchandisers | Client | Phase 0 exit | Unknown [SCR/oct27/byo/REPORT.md:282-284] |
| Written answers: Qwen3-Next from Azure EU, PT plans, tokens per second per model unit for gpt-oss-120b, $/DBU in the client's region and tier, a no-training commitment for FM API prompts and outputs | Databricks account team | Track R; any EU or high-traffic plan | Unknown |
| A workspace load-test slot and budget; deploying the current App; verifying the memo-volume grants | Our workspace admins (dev catalog) | Track R; any throughput statement | Unknown [CLAUDE.md:1785, 1796-1805] |
| Legal sign-off: the AGPL v3 network clause on any image or source handover; the research-only Amazon Reviews 2023 demo data (images hotlinked from Amazon, Amazon titles in sponsored.json's git history) | LatentView legal | Any packaging or collateral | Unknown [LICENSE:1-2; SCR/gtm/research/claims.json CLA-58, CLA-59] |

## 10. Risks and open questions

### 10.1 Risks

Likelihood and impact are our judgement (inference), not measurements.

| Risk | Likelihood | Impact | Mitigation | Section |
|---|---|---|---|---|
| **EU model gap.** No Azure EU region lists Qwen3-Next, which leaves gpt-oss-120b as the usable listed model CAS has run there (gpt-oss-20b failed too many reranks on Groq; Qwen3.5 122B, a preview, is untested). That model was ~3x slower in our one comparison (30.3 s vs 10.4 s), allows ~0.7 new searches per second, costs ~$2.40 per 1,000 searches at West Europe list prices, and has no quality number on today's prompts | High for an Azure-EU fashion client | High | Confirm the region in Phase 0. Get Databricks' written answer. Re-run 08, 05, 10 and 12 on the allowed model. Check the AWS EU listing; evaluate Qwen3.5 122B (preview) | 5, 6, 8 |
| **LLM capacity.** At most ~2.4 new searches per second per workspace on pay-per-token (an upper bound; an undocumented default reservation may lower it). Qwen3-Next is not offered under provisioned throughput, and tokens per second per model unit are undocumented | High at retailer volume; lower under D-lite (tail only) | High | Route only the tail; caches and memo; recommend=false for API callers; per-stage max_tokens; a scores-only rerank A/B; account-team sizing; a load test | 5, 6 |
| **Latency.** ~10 s per new search on Databricks (eval harness); no search that uses the LLM below ~3 s; the App never timed | High | High on head queries; medium on the tail | The router keeps the head on the engine; a client deadline with fail-open; a non-reasoning model; skip rerank where confident; measure the App's p50/p95 | 5, 7 |
| **Value over the incumbent is unproven.** Lift was measured only against embedding search and a BM25 stand-in. Kroger's Empathy already lists hybrid and vector search; H&M rebuilt on Voyado in 2026; Albertsons bought two AI layers | Medium-high | High (no business case) | Phase 0 on the client's own logs and engine, with explicit exit criteria | 7 |
| **Engine hooks unknown** (Empathy filter injection or external rerank, Voyado's API, Coveo) | Medium | Medium-high (B or C may not fit) | Read the API, or ask, before writing code; Empathy first | 7 |
| **Silent catalog misreads** on today's code (the Shopify export showed one blank product with HTTP 200) | High if a catalog is hand-shaped | Medium | The overlay needs no catalog copy; compiler P0, with its coverage report, before any CAS-held catalog | 3 |
| **Connector access** blocked by grants, networks or cross-cloud reads; Hive on HDFS and transactional Hive tables need an export | Medium | Medium (delay) | Start with BigQuery or Snowflake query federation (the lightest proof); ask the admin early; ask Hive clients the three questions | 3 |
| **Catalog held in each process.** ~5.2 GB per worker at 1M products and ~25 GB at 5M; one cross-market fashion catalog could reach ~9.3M rows (estimates) | Medium (D-full only) | High for D-full | One catalog per market; compiler P3 and thin workers; or stay with the overlay | 6 |
| **No load test; the current App is not deployed;** its memory and start-up on 2 vCPUs are unmeasured | Certain (a fact today) | High for any throughput or SLA statement | Track R load test; deploy the current App | 6, 8 |
| **English only:** word lists, tokenizers, prompts, embedding models and eval sets | High for H&M and C&A local-language markets | High for EU fashion | Pilot on an English catalog feed with English queries; scope localisation as its own project | 8 (also 4, 7) |
| **Security gaps:** the API has no authentication; seller-text (indirect) injection is open, which matters most for C; memo keys are plain sha256 and values echo shopper text; the dev catalog is not least-privilege; no restricted-category blocklist | Medium | High | C10 before any pilot; screen seller text before C; secret-keyed (HMAC) memo keys; the client's own catalog and schema; a blocklist | 8 |
| **Licensing:** AGPL v3 section 13 on handover; research-only demo data | Medium | High (could block delivery) | Legal sign-off before packaging; client catalog only | 8 |
| **A new LLM host silently degrades:** a refused seed makes every stage fall back; 1-hour Vertex and 12-hour Bedrock tokens expire mid-session | High if unpatched | High (quality silently drops to the no-LLM baseline) | C2, C3 and the C4 smoke before naming any host | 5, 8 |
| **Evidence on the wrong model:** every recent quality number is on Groq qwen3.8-27b with local embeddings, graded by LLM prompts, with one sample per query | Certain | Medium | Re-run on the deployment model, with the client's queries, and repeats | 5 |
| **Cost at volume:** tokens dominate above ~378K new searches a month; provisioned throughput costs ~2.7x pay-per-token at 33% use | Medium | Medium | Route the tail only; caches; scores-only rerank; recommend=false | 5, 6 |
| **Memo and ANN repeatability:** the memo pins the first answer, good or bad; whether ANN indexes return the same top 60 run to run is unmeasured | Medium | Low-medium | Purge CLI and a maximum age; run notebook 12 on the workspace | 4, 5 |
| **Retrieval configuration traps:** a SEARCH_BACKEND typo silently serves local search; the relevance floor is uncalibrated per backend; the page_cache key lacks the backend | Low-medium | Medium | Backend registry; per-backend calibration; key fix | 4 |
| **Databricks dependence:** the only cloud deployment; Apps cannot be public; 5 instances per app by default | Medium (for a client not on Databricks) | Medium | Overlay as a container (Sum A); API-only deployment | 6, 8 |

### 10.2 Open questions

Merged and de-duplicated from Sections 3-8, grouped by who must answer.

**Client**
1. Which engine serves each search surface today? Does it accept injected filters and an external reranker (Empathy, Voyado, Coveo, Algolia, Elasticsearch...)? Can we have a read-only sandbox?
2. From the query logs:
   - What share of traffic is long natural-language queries (the router's tail)?
   - What share is exact repeats?
   - What are the peak and average searches per second?
3. May query text (B) and product text (C) go to an LLM, and on which host and region? Is cross-Geo processing allowed? Is Private Link to model endpoints required?
4. Which cloud and region does the client use, and does it run Databricks or Snowflake? Will it grant a CAS workspace a federation connection, or must CAS run inside its own workspace and cloud?
5. Where do product data live? (H&M's product data hub, C&A's product information management (PIM) system or Snowflake, and Kroger's search feed are only partly documented; Albertsons and H-E-B are unknown.) How often must data refresh (prices, stock)? That decides whether change capture is worth building.
6. Who wins when CAS's order conflicts with pins, boosts or sponsored slots?
7. What is the serving unit (SKU, style-colour or title)? One catalog per market or one across markets? How large is a grocery chain's union catalog?
8. Would an English catalog feed with English queries be acceptable for an H&M or C&A pilot?
9. For Hive sources:
   - HDFS or cloud storage?
   - Transactional tables or not?
   - Is the metastore's database reachable?
   - Is a read-only Azure role enough, and can the storage account sit in another Entra tenant?
10. Who signs the mapping file (a merchandiser for departments, an engineer for id and price), and where does it live? It must never live in our repo.
11. Can merchandisers write a context eval set for the offline proof?

**Databricks account team and platform admins**
1. Is Qwen3-Next reachable from an Azure EU workspace? (No EU row lists it.) Is provisioned throughput planned for it? On Google Cloud Europe, which page governs: the region table or the Designated Services page?
2. Pricing and capacity:
   - Tokens per second per model unit for gpt-oss-120b at CAS's request shape (~4.2K in, ~0.7K out)?
   - $/DBU in the client's region and tier?
   - A no-training commitment for FM API prompts and outputs?
3. How large is the default output reservation when max_tokens is omitted? Does Azure Premium map to the Enterprise-tier limits table? Which tier is our workspace?
4. AI Search:
   - What does target QPS cost?
   - What QPS does one 1024-dimension endpoint copy sustain at 30-60 results per phrase?
   - Are ANN results repeatable run to run?
   - Is AI Search offered in each client's region?
5. Can the 5-instance App limit be raised in practice? Can a retailer's backend call an App's API with a service-principal token at shopper volume?
6. Who is the metastore admin for dev? Does our account hold CREATE CONNECTION or CREATE CATALOG? What is the lead time for a connection plus foreign catalog? Is Retail_DE_cluster in standard or dedicated access mode? Which region is our workspace in?
7. Are WRITE_VOLUME and the memo-volume grants in place on the workspace? What is Lakebase's lookup latency for 150-300 rows per search?

**LatentView owners (Sai, Niharika, legal)**
1. Can LatentView deliver CAS images or source to retailers under AGPL v3, or must it relicense its own code? May the Amazon Reviews 2023 demo data appear in collateral (images hotlinked, Amazon titles in git history)?
2. Positioning:
   - Do we adopt "a query-understanding layer on top of your search", "we detect and propose; a person confirms once" and "English catalogs and queries today"?
   - Do we drop "understands context" and "explains every result" as differentiators, since competitors claim both [SCR/gtm/research/market.json recommendations]?
3. Do we build the Empathy (Kroger) adapter first, and hold the Voyado (H&M) adapter until a buyer asks?
4. When a client wants a CAS-hosted demo: a hand-shaped CSV (~1 day, silent failures live), or wait for compiler P0?
5. Is German or Swedish search ever in scope?
6. Memo policy for client deployments: secret-keyed (HMAC) hashes for memo keys, a schema only the App identity writes, a maximum age, the assistant stage out of shared volumes?
7. Should production CAS run as a container in the retailer's cloud rather than as a Databricks App?
8. Do the Amazon-tuned clothing, device and gift prompt rules suit each client, or does each client get its own prompt variant and eval set?
9. Who owns the trace store (MLflow) in a client deployment? How do data-subject erasure and retention map to client storage?

**Us (engineering and evaluation)**
1. Run the workspace load test:
   - App start-up and memory on 2 vCPUs at 300K;
   - memo-volume latency;
   - a stepped 1/5/10 searches per second, recording which limit refuses first;
   - then a synthetic 1M-row catalog.
   Also measure the App's p50/p95 with the memo on and off, and the per-stage latency of B and C on Databricks.
2. Measure quality, tokens and latency on gpt-oss-120b and on Databricks Qwen3-Next with today's prompts (re-run 08, 05 with its control, 10, 12). A/B a scores-only rerank. Measure Assistant tokens per chat turn.
3. Smoke-test seed and json_object handling on Bedrock, Vertex, Snowflake REST and vLLM. Confirm Qwen3 Next on Bedrock's OpenAI route and Bedrock's quotas. Get Vertex's price for gpt-oss 120B.
4. Retrieval checks:
   - Does self-hosted gte-large-en-v1.5 give the Databricks endpoint's vectors (a 100-text cosine test)?
   - Is ANN repeatable (run notebook 12)?
   - Does hybrid beat plain ANN on CAS's phrases (needs a new index and a re-calibrated floor)?
   - Which relevance floor should each backend use?
5. Code fixes:
   - validate SEARCH_BACKEND;
   - add the backend and embedding model to the page_cache key;
   - add LLM timeouts, max_retries and max_tokens;
   - make the seed setting explicit;
   - fix the laptop scan, with a top-60 test;
   - correct docs/RESEARCH.md:116-117 ("At 60K products, no vector DB needed").
6. Calibrate the router on client logs: it has no tests and was never measured live. Have someone other than Section 7's author check its derived D-lite figures (0.369 context, 0.929 keyword).
7. Connector checks:
   - live reads: Snowflake fetch_arrow_batches, delta-sharing 1.4.1, BigQuery to_dataframe, HiveServer2 over JDBC (a standard database-driver interface);
   - the pandas version on DBR 19 (Databricks Runtime 19);
   - uc:// read time and memory above 1M rows;
   - the cost of a nightly federated snapshot, and cross-cloud egress;
   - the Lakeflow Shopify output and the Salesforce Commerce Cloud export format.
8. How do Qwen3-Next and gpt-oss answer German or Swedish queries with English prompts, and how does gte-large-en score German or Swedish text?
9. Container image size, start-up and memory on a 2-vCPU host.
10. Still unfetched: prices for OpenSearch, Elastic, MongoDB, Qdrant, Weaviate, Milvus and the Azure AI Search tiers; Vertex AI Vector Search quotas; Snowflake Cortex Search at 20 QPS per service against CAS's 3-8 queries per shopper search; whether BigQuery VECTOR_SEARCH can serve online search.

## 11. Sources

External sources cited in Sections 1-10, de-duplicated and grouped by topic. Each carries the page's own date where the page shows one; otherwise "fetched 2026-11-02". The four prices.azure.com links are API queries; their filter text contains spaces and is quoted as the sections use it.

**Databricks: federation, ingestion, Unity Catalog, sharing**
- https://learn.microsoft.com/en-us/azure/databricks/query-federation/ (ms.date 2026-09-29)
- https://learn.microsoft.com/en-us/azure/databricks/query-federation/bigquery (ms.date 2026-10-09)
- https://learn.microsoft.com/en-us/azure/databricks/query-federation/snowflake (ms.date 2026-10-13)
- https://learn.microsoft.com/en-us/azure/databricks/query-federation/snowflake-catalog-federation (ms.date 2026-10-06)
- https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-concepts (ms.date 2026-09-11)
- https://learn.microsoft.com/en-us/azure/databricks/query-federation/hms-federation-external (ms.date 2026-10-06)
- https://learn.microsoft.com/en-us/azure/databricks/query-federation/onelake (ms.date 2026-10-28)
- https://learn.microsoft.com/en-us/azure/databricks/query-federation/foreign-catalogs (ms.date 2026-09-11)
- https://docs.databricks.com/aws/en/query-federation/hms-federation-concepts (Last updated Sep 11, 2026)
- https://docs.databricks.com/aws/en/query-federation/hms-federation-glue (Last updated Oct 13, 2026)
- https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-hive-compatibility (Last updated Sep 11, 2026)
- https://learn.microsoft.com/en-us/azure/databricks/connect/jdbc-connection (ms.date 2026-10-02)
- https://learn.microsoft.com/en-us/azure/databricks/tables/external (ms.date 2026-10-06)
- https://learn.microsoft.com/en-us/azure/databricks/connect/unity-catalog/cloud-storage/s3/s3-external-location-manual (ms.date 2026-10-06)
- https://learn.microsoft.com/en-us/azure/databricks/connect/unity-catalog/cloud-storage/external-locations-adls (ms.date 2026-10-14)
- https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/access-control/privileges-reference (ms.date 2026-10-12)
- https://learn.microsoft.com/en-us/azure/databricks/ingestion/cloud-object-storage/auto-loader/ (ms.date 2026-10-15)
- https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/copy-into/ (Last updated Oct 6, 2026)
- https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/ (Last updated Oct 30, 2026)
- https://docs.databricks.com/aws/en/ingestion/lakeflow-connect/shopify (Last updated Oct 9, 2026)
- https://docs.databricks.com/aws/en/query/formats/excel (Last updated Sep 11, 2026)
- https://docs.databricks.com/aws/en/delta/delta-change-data-feed (Last updated Oct 9, 2026)
- https://docs.databricks.com/aws/en/delta-sharing/ (Last updated Oct 27, 2026)
- https://docs.databricks.com/aws/en/delta-sharing/read-data-databricks (Last updated Sep 24, 2026)
- https://docs.databricks.com/aws/en/dev-tools/python-sql-connector (Last updated Sep 29, 2026)

**Databricks: AI Search (Vector Search)**
- https://learn.microsoft.com/en-us/azure/databricks/ai-search/ai-search (ms.date 2026-10-29)
- https://learn.microsoft.com/en-us/azure/databricks/ai-search/query-ai-search (ms.date 2026-10-29)
- https://learn.microsoft.com/en-us/azure/databricks/ai-search/best-practices (ms.date 2026-10-29)
- https://learn.microsoft.com/en-us/azure/databricks/ai-search/create-ai-search (ms.date 2026-10-29)
- https://learn.microsoft.com/en-us/azure/databricks/ai-search/high-qps (ms.date 2026-10-27)
- https://learn.microsoft.com/en-us/azure/databricks/ai-search/cost-management (ms.date 2026-10-20)
- https://docs.databricks.com/aws/en/vector-search/create-vector-search (Last updated Oct 29, 2026)
- https://docs.databricks.com/aws/en/vector-search/vector-search (Last updated Oct 29, 2026)
- https://docs.databricks.com/gcp/en/vector-search/vector-search (Last updated Oct 29, 2026)
- https://www.databricks.com/product/pricing/vector-search (no date shown; fetched 2026-11-02)

**Databricks: Foundation Model APIs, models, regions, compliance, pricing**
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits (ms.date 2026-10-30)
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/supported-models (ms.date 2026-10-30)
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/model-units (ms.date 2026-09-23)
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/priority-mode (ms.date 2026-10-26)
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/compliance (ms.date 2026-10-28)
- https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving (ms.date 2026-10-06)
- https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/ (Last updated Oct 30, 2026)
- https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference (Oct 2, 2026)
- https://learn.microsoft.com/en-us/azure/databricks/unity-gateway/model-region-availability (ms.date 2026-10-30)
- https://docs.databricks.com/aws/en/unity-gateway/model-region-availability (Last updated Oct 30, 2026)
- https://docs.databricks.com/gcp/en/unity-gateway/model-region-availability (dated 2026-10-30)
- https://learn.microsoft.com/en-us/azure/databricks/resources/databricks-geos (ms.date 2026-09-11)
- https://learn.microsoft.com/en-us/azure/databricks/resources/designated-services (Oct 26, 2026)
- https://docs.databricks.com/aws/en/resources/designated-services (Oct 26, 2026)
- https://docs.databricks.com/gcp/en/resources/designated-services (Oct 26, 2026)
- https://www.databricks.com/product/pricing/foundation-model-serving (no date shown; fetched 2026-11-02)

**Databricks: Apps, Lakebase, cloud coverage**
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/compute-size (ms.date 2026-09-25)
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/horizontal-scaling (ms.date 2026-09-25)
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/permissions (ms.date 2026-09-11)
- https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/connect-local (ms.date 2026-09-11)
- https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth (updated Oct 9, 2026)
- https://docs.databricks.com/aws/en/dev-tools/databricks-apps/permissions (updated Sep 11, 2026)
- https://docs.databricks.com/aws/en/dev-tools/databricks-apps/compute-size (updated Sep 25, 2026)
- https://learn.microsoft.com/en-us/azure/databricks/oltp/projects/sync-tables (ms.date 2026-10-30)
- https://www.databricks.com/en/company/partners/cloud-partners (fetched 2026-11-02)

**Microsoft Azure**
- https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle (ms.date 2026-05-13)
- https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/json-mode (ms.date 2026-05-13)
- https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reproducible-output (ms.date 2025-11-26)
- https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quotas-limits (ms.date 2026-08-20)
- https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/openai/data-privacy (ms.date 2026-05-18)
- https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy (ms.date 2026-05-18)
- https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview (ms.date 2026-08-31)
- https://learn.microsoft.com/en-us/azure/search/semantic-search-overview (ms.date 2026-08-05)
- https://learn.microsoft.com/en-us/azure/search/search-limits-quotas-capacity (ms.date 2026-09-16)
- https://learn.microsoft.com/en-us/azure/search/vector-search-overview (ms.date 2026-08-05)
- https://learn.microsoft.com/en-us/azure/container-apps/overview (fetched 2026-11-02)
- https://learn.microsoft.com/en-us/azure/container-apps/authentication (fetched 2026-11-02)
- https://learn.microsoft.com/en-us/fabric/data-warehouse/connectivity (ms.date 2026-09-09)
- https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Azure Databricks' and armRegionName eq 'eastus' (fetched 2026-11-02)
- https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Azure Databricks' and armRegionName eq 'westeurope' (fetched 2026-11-02)
- https://prices.azure.com/api/retail/prices?$filter=armRegionName eq 'eastus' and productName eq 'Azure OpenAI OSS Models' (fetched 2026-11-02; meter effective 2026-03-01)
- https://prices.azure.com/api/retail/prices?$filter=armRegionName eq 'eastus' and contains(skuName,'4.1 mini') (fetched 2026-11-02)

**AWS**
- https://aws.amazon.com/bedrock/pricing/ (no date shown; fetched 2026-11-02)
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html (fetched 2026-11-02)
- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html (fetched 2026-11-02)
- https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html (fetched 2026-11-02)
- https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html (fetched 2026-11-02)
- https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html (fetched 2026-11-02)
- https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html (fetched 2026-11-02)
- https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-s3-vectors-generally-available/ (posted Dec 2, 2025)
- https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html (fetched 2026-11-02)
- https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html (fetched 2026-11-02)
- https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html (fetched 2026-11-02)
- https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html (fetched 2026-11-02)
- https://docs.aws.amazon.com/elasticloadbalancing/latest/application/listener-authenticate-users.html (fetched 2026-11-02)

**Google Cloud and Google APIs**
- https://docs.cloud.google.com/bigquery/docs/vector-search-intro (fetched 2026-11-02)
- https://docs.cloud.google.com/bigquery/docs/vector-index (fetched 2026-11-02)
- https://docs.cloud.google.com/bigquery/docs/change-history (fetched 2026-11-02)
- https://docs.cloud.google.com/bigquery/docs/reference/storage (fetched 2026-11-02)
- https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.table.RowIterator (fetched 2026-11-02)
- https://docs.cloud.google.com/vertex-ai/docs/vector-search/overview (fetched 2026-11-02)
- https://docs.cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search (fetched 2026-11-02)
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/data-governance (no date shown; fetched 2026-11-02)
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings (fetched 2026-11-02)
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/use-open-models (fetched 2026-11-02)
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate/openai/auth-and-credentials (fetched 2026-11-02)
- https://docs.cloud.google.com/alloydb/docs/ai/perform-vector-search (fetched 2026-11-02)
- https://docs.cloud.google.com/retail/docs/filter-and-order (fetched 2026-11-02)
- https://docs.cloud.google.com/run/docs/overview/what-is-cloud-run (fetched 2026-11-02)
- https://docs.cloud.google.com/iap/docs/enabling-cloud-run (fetched 2026-11-02)
- https://cloud.google.com/blog/products/ai-machine-learning/introducing-conversational-commerce-agent-on-vertex-ai (2025-09-11)
- https://ai.google.dev/gemini-api/docs/openai (fetched 2026-11-02)
- https://ai.google.dev/gemini-api/docs/pricing (fetched 2026-11-02)
- https://developers.google.com/shopping-content/guides/deprecation-and-sunset (fetched 2026-11-02)

**Snowflake**
- https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview (fetched 2026-11-02)
- https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/query-cortex-search-service (fetched 2026-11-02)
- https://docs.snowflake.com/en/sql-reference/data-types-vector (fetched 2026-11-02)
- https://docs.snowflake.com/en/sql-reference/functions/vector_cosine_similarity (fetched 2026-11-02)
- https://docs.snowflake.com/en/user-guide/snowflake-cortex/vector-embeddings (fetched 2026-11-02)
- https://docs.snowflake.com/user-guide/snowflake-cortex/open_ai_sdk (fetched 2026-11-02)
- https://docs.snowflake.com/en/guides-overview-ai-features (fetched 2026-11-02)
- https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-pandas (fetched 2026-11-02)
- https://docs.snowflake.com/en/user-guide/key-pair-auth (fetched 2026-11-02)
- https://docs.snowflake.com/en/user-guide/streams-intro (fetched 2026-11-02)
- https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview (fetched 2026-11-02)
- https://docs.snowflake.com/en/developer-guide/snowpark-container-services/service-network-communications (fetched 2026-11-02)
- https://docs.snowflake.com/en/release-notes/bcr-bundles/2024_04/bcr-1611 (fetched 2026-11-02)
- https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit (fetched 2026-11-02)
- https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf (effective Oct 29, 2026)

**Search engines and vector stores**
- https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector (fetched 2026-11-02)
- https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion (fetched 2026-11-02)
- https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-elser (fetched 2026-11-02)
- https://www.elastic.co/docs/explore-analyze/query-filter/languages/querydsl (fetched 2026-11-02)
- https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-msearch (fetched 2026-11-02)
- https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/text-similarity-reranker-retriever (fetched 2026-11-02)
- https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-inference-put-custom (fetched 2026-11-02)
- https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/ (fetched 2026-11-02)
- https://docs.opensearch.org/latest/search-plugins/search-pipelines/rerank-processor/ (fetched 2026-11-02)
- https://docs.opensearch.org/latest/search-plugins/search-pipelines/score-ranker-processor/ (fetched 2026-11-02)
- https://solr.apache.org/guide/solr/latest/query-guide/common-query-parameters.html (fetched 2026-11-02)
- https://solr.apache.org/guide/solr/latest/query-guide/query-re-ranking.html (fetched 2026-11-02)
- https://www.algolia.com/doc/api-reference/api-parameters/filters/ (fetched 2026-11-02)
- https://www.algolia.com/doc/rest-api/search/search.md (fetched 2026-11-02)
- https://www.algolia.com/doc/api-reference/api-parameters/hitsPerPage/ (fetched 2026-11-02)
- https://www.algolia.com/doc/guides/ai-relevance/neuralsearch/get-started (fetched 2026-11-02)
- https://documentation.bloomreach.com/discovery/reference/faceting-and-filtering (updated 2026-07-27)
- https://docs.constructor.com/reference/shared-pre-filter-expressions (fetched 2026-11-02)
- https://docs.constructor.com/reference/shared-filter-expressions (updated 2025-11-26)
- https://coveo.github.io/search-ui/interfaces/iquery.html (no date; fetched 2026-11-02)
- https://docs.coveo.com/en/1445/ (returned HTTP 503 on 2026-11-02)
- https://docs.empathy.co/blog/welcome-ai-mode.html (updated 2026-10-30)
- https://commercetools.com/blog/b2b-product-spotlight-storefront-search (2026-05-19)
- https://docs.pinecone.io/reference/api/database-limits/operation-limits (fetched 2026-11-02)
- https://docs.pinecone.io/guides/search/hybrid-search (fetched 2026-11-02)
- https://www.pinecone.io/pricing/ (no date shown; fetched 2026-11-02)
- https://qdrant.tech/documentation/concepts/hybrid-queries/ (fetched 2026-11-02)
- https://github.com/qdrant/qdrant (fetched 2026-11-02)
- https://docs.weaviate.io/weaviate/search/hybrid (fetched 2026-11-02)
- https://github.com/weaviate/weaviate (fetched 2026-11-02)
- https://milvus.io/docs/full-text-search.md (fetched 2026-11-02)
- https://milvus.io/docs/limitations.md (fetched 2026-11-02)
- https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-overview/ (fetched 2026-11-02)
- https://www.mongodb.com/docs/atlas/atlas-vector-search/hybrid-search/ (fetched 2026-11-02)
- https://github.com/pgvector/pgvector (fetched 2026-11-02)

**Commerce platforms and product feeds**
- https://help.shopify.com/en/manual/products/import-export/using-csv (fetched 2026-11-02)
- https://shopify.dev/docs/api/usage/bulk-operations/queries (fetched 2026-11-02)
- https://salesforcecommercecloud.github.io/b2c-dev-doc/docs/current/jobstepapi/html/api/jobstep.ExportCatalog.html (fetched 2026-11-02)
- https://salesforcecommercecloud.github.io/b2c-dev-doc/docs/current/jobstepapi/html/api/jobstep.CatalogDeltaExport.html (fetched 2026-11-02)
- https://learning.sap.com/courses/mastering-the-technical-essentials-of-sap-commerce-cloud/omni-commerce-connect-restful-web-services_a60bebeb-75bc-4a6d-972b-96c17363b909 (fetched 2026-11-02)
- https://docs.commercetools.com/api/projects/productProjections (fetched 2026-11-02)

**Libraries, runtimes and serving software**
- https://arrow.apache.org/docs/python/dataset.html (Arrow v26.0.0 docs; fetched 2026-11-02)
- https://py.iceberg.apache.org/ (fetched 2026-11-02)
- https://delta-io.github.io/delta-rs/latest/ (fetched 2026-11-02)
- https://aws-sdk-pandas.readthedocs.io/en/stable/ (fetched 2026-11-02)
- https://github.com/cloudera/impyla (fetched 2026-11-02)
- https://github.com/trinodb/trino-python-client (fetched 2026-11-02)
- https://github.com/dropbox/PyHive (fetched 2026-11-02)
- https://docs.sqlalchemy.org/en/20/dialects/ (2.0 docs; fetched 2026-11-02)
- https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html (fetched 2026-11-02)
- https://issues.apache.org/jira/browse/SPARK-15348 (resolved 2021-05-25)
- https://issues.apache.org/jira/browse/SPARK-45403 (created 2023-10-03; open)
- https://uvicorn.dev/deployment/ (fetched 2026-11-02)
- https://anyio.readthedocs.io/en/stable/threads.html (fetched 2026-11-02)
- https://fastapi.tiangolo.com/async/ (fetched 2026-11-02)
- https://docs.streamlit.io/develop/concepts/architecture/architecture (fetched 2026-11-02)
- https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/ (dated October 31, 2026)
- https://docs.vllm.ai/en/latest/serving/online_serving/ (fetched 2026-11-02)
- https://huggingface.co/docs/text-embeddings-inference/index (fetched 2026-11-02)
- https://oauth2-proxy.github.io/oauth2-proxy/ (fetched 2026-11-02)

**Models and LLM providers**
- https://huggingface.co/thenlper/gte-small (no date; fetched 2026-11-02)
- https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5 (fetched 2026-11-02)
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (fetched 2026-11-02)
- https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct (fetched 2026-11-02)
- https://huggingface.co/openai/gpt-oss-120b (fetched 2026-11-02)
- https://console.groq.com/docs/models (fetched 2026-11-02)
- https://console.groq.com/docs/rate-limits (no date shown; fetched 2026-11-02)
- https://console.groq.com/docs/your-data (fetched 2026-11-02)
- https://developers.openai.com/api/docs/pricing (fetched 2026-11-02)
- https://company.instacart.com/tech-innovation/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms (Nov 21, 2025)

**Target accounts and grocery assortment sizes**
- Kroger:
  - https://news.microsoft.com/source/2019/01/07/kroger-and-microsoft-partner-to-redefine-customer-experience-introduce-digital-solutions-for-retail-industry/ (2019-01-07)
  - https://www.googlecloudpresscorner.com/2023-01-20-Google-Cloud-and-Deloitte-Boost-Grocery-Associate-Productivity-and-Improve-the-Customer-Experience (2023-01-20)
  - https://empathy.co/assets/media/uploads/resources/kroger-building-and-evolving-ai-search-with-empathy.pdf (undated vendor case study, data to June 2025; fetched 2026-11-02)
  - https://www.elastic.co/industries/retail-ecommerce (fetched 2026-11-02)
  - https://www.elastic.co/elasticon/tour/2020/columbus/protecting-your-cluster-from-your-humans (fetched 2026-11-02)
  - https://ycharts.com/news/story/PRN-CL36718-20211012 (Oct 12, 2021)
- Albertsons:
  - https://news.microsoft.com/source/2019/02/22/albertsons-companies-to-transform-experiences-for-shoppers-with-microsoft-cloud-and-ai/ (2019-02-22)
  - https://www.googlecloudpresscorner.com/2025-09-10-Google-Cloud-Launches-Conversational-Commerce-Agent,-Delivering-AI-Enabled,-Personalized-Shopping-Experiences-for-Customers (2025-09-10)
  - https://www.albertsonscompanies.com/newsroom/press-releases/news-details/2025/Albertsons-Companies-Accelerates-Digital-Transformation-with-the-Albertsons-AI-Shopping-Assistant-Redefining-the-Grocery-Shopping-Experience/default.aspx (2025-12-03)
  - https://www.grocerydive.com/news/albertsons-media-collective-criteo-ai-search-brand-ads-ecommerce-omnichannel/823557/ (2026-06-24)
  - https://www.sec.gov/Archives/edgar/data/0001646972/000164697226000032/aci-20260228.htm (10-K, fiscal year ended 2026-02-28)
- H&M:
  - https://www.googlecloudpresscorner.com/2022-06-30-Google-Cloud-Announces-New-Partnership-with-Global-Fashion-Retailer (2022-06-30)
  - https://www.databricks.com/company/newsroom/press-releases/databricks-launches-data-lakehouse-for-retail-and-consumer-goods-customers (2022-01-13; also names Kroger's 84.51°)
  - https://www.ecommerceexpo.co.uk/news/hms-global-product-information-lead-enhancing-customer-experience-ai-data (27 Aug 2024)
  - https://webbjobb.io/jobb/208490/software-engineer-solution-architect (H&M job ad, 1 Apr 2022)
  - https://centra.com/news/hm-invests-in-centra (7 May 2025)
  - https://retailtechinnovationhub.com/home/2026/9/27/handm-overhauls-its-onsite-search-across-more-than-60-markets-in-partnership-with-voyado (byline 28 Sep 2026)
  - https://www.retail-week.com/technology/handm-has-taken-a-minority-stake-in-a-swedish-retail-technology-platform/7047788.article (2024-12-18)
  - https://hmgroup.com/?p=311 (figures as of 31 Aug 2026)
- C&A:
  - https://www.c-and-a.com/si/sl/corporate/company/careers/jobs/working-student-mfd-onsite-merchandise-1-1-66568-en-gb (undated; fetched 2026-11-02)
  - https://www.snowflake.com/en/blog/c-and-a-build-data-driven-fashion-empire/ (2023-08-17)
  - https://www.casestudies.com/company/vtex/case-study/ca-fashion-retail-ecommerce-at-its-finest (undated; data Jul-Sep 2020; fetched 2026-11-02)
  - https://www.syte.ai/ecommerce-fashion/ (page (c) 2024; fetched 2026-11-02)
- H-E-B:
  - https://careers.heb.com/blog/simplifying-the-assortment-riddle (2025-03-07)
  - https://www.elastic.co/elasticon/tour/2019/austin/elastic-at-h-e-b-providing-a-better-shopping-experience-at-central-market (URL carries 2019; page undated)
  - https://www.algolia.com/customers/h-e-b-mexico (fetched 2026-11-02)
  - https://ktxs.com/news/texas/50000-products-now-available-through-new-h-e-b-online-store (Nov 12, 2015)
  - https://www.supermarketnews.com/retailers/central-market-expands-curbside-home-delivery (Jun 1, 2021)
- Industry: https://fmi.org/our-research/supermarket-facts (2025 figure; fetched 2026-11-02)

**Demo data licensing**
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1 (Mar 8, 2024)

**Internal evidence**
- Repo:
  - CLAUDE.md (the committed file, git HEAD fe206d8; see How to read this);
  - .claude/rules/eval.md and .claude/rules/safety.md;
  - LICENSE; requirements.txt; requirements-databricks.txt;
  - app/ (search.py, vector_search.py, embeddings.py, translator.py, reranker.py, main.py, llm_client.py, llm_memo.py, page_cache.py, hybrid_router.py, keyword_search.py, config.py, assistant.py, recommendations.py, query_guard.py);
  - databricks/ (01_catalog_to_delta.py, 02_vector_index.py, 09_deploy_app.py, 11_idle_guard.py, app_bundle/app.yaml, app_bundle/run_app.py, app_bundle/requirements.txt, README.md);
  - docs/RESEARCH.md; docs/COMPETITIVE_SUMMARY.md.
- Research lanes and their verifier verdicts: SCR/gtm/research/{connectors, retrieval, llm, scalability, architecture, cloud_ops, claims, market}.json and the matching .verify.json files.
- Gap follow-ups:
  - SCR/gtm/research/gap_1.json (cost by scale tier);
  - gap_2.json (incumbents, platforms, assortment sizes);
  - gap_3.json (Hive federation, privileges, pandas 3);
  - gap_4.json (EU regions, English-only components);
  - their extracts in SCR/gtm/gap_4/out/ (azure_matrix_all.txt, aws_matrix_all.txt, lang_probe.json, embed_probe.json, vocab_union.json).
- Probes:
  - SCR/gtm/research/probe_scalability.json (offline, laptop, 2026-11-02);
  - SCR/gtm/research/latency_from_nov02.txt;
  - SCR/gtm/research/arch_sim/sim_patterns_out.json and sim_patterns.py;
  - SCR/gtm/research/retrieval_probe/probe_out.json.
- Earlier reports:
  - SCR/oct27/byo/REPORT.md and SCR/oct27/byo/design_RECOMMENDED.md (bring-your-own catalog, 2026-10-30);
  - SCR/nov02/REPORT.md and SCR/nov02/analyse/score.txt (held-out accuracy and determinism, 2026-11-02).
- Sections 3-8 as checked: SCR/gtm/feasibility/sec3_connectors.txt, sec4_retrieval.txt, sec5_llm.txt, sec6_scalability.txt, sec7_architecture.txt, sec8_cloud.txt, with their .changes.txt logs.
