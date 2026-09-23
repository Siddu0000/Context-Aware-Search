# CLAUDE.md — Context-Aware Agentic Search (CAS)

Consulting PoC for LatentView. Two-stage retail product search: an LLM
translates a natural-language query into search intents, embeddings retrieve
candidates, and an LLM reranks them. The selling point is **context-aware**
search ("breathable outfit for a humid day" → cotton/linen items), not
keyword matching. Stakeholders: Sai (dev),   Ganesan (LatentView lead).

## Commands
- Run API: `uvicorn app.main:app --port 8000` (first boot encodes ~60K rows, ~5 min; cached after)
- Run UI: `streamlit run ui.py` (needs the API running on :8000). `ui.py` is a thin
  `st.navigation` entry point; the real pages are `pages/ai_search.py` ("AI Search" —
  the CAS pipeline), `pages/keyword_search.py` ("Keyword Search" — plain BM25), and
  `pages/shopping_assistant.py` ("Shopping Assistant" — the on-site helper bot).
  Shared rendering lives in `ui_common.py`.
- Eval (main): `python -m eval.run_eval` (add `--no-rerank`, `--exclude-description`, `--tag NAME`)
- Load data: `python -m scripts.load_amazon_data` (reads `data/meta_*` JSONL, writes `data/products_amazon.csv`)
- Install: `pip install -r requirements.txt`

## Architecture (the pipeline, in order)
1. `app/translator.py` — query → N search intents. THREE modes; we use `query_expansion`.
2. `app/embeddings.py` — sentence-transformers (`thenlper/gte-small`) or OpenAI backend.
3. `app/search.py` — scatter-gather retrieval across intents, dedup, NaN-safe. TWO
   backends behind one interface (`SEARCH_BACKEND`): `local` = gte-small + numpy
   cosine; `databricks` = Mosaic AI Vector Search (`app/vector_search.py`). Every
   consumer goes through this module, so the switch moves the whole app AND evals.
4. `app/reranker.py` — LLM reranks a deep pool (RERANK_POOL_K, default 30) with reasons, then blends rating.
5. `app/scoring.py` — Bayesian rating shrinkage + blend into final score.
6. `app/main.py` — paginates the reranked pool (`?page=`, `top_k`=page size); `GET /product?catalog_index=` powers the per-product detail page (product + its own recs).
7. `app/sponsored.py` — featured/paid-ad layer. Reads `data/sponsored.json` (keyed by
   `parent_asin`). **Gate = membership in the RERANKED pool** (`_boost_sponsored`: an ad
   needs a `rerank_score`), so an off-topic ad never shows — e.g. no women's dress on a
   men's-shirt query. Bundles gate within a slot (`_promote_sponsored_options`);
   keyword search gates on BM25 top-k membership. **Placement: gated ads are moved to
   the FRONT of the organic `results` list, bid-ordered, and labelled `is_sponsored` +
   `sponsor`; they are ALSO mirrored into a separate `sponsored` field.** `_bid` is
   stripped before any response. (Corrected 2026-09-23: this line used to say ads were
   "never blended into organic" and gated by a `SPONSORED_REL_RATIO`×median-score
   rule — neither exists in the code. Auditability rests on the `is_sponsored` label.
   Evals are unaffected: run_eval calls the reranker directly and never boosts ads.)
8. `app/recommendations.py` — cross-sell (LLM-proposed complements grounded in the catalog) + upsell (higher Bayesian-rated embedding neighbour). Surfaced on the PRODUCT DETAIL page (per-product), not the results list.
- `app/llm_client.py` — provider abstraction (Gemini/OpenAI/Anthropic). Reads config DYNAMICALLY (see below).
- `app/key_rotator.py` — multi-key Gemini 429 failover.
- `app/config.py` — ALL env knobs live here; read this first.

## Decisions already made — do NOT silently revert these
- **Production LLM = openai/gpt-oss-120b via Groq** (LLM_PROVIDER=openai, OPENAI_BASE_URL=Groq,
  OPENAI_MODEL=openai/gpt-oss-120b). Chosen over Gemini: it matched/beat Gemini on the eval
  (NDCG 0.950 vs ~0.93) and is free with no quota exhaustion. TRADEOFF: it's a reasoning model
  and slow (~20-25s/query) — watch demo latency; gpt-oss-20b or a non-reasoning model is faster.
  Needs GROQ_REASONING_FORMAT=hidden (reasoning tokens else break JSON) and RERANK_INPUT_K=15
  (else Groq 8000 TPM → 429). REVERT to Gemini = set LLM_PROVIDER=gemini (one line).
- **EMBEDDING_MODEL = thenlper/gte-small — comparison done, it WON both rounds.** Retrieval-only
  benchmark (rerank off) vs MiniLM, bge-small-en-v1.5, e5-small-v2, mpnet, bge-m3. gte-small led on
  the Fashion set (NDCG 0.955) AND the 18-query 3-vertical set (NDCG 0.894 vs MiniLM 0.857; P@1/MRR
  1.000), and is fastest of the small models (~33M, CPU). MiniLM only "won" earlier before the strong
  small models were tested. bge-m3 (568M) was worse AND ~5h to encode; 8B impractical on CPU. REVERT =
  EMBEDDING_MODEL=all-MiniLM-L6-v2 (one line). Switching invalidates the on-disk embedding cache ->
  next boot re-encodes 60K once. Shared weak queries ("warm wool sweater", "power bank", "chocolate
  protein bars") are low for every SMALL model. This was recorded as "catalog
  coverage gaps, not a ranking bug" — that was WRONG, disproven 2026-09-11:
  databricks-gte-large-en recovers all three (NDCG 0.93 / 0.78 / 0.80). The
  products were in the catalog; gte-small could not retrieve them. Treat weak
  queries as a retrieval-capacity signal, not evidence of missing data.
- **TRANSLATOR_MODE = query_expansion.** Benchmarked 2026-06-10 vs HyDE and hybrid
  (`eval/compare_translators.py`). query_expansion won decisively: P@1 1.000, NDCG 0.904
  vs HyDE 0.750/0.716. HyDE drifts lexically from short Amazon titles. Hybrid inherits
  HyDE's failures. Keep hyde/hybrid code for re-benchmarking, but query_expansion is the default.
- **RATING_BOOST_WEIGHT = 0.05.**   wants rating "as minimal as possible — not a
  primary filter." It only breaks near-ties. Do not raise without her sign-off.
- **DETERMINISTIC = true** → fixed seed (NOT forced temperature 0). The seed provides
  reproducibility; each call uses its tuned temperature for quality. `effective_temperature`
  in config.py is the single source of truth. `eval/compare_temperature.py` explores the tradeoff.
- **Result caching is ON** via `app/page_cache.py` — caches the ranked pool per query+settings so
  repeat searches and page 2+ are instant. ONLY clean runs are cached; a run where an LLM stage
  fell back (translator/reranker error) is never stored, so a retry hits the LLM again. The old
  `app/cache.py` LRU was removed. The on-disk embedding cache is separate and stays.
- **Data is Amazon Reviews 2023** (McAuley Lab), not the original synthetic eBay catalog. Real
  titles/prices/images/ratings. License is RESEARCH-ONLY — fine for PoC, flag before client demo.

## Data schema gotchas (Amazon Reviews 2023 JSONL — verified against real files)
The McAuley JSONL differs from older docs. `scripts/load_amazon_data.py` already handles these:
- `images` is a LIST of dicts `[{hi_res, large, thumb, variant}]`, NOT a dict-of-lists. Prefer variant=MAIN.
- `details` is ALREADY a dict (not a JSON string).
- `categories` is empty `[]` — infer sub-category from `details.Department` + title gender words.
- `price` is null or float (not a "$x" string).
- `main_category` is UPPERCASE ("AMAZON FASHION").
- Bonus fields kept: `average_rating`, `rating_number`, `store`, `parent_asin`, `bought_together`.
- **`bought_together` is NULL across the ENTIRE dump** (verified 2026-06-12, all three meta_*.jsonl).
  So cross-sell can NOT use real market-basket data. `app/recommendations.py` instead has an LLM
  propose complementary items and grounds them in the catalog via embedding retrieval (decision: Sai,
  2026-06-12). If a future dump populates `bought_together`, prefer it over the LLM path.

## products.csv columns
`bsns_vrtcl_name, categ_lvl2_name, Product_title, img_url, color, material, occasion, price,
prod_description, average_rating, rating_number, store, parent_asin`

## IMPORTANT: llm_client reads config dynamically
`app/llm_client.py` uses `import app.config as cfg` and references `cfg.X`, NOT
`from app.config import X`. This is deliberate: eval scripts mutate `cfg.LLM_PROVIDER` /
`cfg.TEMPERATURE_OVERRIDE` at runtime and need it to take effect. Preserve this pattern.
`cfg.effective_temperature()` resolves the actual temperature (override > deterministic-0 > requested).

## Eval suite (eval/)
- `run_eval.py` — main harness; reads `data/eval_queries.json` (12 multi-context queries).
- `compare_translators.py` — query_expansion vs hyde vs hybrid (rerank off).
- `compare_llms.py` — compares any models on the same eval set (rerank on). Built-ins
  gemini/openai/anthropic; any other string is an OpenAI-compatible model id via OPENAI_BASE_URL
  (Groq recommended — free, no card; current models openai/gpt-oss-120b, openai/gpt-oss-20b —
  the older llama-3.3-70b / qwen3-32b were deprecated 2026-06-17). Pre-flight checks each before
  spending calls. Add RERANK_INPUT_K=15 (env) if a token-limited free tier returns 429s.
- `compare_temperature.py` — temp sweep: quality + run-to-run stability. Auto-disables seed.
- `eval_recipe_completeness.py` — grocery: % of a dish's ingredients in top results (target 70%).
  Reads `data/recipe_eval.json` (5 dishes w/ synonyms — authored 2026-06-12).
- `stress_test.py` — edge/nonsense/injection queries; checks graceful handling, not precision.
  Reads `data/stress_queries.json` (~32 edge cases — authored 2026-06-12).
- `compare_embeddings.py`, `compare_search_text.py` — slower; rebuild embeddings per variant.
- Outputs land in `eval_results/*.csv`. Metric to lead with: P@10 and MRR (NOT R@10 — see below).

## Eval interpretation caveats
- **R@10 looks tiny (~0.03) and that's fine.** Relevant pools are 150-2000 products; top-10 can't
  cover much of that. Precision@10 and MRR are the right headline metrics.
- The old synthetic catalog had label leakage (description templated from the same attrs eval
  checked). The Amazon swap removed it. `--exclude-description` still available as a check.
- A suspected cache-invalidation bug made all `compare_search_text` variants return identical
  numbers on 2026-06-10. Don't trust that comparison until the search-field cache key is verified.

## Gotchas / things that have bitten us
- **Gemini quota burns fast.** Each /search = 2 LLM calls. Free tier ~15 RPM. Multiple keys from
  the SAME Google account share one project quota — they do NOT multiply it. Need DIFFERENT accounts.
- **NaN handling:** pandas reads missing ratings as float NaN, which passes `is not None`. Use
  `math.isnan()` / `pd.isna()`. This bug previously hid the rating count in the UI.
- **Embedding cache** (on-disk, `.cache/`) is keyed by CSV content hash — swapping the catalog
  triggers a ~5-min re-encode on next boot. This is separate from the (disabled) LLM cache.
- Windows: torch CPU-only + MSVC redistributable needed (past WinError 1114 on `c10.dll`).
- **`MIN_RESULT_RELEVANCE` differs by machine and has never actually fired.** The
  local `.env` sets 0.28, but `.env` is gitignored, so Databricks gets the config
  default 0.5. And on gte-small EVERY query scores 0.83–0.94 — gibberish like
  `asdfghjkl` scores ~0.85 (gte's high baseline cosine) — so locally "no products
  found" has only ever come from the translator's gibberish detection. Measure it
  with `python -m eval.calibrate_relevance` rather than trusting any old value.
- **Never add `databricks/__init__.py`.** That folder merges with the pip
  `databricks` namespace package; a regular package there would shadow
  `databricks.sdk` and `databricks.vector_search`.
- **MLflow 3.x refuses the old `./mlruns` file store** (raises unless
  `MLFLOW_ALLOW_FILE_STORE=true`). Locally use `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`;
  Databricks notebooks track to the workspace and are unaffected.
- Databricks returns reasoning replies as a LIST where the OpenAI SDK schema says
  `str`, so MLflow autolog would print a Pydantic serializer warning on every
  call. `tracking.enable_autolog()` silences exactly that message.

## Open work (priority order)
- P2: diversity/dedup ("70 paneer sellers" problem — recipe eval grid exposes it). STILL OPEN.
- ~~P2: cross-sell via Amazon `bought_together`~~ — DONE 2026-06-12 via LLM+embedding (`app/recommendations.py`);
  `bought_together` was empty so we don't use it. Upsell (same-category, higher Bayesian rating) included.
- ~~P3: featured/paid-ad prioritization; pagination beyond 10 results~~ — DONE 2026-06-12
  (`app/sponsored.py` + `?page=` in `app/main.py`). Electronics cross-sell tuning still light.
- Backlog: US-locale filter for the catalog (Indian products leak into grocery results).
- Backlog: real ad inventory to replace the curated `data/sponsored.json` stub.

## Positioning & priorities (Niharika call, 2026-08-13) — READ BEFORE PLANNING
- **CAS is NOT a standalone product.** It is an addition that sits INSIDE the
  client's existing search stack (their website, their ecosystem). Delivery shape:
  swap backend connectors, or package as an API that lets them TOGGLE semantic vs
  contextual search. "How do we plug into their stack" is the recurring question.
- **Target accounts (P0): fashion retailers** (H&M, C&A tier) **and grocery
  retailers** (Kroger, Albertsons, HEB tier). **Walmart/Amazon are benchmarks and
  reference points ONLY — never pitch targets.** (Correcting an earlier note that
  said "small enterprise e-commerce".)
- **Priority order agreed:** (1) helper bot on the client's site — DONE, see below;
  (2) **Databricks migration — non-negotiable**, whole solution, then start GTM
  outreach; (3) **agentic commerce — ON HOLD** pending Niharika's call on whether it
  even belongs in this project.
- **Do not conflate these two** (the main confusion on the call):
  * **Helper bot** = self-service conversational assistant ON the client's website
    (Sphere-style). That's `POST /chat` + `pages/shopping_assistant.py`.
  * **Agentic commerce** = shopper browses/carts/pays INSIDE ChatGPT/Gemini via a
    third-party provider connected to the client's data. Separate workstream, on hold.
    Only thing in common is "search".
- **GTM philosophy:** do NOT build to completion before going to market. Demo early,
  pitch, get validation from search/industry experts, then rework.
- **Always attach source links** to any stat or claim — standing ask.

## Shopping Assistant (helper bot) — 2026-08-13
`POST /chat` (`app/assistant.py`) is a thin CONVERSATIONAL LAYER over the existing
pipeline, not a second engine. One extra LLM call per turn decides search-vs-reply
and rewrites the turn into a SELF-CONTAINED query resolving conversation references
("cheaper ones, for men" -> "men's warm wool sweater winter cheap"), then the normal
`/search` runs — so the bot inherits constraints, rerank reasons, recipe
shopping-lists and sponsored gating for free. Server stays STATELESS: the client
posts `history` back each turn (capped at MAX_HISTORY_TURNS). Shopper text is
inserted as DATA with an explicit anti-injection instruction (verified: it refuses
to leak the prompt).

## Scope decisions + fixes (Niharika call, 2026-08-24)
Scope calls made against docs/COMPETITIVE_SUMMARY.md section A (full table there):
- OUT: A/B testing (another team owns it), visual/image search, multilingual, SLA/SOC2.
- HOLD: purchase-history personalization (returns/cancellation staleness — POS lags
  1-2 days; maybe premium later), store-level inventory, GEO/agent-channel (agentic
  commerce workstream), analytics dashboard (after Databricks).
- DO: persisted catalog attribute enrichment (scripts/enhance_attributes.py exists;
  make it a one-time/seasonal batch), retrieval diversity, autocomplete ONLY if simple
  (verdict: NOT simple in Streamlit — no per-keystroke callbacks without a custom
  component; backend /suggest is trivial later. Deferred).
- Databricks runs IN PARALLEL with local enhancements (databricks/ folder is the
  runbook; blocked on Tarun's credentials). Migrate only what's tested and agreed.
Fixes shipped same day:
- **Gender-skew fix is at RETRIEVAL, not display**: catalog skews ~4.5:1 women's, so
  gender-unspecified apparel queries now retrieve 2x deep and gender-interleave the
  candidate pool BEFORE rerank (`_balance_candidate_pool`) — the reranker's
  RERANK_INPUT_K window otherwise never sees men's items and the post-rerank
  `_balance_by_gender` can't fix what wasn't retrieved.
- **Chat refinement is pool-stable**: "remove the socks" must NOT re-retrieve (that
  dropped pants / pulled in bags). `interpret()` now has a third action "refine"
  (exclusions) → /chat re-pages the CACHED pool of `last_search_query` (zero LLM
  calls), filters exclusion terms deterministically (substring + naive plural fold),
  backfills to top_k. Client echoes `last_search_query` + `exclusions` each turn
  (server stays stateless); a new-topic search resets exclusions.
- **Chat UI shows ONE results panel** (pages/shopping_assistant.py): transcript is
  text-only bubbles; new search REPLACES the panel, refine UPDATES it in place.
  Products never pile up turn after turn ("the page gets lengthy" complaint).
- **Topic boundaries in chat** (Sai, 2026-08-26): `interpret()` also returns
  `new_topic`. History is only kept while the shopper stays on one goal — an
  UNRELATED request (reunion outfits -> pancake ingredients) sets new_topic=true, and
  the UI then CLEARS the transcript to just that exchange (fresh page) AND the prompt
  writes `search_query` from the latest message alone so the finished topic can't leak
  into the new search. Continuations ("cheaper ones") and refines keep the transcript.
  new_topic is forced True when there's no prior search, False for refine/reply, and
  True on LLM failure (never silently inherit context we couldn't interpret).
  Verified: unrelated turn -> 2 bubbles; related turn -> 4 bubbles retained.
  Also fixed here: the client used to send the current message BOTH as `message` and
  as the last `history` entry — the model saw it twice, muddying the judgement.
- **No example prompts in the assistant copy** (Sai, 2026-08-26): the greeting is just
  "Hi! What are you shopping for today?" and the caption carries no sample queries.
  Do not reintroduce "try: something breathable…" style hints.

## Bundles, cart, surface isolation (Sai, 2026-08-27)
- **BUNDLES generalise the recipe pattern.** `bundle_type` = `recipe` | `outfit` |
  `setup` | None, returned by `/search`, `/chat` and `/unified_search`. The backend
  machinery was ALREADY generic: `_recipe_slots_with_alternatives()` groups by
  `source_intent`, so one card per component with 3 options works unchanged for all
  three kinds. What was added is translator-side (OUTFIT_PROMPT / SETUP_PROMPT +
  `detect_bundle_type()` regex fast-path) and label-side (`ui_common.BUNDLE_UI`).
  * outfit -> garment SLOTS (top/bottom/footwear/outerwear/accessory). Gender or age
    stated in the query is pushed into EVERY slot phrase; never mix genders in one
    outfit; gender-neutral when unstated. Kids/boys/girls work via the same rule.
  * setup -> devices AND the peripherals people forget (cables, stands, surge).
  * Bundles deliberately SKIP `_balance_candidate_pool` / `_balance_by_gender` — an
    outfit must stay gender-coherent, unlike a generic apparel query.
  * `is_recipe` is kept as a derived alias (`bundle_type == "recipe"`) so the eval
    suite and older callers keep working. Don't delete it.
- **Advice-shaped questions are SEARCHES, not chat replies.** "What should I wear
  to an interview?" was being answered with prose. `app/assistant.py` now says so
  explicitly; the assistant must never name products in `reply` (it can't see the
  catalogue).
- **SURFACE ISOLATION.** Each page calls `ui_common.set_surface("ai"|"keyword"|
  "assistant"|"cart")` and all state goes through `sget/sset/sinit`, which namespace
  keys as `<surface>__<key>`. Verified: no cross-surface key leakage. Opening a
  product in the assistant no longer moves AI Search. NEVER go back to bare
  `st.session_state["view"]` in a page.
- **Cart** (`ui_cart.py` + `pages/cart.py`) is session-only and deliberately SHARED
  across surfaces — one basket is what a shopper expects; it's the single exception
  to isolation. Add-to-cart sits on grid cards, the product detail card AND the
  cross-sell strip. Many catalog rows have a null price, so the cart reports
  "N items without a listed price" instead of silently under-totalling.
- **Do NOT use `st.page_link` in a sidebar.** It raised `KeyError: 'url_pathname'`
  outside full page context and would have taken the page down whenever the cart was
  non-empty (invisible in manual testing because the cart starts empty). Cart is a
  top-level nav item; a caption is enough.
- **Verify UI with `streamlit.testing.v1.AppTest`, not the browser.** Driving
  Streamlit's `text_input` via synthetic browser events does not commit reliably;
  AppTest runs the page in-process and catches real errors (it found the page_link
  bug). Remember `sys.path.insert(0, os.getcwd())` — AppTest doesn't put the project
  root on the path the way `streamlit run` does.

## UI naming conventions (2026-08-12)
User-facing copy is PLAIN LANGUAGE; pipeline internals are shown only behind each
page's **"Developer details"** toggle. Keep it that way — Niharika's standing ask is
that demos stay clean.
- Nav/pages: **AI Search** 🧠 and **Keyword Search** 🔍 (previously the main page
  showed as a bare "ui" because auto-discovery names pages after the FILENAME —
  fixed by `st.navigation` in `ui.py`).
- Cards: "**93% match**" (rerank blend), "≈ 90% similar" (embedding-only),
  "🔤 Keyword match" (BM25). Dev toggle appends `rerank / Bayes / embed / BM25`.
- Sections: "🧠 What we searched for" (intents), "🛍️ Results", "🧺 Shopping list"
  for recipes, "🧺 Frequently bought together", "⬆️ Better-rated alternative".
- Backend `reason` strings that are pipeline notes ("(beyond reranked pool …)",
  "(rerank disabled)") are mapped to nothing via `ui_common.friendly_reason()` —
  the backend strings are UNCHANGED because evals/tests match on them.
- `is_recipe` MUST come from the API response, never inferred from `source_intent`
  (which is set on every result for every query — inferring it labelled a
  wool-sweater search as a "Shopping list").

## Databricks migration — VERIFIED RESULTS (2026-09-11)

Ran end to end on Azure workspace `adb-7855330416659580`, catalog `dev`, schema
`cas`. Everything below is measured, not projected.

**Both models are now Databricks-hosted. No external API key, no Groq.**
- embeddings: `databricks-gte-large-en` (1024-dim), managed delta-sync index
  `dev.cas.products_index` — the platform embeds the `search_text` column, so
  `app/embeddings.py` and the local encode are unused on this path.
- LLM: `databricks-gpt-oss-120b` via Foundation Model APIs. Reached through the
  existing OpenAI-compatible backend (`OPENAI_BASE_URL` = `<host>/serving-endpoints`,
  `OPENAI_API_KEY` = a Databricks token). Those env names describe the PROTOCOL,
  not the vendor — nothing leaves the workspace.

### Retrieval quality: gte-large-en BEATS gte-small on every metric
18-query 3-vertical set, rerank OFF, clean run (no degraded queries):

| metric | gte-small 384d (local) | gte-large-en 1024d | delta |
|---|---|---|---|
| P@1     | 0.944 | **1.000** | +0.056 |
| P@10    | 0.917 | **0.944** | +0.027 |
| MRR     | 0.972 | **1.000** | +0.028 |
| NDCG@10 | 0.927 | **0.955** | +0.028 |
| ms_total| 3,230 | 3,797 | +567 (+18%) |

MRR 1.000 = the top hit was relevant for all 18 queries. **DECISION: use managed
embeddings.** The argument for a self-managed index (preserving gte-small's
numbers) is dead — gte-large-en is simply better. Latency is +18% on the full
pipeline, NOT the 10x an earlier note implied; that comparison wrongly used an
isolated retrieval microbenchmark (~373ms) against a full-pipeline total.

Lost on managed embeddings: the index embeds ONE column fixed at creation, so
`--exclude-description` and `compare_search_text` have no cheap equivalent, and
each extra embedding contender costs a whole index (see timing below).

### Provider quirks that cost us three failed runs — all now handled in code
1. **`seed` is REJECTED, not ignored.** `DETERMINISTIC=true` sends a fixed seed;
   FM APIs 400 with `unknown field "seed"`. Gated on the base URL in
   `llm_client.py`, same as GROQ_REASONING_FORMAT. Groq/OpenAI still get it.
2. **Content comes back as TYPED BLOCKS, not a string.** Reasoning models return
   `[{"type":"reasoning",...},{"type":"text","text":"{...}"}]`, so `raw.strip()`
   raised `AttributeError: 'list' object has no attribute 'strip'`.
   `_message_text()` drops reasoning blocks and keeps text ones — the structural
   equivalent of Groq's `reasoning_format=hidden`, which does not exist here.
3. **Claude endpoints REJECT JSON mode entirely** (`Response format type json_object is
   not supported for this model`, 2026-09-23) — both Haiku and Sonnet runs silently
   measured the no-LLM baseline. `_OpenAIBackend` now retries once without
   `response_format`, remembers the model in `_NO_JSON_MODE`, restates "JSON only"
   in the prompt, and `_parse_json_reply()` tolerates fences and prose around the
   object. A `finish_reason=length` reply now raises "truncated", not "invalid JSON".
3b. **The prompt MUST contain the literal word "json"** when
   `response_format={"type":"json_object"}` is sent, else 400. All 9 current
   prompts satisfy this by luck ("Output ONLY valid JSON"). A new prompt saying
   "return a dict" would work on Groq and 400 here.
4. **Index build takes ~2 HOURS for 300K rows** (~2,350 rows/min), not "minutes".
   MEASURED COST (system.billing.usage, list prices, 2026-09-23): **one clean build
   ≈ $20** — Sep 11's successful build + eval was $20.27. The Sep 10 build that went
   OFFLINE_FAILED cost $23.39 on its own (cause never pinned down), which is why the
   first round totalled ~$44. Split: ~79% is the sync/embedding pipeline at
   $0.45/DBU (unlabelled VECTOR_SEARCH rows); ~21% is the `cas-search` endpoint at
   $0.07/DBU: 130.75 DBU = $9.15 over Sep 10-11, but the endpoint existed only
   PART of each day (~29-48h), so the true idle rate is **~$5-8/day**, not the
   $4-5 first recorded here. Rule: keep the index up if (days until next use ×
   ~$7) < ~$20 -- a break-even of ~3 days -- else delete and rebuild. Once the app
   is deployed it must stay up: ~$150-240/month at list price. `02` now uses
   TRIGGERED sync (the catalog is static) so an idle index costs only the endpoint.
   When querying billing as a shared account, `current_user()` returns that
   account's whole usage (e.g. $23.59 of unrelated Genie) — attribute by endpoint.

### Why (1) and (2) were dangerous rather than just annoying
Both failed INVISIBLY: `translate_query` degrades to the raw query on error, so
the eval measured raw-query retrieval and reported it as the CAS pipeline —
plausible numbers (NDCG 0.654), completely invalid. Same failure that produced
the bogus `gemma2-9b-it` row in `eval_results` (metrics identical to the
embedding-only baseline, 66ms "rerank"). `run_eval.py` now threads an errors
list through translate AND rerank, writes a **`fell_back`** column per query, and
prints a loud banner. **Never read an eval number without checking `fell_back`.**

### Running the eval on Databricks (databricks/) — full runbook in databricks/README.md
`vs_shim.patch_eval_harness(spark)` sets `cfg.SEARCH_BACKEND="databricks"`, loads the
catalog and points eval CSVs at the UC Volume. There is NO monkey-patching any more:
the earlier shim patched `run_eval` with a parallel retrieval function that returned
only 11 index columns — fine for the eval, but in the app it would have blanked
`img_url` and broken sponsored matching (keyed by `parent_asin`). Now the index
returns only `catalog_index` + score and rows are HYDRATED from the catalog, so both
backends return identical row shapes and the eval runs the app's real code.
- `04_run_eval.py` — retrieval eval; `--rerank` for the full pipeline
- `05_compare_llms.py` — `list_models()` then `run([...])`; rerank ON, so it
  tests intent generation AND reranking. Endpoint NAMES, not vendor model ids.
- `06_compare_translators.py` — query_expansion vs hyde vs hybrid, rerank OFF,
  isolating intent generation. The local verdict (query_expansion 0.904 vs HyDE
  0.716) was measured on gte-small and is worth re-checking on gte-large-en.
Catalog/schema come from `CAS_CATALOG` / `CAS_SCHEMA` (default `dev`/`cas`).

### Still open
- `catalog_index`: FIXED in `01` (dense `row_number()` over file order + an
  assertion). The live table `dev.cas.products` was VERIFIED dense and aligned with
  the local CSV on 2026-09-23 (the multiLine read was a single partition), so `01`
  does NOT need re-running — only `02`. `vector_search.load_catalog()` refuses a non-dense key loudly,
  because hydration and `GET /product` use it as a ROW POSITION.
- `MIN_RESULT_RELEVANCE` not yet calibrated for the Vector Search score scale —
  run `databricks/07` and set the env var for the deployed app (see Gotchas).
- LLM comparison (05) run 2026-09-23 on the KEYWORD set: Qwen3-Next matched
  gpt-oss-120b (NDCG 0.953 vs 0.943) at 2.9x the speed (10.4s vs 30.3s/query) and
  3.3x fewer completion tokens -- but a no-LLM control scored the SAME (0.954), so
  that set cannot rank LLMs on quality. Re-run on `data/eval_queries_context.json`
  (see .claude/rules/eval.md). Provisional: Qwen for the demo, on latency alone.
- **The SETUP bundle over-triggers** the way OUTFIT did before Aug 31: "watch
  movies on my bedroom wall" becomes a 6-part home theatre (projector, mount,
  screen, soundbar, HDMI cable, streaming player); "keep my phone alive camping" a
  4-part setup. Niharika's principle (don't make the upsell obvious) suggests gating
  setups behind an explicit ask too -- a PRODUCT decision, not yet made or changed.
- Local-only bug, pre-existing: `_build_search_text` lowercases display columns IN
  PLACE, so the local app shows titles like "yullser wireless mouse" and blanks
  become "". The Databricks backend reads the Delta table and is unaffected -- so
  the two backends now DISPLAY differently (and dedup by title differently).
- The app itself is not yet deployed (`app.yaml` untested). Databricks Apps have
  no Spark, so `load_catalog()` falls back to `VS_CATALOG_CSV` on the UC Volume —
  confirm the App can read `/Volumes/...` before relying on that path.

## MLflow + real token accounting (2026-09-23)
- `app/llm_client.py` now records the PROVIDER-REPORTED usage of every call
  (`resp.usage`, Gemini `usage_metadata`, Anthropic `usage`) — including
  **reasoning tokens**, which are billed but never visible in the reply. Collect it
  with `with track_usage() as u:`; scopes nest additively and are ContextVar-based,
  so a scope opened in async middleware sees calls made in the threadpool endpoint.
  `approx_tokens` (`len//4`, input only) is kept only so old CSVs stay comparable.
- Every HTTP response carries `X-LLM-Calls` / `X-LLM-Prompt-Tokens` /
  `X-LLM-Completion-Tokens`, and requests with LLM calls log counts (never the
  query text — safety.md).
- Eval CSVs gained `llm_calls, tokens_prompt, tokens_completion, tokens_reasoning,
  tokens_total`; the summary prints totals.
- `app/tracking.py` (opt-in, `MLFLOW_ENABLED=true` + mlflow installed; otherwise
  every call is a no-op): `mlflow.openai.autolog()` traces each LLM call, the request
  middleware groups one request's calls into ONE trace (verified:
  `GET /search` -> 2 nested `Completions` spans), and `run_eval.evaluate()` logs one
  MLflow run per eval — params (models/backend/modes), metrics (quality + tokens),
  the CSV as an artifact, tag `valid=true|false` from `fell_back`.
- Tracking can NEVER break a search or an eval: every mlflow call is wrapped, and a
  failed `start_run` just runs untracked (verified against a real failure).
- MLflow metric keys reject `@`: `P@10` is logged as `P_at_10`.
- **Databricks does not break out reasoning tokens** (verified 2026-09-23: a
  `{"ok": true}` reply cost 61 completion tokens, `reasoning_tokens` 0). gpt-oss's
  reasoning is billed INSIDE `completion_tokens`, so `tokens_reasoning=0` there means
  "not reported", not "no reasoning". Compare models on `tokens_completion` /
  `tokens_total`. Groq and api.openai.com do report it.
- Traces go to the workspace MLflow store (100K-trace limit). Fine for evals; a
  DEPLOYED app traces every request, so decide on Unity Catalog trace storage or
  sampling before turning MLFLOW_ENABLED on in production.


## Conventions
- Prose comments explaining WHY, not what. Keep functions small and pure where possible.
- New env knobs go in `app/config.py` with a comment, plus `.env.example`.
- Don't add dependencies casually; openai/anthropic are optional/lazy-imported.
- Secrets live in `.env` (gitignored). Never commit keys.
