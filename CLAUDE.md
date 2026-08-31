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
2. `app/embeddings.py` — sentence-transformers (`all-MiniLM-L6-v2`) or OpenAI backend.
3. `app/search.py` — cosine retrieval, scatter-gather across intents, dedup. NaN-safe.
4. `app/reranker.py` — LLM reranks a deep pool (RERANK_POOL_K, default 30) with reasons, then blends rating.
5. `app/scoring.py` — Bayesian rating shrinkage + blend into final score.
6. `app/main.py` — paginates the reranked pool (`?page=`, `top_k`=page size); `GET /product?catalog_index=` powers the per-product detail page (product + its own recs).
7. `app/sponsored.py` — featured/paid-ad layer. Reads `data/sponsored.json`, returns a SEPARATE `sponsored` list (never blended into organic — see safety.md). RELEVANCE-GATED: an ad shows only if its similarity to the query intents ≥ SPONSORED_REL_RATIO×median organic score, else none (fixes off-topic ads like a women's dress on a men's-shirt query).
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
  protein bars") are low for ALL models = catalog coverage gaps, not a ranking bug.
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

## Conventions
- Prose comments explaining WHY, not what. Keep functions small and pure where possible.
- New env knobs go in `app/config.py` with a comment, plus `.env.example`.
- Don't add dependencies casually; openai/anthropic are optional/lazy-imported.
- Secrets live in `.env` (gitignored). Never commit keys.
