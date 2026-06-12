# CLAUDE.md — Context-Aware Agentic Search (CAS)

Consulting PoC for LatentView. Two-stage retail product search: an LLM
translates a natural-language query into search intents, embeddings retrieve
candidates, and an LLM reranks them. The selling point is **context-aware**
search ("breathable outfit for a humid day" → cotton/linen items), not
keyword matching. Stakeholders: Sai (dev), Niharika Ganesan (LatentView lead).

## Commands
- Run API: `uvicorn app.main:app --port 8000` (first boot encodes ~60K rows, ~5 min; cached after)
- Run UI: `streamlit run ui.py` (needs the API running on :8000)
- Eval (main): `python -m eval.run_eval` (add `--no-rerank`, `--exclude-description`, `--tag NAME`)
- Load data: `python -m scripts.load_amazon_data` (reads `data/meta_*` JSONL, writes `data/products_amazon.csv`)
- Install: `pip install -r requirements.txt`

## Architecture (the pipeline, in order)
1. `app/translator.py` — query → N search intents. THREE modes; we use `query_expansion`.
2. `app/embeddings.py` — sentence-transformers (`all-MiniLM-L6-v2`) or OpenAI backend.
3. `app/search.py` — cosine retrieval, scatter-gather across intents, dedup. NaN-safe.
4. `app/reranker.py` — LLM reranks top-30 → top-10 with reasons, then blends rating.
5. `app/scoring.py` — Bayesian rating shrinkage + blend into final score.
- `app/llm_client.py` — provider abstraction (Gemini/OpenAI/Anthropic). Reads config DYNAMICALLY (see below).
- `app/key_rotator.py` — multi-key Gemini 429 failover.
- `app/config.py` — ALL env knobs live here; read this first.

## Decisions already made — do NOT silently revert these
- **TRANSLATOR_MODE = query_expansion.** Benchmarked 2026-06-10 vs HyDE and hybrid
  (`eval/compare_translators.py`). query_expansion won decisively: P@1 1.000, NDCG 0.904
  vs HyDE 0.750/0.716. HyDE drifts lexically from short Amazon titles. Hybrid inherits
  HyDE's failures. Keep hyde/hybrid code for re-benchmarking, but query_expansion is the default.
- **RATING_BOOST_WEIGHT = 0.05.** Niharika wants rating "as minimal as possible — not a
  primary filter." It only breaks near-ties. Do not raise without her sign-off.
- **DETERMINISTIC = true** → temperature 0 + fixed seed. This is the answer to Niharika's
  repeated "are results deterministic?" question. `eval/compare_temperature.py` justifies temp=0.
- **Caching is DISABLED on purpose** (early-stage dev). Code exists in `app/cache.py` but is
  commented out at every integration point (search `# CACHE DISABLED`). Every /search hits the
  real LLM so we see true behaviour. Re-enable later by uncommenting those blocks.
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
- `compare_llms.py` — Gemini vs OpenAI vs Anthropic (rerank on; needs keys + SDKs).
- `compare_temperature.py` — temp sweep: quality + run-to-run stability. Auto-disables seed.
- `eval_recipe_completeness.py` — grocery: % of a dish's ingredients in top results (target 70%).
- `stress_test.py` — edge/nonsense/injection queries; checks graceful handling, not precision.
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
- P2: diversity/dedup ("70 paneer sellers" problem — recipe eval grid exposes it).
- P2: cross-sell via Amazon `bought_together` (groceries first, then fashion).
- P3: featured/paid-ad prioritization; pagination/scalability beyond 10 results.
- Backlog: US-locale filter for the catalog (Indian products leak into grocery results).

## Conventions
- Prose comments explaining WHY, not what. Keep functions small and pure where possible.
- New env knobs go in `app/config.py` with a comment, plus `.env.example`.
- Don't add dependencies casually; openai/anthropic are optional/lazy-imported.
- Secrets live in `.env` (gitignored). Never commit keys.
