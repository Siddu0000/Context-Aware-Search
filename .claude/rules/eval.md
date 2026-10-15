# Eval methodology (load when working on evaluation)

## How relevance is defined
`eval/run_eval.py::build_relevance_set` resolves each query's `relevance`
criteria against the catalog. Supported keys (AND-combined):
- `title_must_include` — all terms in Product_title
- `title_any_of` — any term in Product_title
- `color_in` / `material_in` / `occasion_in` — exact match on that column
- `category_in` — matches bsns_vrtcl_name OR categ_lvl2_name
- `title_none_of` — NO term in Product_title (added 2026-09-23). Needed because
  Amazon titles are keyword-stuffed: "roku remote", "airtag case", "nightgown" for
  gown, "lamp for projector". Build relevance from `get_dataframe()`, never a fresh
  `read_csv`: `_build_search_text` lowercases titles IN PLACE on the local backend.

**TWO eval sets — never compare scores across them** (MLflow param `eval_set`):
- `data/eval_queries.json` — 18 KEYWORD queries, relevance = product-type word
  only. SATURATED: raw retrieval with NO LLM scores P@10 0.944 locally, NDCG 0.954
  on Databricks — the same as the full pipeline. It cannot see what the LLM does.
- `data/eval_queries_context.json` — 16 INFERRED-intent queries (2026-09-23): the
  product type never appears in the query ("keep my coffee hot" -> travel mug).
  Raw retrieval P@10 0.325 vs query expansion 0.487 (+50%), NDCG 0.342 -> 0.517
  (local gte-small, Groq). THIS is the set that measures CAS's value and can
  separate LLMs. Select with `EVAL_QUERIES_JSON=data/eval_queries_context.json` or
  `run_eval --queries`. Every rule was checked against sampled catalog titles.
  Three queries are BUNDLE-SENSITIVE (see `_context`): a translator that builds a
  'setup' for "watch movies on my wall" is penalised, deliberately.

Historical: the ORIGINAL `data/eval_queries.json` (commit f844792, v4) held 12
fashion-only MULTI-CONTEXT queries ( 's 2026-06-11
ask: every query carries 2-3 context signals like weather + activity + fabric).
The relevance criteria encode the CORRECT interpretation of the context, e.g.
"breathable top for a humid day" → `material_in: [cotton, linen, rayon]`.
The `_context` field in each query is documentation only; the builder ignores it.

## Metrics — what to report and how to read them
- **Lead with P@10 and MRR.** These are the credible headline numbers.
- **R@10 is intentionally tiny (~0.03).** Relevant pools are large (hundreds to
  thousands of products); top-10 mathematically can't cover much. Low R@10 is
  NOT a failure — don't let anyone misread it that way.
- **NDCG@10** rewards correct ordering; good secondary metric.

## Quota discipline for eval runs
Each /search = 2 LLM calls. The full eval set with rerank = ~24 calls/run.
Gemini free tier is ~15 RPM and will throttle. Guidance:
- `compare_temperature` and `compare_translators` default to rerank OFF
  (translator-only, ~12 calls/run) to stay light.
- Latency is honest: run_eval calls the pipeline stages directly and never touches the
  app's `page_cache` (which IS on in the app).
- Databricks pay-per-token limits are far above eval volume: ITPM 1M / OTPM 100K for
  Qwen3-Next on Enterprise-tier workspaces (tier-dependent; our tier is not recorded) per
  https://learn.microsoft.com/en-us/azure/databricks/machine-learning/foundation-model-apis/limits
  -- the constraint there is cost, not quota.
- If you need many runs, get a paid key or space them out.

## Known eval issues
- **Single runs are noisy.** The translator is non-deterministic (Groq's seed is
  best-effort; Databricks gets no seed at all) — the same query produced different
  intents on two calls. Treat gaps under ~0.05 P@10 on 16 queries as a tie.
- **Local and Databricks run DIFFERENT LLMs** (since 2026-09-24): local = Groq
  `openai/gpt-oss-120b`, Databricks = `databricks-qwen3-next-80b-a3b-instruct`. A local number
  and a Databricks number differ in embedding model AND LLM -- never attribute the gap to one.
  Read run_eval's printed `llm = <model>` summary line (every run) or `llm_model_used`
  (MLflow, when enabled), not the notebook's Cell 2. CSV names carry the model only for
  04/05/06 and compare_translators; local run_eval's default is `rerank_on_<ts>.csv`.
- `run_eval --no-translate` is the NO-LLM CONTROL. Run it alongside any model
  comparison — without it you cannot tell whether the LLM added anything.
- `compare_search_text` returned identical numbers across all field variants on
  2026-06-10 — suspected the embedding cache key doesn't include the field set,
  so every variant reused the first variant's embeddings. Verify the cache key
  in `app/search.py` before trusting that comparison.
- Old synthetic catalog had label leakage; the Amazon swap fixed it. The
  `--exclude-description` flag remains as a leakage probe.

## App-path eval (`eval/eval_page_types.py`, 2026-10-14)
- **run_eval cannot see app/main.py.** It calls translator -> `search_products` ->
  reranker directly and scores the reranker's own list, so the ordering main.py applies
  around the reranker -- the rerank window, gender balancing, display sort, bundle slot
  order, sponsored boosts -- never runs there. That is where the type-aware ranking fix
  lives (CLAUDE.md, TYPE_AWARE_RANKING); run_eval sees it only through the prompt edits,
  the narrowed setup-routing regex (`detect_bundle_type`, reached via translate_query ->
  understand_query) and the reranker's `type=` field.
- eval_page_types calls `app.main.search` in-process (page 1, 12 cards, rerank on,
  sponsored on, no cross-sell) and scores what a shopper sees. Flat pages:
  `mismatch_top4` / `mismatch_top12` (share not of the expected type), first mismatch,
  context fit (title words that betray the setting: 'bridal' or 'wool' at a beach
  wedding), vertical share, wrong gender. Bundles: routing (`bundle_ok`), good/bad
  components in the intents and slot primaries, `garment_slots_first`, mixed gender.
  Queries: `data/page_type_queries.json` (34; groups stakeholder / case / collision /
  routing; each entry's `_context` says what it checks). No relevance labels by default.
- `--mode both` (the default) runs `TYPE_AWARE_RANKING` off then on per query and repeat.
  The on arm REPLAYS the off arm's translation and any rerank call with the same window
  (catalog_index order) and constraints, errors included -- the arms differ by ordering
  policy, not LLM sampling; the `memo` column says what was replayed (latency blank for
  it). A different window or constraint list (the on arm can add `[product_type]`) is a
  real call. "off" is NOT the pre-change code: prompts, routing, the gender classifier
  and `type=` sit outside the knob. A true before = a worktree at the previous commit
  with only this script, its JSON and `app/product_type.py` copied in (one 'head' arm).
  Every run starts from an EMPTY page_cache (its key ignores repeats).
- `--relevance data/eval_queries_context.json` alone runs THAT set's queries and adds
  served-page P@4 / P@12 from `build_relevance_set` criteria -- the context set scored on
  the page the App serves, not the reranker's list. With `--queries` too, criteria are
  matched by query text.
- Cost: 2 LLM calls per query run; `both` adds only the reranks it cannot replay (34
  queries ~ 70-100 calls per repeat). On Databricks: `databricks/10_page_eval.py`.
- **Groq free tier = 200K tokens/day PER MODEL** (TPD, 429 "tokens per day"). On
  2026-10-14 gpt-oss-120b ran out mid-baseline (30 of 68 rows degraded): a full
  before/after (HEAD arm + both arms + the context set, rerank on) does not fit in one
  day on one model. Split across days, never across models.
- **gpt-oss-20b is unusable for rerank comparisons**: with rerank ON it fell back on
  5 of 16 context-set reranks (2026-10-14: 4 Groq json_validate_failed 400s, 1 truncated
  reply). Translator-only (`--no-rerank`) runs work,
  but are noisy: an untouched grocery query swung P@10 0.9 -> 0.0 between two runs.

## Recipe-completeness eval (grocery)
`eval/eval_recipe_completeness.py` + `data/recipe_eval.json`. For each dish,
search "ingredients to make <dish>", inspect top-K titles, count how many
canonical ingredients (with synonyms) appear. Target 70% coverage. The
per-ingredient grid exposes the "70 paneer sellers" diversity problem: you'll
see the headline ingredient FOUND but others MISSING when retrieval lacks
diversity. Needs grocery rows loaded in products.csv.

## Stress test
`eval/stress_test.py` + `data/stress_queries.json`. Edge/nonsense/injection/
unicode queries. Measures GRACEFUL HANDLING (no crashes; injection strings
treated as inert text), not precision. A healthy run reports 0 hard errors.
