# Eval methodology (load when working on evaluation)

## How relevance is defined
`eval/run_eval.py::build_relevance_set` resolves each query's `relevance`
criteria against the catalog. Supported keys (AND-combined):
- `title_must_include` — all terms in Product_title
- `title_any_of` — any term in Product_title
- `color_in` / `material_in` / `occasion_in` — exact match on that column
- `category_in` — matches bsns_vrtcl_name OR categ_lvl2_name

`data/eval_queries.json` holds 12 MULTI-CONTEXT queries (Niharika's 2026-06-11
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
- Run with cache OFF for honest latency (cache is disabled anyway right now).
- If you need many runs, get a paid key or space them out.

## Known eval issues
- `compare_search_text` returned identical numbers across all field variants on
  2026-06-10 — suspected the embedding cache key doesn't include the field set,
  so every variant reused the first variant's embeddings. Verify the cache key
  in `app/search.py` before trusting that comparison.
- Old synthetic catalog had label leakage; the Amazon swap fixed it. The
  `--exclude-description` flag remains as a leakage probe.

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
