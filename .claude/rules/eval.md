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
  gown, "lamp for projector". Build relevance from `get_dataframe()` (the frame
  retrieval serves), never a fresh `read_csv`. Until 2026-10-15 `_build_search_text`
  lowercased titles IN PLACE on the local backend; both backends now keep the
  seller's case and the builder matches case-insensitively. Relevance sets are sets
  of TITLES, so two titles differing only by case now count separately: n_relevant
  grew by 1 on 5 keyword and 3 context queries (R@10 equal to 6 decimals; P@k, MRR,
  NDCG and every 30-row retrieval pool identical, no-LLM runs, measured before the
  `title_key` dedup). Retrieval dedups them (`search.title_key`), so R@10 cannot
  reach them both.

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
- **Rerank-on runs across 2026-10-15 are not strictly comparable**: local rerank and
  cross-sell prompts now carry the seller's case (`row_text_case`: run_eval logs it as an
  MLflow param only, when MLFLOW_ENABLED -- its CSV never carries it, so absence there
  proves nothing; eval_page_types writes a CSV column, 'absent' on older code), the
  rubric's guest line and the QE/SETUP prompts changed (`prompts_sha`), and
  `outfit_bundles` picks the general prompt (and the outfit route).
- **Nor across 2026-10-19**: rubric 8aa34da0 -> 8e7b7f67, typed Fashion rerank lines carry
  ` | subtype= | audience=` (`row_text_case` 'original+fine_fields'), QE a56801f1 -> 94fffb8d
  (`prompts_sha`). Cost: +15.6% tokens per live rerank-on query (8 queries, 3.8K -> 4.4K mean;
  30,571 -> 35,334 in all): the QE prompt adds ~245 prompt tokens per query, the rerank call ~375
  (longer rubric plus subtype= / audience=; by size about 200 of it is the rubric text);
  completions -197 in all. run_eval sees those three; it cannot see
  the new window partition, the guest constraint or the display knobs (REJECTS_AFTER_TAIL,
  VARIANT_COLLAPSE), which only eval_page_types and the apparel battery run.
- **Nor across 2026-10-27**: QE 94fffb8d -> d3e184c6 (a DEVICE and a GIFT bullet; 4,729 -> 5,859
  chars), rubric 8e7b7f67 -> 089fb621 (4,175 -> 4,838 chars), `row_text_case`
  'original+fine_fields+role' (an acted device window's lines carry ` | role=device|accessory`;
  Databricks lines with blank fields no longer read "None" / "nan", local lines byte-identical).
  `prompts_sha` is now 0af957a6337a, and it moved once with NO prompt edit: the digest
  (`llm_memo.prompts_digest`, which run_eval's `_prompts_sha()` and the start-up log share) now also
  hashes COMPLEMENTARY_PROMPT (the snapshot's own texts: f448928b67aa -> 1685ce703d24), so compare
  the per-prompt shas across that date, not the digest. Cost, by size: QE ~+285 prompt tokens per
  translate call (at the draft 45138beb), rubric ~+165 per rerank call, ` | role=` ~3 per acted
  window line; live on the qwen app path, e01-e14 spent 5,113 tokens per query vs 4,649 on Oct 27
  (+10%; Electronics battery). run_eval sees the QE and rubric edits; the device ask, the role
  fields and the gift reserve (DEVICE_ROLE_RANKING, in the page_cache key) act in main.py, so only
  eval_page_types and the batteries see them.
- **The LLM reply memo must be OFF in every eval** (LLM_MEMO, 2026-10-27, default ON in the app): a
  memo hit replays the first accepted reply with 0 provider calls, so a repeat, a model / prompt /
  temperature comparison or a stability run would report one pinned sample as the model's behaviour.
  The scripts ensure it: every LLM-calling harness turns it off before its first call
  (`llm_memo.disable_for_eval()`: run_eval's `evaluate()`, which databricks/04-06 go through, the
  compare_* scripts, calibrate_relevance (so 07), eval_crosssell, eval_recipe_completeness,
  stress_test; `eval_page_types.run` sets `cfg.LLM_MEMO = False`). run_eval records `llm_memo` (an
  MLflow param, when MLFLOW_ENABLED) and a per-query `memo_hits` CSV column, and prints
  `llm_memo = off (N memo hits)`, with a "!! replies were REPLAYED" warning when N > 0;
  eval_page_types writes `llm_memo` / `memo_hits` columns (its older `memo` column is the off/on arm
  replay, not this memo). databricks/08 runs with it off (the notebook's value comes back
  afterwards) and FAILs any row a memo hit answered; 10 calls
  `vs_shim.patch_eval_harness(spark, memo=False)` and raises if it is still on; vs_shim prints a
  `memo :` header line. Traps: `disable_for_eval()` and `eval_page_types.run` leave it off for the
  rest of a notebook process, and app calls a notebook makes before any harness or
  `patch_eval_harness` (memo=False by default) run with it ON (Cell 2 sets no LLM_MEMO: SQLite
  under /Workspace). The one harness that wants it on is check_determinism, in its own memo dir
  (Determinism check below).
- `run_eval --no-translate` is the NO-LLM CONTROL. Run it alongside any model
  comparison — without it you cannot tell whether the LLM added anything.
- `compare_search_text` returned identical numbers across all field variants on
  2026-06-10. The embedding cache key DOES include the field set (`app/search.py`
  `_cache_key`, since v2 2026-06-03; verified 2026-10-15), so that was not the cause;
  the cause is unknown, so don't trust that comparison until it is re-run.
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
  Queries: `data/page_type_queries.json` (48 since 2026-10-27, was 38; groups stakeholder / case /
  collision / routing / bridal / device; each entry's `_context` says what it checks). No relevance
  labels by default.
  2026-10-15: the 4 'bridal' rows ("wedding dress", "wedding costume", "bride costume",
  the rajasthan query: apparel, flat, a bridal `context_any_of`) check the guest-rule
  narrowing; the 6 outfit rows now expect flat apparel. The outfit metrics
  (`garment_slots_first`, mixed gender) exist only with OUTFIT_BUNDLES_ENABLED on, and
  `expected_bundles()` follows that knob (on: 'outfit' is also accepted for a row the
  regex routes there), so the documented revert is not scored as misrouting. CSV columns
  `outfit_bundles` and `row_text_case` ('absent' on older code); since 2026-10-19 also
  `rejects_after_tail`, `variant_collapse` and `prompts_sha` (run_eval's digest; 'absent' on
  older trees), printed in the summary. 2026-10-27: 10 'device' rows ("gift for a vintage watch
  collector", "laptop for college", "airtag", "iphone", "noise cancelling headphones for travel",
  "watch winder", "stop my laptop from overheating", "women's watch", "ipad", "headphone pads"),
  scored by `core_flags` (`device_role.is_core`; a pre-change worktree's rows get the taxonomy
  columns derived first, so before and after are scored by one ruler).
- `--mode both` (the default) runs `TYPE_AWARE_RANKING` off then on per query and repeat (only
  that knob: both arms run REJECTS_AFTER_TAIL and VARIANT_COLLAPSE as cfg has them).
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
- Cost: 2 LLM calls per query run; `both` adds only the reranks it cannot replay (48
  queries ~ 100-145 calls and ~0.24-0.38M tokens per repeat, ESTIMATED; the 38 were ~ 80-115
  calls). On Databricks: `databricks/10_page_eval.py`.
- **Groq free tier = 200K tokens/day PER MODEL** (TPD, 429 "tokens per day"). On
  2026-10-14 gpt-oss-120b ran out mid-baseline (30 of 68 rows degraded): a full
  before/after (HEAD arm + both arms + the context set, rerank on) does not fit in one
  day on one model. Split across days, never across models. gpt-oss-120b is also the
  local DEMO model (the Oct 15 meeting repro alone spent ~107K of its day): no evals on
  it within a day of a demo.
- **gpt-oss-20b is unusable for rerank comparisons**: with rerank ON it fell back on
  5 of 16 context-set reranks (2026-10-14: 4 Groq json_validate_failed 400s, 1 truncated
  reply). RE-CONFIRMED 2026-10-15: 5 of 9 live /search reranks fell back, and "wedding
  costume" failed on all 4 tries (2 truncated at max_tokens in fixed-intent runs -- a
  15-row app-path window, then an 8-title direct probe; 2 json_validate_failed live, on
  different translations and pools), so the guest-rule narrowing is unverified live.
  Translator-only (`--no-rerank`)
  runs work (0/16 fell back on 2026-10-15),
  but are noisy: an untouched grocery query swung P@10 0.9 -> 0.0 between two runs.
  2026-10-19: the app path again lost 4 of 7 reranks (3 json_validate_failed, 1 truncated;
  a direct probe returned 3 of 15 entries); translator-only, 2 of 60 calls hit
  json_validate_failed (a retry worked), and the same prompt and seed gave different
  intents on two calls (single-query P@10 swings of +-0.4 are noise there).

## Apparel battery (2026-10-19): the graded app-path before/after
- **What.** 41 apparel-led queries (`battery.json`, each with an `expect`) through
  `app.main.search` in-process, pages 1 AND 2 at 12 cards, rerank + sponsored on, local
  gte-small, Groq `qwen/qwen3.8-27b` ONLY (not the demo model; Databricks untested). It sees
  everything main.py does around the reranker.
- **Query ids** (CLAUDE.md and this file cite them; the `expect` per query lives only in the
  archive's `battery.json`). Wedding: m01 "wedding costume", m02 "beach wedding outfit", m03
  "cloths for a beach wedding in summer", m04 "Wedding dresses for a destination wedding in
  rajasthan", m05 "summer beach wedding outfit dress". Plain types: p01 "black dress", p02 "men's
  slim fit jeans", p03 "white sneakers for women", p04 "linen shirt for men", p05 "women's winter
  coat", p06 "kids rain jacket", p07 "silk saree". Context: c01 "what to wear to a job interview
  for men", c02 "breathable top for a humid day", c03 "something warm for a ski trip", c04 "comfy
  clothes for a long flight", c05 "office wear for women", c06 "date night dress", c07 "music
  festival outfit", c08 "gym clothes for women", c09 "funeral attire for men", c10 "Diwali outfit
  for women", c11 "black tie event men", c12 "business casual for men", c13 "hiking clothes for
  rainy weather", c14 "beach vacation clothes for women". Demographic: d01 "plus size jeans for
  women", d02 "maternity dress for a baby shower", d03 "toddler girl party dress", d04 "teenage
  boy back to school clothes", d05 "petite work pants". Attributes: a01 "sleeveless summer
  dress", a02 "dress that is not black", a03 "cotton t-shirt under $20", a04 "waterproof boots for
  snow". Gift / style: g01 "gift for my boyfriend who likes streetwear", g02 "matching couple
  outfits", g03 "halloween costume for kids". Shoes / bag / jewelry: x01 "handbag for work", x02
  "running shoes for flat feet", x03 "minimalist gold jewelry".
- **Where.** OUTSIDE the repo, with its cache and labels: the 2026-10-19 session archive, an
  `oct19/` folder in that Claude Code session's TEMP scratchpad on the dev laptop (the
  machine-specific path is deliberately not checked in). `audit/` = BEFORE and the synthesis,
  `measure/` = AFTER; paths below are relative to `oct19/`. TEMP gets cleaned: persisting it is
  Open work (CLAUDE.md).
- **Harness** `run_battery.py`: one patched `chat.completions.create` records / replays every
  call in `llm_cache.json`, keyed by sha256(model + exact messages + response_format +
  temperature); a reply is cached only when its stage was accepted (a fallen-back query is
  retried, `--attempts`); page 2 must come from page_cache (0 LLM calls; checked, flagged with
  `!! PAGE 2 NOT FROM CACHE` and listed in the summary -- not asserted; `--resume` re-runs such a
  query); a token ledger enforces `--cap`; `--replay-only` makes a miss an error; `--only`,
  `--resume`. ~3.7-4.4K tokens per fresh query: 41 queries = one Groq qwen day (200K TPD). Stage
  latencies are mostly 429 pacing: read `llm_calls_detail[].latency_ms`.
- **Grading.** Two LLM grader LENSES (shopper, merchandiser: two prompts of one LLM, not humans)
  label every card 0 / 1 / 2 with failure tags (wrong_type, attribute_violation,
  wrong_occasion_formality, duplicate_variant, bad_order, wrong_age, costume_or_novelty,
  wrong_season_material, bridal_vs_guest, junk_listing, wrong_gender) and judge the intents.
  Merged label = the lens mean; P@12 strict = share of 2 (a 2/1 split counts 0.5), lenient =
  share >= 1; "both-agree" strict needs 2 from both. BEFORE agreement: exact 0.908, kappa 0.859
  (quadratic 0.925), never 0 vs 2. Labels judge each card alone: P@12 cannot see variety or
  gender balance (only `duplicate_variant` sees duplicates). Labels are reused by (query id,
  catalog_index); grade only NEW pairs, same lenses and rubric, and report coverage. The GRADER
  MODEL IS NOT RECORDED: the lenses were agent sessions of the Claude Code workflow that ran the
  audit, and the saved files hold the outputs (`audit/grades.json`, six grader outputs; the
  2026-10-20 labels in `measure/live/make_grades.py` -> `new_grades.json`) and the card dumps they
  read (`audit/grade_*/dump.txt`), not the model id or the lens prompts. Grading of the FULL
  after-battery must use sessions that did not write the fixes (ideally another model), blind to
  run and rank as in the re-grade below, and record the model id and prompts. The 2026-10-27
  round's graders read HIGHER on 103 shared apparel items (shopper +0.07, merchandiser +0.17):
  never set its apparel numbers beside these (Electronics battery, "Grader calibration").
- **BEFORE** (QE a56801f1, rubric 8aa34da0, 0 fell back, 158,823 tokens): page 1 strict 0.477 /
  lenient 0.869, page 2 0.279 / 0.618; wedding asks 0.283 / 0.875; worst x02, p03, m04 (0.00),
  c03, c11, c09, g01, m03, p07, c07. The rerank score is calibrated: > 40 mean label 1.41,
  rejected (<= 40) 0.32, never reranked 1.01 (rr 0-30: 0.14 ... 86-100: 1.81). Headroom: an
  oracle reorder of the 24 rows 0.676 / 0.975; 24 judged rows instead of 15 ~0.58 / 0.92 (Monte
  Carlo from the reranker's own score-per-label distribution, not live; 109 page-2 rows both
  lenses graded 2 were never reranked). The cost of RERANK_INPUT_K 24-30 (owner call 8, CLAUDE.md
  Open work), estimated, not measured: ~+40-70% rerank tokens per query -- the 2026-10-20 live
  15-row rerank calls averaged ~2.5K prompt / ~0.58K completion tokens (20,194 / 4,656 over 8
  calls), the template ~1K of the prompt by size, so ~100 prompt and ~40 completion tokens per
  extra row: ~3.4K prompt at 24 rows, ~4.0K at 30.
- **Replay** (display / window code, 0 LLM): `measure/replay/replay_measure.py` patches the
  snapshot's `QUERY_EXPANSION_PROMPT`, `RERANK_PROMPT_TEMPLATE` and `_type_field` into the live
  modules (it does not swap whole modules), so prompts AND rerank row text (no subtype= /
  audience=) are byte-identical to the BEFORE cache, and runs `--replay-only`. A changed rerank
  window or constraint list misses: list those queries
  (they need live calls) or answer them from BEFORE scores by catalog_index (a "memo" reranker:
  approximate, and its match_quality is meaningless). `audit/synth/eval_variants.py` scores pages 1
  and 2: an ungraded row is bounded as 0 (lo) or 2 (hi) and imputed with the query's mean label of
  graded unjudged rows; always print coverage.
- **AFTER, replay** (old prompts, new code): 13 of 41 replay exactly (28 miss: 21 window seats, 5
  seats + constraints, 2 the guest constraint only). On the 13, page 1 strict / lenient: BEFORE
  0.532 / 0.929, knobs off 0.532 / 0.929, REJECTS_AFTER_TAIL 0.551 / 0.946, VARIANT_COLLAPSE
  0.545 / 0.923, both (shipped) 0.571 / 0.952, no page-1 loss; page 2 strict 0.394 -> 0.456
  imputed (lo 0.353, hi 0.571, 78% graded). Knobs off restores pages 1+2 on 11 of 13 (g02 / g03
  differ by one tail row: the window's age partition). Memo, all 41 (approximate): 0.477 -> 0.525.
- **AFTER, live** (2026-10-20, QE 94fffb8d + rubric 8e7b7f67, 8 of the worst 10: m04 x02 p03 c03
  c11 c09 g01 m03 -- p07 and c07 not run; m01, also on the Measure list, not run either (budget);
  16 calls, 35,334 tokens, 0 fell back): page 1 strict 0.120 -> 0.339, lenient 0.719 -> 0.901;
  page 2 0.083 / 0.443 -> 0.125 / 0.620. Per query, page-1 strict: c03 0.08 -> 1.00, c11 0.21 ->
  0.58, m03 0.25 -> 0.54, p03 0.00 -> 0.12, m04 / c09 / g01 unchanged (m04 0.00); x02 lenient
  0.25 -> 0.71. The same replies with both display knobs off give 0.323 / 0.875: +0.203 strict
  came from upstream (prompt, window, rubric, guest constraint), +0.016 strict / +0.026 lenient
  from display (c09 lenient 0.88 knobs off -> 0.75 on). Those labels came from the measuring
  session, which knew the intended fixes. BLIND RE-GRADE (2026-10-20, `oct19/blind/`): the union
  of BEFORE and AFTER page-1+2 rows of the 8 queries (322 items) was shuffled per query with rank,
  scores, reasons and run hidden (`blind_sheet.json`; key kept apart), and two fresh agent sessions
  (shopper, merchandiser lenses; model not recorded, same caveat as above) labelled every item;
  `score_blind.py`, mean of the two lenses: page 1 strict 0.125 -> 0.339, lenient 0.646 -> 0.844;
  page 2 0.089 / 0.411 -> 0.130 / 0.557; agreement exact 0.904, within 1 1.000. Per query page-1
  lenient: m04 0.54 -> 0.92, x02 0.29 -> 0.71, p03 0.33 -> 0.62, m03 0.54 -> 1.00, c03 strict 0.08
  -> 0.96; losses: c09 lenient 0.83 -> 0.75, c11 / p03 / g01 page 2. `match_quality.weak` fired
  on 0 of the 8 (the base rule, WEAK_MATCH_MIN_STRONG=0).
- **Cross-vertical guard** (required for any ranking change, 0 LLM): 16 xvertical + 16 context
  queries with recorded qwen intents and a deterministic fake reranker (md5(catalog_index) mod
  100 + 1), BEFORE code (snapshot main.py + product_type.py) vs AFTER with the display knobs off
  and on. 2026-10-19: all 25 non-Fashion queries keep window, constraints and knobs-off display
  identical; knobs on, only rejected rows move after the tail (1-9 per query), 0 non-Fashion rows
  collapse. Fashion rows change by design (2 context windows, the guest constraint on the gala).
- **Translator A/B** (prompt edits; gpt-oss-20b, translator only, 1 sample per arm, calls paced
  >= 16 s): swap `translator.QUERY_EXPANSION_PROMPT` (`general_prompt()` reads the module global)
  in an isolated tree; a 0-LLM window proxy runs the snapshot's `_gender_stated` ->
  `search_products` -> `_effective_request` -> `_flat_window` on each arm's intents, so the arms
  differ only by intents. Title-regex target / drift counts are proxies, not grader labels.
- **Owed: the full after-battery** on a fresh qwen day with no other qwen use:
  `measure/full/run_full_after.sh` (copies the harness and the after-cache, so the 8 live queries
  replay at 0 tokens while code and prompts are unchanged; `--cap 190000 --resume`: finish the
  next day if capped; ~146K tokens, ~181K if all 41 go live). Check meta prompt sha 94fffb8d /
  8e7b7f67, `degraded` [] and the 8 replayed ids; grade the ~650 new pairs
  (`measure/live/grade_dump.py`, `make_grades.py`'s format, `score_live.py`) and compare with
  BEFORE 0.477 / 0.869, 0.279 / 0.618. Never split it across models.

## Electronics battery (2026-10-27/30): device asks, the graded before/after
- **What.** The Oct 27 meeting's watch-gift ask ("gift for a vintage watch collector" showed boxes,
  winders and cleaners) and its "redo the electronics category": 24 queries
  (`live_c/battery_c.json`, each with an `expect`) through `app.main.search` in-process, pages 1 AND
  2 at 12 cards, rerank + sponsored on, LLM_MEMO off, local gte-small, Groq `qwen/qwen3.8-27b` ONLY,
  one sample per arm (not the demo model; Databricks untested). BEFORE: e01-e14 = the 2026-10-27
  live run on the same model and the pre-round code (QE 94fffb8d, rubric 8e7b7f67); c03 / m03 / c11
  / x02 = the Apparel battery's 2026-10-20 live run; w01-w06 have NO before. AFTER: 2026-10-30, QE
  d3e184c6, rubric 089fb621, DEVICE_ROLE_RANKING on. Harness `live_c/run_battery.py` (an LLM record
  / replay cache, `llm_cache_c.json`, and a token ledger, `token_ledger_c.json`); `analyse_after.py`
  writes `after_summary.md`.
- **Query ids** (the `expect` per query lives only in `battery_c.json`). Watch: e01 "gift for a
  vintage watch collector" (the meeting query), e02 "watch for men", e03 "smart watch for running".
  Electronics: e04 "airtag", e05 "iphone", e06 "laptop for college", e07 "noise cancelling
  headphones for travel", e08 "wireless earbuds under $50", e09 "bluetooth speaker for the beach",
  e10 "camera for vlogging", e12 "power bank for camping", e13 "tablet for kids". Gift: e11 "gift
  for a gamer", e14 "gift for a photographer". New this round, AFTER only: w01 "gift for my dad who
  loves watches", w02 "women's watch", w03 "dress watch for men", w04 "apple watch band", w05
  "pocket watch", w06 "gift for someone who collects cameras". Apparel regression (Apparel battery
  ids): c03, m03, c11, x02. "With a family" = names a D2 device family (watch, laptop, headphones,
  phone, tablet, camera, speaker): the 9 plain device asks e02, e03, e05-e10, e13, plus e01 and
  w01-w06 (w04 asks for an accessory: `accessory_asked`, no device ask). "No family" = e04, e11,
  e12, e14 (a tracker, a power bank and a profession are no family).
- **AFTER, the classifier read** (`after_summary.md`; "core" = `device_role.is_core`, the app's own
  classifier, on both arms, not a grader label; e12 on the Oct 27 title-regex proxy). 23 of 24
  valid (m03 fell back: see "m03 after the F1 fix"); page 2 from page_cache with 0 LLM calls on 23
  of 24. e01: page 1 0 -> 12 of 12 watches, pages 1+2 24/24, first watch #19 -> #1, the reranker
  judged 12 core rows of 15 (was 0); intents `vintage mechanical watch` / `vintage watch display
  case` / `vintage watch repair kit`. The 9 plain device asks: core share top 4 0.92 -> 1.00, top 12
  0.86 -> 0.94. e05 "iphone": 3 iPhones at #1-#3 (rerank 95/85/80; the catalog holds only 3 iPhone
  handsets), 3 other phones #4-#6, #7-#12 unjudged MacBooks / iPods / iPads; weak banner True ->
  False; core top 4 0.25 -> 1.00, top 12 0.08 -> 0.50. w01-w06: core at #1, top 4 = top 12 = 1.00
  (w02 12/12 women's watches, all typed `accessory`; w04 12/12 bands; w06 12/12 cameras). The device
  ask acted on every e/w query naming a family and nowhere else (not w04, the 4 apparel queries,
  airtag, gamer, power bank, photographer). No sponsored card on any page 1 (`data/sponsored.json`
  is stale).
- **Gift add-ons (D1: 3 window seats for non-core intents, storage first, care last) do NOT open
  page 2 on qwen.** The reserve seats on e01 / w01 / w06 were judged <= 35 ("capped as an
  accessory/storage piece below actual watches"), so under REJECTS_AFTER_TAIL they sit after the
  unjudged tail: pages 1-2 = 24/24 core. An owner question (amend D1, exempt storage on a gift ask,
  or send the reserve rows without the role field).
- **Unchanged or worse (classifier read).** e04 "airtag" unchanged: 12 holders / mounts scored
  65-92, and the weak banner stays off. e12 0.75 -> 0.25 on the regex proxy (new intents pulled
  speakers and power-bank cases; no device code involved). e14 0 cameras both times. e11 lost its
  mouse-pad intent. Taxonomy residuals seen live (3 camera straps typed `watch_accessories` on e14,
  one Apple Watch band typed `watch_kind` watch on w04) are fixed offline by F2 and NOT re-run live
  (nor e13).
- **Blind grading** (`grade_c/`; lead with it). Two LLM grader lenses (shopper, merchandiser)
  labelled every card of both arms' pages 1+2, blind: the sheets in `sheet/`, the key apart
  (`key/_key_do_not_read.json`), and the grader-facing `expect` rewritten for e01, e04, e05, e08,
  w01, w02, w04, w05, w06 and m03 to keep the graders blind (originals in the key). 775 items
  (electronics 505, watch_gift 144, apparel 126), coverage 1.000 (`score.py --require-full`). P@12
  strict / lenient = the lens mean, as in the Apparel battery; core@4 / core@12 = share of the
  page-1 top 4 / top 12 that BOTH lenses tagged `core` (the item itself, not an add-on for it). A
  fallen-back arm is out of every mean; "paired" = both arms valid.

| group (blind, BEFORE -> AFTER) | n | p1 strict | p1 lenient | p2 strict | p2 lenient | core@4 | core@12 | lenses agree exact / within 1 |
|---|---|---|---|---|---|---|---|---|
| electronics e01-e14 (paired) | 14 | 0.69 -> 0.68 | 0.84 -> 0.87 | 0.60 -> 0.62 | 0.73 -> 0.82 | 0.80 -> 0.84 | 0.73 -> 0.75 | 0.915 / 1.000 |
| watch/gift w01-w06 (AFTER only) | 6 | 0.95 | 1.00 | 0.72 | 0.93 | 1.00 | 1.00 | 0.944 / 1.000 |
| apparel paired c03, c11, x02 | 3 | 0.51 -> 0.47 | 0.85 -> 0.86 | 0.24 -> 0.22 | 0.56 -> 0.62 | - | - | 0.937 / 1.000 |
| apparel paired + m03 fixed | 4 | 0.53 -> 0.50 | 0.89 -> 0.90 | 0.26 -> 0.25 | 0.65 -> 0.70 | - | - | 0.934 / 1.000 |
| all paired (`score.txt`) | 17 | 0.66 -> 0.64 | 0.84 -> 0.87 | 0.54 -> 0.55 | 0.70 -> 0.79 | | | |
| all paired + m03 fixed | 18 | 0.66 -> 0.64 | 0.85 -> 0.87 | 0.53 -> 0.54 | 0.71 -> 0.80 | | | |

- **Read.** Electronics (n=14) deltas: page-1 strict -0.012, lenient +0.027, page-2 strict +0.018,
  lenient +0.095, core@4 +0.036, core@12 +0.024 (all paired, n=18 with m03 fixed: -0.016, +0.023,
  +0.012, +0.086). e01: page-1 strict 0.00 -> 0.50, page-2 lenient 0.33 -> 1.00, core@4 0.00 ->
  0.50, core@12 0.00 -> 0.67. The 9 plain device asks: core@4 0.917 -> 0.972, core@12 0.889 ->
  0.917 (mostly e05: core@4 0.25 -> 0.75). Asks with no family (e04, e11, e12, e14) hold every core
  loss: core@4 0.750 -> 0.625, core@12 0.542 -> 0.396.
- **Losses and their causes** (`grade_c/loss_detail.txt` has the rows; it quotes titles: keep it out
  of shared places). e12 "power bank for camping" page-1 lenient 0.75 -> 0.33: the translator
  dropped the high-capacity USB-C intent, "rugged waterproof" took 9 of 15 window seats, and no
  device code acted. e14 "gift for a photographer" page-1 strict 0.83 -> 0.17: strap rows took 12 of
  15 window seats by score, and a profession is outside D1. x02 page-1 lenient 0.75 -> 0.67, page-2
  lenient 0.50 -> 0.25: a new `arch support` intent brought trail and hiking shoes, two real running
  shoes fell to window order 17 and 19, and the Revels scored 40, not 45 (-> the rejected band). m03
  degraded (fixed: next bullet). Smaller: e02 strict 1.00 -> 0.83 (smartwatches labelled 1), c11
  strict 0.54 -> 0.38 / lenient 0.79 -> 0.92, e06 page-2 strict 1.00 -> 0.79, e13 a page-2
  kids-tablet case typed a device (fixed offline by F2). e12 and e14 are the live evidence for the
  untyped window's seat cap (D6, not this round).
- **m03 after the F1 fix.** m03's AFTER fell back on both attempts: qwen answered the rerank with a
  top-level JSON array and the reranker crashed on it. Since F1 a top-level array IS the ranking
  (`app/reranker.py`); other non-object replies (translate, cross-sell, assistant) are
  `LLMError "Bad shape: reply is <type>"`. Re-run 2026-10-30 (`oct30_D_m03`; translation replayed
  from a copy of the harness cache, 1 live rerank call, 3,414 tokens): 0 errors, page 2 from
  page_cache at 0 calls; the same intents and 15-row window as 2026-10-20, page 1 the same 12 cards
  as BEFORE in another order, page 2 identical; 159 results / 14 pages / 168 pool rows,
  match_quality not weak (band0 10, strong 7, best 92), no bridal gown on page 1. Blind P@12 (this
  round's labels), BEFORE vs AFTER-fixed: page 1 0.58 / 1.00 both, page 2 0.33 / 0.92 both, top-4
  strict 0.88 both, coverage 24/24 (no new item to grade); the degraded AFTER had read 0.50 / 0.83.
  Its reply sits only in `live_c/m03_fixed/llm_cache_m03.json`: merge it into `llm_cache_c.json`
  before a whole-battery replay.
- **Grader calibration** (read before comparing rounds). The graders are LLM agent sessions of the
  same model family, not people; the model id is not recorded. On the 103 apparel items both rounds
  labelled, this round's graders read higher than the Apparel battery's blind graders (2026-10-20,
  `oct19/blind/`): shopper exact 0.932, mean new - old +0.07; merchandiser exact 0.825, +0.17. So
  this round's apparel numbers must NOT sit next to the Apparel battery's blind numbers (before /
  after WITHIN a round is fine): m03 BEFORE page-1 strict is 0.46 on the 2026-10-20 labels and 0.58
  on this round's. `after_summary.md` also reads c03 / c11 / x02 on the 2026-10-20 labels: those
  belong with the Apparel battery, not this table. Lens quirks: e04 agreement exact 0.04 (shopper 25
  of 26 holders = 1, merchandiser 26 of 26 = 0; the page is identical before and after); e01 core@12
  is 0.67, not 1.00, because the shopper lens read "wristwatches" literally and did not tag the 4
  pocket watches core. P@12 cannot see order (c03 shows the same 24 items in both runs). The Oct 27
  BEFORE page-1 grades that `after_summary.md` section 1b reuses were ONE non-blind grader.
- **Tokens** (provider-reported). qwen3.8-27b on 2026-10-30: 157,967 in 69 calls -- the battery's
  main run 119,010 / 47, x02 run on its own (translate 1,561 / 1, rerank 3,343 / 1), the determinism
  check 30,639 / 19, m03's re-run 3,414 / 1 -- under Groq's 200K/day per model. The battery (main
  run + x02): 123,914 in 49 calls; per query, final attempts, 4,891 over all 24 (m03 and x02 made 1
  live call each) and 5,113 over e01-e14 against 4,649 on Oct 27 (+10%, 28 calls each); translate
  ~1,500 prompt tokens, rerank ~2,630-3,690 prompt. gpt-oss-20b (the QE A/B): 118,335 in 60 calls
  (cap 120,000). gpt-oss-120b (the demo model): 0 calls; a preflight to it was refused before any
  traffic.
- **Where.** OUTSIDE the repo: the 2026-10-27 session archive, an `oct27/` folder in that Claude
  Code session's TEMP scratchpad on the dev laptop (machine path kept out, as for `oct19/`), 555 MB
  in all; paths in this section are relative to it. `SPEC.md` (decisions D1-D6); `before_snapshot/`
  (2.2 MB: the only byte-level baseline of the pre-round UNCOMMITTED tree; git HEAD is not it);
  `understand/` (the Oct 27 BEFORE run: `watch_electronics_live.*`, `grades_page1.json`,
  `devices.py`); `live_c/` (5.0 MB: battery, cache, results, `after_summary.md`,
  `token_ledger_c.json`, `m03_fixed/`, `qe_ab/`, `determinism/`); `grade_c/` (874 KB: `sheet/`,
  `labels/`, `key/`, `score.py`, `score.txt`, `per_lens.*`, `loss_detail.*`); `build/`
  (`NOTES_*.md`, `VERIFY*.md`, `phC_result.json`, `FACTS_OCT27.md`). TEMP gets cleaned: persisting
  those is owed, as for `oct19/` -- but not `understand/taxonomy/`, the `build/` scratch trees or
  any memo folder.
- **Owed.** A translator A/B on e12 (did the QE DEVICE bullet's "waterproof" variant cause the
  drift?); the F2 rows live (e13, e14, w04 were checked offline only); repeats (one sample per arm);
  everything on the demo model gpt-oss-120b and on Databricks (gte-large-en + Qwen3-Next: 08's new
  rows, 10's 10 device rows and 12 never run).

## QE A/B 94fffb8d vs d3e184c6 (2026-10-30)
- **What.** The 2026-10-27 QE edit (a DEVICE bullet and a GIFT bullet; 94fffb8d -> the draft
  45138beb -> d3e184c6, 4,729 -> 5,859 chars) on BOTH eval sets: translator only (rerank off), Groq
  `openai/gpt-oss-20b`, paired per query, 1 sample per arm. The arms differ only by the
  `QUERY_EXPANSION_PROMPT` constant, swapped in-process, and every sent prompt was sha-checked
  against its arm. The no-LLM control ran alongside and equals run_eval's per query. Files (archive
  above): `live_c/qe_ab/` (`run_qe_ab.py`, `before_prompt.py`, `summarize.py`, `results.json`,
  `summary_tables.md`, `moved_intents.md`, `headline.json`, `csv/`).

| set | arm | n | P@1 | P@10 | MRR | NDCG@10 | fell_back | partial | tokens |
|---|---|---|---|---|---|---|---|---|---|
| context | control (no LLM) | 16 | 0.375 | 0.319 | 0.526 | 0.324 | - | - | 0 |
| context | BEFORE 94fffb8d | 16 | 0.562 | 0.450 | 0.638 | 0.470 | 0 | 0 | 28,667 |
| context | AFTER d3e184c6 | 16 | 0.500 | 0.463 | 0.615 | 0.481 | 0 | 0 | 36,184 |
| keyword | control (no LLM) | 18 | 0.944 | 0.944 | 0.972 | 0.950 | - | - | 0 |
| keyword | BEFORE | 14 | 1.000 | 0.986 | 1.000 | 0.989 | 0 | 0 | 25,086 |
| keyword | AFTER | 14 | 1.000 | 0.979 | 1.000 | 0.986 | 0 | 0 | 28,398 |

- **Read.** Context: P@10 +0.013 (a TIE by the 0.05 rule), MRR -0.023, NDCG@10 +0.011, P@1 -0.062;
  6 wins / 6 losses / 4 ties. Keyword: a tie on a saturated set, over 14 of 18 queries only (the
  token cap stopped it; 4 of its 6 grocery queries did not run), so its arm rows and its control row
  (18) are not over the same queries. The two big context swings are the model's own recipe routing:
  "breakfast I can make in two minutes before work" 1.0 -> 0.0, "cut sugar from my coffee without
  losing the sweetness" 0.1 -> 1.0. 0.450 is the first eval-set number for the shipped 94fffb8d
  text; single gpt-oss-20b samples are noisy (App-path eval, "gpt-oss-20b").
- **What the bullets reach.** Beyond the device families: on a food gift ("a gift for someone who
  loves spicy food") AFTER wrote a "spice jar display case" intent (P@10 0.3 -> 0.2); on an inferred
  device ("watch movies on my bedroom wall") AFTER wrote three projector variants. The GIFT bullet
  and the rubric's GIFTS sentence reach every persona gift and sit under no knob (only
  OUTFIT_BUNDLES_ENABLED=true drops the QE bullet): an owner question.
- **Cost.** AFTER adds ~256 prompt + ~214 reasoning tokens per translate call on the context set
  (+13-26% per call); the A/B spent 118,335 tokens in 60 calls (cap 120,000).
- **Not measured here.** The rubric (089fb621) and the app-path device / gift code (rerank was off,
  and run_eval cannot see main.py: the Electronics battery covers them on qwen); the demo model;
  Databricks.

## Determinism check (`eval/check_determinism.py`, 2026-10-27)
- **What.** The Oct 27 meeting asked that the same search give the same results. The variation is
  the LLM's, and the app pins it with the reply memo (LLM_MEMO, default ON). The check runs
  `app.main.search` pages 1 and 2 (12 cards, rerank, ads and filters as the UI sends) and
  `app.main.product` on the page-1 top card, `--runs` times, each COLD (page_cache, the degraded
  store and the /product cache cleared; the memo is not: that is the test), and compares intents,
  constraints, bundle type, both pages' catalog_index lists, variants, ad labels, facets,
  match_quality, the result count and the product page's cross-sell / upsell / source.
  - `--memo on` (default): identical snapshots on every run and 0 provider calls on runs 2..N, else
    exit 1. `--memo off`: the model's raw stability instead (page-1 overlap@12, top-4 overlap,
    Kendall tau on the shared cards, identical intents, the fallback rate). `--subprocess`: run 1 in
    a fresh Python process with another PYTHONHASHSEED (stands in for an API / App restart). A query
    whose run 1 fell back is "not pinned": retried (`--attempts`), never compared. `--out` writes
    the snapshots (catalog indexes, intents, facets) for a later diff; the script never logs or
    saves reply text.
  - The memo lives in `--memo-dir`: a new temp dir by default, never the app's LLM_MEMO_DIR and
    never the App's memo volume (`/Volumes/<catalog>/<schema>/app_state`, refused in any spelling: a
    check must not pre-warm, or read, what the API or the App serves); another /Volumes path needs
    `--allow-volume`.
  - Usage, as run live: `python -X utf8 -m eval.check_determinism --memo on --runs 3 --subprocess
    --memo-dir <scratch>/memo --queries <file>.json` (`--query TEXT` is repeatable; with neither,
    the meeting's queries plus the head of `data/page_type_queries.json`). COST: real LLM calls, 3
    per query in run 1 (translate, rerank, cross-sell); with `--memo off` every run pays.
  - Databricks: `databricks/12_determinism.py` `run(queries=None, runs=3, memo=True,
    memo_dir=None, volume="evals", attempts=2, product=True)`, its memo under
    `/Volumes/<CAS_CATALOG>/<CAS_SCHEMA>/evals/determinism/<user>/<UTC>/llm_memo` (never app_state);
    the restart case is `run(memo_dir=...)` after restartPython (it does not use `--subprocess`,
    which cannot work from a notebook); `cleanup(folder)`. Default 21 queries; run 1 ~63 calls /
    ~0.11M tokens ESTIMATED, runs 2+ 0 calls. Never run on the workspace.
- **Live, memo ON** (2026-10-30, Groq qwen3.8-27b; `live_c/determinism/table.md`): 6 queries ("gift
  for a vintage watch collector", "laptop for college", "black dress", "gluten free snacks", "noise
  cancelling headphones for travel", "ingredients to make pancakes"), `--runs 3 --subprocess`: PASS
  6/6. Runs 2 and 3 were identical to run 1 on every compared field, overlap@12 1.000, tau 1.0, 0
  provider calls, 3 memo hits each. Run 1 in a fresh child process (PYTHONHASHSEED=4242): 18 calls
  (3 per query), 30,613 tokens (26,542 prompt / 4,071 completion) + a 26-token preflight = 30,639;
  two TPM 429s were retried, nothing fell back; 453 s. The memo then held 18 entries (6 translate, 6
  rerank, 6 cross-sell), 12.9 KB. That memo folder (`live_c/determinism/memo/`) holds model
  replies: delete it or keep it private.
- **Offline.** `tests/test_determinism.py`, 18 tests on a fake LLM (two cold passes, a fresh process
  with another PYTHONHASHSEED, single flight, a memo restart); under PYTHONHASHSEED 0 and 12345 it
  passed 16/16 at the Phase C gate (not re-run on the 18). The watch query's snapshot is identical
  across seeds 0-7 (a set-iterating mutant gives 2 different ones).
- **What it does not cover.** No ad on page 1 and no variant merge on those flat pages (compared,
  never exercised); memo-off stability (`--memo off` was never run live); Databricks. Pinned does
  not mean good: the memo keeps the FIRST accepted sample, good or bad (e01's `vintage watch repair
  kit` intent is pinned; owner question), and a prompt edit starts a new sample (new prompt bytes =
  new memo keys). An empty cross-sell reply is never pinned, so the product page asks again on every
  open (and can flip "Similar items" -> "Pairs well with this"), and the check reports a FAIL with
  `calls_after_run1 [1, 1]` though the pipeline worked as designed (owner question).

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

## Complex query bank (2026-11-02): the Databricks stress-test set
- **What.** `data/complex_queries.json`: 107 shopper-style queries in eval_page_types' own format
  (typos, slang, budgets, negations, occasions, personas, problem phrasings, 2-5 signals each):
  groups `complex_fashion` 29, `complex_grocery` 27, `complex_electronics` 31, `complex_cross`
  13 (the only rows spanning 2-3 departments), `complex_collision` 7 (brand / noun collisions:
  "watch the baby", "galaxy print", "electrolyte tablets"...). 8 deliberate gap rows (2 per
  lane: fashion-27/28, grocery-29/30, electronics-01/30, cross-19/23) test graceful handling,
  judged by `match_weak` and their `_context`, not by shares. 41 rows carry `relevance`
  (served-page P@4 / P@12). Each `_context` opens with the row's id and states the pass bar.
- **Meta.** `data/complex_queries_meta.json`, same order: id, lane, signals, `expect`, `gap`,
  `core_target` ('0' on 12 accessory / owner-need rows where the device itself is WRONG, '>=0.8'
  on 10 device asks), `dept_min` (13 multi-department rows: >= 3 of the first 12 cards per named
  department on a flat page, >= 1 slot on a bundle), and `supply_rows` / `supply_regex`: the
  matching titles in data/products_enriched.csv, MEASURED 2026-11-02 with pandas (every non-gap
  row >= 20). The expectations are INFERENCE (written by agents, reviewed by an adversarial
  pass, not by merchandisers); nothing in the bank has been run against any LLM yet.
- **Run.** `databricks/10_page_eval.py`: `run(mode="on", queries="data/complex_queries.json")`
  (databricks/README.md, run order), then `eval/score_complex_queries.py` on its rows or CSV:
  per-group means, core split by `core_target` (eval_page_types' single mean mixes the two
  directions), the `dept_min` check from `types_page` (vertical_top12 scores each card alone, so
  a one-department page passes it), validity (degraded, partial, misrouted, memo hits must be 0).
  Locally: `python -m eval.eval_page_types --mode on --queries data/complex_queries.json`, then
  `python -m eval.score_complex_queries <csv>`. Read validity first, as everywhere.
- **Cost (ESTIMATES).** 2 LLM calls per query run: mode "on" 214 calls, ~0.36-0.55M tokens (per
  query 3.4K on Databricks Qwen3-Next with 2026-09 prompts to 5.1K on qwen3.8-27b, Oct 30), ~20
  min at ~10 s per query on Databricks; mode "both" 214-321 calls. Repeatability subset:
  `12_determinism.run(queries=[...], runs=3)` with a LIST of query strings (a str is one query).
- **Known traps.** grocery-18 ("egg replacer for baking cookies") is routed to `recipe` by the
  regex fast path (it accepts recipe or flat); cross-04 reads `requested_types` {accessory}, so
  its snack half may score as type mismatches; dash-cam rows count as core cameras on the camera
  asks; Kids rows score gender-neutral in eval_page_types, so wrong_gender sees adult rows only.
  `tests/test_score_complex_queries.py` pins the bank's loadability and the scorer.
- **Not a load test.** There is NO concurrency or load test in the repo (eval/ and databricks/
  hold none); every latency here is sequential, one request at a time.
