# Safety & content rules (load if the search faces real end users)

This is a retail product search. If/when it serves real user queries, apply
these guardrails. Most are not yet implemented — treat as a checklist when the
demo moves toward production.

## Input handling
- Treat ALL query text as untrusted input. Never interpolate it into shell
  commands, SQL, or eval(). The pipeline currently embeds it as text only —
  keep it that way. (The stress test verifies injection-like strings stay inert.)
- Don't log full user queries with PII to shared logs. `logs/feedback.jsonl`
  is local-only and gitignored. errors[].detail and log lines carry flag names,
  counts and exception TYPES only, never query or reply text
  (`query_guard.describe` / `query_guard.error_detail`, all four LLM call sites);
  the search exception logs and the feedback log line no longer carry the query,
  and a deployed App runs uvicorn with --no-access-log (URLs carry the query).
- The local API log FILE (2026-10-15, `--log-config uvicorn_log.json` -> `logs/api.log`,
  gitignored, rotating) takes app.*, uvicorn.error and root only. `uvicorn.access`
  goes to the console ONLY (propagate false): its request lines carry the query in
  the URL. Verified after a live session: 0 of the queries used appear in the file.
  Keep any new handler off `uvicorn.access`; the console scrollback still holds the
  URLs, so never paste it anywhere shared.
- 2026-10-27 code logs the same way: never query text, LLM replies, titles or memo
  keys. The LLM memo (below) logs the operation and the exception TYPE, once per type
  (`llm_memo._warn_once`: a message can carry an entry's path, which is its key); only
  the start-up check's FAILED line adds the first 200 characters of the message (the
  store folder or its probe file at most: the check reads no entry by key).
  `python -m app.llm_memo` prints counts only and, on a store error, exits 1 with the
  exception type only; memo counts reach clients only as numbers (`X-LLM-Memo-Hits`,
  `/healthz` `memo`: this process's hits / misses / commits / discards / errors /
  expired / bypassed / trips). The device-ask lines say why an ask was dropped, never
  the family: it comes from the query text. The start-up "Effective config" line
  carries settings and digests (`prompts_sha`, `catalog_fp`, the memo's backend kind
  and folder), no query.

## Output / ranking
- Don't surface adult, weapon, or otherwise restricted product categories to
  general queries. Amazon Reviews 2023 may contain such items; add a category
  blocklist before any public-facing demo.
- Featured/paid-ad prioritization (P3) must be clearly separated from organic
  relevance in the code, so the ranking stays auditable. Every ad card carries a
  visible "Sponsored" tag; under a price/rating sort an ad keeps the label but is
  not forced to the front (2026-10-15). VARIANT_COLLAPSE never merges an ad with
  organic rows (2026-10-19): an ad is always its own labelled card, never a swatch
  inside an organic card or the other way round, under every sort.
- Under a device ask (2026-10-27, DEVICE_ROLE_RANKING, flat pages, inside
  TYPE_AWARE_RANKING) an ad for an ACCESSORY of the asked device is refused: it is not
  boosted, labelled or mirrored into `sponsored`, and stays an organic row where the
  organic order puts it (a winder ad must not stand above every watch on a watch
  gift). It applies only where `_effective_device` acted (the knob, a device ask not
  for an accessory, >= 50% of the raw RERANK_INPUT_K head about the family, >= 1 core
  row in the pool) and is one more condition in `_boost_sponsored` (`_is_addon`), next
  to the rerank-score floor and the type gate: the ranking stays auditable -- every ad
  moved to the front still carries "Sponsored", and a refused ad neither moves nor
  carries the label. Test: `test_device_ask_refuses_an_accessory_ad`
  (tests/test_ranking.py; knob off, the add-on ad leads again in bid order). NOT seen
  live: no sponsored card on any page 1 of the 2026-10-30 live run
  (`data/sponsored.json` is stale), none in the determinism check, 0 sponsored rows in
  any arm of the 0-LLM cross-vertical guard. Bundles keep the slot-membership gate
  (CLAUDE.md Architecture 7). REVERT = `DEVICE_ROLE_RANKING=false`.
- No claim the data cannot back (2026-10-19 marketplace-style UI): badges come only
  from real fields ("Highly rated" = 4.5+ from 50+ ratings); no stock, delivery,
  seller, bestseller, coupon or membership lines, and checkout stays disabled
  ("Demo store: no checkout or payment."). `test_no_badge_or_claim_the_data_cannot_back`
  (tests/test_ui.py) fails on prime / best seller / choice / in stock / deliver /
  coupon / bought in / deal, or either marketplace's name ("amazon" outside the dataset
  citation, "ebay" anywhere; 2026-10-26),
  in the card, product-detail, badge, details-table and cart-line HTML helpers and the
  default AI Search page; it does NOT render the product page's buy box or the cart
  page (Subtotal panel, captions). The LOOK is allowed -- the Amazon-style layout
  (2026-10-19) and, since 2026-10-26, an eBay-style palette from eBay's MIT-licensed
  design tokens -- never either marketplace's name, logo (eBay's four-colour wordmark
  included), marks or proprietary font (e.g. eBay's Market Sans); the footer says the
  demo is not affiliated with or endorsed by any retailer.
  2026-10-27 rail: the Dietary values are enrichment tags (regex matches in the
  listing's own title, specs and description), not certifications, so the section's
  hint says "As the listing words it, not a certification" (Condition: "As the listing
  words it"); the hint is a tooltip, not a visible caption (owner question). OPEN,
  owner: the Electronics "Works with" section offers "Amazon" (1,763 Electronics rows
  carry works_with=amazon), against the rule above -- rename it (e.g. "Kindle, Echo &
  Fire") or record an exception; either way the no-claim test should scan every
  registry label and hint, not only the fixture's apple / samsung / pc.
- Model-written text shown in the UI is screened server-side (links, chat tokens,
  instruction markers) and escaped with `ui_common.escape_md` before any markdown
  sink; intents render with st.text. A NEW markdown sink for model or shopper text
  MUST use escape_md.
- CATALOG (seller) text is untrusted in every UI sink too (2026-10-15 shop UI):
  `st.html` cards escape every value with `html.escape(quote=True)` (`ui_common.esc`,
  `card_html`); markdown sinks use `escape_md` (cart titles go through st.html, so a
  "$...$" title no longer renders as LaTeX); image URLs must be https with no quotes,
  whitespace or control characters (`safe_img` rejects javascript:, http:, data:).
  Chat bubbles (shopper and assistant text) and error messages (the request URL in
  them carries the query) are escaped as well. The only `unsafe_allow_javascript`
  sink is the scroll script: constant text plus a widget key reduced to
  [A-Za-z0-9_-]. A NEW sink for model, shopper OR catalog text MUST escape;
  `tests/test_ui.py` keeps the title `Plan ($200-$249.99) [deal](https://evil.example)
  *X* #1 <img src=x>` inert in grid, product page, recommendations and cart.
- 2026-10-19 sinks (marketplace UI), all escaped in the code: title-as-link
  buttons and their tooltips, rail link labels, Brand and Colour checkbox labels and
  the "Also available in" variant buttons use `escape_md`; the breadcrumb, Product
  details cells, cart brand, weak-match banner (it quotes the query) and masthead use
  `esc`. The EVIL-title test (`test_seller_text_is_inert_in_grid_detail_mini_and_cart`)
  covered the title links, Product details cells, breadcrumb and cart, but ran with no
  facets and no `variants` until 2026-10-27; it now also feeds an EVIL rail (Brand,
  Colour and Style labels, an attribute section's heading, hint and value label), EVIL
  variant titles and colours, and an EVIL `brand` (swatch CSS stays in
  `test_colour_swatches_come_from_the_fixed_map`). `variants[]` entries carry catalog
  text (title, color, img_url): escape them and pass images through `safe_img`. Swatch
  CSS takes colours ONLY from the fixed `COLOR_HEX` map with digest-key selectors,
  never from catalog or API text. The refused-filter warning is fixed text. The scroll
  script stays constant text (the 2026-10-19 wheel dispatch and hold interpolate
  nothing).
- 2026-10-27 sinks (Related items, department rail), all escaped in the code (the EVIL
  test above feeds the attribute heading, hint and value label and `brand`): the
  "Related items" button and its help text are constant text (its key is built from
  `catalog_index`); an attribute section's heading and hint go
  through `_rail_heading` (`esc`, the hint inside a `title=` attribute), its value
  labels through `escape_md` (checkbox labels are markdown), its checkbox keys are
  digests, never label text, and the filter sends the KEY, never a label; the
  `max_picked` caption is constant text plus an int; the Dietary note
  (`MATCH_ALL_NOTE`) and "No related products to suggest for this item." are constants
  (`st.html` + `esc` for the note). The rail's labels are the taxonomy's fixed
  vocabulary, never seller text, and are escaped anyway: a sink never trusts its
  source. `brand` (the listing's maker, else the store: seller text) goes through `esc`
  in the card byline, the product page's "Brand:" line and the cart line.

## LLM usage
- The translator and reranker prompts must not be overridable by user input
  (no prompt injection via the query). Keep the user query as DATA inside the
  prompt, never as instructions. That sentence alone is NOT enough: Qwen3-Next
  obeyed an injected payload 3/3 (databricks/08, 2026-10-12). IMPLEMENTED as
  `app/query_guard.py` (QUERY_GUARD_ENABLED, default on): the query is cut at its
  first instruction marker before any LLM sees it; a query that opens with one
  reaches no LLM; everything an LLM writes is screened before it is searched or
  shown; the chat assistant strips chat tokens and our MESSAGE fence (no override
  cut there -- "ignore my earlier instructions, I want blue" is legitimate chat).
  "Not overridable" now holds in this sense: an override cannot produce anything
  the shopper could not type. Residual + accepted false positives: CLAUDE.md.
- Indirect injection is still OPEN: catalog titles/descriptions (seller text) go
  into the rerank and cross-sell prompts unscreened. Revisit before onboarding a
  marketplace catalog.
- If a query asks the system to ignore instructions or reveal the prompt,
  the LLM should still just return product search intents.

## Stored LLM replies: the LLM memo (2026-10-27, meeting item "deterministic results")
- What: `app/llm_memo.py` (LLM_MEMO, default true in the app; every eval harness and
  databricks/08 turn it off) keeps the RAW reply each stage ACCEPTED -- translate,
  rerank, cross_sell, assistant -- and replays it for an identical request. An entry
  holds that reply, its stage, the model, `query_sha` and its creation time (in SQLite
  also its last use). The VALUES echo shopper text: intents, rerank reasons, the
  assistant's reply and rewritten query (its request is the conversation as posted).
  Treat a memo store like `logs/feedback.jsonl`: shopper data, never copied anywhere
  shared. A hit goes through today's parser and query guard, so a replayed reply passes
  the same output screen as a fresh one.
- Where: locally `LLM_MEMO_DIR` = `.cache/llm_memo` (SQLite `memo.sqlite`, WAL;
  `.cache/` is gitignored), bounded by LLM_MEMO_MAX_ENTRIES (50000, least recently used
  dropped). On the App: `/Volumes/dev/cas/app_state/llm_memo`, one JSON file per entry,
  named by its key, bounded only by `python -m app.llm_memo purge --to-limit`. In a
  notebook (Cell 2 sets no LLM_MEMO) app calls made there use it, SQLite in the
  /Workspace Git folder's `.cache/llm_memo`, until `vs_shim.patch_eval_harness`
  (memo=False by default) or an eval harness turns it off.
  LLM_MEMO_MAX_AGE_DAYS defaults to 0 = entries never expire. check_determinism keeps
  its memo in `--memo-dir` (a new temp folder by default; a /Volumes path only with
  `--allow-volume`), databricks/12 under
  `/Volumes/<CAS_CATALOG>/<CAS_SCHEMA>/evals/determinism/<user>/<UTC>/llm_memo`; both
  refuse the app's own LLM_MEMO_DIR and the App's memo volume in any spelling; a
  determinism run's memo folder holds model replies: delete it or keep it private.
- Key hashing: the key is a PLAIN sha256 of the request (MEMO_VERSION, provider, model,
  base URL, the messages -- the query is in them, case- and space-folded by
  `page_cache.canonical_query` for translate, rerank and cross_sell, as posted for the
  assistant --, JSON mode, temperature, seed, reasoning format and any other parameter
  sent); `query_sha` (what `purge --query` matches) is a plain sha256 of the case- and
  space-folded query. Neither is keyed by a secret, so anyone who can list the store and
  knows the prompts can CONFIRM a guessed query. Never log, print or paste a key (Input
  handling above).
- Exposure on the App: `dev` grants `account users` broad rights (CLAUDE.md, 2026-10-13
  grants). If they hold WRITE VOLUME on `dev.cas.app_state`, any account user can plant
  an entry the App replays to shoppers: the guard re-screens it, but product words and
  assistant prose pass. `09 app_grant()` prints a WARN (`_warn_memo_exposure`) when
  `account users` can read or write the volume.
- OWNER QUESTION (open): HMAC the key and `query_sha` with a deployment secret (a
  planted file is then a miss, a guessed query unconfirmable)? A schema where only the
  App's service principal holds WRITE VOLUME? Keep the assistant stage out of a shared
  volume? A max age on the App? Today the App bundle ships `LLM_MEMO: "true"`
  (databricks/app_bundle/app.yaml); "false" there creates and grants nothing. Purge:
  `python -m app.llm_memo purge` (`--query`, `--stage`, `--older-than-days`,
  `--expired`, `--to-limit`, `--all`); a running API or App keeps purged replies until
  it restarts.

## Data / licensing
- Amazon Reviews 2023 is research-licensed. Do NOT ship this dataset in a
  commercial client deployment. Production would use the client's own catalog.
  The footer's licence strip (`ui_common.footer`) is the only on-screen notice: no
  page may `st.stop()` before it.
- Open Food Facts (grocery) is ODbL — commercially usable, attribution required.
