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

## Output / ranking
- Don't surface adult, weapon, or otherwise restricted product categories to
  general queries. Amazon Reviews 2023 may contain such items; add a category
  blocklist before any public-facing demo.
- Featured/paid-ad prioritization (P3) must be clearly separated from organic
  relevance in the code, so the ranking stays auditable.
- Model-written text shown in the UI is screened server-side (links, chat tokens,
  instruction markers) and escaped with `ui_common.escape_md` before any markdown
  sink; intents render with st.text. A NEW markdown sink for model or shopper text
  MUST use escape_md.

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

## Data / licensing
- Amazon Reviews 2023 is research-licensed. Do NOT ship this dataset in a
  commercial client deployment. Production would use the client's own catalog.
- Open Food Facts (grocery) is ODbL — commercially usable, attribution required.
