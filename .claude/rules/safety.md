# Safety & content rules (load if the search faces real end users)

This is a retail product search. If/when it serves real user queries, apply
these guardrails. Most are not yet implemented — treat as a checklist when the
demo moves toward production.

## Input handling
- Treat ALL query text as untrusted input. Never interpolate it into shell
  commands, SQL, or eval(). The pipeline currently embeds it as text only —
  keep it that way. (The stress test verifies injection-like strings stay inert.)
- Don't log full user queries with PII to shared logs. `logs/feedback.jsonl`
  is local-only and gitignored.

## Output / ranking
- Don't surface adult, weapon, or otherwise restricted product categories to
  general queries. Amazon Reviews 2023 may contain such items; add a category
  blocklist before any public-facing demo.
- Featured/paid-ad prioritization (P3) must be clearly separated from organic
  relevance in the code, so the ranking stays auditable.

## LLM usage
- The translator and reranker prompts must not be overridable by user input
  (no prompt injection via the query). Keep the user query as DATA inside the
  prompt, never as instructions.
- If a query asks the system to ignore instructions or reveal the prompt,
  the LLM should still just return product search intents.

## Data / licensing
- Amazon Reviews 2023 is research-licensed. Do NOT ship this dataset in a
  commercial client deployment. Production would use the client's own catalog.
- Open Food Facts (grocery) is ODbL — commercially usable, attribution required.
