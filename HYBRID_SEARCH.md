# Hybrid Search: eBay-style keyword stack + Swiggy-model CAS integration

This adds a **keyword (lexical) search engine** — the kind of stack a retailer
like eBay already runs — and wires CAS in front of it using the **Swiggy model**
(one search bar, keyword-first, context-aware fallback). It's additive: the
existing CAS pipeline is untouched; these pieces wrap it.

## What was added

| File | Role |
|---|---|
| `app/keyword_search.py` | eBay-Cassini-style lexical engine: BM25 retrieval over title + item-specifics + category, then a "Best Match" rank blending text relevance with business signals (rating quality + review-count popularity). Self-contained, no new deps. |
| `app/hybrid_router.py` | The Swiggy-model router: keyword-first, and on a lexical miss, fall back to the intent (CAS) layer. Intent layer is injected, so CAS stays swappable/testable. |
| `app/main.py` → `GET /unified_search` | Single entry point. Runs the keyword engine; on a miss, calls the **real CAS pipeline** (`/search`) as the fallback. Returns which path served the query. |
| `demo/hybrid_demo.py` | Runnable offline demo (synthetic catalog + lightweight intent stand-in) proving the routing without a catalog or LLM keys. |

## The model

```
query -> keyword engine (BM25 + Best Match)
      -> keyword HIT?  yes -> return keyword results        (the common case, fast)
                       no  -> CAS pipeline (expand -> vectors -> rerank) -> return
```

**Hit vs miss = term coverage of the top result**, not result count. A query is
a keyword hit when the best lexical match actually contains enough of what was
asked (`min_coverage`, default **0.75**). Coverage is measured against the top
result's **title + category tokens only** (`coverage_fields`) — NOT the product
description. Both choices are load-bearing, learned from the real catalog:
descriptions mention words like "breathable"/"day" in passing, and Amazon
titles are keyword-stuffed ("...Breathable...Mothers Day...Gift"), so a lenient
0.5 over full documents let intent queries ("something breathable for a humid
day") false-hit on irrelevant products. At 0.75 the top result may miss at most
a quarter of the asked terms; anything less falls through to CAS — which costs
latency but never quality. A precise query that matches only 1–2 products
("wool sweater") is still a hit. This is the non-disruptive integration from
the runbook: keyword search keeps everything it's good at; CAS only catches its
misses.

## Run the demo (no catalog / keys needed)

```
python -m demo.hybrid_demo
```

You'll see precise queries served by KEYWORD and intent queries falling back to
INTENT (with the expanded query shown). The demo's intent layer is an **offline
stand-in** (concept expansion + BM25) so it runs anywhere.

## Production wiring

`/unified_search` uses the **real CAS pipeline** as the fallback — the stand-in
is demo-only. With your catalog and LLM keys in place:

```
GET /unified_search?query=...&top_k=12&min_coverage=0.75&rerank=true&sponsored=true
-> { "path": "keyword"|"intent", "reason": "...", "keyword_coverage": 0.x,
     "results": [ ... ],
     // intent path also passes through the CAS context:
     "interpreted_as": [...], "errors": [...], "no_match": bool, "message": "..." }
```

The keyword engine is indexed at startup on the **same catalog** the CAS
pipeline loads (`get_dataframe()`), so both paths cover the same products.
Keyword result rows are JSON-safe (NaN → null; Starlette rejects NaN) and carry
`catalog_index`, so they are clickable into the `/product` detail page exactly
like CAS results. `rerank`/`sponsored` are forwarded to the CAS fallback so the
UI toggles behave identically in both modes.

## Knobs

- `min_coverage` (router / endpoint): keyword-hit threshold, default 0.75.
  Higher = more queries fall to CAS.
- `coverage_fields` (`KeywordSearchEngine`): fields the hit/miss coverage is
  measured against — default title + category, deliberately excluding the
  description (incidental words there inflate coverage).
- Best-Match weights (`KeywordSearchEngine`): `w_relevance` 0.7 / `w_popularity`
  0.2 / `w_quality` 0.1; `title_weight` 3 (title matches count more).
- BM25 `k1` 1.5 / `b` 0.75 (standard).

## Honest limitations

- **Miss detection is coverage-based (offline).** It's a solid proxy, but the
  runbook's real trigger uses **GA4/engagement signals** (zero-result / no-click
  queries). Swap that in when integrating with a real client's analytics.
- **The demo intent layer is a stand-in**, not CAS. It shows the *routing*, not
  CAS's real quality. Production uses the actual LLM+vector pipeline.
- **`/unified_search` is backend-only again as of 2026-07-31.** Sai/Niharika's
  2026-07-30 call: the inline "🔀 Hybrid mode" toggle was replaced with a
  SEPARATE page (Amazon's model — regular search bar vs its AI assistant, two
  distinct surfaces, not one page with a mode switch). The endpoint,
  `app/hybrid_router.py`, and `app/keyword_search.py` are untouched and still
  work exactly as documented above; nothing here was disrupted. They're just
  not wired into the UI's auto-routing toggle anymore. If you want to exercise
  the auto-router again, call `/unified_search` directly or re-add a toggle.
- **The new Classic Search page is NOT `/unified_search`.** It's a separate,
  simpler endpoint, `GET /keyword_search` (see `app/main.py`), that is pure
  BM25 with NO fallback to CAS, ever — the deliberate "normal search bar"
  alternative to `/search`, not an auto-router. `ui.py` (Context-Aware Search)
  and `pages/1_Classic_Search.py` (Classic Search) are two independent pages;
  `ui_common.py` holds the rendering/fetch code they share.
- **Streamlit gotcha found 2026-07-31**: an emoji in a `pages/*.py` FILENAME
  silently breaks the entire sidebar (not just page nav) on this Streamlit
  version/Windows setup — confirmed by removing the emoji and watching the
  sidebar reappear. Emoji in page *content* (`st.title`, captions, `st.Page`
  `icon=` if you migrate to `st.navigation()`) is fine; just never put one in
  a `pages/` filename.
- This mirrors, on a synthetic scale, what a client integration would do; the
  exact keyword engine would be *their* stack (Cassini/Elasticsearch), with CAS
  reranking or catching misses via the same router shape.
