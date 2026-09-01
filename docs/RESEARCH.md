# CAS Research Notes — Agentic Commerce, Search Trends, Databricks Build

Web-researched and independently fact-checked 2026-07-29 (every load-bearing
claim verified against a fetched primary source or flagged otherwise).
Prepared for the LatentView CAS PoC. Three sections + the phase-2 fine-tuning
plan.

---

## 1. GPT Shopping / Agentic Commerce

### What actually happened (verified timeline)
- **2025-09-29 — OpenAI + Stripe "Instant Checkout"** in ChatGPT (US Etsy
  merchants first; Shopify brands after). The **Agentic Commerce Protocol
  (ACP)** released as an open standard; Stripe's Shared Payment Token is
  scoped per-merchant/basket, so ChatGPT never holds card credentials.
- **2025-11-25 — Perplexity + PayPal "Instant Buy"**: retailers stay merchant
  of record; PayPal handles identity/fraud/protection.
- **2026-01-08 — Microsoft Copilot Checkout + Brand Agents** (PayPal, Stripe,
  Shopify partners; Shopify merchants auto-enrolled, opt-out). By April 2026:
  500K+ merchants, UCP feeds GA in Merchant Center (Target first).
- **2026-01-11 — Google Universal Commerce Protocol (UCP)** with Shopify,
  Etsy, Wayfair, Target, Walmart + 20 endorsers (Visa, Mastercard, Amex,
  Stripe...). Agentic checkout in AI Mode + Gemini; retailer stays seller of
  record. Compatible with A2A, AP2, MCP.
- **2026-03 — the checkout retreat**: OpenAI deprioritized Instant Checkout
  ("moving to Apps" — Target/Instacart/DoorDash apps in ChatGPT). Walmart
  pulled out and shipped its own **Sparky** agent as a ChatGPT app: in-chat
  checkout converted at **~1/3** the rate of click-through to walmart.com
  ("a very temporary moment in time" — Walmart EVP, Morgan Stanley TMT).
- **Amazon**: Rufus became **"Alexa for Shopping"** (2026-05-13), default in
  the search bar for signed-in US customers; Amazon's own Q4-2025 disclosure
  attributes **~$12B incremental annualized sales** to it. Amazon
  simultaneously blocks OpenAI's crawlers *and* buys sponsored placements in
  ChatGPT that redirect to amazon.com.
- **ChatGPT ads are live**: pilot 2026-02-09, self-serve Ads Manager
  2026-05-05, product feeds for ads 2026-06-02; ~$60 US CPMs. Perplexity
  publicly rejects ads as trust-eroding — monetization models are diverging.
- **The legal fight over agent access**: Amazon v. Perplexity (Comet browser
  agent) — preliminary injunction 2026-03-10, stayed on appeal ~a week later.
  Countermeasure standards: Visa **Trusted Agent Protocol** (with Cloudflare,
  Web Bot Auth), Mastercard Agent Pay / AP4M.

### The equilibrium (four independent datapoints)
"**Discover in AI, buy on site**": Walmart's 1/3 conversion + pullout;
OpenAI's own retreat; Semrush survey (22% bought in-AI vs 50% bought
elsewhere after AI research); Prime Day 2026 — chatbot-referred visitors
were **+40% more likely to convert** than search/email/social.

### What it means for CAS
1. **Context-aware search is NOT commoditized — it relocates behind an
   agent-facing API.** Shopify's storefront MCP (search_catalog /
   get_product / update_cart...) is the concrete shape. Wrapping CAS's
   FastAPI backend as an **MCP server** (product_search, get_product,
   recommend_complements) is a thin wrapper and a credible next deliverable.
2. **Staying out of checkout/payments is the right scope** — checkout comes
   from PSPs/platforms; the protocol camps (ACP/UCP) are converging, so the
   choice is deferrable.
3. **CAS's sponsored design is validated**: ChatGPT and Google AI Mode both
   sell *labeled* product ads; CAS's never-blended, relevance-gated sponsored
   layer matches the pattern and answers Perplexity's trust critique.
4. **Feed quality is the new SEO**: OpenAI feed spec (JSONL, 15-min
   refreshes), Google conversational attributes, Microsoft UCP feeds. CAS's
   attribute schema (color/material/occasion) maps directly; LLM attribute
   enrichment is a companion service pitch.
5. **Latency matters for agent surfaces**: agents call catalog tools
   mid-conversation. The ~20-25s reasoning-model path is disqualifying there;
   the agent endpoint should use the fast path (non-reasoning model +
   page_cache, or retrieval-only + Bayesian blend).

---

## 2. Trends in Context-Aware Search / Retrieval (2025-2026)

### Verified state of the art
- **Hybrid BM25+dense with RRF is the settled baseline.** WANDS product-search
  benchmark: BM25 0.698 NDCG ≈ pure vectors 0.695; RRF fusion 0.707;
  RRF + product-name field boost **0.750 (~+7.4%)**. RRF k=60 is the
  production default.
- **Learned sparse** (Elastic ELSER > SPLADE; OpenSearch 3.3 "SEISMIC" sparse
  ANN: <15ms at 50M docs) is mainstream for out-of-domain robustness.
- **Rerankers**: Cohere Rerank 4 (Dec 2025; `rerank-v4-fast` explicitly for
  e-commerce), Voyage rerank-2.5 (instruction-following). Open-weight:
  bge-reranker-v2-m3 (Apache-2.0, CPU-viable). ⚠️ Jina rerankers v2/v3 are
  CC-BY-NC — same license trap as our research-only dataset.
- **Embeddings**: Qwen3-Embedding leads MTEB (multilingual 70.6); Matryoshka
  truncation to 256-dim loses <1% for the best models (cache shrink + faster
  CPU retrieval). Never mix MTEB v1 and v2 numbers in one table.
- **LLM query understanding in production** — Instacart "Intent Engine"
  (their tech blog, Nov 2025): rewrite coverage 50%→95%+ with 90%+ precision;
  only ~2% of (tail) queries hit live LLM inference — head queries are cached.
  This is the strongest external validation of CAS's translator architecture.
- **Agentic retrieval as a category**: Azure AI Search decompose→parallel
  subqueries→rerank→merge ("up to 40% better relevance" per Microsoft) — this
  is architecturally what CAS's multi-intent scatter-gather already does.
- **2026 Gartner MQ (Search & Product Discovery) Leaders**: Algolia,
  Bloomreach, Constructor, Coveo, Netcore Unbxd. Gartner's 2026 criteria
  explicitly include **"agentic integrations"**.

### What it means for CAS (priority order)
1. **Fusion beats routing** — our keyword-first router picks ONE engine per
   query; the verified numbers say parallel BM25+dense fused with RRF beats
   either alone by ~7% NDCG. Feeding an RRF-fused pool into the existing LLM
   reranker is the single highest-ROI change. (Keep the router as the
   "non-disruptive integration" story; offer fusion as the quality tier.)
2. **Cheap cross-encoder stage** (bge-reranker-v2-m3) between retrieval and
   the slow LLM rerank — trims the LLM pool and is the fast fallback on 429s.
3. **Refresh the embedding bake-off** with one 2025-generation small model
   (Qwen3-Embedding-0.6B, Apache-2.0) + try 256-dim Matryoshka truncation.
4. **Cite Instacart** for the architecture (cache head, LLM only on tail,
   small model for the translator) — it directly addresses our demo-latency
   risk.
5. **Session personalization is our biggest gap** vs 2026 expectations; the
   SIGIR '26 agentic-query-reformulation paper sits ON TOP of an existing
   stack — exactly our architecture. Roadmap item, published pattern.
6. **At 60K products, no vector DB needed** — in-memory cosine is correct
   2026 guidance. Scale answers if asked: pgvector to ~50M, OpenSearch
   SEISMIC for sparse-at-scale.

---

## 3. Build on Databricks — component mapping (verified against docs 2026-07-29)

The product is now branded **Databricks AI Search** (ex Mosaic AI Vector
Search). All mappings below verified against current Databricks docs.

| CAS component (local) | Databricks target | Status |
|---|---|---|
| `data/products*.csv` + loader | **Unity Catalog Delta table** (+ Change Data Feed); loader → Lakeflow Job/Declarative Pipeline over `meta_*.jsonl` in a UC Volume | GA |
| `app/embeddings.py` + `.cache/` + cosine loop | **AI Search delta-sync index** on a **standard endpoint** (~150ms, continuous sync; 60K vectors fits the smallest unit = 2M-vector capacity) | GA |
| Embedding model (gte-small) | `databricks-gte-large-en` pay-per-token (1024-dim; **re-run compare_embeddings — old numbers don't transfer**), or serve exact gte-small via provisioned throughput | GA |
| `app/keyword_search.py` (BM25) + hybrid router | `query_type=hybrid` (GA; RRF, `rrf_param=60`, max 200 results) on the same index — or Beta `FULL_TEXT` queries to keep keyword-first routing. Dedicated BM25 indexes need storage-optimized endpoints (Public Preview) — overkill at 60K. Scoring semantics change → re-run router evals | GA / Beta |
| LLM translate + rerank (Groq gpt-oss-120b) | `databricks-gpt-oss-120b` pay-per-token (~$0.15/$0.60 per M tok; same model → kills `key_rotator.py` + the 8000-TPM workaround). Latency will NOT beat Groq; option: keep Groq as an **External Model** behind **Unity AI Gateway** (failover Beta, guardrails Beta) | GA (gateway bits Beta) |
| — (new tier) | **DatabricksReranker** cross-encoder: ≤50 results, <1s, ~10% lift, SDK ≥0.57 — fast default, keep LLM rerank for the reasons/demo view | GA |
| `eval/` suite | **MLflow 3 GenAI**: `mlflow.genai.evaluate()` + custom code scorers (P@10/MRR/NDCG) offline; ⚠️ production monitoring runs **LLM judges only** — code scorers stay in a scheduled Lakeflow Job. Eval JSONs → UC tables. Per-stage tracing free | GA |
| `ui.py` (Streamlit) + FastAPI | **Databricks Apps** (GA; Streamlit first-class). Medium = 2 vCPU/6GB at 0.5 DBU/hr suffices once vector search replaces in-process encoding | GA |
| `GET /product` row lookup | **Lakebase synced table** / Online Feature Store (legacy Online Tables are dead after 2026-01-15) | GA |
| `sponsored.json` | Small Delta table; relevance gate + separation stay app-side | — |
| `safety.md` checklist | **Unity AI Gateway guardrails** (Beta): prompt-injection detection, PII redaction, payload logging to Delta, per-user rate limits — most of safety.md becomes platform-enforced | Beta |

**Cost mechanics for the PoC budget:** vector endpoint bills from 1 search
unit and only stops **24h after the last index is deleted** (idle ≠ free);
Apps bill 0.5 DBU/hr while running; LLM calls are pennies at 2/search.
**Disclosures:** Amazon Reviews 2023 stays research-only on any platform;
Beta items = FULL_TEXT queries, dedicated full-text indexes, AI Gateway
guardrails/failover. Watch pay-per-token model retirements (Llama 3.1 405B
from 2026-02-15, Gemini 2.5 on 2026-10-02); gpt-oss currently safe.

**What the platform does NOT solve:** reasoning-model latency, catalog
coverage gaps, and the "70 paneer sellers" diversity P2 — those remain
app-level work.

---

## 4. Fine-tuning plan (staged — decided 2026-07-29)

### Phase 1 — embedding fine-tune (local, running)
`scripts/finetune_embeddings.py`: contrastive fine-tune of gte-small with
**MultipleNegativesRankingLoss** on ~20K catalog-derived (anchor, positive)
pairs — (title ↔ attributes/description) and (query-style attribute phrase ↔
full text) — sampled evenly across verticals; in-batch negatives.
Output: `models/gte-small-cas-ft`. Measure honestly (retrieval-only):

```bash
python -m eval.run_eval --no-rerank --tag base_embed
EMBEDDING_MODEL=models/gte-small-cas-ft python -m eval.run_eval --no-rerank --tag ft_embed
```

Ship only if P@10/NDCG beat the base model. Next iterations: mine hard
negatives (same category, wrong color/material), add pairs from
`logs/feedback.jsonl` thumbs-up data as real relevance labels.

### Phase 2 — LLM fine-tune on Databricks (planned)
- **Target tasks**: translator (query → intents+constraints JSON) and
  reranker (candidates → scored JSON). Both have cheap supervision: log
  gpt-oss-120b's outputs in production (MLflow traces) → distill into a
  small model (e.g. Llama 3.x 8B / Qwen3 8B) with **LoRA/QLoRA** on
  Databricks Foundation Model Fine-tuning; judge with MLflow LLM judges
  against the big model's outputs.
- **Why distillation**: the 120B reasoning model is the latency problem
  (~20-25s). A LoRA-tuned 8B matching it on THESE two narrow JSON tasks cuts
  latency an order of magnitude and removes the Groq dependency — the
  Instacart pattern (small tuned model for query understanding) exactly.
- **Techniques**: instruction tuning on traced examples; DPO on
  thumbs-up/down feedback pairs once volume exists; constrained JSON decoding
  at serving time.
