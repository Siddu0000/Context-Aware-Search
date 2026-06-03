# Context-Aware Agentic Search

> Production-style PoC for **intent-aware product retrieval**. Implements a
> two-stage retrieval pipeline (HyDE-style query expansion + LLM cross-encoder
> rerank) with caching, latency instrumentation, multi-model embedding
> support, an IR evaluation harness, and a feedback loop.

The system answers a single question well:

> Given a free-form shopping query, which products from the catalog best
> match the user's *intent* — not just the literal tokens they typed?

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
   User query →  │  1. Query Translator (HyDE)                  │  LLM: Gemini Flash
                 │     Generates N hypothetical product intents │
                 └──────────────────────────────────────────────┘
                                       │
                                       ▼
                 ┌──────────────────────────────────────────────┐
                 │  2. Embedding Layer                          │  sentence-transformers
                 │     Encodes each intent into a vector        │  (or OpenAI, pluggable)
                 └──────────────────────────────────────────────┘
                                       │
                                       ▼
                 ┌──────────────────────────────────────────────┐
                 │  3. Vector Retrieval (scatter-gather)        │  cosine similarity
                 │     Top-K per intent, merged, deduplicated   │  (FAISS-ready interface)
                 └──────────────────────────────────────────────┘
                                       │
                                       ▼
                 ┌──────────────────────────────────────────────┐
                 │  4. LLM Reranker (cross-encoder pattern)     │  LLM: Gemini Flash
                 │     Re-scores against ORIGINAL query         │
                 │     Produces score + one-sentence reason     │
                 └──────────────────────────────────────────────┘
                                       │
                                       ▼
                            Top-K results + reasons
```

Every stage is independently togglable. The reranker can be turned off via
the `rerank` query parameter for A/B comparison.

---

## Tech Stack and Why

| Layer | Choice | Why |
|---|---|---|
| API framework | **FastAPI** | Pydantic validation, async-ready, auto-generated OpenAPI docs at `/docs`, lifespan events for cold-start mitigation |
| LLM (translator + reranker) | **Google Gemini Flash** (`google-genai`) | Free tier sufficient for PoC volume, native JSON-mode output (`response_mime_type=application/json`), sub-second latency, swappable via `GEMINI_MODEL` env var |
| Default embedder | **sentence-transformers/all-MiniLM-L6-v2** | 384-d, CPU-friendly, well-benchmarked on retail-style retrieval, no external API dependency |
| Alternative embedder | **OpenAI text-embedding-3-small** | 1536-d, higher MTEB scores; used for benchmarking, not as default (adds API latency and cost) |
| Vector search | **scikit-learn cosine_similarity** | Deterministic, zero infra, sufficient up to ~1M rows; abstraction is FAISS-ready when scaling demands it |
| Embedding persistence | **NumPy `.npy` + content-hashed cache key** | Boots in ~2 s after first run; cache auto-invalidates when CSV, model, or field set changes |
| UI | **Streamlit** | Fast iteration for data-product demos; thumbs-up/down feedback wired to backend |
| Config | **python-dotenv + `app/config.py`** | Single source of truth; no scattered `os.getenv()` calls |
| Feedback log | **append-only JSONL** | Concurrency-safe enough for a PoC, parses with one line of pandas, survives crashes |

---

## Key Architectural Decisions

### Why HyDE (Hypothetical Document Embeddings)?

Direct query embeddings often miss relevant products because user queries and
product descriptions use different vocabulary ("comfy work-from-home setup"
vs. "ergonomic chair", "LED lamp", "noise-cancelling headphones"). HyDE
(Gao et al., 2022) sidesteps this by generating *hypothetical answer
documents* and embedding those instead — moving the query into the same
linguistic register as the catalog.

We extend single-document HyDE to a **multi-intent variant** (`NUM_INTENTS=3`)
because a single bad generation can dominate retrieval. With three
hypotheticals, one drifted intent contributes at most ~33 % of the candidate
pool — leaving room for the reranker to filter it out.

### Why an LLM rerank step on top of retrieval?

Embedding similarity is fast but coarse. It cannot enforce constraints like
price ranges, occasion fit, or compositional intent ("X but not Y"). It also
suffers from the well-known **HyDE drift failure mode**: if one hypothetical
intent shifts off the user's real meaning, the products retrieved by that
intent flow straight into the top-K with no second filter.

The reranker is the only stage that sees the user's *original* query (not
the expanded intents), so it acts as a coherence check. It produces a
`rerank_score` in [0,100] and a one-line `reason` per result. The reason
field is what powers the UI's "💡 _Matches your request for…_" explanations
and gives clients an explainability story.

### Why scatter-gather instead of single-query retrieval?

Scatter-gather (one retrieval per intent, then merge + dedup) gives
*redundancy*. If intent #1 nails the query and intents #2 and #3 drift,
intent #1's top results still dominate the merged pool after dedup-by-title.
The reranker can then surface the right ones.

### Why a `.npy` cache instead of FAISS?

For 60K rows × 384 dims, scikit-learn's `cosine_similarity` runs in ~5 ms
per query and uses ~24 MB of RAM. FAISS would be premature here. The
embedding layer is abstracted behind `search_vectors()` so swapping in
FAISS later is a 10-line change in one file.

### Why an append-only JSONL feedback log?

Thumbs-up/down events are low-frequency and high-value. A SQLite or Postgres
table adds infra without solving any current problem. JSONL is concurrent-
write safe (POSIX append is atomic for <PIPE_BUF), trivially replayable for
offline analysis, and converts to a DataFrame with `pd.read_json(lines=True)`.

---

## Setup

```bash
# 1. Virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt
# Windows torch issue? Install CPU-only build explicitly:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Credentials
cp .env.example .env              # then edit .env and add GOOGLE_API_KEY

# 4. Place the catalog
# Drop your products.csv at data/products.csv
```

---

## Running the service

Two terminals (venv active in both):

```bash
# Backend
uvicorn app.main:app --reload --port 8000

# UI
streamlit run ui.py
```

Open <http://localhost:8501>.

The first backend boot encodes the catalog (~2-5 min on CPU for 60K rows).
Every subsequent boot loads from `.cache/` in < 2 s. The cache key includes
the CSV hash, embedding model name, and field set — change any of those and
it regenerates automatically.

Interactive API docs are at <http://localhost:8000/docs>.

---

## Evaluation Harness

The eval harness addresses the five evaluation dimensions any serious
retrieval system needs to defend:

### 1. Retrieval quality — Precision@K, MRR, NDCG, Recall

```bash
# Pipeline with rerank on (default)
python -m eval.run_eval

# Pipeline without rerank (baseline)
python -m eval.run_eval --no-rerank
```

Output goes to `eval_results/<tag>_<timestamp>.csv`. The summary line at the
end of the run prints mean P@1, P@5, P@10, R@10, MRR, NDCG@10 and total
latency. Run with and without rerank back-to-back to quantify the rerank
contribution.

Relevance per query is defined in `data/eval_queries.json` using simple
textual criteria (title patterns, color matches, category filters). Extend
this file as you add real labels.

### 2. Embedding-model comparison

```bash
python -m eval.compare_embeddings
# or specify
python -m eval.compare_embeddings --models all-MiniLM-L6-v2 all-mpnet-base-v2 text-embedding-3-small
```

Reranker is disabled so the differences are purely from the embedding
backend. Each model gets its own cached `.npy`, so re-runs are fast.

### 3. "Chunking" / search-text construction

In a product catalog there's no long document to split, but the analogous
question is *which fields to concatenate into each product's search text*.

```bash
python -m eval.compare_search_text
```

Compares four variants:
- `title_only` — minimal
- `title_plus_desc` — title + description
- `title_color` — title + color (for color-heavy queries)
- `all_fields` — every text field concatenated (current default)

### 4. Latency and cost

Every `/search` response carries `latency_ms` per stage (`translate`,
`retrieve`, `rerank`, `total`). The eval CSV captures the same fields per
query plus a token-count proxy (`approx_tokens`, chars/4) for rough cost
estimation. Multiply by published per-token Gemini pricing for $ figures.

### 5. User feedback loop

The Streamlit UI shows 👍 / 👎 buttons per result. Clicks POST to
`/feedback` which appends to `logs/feedback.jsonl`. Each entry contains the
query, product title, rank, and the rerank reason that was displayed.

Replay later with:

```python
import pandas as pd
fb = pd.read_json("logs/feedback.jsonl", lines=True)
# Online relevance at rank k:
fb.assign(relevant=(fb.rating == 1)).groupby("rank")["relevant"].mean()
```

---

## Project Structure

```
.
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── app/
│   ├── config.py            # Centralized env config
│   ├── embeddings.py        # Multi-backend embedder (ST + OpenAI)
│   ├── search.py            # Cached retrieval, NaN-safe, configurable fields
│   ├── translator.py        # Multi-intent HyDE with fallback
│   ├── reranker.py          # LLM rerank with reasoning
│   ├── feedback.py          # JSONL append logger
│   ├── metrics.py           # StageTimings + token proxy
│   └── main.py              # FastAPI app (lifespan, validation, /search, /feedback)
├── data/
│   ├── products.csv         # Your catalog
│   └── eval_queries.json    # Labeled eval set
├── eval/
│   ├── metrics_ir.py        # P@K, MRR, NDCG, Recall (pure)
│   ├── run_eval.py          # End-to-end eval harness
│   ├── compare_embeddings.py
│   └── compare_search_text.py
├── ui.py                    # Streamlit UI
└── logs/                    # JSONL feedback (gitignored)
```

---

## Roadmap

Items intentionally out of scope for this version, kept for the next sprint:

- **Grocery category** (recipe-aware basket composition, nutrition-constrained
  retrieval). Blocked on a grocery catalog with nutrition columns.
- **Upsell / cross-sell** layer. Belongs as a post-rerank step.
- **Multi-attribute weighted scoring** (rating × review-count × semantic).
  Blocked on having rating data.
- **FAISS migration** when catalog crosses ~1M rows.
- **LLM-as-judge eval mode** for harder labels than textual relevance.
- **Listing-completeness scoring** for the seller-fulfillment story.

---

## References

- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*
  (HyDE), 2022 — <https://arxiv.org/abs/2212.10496>
- Reimers & Gurevych, *Sentence-BERT*, 2019 — basis for sentence-transformers
- Nogueira & Cho, *Passage Re-ranking with BERT*, 2019 — classical reranking
  motivation


