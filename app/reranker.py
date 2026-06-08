"""LLM-based reranker with reasoning + cache.

Sees the user's ORIGINAL query and reorders the candidate pool, attaching
a one-sentence justification to each result.

Cache: keyed by (provider, query, candidate-titles, top_k). Identical
inputs return identical outputs within the session — supports Niharika's
determinism requirement.

Fallback: on any LLM failure (quota, parse, all keys cooling), returns
the embedding-score order with a sentinel reason. The demo stays usable.
"""

import logging
from typing import List

from app.cache import make_key, reranker_cache
from app.config import LLM_PROVIDER
from app.llm_client import LLMError, generate_json

logger = logging.getLogger(__name__)


RERANK_PROMPT_TEMPLATE = """\
You are a retail search quality expert.

The user searched for: "{query}"

Below are {n_candidates} candidate products from the catalog. Rank them by
how well each matches the user's ORIGINAL intent. Use these factors:
- Semantic relevance to the query
- Price appropriateness if a budget is implied
- Occasion / use-case fit (each product has an `occasion` attribute)
- Material fit if mentioned (each product has a `material` attribute)
- Specificity (color, size, style if specified)

Return JSON only. Score is integer 0-100. Include only the TOP {top_k}.
Order the array by score descending.

Candidates (use the `idx` field):
{candidates_block}

JSON format:
{{
  "ranked": [
    {{"idx": <int>, "score": <int>, "reason": "<one short sentence>"}}
  ]
}}
"""


def _format_candidate(i: int, product: dict) -> str:
    title = product.get("Product_title", "")
    price = product.get("price")
    color = product.get("color", "")
    material = product.get("material", "")
    occasion = product.get("occasion", "")
    desc = (product.get("prod_description") or "")[:100]
    price_str = f"${price}" if price is not None else "?"
    return (
        f"  idx={i} | {title} | color={color} | material={material} "
        f"| occasion={occasion} | {price_str} | {desc}"
    )


def rerank(query: str, candidates: List[dict], top_k: int) -> List[dict]:
    if not candidates:
        return []

    # Cap candidates sent to the LLM for cost + latency.
    candidates = candidates[: max(top_k * 3, 30)]
    titles = tuple(c.get("Product_title", "") for c in candidates)

    key = make_key("rerank", LLM_PROVIDER, query, titles, top_k)
    cached = reranker_cache.get(key)
    if cached is not None:
        logger.info("Reranker cache hit q=%r", query)
        return [dict(p) for p in cached]

    candidates_block = "\n".join(
        _format_candidate(i, p) for i, p in enumerate(candidates)
    )
    prompt = RERANK_PROMPT_TEMPLATE.format(
        query=query,
        n_candidates=len(candidates),
        top_k=top_k,
        candidates_block=candidates_block,
    )

    try:
        parsed = generate_json(prompt, temperature=0.1)
        ranked = parsed.get("ranked", [])
        if not isinstance(ranked, list) or not ranked:
            raise LLMError(f"Bad shape: {parsed!r}")

        results = []
        for entry in ranked[:top_k]:
            idx = int(entry.get("idx", -1))
            if 0 <= idx < len(candidates):
                product = dict(candidates[idx])
                product["rerank_score"] = int(entry.get("score", 0))
                product["reason"] = str(entry.get("reason", "")).strip()
                results.append(product)
        if not results:
            raise LLMError("Reranker returned no valid indices.")
        reranker_cache.set(key, results)
        return results

    except Exception as e:
        logger.warning("Rerank failed (%s). Embedding-order fallback.", repr(e))
        fallback = []
        for product in candidates[:top_k]:
            product = dict(product)
            product["rerank_score"] = None
            product["reason"] = "(rerank unavailable — embedding score only)"
            fallback.append(product)
        return fallback
