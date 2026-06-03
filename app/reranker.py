"""LLM-based reranker.

Solves the well-known HyDE failure mode where one drifted hypothetical
document pulls in irrelevant products that the embedding-score sort cannot
filter out. The reranker is the only stage that sees the user's *original*
query, so it acts as a coherence check against the entire candidate pool.

It also produces a one-line `reason` per result — used in the UI and exposed
in the API response.

Falls back to embedding-score order on any failure: the demo never breaks
because the rerank failed.
"""

import json
import logging
from typing import List

from google import genai

from app.config import GEMINI_MODEL, GOOGLE_API_KEY

logger = logging.getLogger(__name__)

client = genai.Client(api_key=GOOGLE_API_KEY)


RERANK_PROMPT_TEMPLATE = """\
You are a retail search quality expert.

The user searched for: "{query}"

Below are {n_candidates} candidate products retrieved from the catalog.
Rank them by how well each one matches the user's ORIGINAL intent.

When scoring, consider:
- Semantic relevance to the query
- Price appropriateness if a budget is implied
- Occasion / use-case fit (each product has an explicit `occasion` attribute)
- Material fit if mentioned in the query (each product has a `material` attribute)
- Specificity (color, size, style if specified)

Return JSON only. Score is integer 0-100. Include only the TOP {top_k}.
Order the array by score descending.

Candidates (use the 'idx' value as the index):
{candidates_block}

JSON format:
{{
  "ranked": [
    {{"idx": <int>, "score": <int>, "reason": "<one short sentence>"}},
    ...
  ]
}}
"""


def _format_candidate(i: int, product: dict) -> str:
    """Compact one-line representation of a candidate for the prompt."""
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
    """Return up to top_k candidates re-scored and annotated with reasons.

    On any LLM/JSON failure, returns the candidates' first top_k items
    unchanged (i.e. embedding-score order) so the API stays useful.
    """
    if not candidates:
        return []

    # Cap the number of candidates sent to the LLM for cost/latency control.
    candidates = candidates[: max(top_k * 3, 30)]

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
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )
        text = (
            response.text.strip()
            .removeprefix("```json")
            .removesuffix("```")
            .strip()
        )
        parsed = json.loads(text)
        ranked = parsed.get("ranked", [])
        if not isinstance(ranked, list) or not ranked:
            raise ValueError(f"Bad shape: {parsed!r}")

        results = []
        for entry in ranked[:top_k]:
            idx = int(entry.get("idx", -1))
            if 0 <= idx < len(candidates):
                product = dict(candidates[idx])  # copy
                product["rerank_score"] = int(entry.get("score", 0))
                product["reason"] = str(entry.get("reason", "")).strip()
                results.append(product)
        if not results:
            raise ValueError("Reranker returned no valid indices.")
        return results

    except Exception as e:
        logger.warning(
            "Rerank failed (%s). Falling back to embedding-score order.", repr(e)
        )
        fallback = []
        for product in candidates[:top_k]:
            product = dict(product)
            product["rerank_score"] = None
            product["reason"] = "(rerank unavailable — embedding score only)"
            fallback.append(product)
        return fallback
