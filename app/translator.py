"""Query translator with three pluggable strategies.

Strategies (selected via TRANSLATOR_MODE env var):

  query_expansion  - N short product-search phrases. Each is embedded
                     independently, then scatter-gather retrieval.
                     Recommended for short queries and retail catalogs
                     where titles are short.

  hyde             - One longer hypothetical product listing. Classical
                     HyDE (Gao et al. 2022). Single embedding, simpler
                     retrieval. Better when queries are descriptive /
                     conversational.

  hybrid           - 1 hypothetical document + (N-1) short phrases.
                     Diversity + depth at extra LLM token cost.

All three return a list of strings that downstream search treats uniformly.

Determinism: when DETERMINISTIC=true, temperature is 0 and outputs are
cached, so the same input gives bit-identical results within a session.
"""

import logging
from typing import List

from app.cache import make_key, translator_cache
from app.config import LLM_PROVIDER, NUM_INTENTS, TRANSLATOR_MODE
from app.llm_client import LLMError, generate_json

logger = logging.getLogger(__name__)


QUERY_EXPANSION_PROMPT = f"""\
You are a retail search translator using the multi-intent query expansion pattern.

Convert the user's query into EXACTLY {NUM_INTENTS} concise product search intents.
Each intent reads like a short product listing, not advice.

Rules:
- Infer category from the query; do NOT restrict to any fixed list
- Use product-focused language
- Include type/spec/color where appropriate
- Do NOT assume gender unless stated
- Output ONLY valid JSON

JSON format:
{{"search_terms": ["...", "...", "..."]}}
"""


HYDE_PROMPT = """\
You are a retail search assistant implementing the HyDE
(Hypothetical Document Embeddings, Gao et al. 2022) pattern.

For the user's query, write ONE hypothetical product listing — 2 to 4
sentences — that would be an ideal match. Include type, key attributes
(material, color, occasion), and a short description. This hypothetical
listing will be embedded and used to retrieve real catalog products.

Do NOT recommend a specific brand or invent a price.

Output ONLY valid JSON.
JSON format:
{"hypothetical_listing": "..."}
"""


HYBRID_PROMPT = f"""\
You are a retail search translator combining classical HyDE with multi-intent expansion.

Produce:
1. ONE hypothetical product listing (2-4 sentences) describing an ideal match.
2. {NUM_INTENTS - 1} short search phrases that would also retrieve relevant products.

Output ONLY valid JSON.
JSON format:
{{
  "hypothetical_listing": "...",
  "search_terms": [{", ".join(['"..."'] * (NUM_INTENTS - 1))}]
}}
"""


def _query_expansion(user_query: str) -> List[str]:
    prompt = QUERY_EXPANSION_PROMPT + f'\nUser query: "{user_query}"'
    parsed = generate_json(prompt, temperature=0.2)
    terms = parsed.get("search_terms", [])
    if not isinstance(terms, list) or not terms:
        raise LLMError(f"Bad query_expansion shape: {parsed!r}")
    return _dedupe([str(t).strip() for t in terms])


def _hyde(user_query: str) -> List[str]:
    prompt = HYDE_PROMPT + f'\nUser query: "{user_query}"'
    parsed = generate_json(prompt, temperature=0.2)
    listing = parsed.get("hypothetical_listing", "")
    if not isinstance(listing, str) or not listing.strip():
        raise LLMError(f"Bad hyde shape: {parsed!r}")
    return [listing.strip()]


def _hybrid(user_query: str) -> List[str]:
    prompt = HYBRID_PROMPT + f'\nUser query: "{user_query}"'
    parsed = generate_json(prompt, temperature=0.2)
    listing = (parsed.get("hypothetical_listing") or "").strip()
    terms = parsed.get("search_terms", []) or []
    if not listing and not terms:
        raise LLMError(f"Bad hybrid shape: {parsed!r}")
    intents: List[str] = []
    if listing:
        intents.append(listing)
    intents.extend(str(t).strip() for t in terms if str(t).strip())
    return _dedupe(intents)


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for item in items:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


_STRATEGIES = {
    "query_expansion": _query_expansion,
    "hyde": _hyde,
    "hybrid": _hybrid,
}


def translate_query(
    user_query: str,
    mode: str | None = None,
    use_cache: bool = True,
) -> List[str]:
    """Return a list of intents/documents to embed for retrieval.

    use_cache=False forces a fresh LLM call (won't read from cache, won't
    write to it either). Useful for testing actual response variance.

    On ANY failure (LLM down, bad JSON, all keys exhausted), falls back to
    [user_query] so the search API stays responsive. Fallback results are
    NOT cached — the next call retries the real LLM.
    """
    mode = (mode or TRANSLATOR_MODE).lower()
    if mode not in _STRATEGIES:
        logger.warning("Unknown TRANSLATOR_MODE=%r; using query_expansion.", mode)
        mode = "query_expansion"

    # Cache key includes mode + provider so different configs don't collide.
    key = make_key("translate", LLM_PROVIDER, mode, user_query)
    if use_cache:
        cached = translator_cache.get(key)
        if cached is not None:
            logger.info("Translator cache hit [mode=%s] q=%r", mode, user_query)
            return list(cached)
    else:
        logger.info("Translator cache BYPASSED [mode=%s] q=%r", mode, user_query)

    try:
        result = _STRATEGIES[mode](user_query) or [user_query]
        if use_cache:
            translator_cache.set(key, result)
        return result
    except Exception as e:
        logger.warning(
            "Translator failed [mode=%s] (%s). Using raw query.", mode, repr(e)
        )
        return [user_query]
