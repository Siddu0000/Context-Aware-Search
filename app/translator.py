"""Query translator implementing a multi-intent HyDE pattern.

HyDE (Hypothetical Document Embeddings, Gao et al. 2022) generates a
hypothetical answer document for a query, then embeds the answer instead of
the query. We extend this to N=3 hypothetical product search intents so a
single bad generation doesn't dominate retrieval.

The LLM call is always wrapped in try/except with a fallback to the raw
query. The API must stay responsive even when the LLM is degraded.
"""

import json
import logging
from typing import List

from google import genai

from app.config import GEMINI_MODEL, GOOGLE_API_KEY, NUM_INTENTS

logger = logging.getLogger(__name__)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY missing. Copy .env.example to .env and fill in the key."
    )

client = genai.Client(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = f"""
You are an intelligent retail search translator implementing the HyDE
(Hypothetical Document Embeddings) pattern.

Task: convert the user's free-form query into EXACTLY {NUM_INTENTS} concise
product search intents. Each intent should read like a short product listing,
not advice. These intents will be embedded and used to retrieve real products
from a catalog.

Rules:
- Infer category from the query (do NOT restrict to any fixed list of categories)
- Use product-focused language
- Include brand/type/spec/color where appropriate
- Do NOT assume gender unless explicitly mentioned
- Do NOT over-generalize; stay close to the user's intent
- Do NOT explain anything
- Output ONLY valid JSON

JSON format:
{{
  "search_terms": ["...", "...", "..."]
}}
"""


def translate_query(user_query: str) -> List[str]:
    """Generate N hypothetical product intents from a free-form query.

    Falls back to `[user_query]` on ANY failure (LLM down, malformed JSON,
    empty list). The API never 500s because the translator hiccupped.
    """
    prompt = SYSTEM_PROMPT + f'\nUser query: "{user_query}"'
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.2,
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
        terms = parsed.get("search_terms", [])
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"Bad shape: {parsed!r}")

        seen, clean = set(), []
        for t in terms:
            t = str(t).strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                clean.append(t)
        return clean or [user_query]

    except Exception as e:
        logger.warning("Translator failed (%s). Using raw query.", repr(e))
        return [user_query]
