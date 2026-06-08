"""LLM provider abstraction.

Single interface across Gemini, OpenAI, Anthropic. Eval scripts swap
providers via the LLM_PROVIDER env var without touching translator or
reranker code.

Lazy imports: a missing SDK (e.g. `openai` not installed) only errors when
that provider is actually selected, not at import time.

Determinism: when DETERMINISTIC=true, temperature is forced to 0 and seed
is passed where the provider supports it.
"""

import json
import logging
from typing import Optional

from app.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    DETERMINISTIC,
    GEMINI_MODEL,
    GOOGLE_API_KEYS,
    LLM_PROVIDER,
    LLM_SEED,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails after all retries / fallbacks."""


# ---------- Gemini ----------------------------------------------------------


class _GeminiBackend:
    def __init__(self):
        if not GOOGLE_API_KEYS:
            raise LLMError(
                "No Gemini API keys configured. Set GOOGLE_API_KEY in .env."
            )
        # Use the rotator even with one key — it gives uniform error handling.
        from app.key_rotator import GeminiKeyRotator

        self.rotator = GeminiKeyRotator(GOOGLE_API_KEYS)
        self.model = GEMINI_MODEL

    def generate_json(self, prompt: str, temperature: float) -> str:
        config = {
            "temperature": 0.0 if DETERMINISTIC else temperature,
            "response_mime_type": "application/json",
        }
        # Gemini SDK supports `seed` for determinism.
        if DETERMINISTIC:
            config["seed"] = LLM_SEED
        response = self.rotator.generate_content(
            model=self.model, contents=prompt, config=config
        )
        return response.text


# ---------- OpenAI ----------------------------------------------------------


class _OpenAIBackend:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY not set.")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError(
                "openai package not installed. `pip install openai`."
            ) from e
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL

    def generate_json(self, prompt: str, temperature: float) -> str:
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0 if DETERMINISTIC else temperature,
            "response_format": {"type": "json_object"},
        }
        if DETERMINISTIC:
            params["seed"] = LLM_SEED
        resp = self.client.chat.completions.create(**params)
        return resp.choices[0].message.content


# ---------- Anthropic -------------------------------------------------------


class _AnthropicBackend:
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY not set.")
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMError(
                "anthropic package not installed. `pip install anthropic`."
            ) from e
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = ANTHROPIC_MODEL

    def generate_json(self, prompt: str, temperature: float) -> str:
        # Anthropic does NOT have a strict JSON mode, so we instruct in the
        # prompt and let the caller's existing JSON parser handle stripping.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0 if DETERMINISTIC else temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt + "\n\nRespond with ONLY valid JSON, no prose.",
                }
            ],
        )
        return resp.content[0].text


# ---------- Factory ---------------------------------------------------------


_singleton: Optional[object] = None
_singleton_provider: Optional[str] = None


def get_llm_client():
    """Return the configured backend (singleton per provider).

    Lazy-initialized so the first call discovers config issues; later calls
    just return the cached instance.
    """
    global _singleton, _singleton_provider
    if _singleton is not None and _singleton_provider == LLM_PROVIDER:
        return _singleton

    if LLM_PROVIDER == "gemini":
        _singleton = _GeminiBackend()
    elif LLM_PROVIDER == "openai":
        _singleton = _OpenAIBackend()
    elif LLM_PROVIDER == "anthropic":
        _singleton = _AnthropicBackend()
    else:
        raise LLMError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. "
            f"Expected one of: gemini, openai, anthropic."
        )

    _singleton_provider = LLM_PROVIDER
    logger.info("LLM provider initialized: %s", LLM_PROVIDER)
    return _singleton


def generate_json(prompt: str, temperature: float = 0.2) -> dict:
    """Convenience wrapper that parses the response as JSON.

    Strips common code-fence wrappers (\\`\\`\\`json ... \\`\\`\\`) some models add.
    Raises LLMError if response can't be parsed.
    """
    client = get_llm_client()
    raw = client.generate_json(prompt, temperature)
    text = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM returned invalid JSON: {raw!r}") from e
