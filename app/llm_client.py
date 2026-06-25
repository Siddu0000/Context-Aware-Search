"""LLM provider abstraction.

Single interface across Gemini, OpenAI, Anthropic. Eval scripts swap
providers (LLM_PROVIDER) and temperature (TEMPERATURE_OVERRIDE) at runtime
without touching translator or reranker code.

Why this module reads config dynamically (import app.config as cfg) instead
of `from app.config import X`: a plain `from ... import X` binds a *copy* of
the value at import time, so an eval script that sets `config.X = ...` later
would have no effect here. Referencing cfg.X reads the live module attribute,
so runtime overrides (provider switch, temperature sweep) work as expected.

Lazy imports: a missing SDK (e.g. `openai` not installed) only errors when
that provider is actually selected, not at import time.

Determinism: when cfg.DETERMINISTIC is true, temperature resolves to 0 and a
fixed seed is sent where the provider supports it. See cfg.effective_temperature.
"""

import json
import logging
import re
from typing import Optional

import app.config as cfg

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails after all retries / fallbacks."""


def error_code(exc: Exception) -> str:
    """Extract a short, user-facing code from an exception (e.g. '503').
    Falls back to the HTTP-like status in the message, then the class name."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code:
        return str(code)
    m = re.search(r"\b([4-5]\d{2})\b", str(exc))
    return m.group(1) if m else exc.__class__.__name__


def _use_seed() -> bool:
    """Send a fixed seed only in true deterministic mode (not during a sweep)."""
    return cfg.DETERMINISTIC and cfg.TEMPERATURE_OVERRIDE is None


class _GeminiBackend:
    def __init__(self):
        if not cfg.GOOGLE_API_KEYS:
            raise LLMError(
                "No Gemini API keys configured. Set GOOGLE_API_KEY in .env."
            )
        from app.key_rotator import GeminiKeyRotator

        self.rotator = GeminiKeyRotator(cfg.GOOGLE_API_KEYS)
        self.model = cfg.GEMINI_MODEL

    def generate_json(self, prompt: str, temperature: float) -> str:
        config = {
            "temperature": cfg.effective_temperature(temperature),
            "response_mime_type": "application/json",
        }
        if _use_seed():
            config["seed"] = cfg.LLM_SEED
        response = self.rotator.generate_content(
            model=self.model, contents=prompt, config=config
        )
        return response.text


class _OpenAIBackend:
    def __init__(self):
        if not cfg.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY not set.")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError(
                "openai package not installed. `pip install openai`."
            ) from e
        self.client = OpenAI(
            api_key=cfg.OPENAI_API_KEY,
            **({"base_url": cfg.OPENAI_BASE_URL} if cfg.OPENAI_BASE_URL else {}),
        )
        self.model = cfg.OPENAI_MODEL

    def generate_json(self, prompt: str, temperature: float) -> str:
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg.effective_temperature(temperature),
            "response_format": {"type": "json_object"},
        }
        if _use_seed():
            params["seed"] = cfg.LLM_SEED
        base = (cfg.OPENAI_BASE_URL or "").lower()
        if "groq" in base and cfg.GROQ_REASONING_FORMAT:
            params["extra_body"] = {"reasoning_format": cfg.GROQ_REASONING_FORMAT}
        resp = self.client.chat.completions.create(**params)
        return resp.choices[0].message.content


class _AnthropicBackend:
    def __init__(self):
        if not cfg.ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY not set.")
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMError(
                "anthropic package not installed. `pip install anthropic`."
            ) from e
        self.client = Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        self.model = cfg.ANTHROPIC_MODEL

    def generate_json(self, prompt: str, temperature: float) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=cfg.effective_temperature(temperature),
            messages=[
                {
                    "role": "user",
                    "content": prompt + "\n\nRespond with ONLY valid JSON, no prose.",
                }
            ],
        )
        return resp.content[0].text


_singleton: Optional[object] = None
_singleton_provider: Optional[str] = None


def get_llm_client():
    """Return the configured backend (singleton per provider).

    Reads cfg.LLM_PROVIDER dynamically so eval scripts can switch providers
    at runtime (reset the singleton by setting _singleton = None).
    """
    global _singleton, _singleton_provider
    provider = cfg.LLM_PROVIDER
    if _singleton is not None and _singleton_provider == provider:
        return _singleton

    if provider == "gemini":
        _singleton = _GeminiBackend()
    elif provider == "openai":
        _singleton = _OpenAIBackend()
    elif provider == "anthropic":
        _singleton = _AnthropicBackend()
    else:
        raise LLMError(
            f"Unknown LLM_PROVIDER={provider!r}. "
            f"Expected one of: gemini, openai, anthropic."
        )

    _singleton_provider = provider
    logger.info("LLM provider initialized: %s", provider)
    return _singleton


def generate_json(prompt: str, temperature: float = 0.2) -> dict:
    """Convenience wrapper that parses the response as JSON.

    Strips common code-fence wrappers some models add. Raises LLMError if
    the response can't be parsed.
    """
    client = get_llm_client()
    raw = client.generate_json(prompt, temperature)
    text = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM returned invalid JSON: {raw!r}") from e
