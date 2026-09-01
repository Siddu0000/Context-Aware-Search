"""LLM provider abstraction — one interface over Gemini, OpenAI and Anthropic."""

import json
import logging
import re
from typing import Optional

# Read as cfg.X, never `from app.config import X` — evals override cfg at runtime
import app.config as cfg

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails after all retries / fallbacks."""


def error_code(exc: Exception) -> str:
    """Short code for an exception: explicit code, else HTTP status, else class name."""
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
    """Return the configured backend; rebuilt when cfg.LLM_PROVIDER changes."""
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
    """Call the backend and parse the reply as JSON, tolerating code-fence wrappers."""
    client = get_llm_client()
    raw = client.generate_json(prompt, temperature)
    text = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM returned invalid JSON: {raw!r}") from e
