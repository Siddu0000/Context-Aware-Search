"""LLM provider abstraction — one interface over Gemini, OpenAI and Anthropic."""

import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

# Read as cfg.X, never `from app.config import X` — evals override cfg at runtime
import app.config as cfg

logger = logging.getLogger(__name__)

# Stack of active usage scopes. A tuple (not a list) so each scope set/reset is
# isolated per context; the dicts inside are MUTATED in place, which is what lets
# a scope opened in async middleware see calls made in the threadpool endpoint.
_USAGE_STACK: ContextVar[tuple] = ContextVar("llm_usage_stack", default=())


def _new_usage() -> dict:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }


@contextmanager
def track_usage():
    """Accumulate REAL provider-reported token usage for every LLM call in scope.

    Nested scopes are additive: a call counts toward every enclosing scope.
    Calls made outside any scope are simply not counted.
    """
    usage = _new_usage()
    token = _USAGE_STACK.set(_USAGE_STACK.get() + (usage,))
    try:
        yield usage
    finally:
        _USAGE_STACK.reset(token)


def _record_usage(prompt=None, completion=None, total=None, reasoning=None) -> None:
    """Add one call's usage to every open scope. Never raises."""
    stack = _USAGE_STACK.get()
    if not stack:
        return
    try:
        p, c = int(prompt or 0), int(completion or 0)
        t = int(total) if total else p + c
        r = int(reasoning or 0)
    except (TypeError, ValueError):
        p = c = t = r = 0
    for u in stack:
        u["calls"] += 1
        u["prompt_tokens"] += p
        u["completion_tokens"] += c
        u["total_tokens"] += t
        u["reasoning_tokens"] += r


def _usage_openai(resp) -> None:
    # Reasoning tokens are billed as completion tokens but never shown in the reply
    u = getattr(resp, "usage", None)
    if u is None:
        _record_usage()
        return
    details = getattr(u, "completion_tokens_details", None)
    _record_usage(
        getattr(u, "prompt_tokens", None),
        getattr(u, "completion_tokens", None),
        getattr(u, "total_tokens", None),
        getattr(details, "reasoning_tokens", None) if details is not None else None,
    )


def _usage_gemini(response) -> None:
    m = getattr(response, "usage_metadata", None)
    if m is None:
        _record_usage()
        return
    _record_usage(
        getattr(m, "prompt_token_count", None),
        getattr(m, "candidates_token_count", None),
        getattr(m, "total_token_count", None),
        getattr(m, "thoughts_token_count", None),
    )


def _usage_anthropic(resp) -> None:
    u = getattr(resp, "usage", None)
    if u is None:
        _record_usage()
        return
    _record_usage(getattr(u, "input_tokens", None), getattr(u, "output_tokens", None))


class LLMError(Exception):
    """Raised when an LLM call fails after all retries / fallbacks."""


def error_code(exc: Exception) -> str:
    """Short code for an exception: explicit code, else HTTP status, else class name."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code:
        return str(code)
    m = re.search(r"\b([4-5]\d{2})\b", str(exc))
    return m.group(1) if m else exc.__class__.__name__


def _message_text(content) -> str:
    """Flatten an assistant message to text.

    Most providers return a plain string. Databricks FM APIs return a LIST of
    typed blocks for reasoning models; the chain-of-thought arrives as
    "reasoning" blocks which must be dropped, not parsed as JSON.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                btype, text = block.get("type"), block.get("text")
            else:
                btype = getattr(block, "type", None)
                text = getattr(block, "text", None)
            if btype == "reasoning":
                continue
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


# Models that 400 on response_format=json_object. Learned on first refusal so
# later calls skip the doomed attempt instead of paying a round trip each time.
_NO_JSON_MODE: set = set()

# Only JSON-mode-less models get this appended: without the API guarantee they
# need the format restated (the Anthropic backend already does the same).
_JSON_ONLY_SUFFIX = "\n\nRespond with ONLY the JSON object, no prose or markdown."


def _rejects_json_mode(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "json_object" in msg and ("not supported" in msg or "unsupported" in msg)


def _parse_json_reply(raw: str):
    """Parse a reply that should be one JSON object, tolerating fences and prose."""
    text = raw.strip()
    if text.startswith("```"):
        # drop the opening fence line (```json, ``` or any language tag)
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Models without a JSON mode sometimes wrap the object in a sentence
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("no JSON object in reply", text, 0)


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
        _usage_gemini(response)
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

    def _params(self, prompt: str, temperature: float, json_mode: bool) -> dict:
        params = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt if json_mode else prompt + _JSON_ONLY_SUFFIX,
            }],
            "temperature": cfg.effective_temperature(temperature),
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        base = (cfg.OPENAI_BASE_URL or "").lower()
        # Databricks FM APIs reject unknown fields: `seed` 400s the whole call
        if _use_seed() and "serving-endpoints" not in base:
            params["seed"] = cfg.LLM_SEED
        if "groq" in base and cfg.GROQ_REASONING_FORMAT:
            params["extra_body"] = {"reasoning_format": cfg.GROQ_REASONING_FORMAT}
        return params

    def generate_json(self, prompt: str, temperature: float) -> str:
        json_mode = self.model not in _NO_JSON_MODE
        try:
            resp = self.client.chat.completions.create(
                **self._params(prompt, temperature, json_mode)
            )
        except Exception as e:
            if not (json_mode and _rejects_json_mode(e)):
                raise
            _NO_JSON_MODE.add(self.model)
            logger.info("%s rejects JSON mode; prompting for JSON instead.", self.model)
            resp = self.client.chat.completions.create(
                **self._params(prompt, temperature, json_mode=False)
            )
        _usage_openai(resp)
        choice = resp.choices[0]
        # A truncated reply is unparseable JSON; say so instead of "invalid JSON"
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMError(f"{self.model} reply truncated (hit max_tokens).")
        return _message_text(choice.message.content)


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
        _usage_anthropic(resp)
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
    if not isinstance(raw, str):
        raw = _message_text(raw)
    if not raw.strip():
        raise LLMError("LLM returned no text content (reasoning-only reply?)")
    try:
        return _parse_json_reply(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM returned invalid JSON: {raw!r}") from e
