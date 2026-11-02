"""LLM provider abstraction — one interface over Gemini, OpenAI and Anthropic."""

import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

# Read as cfg.X, never `from app.config import X` — evals override cfg at runtime
import app.config as cfg
from app import llm_memo

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
        # Replies served from the LLM memo: no provider call, no tokens (app/llm_memo.py)
        "memo_hits": 0,
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


def _record_memo_hit() -> None:
    """A memo hit made no provider call: it counts here, never in `calls`."""
    for u in _USAGE_STACK.get():
        u["memo_hits"] += 1


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


# Models that 400 on any `temperature` (claude-sonnet-5 on Databricks). They run
# at the provider default, so their runs are not reproducible call to call.
_NO_TEMPERATURE: set = set()


def _rejects_json_mode(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "json_object" in msg and ("not supported" in msg or "unsupported" in msg)


def _rejects_temperature(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "temperature" in msg and ("not support" in msg or "unsupported" in msg)


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


def _openai_endpoint_takes_seed() -> bool:
    # Databricks FM APIs reject unknown fields: `seed` 400s the whole call
    return "serving-endpoints" not in (cfg.OPENAI_BASE_URL or "").lower()


def seed_sent() -> bool:
    """Whether LLM calls made now carry a fixed seed, i.e. can be reproduced.

    DETERMINISTIC=true alone does not mean seeded: Databricks and Anthropic get none.
    Mirrors the backends' own gates, so an eval can log what was actually sent.
    """
    if not _use_seed():
        return False
    if cfg.LLM_PROVIDER == "gemini":
        return True
    if cfg.LLM_PROVIDER == "openai":
        return _openai_endpoint_takes_seed()
    return False


def param_downgrades() -> dict:
    """Models that silently run WITHOUT JSON mode or temperature (learned on refusal).

    Neither is an error, so fell_back never shows them; evals must report them.
    """
    return {
        "json_mode_dropped": sorted(_NO_JSON_MODE),
        "temperature_dropped": sorted(_NO_TEMPERATURE),
    }


class _GeminiBackend:
    base_url = None  # identity for get_llm_client(); Gemini has no configurable endpoint

    def __init__(self):
        if not cfg.GOOGLE_API_KEYS:
            raise LLMError(
                "No Gemini API keys configured. Set GOOGLE_API_KEY in .env."
            )
        from app.key_rotator import GeminiKeyRotator

        self.rotator = GeminiKeyRotator(cfg.GOOGLE_API_KEYS)
        self.model = cfg.GEMINI_MODEL

    def _config(self, temperature: float) -> dict:
        config = {
            "temperature": cfg.effective_temperature(temperature),
            "response_mime_type": "application/json",
        }
        if _use_seed():
            config["seed"] = cfg.LLM_SEED
        return config

    def memo_request(self, prompt: str, temperature: float) -> dict:
        """What generate_json sends, as app/llm_memo.make_key reads it (no network)."""
        config = self._config(temperature)
        return {"model": self.model, "messages": prompt, "json_mode": True,
                "temperature": config["temperature"], "seed": config.get("seed"),
                "reasoning_format": None}

    def generate_json(self, prompt: str, temperature: float) -> str:
        response = self.rotator.generate_content(
            model=self.model, contents=prompt, config=self._config(temperature)
        )
        _usage_gemini(response)
        return response.text


def _databricks_token_provider():
    """api_key callable for OPENAI_AUTH=databricks; openai calls it before every request.

    databricks-sdk caches the token and refreshes it before it expires, so a call
    costs microseconds. In a Databricks App the token is the app's service principal.
    """
    import openai

    # Older openai formats a callable into "Bearer <function ...>": every call would 401
    if tuple(int(p) for p in re.findall(r"\d+", openai.__version__)[:2]) < (1, 106):
        raise LLMError(f"OPENAI_AUTH=databricks needs openai>=1.106 (found {openai.__version__}).")
    from app.vector_search import _workspace_client  # one SDK client, one token cache

    sdk = _workspace_client().config
    host = (sdk.host or "").rstrip("/").lower()
    if not host or not (cfg.OPENAI_BASE_URL or "").lower().startswith(host + "/"):
        # cfg's DEFAULT base URL is Groq: never hand a workspace token to another host
        raise LLMError("OPENAI_AUTH=databricks needs OPENAI_BASE_URL on the workspace host "
                       f"({host or 'unknown'}), e.g. {host or 'https://<host>'}/serving-endpoints.")

    def token() -> str:
        return sdk.authenticate()["Authorization"].split(" ", 1)[1]

    return token


class _OpenAIBackend:
    def __init__(self):
        if cfg.OPENAI_AUTH not in {"key", "databricks"}:
            raise LLMError(f"Unknown OPENAI_AUTH={cfg.OPENAI_AUTH!r}; expected key or databricks.")
        if cfg.OPENAI_AUTH == "key" and not cfg.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY not set (a Databricks App uses OPENAI_AUTH=databricks).")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError(
                "openai package not installed. `pip install openai`."
            ) from e
        self.auth = cfg.OPENAI_AUTH
        api_key = (_databricks_token_provider() if self.auth == "databricks"
                   else cfg.OPENAI_API_KEY)
        self.client = OpenAI(
            api_key=api_key,
            **({"base_url": cfg.OPENAI_BASE_URL} if cfg.OPENAI_BASE_URL else {}),
        )
        self.model = cfg.OPENAI_MODEL
        self.base_url = cfg.OPENAI_BASE_URL

    def _params(self, prompt: str, temperature: float, json_mode: bool,
                send_temperature: bool = True) -> dict:
        params = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt if json_mode else prompt + _JSON_ONLY_SUFFIX,
            }],
        }
        if send_temperature:
            params["temperature"] = cfg.effective_temperature(temperature)
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        base = (cfg.OPENAI_BASE_URL or "").lower()
        if _use_seed() and _openai_endpoint_takes_seed():
            params["seed"] = cfg.LLM_SEED
        if "groq" in base and cfg.GROQ_REASONING_FORMAT:
            params["extra_body"] = {"reasoning_format": cfg.GROQ_REASONING_FORMAT}
        return params

    def memo_request(self, prompt: str, temperature: float) -> dict:
        """What generate_json sends NOW, as app/llm_memo.make_key reads it (no network).

        Read after a call it describes that call's successful attempt: a refusal adds the
        model to _NO_JSON_MODE / _NO_TEMPERATURE for good, so the sets only grow."""
        params = self._params(prompt, temperature, self.model not in _NO_JSON_MODE,
                              self.model not in _NO_TEMPERATURE)
        return {
            "model": params["model"],
            "messages": params["messages"],
            "json_mode": "response_format" in params,
            "temperature": params.get("temperature"),
            "seed": params.get("seed"),
            "reasoning_format": (params.get("extra_body") or {}).get("reasoning_format"),
        }

    def generate_json(self, prompt: str, temperature: float) -> str:
        # Each refusal permanently drops one param, so this ends within 3 attempts
        while True:
            json_mode = self.model not in _NO_JSON_MODE
            send_temperature = self.model not in _NO_TEMPERATURE
            try:
                resp = self.client.chat.completions.create(
                    **self._params(prompt, temperature, json_mode, send_temperature)
                )
                break
            except Exception as e:
                if json_mode and _rejects_json_mode(e):
                    _NO_JSON_MODE.add(self.model)
                    logger.info("%s rejects JSON mode; prompting for JSON instead.", self.model)
                elif send_temperature and _rejects_temperature(e):
                    _NO_TEMPERATURE.add(self.model)
                    logger.warning("%s rejects temperature; using the provider default "
                                   "(runs are not reproducible).", self.model)
                else:
                    raise
        _usage_openai(resp)
        choice = resp.choices[0]
        # A truncated reply is unparseable JSON; say so instead of "invalid JSON"
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMError(f"{self.model} reply truncated (hit max_tokens).")
        return _message_text(choice.message.content)


class _AnthropicBackend:
    base_url = None  # identity for get_llm_client(); direct Anthropic has one endpoint

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

    def _request(self, prompt: str, temperature: float) -> dict:
        return {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": cfg.effective_temperature(temperature),
            "messages": [
                {
                    "role": "user",
                    "content": prompt + "\n\nRespond with ONLY valid JSON, no prose.",
                }
            ],
        }

    def memo_request(self, prompt: str, temperature: float) -> dict:
        """What generate_json sends, as app/llm_memo.make_key reads it (no network)."""
        req = self._request(prompt, temperature)
        return {"model": req["model"], "messages": req["messages"], "json_mode": False,
                "temperature": req["temperature"], "seed": None, "reasoning_format": None,
                "extra": {"max_tokens": req["max_tokens"]}}

    def generate_json(self, prompt: str, temperature: float) -> str:
        resp = self.client.messages.create(**self._request(prompt, temperature))
        _usage_anthropic(resp)
        return resp.content[0].text


# Set both to None to force a rebuild (eval/compare_llms.py does). The model and
# base URL a backend was built for live on the backend itself, not here.
_singleton: Optional[object] = None
_singleton_provider: Optional[str] = None


def _wanted() -> tuple:
    """(provider, model, base_url, auth) that cfg selects right now."""
    provider = cfg.LLM_PROVIDER
    if provider == "openai":
        return provider, cfg.OPENAI_MODEL, cfg.OPENAI_BASE_URL, cfg.OPENAI_AUTH
    model = {"gemini": cfg.GEMINI_MODEL, "anthropic": cfg.ANTHROPIC_MODEL}.get(provider)
    return provider, model, None, None


def _is_current(backend) -> bool:
    # Keyed on model and endpoint too: a notebook that changes cfg.OPENAI_MODEL
    # must not keep calling the old model while evals log the new name
    built_for = (_singleton_provider, getattr(backend, "model", None),
                 getattr(backend, "base_url", None), getattr(backend, "auth", None))
    return backend is not None and built_for == _wanted()


def configured_model() -> Optional[str]:
    """The model cfg selects now; get_llm_client() rebuilds to match it."""
    return _wanted()[1]


def active_model() -> Optional[str]:
    """Model of the backend already built for the CURRENT cfg, else None. Never builds."""
    return getattr(_singleton, "model", None) if _is_current(_singleton) else None


def get_llm_client():
    """Return the configured backend; rebuilt when provider, model or base URL changes."""
    global _singleton, _singleton_provider
    provider = cfg.LLM_PROVIDER
    if _is_current(_singleton):
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
    logger.info("LLM provider initialized: %s (model=%s)", provider, _singleton.model)
    return _singleton


# ------------------------------------------------------------------ the LLM reply memo
# app/llm_memo.py keeps replies a stage ACCEPTED. A stage declares its calls with
# memo_stage(...) and reports its verdict with accept_reply / reject_reply, so only an
# accepted reply is ever stored and a refused replay is forgotten. ContextVars, like the
# usage stack above: each request (a uvicorn threadpool context) has its own.
_MEMO_STAGE: ContextVar[Optional[str]] = ContextVar("llm_memo_stage", default=None)
# memo_stage(..., key_text=...): what the memo keys the calls inside on, instead of their
# prompt (a ContextVar, not a generate_json argument at the call sites, so the tests'
# stand-ins for generate_json keep their (prompt, temperature) signature)
_MEMO_KEY_TEXT: ContextVar[Optional[str]] = ContextVar("llm_memo_key_text", default=None)
# (stage, key, raw or None for a memo hit, model) of the last memo-keyed call in context
_PENDING: ContextVar[Optional[tuple]] = ContextVar("llm_memo_pending", default=None)
# Open memo_outcome() scopes: dicts the stages' verdicts are written into (a tuple, like
# the usage stack, so a scope is isolated per context and seen by nested calls)
_OUTCOMES: ContextVar[tuple] = ContextVar("llm_memo_outcomes", default=())


@contextmanager
def memo_stage(stage: str, key_text: Optional[str] = None):
    """Mark the generate_json calls inside as `stage`'s (llm_memo.STAGES).

    Only such calls use the memo: a reply is safe to replay only once a stage accepted
    it, and a call outside any stage (the start-up probe, an ad-hoc notebook call) has
    no accept point. `key_text`: see generate_json (the stages pass their prompt
    rendered with the canonical query)."""
    token = _MEMO_STAGE.set(stage)
    key_token = _MEMO_KEY_TEXT.set(key_text)
    try:
        yield
    finally:
        _MEMO_KEY_TEXT.reset(key_token)
        _MEMO_STAGE.reset(token)


@contextmanager
def memo_outcome():
    """What the LLM stages inside said about their replies, for main.py's caches.

    `superseded`: a stage's reply lost the memo race -- a concurrent request (another
    thread, process or App instance) stored another reply for the same request first.
    That stored one is what every later request gets, so a run built on ours must not be
    cached (page_cache, the /product cache): the next request rebuilds from the winner.
    `rejected`: the stages that fell back on (or refused) their reply."""
    outcome = {"superseded": False, "rejected": []}
    token = _OUTCOMES.set(_OUTCOMES.get() + (outcome,))
    try:
        yield outcome
    finally:
        _OUTCOMES.reset(token)


def accept_reply(stage: str, query: Optional[str] = None) -> bool:
    """`stage` USED the reply it just got: keep it (a memo hit is already kept).

    `query` (the sanitised shopper text) is stored only as a hash, for `purge --query`.
    Returns False when the memo already held ANOTHER reply for this request (a concurrent
    request committed first); the open memo_outcome() scopes are marked superseded."""
    pending = _PENDING.get()
    if pending is None or pending[0] != stage:
        return True
    _PENDING.set(None)
    _, key, raw, model = pending
    if raw is None:
        return True
    ours = llm_memo.commit(stage, key, raw, model=model, query=query)
    if not ours:
        for outcome in _OUTCOMES.get():
            outcome["superseded"] = True
    return ours


def reject_reply(stage: str) -> None:
    """`stage` fell back on the reply it just got: never keep it, and forget a replayed
    one (a guard or parser change now refuses it), so the next search asks again."""
    for outcome in _OUTCOMES.get():
        outcome["rejected"].append(stage)
    pending = _PENDING.get()
    if pending is None or pending[0] != stage:
        return
    _PENDING.set(None)
    _, key, raw, _ = pending
    if raw is None:
        llm_memo.discard(stage, key)


def _memo_key(client, prompt: str, temperature: float) -> str:
    return llm_memo.make_key(cfg.LLM_PROVIDER, getattr(client, "base_url", None),
                             client.memo_request(prompt, temperature))


def _reply_text(raw) -> str:
    if not isinstance(raw, str):
        raw = _message_text(raw)
    if not raw.strip():
        raise LLMError("LLM returned no text content (reasoning-only reply?)")
    return raw


def _parse_reply(raw: str):
    try:
        return _parse_json_reply(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM returned invalid JSON: {raw!r}") from e


def generate_json(prompt: str, temperature: float = 0.2, *, memo: bool = True,
                  key_text: Optional[str] = None) -> dict:
    """Call the backend and parse the reply as JSON, tolerating code-fence wrappers.

    Inside memo_stage(...) with cfg.LLM_MEMO on, a reply the stage accepted earlier for
    the byte-identical request is replayed instead: 0 provider calls, no usage recorded
    (`memo_hits` counts it), and it is parsed and screened by today's code like a fresh
    one. A backend that cannot describe its request (no memo_request, e.g. a test fake)
    is never memoised. memo=False always asks the provider: the start-up probe, which a
    replayed reply would let pass over a revoked grant.

    `key_text` (or memo_stage's), when given, stands in for `prompt` in the memo KEY only
    (the provider is always sent `prompt`): the stages pass the prompt rendered with the
    canonical query (page_cache.canonical_query), so "Black Dress" and "black  dress" --
    one page_cache entry within a process -- also share their replies after a restart."""
    _PENDING.set(None)
    client = get_llm_client()
    stage = _MEMO_STAGE.get() if memo else None
    keyed = bool(stage) and llm_memo.enabled() and hasattr(client, "memo_request")
    if key_text is None:
        key_text = _MEMO_KEY_TEXT.get()
    keyed_text = prompt if key_text is None else key_text
    if keyed:
        key = _memo_key(client, keyed_text, temperature)
        stored = llm_memo.lookup(stage, key)
        if stored is not None:
            try:
                parsed = _parse_reply(_reply_text(stored))
            except LLMError:
                llm_memo.discard(stage, key)   # unreadable by today's parser: ask again
            else:
                _record_memo_hit()
                _PENDING.set((stage, key, None, getattr(client, "model", None)))
                return parsed
    raw = _reply_text(client.generate_json(prompt, temperature))
    parsed = _parse_reply(raw)
    if keyed:
        # Keyed again from what was SENT: a JSON-mode or temperature refusal during this
        # call changed the request, and only the final attempt's reply came back
        _PENDING.set((stage, _memo_key(client, keyed_text, temperature), raw,
                      getattr(client, "model", None)))
    return parsed
