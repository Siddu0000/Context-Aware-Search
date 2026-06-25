"""Multi-key Gemini API rotator with automatic 429 failover.

Implements  's suggestion (June 4): create multiple free Gemini API
keys with personal Gmail IDs, rotate between them when one hits quota.

Strategy:
1. Round-robin across the configured keys (load-balances burn).
2. On 429 RESOURCE_EXHAUSTED, mark that key as cooling for `cool_seconds`
   and immediately retry the next key.
3. If ALL keys are cooling, raise — the caller's try/except will fall back.

Why round-robin instead of "use key 1 until exhausted": round-robin distributes
load and avoids one key burning to zero while others sit idle.
"""

import logging
import threading
import time
from typing import List, Optional

from google import genai

logger = logging.getLogger(__name__)


class KeyExhausted(Exception):
    """Raised when every configured key is in cool-down."""


class GeminiKeyRotator:
    """Wraps a pool of API keys, presenting a single .generate_content() interface.

    Thread-safe: a Lock guards the rotation pointer and cool-down map.
    """

    def __init__(self, api_keys: List[str], cool_seconds: int = 60):
        if not api_keys:
            raise ValueError("At least one Gemini API key is required.")
        self._keys = list(api_keys)
        self._clients = [genai.Client(api_key=k) for k in self._keys]
        self._cursor = 0
        self._cool_until: dict[int, float] = {}
        self._lock = threading.Lock()
        self._cool_seconds = cool_seconds
        logger.info("Gemini key rotator initialized with %d key(s).", len(self._keys))

    def _next_index(self, skip: Optional[set] = None) -> int:
        """Advance the round-robin cursor, skipping cooling keys."""
        skip = skip or set()
        with self._lock:
            n = len(self._clients)
            now = time.time()
            for _ in range(n):
                idx = self._cursor
                self._cursor = (self._cursor + 1) % n
                if idx in skip:
                    continue
                if self._cool_until.get(idx, 0) > now:
                    continue
                return idx
        raise KeyExhausted("All Gemini API keys are currently cooling.")

    def _mark_cooling(self, idx: int) -> None:
        with self._lock:
            self._cool_until[idx] = time.time() + self._cool_seconds
        logger.warning(
            "Marked Gemini key #%d as cooling for %ds.", idx, self._cool_seconds
        )

    def generate_content(self, model: str, contents: str, config: dict):
        """Identical signature to genai.Client.models.generate_content().

        Rotates keys on 429 errors. Raises the last error if all keys are
        cooling.
        """
        tried = set()
        last_err: Optional[Exception] = None
        for _ in range(len(self._clients)):
            try:
                idx = self._next_index(skip=tried)
            except KeyExhausted as e:
                last_err = e
                break
            client = self._clients[idx]
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as e:
                tried.add(idx)
                last_err = e
                msg = repr(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    self._mark_cooling(idx)
                    logger.info("Key #%d hit quota; trying next key.", idx)
                    continue
                raise
        if last_err:
            raise last_err
        raise KeyExhausted("No keys available.")

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                "total_keys": len(self._clients),
                "cooling_now": sum(
                    1 for t in self._cool_until.values() if t > now
                ),
            }
