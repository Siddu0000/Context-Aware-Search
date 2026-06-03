"""Lightweight latency + cost instrumentation.

Used by main.py to attach per-stage timing to every response. The eval
harness also uses these to produce latency tables.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StageTimings:
    """Accumulator for per-stage latencies in a single request."""

    timings_ms: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = round((time.perf_counter() - t0) * 1000, 2)

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)

    def to_dict(self) -> dict:
        out = dict(self.timings_ms)
        out["total"] = self.total_ms
        return out


# Very rough token-count proxy. For Gemini we don't have a free tokenizer
# always available, so this is a chars/4 approximation — good enough for
# order-of-magnitude cost tracking, not for billing.
def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
