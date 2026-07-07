"""Sponsored (paid-ad) products.

Sponsored items are ordinary catalog products flagged in data/sponsored.json
(by parent_asin, with a sponsor name and bid). They are NOT ranked separately:
the normal pipeline retrieves and reranks everything, and any sponsored product
that lands in the reranked pool is boosted to the top of the results (ordered
by bid). A sponsored product that isn't relevant enough to reach the reranked
pool simply doesn't appear — relevance is enforced by the ranking itself.
"""

import json
import logging
from typing import Dict, Optional

import app.config as cfg

logger = logging.getLogger(__name__)

_map: Optional[Dict[str, dict]] = None
_mtime: Optional[float] = None


def sponsored_map() -> Dict[str, dict]:
    """Return {parent_asin: {"sponsor", "bid"}}, reloading if the file changed.
    Empty dict if the config is missing or unreadable (best-effort)."""
    global _map, _mtime
    path = cfg.SPONSORED_CONFIG
    if not path.exists():
        return {}
    try:
        m = path.stat().st_mtime
        if _map is None or m != _mtime:
            spec = json.loads(path.read_text(encoding="utf-8"))
            entries = spec.get("sponsored", []) if isinstance(spec, dict) else spec
            built: Dict[str, dict] = {}
            for e in entries:
                asin = e.get("parent_asin")
                if asin:
                    built[str(asin)] = {
                        "sponsor": e.get("sponsor", "Sponsored"),
                        "bid": float(e.get("bid", 0.0)),
                    }
            _map, _mtime = built, m
        return _map or {}
    except Exception as e:
        logger.warning("Could not load sponsored config (%s).", repr(e))
        return {}
