"""Sponsored (paid-ad) products flagged by parent_asin in data/sponsored.json."""

import json
import logging
from typing import Dict, Optional

import app.config as cfg

logger = logging.getLogger(__name__)

_map: Optional[Dict[str, dict]] = None
_mtime: Optional[float] = None


def sponsored_map() -> Dict[str, dict]:
    """Return {parent_asin: {"sponsor", "bid"}}, reloading if the file changed."""
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
