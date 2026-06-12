"""Featured / sponsored (paid-ad) layer — kept auditable and SEPARATE.

There is no real ad inventory in this PoC, so sponsored placements are read
from data/sponsored.json (curated). Each entry names a product by parent_asin,
plus a sponsor name, an optional bid (for ordering between competing ads),
and optional targeting (keywords / verticals). At search time we surface the
sponsored products whose targeting matches the query — or, failing a keyword
hit, the vertical of the organic results — capped at SPONSORED_MAX, each
flagged with is_sponsored=true and the sponsor name.

Per .claude/rules/safety.md this layer is intentionally NOT blended into the
organic ranking. Sponsored items travel in their own `sponsored` response key
so the organic relevance order stays auditable and the paid placement is
always distinguishable from earned placement.
"""

import json
import logging
from typing import List, Optional

import pandas as pd

import app.config as cfg
from app.search import get_dataframe

logger = logging.getLogger(__name__)

# Cached config + a parent_asin -> row lookup, built lazily and cheaply.
_config: Optional[List[dict]] = None
_config_mtime: Optional[float] = None


def _load_config() -> List[dict]:
    """Read data/sponsored.json, reloading if the file changed on disk."""
    global _config, _config_mtime
    path = cfg.SPONSORED_CONFIG
    if not path.exists():
        return []
    try:
        mtime = path.stat().st_mtime
        if _config is None or mtime != _config_mtime:
            spec = json.loads(path.read_text(encoding="utf-8"))
            _config = spec.get("sponsored", []) if isinstance(spec, dict) else spec
            _config_mtime = mtime
        return _config or []
    except Exception as e:  # noqa: BLE001 — ads are best-effort
        logger.warning("Could not load sponsored config (%s).", repr(e))
        return []


def _targets(entry: dict, query_lc: str, verticals: set) -> bool:
    """True if this sponsored entry should show for the current query.

    Keyword match wins (intent-level targeting). Otherwise fall back to a
    vertical match against the organic results, so a sponsored grocery item
    only appears on grocery-shaped searches — never on a random query.
    """
    keywords = [str(k).lower() for k in entry.get("keywords", [])]
    if any(kw in query_lc for kw in keywords):
        return True
    entry_verticals = {str(v).lower() for v in entry.get("verticals", [])}
    return bool(entry_verticals & verticals)


def _clean_row(row: dict) -> dict:
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}


def get_sponsored(
    query: str, organic_results: List[dict], max_slots: int = None
) -> List[dict]:
    """Return up to max_slots sponsored products relevant to this query.

    Best-effort: returns [] on any problem. Sponsored items are deduped
    against the organic results so the same product never appears twice.
    """
    if max_slots is None:
        max_slots = cfg.SPONSORED_MAX

    entries = _load_config()
    if not entries:
        return []

    try:
        df = get_dataframe()
    except Exception as e:  # noqa: BLE001
        logger.warning("Sponsored layer could not load catalog (%s).", repr(e))
        return []

    query_lc = query.lower()
    organic_titles = {r.get("Product_title", "") for r in organic_results}
    organic_asins = {
        r.get("parent_asin") for r in organic_results if r.get("parent_asin")
    }
    verticals = {
        str(r.get("bsns_vrtcl_name", "")).lower()
        for r in organic_results
        if r.get("bsns_vrtcl_name")
    }

    # Highest bid first; that's the only place bid influences anything.
    matched = [e for e in entries if _targets(e, query_lc, verticals)]
    matched.sort(key=lambda e: e.get("bid", 0.0), reverse=True)

    out: List[dict] = []
    for entry in matched:
        if len(out) >= max_slots:
            break
        asin = entry.get("parent_asin")
        if not asin:
            continue
        rows = df[df["parent_asin"].astype(str) == str(asin)]
        if rows.empty:
            logger.warning("Sponsored asin %s not found in catalog.", asin)
            continue
        product = _clean_row(rows.iloc[0].to_dict())
        # Dedup against organic by ASIN first (the real product key); fall back
        # to a NON-EMPTY title match. Empty titles must not collide.
        title = product.get("Product_title", "")
        if asin in organic_asins or (title and title in organic_titles):
            continue  # already shown organically; don't double up
        product["is_sponsored"] = True
        product["sponsor"] = entry.get("sponsor", "Sponsored")
        product["reason"] = f"Sponsored by {entry.get('sponsor', 'a partner')}"
        out.append(product)

    return out
