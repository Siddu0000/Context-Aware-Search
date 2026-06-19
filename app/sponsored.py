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
import statistics
from typing import List, Optional

import pandas as pd

import app.config as cfg
from app.search import get_dataframe, intent_similarity

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


def _clean_row(row: dict) -> dict:
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}


def get_sponsored(
    query: str,
    organic_results: List[dict],
    intents: Optional[List[str]] = None,
    max_slots: int = None,
) -> List[dict]:
    """Return up to max_slots sponsored products that are RELEVANT to the query.

    Relevance gate (the important bit): each candidate ad is scored by its max
    cosine similarity to the query intents — the SAME vectors retrieval used —
    and must reach SPONSORED_REL_RATIO x the MEDIAN organic score on the page,
    i.e. be about as on-topic as the products we're already showing. That
    relative gate is what keeps an off-topic ad (e.g. a women's dress on a
    men's-shirt search) out, where a fixed threshold can't (all apparel embeds
    ~0.4 similar). We do NOT fall back to a vertical match: if nothing clears
    the bar, we return [] (relevance over fill rate).

    Ordering among relevant ads is by bid, then by relevance. Best-effort:
    returns [] on any problem. Deduped against the organic results by
    parent_asin (then non-empty title).
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

    organic_titles = {r.get("Product_title", "") for r in organic_results}
    organic_asins = {
        r.get("parent_asin") for r in organic_results if r.get("parent_asin")
    }
    terms = [t for t in (intents or [query]) if t]

    # Resolve each entry to a real catalog row (+ embedding index), skipping
    # anything already shown organically.
    candidates = []  # list of (entry, product_dict, catalog_index)
    for entry in entries:
        asin = entry.get("parent_asin")
        if not asin:
            continue
        rows = df[df["parent_asin"].astype(str) == str(asin)]
        if rows.empty:
            logger.warning("Sponsored asin %s not found in catalog.", asin)
            continue
        idx = int(rows.index[0])
        product = _clean_row(rows.iloc[0].to_dict())
        title = product.get("Product_title", "")
        if asin in organic_asins or (title and title in organic_titles):
            continue  # already shown organically; don't double up
        candidates.append((entry, product, idx))

    if not candidates:
        return []

    # Relevance gate (relative to organic) + bid/relevance ordering.
    organic_scores = [
        r["score"]
        for r in organic_results
        if isinstance(r.get("score"), (int, float)) and not pd.isna(r["score"])
    ]
    if organic_scores:
        floor = max(
            cfg.SPONSORED_MIN_RELEVANCE,
            cfg.SPONSORED_REL_RATIO * statistics.median(organic_scores),
        )
    else:
        floor = cfg.SPONSORED_MIN_RELEVANCE

    sims = intent_similarity(terms, [idx for _, _, idx in candidates])
    relevant = []
    for entry, product, idx in candidates:
        score = sims.get(idx, 0.0)
        if score < floor:
            continue  # not as relevant as the products we're already showing
        product = dict(product)
        product["is_sponsored"] = True
        product["sponsor"] = entry.get("sponsor", "Sponsored")
        product["reason"] = f"Sponsored by {entry.get('sponsor', 'a partner')}"
        product["catalog_index"] = idx
        product["sponsored_relevance"] = round(score, 3)
        relevant.append((entry.get("bid", 0.0), score, product))

    relevant.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [p for _, _, p in relevant[:max_slots]]
