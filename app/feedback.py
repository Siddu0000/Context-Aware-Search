"""User feedback collection: one thumbs-up/down event per row, appended to JSONL."""

import json
import logging
import time
from typing import Optional

from app.config import FEEDBACK_LOG

logger = logging.getLogger(__name__)


def record_feedback(
    query: str,
    product_title: str,
    rating: int,
    rank: int,
    reason: Optional[str] = None,
) -> None:
    """Append one feedback event."""
    if rating not in (-1, 1):
        raise ValueError("rating must be -1 or +1")
    event = {
        "ts": time.time(),
        "query": query,
        "product_title": product_title,
        "rating": rating,
        "rank": rank,
        "reason": reason,
    }
    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except OSError as e:
        # A Databricks App's disk can be read-only (and is wiped on restart anyway):
        # losing a thumbs-up must not turn into a 500 for the shopper
        logger.warning("Feedback not persisted (%s).", type(e).__name__)
        return
    # Rating and rank only: query text stays out of logs (safety.md), and a
    # Databricks App's logs are readable by everyone who can manage the app
    logger.info("Feedback recorded: rating=%+d rank=%d", rating, rank)
