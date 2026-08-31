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
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    logger.info(
        "Feedback recorded: q=%r prod=%r rating=%+d", query, product_title, rating
    )
