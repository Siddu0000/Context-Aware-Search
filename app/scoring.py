"""Rating-aware scoring: Bayesian shrinkage of a product's rating toward a prior."""

import math
from typing import Optional

# (n*r + C*m)/(n + C): low n shrinks toward m, so a lone 5.0 can't beat 4.3 of 3000
BAYESIAN_PRIOR_MEAN = 4.0
BAYESIAN_PRIOR_WEIGHT = 10.0


def _is_real_number(x) -> bool:
    """Pandas reads missing ratings as float NaN, which passes `is not None`."""
    if x is None:
        return False
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return not math.isnan(v)


def bayesian_rating(
    average_rating: Optional[float],
    rating_count: Optional[int],
    prior_mean: float = BAYESIAN_PRIOR_MEAN,
    prior_weight: float = BAYESIAN_PRIOR_WEIGHT,
) -> float:
    """Shrinkage-adjusted rating in [0, 5]; prior_mean when the rating is missing."""
    if not _is_real_number(average_rating) or not _is_real_number(rating_count):
        return prior_mean
    r = float(average_rating)
    n = float(rating_count)
    if n <= 0:
        return prior_mean
    return (n * r + prior_weight * prior_mean) / (n + prior_weight)


def rating_quality_tag(rating_count: Optional[int]) -> str:
    """Confidence tag shown to the LLM so it can weigh rating reliability itself."""
    if not _is_real_number(rating_count) or float(rating_count) <= 0:
        return "no_ratings"
    n = int(float(rating_count))
    if n < 10:
        return "low_confidence"
    if n < 100:
        return "medium_confidence"
    return "high_confidence"


def blend_score(rerank_score: float, bayesian_avg: float, weight: float = 0.15) -> float:
    """Blend a 0-100 rerank score with the rating (weight=0 rerank, 1 rating)."""
    if weight < 0 or weight > 1:
        raise ValueError(f"weight must be in [0,1], got {weight}")
    rating_normalized = (bayesian_avg / 5.0) * 100
    return (1 - weight) * float(rerank_score) + weight * rating_normalized
