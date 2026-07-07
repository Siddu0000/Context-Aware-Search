"""Rating-aware scoring helpers.

Combines `average_rating` and `rating_number` into a confidence-weighted
signal that the reranker mixes with semantic relevance. The math is
Bayesian-average shrinkage toward a global prior — the standard fix for
the "5.0 from 1 reviewer outranks 4.3 from 3000 reviewers" pathology.

Formula:
    bayesian_avg = (n * r + C * m) / (n + C)

  r = product's average rating
  n = number of ratings the product has
  m = global prior mean rating (we use 4.0 for Amazon)
  C = prior weight, i.e. effective sample size of the prior (we use 10)

Higher n means the result stays close to r. Lower n pulls it toward m.

Why this is the right design:
- Deterministic. Same inputs → bit-identical outputs every time
  (preserves  's "are results deterministic" requirement).
- Non-destructive. The LLM still owns the relevance ranking; rating
  is a small post-processing nudge.
- Transparent. The weight is a single tunable parameter you can defend.
"""

import math
from typing import Optional

BAYESIAN_PRIOR_MEAN = 4.0
BAYESIAN_PRIOR_WEIGHT = 10.0


def _is_real_number(x) -> bool:
    """Treat None and NaN as "missing". Both arrive from pandas CSV reads."""
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
    """Shrinkage-adjusted rating in [0, 5].

    Examples:
        >>> round(bayesian_rating(5.0, 2), 2)
        4.17
        >>> round(bayesian_rating(4.3, 3000), 3)
        4.299
        >>> round(bayesian_rating(4.6, 16), 2)
        4.37
        >>> bayesian_rating(None, 0)
        4.0
    """
    if not _is_real_number(average_rating) or not _is_real_number(rating_count):
        return prior_mean
    r = float(average_rating)
    n = float(rating_count)
    if n <= 0:
        return prior_mean
    return (n * r + prior_weight * prior_mean) / (n + prior_weight)


def rating_quality_tag(rating_count: Optional[int]) -> str:
    """Human-readable confidence tag for the LLM prompt.

    The LLM sees this in the candidate block so it can reason about
    rating reliability without needing to do math itself.
    """
    if not _is_real_number(rating_count) or float(rating_count) <= 0:
        return "no_ratings"
    n = int(float(rating_count))
    if n < 10:
        return "low_confidence"
    if n < 100:
        return "medium_confidence"
    return "high_confidence"


def blend_score(rerank_score: float, bayesian_avg: float, weight: float = 0.15) -> float:
    """Combine 0-100 rerank score with rating signal (mapped to 0-100).

    weight=0 → pure rerank. weight=1 → pure rating. Default 0.15 means
    relevance dominates 85% and rating nudges 15%.

    Returns float; callers can int() it for display.
    """
    if weight < 0 or weight > 1:
        raise ValueError(f"weight must be in [0,1], got {weight}")
    rating_normalized = (bayesian_avg / 5.0) * 100
    return (1 - weight) * float(rerank_score) + weight * rating_normalized
