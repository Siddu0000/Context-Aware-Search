"""Standard IR metrics. Pure functions, no dependencies.

A "relevant_set" is a set of product titles considered relevant for a query
(usually derived from textual criteria in eval_queries.json).
A "retrieved" list is the ranked product titles returned by the pipeline.
"""

import math
from typing import List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """1/rank of first relevant item, else 0."""
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Binary-relevance NDCG@K."""
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 1)
    # Ideal DCG assumes all relevant items are at the top.
    n_rel = min(len(relevant), k)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg
