"""Keyword (lexical) search engine — the "existing retailer stack" that CAS
plugs into. Modeled on eBay's Cassini: retrieve by keyword match over the
title + item-specifics + category, then rank with a "Best Match" blend of text
relevance and business signals (rating quality + review-count popularity).

Deliberately NOT semantic. This is the baseline that context-aware search sits
in front of, and the thing whose misses CAS is meant to catch. Self-contained
BM25 (Okapi), no extra dependencies beyond numpy/pandas.
"""

import math
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "for", "of", "to", "and", "or", "in", "on", "with",
    "my", "your", "i", "me", "is", "are", "be", "this", "that", "it",
    "something", "some", "any", "at", "by", "from", "as",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


class KeywordSearchEngine:
    """Okapi BM25 over a product catalog with Best-Match re-ranking.

    text_fields are concatenated into the searchable document; title_weight
    repeats the title so title matches count more (field boosting, like
    Cassini weighting the title). Business signals: average_rating (quality)
    and rating_number (popularity proxy — we have no sell-through data)."""

    def __init__(
        self,
        text_fields: Tuple[str, ...] = (
            "Product_title", "categ_lvl2_name", "color", "material",
            "occasion", "prod_description",
        ),
        title_field: str = "Product_title",
        rating_field: str = "average_rating",
        count_field: str = "rating_number",
        title_weight: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
        w_relevance: float = 0.7,
        w_popularity: float = 0.2,
        w_quality: float = 0.1,
        # Fields the router's coverage (hit/miss) signal is measured against.
        # Deliberately EXCLUDES prod_description: long descriptions contain
        # incidental words ("breathable", "day"...) that inflate coverage and
        # make intent queries look like keyword hits. Title + category +
        # structured color/material is "what the product is" — the honest
        # basis for "did the best lexical match actually contain what was
        # asked?". color/material are short, precise values ("black",
        # "cotton") that legit lexical queries name but titles sometimes omit.
        # `occasion` is deliberately NOT here: its values (Casual, Party Wear,
        # Wedding) overlap intent-query vocabulary and would false-hit the
        # very queries the CAS fallback exists for.
        coverage_fields: Tuple[str, ...] = (
            "Product_title", "categ_lvl2_name", "color", "material",
        ),
    ):
        self.text_fields = text_fields
        self.title_field = title_field
        self.coverage_fields = coverage_fields
        self.rating_field = rating_field
        self.count_field = count_field
        self.title_weight = title_weight
        self.k1, self.b = k1, b
        self.w_relevance = w_relevance
        self.w_popularity = w_popularity
        self.w_quality = w_quality
        self._df: Optional[pd.DataFrame] = None
        self._docs: List[List[str]] = []
        self._coverage_tokens: List[set] = []
        self._tf: List[dict] = []
        self._doc_len: np.ndarray = np.array([])
        self._avgdl: float = 0.0
        self._idf: dict = {}

    def index(self, df: pd.DataFrame) -> None:
        self._df = df.reset_index(drop=True)
        self._docs, self._tf, doc_len = [], [], []
        self._coverage_tokens: List[set] = []
        df_count: dict = {}
        for _, row in self._df.iterrows():
            parts = []
            cov_parts = []
            for f in self.text_fields:
                val = row.get(f)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                reps = self.title_weight if f == self.title_field else 1
                parts.extend([str(val)] * reps)
                if f in self.coverage_fields:
                    cov_parts.append(str(val))
            tokens = _tokenize(" ".join(parts))
            self._docs.append(tokens)
            self._coverage_tokens.append(set(_tokenize(" ".join(cov_parts))))
            counts: dict = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self._tf.append(counts)
            doc_len.append(len(tokens))
            for term in counts:
                df_count[term] = df_count.get(term, 0) + 1
        self._doc_len = np.array(doc_len, dtype=float)
        self._avgdl = float(self._doc_len.mean()) if len(self._doc_len) else 0.0
        n = len(self._docs)
        self._idf = {
            term: math.log(1 + (n - dfq + 0.5) / (dfq + 0.5))
            for term, dfq in df_count.items()
        }
        # Business signals, coerced once (robust to messy numeric columns).
        self._rating = self._numeric_column(self._df, self.rating_field)
        self._count = self._numeric_column(self._df, self.count_field)

    def _bm25_scores(self, q_terms: List[str]) -> np.ndarray:
        n = len(self._docs)
        scores = np.zeros(n, dtype=float)
        if not q_terms or self._avgdl == 0:
            return scores
        for i in range(n):
            tf, dl = self._tf[i], self._doc_len[i]
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s += idf * (f * (self.k1 + 1)) / denom
            scores[i] = s
        return scores

    @staticmethod
    def _numeric_column(df: pd.DataFrame, col: str) -> np.ndarray:
        """Coerce a catalog column to a float array, robust to real-world mess:
        object dtype, thousands separators ('1,203'), blanks, 'N/A', etc. all
        become 0.0 rather than raising. Amazon data has these in rating_number
        / average_rating, which otherwise crash the Best-Match math."""
        if col not in df.columns:
            return np.zeros(len(df), dtype=float)
        s = df[col]
        if s.dtype == object:
            s = s.astype(str).str.replace(",", "", regex=False).str.strip()
        return pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def search(self, query: str, k: int = 12) -> Tuple[List[dict], int, float, float]:
        """Return (results, num_matched, top_relevance, top_coverage).

        num_matched = docs sharing >=1 content term with the query.
        top_relevance = best raw BM25 score.
        top_coverage = fraction of the query's distinct content terms that
        appear in the top-ranked result — the router's primary miss signal
        ("did the best lexical match actually contain what was asked?").
        Robust to result count and query length. Results are Best-Match ranked."""
        if self._df is None:
            raise RuntimeError("KeywordSearchEngine.index() not called.")
        q_terms = _tokenize(query)
        bm25 = self._bm25_scores(q_terms)
        matched_mask = bm25 > 0
        num_matched = int(matched_mask.sum())
        top_relevance = float(bm25.max()) if len(bm25) else 0.0
        if num_matched == 0:
            return [], 0, 0.0, 0.0

        idx = np.where(matched_mask)[0]
        rel = self._norm(bm25[idx])
        rating_arr = self._rating[idx] / 5.0
        count_arr = np.log1p(self._count[idx])
        pop = self._norm(count_arr)
        best_match = (
            self.w_relevance * rel
            + self.w_popularity * pop
            + self.w_quality * rating_arr
        )
        order = np.argsort(-best_match)

        # Coverage of the BEST LEXICAL match (BM25 argmax) — NOT the blended
        # rank-1, whose popularity/quality terms can put a partial match on
        # top and misreport an exact-match query as a miss. Measured against
        # coverage_fields tokens only (title/category/color/material) — NOT
        # the full document: a description mentioning "breathable"/"day" in
        # passing must not make an intent query look like a keyword hit.
        # (num_matched > 0 here, so argmax is a genuine positive match.)
        q_set = set(q_terms)
        top_doc_idx = int(np.argmax(bm25))
        top_tokens = self._coverage_tokens[top_doc_idx]
        top_coverage = (
            len(q_set & top_tokens) / len(q_set) if q_set else 0.0
        )

        results = []
        for j in order[:k]:
            cat_idx = int(idx[j])
            raw = self._df.iloc[cat_idx].to_dict()
            # JSON-safe: ~99% of raw catalog rows carry NaN somewhere (price,
            # color, rating...). Starlette serializes with allow_nan=False, so
            # a NaN would 500 the endpoint after it returns; the UI's string
            # handling (e.g. color.title()) also chokes on NaN floats.
            row = {
                key: (None if isinstance(v, float) and math.isnan(v) else v)
                for key, v in raw.items()
            }
            # The UI's click-through (/product detail page) is keyed on the
            # catalog row id; the engine indexes the same catalog dataframe in
            # the same order, so the positional index IS the catalog_index.
            row["catalog_index"] = cat_idx
            row["keyword_score"] = round(float(bm25[idx[j]]), 4)
            row["best_match_score"] = round(float(best_match[j]), 4)
            results.append(row)
        return results, num_matched, top_relevance, top_coverage
