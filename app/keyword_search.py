"""Self-contained BM25 keyword search with a Best-Match blend of relevance and
business signals. Deliberately NOT semantic — this is the CAS baseline."""

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
    """Okapi BM25 over a product catalog with Best-Match re-ranking."""

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
        # Hit/miss basis: excludes prod_description/occasion (they false-hit intents)
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
        # Business signals, coerced once (robust to messy numeric columns)
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
        """Coerce a column to floats; mess ('1,203', 'N/A', blanks) becomes 0.0."""
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
        """Return (results, num_matched, top_relevance, top_coverage), Best-Match ranked.

        top_coverage = fraction of the query's terms present in the top result."""
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

        # BM25 argmax, not blended rank-1: popularity can top-rank a partial match
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
            # Starlette serializes with allow_nan=False — a NaN would 500 the endpoint
            row = {
                key: (None if isinstance(v, float) and math.isnan(v) else v)
                for key, v in raw.items()
            }
            # Same catalog, same order — the positional index IS the catalog_index
            row["catalog_index"] = cat_idx
            row["keyword_score"] = round(float(bm25[idx[j]]), 4)
            row["best_match_score"] = round(float(best_match[j]), 4)
            results.append(row)
        return results, num_matched, top_relevance, top_coverage
