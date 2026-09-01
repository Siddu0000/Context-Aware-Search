"""Embedding layer: sentence-transformers or OpenAI behind one interface."""

import logging
import os
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class Embedder:
    """Single object that owns one embedding backend, loaded lazily."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._st_model = None
        self._openai_client = None

    @property
    def is_openai(self) -> bool:
        return self.model_name.startswith("text-embedding-")

    def _ensure_loaded(self):
        if self.is_openai:
            if self._openai_client is None:
                try:
                    from openai import OpenAI
                except ImportError as e:
                    raise RuntimeError(
                        "openai package not installed. Run `pip install openai` "
                        "or set EMBEDDING_MODEL to a sentence-transformers model."
                    ) from e
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "OPENAI_API_KEY is required for OpenAI embedding models."
                    )
                self._openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI embedding client ready: %s", self.model_name)
        else:
            if self._st_model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading sentence-transformer: %s", self.model_name)
                self._st_model = SentenceTransformer(self.model_name)
                max_seq = int(os.getenv("EMBED_MAX_SEQ_LEN", "128"))
                try:
                    if self._st_model.max_seq_length and self._st_model.max_seq_length > max_seq:
                        logger.info(
                            "Capping max_seq_length %s -> %s for speed.",
                            self._st_model.max_seq_length, max_seq,
                        )
                        self._st_model.max_seq_length = max_seq
                except Exception:
                    pass
                logger.info("Sentence-transformer loaded.")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into a (N, D) float32 array."""
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        if self.is_openai:
            batch_size = 100
            chunks = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = self._openai_client.embeddings.create(
                    model=self.model_name, input=batch
                )
                chunks.extend(d.embedding for d in resp.data)
            return np.asarray(chunks, dtype=np.float32)

        n = len(texts)
        show_progress = n >= 500
        if show_progress:
            import time as _time
            t0 = _time.perf_counter()
            logger.info("Encoding %s texts with %s ...", f"{n:,}", self.model_name)

        out = self._st_model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "64")),
        ).astype(np.float32)

        if show_progress:
            elapsed = _time.perf_counter() - t0
            rate = n / elapsed if elapsed > 0 else 0
            logger.info(
                "Encoded %s texts in %.1fs (%.0f texts/sec).",
                f"{n:,}", elapsed, rate,
            )
        return out


_default_embedder: Embedder | None = None


def get_default_embedder() -> Embedder:
    global _default_embedder
    if _default_embedder is None:
        from app.config import EMBEDDING_MODEL

        _default_embedder = Embedder(EMBEDDING_MODEL)
    return _default_embedder


def embed_text(texts: List[str]) -> np.ndarray:
    return get_default_embedder().encode(texts)


def search_vectors(query_embedding, product_embeddings, top_k: int = 10):
    """Cosine-similarity top-K. Returns (indices, scores)."""
    similarities = cosine_similarity([query_embedding], product_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return top_indices, similarities[top_indices]
