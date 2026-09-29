"""
Embedding generation.

Wraps a sentence-transformers bi-encoder so the rest of the system never
touches a specific model API directly (swap models here only).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import get_settings


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,   # cosine similarity via inner product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return Embedder(settings.embedding_model_name)
