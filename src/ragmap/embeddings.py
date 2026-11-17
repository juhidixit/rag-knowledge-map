"""Embedding backends.

`SentenceTransformerEmbedder` is the real model. `HashingEmbedder` is a dependency-free,
deterministic stand-in used by unit tests and CI so they never download weights.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

DEFAULT_EMBED_MODEL = os.environ.get("RAGMAP_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_RERANK_MODEL = os.environ.get("RAGMAP_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = min(self.model.max_seq_length or 256, 256)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model_name

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return emb.astype(np.float32)


class HashingEmbedder:
    """Bag-of-words feature hashing + L2 norm. Not semantic, but deterministic and fast."""

    def __init__(self, dim: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.name = f"hashing-{dim}"
        self._vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2", ngram_range=(1, 2))

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype(np.float32)


def get_embedder(name: str | None = None) -> Embedder:
    name = name or DEFAULT_EMBED_MODEL
    if name.startswith("hashing"):
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name)
