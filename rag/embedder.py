"""Stage 3 - Embedding generator.

Input : text (chunks when indexing, a question when searching).
Output: unit-length float32 vectors. Texts with similar meaning end up with
        vectors that point in similar directions.

The embedding model runs locally (no API key, no quota). It is deliberately
separate from the generation model: the two never need to match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .errors import EmbeddingError


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dimension)."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a vector of shape (dimension,)."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every vector to length 1, so inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class FastEmbedEmbedder:
    """Local embeddings via the `fastembed` library (ONNX runtime, no PyTorch)."""

    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._model = None  # loaded on first use; the first run downloads the model

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(
                    model_name=self.model_name, cache_dir=str(self.cache_dir)
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}': {exc}\n"
                    "The first run must download the model (about 70 MB), so an "
                    "internet connection is required once. Check EMBEDDING_MODEL "
                    "in your .env if you changed it."
                ) from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        model = self._load()
        try:
            vectors = np.array(list(model.passage_embed(texts)), dtype="float32")
        except Exception as exc:
            raise EmbeddingError(f"Embedding the documents failed: {exc}") from exc
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        try:
            vector = np.array(next(iter(model.query_embed(text))), dtype="float32")
        except Exception as exc:
            raise EmbeddingError(f"Embedding the question failed: {exc}") from exc
        return _normalize(vector[None, :])[0]
