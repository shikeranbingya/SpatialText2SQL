"""Embedding providers for relation-aware spatial database synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingProvider(ABC):
    """Abstract embedding interface."""

    model_name: str

    @abstractmethod
    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by sentence-transformers."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 32
    device: Optional[str] = None
    _model: Any = field(default=None, init=False, repr=False)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        return np.asarray(embeddings, dtype=float)


@dataclass
class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for tests."""

    model_name: str = "mock-embedding"
    vectors_by_text: Optional[dict[str, Sequence[float]]] = None
    resolver: Optional[Callable[[str], Sequence[float]]] = None
    dimension: int = 8

    def _hash_vector(self, text: str) -> np.ndarray:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest, dtype=np.uint8).astype(float)
        if self.dimension <= len(raw):
            vector = raw[: self.dimension]
        else:
            repeats = int(np.ceil(self.dimension / len(raw)))
            vector = np.tile(raw, repeats)[: self.dimension]
        return vector

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            if self.vectors_by_text and text in self.vectors_by_text:
                vector = np.asarray(self.vectors_by_text[text], dtype=float)
            elif self.resolver is not None:
                vector = np.asarray(self.resolver(text), dtype=float)
            else:
                vector = self._hash_vector(text)
            vectors.append(vector)
        matrix = np.vstack(vectors) if vectors else np.zeros((0, self.dimension), dtype=float)
        if normalize_embeddings and matrix.size:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix
