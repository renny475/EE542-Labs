"""Embedding providers and the on-disk vector format.

Vectors are stored in MySQL as little-endian float32 bytes and compared by
cosine similarity, so every stored vector is L2-normalised first.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """``text-embedding-3-small`` (1536 dims) through the official client."""

    dim = 1536

    def __init__(self, model: str, api_key: str, batch_size: int = 100) -> None:
        from openai import OpenAI  # imported here so tests never need the key

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key, max_retries=3)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            # The API may return items out of order; sort by index to be safe.
            for item in sorted(response.data, key=lambda d: d.index):
                out.append(list(item.embedding))
        return out


class FakeEmbedder:
    """Deterministic stand-in for tests: sha256(text) bytes become a unit vector."""

    def __init__(self, dim: int = 8, model: str = "fake") -> None:
        self.dim = dim
        self.model = model

    def _one(self, text: str) -> list[float]:
        digest = b""
        while len(digest) < self.dim:
            digest += hashlib.sha256((text + str(len(digest))).encode("utf-8")).digest()
        raw = np.frombuffer(digest[: self.dim], dtype=np.uint8).astype(np.float32) - 127.5
        return normalise(raw[np.newaxis, :])[0].tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


def to_blob(vec: Sequence[float]) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").astype(np.float32)


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise each row; all-zero rows stay zero instead of becoming NaN."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
