"""Text embeddings for segments and titles.

``SentenceTransformerEmbedder`` is the production path (bge-m3). ``HashingEmbedder`` is a
dependency-free fallback that gives deterministic, roughly cohesion-preserving vectors so the
rest of the pipeline (segmentation, kNN retrieval, two-tower content tower) can be exercised
without a GPU or model download.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Feature-hashed unigram+bigram embedding with signed hashing and L2 normalisation."""

    def __init__(self, dim: int = 256, seed: int = 13):
        self.dim = dim
        self.name = f"hashing-{dim}"
        self._seed = seed

    def _index(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(f"{self._seed}:{token}".encode(), digest_size=8).digest()
        v = int.from_bytes(h, "little")
        return v % self.dim, 1.0 if (v >> 60) & 1 else -1.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _TOKEN_RE.findall(text.lower())
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for g in grams:
                j, sign = self._index(g)
                out[i, j] += sign
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-m3", normalize: bool = True, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install storygraph[llm] for sentence-transformers") from exc
        self._m = SentenceTransformer(model)
        self.dim = int(self._m.get_sentence_embedding_dimension() or 0)
        self.name = model
        self.normalize = normalize
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs model
        return np.asarray(
            self._m.encode(
                list(texts), batch_size=self.batch_size, normalize_embeddings=self.normalize
            )
        )


def pool_title_embedding(
    segment_vecs: np.ndarray, weights: Sequence[float] | None = None
) -> np.ndarray:
    """Length-weighted mean of segment embeddings, renormalised."""
    if segment_vecs.ndim != 2 or segment_vecs.shape[0] == 0:
        raise ValueError("need at least one segment vector")
    w = np.ones(segment_vecs.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    v = (segment_vecs * w[:, None]).sum(axis=0)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


def get_embedder(name: str, dim: int = 256) -> Embedder:
    if name.startswith("hashing"):
        return HashingEmbedder(dim=dim)
    return SentenceTransformerEmbedder(model=name)
