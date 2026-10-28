"""Candidate generation from three sources, merged with per-source provenance.

* two-tower: user vector vs title-tower vectors (OpenSearch kNN, inner product)
* content kNN: mean content embedding of the listener's recent finishes vs ``content_vec``
* popularity: top titles by 7-day popularity as a cold-start / exploration floor

:class:`InMemoryIndex` mirrors the OpenSearch kNN semantics for tests and local runs;
:class:`OpenSearchIndex` is the production client.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from storygraph.config import CandidateConfig


@dataclass
class Candidate:
    title_id: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def sources(self) -> list[str]:
        return sorted(self.scores)


class VectorIndex(Protocol):
    def knn(
        self, field_name: str, query: np.ndarray, k: int, exclude: set[str]
    ) -> list[tuple[str, float]]: ...

    def top_popular(self, k: int, exclude: set[str]) -> list[tuple[str, float]]: ...


class InMemoryIndex:
    """Exact brute-force kNN with the same (cosine / inner-product) semantics as the OpenSearch mapping."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._fields: dict[str, list[np.ndarray]] = {}
        self._popularity: list[float] = []
        self._matrices: dict[str, np.ndarray] = {}

    def add(
        self, title_id: str, vectors: Mapping[str, np.ndarray], popularity: float = 0.0
    ) -> None:
        self._ids.append(title_id)
        self._popularity.append(popularity)
        for f, v in vectors.items():
            self._fields.setdefault(f, []).append(np.asarray(v, dtype=np.float32))
        self._matrices.clear()

    def _matrix(self, field_name: str) -> np.ndarray:
        if field_name not in self._matrices:
            self._matrices[field_name] = np.stack(self._fields[field_name])
        return self._matrices[field_name]

    def knn(
        self, field_name: str, query: np.ndarray, k: int, exclude: set[str]
    ) -> list[tuple[str, float]]:
        if field_name not in self._fields:
            return []
        m = self._matrix(field_name)
        q = np.asarray(query, dtype=np.float32)
        sims = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-9)
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for i in order:
            tid = self._ids[i]
            if tid in exclude:
                continue
            out.append((tid, float(sims[i])))
            if len(out) >= k:
                break
        return out

    def top_popular(self, k: int, exclude: set[str]) -> list[tuple[str, float]]:
        order = np.argsort(-np.asarray(self._popularity))
        mx = max(self._popularity) if self._popularity else 1.0
        out = [
            (self._ids[i], self._popularity[i] / (mx or 1.0))
            for i in order
            if self._ids[i] not in exclude
        ]
        return out[:k]

    def __len__(self) -> int:
        return len(self._ids)


class OpenSearchIndex:  # pragma: no cover - needs a cluster
    def __init__(self, hosts: Sequence[str], index: str, **client_kwargs: Any):
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:
            raise ImportError("install storygraph[search] for OpenSearch") from exc
        self.client = OpenSearch(hosts=list(hosts), **client_kwargs)
        self.index = index

    def knn(
        self, field_name: str, query: np.ndarray, k: int, exclude: set[str]
    ) -> list[tuple[str, float]]:
        body = {
            "size": k,
            "_source": False,
            "query": {
                "bool": {
                    "must": [
                        {"knn": {field_name: {"vector": query.tolist(), "k": k + len(exclude)}}}
                    ],
                    "must_not": [{"ids": {"values": list(exclude)}}] if exclude else [],
                }
            },
        }
        hits = self.client.search(index=self.index, body=body)["hits"]["hits"]
        return [(h["_id"], float(h["_score"])) for h in hits][:k]

    def top_popular(self, k: int, exclude: set[str]) -> list[tuple[str, float]]:
        body = {
            "size": k,
            "_source": ["popularity_7d"],
            "sort": [{"popularity_7d": "desc"}],
            "query": {
                "bool": {"must_not": [{"ids": {"values": list(exclude)}}] if exclude else []}
            },
        }
        hits = self.client.search(index=self.index, body=body)["hits"]["hits"]
        return [(h["_id"], float(h["_source"]["popularity_7d"])) for h in hits]

    def bulk_index(self, docs: Iterable[dict[str, Any]]) -> int:
        from opensearchpy.helpers import bulk

        actions = ({"_index": self.index, "_id": d["title_id"], "_source": d} for d in docs)
        ok, _ = bulk(self.client, actions, refresh=False)
        return int(ok)


def retrieve_candidates(
    index: VectorIndex,
    user_tower_vec: np.ndarray | None,
    content_query_vec: np.ndarray | None,
    exclude: set[str],
    cfg: CandidateConfig,
    title_vector_field: str = "title_tower_vec",
    content_vector_field: str = "content_vec",
) -> list[Candidate]:
    """Union of the three sources; each candidate keeps every source score it came from."""
    merged: dict[str, Candidate] = {}

    def absorb(source: str, hits: Sequence[tuple[str, float]]) -> None:
        for tid, score in hits:
            merged.setdefault(tid, Candidate(title_id=tid)).scores[source] = score

    if user_tower_vec is not None:
        absorb("two_tower", index.knn(title_vector_field, user_tower_vec, cfg.k_two_tower, exclude))
    if content_query_vec is not None:
        absorb(
            "content_knn",
            index.knn(content_vector_field, content_query_vec, cfg.k_content_knn, exclude),
        )
    absorb("popular", index.top_popular(cfg.k_popular, exclude))

    # keep the strongest candidates if the union is too large: max of normalised source scores
    ranked = sorted(merged.values(), key=lambda c: -max(c.scores.values()))
    return ranked[: cfg.k_total_max]
