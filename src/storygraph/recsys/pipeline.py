"""End-to-end recommend(): retrieve -> featurise -> rank -> explain.

This is the object the serving API holds; everything it needs (index, title features, towers,
ranker) is injected so the same code runs against OpenSearch + LightGBM in production and the
in-memory index + linear ranker in tests.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from storygraph.config import CandidateConfig
from storygraph.recsys.explain import template_explanation
from storygraph.recsys.features import (
    CROSS_FEATURES,
    ListenerFeatures,
    TitleFeatures,
    cross_features,
)
from storygraph.recsys.retrieval import VectorIndex, retrieve_candidates
from storygraph.schemas import Recommendation, RecommendationResponse
from storygraph.training.ranker import Ranker
from storygraph.training.two_tower import TwoTowerWeights


@dataclass
class Recommender:
    index: VectorIndex
    titles: Mapping[str, TitleFeatures]
    ranker: Ranker
    towers: TwoTowerWeights | None = None
    candidates_cfg: CandidateConfig = field(default_factory=CandidateConfig)
    model_version: str = "two_tower_lgbm_v3"
    explain: bool = True

    def _content_query(self, listener: ListenerFeatures) -> np.ndarray | None:
        vecs = [self.titles[t].content_vec for t in listener.history[-10:] if t in self.titles]
        if not vecs:
            return None
        v = np.mean(vecs, axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    def recommend(
        self,
        listener: ListenerFeatures,
        k: int = 10,
        variant: str = "treatment",
        as_of: date | None = None,
    ) -> RecommendationResponse:
        t0 = time.perf_counter()
        exclude = set(listener.history)
        user_vec = (
            self.towers.encode_users(listener.vector()[None, :])[0]
            if self.towers is not None
            else None
        )
        cands = retrieve_candidates(
            self.index, user_vec, self._content_query(listener), exclude, self.candidates_cfg
        )
        cands = [c for c in cands if c.title_id in self.titles]
        if not cands:
            return RecommendationResponse(
                listener_id=listener.listener_id,
                variant=variant,
                model_version=self.model_version,
                items=[],
                latency_ms=0.0,
            )

        X = np.stack(
            [
                cross_features(
                    listener,
                    self.titles[c.title_id],
                    c.scores.get("two_tower", 0.0),
                    c.scores.get("content_knn", 0.0),
                    c.scores.get("popular", 0.0),
                    len(c.scores),
                    as_of=as_of,
                )
                for c in cands
            ]
        )
        assert X.shape[1] == len(CROSS_FEATURES)
        scores = self.ranker.predict(X)
        order = np.argsort(-scores)[:k]
        items = [
            Recommendation(
                title_id=cands[i].title_id,
                score=float(scores[i]),
                rank=r + 1,
                retrieval_sources=cands[i].sources,
                explanation=template_explanation(
                    listener, self.titles[cands[i].title_id], cands[i].sources
                )
                if self.explain
                else None,
            )
            for r, i in enumerate(order)
        ]
        return RecommendationResponse(
            listener_id=listener.listener_id,
            variant=variant,
            model_version=self.model_version,
            items=items,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
