"""FastAPI service: recommendations, explained picks, chapter summaries and health.

The app is built by :func:`create_app` with explicit dependencies so tests can inject an in-memory
index, stub LLM and linear ranker. ``app`` at module level builds a demo instance from the sample
catalog so ``uvicorn storygraph.serving.api:app`` works out of the box.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from storygraph.config import SegmentationConfig
from storygraph.genai.summaries import ChapterSummariser
from storygraph.recsys.features import listener_from_history
from storygraph.recsys.pipeline import Recommender
from storygraph.schemas import (
    ChapterSummary,
    ContentSignals,
    RecommendationResponse,
    Segment,
    Title,
)
from storygraph.serving.ab import Bucketing

log = logging.getLogger(__name__)


class InteractionIn(BaseModel):
    title_id: str
    date: dt.date | None = None
    started_seconds: float = 0.0
    listened_seconds: float = 0.0
    finished: bool = False


class RecommendRequest(BaseModel):
    listener_id: str
    history: list[InteractionIn] = Field(default_factory=list)
    k: int = Field(default=10, ge=1, le=100)
    variant_override: str | None = None


@dataclass
class LatencyTracker:
    window: int = 2000
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    errors: int = 0
    total: int = 0

    def record(self, ms: float, ok: bool = True) -> None:
        self.samples.append(ms)
        self.total += 1
        self.errors += 0 if ok else 1

    def p95(self) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        return s[min(len(s) - 1, int(0.95 * len(s)))]

    @property
    def availability(self) -> float:
        return 1.0 if self.total == 0 else 1 - self.errors / self.total


@dataclass
class ServingContext:
    recommender: Recommender
    summariser: ChapterSummariser | None
    bucketing: Bucketing
    titles: dict[str, Title]
    segments: dict[str, list[Segment]]
    signals: dict[str, list[ContentSignals]]
    latency: LatencyTracker = field(default_factory=LatencyTracker)


def create_app(ctx: ServingContext) -> FastAPI:
    app = FastAPI(
        title="StoryGraph",
        version="0.4.0",
        description="Content understanding + recommendations for long-form audio",
    )
    app.state.ctx = ctx

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "titles": len(ctx.titles),
            "p95_ms": round(ctx.latency.p95(), 2),
            "availability": round(ctx.latency.availability, 5),
        }

    @app.post("/v1/recommend", response_model=RecommendationResponse)
    def recommend(req: RecommendRequest) -> RecommendationResponse:
        t0 = time.perf_counter()
        try:
            variant = ctx.bucketing.variant_for(req.listener_id, req.variant_override)
            history = [r.model_dump() for r in req.history]
            listener = listener_from_history(req.listener_id, history, ctx.recommender.titles)
            resp = ctx.recommender.recommend(listener, k=req.k, variant=variant.name)
            resp.model_version = variant.model
            ctx.latency.record((time.perf_counter() - t0) * 1000, ok=True)
            return resp
        except Exception as exc:  # noqa: BLE001 - convert to 500 but track availability
            ctx.latency.record((time.perf_counter() - t0) * 1000, ok=False)
            log.exception("recommend failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/v1/titles/{title_id}/signals", response_model=ContentSignals)
    def title_signals(title_id: str) -> ContentSignals:
        sigs = ctx.signals.get(title_id)
        if not sigs:
            raise HTTPException(status_code=404, detail="unknown title")
        return sigs[-1]  # title-level aggregate is stored last

    @app.get("/v1/titles/{title_id}/chapters", response_model=list[ChapterSummary])
    def chapters(
        title_id: str, limit: int = Query(default=50, ge=1, le=500)
    ) -> list[ChapterSummary]:
        if ctx.summariser is None:
            raise HTTPException(status_code=503, detail="summariser unavailable")
        title, segs = ctx.titles.get(title_id), ctx.segments.get(title_id)
        if title is None or not segs:
            raise HTTPException(status_code=404, detail="unknown title")
        sigs = ctx.signals.get(title_id)
        return ctx.summariser.summarise_title(title, segs[:limit], sigs[:limit] if sigs else None)

    @app.get("/v1/experiment/{listener_id}")
    def experiment(listener_id: str) -> dict[str, str]:
        v = ctx.bucketing.variant_for(listener_id)
        return {"listener_id": listener_id, "variant": v.name, "model": v.model}

    return app


def build_demo_context(
    catalog: str = "data/sample/catalog.jsonl",
    experiment_cfg: str | None = "configs/experiment.yaml",
) -> ServingContext:
    """Index the sample catalog with stub models so the API runs locally with zero infra."""
    import json
    from pathlib import Path

    import numpy as np

    from storygraph.recsys.features import CROSS_FEATURES, TitleFeatures
    from storygraph.recsys.retrieval import InMemoryIndex
    from storygraph.schemas import Transcript
    from storygraph.serving.llm_client import StubLLM
    from storygraph.training.ranker import LinearRanker
    from storygraph.understanding.embeddings import HashingEmbedder
    from storygraph.understanding.pipeline import understand_title
    from storygraph.understanding.tagger import SegmentTagger

    llm = StubLLM()
    embedder = HashingEmbedder(dim=256)
    tagger = SegmentTagger(llm)
    index = InMemoryIndex()
    titles: dict[str, Title] = {}
    tfeats: dict[str, TitleFeatures] = {}
    segments: dict[str, list[Segment]] = {}
    signals: dict[str, list[ContentSignals]] = {}
    with open(catalog, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = Title(**json.loads(line))
            if not t.transcript_path or not Path(t.transcript_path).exists():
                continue
            tr = Transcript.from_text(
                t.title_id, Path(t.transcript_path).read_text(encoding="utf-8")
            )
            u = understand_title(t, tr, embedder, tagger, seg_cfg=_demo_seg_cfg())
            titles[t.title_id] = t
            segments[t.title_id] = u.segments
            signals[t.title_id] = [*u.segment_signals, u.title_signals]
            s = u.title_signals
            tfeats[t.title_id] = TitleFeatures(
                title_id=t.title_id,
                content_vec=u.title_embedding,
                genres=s.label_names("genres"),
                themes=s.label_names("themes"),
                moods=s.label_names("moods"),
                duration_min=t.duration_min,
                narrator=t.narrator,
                release_date=t.release_date,
                popularity_7d=float(len(u.segments)),
                narrative_arc=s.narrative_arc.value if s.narrative_arc else None,
            )
            index.add(
                t.title_id, {"content_vec": u.title_embedding}, popularity=float(len(u.segments))
            )
    # a hand-set linear ranker: prefer retrieval scores and affinity matches
    w = np.zeros(len(CROSS_FEATURES))
    for name, val in {
        "two_tower_score": 1.0,
        "content_knn_score": 1.0,
        "popularity_score": 0.3,
        "genre_match": 0.8,
        "theme_overlap": 0.5,
        "mood_match": 0.5,
        "same_narrator": 0.2,
    }.items():
        w[CROSS_FEATURES.index(name)] = val
    ranker = LinearRanker(
        weights=w, feature_names=list(CROSS_FEATURES), mean=np.zeros(len(w)), std=np.ones(len(w))
    )
    rec = Recommender(
        index=index, titles=tfeats, ranker=ranker, towers=None, model_version="demo-linear"
    )
    return ServingContext(
        recommender=rec,
        summariser=ChapterSummariser(llm),
        bucketing=Bucketing.from_file(experiment_cfg),
        titles=titles,
        segments=segments,
        signals=signals,
    )


def _demo_seg_cfg() -> SegmentationConfig:
    return SegmentationConfig(
        window_sentences=4, min_segment_words=80, max_segment_words=400, smoothing=1
    )


def _maybe_demo_app() -> FastAPI:
    if os.environ.get("STORYGRAPH_NO_DEMO"):
        return FastAPI(title="StoryGraph (empty)")
    try:
        return create_app(build_demo_context())
    except FileNotFoundError:  # pragma: no cover - running outside the repo
        return FastAPI(title="StoryGraph (no sample data)")


app = _maybe_demo_app()
