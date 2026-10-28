"""One-title orchestration of the understanding stages (segment -> embed -> tag -> narrative).

Batch jobs call :func:`understand_title` per title inside a shard; the local CLI and tests call it
directly with stub backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from storygraph.config import AggregationConfig, SegmentationConfig
from storygraph.schemas import ContentSignals, Segment, Title, Transcript
from storygraph.understanding.embeddings import Embedder, pool_title_embedding
from storygraph.understanding.narrative import classify_arc, tension_curve
from storygraph.understanding.segmentation import segment_transcript
from storygraph.understanding.speakers import infer_roles, prune, speaker_shares
from storygraph.understanding.tagger import SegmentTagger, aggregate_title_signals


@dataclass
class TitleUnderstanding:
    title: Title
    segments: list[Segment]
    segment_signals: list[ContentSignals]
    title_signals: ContentSignals
    segment_embeddings: np.ndarray
    title_embedding: np.ndarray
    tension: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_record(self) -> dict:
        """Flat record for the feature store / OpenSearch document."""
        s = self.title_signals
        return {
            "title_id": self.title.title_id,
            "title": self.title.title,
            "kind": self.title.kind.value,
            "duration_min": self.title.duration_min,
            "genres": s.label_names("genres"),
            "themes": s.label_names("themes"),
            "moods": s.label_names("moods"),
            "narrative_arc": s.narrative_arc.value if s.narrative_arc else None,
            "tension_curve": self.tension.round(4).tolist(),
            "speaker_count": len(s.speakers),
            "content_warnings": s.content_warnings,
            "segment_count": len(self.segments),
            "content_vec": self.title_embedding.round(6).tolist(),
            "release_date": self.title.release_date.isoformat()
            if self.title.release_date
            else None,
        }


def understand_title(
    title: Title,
    transcript: Transcript,
    embedder: Embedder,
    tagger: SegmentTagger,
    seg_cfg: SegmentationConfig | None = None,
    agg_cfg: AggregationConfig | None = None,
    arc_bins: int = 12,
) -> TitleUnderstanding:
    seg_cfg = seg_cfg or SegmentationConfig()
    segments = segment_transcript(
        transcript,
        method=seg_cfg.method,
        window_sentences=seg_cfg.window_sentences,
        min_segment_words=seg_cfg.min_segment_words,
        max_segment_words=seg_cfg.max_segment_words,
        smoothing=seg_cfg.smoothing,
        embed=embedder.encode if seg_cfg.method == "embedding-breakpoints" else None,
    )
    if not segments:
        raise ValueError(f"{title.title_id}: empty transcript")

    seg_vecs = embedder.encode([s.text for s in segments])
    title_vec = pool_title_embedding(seg_vecs, [s.word_count for s in segments])

    seg_signals = tagger.tag_many(segments)
    title_signals = aggregate_title_signals(segments, seg_signals, agg_cfg)

    curve = tension_curve(segments, seg_signals, bins=arc_bins)
    if title_signals.narrative_arc is None:
        title_signals.narrative_arc = classify_arc(curve)
    if not title_signals.speakers:
        title_signals.speakers = prune(
            infer_roles(speaker_shares(transcript.utterances), title.kind)
        )

    return TitleUnderstanding(
        title=title,
        segments=segments,
        segment_signals=seg_signals,
        title_signals=title_signals,
        segment_embeddings=seg_vecs,
        title_embedding=title_vec,
        tension=curve,
    )
