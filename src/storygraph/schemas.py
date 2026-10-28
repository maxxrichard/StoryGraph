"""Core data contracts.

These pydantic models are shared by the pipeline, the feature store writers, the LLM
tagger (whose JSON output is validated against :class:`ContentSignals`), the OpenSearch
indexer and the serving API. Changing a field here is a schema migration.
"""

from __future__ import annotations

import hashlib
from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Kind(str, Enum):
    AUDIOBOOK = "audiobook"
    PODCAST = "podcast"
    LECTURE = "lecture"
    OTHER = "other"


class NarrativeArc(str, Enum):
    """Coarse narrative shapes (after Vonnegut / Reagan et al.)."""

    RAGS_TO_RICHES = "rags_to_riches"
    RICHES_TO_RAGS = "riches_to_rags"
    MAN_IN_HOLE = "man_in_hole"
    ICARUS = "icarus"
    CINDERELLA = "cinderella"
    OEDIPUS = "oedipus"
    EPISODIC = "episodic"
    EXPOSITORY = "expository"


class Title(BaseModel):
    title_id: str
    title: str
    kind: Kind = Kind.OTHER
    duration_min: float = Field(ge=0)
    narrator: str | None = None
    release_date: date | None = None
    transcript_path: str | None = None
    audio_path: str | None = None

    @property
    def content_key(self) -> str:
        """Stable idempotency key: changes if the source asset path or title changes."""
        raw = f"{self.title_id}|{self.title}|{self.transcript_path}|{self.audio_path}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class Word(BaseModel):
    text: str
    start: float
    end: float
    speaker: str | None = None
    confidence: float | None = None


class Utterance(BaseModel):
    text: str
    start: float
    end: float
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    title_id: str
    language: str = "en"
    utterances: list[Utterance]
    asr_model: str | None = None

    @property
    def text(self) -> str:
        return " ".join(u.text.strip() for u in self.utterances if u.text.strip())

    @property
    def duration(self) -> float:
        return max((u.end for u in self.utterances), default=0.0)

    @classmethod
    def from_text(cls, title_id: str, text: str, words_per_second: float = 2.5) -> Transcript:
        """Build a transcript with synthetic timings from plain text (used for tests / demos)."""
        utterances: list[Utterance] = []
        t = 0.0
        for para in (p.strip() for p in text.split("\n")):
            if not para:
                continue
            dur = max(1.0, len(para.split()) / words_per_second)
            utterances.append(Utterance(text=para, start=t, end=t + dur))
            t += dur
        return cls(title_id=title_id, utterances=utterances, asr_model="synthetic")


class Segment(BaseModel):
    """A topically coherent span of a transcript (roughly a chapter or a podcast section)."""

    title_id: str
    index: int
    start: float
    end: float
    text: str
    word_count: int
    speakers: dict[str, float] = Field(
        default_factory=dict, description="speaker -> share of words"
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> Segment:
        if self.end < self.start:
            raise ValueError("segment end precedes start")
        return self


class Label(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)


class SpeakerProfile(BaseModel):
    speaker_id: str
    share: float = Field(ge=0.0, le=1.0)
    role: Literal["narrator", "host", "guest", "character", "unknown"] = "unknown"


class ContentSignals(BaseModel):
    """Structured output of the content-understanding pipeline for one unit (segment or title).

    This is also the JSON schema used for guided decoding in the LLM tagger, so keep it flat
    and keep enum-like fields as plain strings validated against the taxonomy.
    """

    title_id: str
    segment_index: int | None = None
    genres: list[Label] = Field(default_factory=list, max_length=3)
    themes: list[Label] = Field(default_factory=list, max_length=8)
    moods: list[Label] = Field(default_factory=list, max_length=4)
    narrative_arc: NarrativeArc | None = None
    tension: float | None = Field(default=None, ge=0.0, le=1.0)
    content_warnings: list[str] = Field(default_factory=list)
    speakers: list[SpeakerProfile] = Field(default_factory=list)
    one_line_summary: str | None = Field(default=None, max_length=300)

    @field_validator("genres", "themes", "moods")
    @classmethod
    def _dedupe(cls, labels: list[Label]) -> list[Label]:
        seen: set[str] = set()
        out: list[Label] = []
        for lab in sorted(labels, key=lambda x: -x.confidence):
            key = lab.name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(Label(name=key, confidence=lab.confidence))
        return out

    def label_names(self, field: str) -> list[str]:
        return [lab.name for lab in getattr(self, field)]


class ChapterSummary(BaseModel):
    title_id: str
    segment_index: int
    heading: str = Field(max_length=80)
    summary: str = Field(max_length=1200)
    spoiler_free: bool = True


class Recommendation(BaseModel):
    title_id: str
    score: float
    rank: int
    retrieval_sources: list[str] = Field(default_factory=list)
    explanation: str | None = None


class RecommendationResponse(BaseModel):
    listener_id: str
    variant: str
    model_version: str
    items: list[Recommendation]
    latency_ms: float
