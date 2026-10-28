"""LLM tagging of segments into :class:`ContentSignals`.

Flow per segment:

1. Build a prompt with the taxonomy and the segment text (truncated to ``max_input_tokens``).
2. Call the LLM (vLLM OpenAI-compatible endpoint, guided JSON decoding) via an :class:`LLMBackend`.
3. Parse + validate the JSON into ``ContentSignals``; snap labels to the taxonomy.
4. Aggregate per-segment signals into a title-level ``ContentSignals``.

The LoRA adapter is trained in :mod:`storygraph.training.lora` on editor-labelled segments.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, Literal, Protocol, cast

from storygraph.config import AggregationConfig
from storygraph.schemas import ContentSignals, Label, NarrativeArc, Segment, SpeakerProfile
from storygraph.understanding.taxonomy import DEFAULT_TAXONOMY, Taxonomy

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert editor cataloguing long-form audio. Read the transcript excerpt and return "
    "ONLY a JSON object matching the schema. Use labels from the provided vocabularies; never invent "
    "genres. Confidences are calibrated probabilities. Do not include spoilers in one_line_summary."
)


class LLMBackend(Protocol):
    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> str: ...


def build_prompt(segment: Segment, taxonomy: Taxonomy, max_chars: int = 18_000) -> str:
    text = segment.text if len(segment.text) <= max_chars else segment.text[:max_chars] + " …"
    return (
        f"{taxonomy.prompt_block()}\n\n"
        f"NARRATIVE_ARCS: {', '.join(a.value for a in NarrativeArc)}\n\n"
        f"TITLE_ID: {segment.title_id}\nSEGMENT_INDEX: {segment.index}\n"
        f"WORDS: {segment.word_count}\n\nTRANSCRIPT:\n{text}\n\n"
        "Return JSON with keys: title_id, segment_index, genres, themes, moods (each a list of "
        "{name, confidence}), narrative_arc, tension (0-1), content_warnings, speakers, one_line_summary."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_signals(
    raw: str, taxonomy: Taxonomy, title_id: str, segment_index: int | None
) -> ContentSignals:
    """Parse model output into ContentSignals, tolerating prose around the JSON and loose labels."""
    m = _JSON_RE.search(raw)
    if not m:
        raise ValueError("no JSON object in model output")
    data = json.loads(m.group(0))
    data["title_id"] = title_id
    data["segment_index"] = segment_index
    for field in ("genres", "themes", "moods"):
        snapped: list[dict[str, Any]] = []
        for item in data.get(field, []) or []:
            if isinstance(item, str):
                item = {"name": item, "confidence": 0.6}
            name = taxonomy.snap(str(item.get("name", "")), field)
            if name is None:
                log.debug("dropping off-taxonomy %s label %r", field, item.get("name"))
                continue
            conf = float(item.get("confidence", 0.6))
            snapped.append({"name": name, "confidence": min(1.0, max(0.0, conf))})
        limit = {"genres": 3, "themes": 8, "moods": 4}[field]
        data[field] = sorted(snapped, key=lambda x: -x["confidence"])[:limit]
    arc = data.get("narrative_arc")
    if arc is not None and arc not in {a.value for a in NarrativeArc}:
        data["narrative_arc"] = None
    return ContentSignals(**data)


class SegmentTagger:
    def __init__(
        self, backend: LLMBackend, taxonomy: Taxonomy = DEFAULT_TAXONOMY, max_chars: int = 18_000
    ):
        self.backend = backend
        self.taxonomy = taxonomy
        self.max_chars = max_chars
        self._schema = ContentSignals.model_json_schema()

    def tag(self, segment: Segment) -> ContentSignals:
        raw = self.backend.complete_json(
            SYSTEM_PROMPT, build_prompt(segment, self.taxonomy, self.max_chars), self._schema
        )
        return parse_signals(raw, self.taxonomy, segment.title_id, segment.index)

    def tag_many(self, segments: Sequence[Segment]) -> list[ContentSignals]:
        return [self.tag(s) for s in segments]


def aggregate_title_signals(
    segments: Sequence[Segment],
    per_segment: Sequence[ContentSignals],
    cfg: AggregationConfig | None = None,
) -> ContentSignals:
    """Roll per-segment signals up to one title-level ContentSignals."""
    cfg = cfg or AggregationConfig()
    if len(segments) != len(per_segment) or not segments:
        raise ValueError("segments and per_segment signals must align and be non-empty")
    total_words = sum(s.word_count for s in segments) or 1
    weights = [s.word_count / total_words for s in segments]

    def weighted(field: str) -> dict[str, float]:
        acc: dict[str, float] = defaultdict(float)
        for w, sig in zip(weights, per_segment):
            for lab in getattr(sig, field):
                acc[lab.name] += w * lab.confidence
        return acc

    # genres: weighted vote, keep top-3 above floor
    g = weighted("genres")
    genres = [
        Label(name=n, confidence=min(1.0, c)) for n, c in sorted(g.items(), key=lambda x: -x[1])
    ]
    genres = [
        x
        for x in genres
        if x.confidence >= cfg.confidence_floor * max((y.confidence for y in genres), default=1)
    ][:3]

    # themes: keep those present in >= themes_min_share of the words
    share: dict[str, float] = defaultdict(float)
    conf: dict[str, list[float]] = defaultdict(list)
    for w, sig in zip(weights, per_segment):
        for lab in sig.themes:
            share[lab.name] += w
            conf[lab.name].append(lab.confidence)
    themes = [
        Label(name=n, confidence=sum(conf[n]) / len(conf[n]))
        for n, s in sorted(share.items(), key=lambda x: -x[1])
        if s >= cfg.themes_min_share
    ][:8]

    # moods: trajectory — first appearance order, de-duplicated, top-4 by weighted mass
    m = weighted("moods")
    order: list[str] = []
    for sig in per_segment:
        for lab in sig.moods:
            if lab.name not in order:
                order.append(lab.name)
    top = set(n for n, _ in sorted(m.items(), key=lambda x: -x[1])[:4])
    moods = [Label(name=n, confidence=min(1.0, m[n])) for n in order if n in top]

    arcs = Counter(sig.narrative_arc for sig in per_segment if sig.narrative_arc)
    tensions = [(w, sig.tension) for w, sig in zip(weights, per_segment) if sig.tension is not None]
    warnings = sorted({w for sig in per_segment for w in sig.content_warnings})

    sp_share: dict[str, float] = defaultdict(float)
    sp_role: dict[str, Counter[str]] = defaultdict(Counter)
    for w, sig in zip(weights, per_segment):
        for sp in sig.speakers:
            sp_share[sp.speaker_id] += w * sp.share
            sp_role[sp.speaker_id][sp.role] += 1
    speakers = [
        SpeakerProfile(
            speaker_id=k,
            share=min(1.0, v),
            role=cast(
                Literal["narrator", "host", "guest", "character", "unknown"],
                sp_role[k].most_common(1)[0][0],
            ),
        )
        for k, v in sorted(sp_share.items(), key=lambda x: -x[1])
    ]

    return ContentSignals(
        title_id=segments[0].title_id,
        segment_index=None,
        genres=genres,
        themes=themes,
        moods=moods,
        narrative_arc=arcs.most_common(1)[0][0] if arcs else None,
        tension=sum(w * t for w, t in tensions) / sum(w for w, _ in tensions) if tensions else None,
        content_warnings=warnings,
        speakers=speakers,
        one_line_summary=per_segment[0].one_line_summary,
    )
