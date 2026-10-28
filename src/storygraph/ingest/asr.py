"""Automatic speech recognition over audio chunks with overlap stitching.

The production backend is faster-whisper (CTranslate2) on GPU; the :class:`ASRBackend`
protocol lets tests and local runs use a stub.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from storygraph.ingest.audio import Chunk
from storygraph.schemas import Transcript, Utterance, Word


class ASRBackend(Protocol):
    name: str

    def transcribe(self, chunk: Chunk, sr: int, language: str | None) -> list[Utterance]: ...


class FasterWhisperBackend:
    """Thin adapter over faster-whisper with word timestamps."""

    def __init__(
        self, model: str = "large-v3", compute_type: str = "int8_float16", beam_size: int = 5
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install storygraph[audio] for faster-whisper") from exc
        self.name = f"faster-whisper/{model}"
        self.beam_size = beam_size
        self._model = WhisperModel(model, compute_type=compute_type)

    def transcribe(
        self, chunk: Chunk, sr: int, language: str | None
    ) -> list[Utterance]:  # pragma: no cover
        segments, _ = self._model.transcribe(
            chunk.samples,
            beam_size=self.beam_size,
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
        out: list[Utterance] = []
        for seg in segments:
            words = [
                Word(
                    text=w.word.strip(),
                    start=chunk.start + w.start,
                    end=chunk.start + w.end,
                    confidence=w.probability,
                )
                for w in (seg.words or [])
            ]
            out.append(
                Utterance(
                    text=seg.text.strip(),
                    start=chunk.start + seg.start,
                    end=chunk.start + seg.end,
                    words=words,
                )
            )
        return out


def stitch(utterances: Iterable[Utterance], overlap_seconds: float) -> list[Utterance]:
    """Merge utterances from overlapping chunks, dropping duplicates inside the overlap.

    Utterances are assumed to be in chunk order. A later utterance that ends inside the region
    already covered by earlier chunks (up to a small tolerance) is a duplicate from the overlap
    and is dropped; one that extends past the frontier carries new content and is kept.
    """
    stitched: list[Utterance] = []
    frontier = -1.0
    tol = min(0.5, overlap_seconds / 10)
    for u in utterances:
        if u.end <= frontier + tol:
            continue
        stitched.append(u)
        frontier = max(frontier, u.end)
    return stitched


def transcribe_chunks(
    title_id: str,
    chunks: Iterable[Chunk],
    sr: int,
    backend: ASRBackend,
    language: str | None = None,
    overlap_seconds: float = 5.0,
) -> Transcript:
    raw: list[Utterance] = []
    for chunk in chunks:
        raw.extend(backend.transcribe(chunk, sr, language))
    return Transcript(
        title_id=title_id,
        language=language or "en",
        utterances=stitch(raw, overlap_seconds),
        asr_model=backend.name,
    )
