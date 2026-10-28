"""Audio loading and chunking.

Long-form audio (6-20 hours per audiobook) cannot be transcribed in one pass; we cut it into
overlapping fixed-length chunks on silence boundaries where possible, transcribe each chunk,
then stitch transcripts back together using the overlap to de-duplicate words.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float
    samples: np.ndarray

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_audio(path: str, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load mono float32 PCM at ``target_sr`` using soundfile/librosa (optional deps)."""
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install storygraph[audio] to load audio files") from exc
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), int(sr)


def energy_vad(
    samples: np.ndarray, sr: int, frame_ms: int = 30, threshold_db: float = -40.0
) -> np.ndarray:
    """Frame-level speech mask from short-time energy. Cheap stand-in for a neural VAD."""
    frame = max(1, int(sr * frame_ms / 1000))
    n = len(samples) // frame
    if n == 0:
        return np.zeros(0, dtype=bool)
    frames = samples[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    return db > threshold_db


def _nearest_silence(mask: np.ndarray, target_frame: int, search: int) -> int:
    """Return the frame index of the nearest non-speech frame around ``target_frame``."""
    lo, hi = max(0, target_frame - search), min(len(mask), target_frame + search)
    candidates = np.where(~mask[lo:hi])[0]
    if candidates.size == 0:
        return target_frame
    return int(lo + candidates[np.argmin(np.abs(candidates + lo - target_frame))])


def chunk_audio(
    samples: np.ndarray,
    sr: int,
    chunk_seconds: float = 600.0,
    overlap_seconds: float = 5.0,
    snap_to_silence: bool = True,
    frame_ms: int = 30,
) -> Iterator[Chunk]:
    """Yield overlapping chunks, snapping boundaries to silence when a VAD mask allows it."""
    if chunk_seconds <= overlap_seconds:
        raise ValueError("chunk_seconds must exceed overlap_seconds")
    total = len(samples) / sr
    mask = energy_vad(samples, sr, frame_ms=frame_ms) if snap_to_silence else None
    frame_s = frame_ms / 1000
    step = chunk_seconds - overlap_seconds
    start, idx = 0.0, 0
    while start < total:
        end = min(total, start + chunk_seconds)
        if mask is not None and end < total:
            f = _nearest_silence(mask, int(end / frame_s), search=int(10 / frame_s))
            end = max(start + 1.0, min(total, f * frame_s))
        a, b = int(start * sr), int(end * sr)
        yield Chunk(index=idx, start=start, end=end, samples=samples[a:b])
        if end >= total:
            break
        start = max(end - overlap_seconds, start + step * 0.5)
        idx += 1
