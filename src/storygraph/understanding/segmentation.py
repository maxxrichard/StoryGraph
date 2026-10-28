"""Topical segmentation of transcripts into chapter-like segments.

Two methods:

* ``texttiling`` – lexical cohesion between adjacent sentence windows (Hearst, 1997). Cheap,
  language-agnostic, good enough for audiobooks whose chapters shift vocabulary.
* ``embedding-breakpoints`` – cosine distance between consecutive sentence-window embeddings;
  better on podcasts where topics shift without vocabulary shifting much.

Both produce depth scores; boundaries are placed at local maxima above mean + std * k, then
merged / split to respect ``min_segment_words`` / ``max_segment_words``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from storygraph.schemas import Segment, Transcript, Utterance

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "the a an and or but of to in on at for with by from as is are was were be been being it its this that "
    "these those he she they them his her their we you i me my our your not no yes so if then than there here "
    "what which who whom when where why how all any some such only own same too very just can will would should "
    "could do does did done have has had having".split()
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2]


def _cosine_counts(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a if t in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def texttiling_scores(sentences: Sequence[str], window: int) -> np.ndarray:
    """Gap score between the ``window`` sentences before and after each sentence boundary."""
    toks = [Counter(tokenize(s)) for s in sentences]
    n = len(sentences)
    scores = np.zeros(max(0, n - 1))
    for gap in range(n - 1):
        left: Counter[str] = Counter()
        right: Counter[str] = Counter()
        for c in toks[max(0, gap - window + 1) : gap + 1]:
            left.update(c)
        for c in toks[gap + 1 : gap + 1 + window]:
            right.update(c)
        scores[gap] = _cosine_counts(left, right)
    return scores


def embedding_breakpoint_scores(
    sentences: Sequence[str], window: int, embed: Callable[[list[str]], np.ndarray]
) -> np.ndarray:
    """Similarity between the mean embedding of the windows either side of each boundary."""
    n = len(sentences)
    if n < 2:
        return np.zeros(0)
    vecs = embed(list(sentences))
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    scores = np.zeros(n - 1)
    for gap in range(n - 1):
        left = vecs[max(0, gap - window + 1) : gap + 1].mean(axis=0)
        right = vecs[gap + 1 : gap + 1 + window].mean(axis=0)
        scores[gap] = float(left @ right / ((np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9))
    return scores


def depth_scores(sim: np.ndarray, smoothing: int = 3) -> np.ndarray:
    """Hearst depth: how far a valley sits below the peaks on either side."""
    if sim.size == 0:
        return sim
    if smoothing > 1:
        k = np.ones(smoothing) / smoothing
        sim = np.convolve(sim, k, mode="same")
    depth = np.zeros_like(sim)
    for i in range(len(sim)):
        left_peak = sim[: i + 1].max()
        right_peak = sim[i:].max()
        depth[i] = (left_peak - sim[i]) + (right_peak - sim[i])
    return depth


def pick_boundaries(depth: np.ndarray, k: float = 0.5) -> list[int]:
    """Boundaries are local maxima of depth above ``mean + k * std``."""
    if depth.size == 0:
        return []
    thr = depth.mean() + k * depth.std()
    out: list[int] = []
    for i, d in enumerate(depth):
        if d <= thr:
            continue
        if (i == 0 or depth[i - 1] <= d) and (i == len(depth) - 1 or depth[i + 1] <= d):
            out.append(i + 1)  # boundary *before* sentence i+1
    return out


def _enforce_lengths(
    sentences: Sequence[str], boundaries: list[int], min_words: int, max_words: int
) -> list[int]:
    """Merge segments under ``min_words`` into a neighbour; split those over ``max_words``."""
    counts = [len(s.split()) for s in sentences]
    bounds = sorted(set(b for b in boundaries if 0 < b < len(sentences)))
    # merge short
    changed = True
    while changed and bounds:
        changed = False
        edges = [0, *bounds, len(sentences)]
        for j in range(len(edges) - 1):
            seg_words = sum(counts[edges[j] : edges[j + 1]])
            if seg_words < min_words and len(edges) > 2:
                # drop whichever adjacent boundary exists (prefer merging into the shorter neighbour)
                if j == 0:
                    bounds.remove(edges[1])
                elif j == len(edges) - 2:
                    bounds.remove(edges[j])
                else:
                    left = sum(counts[edges[j - 1] : edges[j]])
                    right = sum(counts[edges[j + 1] : edges[j + 2]])
                    bounds.remove(edges[j] if left <= right else edges[j + 1])
                changed = True
                break
    # split long
    edges = [0, *bounds, len(sentences)]
    final: list[int] = []
    for j in range(len(edges) - 1):
        a, b = edges[j], edges[j + 1]
        acc = 0
        for i in range(a, b):
            acc += counts[i]
            if acc >= max_words and i + 1 < b:
                final.append(i + 1)
                acc = 0
        if b < len(sentences):
            final.append(b)
    return sorted(set(final))


def _locate(
    utterances: Sequence[Utterance], sentence_char_offsets: list[int], text_len: int
) -> list[float]:
    """Map sentence character offsets to timestamps by linear interpolation over utterances."""
    if not utterances:
        return [0.0 for _ in sentence_char_offsets]
    starts: list[float] = []
    cum = 0
    spans: list[tuple[int, int, float, float]] = []
    for u in utterances:
        ln = len(u.text) + 1
        spans.append((cum, cum + ln, u.start, u.end))
        cum += ln
    for off in sentence_char_offsets:
        t = utterances[-1].end
        for a, b, s, e in spans:
            if a <= off < b:
                t = s + (e - s) * ((off - a) / max(1, b - a))
                break
        starts.append(t)
    return starts


def segment_transcript(
    transcript: Transcript,
    method: str = "texttiling",
    window_sentences: int = 20,
    min_segment_words: int = 300,
    max_segment_words: int = 2500,
    smoothing: int = 3,
    embed: Callable[[list[str]], np.ndarray] | None = None,
) -> list[Segment]:
    text = transcript.text
    sentences = split_sentences(text)
    if not sentences:
        return []
    if method == "embedding-breakpoints":
        if embed is None:
            raise ValueError("embedding-breakpoints requires an embed callable")
        sim = embedding_breakpoint_scores(sentences, window_sentences, embed)
    elif method == "texttiling":
        sim = texttiling_scores(sentences, window_sentences)
    else:
        raise ValueError(f"unknown segmentation method {method!r}")
    bounds = pick_boundaries(depth_scores(sim, smoothing))
    bounds = _enforce_lengths(sentences, bounds, min_segment_words, max_segment_words)

    offsets: list[int] = []
    cursor = 0
    for s in sentences:
        pos = text.find(s, cursor)
        offsets.append(pos if pos >= 0 else cursor)
        cursor = offsets[-1] + len(s)
    times = _locate(transcript.utterances, offsets, len(text))

    edges = [0, *bounds, len(sentences)]
    segments: list[Segment] = []
    for idx in range(len(edges) - 1):
        a, b = edges[idx], edges[idx + 1]
        seg_text = " ".join(sentences[a:b])
        start = times[a]
        end = times[b] if b < len(times) else transcript.duration
        segments.append(
            Segment(
                title_id=transcript.title_id,
                index=idx,
                start=start,
                end=max(start, end),
                text=seg_text,
                word_count=len(seg_text.split()),
                speakers=_speaker_shares(transcript.utterances, start, max(start, end)),
            )
        )
    return segments


def _speaker_shares(utterances: Sequence[Utterance], start: float, end: float) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for u in utterances:
        if u.speaker and u.end > start and u.start < end:
            counts[u.speaker] += len(u.text.split())
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}
