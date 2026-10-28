"""Narrative structure signals: a tension curve over the title and a coarse arc label.

The tension curve is the per-bin mean of (a) the LLM's segment tension where available and
(b) a lexicon score, so it degrades gracefully when the LLM pass is skipped. The arc is
classified from the shape of the curve with a handful of templates, which is both cheap and
explainable ("tension rises, falls, then rises again -> man_in_hole").
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from storygraph.schemas import ContentSignals, NarrativeArc, Segment

_TENSION_WORDS = {
    "fear": 1.0,
    "afraid": 1.0,
    "terror": 1.2,
    "scream": 1.1,
    "blood": 0.9,
    "dead": 0.9,
    "death": 0.9,
    "died": 0.8,
    "storm": 0.7,
    "dark": 0.5,
    "silence": 0.4,
    "footsteps": 0.8,
    "gone": 0.5,
    "lost": 0.6,
    "run": 0.5,
    "panic": 1.1,
    "frightening": 1.0,
    "catastrophe": 1.0,
    "danger": 0.9,
    "under": 0.3,
    "stumbled": 0.6,
    "roar": 0.6,
    "frozen": 0.4,
    "empty": 0.5,
    "wet": 0.3,
    "still": 0.2,
    "worse": 0.6,
}
_CALM_WORDS = {
    "laughing": -0.8,
    "warm": -0.6,
    "gold": -0.4,
    "songs": -0.6,
    "kindness": -0.7,
    "normal": -0.4,
    "quiet": -0.3,
    "morning": -0.2,
    "home": -0.4,
    "adequately": -0.3,
    "learned": -0.3,
    "smile": -0.6,
}
_WORD_RE = re.compile(r"[a-z']+")


def lexicon_tension(text: str) -> float:
    toks = _WORD_RE.findall(text.lower())
    if not toks:
        return 0.0
    score = sum(_TENSION_WORDS.get(t, 0.0) + _CALM_WORDS.get(t, 0.0) for t in toks)
    # squash per-1000-words score into (0,1)
    z = score / (len(toks) / 1000 + 1e-9) / 20
    return float(1 / (1 + np.exp(-z)))


def tension_curve(
    segments: Sequence[Segment], signals: Sequence[ContentSignals] | None, bins: int = 12
) -> np.ndarray:
    """Word-position-weighted tension resampled to ``bins`` equal-length windows."""
    if not segments:
        return np.zeros(bins)
    per_seg: list[float] = []
    for i, seg in enumerate(segments):
        llm = signals[i].tension if signals and i < len(signals) else None
        lex = lexicon_tension(seg.text)
        per_seg.append(lex if llm is None else 0.5 * llm + 0.5 * lex)
    words = np.array([max(1, s.word_count) for s in segments], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(words)]) / words.sum()
    curve = np.zeros(bins)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        overlap = np.clip(np.minimum(hi, cum[1:]) - np.maximum(lo, cum[:-1]), 0, None)
        curve[b] = float((overlap * np.array(per_seg)).sum() / max(overlap.sum(), 1e-9))
    return curve


_TEMPLATES: dict[NarrativeArc, np.ndarray] = {
    NarrativeArc.RAGS_TO_RICHES: np.linspace(1, 0, 12),
    NarrativeArc.RICHES_TO_RAGS: np.linspace(0, 1, 12),
    NarrativeArc.MAN_IN_HOLE: np.concatenate([np.linspace(0, 1, 6), np.linspace(1, 0, 6)]),
    NarrativeArc.ICARUS: np.concatenate([np.linspace(1, 0, 6), np.linspace(0, 1, 6)]),
    NarrativeArc.CINDERELLA: np.array([1, 0.6, 0.3, 0.5, 0.9, 1, 0.8, 0.4, 0.2, 0.5, 0.8, 0.1]),
    NarrativeArc.OEDIPUS: np.array([0.2, 0.5, 0.8, 0.6, 0.3, 0.2, 0.5, 0.7, 0.9, 0.7, 0.9, 1.0]),
}


def classify_arc(curve: np.ndarray, flat_threshold: float = 0.05) -> NarrativeArc:
    if curve.size == 0:
        return NarrativeArc.EXPOSITORY
    if curve.std() < flat_threshold:
        return NarrativeArc.EXPOSITORY
    c = (curve - curve.mean()) / (curve.std() + 1e-9)
    best, best_r = NarrativeArc.EPISODIC, -1.0
    for arc, tmpl in _TEMPLATES.items():
        t = np.interp(np.linspace(0, 1, curve.size), np.linspace(0, 1, tmpl.size), tmpl)
        t = (t - t.mean()) / (t.std() + 1e-9)
        r = float(np.corrcoef(c, t)[0, 1])
        if r > best_r:
            best, best_r = arc, r
    return best if best_r >= 0.5 else NarrativeArc.EPISODIC
