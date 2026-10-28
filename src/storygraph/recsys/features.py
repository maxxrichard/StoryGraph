"""Feature engineering for the ranker.

The Spark job in :mod:`storygraph.orchestration.spark_jobs` materialises listener and title
features into the feature store; this module holds the *schema* and the pure-Python/numpy
versions used at serving time and in tests so both paths compute identical features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from storygraph.understanding.taxonomy import GENRES, MOODS

GENRE_INDEX = {g: i for i, g in enumerate(GENRES)}
MOOD_INDEX = {m: i for i, m in enumerate(MOODS)}

CROSS_FEATURES = [
    "two_tower_score",
    "content_knn_score",
    "popularity_score",
    "genre_match",
    "theme_overlap",
    "mood_match",
    "same_narrator",
    "duration_min",
    "release_age_days",
    "listener_finish_rate_90d",
    "listener_listen_hours_30d",
    "n_sources",
]


@dataclass
class ListenerFeatures:
    listener_id: str
    listen_hours_30d: float = 0.0
    finish_rate_90d: float = 0.0
    genre_affinity: np.ndarray = field(
        default_factory=lambda: np.zeros(len(GENRES), dtype=np.float32)
    )
    mood_affinity: np.ndarray = field(
        default_factory=lambda: np.zeros(len(MOODS), dtype=np.float32)
    )
    recent_narrators: set[str] = field(default_factory=set)
    recent_themes: set[str] = field(default_factory=set)
    history: list[str] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return 2 + len(GENRES) + len(MOODS)

    def vector(self) -> np.ndarray:
        """Input to the user tower."""
        return np.concatenate(
            [
                np.array([np.log1p(self.listen_hours_30d), self.finish_rate_90d], dtype=np.float32),
                self.genre_affinity,
                self.mood_affinity,
            ]
        )


@dataclass
class TitleFeatures:
    title_id: str
    content_vec: np.ndarray
    genres: list[str]
    themes: list[str]
    moods: list[str]
    duration_min: float
    narrator: str | None = None
    release_date: date | None = None
    popularity_7d: float = 0.0
    narrative_arc: str | None = None
    speaker_count: int = 1


def listener_from_history(
    listener_id: str,
    history: Sequence[Mapping[str, Any]],
    titles: Mapping[str, TitleFeatures],
    half_life_days: float = 30.0,
    as_of: date | None = None,
) -> ListenerFeatures:
    """Build listener features from interaction rows with exponential recency decay."""
    lf = ListenerFeatures(listener_id=listener_id)
    as_of = as_of or date.today()
    hours, finishes, starts = 0.0, 0.0, 0
    for row in history:
        tid = str(row["title_id"])
        t = titles.get(tid)
        when = row.get("date")
        age = (as_of - when).days if isinstance(when, date) else 0
        w = 0.5 ** (max(age, 0) / half_life_days)
        listened = float(row.get("listened_seconds", 0.0) or 0.0)
        hours += listened / 3600 * (1.0 if age <= 30 else 0.0)
        if age <= 90:
            starts += 1
            finishes += 1.0 if row.get("finished") else 0.0
        lf.history.append(tid)
        if t is None:
            continue
        engagement = min(1.0, listened / max(1.0, float(row.get("started_seconds", 1.0) or 1.0)))
        for g in t.genres:
            if g in GENRE_INDEX:
                lf.genre_affinity[GENRE_INDEX[g]] += w * engagement
        for m in t.moods:
            if m in MOOD_INDEX:
                lf.mood_affinity[MOOD_INDEX[m]] += w * engagement
        if t.narrator:
            lf.recent_narrators.add(t.narrator)
        lf.recent_themes.update(t.themes)
    lf.listen_hours_30d = hours
    lf.finish_rate_90d = finishes / starts if starts else 0.0
    for vec in (lf.genre_affinity, lf.mood_affinity):
        s = vec.sum()
        if s > 0:
            vec /= s
    return lf


def cross_features(
    listener: ListenerFeatures,
    title: TitleFeatures,
    two_tower_score: float,
    content_knn_score: float,
    popularity_score: float,
    n_sources: int,
    as_of: date | None = None,
) -> np.ndarray:
    as_of = as_of or date.today()
    genre_match = float(
        sum(listener.genre_affinity[GENRE_INDEX[g]] for g in title.genres if g in GENRE_INDEX)
    )
    mood_match = float(
        sum(listener.mood_affinity[MOOD_INDEX[m]] for m in title.moods if m in MOOD_INDEX)
    )
    theme_overlap = len(set(title.themes) & listener.recent_themes) / max(1, len(title.themes))
    age = (as_of - title.release_date).days if title.release_date else 365
    return np.array(
        [
            two_tower_score,
            content_knn_score,
            popularity_score,
            genre_match,
            theme_overlap,
            mood_match,
            1.0 if title.narrator and title.narrator in listener.recent_narrators else 0.0,
            np.log1p(title.duration_min),
            np.log1p(max(age, 0)),
            listener.finish_rate_90d,
            np.log1p(listener.listen_hours_30d),
            float(n_sources),
        ],
        dtype=np.float32,
    )
