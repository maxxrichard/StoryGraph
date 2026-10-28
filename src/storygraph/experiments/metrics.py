"""Metric definitions over exposure-joined interaction logs.

Each metric reduces a listener's rows to one value per listener (the randomisation unit), so
the analysis in :mod:`storygraph.experiments.analysis` can treat listeners as i.i.d.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UnitMetrics:
    listener_id: str
    variant: str
    listen_through: float  # listened / started seconds
    conversion: float  # 1 if any conversion
    finish_rate: float
    sessions: int


def per_listener(rows: Iterable[Mapping[str, Any]]) -> list[UnitMetrics]:
    """Aggregate interaction rows (one per start) to one record per listener."""
    acc: dict[str, dict[str, float]] = {}
    variant: dict[str, str] = {}
    for r in rows:
        lid = str(r["listener_id"])
        variant[lid] = str(r.get("variant", "control"))
        a = acc.setdefault(
            lid, {"started": 0.0, "listened": 0.0, "conv": 0.0, "finished": 0.0, "n": 0.0}
        )
        a["started"] += float(r.get("started_seconds", 0) or 0)
        a["listened"] += float(r.get("listened_seconds", 0) or 0)
        a["conv"] = max(a["conv"], float(bool(r.get("converted"))))
        a["finished"] += float(bool(r.get("finished")))
        a["n"] += 1
    return [
        UnitMetrics(
            listener_id=lid,
            variant=variant[lid],
            listen_through=a["listened"] / a["started"] if a["started"] else 0.0,
            conversion=a["conv"],
            finish_rate=a["finished"] / a["n"] if a["n"] else 0.0,
            sessions=int(a["n"]),
        )
        for lid, a in acc.items()
    ]


def split(units: list[UnitMetrics], metric: str) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {}
    for u in units:
        out.setdefault(u.variant, []).append(float(getattr(u, metric)))
    return {k: np.asarray(v) for k, v in out.items()}
