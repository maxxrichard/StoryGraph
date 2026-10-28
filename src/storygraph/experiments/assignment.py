"""Deterministic experiment assignment.

Listeners are hashed into 10,000 buckets per (experiment, salt); variants own contiguous bucket
ranges so ramping a variant from 1% to 5% keeps every listener already in it. Layers use different
salts so experiments in different layers are orthogonal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

BUCKETS = 10_000


@dataclass(frozen=True)
class Variant:
    name: str
    weight: float
    model: str


@dataclass
class Experiment:
    name: str
    salt: str
    variants: tuple[Variant, ...]
    ramp: float = 1.0  # share of eligible traffic exposed; the rest see control

    def __post_init__(self) -> None:
        total = sum(v.weight for v in self.variants)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"variant weights must sum to 1, got {total}")
        if not 0.0 <= self.ramp <= 1.0:
            raise ValueError("ramp must be in [0,1]")

    @property
    def control(self) -> Variant:
        return next(v for v in self.variants if v.name == "control")

    def bucket(self, unit_id: str) -> int:
        h = hashlib.sha256(f"{self.name}:{self.salt}:{unit_id}".encode()).digest()
        return int.from_bytes(h[:8], "big") % BUCKETS

    def assign(self, unit_id: str) -> Variant:
        """Variant ranges are fixed by weight over all buckets; ``ramp`` exposes the first
        ``ramp`` share of *each* range, so raising the ramp never moves a listener between arms."""
        b = self.bucket(unit_id)
        lo = 0.0
        for v in self.variants:
            hi = lo + v.weight * BUCKETS
            if b < hi:
                return v if (b - lo) < self.ramp * (hi - lo) else self.control
            lo = hi
        return self.control

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], layer: str, ramp: float = 1.0) -> Experiment:
        exp: dict[str, Any] = dict(cfg["experiment"])
        layer_cfg = next(layer_item for layer_item in exp["layers"] if layer_item["name"] == layer)
        variants = tuple(
            Variant(name=n, weight=float(v["weight"]), model=str(v["model"]))
            for n, v in layer_cfg["variants"].items()
        )
        return cls(
            name=f"{exp['name']}/{layer}", salt=str(exp["salt"]), variants=variants, ramp=ramp
        )
