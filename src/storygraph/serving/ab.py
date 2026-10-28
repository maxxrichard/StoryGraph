"""Request-time A/B bucketing for the API: maps a listener to a variant and the model it serves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from storygraph.config import load_yaml
from storygraph.experiments.assignment import Experiment, Variant


@dataclass
class Bucketing:
    experiment: Experiment | None

    @classmethod
    def from_file(
        cls, path: str | Path | None, layer: str = "homepage_recs", ramp: float = 1.0
    ) -> Bucketing:
        if path is None or not Path(path).exists():
            return cls(experiment=None)
        return cls(experiment=Experiment.from_config(load_yaml(path), layer=layer, ramp=ramp))

    def variant_for(self, listener_id: str, override: str | None = None) -> Variant:
        if self.experiment is None:
            return Variant(name="treatment", weight=1.0, model="two_tower_lgbm_v3")
        if override:
            for v in self.experiment.variants:
                if v.name == override:
                    return v
        return self.experiment.assign(listener_id)
