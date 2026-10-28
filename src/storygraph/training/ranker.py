"""Learning-to-rank over retrieved candidates.

Production: LightGBM LambdaRank with graded relevance (skip < sample < listen-through < finish).
Fallback: a tiny pairwise-logistic linear ranker in numpy so serving and tests work without LightGBM.
Both expose ``predict(X) -> scores`` and ``feature_names``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from storygraph.config import RankerConfig


class Ranker(Protocol):
    feature_names: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class LinearRanker:
    """Pairwise logistic ranker trained with SGD within query groups. Deterministic, dependency-free."""

    weights: np.ndarray
    feature_names: list[str]
    mean: np.ndarray
    std: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std) @ self.weights

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "weights": self.weights.tolist(),
                    "feature_names": self.feature_names,
                    "mean": self.mean.tolist(),
                    "std": self.std.tolist(),
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearRanker:
        d = json.loads(Path(path).read_text())
        return cls(
            np.array(d["weights"]), d["feature_names"], np.array(d["mean"]), np.array(d["std"])
        )

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        feature_names: list[str],
        epochs: int = 30,
        lr: float = 0.05,
        l2: float = 1e-3,
        seed: int = 0,
    ) -> LinearRanker:
        rng = np.random.default_rng(seed)
        mean, std = X.mean(0), X.std(0) + 1e-6
        Z = (X - mean) / std
        w = np.zeros(X.shape[1])
        offsets = np.concatenate([[0], np.cumsum(group)])
        for _ in range(epochs):
            for g in rng.permutation(len(group)):
                a, b = offsets[g], offsets[g + 1]
                zi, yi = Z[a:b], y[a:b]
                ii, jj = np.where(yi[:, None] > yi[None, :])
                if ii.size == 0:
                    continue
                diff = zi[ii] - zi[jj]
                gain = (yi[ii] - yi[jj])[:, None]
                p = 1 / (1 + np.exp(-(diff @ w)))
                grad = (diff * (p - 1)[:, None] * gain).mean(0) + l2 * w
                w -= lr * grad
        return cls(weights=w, feature_names=feature_names, mean=mean, std=std)


class LightGBMRanker:
    def __init__(self, cfg: RankerConfig, feature_names: list[str]):
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install storygraph[ml] for LightGBM") from exc
        self._lgb = lgb
        self.cfg = cfg
        self.feature_names = feature_names
        self.booster: Any = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        group_val: np.ndarray | None = None,
    ) -> LightGBMRanker:
        lgb = self._lgb
        params = {
            "objective": self.cfg.objective,
            "metric": self.cfg.metric,
            "eval_at": self.cfg.eval_at,
            "num_leaves": self.cfg.num_leaves,
            "learning_rate": self.cfg.learning_rate,
            "min_data_in_leaf": self.cfg.min_data_in_leaf,
            "feature_fraction": self.cfg.feature_fraction,
            "bagging_fraction": self.cfg.bagging_fraction,
            "bagging_freq": self.cfg.bagging_freq,
            "lambda_l2": self.cfg.lambda_l2,
            "label_gain": self.cfg.label_gain,
            "verbosity": -1,
        }
        train = lgb.Dataset(X, label=y, group=group, feature_name=self.feature_names)
        valid_sets, callbacks = [train], []
        if X_val is not None and y_val is not None and group_val is not None:
            valid_sets.append(lgb.Dataset(X_val, label=y_val, group=group_val, reference=train))
            callbacks.append(lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False))
        self.booster = lgb.train(
            params,
            train,
            num_boost_round=self.cfg.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("ranker not fitted")
        return np.asarray(
            self.booster.predict(X, num_iteration=self.booster.best_iteration or None)
        )

    def feature_importance(self) -> dict[str, float]:
        imp = self.booster.feature_importance(importance_type="gain")
        total = float(imp.sum()) or 1.0
        return {
            n: float(v) / total
            for n, v in sorted(zip(self.feature_names, imp), key=lambda kv: -kv[1])
        }

    def save(self, path: str | Path) -> None:
        self.booster.save_model(str(path))

    def load(self, path: str | Path) -> LightGBMRanker:
        self.booster = self._lgb.Booster(model_file=str(path))
        return self


def relevance_grade(
    listened_seconds: float, started_seconds: float, finished: bool, converted: bool
) -> int:
    """Map an interaction to a graded label aligned with ``RankerConfig.label_gain``."""
    if finished or converted:
        return 3
    ratio = listened_seconds / started_seconds if started_seconds > 0 else 0.0
    if ratio >= 0.6:
        return 2
    if ratio >= 0.1:
        return 1
    return 0
