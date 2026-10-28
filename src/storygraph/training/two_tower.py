"""Two-tower retrieval model.

* User tower: listener behavioural features (genre/mood affinities, recency stats) -> MLP.
* Title tower: frozen content embedding from the understanding pipeline + categorical signals -> MLP.
* Loss: sampled-softmax over in-batch negatives (optionally mixed with hard negatives drawn from
  the content-kNN neighbourhood of the positive), temperature-scaled dot product.

Both towers export to plain numpy so retrieval can run with OpenSearch kNN and no torch at serve time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from storygraph.config import TwoTowerConfig


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("install storygraph[ml] for the two-tower model") from exc
    return torch


def build_tower(in_dim: int, hidden: list[int], out_dim: int, dropout: float) -> Any:
    torch = _torch()
    nn = torch.nn
    layers: list[Any] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TwoTower:  # not a nn.Module subclass so the module imports without torch
    def __init__(self, user_dim: int, cfg: TwoTowerConfig):
        torch = _torch()
        self.cfg = cfg
        self.user_tower = build_tower(user_dim, cfg.hidden, cfg.embedding_dim, cfg.dropout)
        self.title_tower = build_tower(cfg.content_dim, cfg.hidden, cfg.embedding_dim, cfg.dropout)
        self._torch = torch

    def parameters(self) -> list[Any]:
        return list(self.user_tower.parameters()) + list(self.title_tower.parameters())

    def encode_users(self, x: Any) -> Any:
        return self._torch.nn.functional.normalize(self.user_tower(x), dim=-1)

    def encode_titles(self, x: Any) -> Any:
        return self._torch.nn.functional.normalize(self.title_tower(x), dim=-1)

    def loss(self, user_x: Any, pos_title_x: Any, hard_title_x: Any | None = None) -> Any:
        """Sampled softmax: each user's positive vs all other positives in the batch (+ hard negs)."""
        torch = self._torch
        u = self.encode_users(user_x)
        t = self.encode_titles(pos_title_x)
        logits = u @ t.T / self.cfg.temperature  # [B, B]
        if hard_title_x is not None:
            h = self.encode_titles(hard_title_x.reshape(-1, hard_title_x.shape[-1]))
            h = h.reshape(user_x.shape[0], -1, self.cfg.embedding_dim)  # [B, H, D]
            hard_logits = torch.einsum("bd,bhd->bh", u, h) / self.cfg.temperature
            logits = torch.cat([logits, hard_logits], dim=1)
        labels = torch.arange(user_x.shape[0], device=user_x.device)
        return torch.nn.functional.cross_entropy(logits, labels)

    def train_mode(self, flag: bool = True) -> None:
        self.user_tower.train(flag)
        self.title_tower.train(flag)

    def export(self) -> TwoTowerWeights:
        """Export to numpy for torch-free inference."""

        def dump(seq: Any) -> list[dict[str, np.ndarray]]:
            out = []
            for layer in seq:
                name = type(layer).__name__
                if name == "Linear":
                    out.append(
                        {
                            "kind": "linear",
                            "w": layer.weight.detach().cpu().numpy().T.copy(),
                            "b": layer.bias.detach().cpu().numpy().copy(),
                        }
                    )
                elif name == "LayerNorm":
                    out.append(
                        {
                            "kind": "ln",
                            "g": layer.weight.detach().cpu().numpy().copy(),
                            "b": layer.bias.detach().cpu().numpy().copy(),
                        }
                    )
                elif name == "GELU":
                    out.append({"kind": "gelu"})
            return out

        return TwoTowerWeights(user=dump(self.user_tower), title=dump(self.title_tower))


@dataclass
class TwoTowerWeights:
    """Torch-free forward pass of exported towers (used at serving time)."""

    user: list[dict[str, Any]]
    title: list[dict[str, Any]]

    @staticmethod
    def _forward(layers: list[dict[str, Any]], x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float32)
        for layer in layers:
            if layer["kind"] == "linear":
                h = h @ layer["w"] + layer["b"]
            elif layer["kind"] == "ln":
                mu = h.mean(-1, keepdims=True)
                var = h.var(-1, keepdims=True)
                h = (h - mu) / np.sqrt(var + 1e-5) * layer["g"] + layer["b"]
            elif layer["kind"] == "gelu":
                # exact (erf) GELU to match torch.nn.GELU default
                from math import erf, sqrt

                h = 0.5 * h * (1 + np.vectorize(erf)(h / sqrt(2)))
        n = np.linalg.norm(h, axis=-1, keepdims=True) + 1e-9
        return h / n

    def encode_users(self, x: np.ndarray) -> np.ndarray:
        return self._forward(self.user, x)

    def encode_titles(self, x: np.ndarray) -> np.ndarray:
        return self._forward(self.title, x)

    def save(self, path: str) -> None:
        np.save(
            path,
            np.array({"user": self.user, "title": self.title}, dtype=object),
            allow_pickle=True,
        )

    @classmethod
    def load(cls, path: str) -> TwoTowerWeights:
        d = np.load(path, allow_pickle=True).item()
        return cls(user=d["user"], title=d["title"])


def train_two_tower(
    user_x: np.ndarray,
    title_x: np.ndarray,
    pairs: np.ndarray,
    cfg: TwoTowerConfig,
    hard_negatives: np.ndarray | None = None,
    seed: int = 0,
    log_every: int = 0,
) -> tuple[TwoTower, list[float]]:
    """Train on (user_idx, title_idx) positive pairs. Returns the model and per-epoch losses."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwoTower(user_dim=user_x.shape[1], cfg=cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ux = torch.as_tensor(user_x, dtype=torch.float32)
    tx = torch.as_tensor(title_x, dtype=torch.float32)
    losses: list[float] = []
    model.train_mode(True)
    for epoch in range(cfg.epochs):
        perm = rng.permutation(len(pairs))
        total, steps = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            if len(idx) < 2:
                continue
            u_idx, t_idx = pairs[idx, 0], pairs[idx, 1]
            hard = None
            if hard_negatives is not None and cfg.negatives == "mixed":
                hard = tx[torch.as_tensor(hard_negatives[idx])]
            loss = model.loss(ux[u_idx], tx[t_idx], hard)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        losses.append(total / max(steps, 1))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch + 1}/{cfg.epochs} loss={losses[-1]:.4f}")
    model.train_mode(False)
    return model, losses
