"""Ranking metrics over graded or binary relevance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    g = np.asarray(gains[:k], dtype=float)
    if g.size == 0:
        return 0.0
    return float((g / np.log2(np.arange(2, g.size + 2))).sum())


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int = 10) -> float:
    """NDCG@k with gains taken directly from ``relevance`` (0 for unseen ids)."""
    gains = [float(relevance.get(i, 0.0)) for i in ranked_ids]
    ideal = sorted((float(v) for v in relevance.values()), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int = 50) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mrr(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    for i, tid in enumerate(ranked_ids):
        if tid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    relevance: Mapping[str, Mapping[str, float]],
    ks: Sequence[int] = (10,),
    recall_k: int = 50,
) -> dict[str, float]:
    """Average metrics over listeners. ``rankings`` and ``relevance`` are keyed by listener."""
    out: dict[str, list[float]] = {f"ndcg@{k}": [] for k in ks}
    out[f"recall@{recall_k}"] = []
    out["mrr"] = []
    for lid, ranked in rankings.items():
        rel = relevance.get(lid, {})
        if not rel:
            continue
        relevant = {t for t, g in rel.items() if g > 0}
        for k in ks:
            out[f"ndcg@{k}"].append(ndcg_at_k(ranked, rel, k))
        out[f"recall@{recall_k}"].append(recall_at_k(ranked, relevant, recall_k))
        out["mrr"].append(mrr(ranked, relevant))
    return {m: float(np.mean(v)) if v else 0.0 for m, v in out.items()}


def relative_lift(
    treatment: Mapping[str, float], baseline: Mapping[str, float]
) -> dict[str, float]:
    return {
        m: (treatment[m] - baseline[m]) / baseline[m] if baseline.get(m) else float("nan")
        for m in treatment
    }
