#!/usr/bin/env python
"""Offline evaluation of a ranker against an item-CF style popularity baseline on a time split.

Writes report.json with ndcg@10 / recall@50 / mrr for both and the relative lift (consumed by the
SageMaker NDCG gate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from storygraph.evaluation.ranking_metrics import evaluate_rankings, relative_lift
from storygraph.recsys.features import CROSS_FEATURES
from storygraph.training.ranker import LinearRanker


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="artifacts/ranker_dataset.parquet")
    p.add_argument("--model", default="artifacts/models")
    p.add_argument("--baseline", default="item_cf")
    p.add_argument("--cutoff", default=None)
    p.add_argument("--out", default="artifacts/report.json")
    args = p.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    model_dir = Path(args.model)
    if (model_dir / "ranker.lgb.txt").exists():
        from storygraph.config import load_recsys_config
        from storygraph.training.ranker import LightGBMRanker

        ranker = LightGBMRanker(load_recsys_config().ranker, list(CROSS_FEATURES)).load(
            model_dir / "ranker.lgb.txt"
        )
    else:
        ranker = LinearRanker.load(model_dir / "ranker.linear.json")

    X = df[CROSS_FEATURES].to_numpy(dtype=np.float32)
    df["score"] = ranker.predict(X)
    df["baseline_score"] = df["popularity_score"] + 1e-3 * df["content_knn_score"]

    rankings, baseline, relevance = {}, {}, {}
    for lid, g in df.groupby("listener_id"):
        rankings[lid] = g.sort_values("score", ascending=False)["title_id"].tolist()
        baseline[lid] = g.sort_values("baseline_score", ascending=False)["title_id"].tolist()
        relevance[lid] = dict(zip(g["title_id"], g["label"].astype(float)))

    treat = evaluate_rankings(rankings, relevance)
    base = evaluate_rankings(baseline, relevance)
    report = {
        "model": treat,
        "baseline": base,
        "lift": relative_lift(treat, base),
        "baseline_name": args.baseline,
        "cutoff": args.cutoff,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
