#!/usr/bin/env python
"""Train the two-tower retrieval model and / or the LightGBM ranker.

python scripts/train_recsys.py --stage two_tower --dataset artifacts/ranker_dataset.parquet --out artifacts/models
python scripts/train_recsys.py --stage ranker    --dataset artifacts/ranker_dataset.parquet --out artifacts/models
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from storygraph.config import load_recsys_config
from storygraph.recsys.features import CROSS_FEATURES


def train_ranker(df: pd.DataFrame, out: Path, cfg) -> None:  # type: ignore[no-untyped-def]
    df = df.sort_values("listener_id")
    X = df[CROSS_FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    group = df.groupby("listener_id", sort=False).size().to_numpy()
    ranker_cfg = cfg.ranker
    if len(df) < 1000:  # tiny / demo datasets: production leaf sizes would yield an empty model
        ranker_cfg = ranker_cfg.model_copy(
            update={"min_data_in_leaf": 1, "num_leaves": 7, "n_estimators": 50}
        )
    try:
        from storygraph.training.ranker import LightGBMRanker

        model = LightGBMRanker(ranker_cfg, list(CROSS_FEATURES)).fit(X, y, group)
        model.save(out / "ranker.lgb.txt")
        print("LightGBM feature importance:", model.feature_importance())
    except ImportError:
        from storygraph.training.ranker import LinearRanker

        LinearRanker.fit(X, y, group, list(CROSS_FEATURES)).save(out / "ranker.linear.json")
        print("LightGBM not installed; trained LinearRanker fallback")


def train_towers(df: pd.DataFrame, out: Path, cfg, signals: str) -> None:  # type: ignore[no-untyped-def]
    import json

    from storygraph.training.two_tower import train_two_tower

    content = {}
    with open(signals, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            content[r["title_id"]] = np.asarray(r["content_vec"], dtype=np.float32)
    title_ids = sorted(content)
    tidx = {t: i for i, t in enumerate(title_ids)}
    title_x = np.stack([content[t] for t in title_ids])
    listeners = sorted(df["listener_id"].unique())
    lidx = {lid: i for i, lid in enumerate(listeners)}
    user_cols = ["listener_finish_rate_90d", "listener_listen_hours_30d"]
    user_x = df.groupby("listener_id")[user_cols].first().loc[listeners].to_numpy(dtype=np.float32)
    pos = df[df["label"] >= 2]
    pairs = np.array(
        [[lidx[lid], tidx[t]] for lid, t in zip(pos["listener_id"], pos["title_id"]) if t in tidx]
    )
    if len(pairs) < 2:
        print("not enough positive pairs to train towers")
        return
    tt_cfg = cfg.retrieval.two_tower.model_copy(
        update={"content_dim": title_x.shape[1], "batch_size": min(256, len(pairs))}
    )
    model, losses = train_two_tower(user_x, title_x, pairs, tt_cfg, log_every=1)
    model.export().save(str(out / "two_tower.npy"))
    print(f"final loss {losses[-1]:.4f}; towers saved to {out / 'two_tower.npy'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["two_tower", "ranker", "both"], default="both")
    p.add_argument("--dataset", default="artifacts/ranker_dataset.parquet")
    p.add_argument("--signals", default="artifacts/signals.jsonl")
    p.add_argument("--config", default="configs/recsys.yaml")
    p.add_argument("--out", default="artifacts/models")
    p.add_argument("--version", default="two_tower_lgbm_v3")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "both":
        # torch and LightGBM ship separate OpenMP runtimes; run each stage in its own process.
        for stage in ("two_tower", "ranker"):
            cmd = [
                sys.executable,
                __file__,
                "--stage",
                stage,
                "--dataset",
                args.dataset,
                "--signals",
                args.signals,
                "--config",
                args.config,
                "--out",
                args.out,
            ]
            subprocess.run(cmd, check=True)
    else:
        cfg = load_recsys_config(args.config)
        df = pd.read_parquet(args.dataset)
        if args.stage == "two_tower":
            train_towers(df, out, cfg, args.signals)
        else:
            train_ranker(df, out, cfg)
    (out / "VERSION").write_text(args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
