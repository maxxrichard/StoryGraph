#!/usr/bin/env python
"""Build a ranker training set from interaction logs + title signals (local, pandas version).

Mirrors the Spark job ``build_ranker_dataset`` for small data. Output: parquet with one row per
(listener, candidate title) with the cross features and a graded label.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from storygraph.recsys.features import (
    CROSS_FEATURES,
    TitleFeatures,
    cross_features,
    listener_from_history,
)
from storygraph.training.ranker import relevance_grade


def load_titles(signals_path: str) -> dict[str, TitleFeatures]:
    out: dict[str, TitleFeatures] = {}
    with open(signals_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["title_id"]] = TitleFeatures(
                title_id=r["title_id"],
                content_vec=np.asarray(r["content_vec"], dtype=np.float32),
                genres=r["genres"],
                themes=r["themes"],
                moods=r["moods"],
                duration_min=float(r["duration_min"]),
                release_date=date.fromisoformat(r["release_date"])
                if r.get("release_date")
                else None,
                narrative_arc=r.get("narrative_arc"),
                speaker_count=int(r.get("speaker_count", 1)),
            )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interactions", default="data/sample/interactions.csv")
    p.add_argument("--signals", default="artifacts/signals.jsonl")
    p.add_argument("--cutoff", default="2026-09-06")
    p.add_argument("--out", default="artifacts/ranker_dataset.parquet")
    args = p.parse_args(argv)

    titles = load_titles(args.signals)
    df = pd.read_csv(args.interactions, parse_dates=["ts"])
    cutoff = datetime.fromisoformat(args.cutoff)
    df["date"] = df["ts"].dt.date
    past, future = (
        df[df["ts"].dt.tz_localize(None) < cutoff],
        df[df["ts"].dt.tz_localize(None) >= cutoff],
    )
    rows = []
    for lid, fut in future.groupby("listener_id"):
        hist = past[past["listener_id"] == lid].to_dict("records")
        listener = listener_from_history(str(lid), hist, titles, as_of=cutoff.date())
        outcomes = {r["title_id"]: r for r in fut.to_dict("records")}
        for tid, t in titles.items():
            if tid in listener.history:
                continue
            o = outcomes.get(tid)
            label = (
                relevance_grade(
                    o["listened_seconds"],
                    o["started_seconds"],
                    bool(o["finished"]),
                    bool(o["converted"]),
                )
                if o
                else 0
            )
            feats = cross_features(listener, t, 0.0, 0.0, 0.0, 1, as_of=cutoff.date())
            rows.append(
                {
                    "listener_id": lid,
                    "title_id": tid,
                    "label": label,
                    **dict(zip(CROSS_FEATURES, feats.tolist())),
                }
            )
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(
        f"wrote {len(out)} rows for {out['listener_id'].nunique() if len(out) else 0} listeners to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
