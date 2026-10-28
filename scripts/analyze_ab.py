#!/usr/bin/env python
"""Analyse an A/B export: SRM check, CUPED-adjusted Welch tests on primary + guardrail metrics.

python scripts/analyze_ab.py --exposures exports/recsys_v3.csv --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from storygraph.config import load_yaml
from storygraph.experiments.analysis import guardrail_ok, srm_check, welch_ttest
from storygraph.experiments.metrics import per_listener, split

METRIC_MAP = {
    "listen_through_rate": "listen_through",
    "conversion_rate": "conversion",
    "finish_rate": "finish_rate",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exposures",
        required=True,
        help="CSV: listener_id, variant, started_seconds, listened_seconds, finished, converted[, pre_*]",
    )
    p.add_argument("--config", default="configs/experiment.yaml")
    args = p.parse_args(argv)

    cfg = load_yaml(args.config)
    df = pd.read_csv(args.exposures)
    units = per_listener(df.to_dict("records"))
    weights = {n: v["weight"] for n, v in cfg["experiment"]["layers"][0]["variants"].items()}
    counts = {v: sum(1 for u in units if u.variant == v) for v in weights}
    ok, pval = srm_check(counts, weights, alpha=cfg["analysis"]["srm_alpha"])
    print(f"SRM: {'ok' if ok else 'MISMATCH - results invalid'} p={pval:.4f} counts={counts}")

    pre = {}
    if "pre_listen_through" in df.columns:
        pre = df.groupby("listener_id")["pre_listen_through"].first().to_dict()

    verdict = ok
    for section in ("primary", "guardrails", "secondary"):
        for m in cfg["metrics"].get(section, []):
            name = METRIC_MAP.get(m["name"])
            if name is None:
                continue
            arms = split(units, name)
            if "control" not in arms or "treatment" not in arms:
                continue
            kwargs = {}
            if pre and name == "listen_through":
                c_ids = [u.listener_id for u in units if u.variant == "control"]
                t_ids = [u.listener_id for u in units if u.variant == "treatment"]
                kwargs = {
                    "control_pre": [pre.get(i, 0.0) for i in c_ids],
                    "treatment_pre": [pre.get(i, 0.0) for i in t_ids],
                }
            res = welch_ttest(
                arms["control"],
                arms["treatment"],
                metric=m["name"],
                alpha=cfg["analysis"]["alpha"],
                **kwargs,
            )
            line = res.summary()
            if section == "guardrails":
                passed = guardrail_ok(res, m["direction"], m["margin"])
                verdict &= passed
                line += f"  guardrail {'PASS' if passed else 'FAIL'}"
            elif section == "primary":
                verdict &= res.significant and res.abs_lift > 0
            print(f"[{section}] {line}")
    print("VERDICT:", "ship" if verdict else "hold")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
