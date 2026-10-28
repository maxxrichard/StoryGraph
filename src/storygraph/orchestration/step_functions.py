"""Helpers for the Step Functions content pipeline (infra/step_functions/content_pipeline.asl.json).

* :func:`render_definition` substitutes ``${Placeholders}`` with deployed ARNs.
* :func:`plan_shards_handler` is the Lambda behind the PlanShards state.
* :func:`quality_gate` decides whether a run's signals are good enough to promote.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from storygraph.orchestration.sharding import plan_shards
from storygraph.schemas import Title

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


def render_definition(path: str | Path, values: Mapping[str, str]) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    missing = sorted(set(_PLACEHOLDER.findall(text)) - set(values))
    if missing:
        raise KeyError(f"unresolved placeholders: {missing}")
    return json.loads(_PLACEHOLDER.sub(lambda m: values[m.group(1)], text))


def plan_shards_handler(
    titles: Sequence[Title],
    shard_count: int,
    artifact_root: str,
    run_id: str,
    done_keys: set[str] | None = None,
) -> dict[str, Any]:
    shards = plan_shards(titles, shard_count, done_keys)
    return {
        "shards": [
            {
                "shard_id": sid,
                "shard_uri": f"{artifact_root}/runs/{run_id}/shards/{sid:04d}.jsonl",
                "n_titles": len(ts),
            }
            for sid, ts in sorted(shards.items())
        ],
        "skipped": sum(1 for _ in titles) - sum(len(ts) for ts in shards.values()),
    }


def quality_gate(
    stats: Mapping[str, float],
    *,
    min_tagged_share: float = 0.98,
    min_mean_confidence: float = 0.6,
    max_empty_segments_share: float = 0.01,
) -> tuple[bool, dict[str, Any]]:
    checks = {
        "tagged_share": stats.get("tagged_share", 0.0) >= min_tagged_share,
        "mean_confidence": stats.get("mean_confidence", 0.0) >= min_mean_confidence,
        "empty_segments_share": stats.get("empty_segments_share", 1.0) <= max_empty_segments_share,
    }
    return all(checks.values()), {"checks": checks, "stats": dict(stats)}
