import json

import pytest

from storygraph.orchestration.sharding import plan_shards, shard_of
from storygraph.orchestration.step_functions import (
    plan_shards_handler,
    quality_gate,
    render_definition,
)
from storygraph.schemas import Title


def _titles(n=50):
    return [Title(title_id=f"t{i}", title=f"T{i}", duration_min=10) for i in range(n)]


def test_sharding_is_stable_and_skips_done():
    ts = _titles()
    assert shard_of("t1", 8) == shard_of("t1", 8)
    shards = plan_shards(ts, 8)
    assert sum(len(v) for v in shards.values()) == 50 and set(shards) <= set(range(8))
    done = {ts[0].content_key, ts[1].content_key}
    assert sum(len(v) for v in plan_shards(ts, 8, done).values()) == 48


def test_plan_shards_handler_payload():
    out = plan_shards_handler(_titles(10), shard_count=4, artifact_root="s3://b", run_id="r1")
    assert out["skipped"] == 0 and all(
        s["shard_uri"].startswith("s3://b/runs/r1/shards/") for s in out["shards"]
    )


def test_render_definition_substitutes_and_validates(repo_root):
    path = repo_root / "infra/step_functions/content_pipeline.asl.json"
    with pytest.raises(KeyError):
        render_definition(path, {})
    values = {
        k: f"arn:{k}"
        for k in [
            "PlanShardsFunctionArn",
            "PipelineImageUri",
            "SageMakerRoleArn",
            "EmrApplicationId",
            "EmrRoleArn",
            "IndexFunctionArn",
            "QualityGateFunctionArn",
            "PromoteFunctionArn",
        ]
    }
    d = render_definition(path, values)
    assert d["StartAt"] == "PlanShards" and d["States"]["ProcessShards"]["Type"] == "Map"
    json.dumps(d)


def test_quality_gate():
    ok, rep = quality_gate(
        {"tagged_share": 0.99, "mean_confidence": 0.7, "empty_segments_share": 0.0}
    )
    assert ok and all(rep["checks"].values())
    ok, rep = quality_gate(
        {"tagged_share": 0.9, "mean_confidence": 0.7, "empty_segments_share": 0.0}
    )
    assert not ok and not rep["checks"]["tagged_share"]


def test_opensearch_mapping_matches_config(repo_root):
    mapping = json.loads((repo_root / "infra/opensearch/index_mapping.json").read_text())
    props = mapping["mappings"]["properties"]
    assert (
        props["content_vec"]["dimension"] == 1024 and props["title_tower_vec"]["dimension"] == 128
    )
