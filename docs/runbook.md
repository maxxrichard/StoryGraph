# Runbook

## Backfill the catalogue

1. Export the catalogue as JSONL of `Title` records to `s3://.../catalog.jsonl`.
2. Start the Step Functions state machine with `{"catalog_uri": ..., "shard_count": 256}`.
3. Watch the Map state; failed shards retry twice. Re-running the execution skips completed keys.
4. The QualityGate must pass before `storygraph-titles-live` is re-pointed.

## Daily increment

Same state machine, triggered by EventBridge at 02:00 UTC with the delta catalogue.

## Retrain the recommender

```
sagemaker pipeline: storygraph-recsys-training
  BuildRankerDataset -> TrainTwoTower -> TrainRanker -> OfflineEvaluation -> NDCGGate -> RegisterModel
```

Promotion requires >= 5 % NDCG@10 lift over the item-CF baseline on the time split.

## Ship a new tagger adapter

1. `scripts/train_tagger.py --labels <editor labels> --out artifacts/tagger`
2. Evaluate agreement on the held-out editor set (`storygraph.evaluation.agreement`).
3. Upload the adapter, add it to `configs/serving.yaml` `lora_modules`, roll vLLM.
4. Re-run the tagging stage only (`STAGE=tag`) over the catalogue.

## Incident: LLM endpoint down

Recommendations keep serving (no LLM in the hot path). Chapter summaries fall back to cache,
then to 503 after the circuit breaker opens. Check vLLM health, GPU memory, and `max_num_seqs`.

## Incident: SRM in a running experiment

Stop the ramp. Common causes: bucketing salt changed mid-experiment, exposure logging missing
for one variant's error path, bots not excluded. Fix, re-salt, restart.
