# Architecture

## Goals

* Turn every title in a large long-form audio catalogue into a compact, stable set of signals
  (genre, themes, mood, narrative arc, tension curve, speakers, chapter summaries).
* Use those signals for retrieval, ranking and GenAI features, and prove the value online.
* Keep every stage idempotent, shardable and re-runnable so catalogue-scale backfills and daily
  increments share one code path.

## Batch: content understanding

```
catalog.jsonl ──► PlanShards (Lambda) ──► Distributed Map over shards (Step Functions)
                                            ├── Transcribe        SageMaker Processing, ml.g5.2xlarge, faster-whisper
                                            ├── EmbedAndSegment   SageMaker Processing, ml.g5.xlarge, bge-m3
                                            └── TagWithLLM        SageMaker Processing, ml.g5.12xlarge, vLLM + tagger LoRA
                                          ──► BuildFeatureStore   EMR Serverless Spark -> Delta
                                          ──► IndexOpenSearch     bulk index + kNN fields
                                          ──► QualityGate         tagged share, confidence, empty segments
                                          ──► PromoteAlias        swap the live alias only if the gate passes
```

Idempotency key: `Title.content_key` (title id + asset paths). `PlanShards` drops titles whose key
already has outputs, so re-running the state machine after a partial failure only touches the
unfinished shards.

Throughput levers: `shard_count` (parallelism), `max_concurrency` (Map), chunk length and ASR
batch size (GPU utilisation), vLLM `max_num_seqs` and prefix caching (the taxonomy block is a
shared prefix for every tagging prompt).

## Models

| Model | Purpose | Training data | Serving |
|-------|---------|---------------|---------|
| faster-whisper large-v3 | ASR | – | SageMaker Processing |
| bge-m3 | segment / title embeddings | – | SageMaker Processing |
| Llama-3.1-8B-Instruct + tagger LoRA | per-segment `ContentSignals` JSON | editor-labelled segments (`scripts/train_tagger.py`) | vLLM, guided JSON |
| Llama-3.1-8B-Instruct + summariser LoRA | chapter summaries, explanations | editor-written summaries | vLLM |
| Two-tower | candidate retrieval | listener × title positives from listen-through | exported to numpy; OpenSearch kNN |
| LightGBM LambdaRank | final ranking | exposures joined to graded outcomes | FastAPI |

## Online: serving

```
client ──► FastAPI /v1/recommend ──► bucketing (experiment.yaml)
                                 ──► listener features (feature store or request history)
                                 ──► retrieval: OpenSearch kNN ×2 + popularity
                                 ──► cross features ──► LightGBM ──► top-k + explanations
       ──► FastAPI /v1/titles/{id}/chapters ──► cache ──► vLLM summariser LoRA ──► spoiler guard
```

SLOs in `configs/serving.yaml`: recommendation p95 120 ms, LLM p95 250 ms, 99.9 % availability.
The vLLM client has bounded retries and a circuit breaker so an LLM outage degrades summaries to
cached / empty rather than failing recommendations.

## Data contracts

All stages exchange pydantic models from `storygraph.schemas`. `ContentSignals` doubles as the
JSON schema for guided decoding in the tagger, so the model literally cannot emit an invalid
record; taxonomy snapping handles the remaining label drift.
