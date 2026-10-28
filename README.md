# StoryGraph

**LLM content understanding and recommendation for long-form audio.**

StoryGraph turns audiobooks, podcasts and transcripts into structured signals (genre, themes, mood,
narrative structure, speakers) and uses them for retrieval, personalised recommendations and GenAI
features such as spoiler-free chapter summaries and explained picks. Everything in this repository
is built to be validated in production experiments: the recommender ships behind a deterministic
A/B framework with CUPED, SRM checks and guardrails.

```
audio / transcripts ──► ASR ──► segmentation ──► embeddings ──► LoRA-tuned LLM tagging ──► title signals
                                                                                             │
                     ┌───────────────────────────────────────────────────────────────────────┘
                     ▼
     feature store (Spark) + OpenSearch kNN ──► two-tower retrieval ──► LightGBM ranker ──► explained picks
                                                                                             │
                                                           vLLM (tagger / summariser LoRA) ──► chapter summaries
```

## What is in the box

| Area | Module | Highlights |
|------|--------|-----------|
| Ingest | `storygraph.ingest` | silence-snapped audio chunking, faster-whisper ASR, overlap stitching |
| Understanding | `storygraph.understanding` | TextTiling / embedding-breakpoint segmentation, bge-m3 embeddings, LLM tagging with guided JSON decoding, taxonomy snapping, speaker roles, tension curve + narrative-arc classification |
| Training | `storygraph.training` | LoRA SFT for the tagger (PEFT), two-tower retrieval (PyTorch, in-batch + hard negatives, numpy export), LightGBM LambdaRank ranker with a dependency-free fallback |
| Recommender | `storygraph.recsys` | three-source candidate retrieval (two-tower, content kNN, popularity), cross features, ranking, template + LLM explanations |
| GenAI | `storygraph.genai` | cached chapter summariser with spoiler guard, LLM-as-judge with human calibration |
| Serving | `storygraph.serving` | vLLM OpenAI-compatible client (LoRA selection, retries, circuit breaker), FastAPI API, request-time A/B bucketing |
| Experiments | `storygraph.experiments` | hashed assignment with ramp-safe bucketing, listener-level metrics, Welch + CUPED, SRM, power |
| Evaluation | `storygraph.evaluation` | NDCG@k / recall / MRR, multi-label editor agreement (macro-F1, kappa) |
| Orchestration | `storygraph.orchestration` + `infra/` | Step Functions distributed map over catalog shards, SageMaker Pipeline with an NDCG gate, Spark feature-store jobs, OpenSearch index mapping |

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"            # add [ml] for torch + LightGBM, [llm] for PEFT / sentence-transformers
make test
```

Run the understanding pipeline over the bundled sample catalog (stub LLM, hashing embedder, no GPU):

```bash
storygraph understand data/sample/catalog.jsonl --output artifacts/signals.jsonl
```

Serve the API with a demo context built from the sample catalog, then ask for recommendations:

```bash
storygraph serve --port 8080
```

```bash
curl -s localhost:8080/v1/recommend -H 'content-type: application/json' \
  -d '{"listener_id":"u-1","history":[{"title_id":"t-0001","started_seconds":3600,"listened_seconds":3300}],"k":5}' | jq
```

Point the pipeline at a real vLLM endpoint and embedding model:

```bash
python scripts/run_pipeline.py --input catalog.jsonl --output signals.jsonl \
  --embedder BAAI/bge-m3 --llm-endpoint http://vllm.internal:8000/v1
```

## Pipeline

1. **ASR.** Audio is resampled to 16 kHz, cut into 10-minute chunks with 5 s overlap snapped to
   silence, transcribed with faster-whisper (word timestamps, VAD), and stitched back together.
2. **Segmentation.** Transcripts are split into chapter-like segments with TextTiling (lexical
   cohesion) or embedding breakpoints, with min/max word bounds enforced.
3. **Embeddings.** Each segment gets a bge-m3 vector; the title vector is a length-weighted mean.
4. **Tagging.** A LoRA-tuned instruction model emits one JSON object per segment, validated
   against `ContentSignals` and snapped to the controlled taxonomy. Segment labels are aggregated
   into title labels (weighted vote for genre, share threshold for themes, trajectory for mood).
5. **Narrative + speakers.** A tension curve (LLM + lexicon) is resampled into 12 bins and matched
   against arc templates; diarized speakers become profiles with heuristic roles.
6. **Sinks.** Records land in a Spark/Delta feature store and an OpenSearch index with two kNN
   fields (content vector, title-tower vector).

## Recommender

* **Retrieval.** Union of two-tower kNN, content kNN from the listener's recent finishes, and a
  popularity floor. Each candidate keeps its per-source scores as features.
* **Ranking.** LightGBM LambdaRank on graded relevance (skip < sample < listen-through < finish),
  with cross features such as genre / mood affinity match, theme overlap and narrator familiarity.
  Offline evaluation is a time split against the collaborative-filtering baseline, gated on
  NDCG@10 lift in the SageMaker pipeline.
* **Explanations.** Template explanations from the same features, or a one-sentence LLM reason
  grounded only in the provided signals.

## Experimentation

`configs/experiment.yaml` defines layers, variants, metrics and ramp. Assignment hashes
`(experiment, salt, listener)` into 10k buckets so ramping keeps listeners stable. Analysis runs
listener-level Welch tests with CUPED (pre-period covariate), an SRM check, non-inferiority
guardrails and a ship / hold verdict:

```bash
python scripts/analyze_ab.py --exposures exports/recsys_v3.csv --config configs/experiment.yaml
```

## Repository layout

```
configs/        pipeline, tagging, recsys, experiment, serving YAML
docs/           architecture, taxonomy, experiment design, runbook
infra/          Step Functions ASL, OpenSearch mapping, Dockerfiles
scripts/        pipeline, training, evaluation and analysis entrypoints
src/storygraph/ library code (see table above)
tests/          unit + integration tests (ML-dependent tests are marked `ml`)
```

## Development

```bash
make lint typecheck test      # ruff, mypy, pytest (no torch / LightGBM needed)
make test-ml                  # adds LightGBM and two-tower tests (run in separate processes: both bundle an OpenMP runtime)
```

CI runs lint, type checks and the test matrix on Python 3.10–3.12.

## License

MIT
