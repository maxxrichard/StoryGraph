"""Typed configuration loading.

YAML files under ``configs/`` are the single source of truth for every stage. They are loaded
into pydantic models here so a typo in a config fails at start-up rather than three hours into
a SageMaker job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SegmentationConfig(BaseModel):
    method: str = "texttiling"
    window_sentences: int = 20
    min_segment_words: int = 300
    max_segment_words: int = 2500
    smoothing: int = 3


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    dim: int = 1024
    normalize: bool = True
    pooling: str = "mean"
    batch_size: int = 64


class AggregationConfig(BaseModel):
    genre: str = "weighted_vote"
    themes: str = "frequency_threshold"
    themes_min_share: float = 0.15
    mood: str = "trajectory"
    confidence_floor: float = 0.55


class TaggingModelConfig(BaseModel):
    base: str
    adapter: str | None = None
    max_input_tokens: int = 6144
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    json_schema_guided: bool = True


class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    per_device_batch_size: int = 4
    gradient_accumulation: int = 8
    bf16: bool = True


class TaggingConfig(BaseModel):
    model: TaggingModelConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    taxonomy: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)


class TwoTowerConfig(BaseModel):
    embedding_dim: int = 128
    content_dim: int = 1024
    hidden: list[int] = Field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    temperature: float = 0.05
    negatives: str = "in_batch"
    hard_negatives_per_positive: int = 4
    epochs: int = 10
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class CandidateConfig(BaseModel):
    k_two_tower: int = 300
    k_content_knn: int = 150
    k_popular: int = 50
    k_total_max: int = 500


class RankerConfig(BaseModel):
    objective: str = "lambdarank"
    metric: str = "ndcg"
    eval_at: list[int] = Field(default_factory=lambda: [10])
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 800
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    early_stopping_rounds: int = 50
    label_gain: list[int] = Field(default_factory=lambda: [0, 1, 3, 7])


class RetrievalConfig(BaseModel):
    two_tower: TwoTowerConfig = Field(default_factory=TwoTowerConfig)
    candidates: CandidateConfig = Field(default_factory=CandidateConfig)
    opensearch: dict[str, Any] = Field(default_factory=dict)


class RecsysConfig(BaseModel):
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ranker: RankerConfig = Field(default_factory=RankerConfig)
    features: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    pipeline: dict[str, Any] = Field(default_factory=dict)
    ingest: dict[str, Any] = Field(default_factory=dict)
    understanding: dict[str, Any] = Field(default_factory=dict)
    sinks: dict[str, Any] = Field(default_factory=dict)

    @property
    def segmentation(self) -> SegmentationConfig:
        return SegmentationConfig(**self.understanding.get("segmentation", {}))

    @property
    def embeddings(self) -> EmbeddingConfig:
        return EmbeddingConfig(**self.understanding.get("embeddings", {}))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def load_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> PipelineConfig:
    return PipelineConfig(**load_yaml(path))


def load_tagging_config(path: str | Path = "configs/tagging.yaml") -> TaggingConfig:
    return TaggingConfig(**load_yaml(path))


def load_recsys_config(path: str | Path = "configs/recsys.yaml") -> RecsysConfig:
    return RecsysConfig(**load_yaml(path))
