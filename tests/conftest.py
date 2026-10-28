from __future__ import annotations

from pathlib import Path

import pytest

from storygraph.config import SegmentationConfig
from storygraph.schemas import Title, Transcript
from storygraph.serving.llm_client import StubLLM
from storygraph.understanding.embeddings import HashingEmbedder
from storygraph.understanding.tagger import SegmentTagger

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def small_seg_cfg() -> SegmentationConfig:
    return SegmentationConfig(
        window_sentences=4, min_segment_words=80, max_segment_words=400, smoothing=1
    )


@pytest.fixture
def sample_titles() -> list[Title]:
    import json

    with open(ROOT / "data/sample/catalog.jsonl", encoding="utf-8") as fh:
        return [Title(**json.loads(line)) for line in fh if line.strip()]


@pytest.fixture
def sample_transcripts(sample_titles: list[Title]) -> dict[str, Transcript]:
    return {
        t.title_id: Transcript.from_text(
            t.title_id, (ROOT / t.transcript_path).read_text(encoding="utf-8")
        )
        for t in sample_titles
    }  # type: ignore[arg-type]


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=128)


@pytest.fixture
def tagger() -> SegmentTagger:
    return SegmentTagger(StubLLM())


@pytest.hookimpl(trylast=True)  # run after -m / -k deselection
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """torch and LightGBM each bundle their own OpenMP runtime; importing both into one process
    deadlocks or aborts on macOS. If a session selects both, skip the torch tests and say how to
    run them on their own (``pytest -m torch``). CI and ``make test-ml`` run them separately."""
    has_torch = any(it.get_closest_marker("torch") for it in items)
    has_lgbm = any(it.get_closest_marker("lightgbm") for it in items)
    if has_torch and has_lgbm:
        skip = pytest.mark.skip(reason="torch tests run in their own process: pytest -m torch")
        for it in items:
            if it.get_closest_marker("torch"):
                it.add_marker(skip)
