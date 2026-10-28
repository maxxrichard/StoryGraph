"""Entrypoint for SageMaker Processing jobs: ``python -m storygraph.orchestration.jobs <stage>``.

Each stage reads a shard manifest (JSONL of Title records) from ``SHARD_URI``, processes every
title and writes one JSONL of outputs next to it. Stages are idempotent per content_key.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from storygraph.config import load_pipeline_config, load_tagging_config
from storygraph.schemas import Title, Transcript

log = logging.getLogger("storygraph.jobs")


def _read_titles(path: str) -> list[Title]:
    with open(path, encoding="utf-8") as fh:
        return [Title(**json.loads(line)) for line in fh if line.strip()]


def run_asr(titles: list[Title], out: Path, cfg: dict) -> None:  # pragma: no cover - GPU job
    from storygraph.ingest.asr import FasterWhisperBackend, transcribe_chunks
    from storygraph.ingest.audio import chunk_audio, load_audio

    asr_cfg, audio_cfg = cfg["asr"], cfg["audio"]
    backend = FasterWhisperBackend(asr_cfg["model"], asr_cfg["compute_type"], asr_cfg["beam_size"])
    with open(out, "w", encoding="utf-8") as fh:
        for t in titles:
            if not t.audio_path:
                continue
            y, sr = load_audio(t.audio_path, audio_cfg["target_sample_rate"])
            chunks = chunk_audio(y, sr, audio_cfg["chunk_seconds"], audio_cfg["overlap_seconds"])
            tr = transcribe_chunks(
                t.title_id, chunks, sr, backend, overlap_seconds=audio_cfg["overlap_seconds"]
            )
            fh.write(tr.model_dump_json() + "\n")


def run_understand(
    titles: list[Title],
    transcripts: dict[str, Transcript],
    out: Path,
    embedder_name: str,
    llm_endpoint: str | None,
) -> None:
    from storygraph.serving.llm_client import StubLLM, VLLMClient
    from storygraph.understanding.embeddings import get_embedder
    from storygraph.understanding.pipeline import understand_title
    from storygraph.understanding.tagger import SegmentTagger

    tag_cfg = load_tagging_config()
    llm = (
        VLLMClient(base_url=llm_endpoint, model=tag_cfg.model.base, adapter="tagger")
        if llm_endpoint
        else StubLLM()
    )
    tagger = SegmentTagger(llm)
    embedder = get_embedder(embedder_name)
    pipe_cfg = load_pipeline_config()
    with open(out, "w", encoding="utf-8") as fh:
        for t in titles:
            tr = transcripts.get(t.title_id)
            if tr is None:
                log.warning("%s: no transcript, skipping", t.title_id)
                continue
            result = understand_title(t, tr, embedder, tagger, pipe_cfg.segmentation)
            fh.write(json.dumps(result.to_record()) + "\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["asr", "embed", "tag"])
    p.add_argument("--shard", default=os.environ.get("SHARD_URI"))
    p.add_argument("--llm-endpoint", default=os.environ.get("LLM_ENDPOINT"))
    p.add_argument("--embedder", default=os.environ.get("EMBEDDER", "hashing"))
    args = p.parse_args(argv)
    if not args.shard:
        p.error("--shard or SHARD_URI is required")
    shard = Path(args.shard)
    titles = _read_titles(str(shard))
    cfg = load_pipeline_config().ingest
    if args.stage == "asr":
        run_asr(titles, shard.with_suffix(".transcripts.jsonl"), cfg)
    else:
        transcripts: dict[str, Transcript] = {}
        tpath = shard.with_suffix(".transcripts.jsonl")
        if tpath.exists():
            with open(tpath, encoding="utf-8") as fh:
                for line in fh:
                    tr = Transcript.model_validate_json(line)
                    transcripts[tr.title_id] = tr
        for t in titles:
            if (
                t.title_id not in transcripts
                and t.transcript_path
                and Path(t.transcript_path).exists()
            ):
                transcripts[t.title_id] = Transcript.from_text(
                    t.title_id, Path(t.transcript_path).read_text(encoding="utf-8")
                )
        run_understand(
            titles,
            transcripts,
            shard.with_suffix(".signals.jsonl"),
            args.embedder,
            args.llm_endpoint,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
