#!/usr/bin/env python
"""Run the content-understanding pipeline over a catalog JSONL (local / dev entrypoint).

python scripts/run_pipeline.py --input data/sample/catalog.jsonl --output artifacts/signals.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from storygraph.config import SegmentationConfig, load_pipeline_config
from storygraph.schemas import Title, Transcript
from storygraph.serving.llm_client import StubLLM, VLLMClient
from storygraph.understanding.embeddings import get_embedder
from storygraph.understanding.pipeline import understand_title
from storygraph.understanding.tagger import SegmentTagger


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--embedder", default="hashing")
    p.add_argument("--llm-endpoint", default=None)
    p.add_argument(
        "--small-segments",
        action="store_true",
        help="use small segment sizes (for short demo transcripts)",
    )
    args = p.parse_args(argv)

    cfg = load_pipeline_config(args.config)
    seg_cfg = (
        SegmentationConfig(
            window_sentences=4, min_segment_words=80, max_segment_words=400, smoothing=1
        )
        if args.small_segments
        else cfg.segmentation
    )
    llm = (
        VLLMClient(base_url=args.llm_endpoint, adapter="tagger") if args.llm_endpoint else StubLLM()
    )
    tagger = SegmentTagger(llm)
    embedder = get_embedder(
        args.embedder, dim=256 if args.embedder == "hashing" else cfg.embeddings.dim
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.input, encoding="utf-8") as fh, open(out, "w", encoding="utf-8") as fo:
        for line in fh:
            if not line.strip():
                continue
            t = Title(**json.loads(line))
            if not t.transcript_path or not Path(t.transcript_path).exists():
                print(f"{t.title_id}: no transcript, skipping", file=sys.stderr)
                continue
            tr = Transcript.from_text(
                t.title_id, Path(t.transcript_path).read_text(encoding="utf-8")
            )
            u = understand_title(t, tr, embedder, tagger, seg_cfg)
            fo.write(json.dumps(u.to_record()) + "\n")
            n += 1
    print(f"wrote {n} title records to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
