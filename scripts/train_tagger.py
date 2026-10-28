#!/usr/bin/env python
"""Build the SFT dataset from editor-labelled segments and LoRA-tune the tagging model.

python scripts/train_tagger.py --labels data/labels/editor_segments.jsonl --out artifacts/tagger
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from storygraph.config import load_tagging_config
from storygraph.training.lora import iter_labelled, train_lora, write_sft_dataset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/tagging.yaml")
    p.add_argument("--dataset-only", action="store_true")
    args = p.parse_args(argv)

    cfg = load_tagging_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sft = out / "sft.jsonl"
    n = write_sft_dataset(iter_labelled(args.labels), sft)
    print(f"wrote {n} SFT examples to {sft}")
    if args.dataset_only:
        return 0
    adapter = train_lora(sft, cfg.model, cfg.lora, out, max_seq_len=cfg.model.max_input_tokens)
    print(f"adapter saved to {adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
