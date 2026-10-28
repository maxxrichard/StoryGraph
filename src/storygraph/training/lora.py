"""LoRA fine-tuning of the tagging LLM on editor-labelled segments.

Training data: JSONL rows ``{"segment": Segment, "signals": ContentSignals}`` produced by the
editorial labelling tool. Each row becomes a chat example whose assistant turn is the canonical
JSON of ``ContentSignals``. We train with PEFT + transformers; vLLM then serves the adapter.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from storygraph.config import LoraConfig, TaggingModelConfig
from storygraph.schemas import ContentSignals, Segment
from storygraph.understanding.tagger import SYSTEM_PROMPT, build_prompt
from storygraph.understanding.taxonomy import DEFAULT_TAXONOMY, Taxonomy


def to_chat_example(
    segment: Segment, signals: ContentSignals, taxonomy: Taxonomy = DEFAULT_TAXONOMY
) -> dict[str, Any]:
    """Turn a labelled segment into a chat-format training example."""
    target = signals.model_dump(mode="json", exclude_none=True)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(segment, taxonomy)},
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def iter_labelled(path: str | Path) -> Iterator[tuple[Segment, ContentSignals]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            yield Segment(**row["segment"]), ContentSignals(**row["signals"])


def write_sft_dataset(rows: Iterable[tuple[Segment, ContentSignals]], out: str | Path) -> int:
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for seg, sig in rows:
            fh.write(json.dumps(to_chat_example(seg, sig), ensure_ascii=False) + "\n")
            n += 1
    return n


def train_lora(
    sft_path: str | Path,
    model_cfg: TaggingModelConfig,
    lora_cfg: LoraConfig,
    output_dir: str | Path,
    max_seq_len: int = 6144,
) -> str:  # pragma: no cover - needs GPU + model weights
    """Run supervised fine-tuning with a LoRA adapter and return the adapter directory."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("install storygraph[ml,llm] to train the tagger") from exc

    tok = AutoTokenizer.from_pretrained(model_cfg.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.base, torch_dtype=torch.bfloat16 if lora_cfg.bf16 else torch.float32
    )
    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            task_type="CAUSAL_LM",
        ),
    )

    ds = load_dataset("json", data_files=str(sft_path), split="train")

    def _tokenize(row: dict[str, Any]) -> dict[str, Any]:
        prompt = tok.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        full = prompt + row["messages"][-1]["content"] + tok.eos_token
        enc = tok(full, truncation=True, max_length=max_seq_len)
        p_len = len(tok(prompt, truncation=True, max_length=max_seq_len)["input_ids"])
        labels = list(enc["input_ids"])
        labels[:p_len] = [-100] * p_len  # loss only on the assistant JSON
        enc["labels"] = labels
        return enc

    ds = ds.map(_tokenize, remove_columns=ds.column_names)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=lora_cfg.epochs,
        learning_rate=lora_cfg.learning_rate,
        warmup_ratio=lora_cfg.warmup_ratio,
        per_device_train_batch_size=lora_cfg.per_device_batch_size,
        gradient_accumulation_steps=lora_cfg.gradient_accumulation,
        bf16=lora_cfg.bf16,
        logging_steps=20,
        save_strategy="epoch",
        report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=ds, tokenizer=tok).train()
    adapter_dir = Path(output_dir) / "adapter"
    model.save_pretrained(adapter_dir)
    return str(adapter_dir)
