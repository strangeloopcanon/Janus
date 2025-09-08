#!/usr/bin/env python
"""Train a LoRA student on rewrite pairs (prompt -> output).

Given a JSONL with fields {prompt, output}, performs causal LM training with
labels masked on the prompt tokens and LoRA adapters on attention/MLP.

Example (paranoid student on first 800 rows):
  python scripts/train_lora_student.py \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --train /tmp/ccnews_train_800_paranoid.jsonl \
    --eval  /tmp/ccnews_eval_200_base.jsonl \
    --out results/students/paranoid_lora \
    --epochs 1 --lr 2e-4 --batch-size 1 --grad-accum 8 \
    --max-seq-len 512 --dtype fp32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, PeftModel


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        for ln in fp:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


class PromptOutputDataset(torch.utils.data.Dataset):
    def __init__(self, tok, path: Path, max_len: int):
        self.samples: List[Dict] = []
        self.tok = tok
        self.max_len = max_len
        for ex in iter_jsonl(path):
            p = (ex.get("prompt") or "").strip()
            o = (ex.get("output") or "").strip()
            if not p or not o:
                continue
            self.samples.append({"prompt": p, "output": o})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ex = self.samples[idx]
        p = ex["prompt"]
        o = ex["output"]
        # Encode prompt and output; truncate to fit max_len
        enc_p = self.tok(p, add_special_tokens=True)
        enc_o = self.tok(o, add_special_tokens=False)
        p_ids = enc_p["input_ids"]
        o_ids = enc_o["input_ids"]
        cap = max(16, self.max_len)
        # Keep full output; trim prompt from the left if needed
        if len(p_ids) + len(o_ids) > cap:
            keep_p = max(1, cap - len(o_ids))
            if keep_p < len(p_ids):
                p_ids = p_ids[-keep_p:]
        input_ids = p_ids + o_ids
        # Labels: mask prompt tokens
        labels = [-100] * len(p_ids) + o_ids[:]
        attn_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LoRA student on rewrites (prompt->output)")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default=None, help="Optional eval JSONL (uses prompts/outputs for loss only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--dtype", choices=["auto","fp32","fp16","bf16"], default="auto")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--init-lora", default=None, help="Optional existing LoRA adapter to initialize from (resume)")
    args = ap.parse_args()

    device = best_device()
    print(f"[init] device={device}, dtype={args.dtype}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "right"
    tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(args.base_model)
    if args.dtype != "auto":
        if args.dtype == "fp16":
            mdl = mdl.to(torch.float16)
        elif args.dtype == "bf16":
            mdl = mdl.to(torch.bfloat16)
        elif args.dtype == "fp32":
            mdl = mdl.to(torch.float32)
    mdl.to(device)

    # LoRA on attention and MLP projections (Qwen-like architectures)
    if args.init_lora:
        # Resume from existing adapter weights
        mdl = PeftModel.from_pretrained(mdl, args.init_lora, is_trainable=True)
    else:
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj",
            ],
        )
        mdl = get_peft_model(mdl, lora)
    mdl.print_trainable_parameters()

    # Ensure checkpointing works with LoRA by enabling input gradients
    try:
        mdl.enable_input_require_grads()
    except Exception:
        pass
    try:
        mdl.config.use_cache = False  # required when gradient checkpointing is on
    except Exception:
        pass

    train_ds = PromptOutputDataset(tok, Path(args.train), args.max_seq_len)
    eval_ds = PromptOutputDataset(tok, Path(args.eval), args.max_seq_len) if args.eval else None

    # We provide labels explicitly (prompt masked as -100), so use the default collator.
    data_collator = default_data_collator

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    args_tr = TrainingArguments(
        output_dir=str(outdir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=mdl,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tok,
    )
    trainer.train()

    # Save LoRA adapter and tokenizer
    mdl.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    print(f"✓ Saved LoRA to {outdir}")


if __name__ == "__main__":
    main()
