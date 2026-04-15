#!/usr/bin/env python3
"""lora_train_metatool.py — L1 stage of LoRA + Rotation hybrid.

Fine-tunes LoRA (rank 8) on q_proj/k_proj/v_proj of Qwen2.5-7B-Instruct
using MetaTool Subtask1 training examples (GT tool name prediction).

Expected effects under Theorem 6.16:
  - K distribution develops bimodal facet structure (H-cat satisfied)
  - K subspace aligns with δW_K column space
  - B_ont re-built from LoRA-adapted K → tighter Cor 6.8 bound

Usage:
  python scripts/ocq/lora_train_metatool.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --train-dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --train-size 500 --epochs 3 --lora-r 8 \
    --out-dir lora_adapters/qwen25_7b_subtask1_r8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--train-dataset", required=True,
                   help="Path to MetaTool Subtask1 JSON (train split).")
    p.add_argument("--train-size", type=int, default=500)
    p.add_argument("--val-size", type=int, default=50)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-target", nargs="+",
                   default=["q_proj", "k_proj", "v_proj"])
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


class MetaToolDataset(Dataset):
    """Build training examples: action_prompt + GT tool_name
    Loss masked so only the GT tool name tokens contribute.
    """
    def __init__(self, entries: List[dict], tokenizer, max_len: int):
        self.entries = entries
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        prompt = e["action_prompt"]
        # GT tool name (Subtask1 single-tool)
        tool = e["tool"] if isinstance(e["tool"], str) else e["tool"][0]
        # Tokenize prompt alone (for boundary)
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        # Tokenize full sequence
        full_text = prompt + tool + self.tok.eos_token
        full_ids = self.tok(full_text, add_special_tokens=False,
                            truncation=True, max_length=self.max_len)["input_ids"]
        # Labels: -100 for prompt tokens, actual for completion
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch, pad_id=0):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attention_mask[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[load] base={args.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32, "auto": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    with open(args.train_dataset) as f:
        all_entries = json.load(f)
    random.shuffle(all_entries)
    train_entries = all_entries[:args.train_size]
    val_entries = all_entries[args.train_size:args.train_size + args.val_size]
    print(f"[data] train={len(train_entries)} val={len(val_entries)}", flush=True)

    train_ds = MetaToolDataset(train_entries, tok, args.max_seq_len)
    val_ds = MetaToolDataset(val_entries, tok, args.max_seq_len)

    pad_id = tok.pad_token_id
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, pad_id))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    device = args.device

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
            n += 1
            if step % 20 == 0:
                print(f"  ep{epoch} step{step} loss={loss.item():.4f}", flush=True)

        # Validation
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_loss += out.loss.item()
                vn += 1
        print(f"[ep{epoch}] train_loss={total_loss/max(n,1):.4f} val_loss={val_loss/max(vn,1):.4f}",
              flush=True)

    # Save LoRA adapter
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[save] LoRA adapter → {out_dir}", flush=True)

    # Meta info
    (out_dir / "train_meta.json").write_text(json.dumps({
        "base_model": args.base_model,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "epochs": args.epochs,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_target": args.lora_target,
        "lr": args.lr, "batch_size": args.batch_size,
    }, indent=2))
    print(f"[done]", flush=True)


if __name__ == "__main__":
    main()
