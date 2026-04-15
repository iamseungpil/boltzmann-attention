#!/usr/bin/env python3
"""lora_train_metatool_v3.py — L1'' fix for v2's single-tool bias.

Root cause of v2 failure: training only on Subtask1 (single-tool GT)
compressed the LoRA-merged model's output policy to "emit exactly one
<tool_call> block", catastrophically harming Subtask4 multi-tool eval
(F1 0.219 at full 497).

v3 fix: augment training data with synthetic 2-tool examples constructed
from the same Subtask1 query's candidate list. Each Subtask1 entry yields:
  - 1 single-tool example (GT = [original_tool])
  - 1 synthetic 2-tool example (GT = [original_tool, second_candidate])
where second_candidate is sampled from the remaining 9 candidates.

The synthetic 2-tool examples teach the model to emit MULTIPLE <tool_call>
blocks; the original single-tool examples preserve correct selection.
Validation remains on real Subtask4 (held-out, no contamination).

Usage:
  python scripts/ocq/lora_train_metatool_v3.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --train-dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --val-dataset   /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json \
    --train-size 600 --val-size 50 --epochs 4 --lora-r 16 \
    --out-dir lora_adapters/qwen25_7b_subtask4_synth_r16
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Dict

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from eval_metatool_subtask1 import parse_candidates  # noqa: E402
from lora_train_metatool_v2 import (  # noqa: E402
    SYSTEM_TEMPLATE, _extract_user_query, build_chat_example, collate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--train-dataset", required=True)
    p.add_argument("--val-dataset", required=True)
    p.add_argument("--train-size", type=int, default=600,
                   help="Number of Subtask1 source entries; each yields 2 train examples.")
    p.add_argument("--val-size", type=int, default=50)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target", nargs="+",
                   default=["q_proj", "k_proj", "v_proj",
                            "o_proj", "up_proj", "down_proj"])
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop-patience", type=int, default=2)
    p.add_argument("--synth-frac", type=float, default=0.5,
                   help="Fraction of train examples that are synthetic 2-tool (vs single-tool).")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def build_synth_pair(rng: random.Random, entry: dict) -> List[str]:
    """Construct a synthetic 2-tool GT for a Subtask1 entry by pairing the
    original GT with a randomly sampled second candidate from the same
    query's candidate list."""
    cands = parse_candidates(entry["action_prompt"])
    gt = entry["tool"] if isinstance(entry["tool"], str) else entry["tool"][0]
    others = [c for c in cands if c != gt]
    if not others:
        return [gt]
    second = rng.choice(others)
    # Random order so model learns both orderings
    pair = [gt, second]
    rng.shuffle(pair)
    return pair


class MixedToolDataset(Dataset):
    """Mixes single-tool (Subtask1 GT as-is) and synthetic 2-tool examples."""
    def __init__(self, source_entries: List[dict], tokenizer, max_len: int,
                 synth_frac: float, seed: int):
        self.tok = tokenizer
        self.max_len = max_len
        rng = random.Random(seed)
        self.examples = []
        for e in source_entries:
            # Always include the single-tool example
            gt_single = [e["tool"] if isinstance(e["tool"], str) else e["tool"][0]]
            self.examples.append((e["action_prompt"], gt_single, "single"))
            # Synthetic 2-tool example with prob = synth_frac
            if rng.random() < synth_frac:
                gt_pair = build_synth_pair(rng, e)
                self.examples.append((e["action_prompt"], gt_pair, "synth2"))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, gt_tools, _kind = self.examples[idx]
        item = build_chat_example(self.tok, prompt, gt_tools)
        for k in ("input_ids", "labels"):
            if item[k].shape[0] > self.max_len:
                item[k] = item[k][:self.max_len]
        return item


class Subtask4ValDataset(Dataset):
    def __init__(self, entries: List[dict], tokenizer, max_len: int):
        self.entries = entries
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        gt = e["tool"] if isinstance(e["tool"], list) else [e["tool"]]
        item = build_chat_example(self.tok, e["action_prompt"], gt)
        for k in ("input_ids", "labels"):
            if item[k].shape[0] > self.max_len:
                item[k] = item[k][:self.max_len]
        return item


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

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    with open(args.train_dataset) as f:
        train_all = json.load(f)
    random.shuffle(train_all)
    source_entries = train_all[:args.train_size]
    print(f"[data] train sources={len(source_entries)} synth_frac={args.synth_frac}",
          flush=True)

    with open(args.val_dataset) as f:
        val_all = json.load(f)
    random.shuffle(val_all)
    val_entries = val_all[:args.val_size]
    print(f"[data] val (Subtask4 multi-tool)={len(val_entries)}", flush=True)

    train_ds = MixedToolDataset(source_entries, tok, args.max_seq_len,
                                args.synth_frac, args.seed)
    val_ds = Subtask4ValDataset(val_entries, tok, args.max_seq_len)
    print(f"[data] train examples (after augmentation)={len(train_ds)}",
          flush=True)
    n_single = sum(1 for _, _, k in train_ds.examples if k == "single")
    n_synth = sum(1 for _, _, k in train_ds.examples if k == "synth2")
    print(f"[data]   single-tool={n_single}, synth-2-tool={n_synth}",
          flush=True)

    pad_id = tok.pad_token_id
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, pad_id))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=0.01)
    device = args.device

    best_val = float("inf")
    patience_left = args.early_stop_patience
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        optim.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                optim.step()
                optim.zero_grad()
            total_loss += out.loss.item()
            n += 1
            if step % 30 == 0:
                print(f"  ep{epoch} step{step} loss={out.loss.item():.4f}",
                      flush=True)

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_loss += out.loss.item()
                vn += 1
        avg_val = val_loss / max(vn, 1)
        print(f"[ep{epoch}] train_loss={total_loss/max(n,1):.4f} "
              f"val_loss(Subtask4)={avg_val:.4f}", flush=True)

        if avg_val < best_val - 1e-3:
            best_val = avg_val
            patience_left = args.early_stop_patience
            model.save_pretrained(str(out_dir))
            tok.save_pretrained(str(out_dir))
            print(f"  → new best, saved.", flush=True)
        else:
            patience_left -= 1
            print(f"  → no improvement, patience_left={patience_left}",
                  flush=True)
            if patience_left <= 0:
                print(f"[early-stop] at ep{epoch}", flush=True)
                break

    (out_dir / "train_meta.json").write_text(json.dumps({
        "base_model": args.base_model,
        "format_version": "v3_synth_multi_tool",
        "train_dataset": args.train_dataset,
        "val_dataset": args.val_dataset,
        "train_sources": len(source_entries),
        "train_examples_after_aug": len(train_ds),
        "synth_frac": args.synth_frac,
        "single_count": n_single,
        "synth_count": n_synth,
        "val_size": len(val_entries),
        "epochs": args.epochs,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_target": args.lora_target,
        "lr": args.lr, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "best_val_loss_subtask4": best_val,
    }, indent=2))
    print(f"[done] best_val_loss(Subtask4)={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
