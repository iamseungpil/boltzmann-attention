#!/usr/bin/env python3
"""lora_train_metatool_v2.py — L1' fix of LoRA training.

Root-cause fixes vs v1 (lora_train_metatool.py):

  (1) Format match: training examples are built in the SAME chat-template +
      <tool_call> JSON format as Subtask4 evaluation, instead of plain
      "action_prompt + tool_name". This is the dominant fix — without it,
      gradient signal lands on a token distribution disjoint from the eval
      distribution.

  (2) Target modules: include o_proj + MLP up_proj/down_proj alongside
      q/k/v_proj. The MLP up/down channels are responsible for special
      token (<tool_call>) routing.

  (3) Held-out validation: validation set is drawn from MetaTool Subtask4
      (multi-tool) NOT from Subtask1, so val_loss measures actual transfer.

  (4) Lowered lr (5e-5) and earlier-stopping on val_loss to mitigate
      over-fit. Also gradient_checkpointing is already enabled.

Usage:
  python scripts/ocq/lora_train_metatool_v2.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --train-dataset /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json \
    --val-dataset   /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json \
    --train-size 800 --val-size 50 --epochs 3 \
    --lora-r 16 \
    --lora-target q_proj k_proj v_proj o_proj up_proj down_proj \
    --out-dir lora_adapters/qwen25_7b_subtask4_chat_r16
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--train-dataset", required=True,
                   help="Subtask1 JSON for training (single-tool GT).")
    p.add_argument("--val-dataset", required=True,
                   help="Subtask4 JSON for held-out val (multi-tool GT).")
    p.add_argument("--train-size", type=int, default=800)
    p.add_argument("--val-size", type=int, default=50)
    p.add_argument("--epochs", type=int, default=3)
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
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


SYSTEM_TEMPLATE = (
    "You are a tool-selection agent. Given a user query, emit ONE OR MORE "
    "<tool_call> blocks naming tools from this list: [{tool_list}]. "
    "Format each tool call exactly as:\n"
    '<tool_call>{{"name": "ToolName", "arguments": {{}}}}</tool_call>\n'
    "Emit MULTIPLE blocks if the query needs multiple tools. "
    "Do not include explanations. Output ONLY the <tool_call> blocks."
)


def _extract_user_query(action_prompt: str) -> str:
    m = re.search(r'User query:\s*"([^"]+)"\s*tool:\s*$', action_prompt)
    if m:
        return m.group(1)
    return action_prompt.split("User query:")[-1].strip().split('"')[1]


def build_chat_example(tokenizer, action_prompt: str, gt_tools: List[str]) -> Dict:
    """Build a single chat-template training example whose target is the
    <tool_call> JSON sequence for the GT tools. Loss is masked on the prompt
    portion."""
    cands = parse_candidates(action_prompt)
    user_query = _extract_user_query(action_prompt)
    sys_msg = SYSTEM_TEMPLATE.format(tool_list=", ".join(cands))
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    target_blocks = "".join(
        f'<tool_call>{{"name": "{t}", "arguments": {{}}}}</tool_call>'
        for t in gt_tools
    )
    target = target_blocks + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + target, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[:len(full_ids)]
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class ChatToolDataset(Dataset):
    def __init__(self, entries: List[dict], tokenizer, max_len: int,
                 single_tool: bool):
        self.entries = entries
        self.tok = tokenizer
        self.max_len = max_len
        self.single_tool = single_tool

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        prompt = e["action_prompt"]
        if self.single_tool:
            gt_tools = [e["tool"] if isinstance(e["tool"], str)
                        else e["tool"][0]]
        else:
            tool = e["tool"]
            gt_tools = tool if isinstance(tool, list) else [tool]
        item = build_chat_example(self.tok, prompt, gt_tools)
        # Truncate to max_len
        for k in ("input_ids", "labels"):
            if item[k].shape[0] > self.max_len:
                item[k] = item[k][:self.max_len]
        return item


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
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": attention_mask}


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
    train_entries = train_all[:args.train_size]
    print(f"[data] train (Subtask1 chat-formatted)={len(train_entries)}",
          flush=True)

    with open(args.val_dataset) as f:
        val_all = json.load(f)
    random.shuffle(val_all)
    val_entries = val_all[:args.val_size]
    print(f"[data] val (Subtask4 multi-tool chat-formatted)={len(val_entries)}",
          flush=True)

    train_ds = ChatToolDataset(train_entries, tok, args.max_seq_len,
                               single_tool=True)
    val_ds = ChatToolDataset(val_entries, tok, args.max_seq_len,
                             single_tool=False)

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
            if step % 20 == 0:
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
        "format_version": "v2_chat_template",
        "train_dataset": args.train_dataset,
        "val_dataset": args.val_dataset,
        "train_size": len(train_entries),
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
