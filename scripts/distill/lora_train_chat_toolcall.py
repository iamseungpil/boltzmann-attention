#!/usr/bin/env python3
"""lora_train_chat_toolcall.py — multi-turn tool-use chat LoRA-SFT (assistant-only loss).

Track-A deliverable for the facet-distillation campaign (EXPERIMENT_DESIGN §13).
Consumes the SFT JSONL produced by build_sft_dataset.py:
    {"messages":[{role: system|user|assistant|tool, content, tool_calls?, tool_call_id?}],
     "tools":[<openai function schema>], "meta":{...}}
and fine-tunes a base chat model (default Qwen2.5-7B-Instruct; coworker scales to 32B)
to internalize the teacher's goal->tool selection procedure.

Loss masking — ASSISTANT TURNS ONLY:
  The huge system policy (~6K tok for telecom), user turns, and tool responses are
  masked to -100; only assistant content + assistant tool_calls (+ their closing
  <|im_end|>) are supervised. Qwen2.5's chat template lacks `{% generation %}`, so
  `return_assistant_tokens_mask` returns all-zero — we use an INCREMENTAL BOUNDARY
  method instead (validated: 15% supervised, correct spans):
    for each assistant message i:
       pre = template(messages[:i],   add_generation_prompt=True)   # up to assistant header
       inc = template(messages[:i+1], add_generation_prompt=False)  # includes the turn
       supervise tokens [len(pre) : len(inc))
  tool_call arguments stored as JSON strings are normalized back to dicts so the
  chat template renders them correctly.

ENV (NOT seka_env — it lacks peft): a training env with
    torch, transformers>=4.51, peft, accelerate   (flash-attn optional)
Records are long; default --max-seq-len 10240, --batch-size 1 + grad-ckpt.

Usage:
  python scripts/distill/lora_train_chat_toolcall.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --train-jsonl reports/facet_rft_2026/phase4_distill/sft_data/sft_plain_train_all.jsonl \
    --out-dir lora_adapters/qwen7b_plain --epochs 3 --lora-r 16 --val-frac 0.05
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
# peft imported lazily inside main() so --encode-only works without peft installed


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", default=None, help="optional explicit val set")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="held-out fraction if --val-jsonl not given")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--max-seq-len", type=int, default=14336)
    p.add_argument("--system-mode", default="full", choices=["full", "none", "minimal"],
                   help="full=keep policy prompt (plain SFT, needs prompt at inference); "
                        "none=drop system policy (INTERNALIZATION arm: student absorbs policy "
                        "into weights, §13 efficiency test); minimal=short stub only. "
                        "tools schema kept in all modes.")
    p.add_argument("--minimal-stub", default=(
        "You are a customer service agent. Use the available tools to help the user "
        "according to the system's policy. Output valid JSON tool calls only."))
    p.add_argument("--skip-overlong", action="store_true",
                   help="skip records exceeding max-seq-len (default: right-truncate)")
    p.add_argument("--max-examples", type=int, default=0, help="cap for debug (0=all)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=0,
                   help="save adapter+optimizer state every N optimizer steps "
                        "(preemption-safe resume; 0=off)")
    p.add_argument("--init-adapter", default=None,
                   help="path to an existing LoRA adapter dir to warm-start from "
                        "(must match lora-r/alpha/target)")
    p.add_argument("--resume", action="store_true",
                   help="resume from <out-dir>/ckpt_state.pt if present "
                        "(deterministic per-epoch order; fast-forwards skipped batches)")
    p.add_argument("--encode-only", action="store_true",
                   help="build+report dataset encoding then exit (no model/training)")
    p.add_argument("--mask-toolcalls", action="store_true",
                   help="supervise only assistant CONTENT (e.g. 'Plan: <step>'), mask "
                        "tool_call tokens to -100 — trains a TBox-only planner "
                        "(use with build_tbox_sft.py data).")
    return p.parse_args()


# ---------------------------------------------------------------------------
def normalize_messages(msgs):
    """Deepcopy + convert tool_call arguments from JSON string to dict (template needs dict)."""
    m2 = copy.deepcopy(msgs)
    for m in m2:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    fn["arguments"] = json.loads(a)
                except Exception:
                    fn["arguments"] = {}
    return m2


def apply_system_mode(msgs, mode, minimal_stub):
    """full=keep system policy; none=empty system (policy removed, tools kept =
    internalization arm); minimal=short stub. The system message is RETAINED (with
    emptied/stub content) so the conversation never starts with an assistant turn
    (empty-prefix would crash apply_chat_template) and tools still render."""
    if mode == "full":
        return msgs
    new_content = "" if mode == "none" else minimal_stub
    out, has_sys = [], False
    for m in msgs:
        if m.get("role") == "system":
            has_sys = True
            out.append({"role": "system", "content": new_content})
        else:
            out.append(m)
    if not has_sys:
        out = [{"role": "system", "content": new_content}] + out
    return out


def encode_record(tok, rec, max_len, skip_overlong, system_mode="full", minimal_stub="",
                  mask_toolcalls=False):
    """Return (input_ids, labels) with assistant-only supervision, or None to skip.

    mask_toolcalls=True: for an assistant turn that carries tool_calls, supervise ONLY
    its content (e.g. 'Plan: <step>') tokens and leave the tool_call serialization at
    -100. The tool_calls stay in the rendered sequence (chat-template validity: the
    following tool result needs a parent) but get no gradient -> TBox-only planner.
    Content end = longest common prefix between the full turn render and a render of
    the same turn with tool_calls stripped.
    """
    msgs = normalize_messages(rec["messages"])
    msgs = apply_system_mode(msgs, system_mode, minimal_stub)
    tools = rec.get("tools") or None
    full = tok.apply_chat_template(msgs, tools=tools, tokenize=True)
    labels = [-100] * len(full)
    sup = 0
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        pre = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=True,
                                      add_generation_prompt=True)
        inc = tok.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=True,
                                      add_generation_prompt=False)
        s, e = len(pre), len(inc)
        # boundary sanity: pre must be a prefix of full, inc within full
        if e > len(full) or full[:s] != pre[:s]:
            return None, "boundary_mismatch"
        if mask_toolcalls and m.get("tool_calls"):
            # render the SAME turn without tool_calls; supervise only the shared
            # (content) prefix, mask the tool_call tokens.
            mc = dict(m); mc.pop("tool_calls", None)
            inc_nc = tok.apply_chat_template(msgs[:i] + [mc], tools=tools, tokenize=True,
                                             add_generation_prompt=False)
            k = s
            lim = min(len(inc_nc), e)
            while k < lim and full[k] == inc_nc[k]:
                k += 1
            e = k  # supervise only [s, content_end)
        for j in range(s, e):
            labels[j] = full[j]
        sup += (e - s)
    if sup == 0:
        return None, "no_assistant"
    if len(full) > max_len:
        if skip_overlong:
            return None, "overlong"
        full = full[:max_len]
        labels = labels[:max_len]
        if all(l == -100 for l in labels):
            return None, "overlong_no_sup"
    return {"input_ids": full, "labels": labels}, "ok"


class ChatToolDataset(Dataset):
    def __init__(self, records, tok, max_len, skip_overlong, tag="",
                 system_mode="full", minimal_stub="", mask_toolcalls=False):
        self.items = []
        stats = {"ok": 0, "boundary_mismatch": 0, "no_assistant": 0,
                 "overlong": 0, "overlong_no_sup": 0}
        for k, rec in enumerate(records):
            item, why = encode_record(tok, rec, max_len, skip_overlong,
                                      system_mode, minimal_stub, mask_toolcalls)
            stats[why] = stats.get(why, 0) + 1
            if item is not None:
                self.items.append(item)
            if (k + 1) % 200 == 0:
                print(f"  [{tag}] encoded {k+1}/{len(records)} kept={len(self.items)}",
                      flush=True)
        print(f"[{tag}] dataset: kept={len(self.items)} stats={stats}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return {"input_ids": torch.tensor(it["input_ids"], dtype=torch.long),
                "labels": torch.tensor(it["labels"], dtype=torch.long)}


def collate(batch, pad_id):
    maxlen = max(x["input_ids"].shape[0] for x in batch)
    ids, lbl, att = [], [], []
    for x in batch:
        n = x["input_ids"].shape[0]
        pad = maxlen - n
        ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        lbl.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        att.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {"input_ids": torch.stack(ids), "labels": torch.stack(lbl),
            "attention_mask": torch.stack(att)}


def load_jsonl(path, cap=0):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
            if cap and len(recs) >= cap:
                break
    return recs


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[load] tokenizer {args.base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_recs = load_jsonl(args.train_jsonl, args.max_examples)
    if args.val_jsonl:
        val_recs = load_jsonl(args.val_jsonl, 0)
    else:
        random.shuffle(train_recs)
        n_val = max(1, int(len(train_recs) * args.val_frac)) if args.val_frac > 0 else 0
        val_recs, train_recs = train_recs[:n_val], train_recs[n_val:]
    print(f"[data] train={len(train_recs)} val={len(val_recs)}", flush=True)

    train_ds = ChatToolDataset(train_recs, tok, args.max_seq_len, args.skip_overlong,
                               "train", args.system_mode, args.minimal_stub,
                               args.mask_toolcalls)
    val_ds = ChatToolDataset(val_recs, tok, args.max_seq_len, args.skip_overlong,
                             "val", args.system_mode, args.minimal_stub,
                             args.mask_toolcalls) \
        if val_recs else None

    if args.encode_only:
        # report supervised-token fraction and exit (CPU smoke, no model)
        import statistics as st
        fr = []
        for it in train_ds.items[:200]:
            n = len(it["input_ids"]); s = sum(1 for x in it["labels"] if x != -100)
            fr.append(s / max(n, 1))
        print(f"[encode-only] supervised frac (first 200): "
              f"mean={st.mean(fr):.3f} min={min(fr):.3f} max={max(fr):.3f}")
        print(f"[encode-only] seq len: "
              f"mean={st.mean([len(it['input_ids']) for it in train_ds.items]):.0f} "
              f"max={max(len(it['input_ids']) for it in train_ds.items)}")
        return 0

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[load] model {args.base_model} dtype={args.dtype} attn={args.attn}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=args.device,
        attn_implementation=args.attn, low_cpu_mem_usage=True)
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model, TaskType
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      task_type=TaskType.CAUSAL_LM, target_modules=args.lora_target)
    model = get_peft_model(model, lcfg)
    if args.init_adapter:
        # warm-start LoRA from an existing adapter (e.g. RAFT continuing from SFT)
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        sd = load_file(str(Path(args.init_adapter) / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, sd)
        print(f"[init] warm-started LoRA from {args.init_adapter}", flush=True)
    model.print_trainable_parameters()
    # grad-ckpt AFTER peft wrap + enable_input_require_grads so checkpointing actually
    # saves activation memory through the frozen base (else backward OOMs at long seq)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    pad_id = tok.pad_token_id
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, pad_id)) if val_ds else None

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    # ---- preemption-safe resume (adapter + optimizer + position) ----
    start_epoch, resume_step = 0, -1
    state_path = out_dir / "ckpt_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        sd = load_file(str(out_dir / "resume_adapter" / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, sd)
        optim.load_state_dict(st["optim"])
        start_epoch, resume_step = st["epoch"], st["step"]
        best = st.get("best", best)
        print(f"[resume] restored ep{start_epoch} step{resume_step} best={best:.4f}",
              flush=True)

    def _save_ckpt(epoch, step):
        model.save_pretrained(str(out_dir / "resume_adapter"))
        tmp = out_dir / "ckpt_state.pt.tmp"
        torch.save({"epoch": epoch, "step": step, "optim": optim.state_dict(),
                    "best": best}, tmp)
        tmp.rename(state_path)
        print(f"  [ckpt] ep{epoch} step{step}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        # deterministic per-epoch shuffle so a resumed run replays the same order
        g = torch.Generator(); g.manual_seed(args.seed * 1000 + epoch)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  generator=g, collate_fn=lambda b: collate(b, pad_id))
        model.train(); tot = 0.0; n = 0; optim.zero_grad()
        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and resume_step >= 0 and step <= resume_step:
                continue  # fast-forward through already-trained batches
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                optim.step(); optim.zero_grad()
                if args.save_every and \
                        ((step + 1) // args.grad_accum) % args.save_every == 0:
                    _save_ckpt(epoch, step)
            tot += out.loss.item(); n += 1
            if step % args.log_every == 0:
                print(f"  ep{epoch} step{step} loss={out.loss.item():.4f}", flush=True)
        tr = tot / max(n, 1)

        avg_val = float("nan")
        if val_loader:
            model.eval(); vl = 0.0; vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(args.device) for k, v in batch.items()}
                    vl += model(**batch).loss.item(); vn += 1
            avg_val = vl / max(vn, 1)
        print(f"[ep{epoch}] train_loss={tr:.4f} val_loss={avg_val:.4f}", flush=True)

        improved = (not val_loader) or (avg_val < best - 1e-3)
        if improved:
            best = avg_val if val_loader else tr
            model.save_pretrained(str(out_dir)); tok.save_pretrained(str(out_dir))
            print("  -> saved", flush=True)
        if args.save_every:
            _save_ckpt(epoch + 1, -1)  # epoch boundary: resume starts next epoch clean

    (out_dir / "train_meta.json").write_text(json.dumps({
        "base_model": args.base_model, "format": "chat_toolcall_v1",
        "train_jsonl": args.train_jsonl, "n_train": len(train_ds),
        "n_val": len(val_ds) if val_ds else 0,
        "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha, "lora_target": args.lora_target,
        "max_seq_len": args.max_seq_len, "best_val_loss": best,
    }, indent=2))
    print(f"[done] best={best:.4f} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
