#!/usr/bin/env python3
"""intervention_metatool_eval.py — E7 of EXPERIMENT_PLAN_v27 addendum.

Multi-tool selection F1 evaluation under static + Q-bias hybrid intervention.

Reuses prompt construction and tool-name parsing from eval_metatool_subtask4.py
(MetaTool Subtask 4 multi-tool benchmark, 497 × 2-tool GT). Compares four
conditions per query:

  full      : full prompt (system + tools + user) + no intervention (anchor)
  noprompt  : user only, no intervention (bottom)
  static    : user only + V_1 V_1^T phi_mean injection at o_proj input
  hybrid    : static + beta * V_k V_k^T Q at q_proj output

Reports F1, F_0.5, EU(α=1,β=2,γ=1), Jaccard, Exact-set per condition.

Usage:
  python intervention_metatool_eval.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --e1-json reports/rank_replaceability_2026_04/qwen_metatool_n256.json \
      --max-samples 64 --beta 4.0 --k-static 1 --k-qbias 8 \
      --device cuda:0 \
      --out reports/rank_replaceability_2026_04/qwen_e7_intervention_n64.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from measure_phi_rank import (  # type: ignore
    DEFAULT_TOOL_SYSTEM_PROMPT,
    detect_model_family,
    load_metatool_st4,
)
from qbias_hybrid_eval import (  # type: ignore
    StaticInjection,
    QBiasHook,
    build_static_injection,
    build_qbias_projector,
)


# =============================================================================
# Prompt construction (replicated from eval_metatool_subtask4.py for portability)
# =============================================================================

FC_SYSTEM_TEMPLATE = (
    "You are a tool-selection agent. Given a user query, emit ONE OR MORE "
    "<tool_call> blocks naming tools from this list: [{tools}]. "
    "Format each tool call exactly as:\n"
    '<tool_call>{{"name": "ToolName", "arguments": {{}}}}</tool_call>\n'
    "Emit MULTIPLE blocks if the query needs multiple tools. "
    "Output ONLY tool_call blocks; no explanation."
)


def build_full_prompt(tokenizer, query: str, tools: List[str]) -> str:
    """Build chat prompt with system (tool list) + user (query)."""
    sys_msg = FC_SYSTEM_TEMPLATE.format(tools=", ".join(tools))
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build_noprompt(tokenizer, query: str) -> str:
    """Chat prompt with user-only (no system / tool list)."""
    msgs = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# =============================================================================
# Tool name parser (replicated from eval_metatool_subtask4.py)
# =============================================================================

TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def extract_tool_names(generation: str, known_tools: List[str]) -> List[str]:
    """Extract all `"name": "X"` mentions from generation, filtered to known set.
    Returns names in order of appearance (may contain duplicates)."""
    found = TOOL_NAME_RE.findall(generation)
    tool_set = set(known_tools)
    return [t for t in found if t in tool_set]


# =============================================================================
# Multi-tool metrics (5 metrics matching eval_metatool_subtask4.py)
# =============================================================================

def compute_metrics(pred: List[str], gt: List[str]) -> dict:
    """Returns F1, F_0.5, EU, Jaccard, Exact for predicted vs GT tool sets.

    Both pred and gt are lists; we evaluate over the *unique* set per query.
    """
    pset = set(pred)
    gset = set(gt)
    if not gset:
        return {"f1": 0.0, "f_05": 0.0, "eu": 0.0, "jaccard": 0.0, "exact": 0.0}

    tp = len(pset & gset)
    fp = len(pset - gset)
    fn = len(gset - pset)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    # F_{0.5}: precision-weighted
    f_05 = (1.25 * precision * recall) / max(0.25 * precision + recall, 1e-9)
    # EU(alpha=1, beta=2, gamma=1): tp - 2*fp - fn (penalize wrong calls more)
    eu_raw = float(tp) - 2.0 * float(fp) - 1.0 * float(fn)
    # Normalize by max possible (perfect: |gset|)
    eu = max(eu_raw / max(len(gset), 1), -1.0)
    jaccard = tp / max(len(pset | gset), 1)
    exact = float(pset == gset)
    return {
        "f1": float(f1),
        "f_05": float(f_05),
        "eu": float(eu),
        "jaccard": float(jaccard),
        "exact": float(exact),
        "precision": float(precision),
        "recall": float(recall),
        "pred": list(pred),
        "gt": list(gt),
    }


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--metatool-path", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--e1-json", required=True)
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--k-static", type=int, default=1)
    p.add_argument("--k-qbias", type=int, default=8)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument(
        "--conditions",
        default="full,noprompt,static,hybrid",
        help="comma-separated subset of {full, noprompt, static, hybrid}",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


@torch.no_grad()
def generate_text(model, tokenizer, prompt_text: str, max_new_tokens: int, device: str) -> str:
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    new_ids = out[0, ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    valid = {"full", "noprompt", "static", "hybrid"}
    for c in conditions:
        if c not in valid:
            raise ValueError(f"unknown condition '{c}'")

    # Load E1
    e1_json = Path(args.e1_json)
    with open(e1_json) as f:
        e1 = json.load(f)
    npz_path = e1_json.parent / e1.get("npz_path", "")
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found", file=sys.stderr)
        return 2
    npz = np.load(npz_path)
    eigvecs = npz["eigvecs"]
    phi_mean = npz["phi_mean"]
    L, H_q, K_save, d_h = eigvecs.shape
    print(f"[e1] L={L} H_q={H_q} d_h={d_h}", flush=True)

    # Data
    items = load_metatool_st4(args.metatool_path, args.max_samples, full_schema=False)
    print(f"[data] N={len(items)}", flush=True)

    # Model
    print(f"[model] loading {args.model}", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    # Build interventions
    static_inj_np = build_static_injection(eigvecs, phi_mean, args.k_static)
    P_kk_np = build_qbias_projector(eigvecs, args.k_qbias)
    static_inj_t = torch.from_numpy(static_inj_np).to(args.device, dtype=dtype)
    P_kk_t = torch.from_numpy(P_kk_np).to(args.device, dtype=dtype)

    inj = StaticInjection(model, static_inj_t)
    inj.install()
    qb = QBiasHook(model, P_kk_t, beta=args.beta)
    qb.install()

    # Per-condition aggregator
    metrics = {c: [] for c in conditions}
    generations = {c: [] for c in conditions}

    t0 = time.time()
    for i, item in enumerate(items):
        query = item["query"]
        tools = item["candidates"]
        gt = item["gt"]
        # Normalize gt to list[str]
        if isinstance(gt, str):
            gt_list = [gt]
        elif isinstance(gt, list):
            gt_list = [str(g) for g in gt]
        else:
            gt_list = []

        prompt_full = build_full_prompt(tokenizer, query, tools)
        prompt_user = build_noprompt(tokenizer, query)

        for cond in conditions:
            if cond == "full":
                inj.enabled = False
                qb.enabled = False
                gen = generate_text(model, tokenizer, prompt_full, args.max_new_tokens, args.device)
            elif cond == "noprompt":
                inj.enabled = False
                qb.enabled = False
                gen = generate_text(model, tokenizer, prompt_user, args.max_new_tokens, args.device)
            elif cond == "static":
                inj.enabled = True
                qb.enabled = False
                gen = generate_text(model, tokenizer, prompt_user, args.max_new_tokens, args.device)
            elif cond == "hybrid":
                inj.enabled = True
                qb.beta = args.beta
                qb.enabled = True
                gen = generate_text(model, tokenizer, prompt_user, args.max_new_tokens, args.device)
            inj.enabled = False
            qb.enabled = False

            pred = extract_tool_names(gen, tools)
            m = compute_metrics(pred, gt_list)
            metrics[cond].append(m)
            generations[cond].append(gen[:512])  # truncate for storage

        if (i + 1) % 4 == 0 or i == len(items) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            f1_now = {c: np.mean([x["f1"] for x in metrics[c]]) for c in conditions}
            print(
                f"[{i+1}/{len(items)}] {elapsed:.1f}s rate={rate:.2f}/s  "
                + " ".join(f"{c[:5]}={f1_now[c]:.3f}" for c in conditions),
                flush=True,
            )

    inj.remove()
    qb.remove()

    # Aggregate
    summary = {}
    for cond in conditions:
        ms = metrics[cond]
        if not ms:
            continue
        summary[cond] = {
            "f1_mean": float(np.mean([m["f1"] for m in ms])),
            "f_05_mean": float(np.mean([m["f_05"] for m in ms])),
            "eu_mean": float(np.mean([m["eu"] for m in ms])),
            "jaccard_mean": float(np.mean([m["jaccard"] for m in ms])),
            "exact_mean": float(np.mean([m["exact"] for m in ms])),
            "precision_mean": float(np.mean([m["precision"] for m in ms])),
            "recall_mean": float(np.mean([m["recall"] for m in ms])),
            "n_pred_mean": float(np.mean([len(m["pred"]) for m in ms])),
        }

    out = {
        "model": args.model,
        "task": "metatool_st4",
        "n_samples": len(items),
        "k_static": args.k_static,
        "k_qbias": args.k_qbias,
        "beta": args.beta,
        "max_new_tokens": args.max_new_tokens,
        "conditions": conditions,
        "summary": summary,
        "details": metrics,
        "generations_sample": {c: generations[c][:5] for c in conditions},
        "wall_seconds": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[done] saved -> {out_path}", flush=True)
    print()
    hdr = "  ".join(f"{c:>9}" for c in conditions)
    print(f"{'metric':<10} {hdr}")
    print("-" * (12 + 11 * len(conditions)))
    for k in ["f1_mean", "f_05_mean", "eu_mean", "jaccard_mean", "exact_mean", "precision_mean", "recall_mean"]:
        row = "  ".join(f"{summary[c][k]:>9.4f}" for c in conditions)
        print(f"{k:<10} {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
