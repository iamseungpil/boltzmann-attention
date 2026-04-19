#!/usr/bin/env python3
"""F12 — Evaluate FacetRot-QK checkpoint on MetaTool Subtask4.

Loads a checkpoint saved by `train_f12_facetrot_qk.py`, re-installs the
hooks with their trained rotation + log_tau, and runs greedy `model.generate`
over the held-out eval split. Computes F1/F_0.5/EU + stepwise metrics.

Usage:
  source /home/woori/venvs/seka_env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python3 scripts/new_theorem_test/eval_subtask4_facetrot_qk.py \\
    --checkpoint external/SEKA/seka_projections/f12b-qwen25-7b-r32-uniform/f12_checkpoint.pt \\
    --subspace external/SEKA/seka_projections/f12-qwen25-7b-metatool-facet-subspace/facet_subspace.pt \\
    --start-idx 350 --max-samples 147 \\
    --baseline-also \\
    --out reports/f12_metatool/f12b_eval_n147.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from train_f12_facetrot_qk import (                                           # noqa: E402
    FacetRotSO2,
    FacetSubspace,
    FacetRotHook,
    build_kq_schedule,
    install_hooks,
    F12Config,
    _repeat_kv_to_q,
)
from eval_metatool_subtask1 import parse_candidates, resolve_dtype            # noqa: E402
from eval_metatool_subtask4 import (                                          # noqa: E402
    build_fc_prompt,
    extract_tool_name_sequence,
    compute_metrics,
    compute_stepwise_metrics,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASET_DEFAULT = "/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--subspace", type=Path, default=None,
                   help="override subspace path; defaults to checkpoint config.")
    p.add_argument("--dataset", default=DATASET_DEFAULT)
    p.add_argument("--start-idx", type=int, default=350)
    p.add_argument("--max-samples", type=int, default=147)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--baseline-also", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    data: List[dict],
    method_name: str,
    args,
    device: torch.device,
) -> Dict:
    macro_keys = ["F1", "F_0.5", "EU", "Jaccard", "Exact", "precision", "recall"]
    stepwise_keys = [
        "emitted_any_rate", "emitted_two_rate", "emitted_two_distinct_rate",
        "first_tool_hit_rate", "second_tool_hit_rate", "second_distinct_hit_rate",
        "second_recovery_given_first_hit_rate", "repeated_first_tool_rate",
    ]
    macro = {k: 0.0 for k in macro_keys}
    stepwise = {k: 0.0 for k in stepwise_keys}
    per_sample: List[dict] = []

    t0 = time.time()
    for i, entry in enumerate(data):
        action_prompt = entry["action_prompt"]
        gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
        cands = parse_candidates(action_prompt)
        fc_prompt = build_fc_prompt(tokenizer, action_prompt, cands)
        ids = tokenizer(fc_prompt, return_tensors="pt").to(device)
        out = model.generate(
            **ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0, ids["input_ids"].shape[1]:]
        generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_sequence = extract_tool_name_sequence(generation, cands)
        pred: List[str] = []
        for name in pred_sequence:
            if name not in pred:
                pred.append(name)

        metrics = compute_metrics(pred, gt, facet_map=None)
        stepwise_m = compute_stepwise_metrics(pred_sequence, gt, facet_map=None)
        for k in macro:
            macro[k] += metrics[k]
        for k in stepwise:
            stepwise[k] += stepwise_m.get(k, 0.0)
        per_sample.append({
            "index": entry.get("index", args.start_idx + i),
            "query": entry.get("query", "")[:150],
            "gt": gt,
            "pred": pred,
            "pred_sequence": pred_sequence,
            "generation_head": generation[:300],
            "metrics": metrics,
            "stepwise": stepwise_m,
        })
        if args.verbose and i < 3:
            print(f"[{method_name} {i}] gt={gt} pred={pred} F1={metrics['F1']:.3f}", flush=True)

    N = max(len(data), 1)
    for k in macro:
        macro[k] /= N
    for k in stepwise:
        stepwise[k] /= N
    runtime = time.time() - t0
    return {
        "method": method_name,
        "n_queries": len(data),
        "macro": macro,
        "stepwise": stepwise,
        "runtime_s": runtime,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _restore_modules(ckpt: dict, subspace_blob: dict, device: torch.device):
    ck_cfg = ckpt["config"]
    n_facets = int(ck_cfg["n_facets"])
    rot_pairs = int(ck_cfg["rot_pairs"])
    rotation = FacetRotSO2(n_facets=n_facets, rot_pairs=rot_pairs).to(device=device, dtype=torch.float32)
    theta_tensor = ckpt["rotation_theta"].to(device=device, dtype=torch.float32)
    with torch.no_grad():
        rotation.theta.copy_(theta_tensor)

    n_q = int(ck_cfg["n_q"])
    n_kv = int(ck_cfg["n_kv"])
    head_dim = int(ck_cfg["head_dim"])
    B_fac_all = subspace_blob["B_fac"].float()         # (L, n_kv, d, R)
    anchors_all = subspace_blob["anchors"].float()     # (F, L, n_kv, d)
    num_layers = B_fac_all.shape[0]

    s_K_schedule = list(ckpt["s_K_schedule"])
    s_Q_schedule = list(ckpt["s_Q_schedule"])

    k_subspaces = nn.ModuleDict()
    q_subspaces = nn.ModuleDict()
    k_log_tau = ckpt.get("k_log_tau", {})
    q_log_tau = ckpt.get("q_log_tau", {})
    for ell in range(num_layers):
        if s_K_schedule[ell] == 0.0 and s_Q_schedule[ell] == 0.0:
            continue
        B_K = B_fac_all[ell]
        anchors_K = anchors_all[:, ell]
        if s_K_schedule[ell] > 0.0:
            sub = FacetSubspace(B_K.clone(), anchors_K.clone(), tau_init=1.0).to(device=device, dtype=torch.float32)
            if str(ell) in k_log_tau:
                with torch.no_grad():
                    sub.log_tau.copy_(torch.tensor(float(k_log_tau[str(ell)]), device=device))
            k_subspaces[str(ell)] = sub
        if s_Q_schedule[ell] > 0.0:
            B_Q, anchors_Q = _repeat_kv_to_q(B_K, anchors_K, n_q, n_kv)
            sub = FacetSubspace(B_Q.clone(), anchors_Q.clone(), tau_init=1.0).to(device=device, dtype=torch.float32)
            if str(ell) in q_log_tau:
                with torch.no_grad():
                    sub.log_tau.copy_(torch.tensor(float(q_log_tau[str(ell)]), device=device))
            q_subspaces[str(ell)] = sub
    return rotation, k_subspaces, q_subspaces, s_K_schedule, s_Q_schedule, n_q, n_kv, head_dim


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[checkpoint] {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_cfg = ckpt["config"]
    subspace_path = Path(args.subspace) if args.subspace else Path(ck_cfg["subspace_path"])
    if not subspace_path.is_absolute():
        subspace_path = REPO / subspace_path
    print(f"[subspace] {subspace_path}", flush=True)
    subspace_blob = torch.load(subspace_path, map_location="cpu", weights_only=False)

    # --- load model ---
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = resolve_dtype(args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    device = torch.device(args.device)
    rotation, k_subs, q_subs, s_K, s_Q, n_q, n_kv, head_dim = _restore_modules(
        ckpt, subspace_blob, device
    )
    for p in rotation.parameters():
        p.requires_grad_(False)
    for sub in list(k_subs.values()) + list(q_subs.values()):
        for p in sub.parameters():
            p.requires_grad_(False)

    # --- load data ---
    with open(args.dataset) as f:
        data_all = json.load(f)
    start = max(0, args.start_idx)
    end = start + args.max_samples if args.max_samples > 0 else len(data_all)
    data = data_all[start:end]
    print(f"[data] {len(data)} queries (slice {start}:{end})", flush=True)

    # --- baseline (optional, no hooks) ---
    baseline_result: Optional[Dict] = None
    if args.baseline_also:
        print("\n[eval] baseline no_steer", flush=True)
        baseline_result = run_eval(model, tokenizer, data, "no_steer", args, device)
        print(
            f"[eval] baseline: F1={baseline_result['macro']['F1']:.3f} "
            f"F0.5={baseline_result['macro']['F_0.5']:.3f} "
            f"EU={baseline_result['macro']['EU']:.3f} "
            f"Exact={baseline_result['macro']['Exact']:.3f} "
            f"runtime={baseline_result['runtime_s']:.1f}s",
            flush=True,
        )

    # --- Install hooks and run F12 eval ---
    # Build a dummy cfg for install_hooks
    cfg = F12Config(
        rot_pairs=int(ck_cfg["rot_pairs"]),
        steered_layers=tuple(int(x) for x in ck_cfg["steered_layers"]),
        schedule=str(ck_cfg["schedule"]),
        skip_layer_28=bool(ck_cfg["skip_layer_28"]),
    )
    handles = install_hooks(
        model, cfg, k_subs, q_subs, rotation,
        n_kv, n_q, head_dim, s_K, s_Q,
    )
    method_name = f"f12_{ck_cfg['schedule']}_r{2*ck_cfg['rot_pairs']}"
    print(f"\n[eval] {method_name} (hooks={len(handles)})", flush=True)
    try:
        f12_result = run_eval(model, tokenizer, data, method_name, args, device)
    finally:
        for h in handles:
            h.remove()
    print(
        f"[eval] {method_name}: F1={f12_result['macro']['F1']:.3f} "
        f"F0.5={f12_result['macro']['F_0.5']:.3f} "
        f"EU={f12_result['macro']['EU']:.3f} "
        f"Exact={f12_result['macro']['Exact']:.3f} "
        f"runtime={f12_result['runtime_s']:.1f}s",
        flush=True,
    )

    # --- save ---
    out_payload = {
        "model": args.model,
        "dataset": args.dataset,
        "n_queries": len(data),
        "start_idx": args.start_idx,
        "max_samples": args.max_samples,
        "method": method_name,
        "macro": f12_result["macro"],
        "stepwise": f12_result["stepwise"],
        "runtime_s": f12_result["runtime_s"],
        "baseline_macro": baseline_result["macro"] if baseline_result else None,
        "baseline_stepwise": baseline_result["stepwise"] if baseline_result else None,
        "baseline_runtime_s": baseline_result["runtime_s"] if baseline_result else None,
        "per_sample": f12_result["per_sample"],
        "baseline_per_sample": baseline_result["per_sample"] if baseline_result else None,
        "checkpoint_config": ck_cfg,
        "checkpoint_path": str(args.checkpoint),
        "subspace_path": str(subspace_path),
        "decode_policy": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        },
        "runtime_config": {
            "device": args.device,
            "dtype": str(dtype),
            "seed": args.seed,
            "attn_implementation": "eager",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
