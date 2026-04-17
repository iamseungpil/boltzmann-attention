#!/usr/bin/env python3
"""
β* logit-lens discriminative-G variant.

Motivation. Schema-G (measure_beta_star.py --gt-mode schema) fails on τ² Telecom:
sign-agreement 31.2% on N=20 smoke (locked). The failure is consistent with the
schema-G proxy being weakly aligned with the attention mass the model actually
spends on GT during generation. Theorem β* itself is G-agnostic (the softmax
gradient identity holds for any G), so redefining G to a proxy more correlated
with end-task attention should restore predictor validity without touching the
proof.

Definition. For each prompt, capture per-layer residual-stream hidden states.
Apply the final RMSNorm + lm_head (logit-lens projection) to produce
per-position next-token logits at every layer. Score each position t by the
sum of layer-wise logits at the GT tool's first-BPE token IDs. Aggregate
across layers via --layer-agg (mean / last_half / top3) and define G as the
top-k positions by this discriminative score (k = max(top-k-min,
top-k-ratio · T)). Rank-based top-k is invariant to any monotone rescaling.

This script extends measure_beta_star.py rather than replacing it: all the
QKCapture / compute_s_query / empirical_best_sign machinery is reused via
direct import from the sibling module.

Usage
-----
    python scripts/ocq/measure_beta_star_logit.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --device cuda:0 \
        --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt \
        --q-sign-json reports/tau2_2026_04_17/telecom_N200.json \
        --domain telecom \
        --top-k-ratio 0.1 \
        --max-samples 20 \
        --out reports/beta_star_2026_04_18/telecom_logit_discriminative.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_metatool_subtask1 import resolve_dtype  # noqa: E402
from eval_tau2_bench import (  # noqa: E402
    build_chat_prompt,
    build_tools_json,
    extract_domain_tools,
    extract_gt_tools,
)
from measure_beta_star import (  # noqa: E402
    QKCapture,
    compute_s_query,
    empirical_best_sign,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--b-ont", type=str, required=True)
    p.add_argument("--q-sign-json", type=str, required=True)
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--domain", required=True,
                   choices=["retail", "airline", "telecom", "banking_knowledge"])
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--tie-threshold", type=float, default=0.0)
    p.add_argument("--pos-method-patterns", nargs="+",
                   default=["ocq_qbias_b0.03", "ocq_qbias_b0.05"])
    p.add_argument("--neg-method-patterns", nargs="+",
                   default=["ocq_qbias_b-0.03", "ocq_qbias_b-0.05"])
    p.add_argument("--top-k-ratio", type=float, default=0.10,
                   help="Fraction of prompt tokens to select as G (min 1).")
    p.add_argument("--top-k-min", type=int, default=1,
                   help="Absolute lower bound on |G|.")
    p.add_argument("--layer-agg", default="mean",
                   choices=["mean", "last_half", "top3"],
                   help="How to aggregate per-layer logit-lens scores: mean="
                        "arithmetic mean across all layers; last_half=mean over "
                        "the last L/2 layers; top3=sum of the three highest "
                        "per-layer scores per position.")
    p.add_argument("--aggregation", default="all_positions",
                   choices=["last", "all_positions"])
    p.add_argument("--out", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def gt_first_token_ids(tokenizer, gt_tool_names: List[str]) -> List[int]:
    """Return the set of first BPE token IDs for each GT tool name.

    Qwen and Llama handle leading-space differently; we take whichever of the
    two encodings produces a valid non-special first token.
    """
    ids: List[int] = []
    for name in gt_tool_names:
        for candidate in (name, " " + name):
            enc = tokenizer.encode(candidate, add_special_tokens=False)
            if enc and enc[0] not in (tokenizer.pad_token_id,):
                ids.append(enc[0])
                break
    # Deduplicate
    seen = set()
    out: List[int] = []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


@torch.no_grad()
def logit_lens_scores(
    model,
    input_ids: torch.Tensor,
    gt_token_ids: List[int],
    layer_agg: str,
) -> torch.Tensor:
    """Return per-position discriminative score of shape (T,).

    For each prompt position t and each layer ℓ, compute
        logit_ℓ,t(v) = (RMSNorm(h_ℓ,t) @ W_U)_v
    for v ∈ gt_token_ids, then sum across v.  Aggregate across ℓ per the
    chosen policy.

    Assumes a LlamaForCausalLM / Qwen2ForCausalLM-style architecture:
        model.model.norm          : final RMSNorm
        model.lm_head.weight      : unembedding (V, H)
    """
    out = model(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    # hidden_states is a tuple (L+1,) of (B=1, T, H) — index 0 is embedding,
    # index ℓ for ℓ in [1, L] is the post-layer-ℓ residual.
    hs = out.hidden_states  # tuple of (1, T, H)
    W_U = model.lm_head.weight  # (V, H)
    norm = model.model.norm

    T = hs[0].shape[1]
    device = hs[0].device
    gt_ids_t = torch.tensor(gt_token_ids, dtype=torch.long, device=device)
    # Hoist: gather unembedding rows for GT IDs once, transpose to (H, |G|).
    W_U_gt_T = W_U[gt_ids_t].float().T.contiguous()

    per_layer: List[torch.Tensor] = []  # each (T,)
    n_layers = len(hs) - 1  # exclude embedding
    for layer_idx in range(1, n_layers + 1):
        h = hs[layer_idx][0]  # (T, H)
        h_norm = norm(h)  # (T, H)
        # Only need logits at gt_ids to save memory: (T, |G_ids|)
        logits_gt = torch.matmul(h_norm.float(), W_U_gt_T)
        # Sum across GT token IDs → (T,)
        per_layer.append(logits_gt.sum(dim=-1))

    stack = torch.stack(per_layer, dim=0)  # (L, T)
    if layer_agg == "mean":
        return stack.mean(dim=0)
    if layer_agg == "last_half":
        return stack[n_layers // 2:].mean(dim=0)
    if layer_agg == "top3":
        topk = torch.topk(stack, k=min(3, n_layers), dim=0).values
        return topk.sum(dim=0)
    raise ValueError(f"unknown layer_agg {layer_agg}")


def top_k_positions_to_spans(
    scores: torch.Tensor,
    k: int,
) -> List[Tuple[int, int]]:
    """Pick the top-k positions by score (z-normalized within prompt) and
    return each as a single-token span (p, p+1)."""
    T = scores.shape[0]
    k = max(1, min(k, T))
    # z-normalize just for reporting; topk works on any monotone transform.
    topk = torch.topk(scores, k=k, dim=0).indices.tolist()
    topk.sort()
    return [(p, p + 1) for p in topk]


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    # Load Q-sign JSON
    js = json.load(open(args.q_sign_json))
    per_sample_by_method: Dict[str, Dict[str, dict]] = {}
    for r in js["results"]:
        per_sample_by_method[r["method"]] = {s["task_id"]: s for s in r["per_sample"]}
    if "no_steer" not in per_sample_by_method:
        raise ValueError("no_steer baseline missing")
    task_ids = list(per_sample_by_method["no_steer"].keys())

    dataset_path = args.dataset or str(
        REPO / "external" / "tau2-bench" / "data" / "tau2" / "domains" / args.domain / "tasks.json"
    )
    tasks_list = json.load(open(dataset_path))
    tasks_by_id = {t["id"]: t for t in tasks_list}
    tasks = [tasks_by_id[tid] for tid in task_ids if tid in tasks_by_id]
    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]
    print(f"[data] {len(tasks)} tasks to measure", flush=True)

    domain_tools = extract_domain_tools(tasks_list)
    tools_json = build_tools_json(domain_tools)

    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = resolve_dtype(args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()
    cfg = model.config
    n_kv = cfg.num_key_value_heads
    n_q = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    L = cfg.num_hidden_layers
    print(f"[load] done in {time.time()-t0:.1f}s; L={L} n_kv={n_kv} n_q={n_q} d={head_dim}",
          flush=True)

    payload = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = payload["B_ont"] if isinstance(payload, dict) else payload
    assert B_ont.shape[0] == L and B_ont.shape[1] == n_kv and B_ont.shape[2] == head_dim, \
        f"B_ont shape {tuple(B_ont.shape)} mismatches model ({L},{n_kv},{head_dim},*)"
    print(f"[ocq] B_ont shape {tuple(B_ont.shape)}", flush=True)

    cap = QKCapture(L, n_kv, n_q, head_dim)
    cap.install(model)

    records = []
    n_match = 0
    n_valid = 0
    t0 = time.time()
    with torch.no_grad():
        for i, task in enumerate(tasks):
            prompt = build_chat_prompt(tok, task, tools_json)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
            # Reset capture buffers
            for li in range(L):
                cap.Q[li] = None
                cap.K[li] = None

            gt_tools = extract_gt_tools(task)
            gt_first_ids = gt_first_token_ids(tok, gt_tools)
            if not gt_first_ids:
                # No GT tokens encodable — skip
                continue

            # Logit-lens: this forward pass also populates Q, K capture.
            scores = logit_lens_scores(model, ids["input_ids"], gt_first_ids, args.layer_agg)
            T = scores.shape[0]
            k_abs = max(args.top_k_min, int(round(args.top_k_ratio * T)))
            gt_spans = top_k_positions_to_spans(scores, k_abs)

            measurement = compute_s_query(
                cap.Q, cap.K, B_ont, gt_spans,
                n_kv=n_kv, n_q=n_q, head_dim=head_dim,
                aggregation=args.aggregation,
            )
            s_val = measurement["s"]
            pred_sign = "+" if s_val > 0 else ("-" if s_val < 0 else "0")

            emp_sign, f1_pos, f1_neg = empirical_best_sign(
                per_sample_by_method, task["id"],
                args.pos_method_patterns, args.neg_method_patterns,
                args.tie_threshold,
            )
            match = None
            if emp_sign in ("+", "-") and pred_sign in ("+", "-"):
                match = emp_sign == pred_sign
                if match:
                    n_match += 1
                n_valid += 1

            records.append({
                "task_id": task["id"],
                "gt_tools": gt_tools,
                "gt_first_token_ids": gt_first_ids,
                "T": T,
                "k_abs": k_abs,
                "n_gt_spans": len(gt_spans),
                "gt_span_positions": [a for (a, b) in gt_spans],
                "s": s_val,
                "s_per_layer": measurement["s_per_layer"],
                "pi_G_mean": measurement["pi_G_mean"],
                "r_bar_mean": measurement["r_bar_mean"],
                "r_bar_G_mean": measurement["r_bar_G_mean"],
                "pred_sign": pred_sign,
                "emp_sign": emp_sign,
                "f1_pos": f1_pos,
                "f1_neg": f1_neg,
                "match": match,
            })
            if (i + 1) % 10 == 0 or i == len(tasks) - 1:
                acc = (n_match / n_valid) if n_valid else float("nan")
                print(f"  [{i+1}/{len(tasks)}]  match_acc={acc:.3f}  valid={n_valid}",
                      flush=True)

    cap.remove()
    runtime = time.time() - t0
    accuracy = (n_match / n_valid) if n_valid else float("nan")
    out_payload = {
        "model": args.model,
        "domain": args.domain,
        "b_ont_shape": list(B_ont.shape),
        "q_sign_json": args.q_sign_json,
        "gt_mode": "logit_discriminative",
        "top_k_ratio": args.top_k_ratio,
        "top_k_min": args.top_k_min,
        "layer_agg": args.layer_agg,
        "aggregation": args.aggregation,
        "n_tasks": len(records),
        "n_valid": n_valid,
        "n_match": n_match,
        "sign_accuracy": accuracy,
        "tie_threshold": args.tie_threshold,
        "pos_methods": args.pos_method_patterns,
        "neg_methods": args.neg_method_patterns,
        "runtime_s": runtime,
        "records": records,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out_payload, indent=2))
        print(f"[save] {args.out}", flush=True)

    print(
        f"\n=== β* logit-lens sign prediction ({args.domain}) ===\n"
        f"  tasks={len(records)}  valid={n_valid}  match={n_match}\n"
        f"  sign-accuracy = {accuracy:.3f}  "
        f"(layer_agg={args.layer_agg}, top_k_ratio={args.top_k_ratio})\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
