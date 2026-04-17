#!/usr/bin/env python3
"""
Per-query measurement of s := π_G (r̄_G - r̄) for Theorem β*.

For each task in a τ²-bench q_sign_test.json (or equivalent), this script:

 1. Reconstructs the prompt (chat template with tools).
 2. Runs a single baseline forward pass with hooks capturing per-layer
    pre-RoPE Q and K outputs of q_proj / k_proj.
 3. Identifies GT token positions in the prompt as the token spans where
    GT tool names appear (from the JSON tools list).
 4. Computes r_t^{(ℓ,h)} = <P Q^{(ℓ,h)}, K_t^{(ℓ,h)}> / sqrt(d_h) at each
    (layer, head), where P = B B^T is the ontology projector.
 5. Computes the GT-attention-weighted vs all-attention-weighted gap
       s^{(ℓ,h)} = π_G^{(ℓ,h)} (r̄_G^{(ℓ,h)} - r̄^{(ℓ,h)})
    where p_0^{(ℓ,h)}(t) = softmax(Q^{(ℓ,h)} · K_t^{(ℓ,h)} / sqrt(d_h))
    evaluated at the *last prompt token's* query row.
 6. Aggregates across (layer, head) with uniform weighting (Cor β*.1 v0).
 7. Compares predicted sign(s_tot) against the empirical best sign taken
    from the per-sample F1 differences in the input JSON.
 8. Writes a summary JSON with per-query predictions and the sign-accuracy
    averaged across tasks.

Usage:
    python scripts/ocq/measure_beta_star.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --device cuda:0 \
        --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-telecom/B_ont.pt \
        --q-sign-json reports/tau2_2026_04_17/telecom_q_sign_test.json \
        --domain telecom \
        --out reports/beta_star_2026_04_17/telecom_prediction.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
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
    build_user_message,
    extract_domain_tools,
    extract_gt_tools,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--b-ont", type=str, required=True)
    p.add_argument("--q-sign-json", type=str, required=True,
                   help="Existing q_sign_test.json (τ² format). Used to pick empirical best sign per task.")
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--domain", required=True,
                   choices=["retail", "airline", "telecom", "banking_knowledge"])
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap number of tasks (0 = all present in q_sign_json).")
    p.add_argument("--tie-threshold", type=float, default=0.0,
                   help="Queries with |F1(Q+) − F1(Q−)| ≤ threshold are marked 'tie' and excluded from accuracy.")
    p.add_argument("--pos-method-patterns", nargs="+",
                   default=["ocq_qbias_b0.03", "ocq_qbias_b0.05"],
                   help="Method names counted as positive-β (used for per-task best-sign decision).")
    p.add_argument("--neg-method-patterns", nargs="+",
                   default=["ocq_qbias_b-0.03", "ocq_qbias_b-0.05"],
                   help="Method names counted as negative-β.")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# =====================================================================
# GT token-span identification inside the tokenized prompt
# =====================================================================

def find_tool_token_spans(
    prompt: str,
    tokenizer,
    gt_tool_names: List[str],
    mode: str = "schema",
) -> List[Tuple[int, int]]:
    """Find token index spans in the tokenized prompt where each GT tool
    is referenced.

    Modes:
      name   : just the tool name tokens (previously default; tiny π_G).
      schema : name + description + parameters block, i.e. the JSON object
               enclosing `"name": "<tool>"`. Matches the tool-schema-token
               interpretation of β* GT proxy (G = schema tokens of correct
               tools, where the model's attention must land to emit the
               correct tool).
    """
    if not gt_tool_names:
        return []
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    def char_span_to_token_span(c_start: int, c_end: int) -> Optional[Tuple[int, int]]:
        tok_start = None
        tok_end = None
        for i, (a, b) in enumerate(offsets):
            if a is None:
                continue
            if tok_start is None and b > c_start:
                tok_start = i
            if a < c_end:
                tok_end = i + 1
            if a >= c_end:
                break
        if tok_start is not None and tok_end is not None and tok_end > tok_start:
            return (tok_start, tok_end)
        return None

    spans: List[Tuple[int, int]] = []
    for name in gt_tool_names:
        for m in re.finditer(re.escape(name), prompt):
            c_start, c_end = m.start(), m.end()
            if mode == "name":
                span = char_span_to_token_span(c_start, c_end)
                if span:
                    spans.append(span)
            elif mode == "schema":
                # Expand outward to encompass the JSON object containing
                # `"name": "<tool>"`. Walk backwards to find the enclosing
                # `{` and forwards to match its closing `}`.
                # Start: most recent `{` before c_start.
                obj_start = prompt.rfind("{", 0, c_start)
                if obj_start < 0:
                    obj_start = c_start
                # End: match braces starting at obj_start.
                depth = 0
                obj_end = c_end
                for i in range(obj_start, len(prompt)):
                    if prompt[i] == "{":
                        depth += 1
                    elif prompt[i] == "}":
                        depth -= 1
                        if depth == 0:
                            obj_end = i + 1
                            break
                span = char_span_to_token_span(obj_start, obj_end)
                if span:
                    spans.append(span)
            else:
                raise ValueError(f"unknown mode {mode}")
    # Merge overlapping spans
    spans = sorted(set(spans))
    merged: List[Tuple[int, int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


# =====================================================================
# Hooks for Q, K capture per layer
# =====================================================================

class QKCapture:
    def __init__(self, n_layers: int, n_kv: int, n_q: int, head_dim: int):
        self.n_layers = n_layers
        self.n_kv = n_kv
        self.n_q = n_q
        self.head_dim = head_dim
        self.Q = [None] * n_layers
        self.K = [None] * n_layers
        self.handles = []

    def install(self, model):
        for li, layer in enumerate(model.model.layers):
            q_proj = layer.self_attn.q_proj
            k_proj = layer.self_attn.k_proj

            def make_q_hook(idx):
                def hook(mod, inp, out):
                    # out: (B, T, n_q * head_dim)
                    self.Q[idx] = out.detach()
                return hook

            def make_k_hook(idx):
                def hook(mod, inp, out):
                    # out: (B, T, n_kv * head_dim)
                    self.K[idx] = out.detach()
                return hook

            self.handles.append(q_proj.register_forward_hook(make_q_hook(li)))
            self.handles.append(k_proj.register_forward_hook(make_k_hook(li)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# =====================================================================
# Per-query s computation
# =====================================================================

@torch.no_grad()
def compute_s_query(
    Q_layers: List[torch.Tensor],      # list of (1, T, n_q * d)
    K_layers: List[torch.Tensor],      # list of (1, T, n_kv * d)
    B_ont: torch.Tensor,               # (L, n_kv, d, r_ont)
    gt_spans: List[Tuple[int, int]],
    n_kv: int,
    n_q: int,
    head_dim: int,
    query_pos: Optional[int] = None,   # which query row to use (default: last prompt token)
) -> Dict[str, float]:
    """Compute aggregated s across (layer, head).

    For each (ℓ, h): use the projector P^{(ℓ,h)} = B B^T, the query at
    position `query_pos` (default T-1), K at all prompt positions, and
    the attention distribution p_0 from that query row.

    Returns {'s': float, 's_per_layer': list, 'pi_G': float, 'r_bar': float,
             'r_bar_G': float, 'n_gt_tokens': int}
    """
    L = len(Q_layers)
    assert len(K_layers) == L
    # infer T
    T = Q_layers[0].shape[1]
    if query_pos is None:
        query_pos = T - 1
    # Build GT mask
    gt_mask = torch.zeros(T, dtype=torch.bool, device=Q_layers[0].device)
    for a, b in gt_spans:
        gt_mask[a:b] = True
    n_gt_tokens = int(gt_mask.sum().item())
    if n_gt_tokens == 0:
        return {"s": 0.0, "s_per_layer": [0.0] * L, "pi_G": 0.0, "r_bar": 0.0,
                "r_bar_G": 0.0, "n_gt_tokens": 0, "T": T}

    # GQA mapping: each q-head maps to kv-head h // (n_q // n_kv)
    g = n_q // n_kv

    sqrt_d = math.sqrt(head_dim)
    s_per_layer: List[float] = []
    s_total = 0.0
    pi_G_total = 0.0
    r_bar_total = 0.0
    r_bar_G_total = 0.0
    count = 0

    for li in range(L):
        Ql = Q_layers[li][0].float()   # (T, n_q * d)
        Kl = K_layers[li][0].float()   # (T, n_kv * d)
        Q_h = Ql.view(T, n_q, head_dim)           # (T, n_q, d)
        K_h = Kl.view(T, n_kv, head_dim)          # (T, n_kv, d)
        B_l = B_ont[li].to(Ql.device, dtype=torch.float32)  # (n_kv, d, r)

        s_layer = 0.0
        for h in range(n_q):
            kvh = h // g
            B_h = B_l[kvh]  # (d, r)
            q = Q_h[query_pos, h]     # (d,)
            P_q = B_h @ (B_h.t() @ q)  # (d,)  = P Q
            K_h_hd = K_h[:, kvh, :]    # (T, d)
            # Attention scores at this head: (K Q) / sqrt(d)
            scores = (K_h_hd @ q) / sqrt_d  # (T,)
            p0 = torch.softmax(scores, dim=0)  # (T,)
            # Ontology-projected score per token
            r_t = (K_h_hd @ P_q) / sqrt_d  # (T,)
            r_bar = torch.dot(p0, r_t).item()
            # GT mass and GT-weighted r
            pi_G = p0[gt_mask].sum().item()
            if pi_G <= 0.0:
                continue
            r_bar_G = (p0[gt_mask] * r_t[gt_mask]).sum().item() / pi_G
            s_head = pi_G * (r_bar_G - r_bar)
            s_layer += s_head
            pi_G_total += pi_G
            r_bar_total += r_bar
            r_bar_G_total += r_bar_G
            count += 1
        s_per_layer.append(s_layer)
        s_total += s_layer

    return {
        "s": s_total,
        "s_per_layer": s_per_layer,
        "pi_G": (pi_G_total / count) if count else 0.0,
        "r_bar": (r_bar_total / count) if count else 0.0,
        "r_bar_G": (r_bar_G_total / count) if count else 0.0,
        "n_gt_tokens": n_gt_tokens,
        "T": T,
    }


# =====================================================================
# Empirical best-sign from q_sign_json
# =====================================================================

def empirical_best_sign(
    per_sample_by_method: Dict[str, Dict[str, dict]],
    task_id: str,
    pos_methods: List[str],
    neg_methods: List[str],
    tie_threshold: float,
) -> Tuple[str, float, float]:
    """For one task_id, compute per-task F1 averaged over positive-β methods
    and over negative-β methods. Return ('+' | '-' | 'tie', f1_pos, f1_neg)."""
    pos_vals = []
    for m in pos_methods:
        if m in per_sample_by_method and task_id in per_sample_by_method[m]:
            pos_vals.append(per_sample_by_method[m][task_id]["metrics"]["F1"])
    neg_vals = []
    for m in neg_methods:
        if m in per_sample_by_method and task_id in per_sample_by_method[m]:
            neg_vals.append(per_sample_by_method[m][task_id]["metrics"]["F1"])
    f1_pos = sum(pos_vals) / len(pos_vals) if pos_vals else float("nan")
    f1_neg = sum(neg_vals) / len(neg_vals) if neg_vals else float("nan")
    if math.isnan(f1_pos) or math.isnan(f1_neg):
        return "na", f1_pos, f1_neg
    diff = f1_pos - f1_neg
    if abs(diff) <= tie_threshold:
        return "tie", f1_pos, f1_neg
    return ("+" if diff > 0 else "-"), f1_pos, f1_neg


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    torch.manual_seed(0)

    # Load Q-sign JSON to extract per-task F1 per method
    js = json.load(open(args.q_sign_json))
    results = js["results"]
    per_sample_by_method: Dict[str, Dict[str, dict]] = {}
    for r in results:
        per_sample_by_method[r["method"]] = {s["task_id"]: s for s in r["per_sample"]}
    # Only use tasks that appear in both baseline and at least one pos/neg
    baseline_method = "no_steer"
    if baseline_method not in per_sample_by_method:
        raise ValueError(f"no_steer method missing in {args.q_sign_json}")
    task_ids = list(per_sample_by_method[baseline_method].keys())

    # Load dataset (for full task objects)
    dataset_path = args.dataset or str(
        REPO / "external" / "tau2-bench" / "data" / "tau2" / "domains" / args.domain / "tasks.json"
    )
    tasks_list = json.load(open(dataset_path))
    tasks_by_id = {t["id"]: t for t in tasks_list}

    # Filter to tasks present in results
    tasks = [tasks_by_id[tid] for tid in task_ids if tid in tasks_by_id]
    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]
    print(f"[data] {len(tasks)} tasks to measure")

    # Build tools_json once per domain
    domain_tools = extract_domain_tools(tasks_list)
    tools_json = build_tools_json(domain_tools)

    # Load model + tokenizer
    print(f"[load] {args.model} on {args.device}")
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
    print(f"[load] done in {time.time()-t0:.1f}s; L={L} n_kv={n_kv} n_q={n_q} head_dim={head_dim}")

    # Load B_ont
    payload = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = payload["B_ont"] if isinstance(payload, dict) else payload
    print(f"[ocq] B_ont shape {tuple(B_ont.shape)}")
    assert B_ont.shape[0] == L and B_ont.shape[1] == n_kv and B_ont.shape[2] == head_dim

    # Install capture hooks
    cap = QKCapture(L, n_kv, n_q, head_dim)
    cap.install(model)

    # Per-query measurement
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
            _ = model(**ids, use_cache=False)

            gt_tools = extract_gt_tools(task)
            gt_spans = find_tool_token_spans(prompt, tok, gt_tools)
            measurement = compute_s_query(
                cap.Q, cap.K, B_ont, gt_spans,
                n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            )
            s_val = measurement["s"]
            pred_sign = "+" if s_val > 0 else ("-" if s_val < 0 else "0")

            # Empirical best sign from q_sign_json
            emp_sign, f1_pos, f1_neg = empirical_best_sign(
                per_sample_by_method,
                task["id"],
                args.pos_method_patterns,
                args.neg_method_patterns,
                args.tie_threshold,
            )
            match = None
            if emp_sign in ("+", "-") and pred_sign in ("+", "-"):
                match = emp_sign == pred_sign
                if match:
                    n_match += 1
                n_valid += 1

            rec = {
                "task_id": task["id"],
                "gt_tools": gt_tools,
                "n_gt_tool_spans": len(gt_spans),
                "s": s_val,
                "s_per_layer": measurement["s_per_layer"],
                "pi_G": measurement["pi_G"],
                "r_bar": measurement["r_bar"],
                "r_bar_G": measurement["r_bar_G"],
                "T": measurement["T"],
                "pred_sign": pred_sign,
                "emp_sign": emp_sign,
                "f1_pos": f1_pos,
                "f1_neg": f1_neg,
                "match": match,
            }
            records.append(rec)
            if (i + 1) % 20 == 0 or i == len(tasks) - 1:
                acc = (n_match / n_valid) if n_valid else float("nan")
                print(f"  [{i+1}/{len(tasks)}]  match_acc={acc:.3f}  valid={n_valid}")

    cap.remove()
    runtime = time.time() - t0

    accuracy = (n_match / n_valid) if n_valid else float("nan")
    out_payload = {
        "model": args.model,
        "domain": args.domain,
        "b_ont_shape": list(B_ont.shape),
        "q_sign_json": args.q_sign_json,
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
        print(f"[save] {args.out}")

    print(
        f"\n=== β* per-query sign prediction ({args.domain}) ===\n"
        f"  tasks={len(records)}  valid (non-tie, non-na)={n_valid}  match={n_match}\n"
        f"  sign-accuracy = {accuracy:.3f}\n"
    )


if __name__ == "__main__":
    main()
