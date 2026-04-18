#!/usr/bin/env python3
"""Phase D first-cut — d* empirical extraction + H1/H3 angular comparison.

Scope (pragmatic first-cut, not full 18-week Phase D plan):
    Step 0 — Extract empirical d*_emp(q) per-(layer, head, query) from
             tau2-bench tasks. Definition:
                d_emp[L, h, q] := mean(K[L, h, pos_GT(q)])
                                − mean(K[L, h, pos_distractor(q)])
             Normalized to unit vector.
    Step 1 — H1 (lm_head-based) hypothesis:
                d_H1[L, h, q] := W_K[L, h]^T · (avg_emb_GT(q) − avg_emb_distr(q))
             where avg_emb_* are mean lm_head rows for the GT / distractor
             tool-name tokens.  Normalized.
    Step 3 — H3 (W_K top singular vector):
                d_H3[L, h] := top-1 left singular vector of W_K[L, h]
             Data-free, query-independent.

    Angular comparison: cos(d_emp, d_Hx) per-(L, h, q).  Aggregate to
    median + quantile distribution across queries.

Skipped (future work): H2 (OV readout), H4 (RoPE), H5 (head-class),
H6 (catalog-position).

Output:
    reports/new_theorem_test/phase_d/d_star_<model>_<bench>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_tau2_bench import (
    build_chat_prompt_marked,
    build_tools_json,
    extract_domain_tools,
)
from measure_lemma_empirical import parse_layers_spec


def locate_tool_token_spans(prompt: str, tokenizer, tool_names: List[str]):
    """Return dict tool_name -> list of token index lists (one per occurrence)."""
    enc = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt",
                   add_special_tokens=False)
    offsets = enc["offset_mapping"][0].tolist()
    out = {name: [] for name in tool_names}
    for name in tool_names:
        for m in re.finditer(re.escape(name), prompt):
            a, b = m.start(), m.end()
            tok_ids = []
            for tid, (cs, ce) in enumerate(offsets):
                if cs >= a and ce <= b and not (cs == ce == 0):
                    tok_ids.append(tid)
            if tok_ids:
                out[name].append(tok_ids)
    return out, enc["input_ids"]


@torch.no_grad()
def run_one_task(model, tok, task, tools_json, sel_layers,
                 n_kv, head_dim, domain_tools, lm_head_w, W_K_per_layer):
    """Return per-(L, h) dicts with d_emp, d_H1, d_H3 and cosines."""
    gt_tools = sorted({a["name"] for a in task.get("evaluation_criteria", {}).get("actions", [])})
    distractors = [t for t in domain_tools if t not in set(gt_tools)]
    if not gt_tools or not distractors:
        return None

    prompt_marked = build_chat_prompt_marked(tok, task, tools_json)
    # strip marker for clean tokenization to find positions
    plain = prompt_marked.replace("**", "")
    spans, ids = locate_tool_token_spans(plain, tok, gt_tools + distractors)

    gt_positions = []
    for name in gt_tools:
        for occ in spans.get(name, []):
            gt_positions.extend(occ)
    distr_positions = []
    for name in distractors:
        for occ in spans.get(name, []):
            distr_positions.extend(occ)
    if not gt_positions or not distr_positions:
        return None

    device = next(model.parameters()).device
    ids = ids.to(device)

    # Capture k_norm/k_proj output at sel_layers
    captures: Dict[int, torch.Tensor] = {}
    def make_capture(i_layer, L):
        def _hook(mod, inputs, output):
            out = output
            if out.dim() == 3:
                B, T, D = out.shape
                if D != n_kv * head_dim:
                    return
                out_view = out.view(B, T, n_kv, head_dim)
            elif out.dim() == 4:
                out_view = out
            else:
                return
            captures[i_layer] = out_view[0].detach().float()  # (T, n_kv, head_dim)
        return _hook

    handles = []
    for i, L in enumerate(sel_layers):
        attn = model.model.layers[L].self_attn
        module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
        handles.append(module.register_forward_hook(make_capture(i, L)))
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # Compute lm_head-based avg emb for GT tokens (first token of each tool name)
    gt_tok_ids = set()
    for name in gt_tools:
        tok_list = tok(name, add_special_tokens=False)["input_ids"]
        if tok_list:
            gt_tok_ids.add(tok_list[0])
    distr_tok_ids = set()
    for name in distractors:
        tok_list = tok(name, add_special_tokens=False)["input_ids"]
        if tok_list:
            distr_tok_ids.add(tok_list[0])
    if not gt_tok_ids or not distr_tok_ids:
        return None
    avg_emb_gt = lm_head_w[list(gt_tok_ids)].mean(dim=0)      # (d_model,)
    avg_emb_distr = lm_head_w[list(distr_tok_ids)].mean(dim=0)  # (d_model,)
    emb_diff = (avg_emb_gt - avg_emb_distr).float()             # (d_model,)

    # For each (layer, head): compute d_emp, d_H1, d_H3 and angles
    per_layer_head = {}
    for i, L in enumerate(sel_layers):
        K = captures[i]  # (T, n_kv, head_dim)
        W_K = W_K_per_layer[L]  # (n_kv, head_dim, d_model)
        for h in range(n_kv):
            K_gt = K[gt_positions, h, :].mean(dim=0)       # (head_dim,)
            K_distr = K[distr_positions, h, :].mean(dim=0)
            d_emp = K_gt - K_distr
            d_emp_norm = d_emp / (d_emp.norm() + 1e-8)

            d_H1 = W_K[h] @ emb_diff.to(W_K[h].device)     # (head_dim,)
            d_H1 = d_H1.to(d_emp.device)
            d_H1_norm = d_H1 / (d_H1.norm() + 1e-8)

            # H3: top-1 left SV of W_K[L, h]
            # W_K[h] has shape (head_dim, d_model); left SV is in head_dim
            try:
                U, _, _ = torch.linalg.svd(W_K[h].float(), full_matrices=False)
                d_H3 = U[:, 0].to(d_emp.device)
            except Exception:
                d_H3 = torch.zeros_like(d_emp)
            d_H3_norm = d_H3 / (d_H3.norm() + 1e-8)

            cos_H1 = float((d_emp_norm @ d_H1_norm).item())
            cos_H3 = float((d_emp_norm @ d_H3_norm).item())
            per_layer_head[(L, h)] = {
                "cos_H1": cos_H1,
                "cos_H3": cos_H3,
                "d_emp_norm": float(d_emp.norm().item()),
                "d_H1_norm": float(d_H1.norm().item()),
                "d_H3_norm": float(d_H3.norm().item()),
            }
    return per_layer_head, {"gt_tools": gt_tools, "n_distr": len(distractors),
                            "n_gt_pos": len(gt_positions), "n_distr_pos": len(distr_positions)}


def quantile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    idx = int(round(q * (len(s) - 1)))
    return s[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", default="external/tau2-bench/data/tau2/domains/telecom/tasks.json")
    p.add_argument("--layers-spec", default="last10")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to(args.device)
    model.eval()

    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_q
    d_model = cfg.hidden_size
    L_total = len(model.model.layers)
    sel_layers = parse_layers_spec(args.layers_spec, L_total)
    print(f"[config] n_q={n_q} n_kv={n_kv} d={head_dim} L_total={L_total} d_model={d_model}")

    # lm_head weights
    lm_head_w = model.lm_head.weight.detach().float().to(args.device)  # (V, d_model)

    # W_K per layer (reshape to (n_kv, head_dim, d_model))
    W_K_per_layer = {}
    for L in sel_layers:
        attn = model.model.layers[L].self_attn
        w = attn.k_proj.weight.detach().float()  # (n_kv * head_dim, d_model)
        W_K_per_layer[L] = w.view(n_kv, head_dim, d_model).to(args.device)
    print(f"[W_K] captured for {len(sel_layers)} layers, shape per head = ({head_dim}, {d_model})")

    tasks_all = json.load(open(args.dataset))
    tasks = tasks_all[: args.n]
    domain_tools = extract_domain_tools(tasks_all)
    tools_json = build_tools_json(domain_tools)
    print(f"[tasks] N={len(tasks)} (of {len(tasks_all)}); n_tools={len(domain_tools)}")

    per_task_records = []
    # aggregate: per-(L, h) list of cos_H1, cos_H3 values
    agg: Dict[Tuple[int, int], Dict[str, List[float]]] = {
        (L, h): {"cos_H1": [], "cos_H3": []} for L in sel_layers for h in range(n_kv)
    }
    t0 = time.time()
    n_ok = 0
    for idx, task in enumerate(tasks):
        try:
            out = run_one_task(model, tok, task, tools_json, sel_layers, n_kv,
                               head_dim, domain_tools, lm_head_w, W_K_per_layer)
        except Exception as exc:
            print(f"  [{idx+1}/{len(tasks)}] ERR {type(exc).__name__}: {exc}", flush=True)
            continue
        if out is None:
            continue
        per_lh, info = out
        for (L, h), d in per_lh.items():
            agg[(L, h)]["cos_H1"].append(d["cos_H1"])
            agg[(L, h)]["cos_H3"].append(d["cos_H3"])
        per_task_records.append({"idx": idx, "task_id": task.get("id", f"t{idx}"),
                                  "info": info,
                                  "per_lh": {f"{L}_{h}": d for (L, h), d in per_lh.items()}})
        n_ok += 1
        if (idx + 1) % 5 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(tasks) - idx - 1)
            # Quick aggregate
            all_H1 = [v for d in agg.values() for v in d["cos_H1"]]
            all_H3 = [v for d in agg.values() for v in d["cos_H3"]]
            print(f"  [{idx+1}/{len(tasks)}] t={elapsed:.1f}s eta={eta:.1f}s  n_ok={n_ok}  "
                  f"H1 median={quantile(all_H1, 0.5):.4f}  "
                  f"H3 median={quantile(all_H3, 0.5):.4f}", flush=True)

    if not per_task_records:
        raise RuntimeError("no tasks produced records")

    # Aggregate per-(L, h)
    per_lh_out = {}
    all_H1, all_H3 = [], []
    for (L, h), d in agg.items():
        if not d["cos_H1"]:
            continue
        per_lh_out[f"{L}_{h}"] = {
            "n": len(d["cos_H1"]),
            "cos_H1_mean": statistics.mean(d["cos_H1"]),
            "cos_H1_abs_mean": statistics.mean(abs(v) for v in d["cos_H1"]),
            "cos_H1_median": statistics.median(d["cos_H1"]),
            "cos_H3_mean": statistics.mean(d["cos_H3"]),
            "cos_H3_abs_mean": statistics.mean(abs(v) for v in d["cos_H3"]),
            "cos_H3_median": statistics.median(d["cos_H3"]),
        }
        all_H1.extend(d["cos_H1"])
        all_H3.extend(d["cos_H3"])

    abs_H1 = [abs(v) for v in all_H1]
    abs_H3 = [abs(v) for v in all_H3]

    # Random baseline for d=head_dim: |cos| ~ 1/sqrt(d). For head_dim=128, ~0.088.
    import math as m
    rand_ref = 1.0 / m.sqrt(head_dim)

    overall = {
        "n_angles_total": len(all_H1),
        "cos_H1_mean": statistics.mean(all_H1),
        "cos_H1_abs_mean": statistics.mean(abs_H1),
        "cos_H1_abs_median": statistics.median(abs_H1),
        "cos_H1_abs_q75": quantile(abs_H1, 0.75),
        "cos_H1_abs_q90": quantile(abs_H1, 0.90),
        "cos_H3_mean": statistics.mean(all_H3),
        "cos_H3_abs_mean": statistics.mean(abs_H3),
        "cos_H3_abs_median": statistics.median(abs_H3),
        "cos_H3_abs_q75": quantile(abs_H3, 0.75),
        "cos_H3_abs_q90": quantile(abs_H3, 0.90),
        "random_cos_abs_ref": rand_ref,
        "H1_over_random_ratio": statistics.mean(abs_H1) / rand_ref,
        "H3_over_random_ratio": statistics.mean(abs_H3) / rand_ref,
    }

    print(f"\n========== Overall summary (all (L, h, q)) ==========")
    print(f"  N total angles: {overall['n_angles_total']}  (tasks × L × h)")
    print(f"  Random baseline |cos|: {rand_ref:.4f} = 1/√{head_dim}")
    print(f"\n  H1 (lm_head-based):")
    print(f"    mean cos       = {overall['cos_H1_mean']:.4f}")
    print(f"    |cos| mean     = {overall['cos_H1_abs_mean']:.4f}")
    print(f"    |cos| median   = {overall['cos_H1_abs_median']:.4f}")
    print(f"    |cos| q75      = {overall['cos_H1_abs_q75']:.4f}")
    print(f"    |cos| q90      = {overall['cos_H1_abs_q90']:.4f}")
    print(f"    H1/random ratio = {overall['H1_over_random_ratio']:.2f}×")
    print(f"\n  H3 (W_K top-SV):")
    print(f"    mean cos       = {overall['cos_H3_mean']:.4f}")
    print(f"    |cos| mean     = {overall['cos_H3_abs_mean']:.4f}")
    print(f"    |cos| median   = {overall['cos_H3_abs_median']:.4f}")
    print(f"    |cos| q75      = {overall['cos_H3_abs_q75']:.4f}")
    print(f"    |cos| q90      = {overall['cos_H3_abs_q90']:.4f}")
    print(f"    H3/random ratio = {overall['H3_over_random_ratio']:.2f}×")

    # Pass/fail per handoff narrowing criterion: angular ≤ 30° → |cos| ≥ 0.866
    # angular ≤ 15° → |cos| ≥ 0.966
    h1_pass_15 = sum(1 for v in abs_H1 if v >= 0.966) / len(abs_H1)
    h1_pass_30 = sum(1 for v in abs_H1 if v >= 0.866) / len(abs_H1)
    h3_pass_15 = sum(1 for v in abs_H3 if v >= 0.966) / len(abs_H3)
    h3_pass_30 = sum(1 for v in abs_H3 if v >= 0.866) / len(abs_H3)
    print(f"\n  Pass rates (per-(L, h, q) angular threshold):")
    print(f"    H1:  ≤ 15° (|cos|≥0.966) = {h1_pass_15*100:5.1f}%   ≤ 30° (|cos|≥0.866) = {h1_pass_30*100:5.1f}%")
    print(f"    H3:  ≤ 15°                = {h3_pass_15*100:5.1f}%   ≤ 30°                = {h3_pass_30*100:5.1f}%")

    out = {
        "args": vars(args),
        "head_dim": head_dim,
        "n_kv": n_kv,
        "sel_layers": sel_layers,
        "n_tasks_used": len(per_task_records),
        "overall": overall,
        "per_lh": per_lh_out,
        "pass_rates": {
            "H1_angle_le_15": h1_pass_15, "H1_angle_le_30": h1_pass_30,
            "H3_angle_le_15": h3_pass_15, "H3_angle_le_30": h3_pass_30,
        },
        "per_task": per_task_records,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
