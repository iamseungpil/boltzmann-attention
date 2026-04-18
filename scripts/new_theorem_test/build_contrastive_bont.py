#!/usr/bin/env python3
"""F6 — Contrastive B_ont construction (CAA-style query-pair).

For each tool T in the catalog:
    d_T[L, h] := mean(K[L, h, prompt_end] over queries where T ∈ GT)
               − mean(K[L, h, prompt_end] over queries where T ∉ GT)
Then per-(L, h):
    B_contrastive[L, h] := top-r left singular vectors of [d_T_1, ..., d_T_M]
This tests whether a STRUCTURALLY different semantic construction
(query-pair contrast at decision-time K) yields a basis with equivalent,
stronger, or weaker direction specificity than the current position-pooling
B_ont.  Phase C tested only content permutation within position pooling;
F6 tests a different pipeline entirely.

Also produces a shuffled-label control: for each tool, randomly permutes
the with_T / without_T labels (set sizes preserved), rebuilds d_T, and
constructs B_contrastive_shuffled.  If semantic contrast is load-bearing,
shuffled → attn_fro ≈ 1.  If not, shuffled still > 1.5.

Output: B_ont.pt artifacts (real + shuffled) + partition JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_tau2_bench import (
    build_chat_prompt,
    build_tools_json,
    extract_domain_tools,
)


@torch.no_grad()
def capture_prompt_end_K(model, tok, prompt_text, sel_layers, n_kv, head_dim):
    """Forward pass, return K at prompt-end for each sel_layer, shape (L_sel, n_kv, head_dim)."""
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(next(model.parameters()).device)
    caps = {}
    def make_hook(i_layer):
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
            caps[i_layer] = out_view[0, -1, :, :].detach().float().cpu()  # (n_kv, head_dim)
        return _hook

    handles = []
    for i, L in enumerate(sel_layers):
        attn = model.model.layers[L].self_attn
        module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
        handles.append(module.register_forward_hook(make_hook(i)))
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if len(caps) != len(sel_layers):
        return None
    stacked = torch.stack([caps[i] for i in range(len(sel_layers))])  # (L_sel, n_kv, head_dim)
    return stacked


def build_bont_from_directions(d_per_tool, sel_layers, n_kv, head_dim, r):
    """Stack tool directions per-(L, h), SVD, keep top-r left singular vectors."""
    L_sel = len(sel_layers)
    M = len(d_per_tool)
    tool_order = list(d_per_tool.keys())
    B = torch.zeros(L_sel, n_kv, head_dim, r)
    per_lh_info = []
    for i in range(L_sel):
        for h in range(n_kv):
            D = torch.stack([d_per_tool[T][i, h, :] for T in tool_order], dim=1)  # (head_dim, M)
            U, S, _ = torch.linalg.svd(D, full_matrices=False)
            r_eff = min(r, U.shape[1])
            B[i, h, :, :r_eff] = U[:, :r_eff]
            per_lh_info.append({
                "layer_idx": i,
                "head": h,
                "singular_values": S.tolist()[:r_eff],
                "r_eff": r_eff,
            })
    return B, per_lh_info, tool_order


def compute_tool_directions(K_cache, partition, tools):
    """Return dict tool → (L_sel, n_kv, head_dim) direction tensor."""
    d_per_tool = {}
    for T in tools:
        with_q = partition[T]["with"]
        without_q = partition[T]["without"]
        K_with = torch.stack([K_cache[q] for q in with_q]).mean(dim=0)
        K_without = torch.stack([K_cache[q] for q in without_q]).mean(dim=0)
        d_per_tool[T] = K_with - K_without
    return d_per_tool


def make_shuffled_partition(partition, tools, seed):
    """For each tool, combine with_T and without_T, randomly reassign while preserving set sizes."""
    rng = random.Random(seed)
    new_part = {}
    for T in tools:
        with_q = list(partition[T]["with"])
        without_q = list(partition[T]["without"])
        n_with = len(with_q)
        pool = with_q + without_q
        rng.shuffle(pool)
        new_part[T] = {"with": pool[:n_with], "without": pool[n_with:]}
    return new_part


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", default="external/tau2-bench/data/tau2/domains/telecom/tasks.json")
    p.add_argument("--sample-n", type=int, default=400, help="random tasks to sample for K extraction")
    p.add_argument("--min-per-bucket", type=int, default=15, help="min with_T and without_T count after sampling (drop tools below)")
    p.add_argument("--target-per-bucket", type=int, default=30, help="cap each bucket to this count (balance)")
    p.add_argument("--rank", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffled-seed", type=int, default=42)
    p.add_argument("--out-real", required=True)
    p.add_argument("--out-shuffled", required=True)
    p.add_argument("--out-partition", required=True)
    args = p.parse_args()

    random.seed(args.seed)

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
    L_total = len(model.model.layers)
    sel_layers = list(range(L_total))  # ALL layers (spec §1.2)
    print(f"[config] n_q={n_q} n_kv={n_kv} head_dim={head_dim} L_total={L_total}")

    tasks_all = json.load(open(args.dataset))
    domain_tools = extract_domain_tools(tasks_all)
    tools_json = build_tools_json(domain_tools)
    print(f"[data] N_tasks={len(tasks_all)}  n_tools={len(domain_tools)}")

    # --- Sample tasks ---
    sample_indices = random.sample(range(len(tasks_all)), min(args.sample_n, len(tasks_all)))
    sample_indices.sort()
    print(f"[sample] {len(sample_indices)} random tasks selected (seed={args.seed})")

    # --- Forward pass, cache K at prompt-end per (L, h) ---
    K_cache = {}
    t0 = time.time()
    for k, q_idx in enumerate(sample_indices):
        task = tasks_all[q_idx]
        try:
            prompt = build_chat_prompt(tok, task, tools_json)
            K = capture_prompt_end_K(model, tok, prompt, sel_layers, n_kv, head_dim)
            if K is None:
                continue
            K_cache[q_idx] = K  # (L_total, n_kv, head_dim)
        except Exception as exc:
            print(f"  [{k+1}/{len(sample_indices)}] ERR {type(exc).__name__}: {exc}")
            continue
        if (k + 1) % 25 == 0 or k == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(sample_indices) - k - 1)
            print(f"  [{k+1}/{len(sample_indices)}] t={elapsed:.1f}s eta={eta:.1f}s cached={len(K_cache)}", flush=True)

    print(f"[K cache] {len(K_cache)} queries captured")

    # --- Build partition from cached queries ---
    cached_idxs = sorted(K_cache.keys())
    partition_raw = {T: {"with": [], "without": []} for T in domain_tools}
    for q_idx in cached_idxs:
        gt = set(a["name"] for a in tasks_all[q_idx].get("evaluation_criteria", {}).get("actions", []))
        for T in domain_tools:
            if T in gt:
                partition_raw[T]["with"].append(q_idx)
            else:
                partition_raw[T]["without"].append(q_idx)

    eligible_tools = []
    for T in domain_tools:
        w = len(partition_raw[T]["with"])
        wo = len(partition_raw[T]["without"])
        if w >= args.min_per_bucket and wo >= args.min_per_bucket:
            eligible_tools.append(T)
        else:
            print(f"  [skip] {T:35s}  with={w}  without={wo}  (below min {args.min_per_bucket})")

    # Balance buckets (cap at target_per_bucket, then match smaller one)
    rng_balance = random.Random(args.seed)
    partition = {}
    for T in eligible_tools:
        w_list = partition_raw[T]["with"][:]
        wo_list = partition_raw[T]["without"][:]
        rng_balance.shuffle(w_list); rng_balance.shuffle(wo_list)
        target = min(args.target_per_bucket, len(w_list), len(wo_list))
        partition[T] = {
            "with": w_list[:target],
            "without": wo_list[:target],
        }
    print(f"[partition] {len(eligible_tools)} eligible tools, balanced to max {args.target_per_bucket} per bucket")
    for T in eligible_tools:
        print(f"  {T:35s}  with={len(partition[T]['with'])}  without={len(partition[T]['without'])}")

    # Save partition JSON
    os.makedirs(os.path.dirname(args.out_partition), exist_ok=True)
    with open(args.out_partition, "w") as f:
        json.dump({
            "sample_indices": sample_indices,
            "cached_indices": cached_idxs,
            "eligible_tools": eligible_tools,
            "partition_raw_counts": {T: {"with": len(partition_raw[T]["with"]),
                                         "without": len(partition_raw[T]["without"])}
                                     for T in domain_tools},
            "partition": {T: partition[T] for T in eligible_tools},
        }, f, indent=2)
    print(f"[saved partition] {args.out_partition}")

    # --- Build real contrastive B_ont ---
    print(f"\n[real] compute d_T + SVD")
    d_real = compute_tool_directions(K_cache, partition, eligible_tools)
    B_real, info_real, tool_order = build_bont_from_directions(
        d_real, sel_layers, n_kv, head_dim, args.rank
    )
    torch.save({
        "B_ont": B_real,
        "model": args.model,
        "r_ont": args.rank,
        "n_layers": L_total,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "target_layers": sel_layers,
        "facet_order": ["contrastive"],
        "construction": "contrastive_query_pair_realF6",
        "eligible_tools": eligible_tools,
        "tool_order": tool_order,
        "per_lh_info": info_real,
    }, args.out_real)
    print(f"[saved real B_ont] {args.out_real}  shape={tuple(B_real.shape)}")

    # --- Build shuffled B_ont ---
    print(f"\n[shuffled] relabel with_T/without_T (seed={args.shuffled_seed}), recompute d_T + SVD")
    part_shuf = make_shuffled_partition(partition, eligible_tools, args.shuffled_seed)
    d_shuf = compute_tool_directions(K_cache, part_shuf, eligible_tools)
    B_shuf, info_shuf, _ = build_bont_from_directions(
        d_shuf, sel_layers, n_kv, head_dim, args.rank
    )
    torch.save({
        "B_ont": B_shuf,
        "model": args.model,
        "r_ont": args.rank,
        "n_layers": L_total,
        "n_kv": n_kv,
        "head_dim": head_dim,
        "target_layers": sel_layers,
        "facet_order": ["contrastive_shuffled"],
        "construction": f"contrastive_query_pair_shuffled_seed{args.shuffled_seed}",
        "eligible_tools": eligible_tools,
        "tool_order": tool_order,
        "per_lh_info": info_shuf,
    }, args.out_shuffled)
    print(f"[saved shuffled B_ont] {args.out_shuffled}  shape={tuple(B_shuf.shape)}")


if __name__ == "__main__":
    main()
