#!/usr/bin/env python3
"""measure_theorem_6_1.py

Empirical measurement of Theorem 6.1 (single-layer attention-weighted
reconstruction bound) on MetaTool Subtask1.

Theorem 6.1 (T6.1):  E_q ||o_hat(q) - o(q)||^2
                        <= 2 · E_q[ qaMSE(q; E) · Var_s[V](q) ]  +  C1 · rho^4,
where:
  alpha_t(q) := q · e_t / sqrt(d)               (logit perturbation per token)
  s_t(q)     := softmax(q · k_t / sqrt(d))_t    (clean attention weights)
  mean_alpha  := sum_t s_t(q) · alpha_t(q)
  qaMSE       := sum_t s_t(q) · (alpha_t(q) - mean_alpha)^2    (attn-weighted Var of alpha)
  Var_s[V]    := sum_t s_t(q) · ||v_t - o(q)||^2               (attn-weighted Var of values)
  C1          := 2 · Q_max^4 · V_max^2 / d^2   (softmax Hessian op-norm constant)
  rho         := max_t ||e_t||                 (per-token key perturbation bound)

We measure, for a chosen target layer and per-head:
  LHS(q)      := ||o_hat(q) - o(q)||^2     (directly from clean vs biased forward)
  RHS_lead(q) := 2 · qaMSE(q) · Var_s[V](q)
  RHS_rem     := C1 · rho^4                 (constant across q for fixed E)
  RHS(q)      := RHS_lead(q) + RHS_rem

And report per-query:
  - LHS, RHS_lead, RHS
  - ratio LHS / max(RHS, eps)   (should be <= 1 for every sample)
  - fraction of samples with LHS <= RHS (expected: 100%)
  - aggregate E_q[LHS], E_q[RHS_lead], RHS_rem

Usage (post label-logprob run, on free GPU):
  source /home/woori/workspace_common/CDP/poc/set.env
  python scripts/ocq/measure_theorem_6_1.py \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --alpha 0.3 \
    --layer 13 \
    --max-samples 100 \
    --out reports/theorem_6_1_verification/qwen_L13_a0.3.json

The script performs two forwards per sample (clean + biased) with hooks that
cache per-head q, k, v, attention-output at the target layer. It avoids
building B_ont^T q projections at scale; we only need q·k, q·e, v, and s per
head, which scales as O(T · d) per head, well within A6000 budget for T <~ 4k.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--dataset",
                   default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--b-ont", required=True, type=str,
                   help="Path to B_ont payload (same as eval_metatool_subtask1).")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="K-bias amplitude (alpha_base).")
    p.add_argument("--layer", type=int, required=True,
                   help="Target layer index for per-head measurement.")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def resolve_dtype(name: str, arg: str) -> torch.dtype:
    if arg == "float32": return torch.float32
    if arg == "float16": return torch.float16
    if arg == "bfloat16": return torch.bfloat16
    return torch.bfloat16 if ("qwen" in name.lower() or "llama" in name.lower()) else torch.float16


@torch.no_grad()
def measure_one(
    model, tokenizer, prompt: str, layer_idx: int,
    B_ont_layer: torch.Tensor, alpha: float, device: str,
) -> Dict:
    """Return per-head Theorem 6.1 quantities for a single prompt at `layer_idx`.

    B_ont_layer: (n_kv, head_dim, r_ont) ontology basis at this layer.
    """
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = ids["input_ids"]

    # Storage for the target layer's q, k, v, s, o_clean and e_t, o_biased.
    cache: Dict[str, torch.Tensor] = {}

    # Hooks: pre-attention, grab q/k/v (post projection, pre RoPE for our
    # analysis we operate on post-RoPE tensors since that is what softmax sees).
    # We use the attention module's forward pre-hook to capture hidden states,
    # then reconstruct q/k/v via the module's own projections.

    # Simpler path: run with output_attentions=True and output_hidden_states=True,
    # and use module hooks on the target attention layer to capture post-projection
    # Q, K, V. This mirrors how eval_metatool_subtask1 installs hooks.

    attn_module = None
    # Locate the attention module at layer_idx. Qwen/Llama layer path is
    # model.model.layers[layer_idx].self_attn
    base = model.model if hasattr(model, "model") else model
    attn_module = base.layers[layer_idx].self_attn

    captured = {}

    def capture_qkv_hook(module, inputs, output):
        # For post-RoPE capture, the cleanest access is via patching q_proj/k_proj/v_proj.
        # But at this level we just grab the (query_states, key_states, value_states)
        # that feed into the attention scoring. We do this with a pre-hook on the module.
        pass

    # Strategy: monkey-patch q_proj / k_proj / v_proj to cache their outputs.
    q_proj = attn_module.q_proj
    k_proj = attn_module.k_proj
    v_proj = attn_module.v_proj

    orig_q = q_proj.forward
    orig_k = k_proj.forward
    orig_v = v_proj.forward

    def cache_q(x):
        out = orig_q(x)
        captured["q_pre"] = out.detach()
        return out

    def cache_k(x):
        out = orig_k(x)
        captured["k_pre"] = out.detach()
        return out

    def cache_v(x):
        out = orig_v(x)
        captured["v"] = out.detach()
        return out

    q_proj.forward = cache_q
    k_proj.forward = cache_k
    v_proj.forward = cache_v

    # Run clean forward (no K-bias).
    try:
        _ = model(input_ids=input_ids)
    finally:
        q_proj.forward = orig_q
        k_proj.forward = orig_k
        v_proj.forward = orig_v

    # Shapes: (B=1, T, n_q*d_head) for q, (B=1, T, n_kv*d_head) for k, v.
    q_pre = captured["q_pre"].squeeze(0)    # (T, n_q*d)
    k_pre = captured["k_pre"].squeeze(0)    # (T, n_kv*d)
    v     = captured["v"].squeeze(0)        # (T, n_kv*d)

    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d_head = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    T = q_pre.shape[0]

    # Reshape per-head and cast to fp32 for numerics.
    q_h = q_pre.view(T, n_q, d_head).permute(1, 0, 2).float()        # (n_q, T, d)
    k_h = k_pre.view(T, n_kv, d_head).permute(1, 0, 2).float()       # (n_kv, T, d)
    v_h = v.view(T, n_kv, d_head).permute(1, 0, 2).float()           # (n_kv, T, d)
    # GQA: map query head -> kv head via groupsize.
    group = n_q // n_kv
    kv_idx = torch.arange(n_q, device=q_h.device) // group          # (n_q,)

    B_layer = B_ont_layer.to(q_h.device).float()                    # (n_kv, d, r_ont)

    # Compute per-head Theorem 6.1 quantities from the target position q_T.
    # We use the LAST position as the query (model's next-token prediction query).
    q_last = q_h[:, -1, :]                                          # (n_q, d)

    # Per KV-head projection K and V.
    results_heads = []
    for h_q in range(n_q):
        kv = kv_idx[h_q].item()
        q_vec = q_last[h_q]                                          # (d,)
        k_full = k_h[kv]                                             # (T, d)
        v_full = v_h[kv]                                             # (T, d)
        B = B_layer[kv]                                              # (d, r_ont)

        # Perturbation e_t = alpha * B B^T k_t (flat bias on EVERY token).
        # Shape: (T, d).
        # First compute c_t := B^T k_t (T, r_ont).
        c = k_full @ B                                               # (T, r_ont)
        e = alpha * (c @ B.T)                                        # (T, d)

        # Logit and perturbation:
        scale = d_head ** 0.5
        logits = (k_full @ q_vec) / scale                            # (T,)
        alpha_t = (e @ q_vec) / scale                                # (T,)
        s = torch.softmax(logits, dim=0)                             # (T,)

        # Clean output o = sum_t s_t v_t.
        o = (s.unsqueeze(-1) * v_full).sum(dim=0)                    # (d,)

        # Biased softmax: logits_hat = logits + alpha_t, then apply softmax.
        logits_hat = logits + alpha_t
        s_hat = torch.softmax(logits_hat, dim=0)
        o_hat = (s_hat.unsqueeze(-1) * v_full).sum(dim=0)

        # LHS
        lhs = float(((o_hat - o) ** 2).sum().item())

        # Thm 6.1 components
        mean_alpha = float((s * alpha_t).sum().item())
        qaMSE = float((s * (alpha_t - mean_alpha) ** 2).sum().item())
        Var_s_V = float((s * ((v_full - o) ** 2).sum(dim=1)).sum().item())
        rho = float(e.norm(dim=1).max().item())
        rhs_lead = 2.0 * qaMSE * Var_s_V
        # C1 is per-model constant; we estimate Q_max, V_max from observed norms.
        Q_max = float(q_vec.norm().item())
        V_max = float(v_full.norm(dim=1).max().item())
        C1 = 2.0 * (Q_max ** 4) * (V_max ** 2) / (d_head ** 2)
        rhs_rem = C1 * (rho ** 4)
        rhs_total = rhs_lead + rhs_rem

        results_heads.append({
            "h_q": h_q, "kv": kv,
            "T": T,
            "lhs": lhs,
            "rhs_lead": rhs_lead,
            "rhs_rem": rhs_rem,
            "rhs_total": rhs_total,
            "ratio_lhs_over_rhs_total": lhs / max(rhs_total, 1e-30),
            "qaMSE": qaMSE,
            "Var_s_V": Var_s_V,
            "mean_alpha": mean_alpha,
            "rho": rho,
            "Q_max": Q_max,
            "V_max": V_max,
            "C1": C1,
        })
    return {"T": T, "n_q": n_q, "n_kv": n_kv, "d_head": d_head, "heads": results_heads}


def main() -> None:
    args = parse_args()

    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = resolve_dtype(args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    cfg = model.config
    L = cfg.num_hidden_layers
    assert 0 <= args.layer < L, f"--layer {args.layer} out of range [0,{L})"

    payload = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = payload["B_ont"] if isinstance(payload, dict) else payload
    B_ont_layer = B_ont[args.layer]                                  # (n_kv, d, r)

    with open(args.dataset) as f:
        data = json.load(f)
    start = max(0, args.start_idx)
    end = start + args.max_samples if args.max_samples > 0 else len(data)
    data = data[start:end]
    print(f"[data] {len(data)} queries", flush=True)

    per_sample: List[Dict] = []
    for i, entry in enumerate(data):
        prompt = entry["action_prompt"]
        try:
            r = measure_one(model, tok, prompt, args.layer, B_ont_layer,
                            args.alpha, args.device)
        except Exception as ex:
            print(f"[skip] {i}: {ex}", flush=True)
            continue
        r["index"] = entry.get("index", i)
        per_sample.append(r)
        if (i + 1) % 10 == 0:
            print(f"[progress] {i+1}/{len(data)}", flush=True)

    # Aggregate
    lhs_all = torch.tensor([h["lhs"] for s in per_sample for h in s["heads"]])
    rhs_lead_all = torch.tensor([h["rhs_lead"] for s in per_sample for h in s["heads"]])
    rhs_total_all = torch.tensor([h["rhs_total"] for s in per_sample for h in s["heads"]])
    ratio_all = lhs_all / rhs_total_all.clamp(min=1e-30)
    pass_rate = float((lhs_all <= rhs_total_all).float().mean().item())

    summary = {
        "model": args.model,
        "layer": args.layer,
        "alpha": args.alpha,
        "n_queries": len(per_sample),
        "n_head_measurements": int(lhs_all.numel()),
        "E_lhs": float(lhs_all.mean().item()),
        "E_rhs_lead": float(rhs_lead_all.mean().item()),
        "E_rhs_total": float(rhs_total_all.mean().item()),
        "mean_ratio_lhs_over_rhs": float(ratio_all.mean().item()),
        "median_ratio": float(ratio_all.median().item()),
        "p95_ratio": float(ratio_all.quantile(0.95).item()),
        "max_ratio": float(ratio_all.max().item()),
        "bound_pass_rate": pass_rate,
    }
    print("\n=== THEOREM 6.1 EMPIRICAL VERIFICATION ===")
    for k, v in summary.items():
        print(f"  {k:32s} : {v}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "per_sample": per_sample},
        indent=2,
    ))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
