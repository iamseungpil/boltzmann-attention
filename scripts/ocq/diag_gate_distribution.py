#!/usr/bin/env python3
"""
Diagnostic: measure the per-facet energy gate g_f distribution on MMLU
vs MetaTool queries. Step 1 of the Cor 6.7 root-cause analysis.

Cor 6.7 predicts `g_f = ||K·B_f||² / ||K||² ≈ 0` on non-tool queries
(phase-closure). Empirically, facet-gated K-bias degrades MMLU by
−4.8pp at α=0.3 and −10.5pp at α=1.0 — worse than flat K-bias. Code
audit passed, so the failure must live in the gate's actual value
distribution. This script dumps that distribution per (layer, facet)
for two datasets and writes JSON so Step 2 can calibrate an
alternative gate.

What it does
------------
Install a READ-ONLY forward hook on every `layer.self_attn.k_proj`
that, for each facet f, computes g_f from the post-projection K
output (exactly as install_facet_gated_hooks does internally) but
does NOT modify the output. The hook accumulates per-(layer, facet)
running sums and a per-layer histogram of Σ_f g_f, grouped by
dataset.

We stream N MMLU prompts and N MetaTool prompts through the model
(forward-only, no generation) and aggregate.

CLI
---
    source /home/woori/workspace_common/CDP/poc/set.env && \\
    python scripts/ocq/diag_gate_distribution.py \\
        --model Qwen/Qwen2.5-7B --device cuda:0 \\
        --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \\
        --n-mmlu 200 --n-metatool 200 --seed 42 \\
        --out reports/axis2_theoretical_verification/diag_gate_distribution_qwen25_7b.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from eval_metatool_subtask1 import build_facet_masks, resolve_dtype  # noqa: E402
from eval_mmlu_subset import build_prompt, load_mmlu  # noqa: E402


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--b-ont", required=True)
    p.add_argument(
        "--metatool-dataset",
        default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json",
    )
    p.add_argument("--mmlu-config", default="all")
    p.add_argument("--n-mmlu", type=int, default=200)
    p.add_argument("--n-metatool", type=int, default=200)
    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-ctx", type=int, default=3072)
    p.add_argument("--n-bins", type=int, default=50,
                   help="Histogram bins for Σ_f g_f over [0, 1+].")
    p.add_argument("--hist-max", type=float, default=1.5,
                   help="Histogram upper edge (Σ_f g_f can exceed 1 if "
                        "facet subspaces overlap, though by construction "
                        "they are disjoint here).")
    p.add_argument("--out", required=True)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------
# Gate accumulator
# ---------------------------------------------------------------------

class GateAcc:
    """Per-layer per-facet running stats, split by token-position bucket.

    Buckets: "bos" (pos==0) and "rest" (pos>=1). The Qwen2.5 attention-sink
    outlier lives at pos==0; the B_ont construction in
    ``build_qwen_metatool_b_ont.py`` already skips pos==0 when collecting
    ontology K's (``K_li = K_li[1:]`` at line 172). The diagnostic must
    therefore also separate the two buckets so we can tell whether the
    L0 ``g_{f0} ≈ 1`` pathology is a BOS-only artifact or a genuine
    non-BOS phenomenon.

    All counters are kept on CPU as float64.
    """

    BUCKETS = ("bos", "rest")

    def __init__(self, L: int, n_facets: int, n_bins: int, hist_max: float):
        self.L = L
        self.n_facets = n_facets
        self.n_bins = n_bins
        self.hist_max = hist_max
        # One set of counters per bucket
        self._s = {
            b: {
                "g_sum": torch.zeros(L, n_facets, dtype=torch.float64),
                "g_sqsum": torch.zeros(L, n_facets, dtype=torch.float64),
                "n_obs": torch.zeros(L, dtype=torch.float64),
                "sigma_sum": torch.zeros(L, dtype=torch.float64),
                "sigma_sqsum": torch.zeros(L, dtype=torch.float64),
                "hist": torch.zeros(L, n_bins, dtype=torch.float64),
            }
            for b in self.BUCKETS
        }

    def add_bucket(
        self, bucket: str, layer: int,
        g_per_f: torch.Tensor, sigma: torch.Tensor,
    ):
        # g_per_f: (n_facets, N), sigma: (N,)
        if sigma.numel() == 0:
            return
        s = self._s[bucket]
        s["g_sum"][layer] += g_per_f.sum(dim=1)
        s["g_sqsum"][layer] += (g_per_f ** 2).sum(dim=1)
        s["n_obs"][layer] += sigma.numel()
        s["sigma_sum"][layer] += sigma.sum()
        s["sigma_sqsum"][layer] += (sigma ** 2).sum()
        idx = torch.clamp(
            (sigma / self.hist_max * self.n_bins).long(),
            min=0, max=self.n_bins - 1,
        )
        s["hist"][layer] += torch.bincount(idx, minlength=self.n_bins).double()

    def _bucket_dict(self, bucket: str) -> Dict:
        s = self._s[bucket]
        n = s["n_obs"].clamp(min=1.0)
        mean = s["g_sum"] / n.unsqueeze(1)
        var = (s["g_sqsum"] / n.unsqueeze(1)) - mean ** 2
        std = var.clamp(min=0.0).sqrt()
        sigma_mean = s["sigma_sum"] / n
        sigma_var = (s["sigma_sqsum"] / n) - sigma_mean ** 2
        sigma_std = sigma_var.clamp(min=0.0).sqrt()
        n_tot = s["n_obs"].sum().clamp(min=1)
        return {
            "n_obs_per_layer": s["n_obs"].tolist(),
            "g_f_mean_per_layer": mean.tolist(),
            "g_f_std_per_layer": std.tolist(),
            "sigma_mean_per_layer": sigma_mean.tolist(),
            "sigma_std_per_layer": sigma_std.tolist(),
            "hist_per_layer": s["hist"].tolist(),
            "g_f_mean_global": (s["g_sum"].sum(dim=0) / n_tot).tolist(),
            "sigma_mean_global": float(s["sigma_sum"].sum() / n_tot),
        }

    def to_dict(self) -> Dict:
        return {
            "L": self.L,
            "n_facets": self.n_facets,
            "n_bins": self.n_bins,
            "hist_max": self.hist_max,
            "buckets": {b: self._bucket_dict(b) for b in self.BUCKETS},
        }


# ---------------------------------------------------------------------
# Read-only diagnostic hook
# ---------------------------------------------------------------------

@contextmanager
def install_diag_hooks(
    model,
    B_ont: torch.Tensor,      # (L, H, d, r)
    facet_mask: torch.Tensor, # (L, H, F, r)
    n_kv: int,
    head_dim: int,
    acc: GateAcc,
    gate_eps: float = 1e-6,
):
    handles = []
    L, H, d, r = B_ont.shape
    n_facets = facet_mask.shape[2]
    try:
        for layer_idx in range(L):
            layer = model.model.layers[layer_idx]
            k_proj = layer.self_attn.k_proj
            B_lh = B_ont[layer_idx]       # (H, d, r)
            M_lh = facet_mask[layer_idx]  # (H, F, r)

            def make_hook(li, B_lh_local, M_lh_local):
                def hook(module, inputs, output):
                    B_sz, T, D = output.shape
                    if D != n_kv * head_dim:
                        return None  # don't modify
                    K = (output.view(B_sz, T, n_kv, head_dim)
                               .permute(0, 2, 1, 3).contiguous())
                    K_f = K.float()
                    B_dev = B_lh_local.to(device=K.device, dtype=torch.float32)
                    M_dev = M_lh_local.to(device=K.device, dtype=torch.float32)
                    # coeffs: (B, H, T, r)
                    coeffs = torch.einsum("bhtd,hdr->bhtr", K_f, B_dev)
                    K_norm_sq = (K_f ** 2).sum(dim=-1) + gate_eps  # (B, H, T)

                    # Compute g_f per (batch, head, token), kept as
                    # (n_facets, B, H, T) on GPU, then split by token
                    # position bucket (pos==0 → "bos", pos>=1 → "rest")
                    # before moving to CPU float64.
                    g_bhtf = torch.empty(
                        n_facets, B_sz, n_kv, T,
                        device=K_f.device, dtype=torch.float32,
                    )
                    for f in range(n_facets):
                        mask_f = M_dev[:, f, :]  # (H, r)
                        mc = coeffs * mask_f.unsqueeze(0).unsqueeze(2)
                        gate_num = (mc ** 2).sum(dim=-1)  # (B, H, T)
                        g_bhtf[f] = gate_num / K_norm_sq

                    # BOS bucket: t == 0
                    g_bos = g_bhtf[:, :, :, 0:1]       # (F, B, H, 1)
                    g_bos_flat = g_bos.reshape(n_facets, -1).double().cpu()
                    sigma_bos_flat = g_bos_flat.sum(dim=0)
                    acc.add_bucket("bos", li, g_bos_flat, sigma_bos_flat)

                    # Rest bucket: t >= 1
                    if T > 1:
                        g_rest = g_bhtf[:, :, :, 1:]     # (F, B, H, T-1)
                        g_rest_flat = g_rest.reshape(n_facets, -1).double().cpu()
                        sigma_rest_flat = g_rest_flat.sum(dim=0)
                        acc.add_bucket("rest", li, g_rest_flat, sigma_rest_flat)
                    return None  # K unchanged
                return hook

            handles.append(k_proj.register_forward_hook(
                make_hook(layer_idx, B_lh, M_lh)
            ))
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------

def mmlu_prompts(args) -> List[str]:
    samples, fewshot = load_mmlu(
        args.mmlu_config, args.n_mmlu, args.n_shot, args.seed,
    )
    return [build_prompt(row, fewshot, args.n_shot) for row in samples]


def metatool_prompts(args) -> List[str]:
    with open(args.metatool_dataset) as f:
        data = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(data)
    data = data[:args.n_metatool]
    return [entry["action_prompt"] for entry in data]


@torch.no_grad()
def run_dataset(
    model, tokenizer, prompts: List[str], args,
    B_ont, facet_mask, n_kv, head_dim, L, n_facets, tag: str,
) -> Dict:
    acc = GateAcc(L=L, n_facets=n_facets,
                  n_bins=args.n_bins, hist_max=args.hist_max)
    t0 = time.time()
    with install_diag_hooks(
        model, B_ont, facet_mask,
        n_kv=n_kv, head_dim=head_dim, acc=acc,
    ):
        for i, prompt in enumerate(prompts):
            ids = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=args.max_ctx,
            ).to(args.device)
            model(**ids)
            if args.verbose and (i % 25 == 0 or i == len(prompts) - 1):
                print(f"  [{tag}] {i+1}/{len(prompts)}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
    out = acc.to_dict()
    out["tag"] = tag
    out["n_prompts"] = len(prompts)
    out["runtime_s"] = time.time() - t0
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[load] {args.model} on {args.device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = resolve_dtype(args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    cfg = model.config
    n_kv = cfg.num_key_value_heads
    n_q = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    L = cfg.num_hidden_layers
    print(f"[cfg] L={L} n_kv={n_kv} n_q={n_q} head_dim={head_dim}", flush=True)

    print(f"[ocq] loading B_ont from {args.b_ont}", flush=True)
    payload = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = payload["B_ont"] if isinstance(payload, dict) else payload
    if B_ont.shape[:2] != (L, n_kv):
        raise ValueError(
            f"B_ont (L,H)={tuple(B_ont.shape[:2])} != model ({L},{n_kv})")
    if B_ont.shape[2] != head_dim:
        raise ValueError(
            f"B_ont head_dim={B_ont.shape[2]} != {head_dim}")
    if not (isinstance(payload, dict) and "r_per_pair" in payload):
        raise ValueError("B_ont payload missing r_per_pair (facet_mask builder)")
    n_facets = len(payload.get("facet_order", ["f0", "f1", "f2", "f3"]))
    facet_mask = build_facet_masks(
        payload["r_per_pair"], L=L, H=n_kv,
        r_ont=B_ont.shape[-1], n_facets=n_facets,
    )
    print(f"[ocq] B_ont {tuple(B_ont.shape)}  facet_mask {tuple(facet_mask.shape)}  "
          f"cols/facet={facet_mask.sum(dim=(0,1,3)).tolist()}", flush=True)

    # --- Build prompts ---
    print(f"[prompts] MMLU n={args.n_mmlu}", flush=True)
    prompts_mmlu = mmlu_prompts(args)
    print(f"[prompts] MetaTool n={args.n_metatool}", flush=True)
    prompts_meta = metatool_prompts(args)

    # --- Run both datasets ---
    print("\n[run] MMLU forward pass (hook read-only) ...", flush=True)
    stats_mmlu = run_dataset(
        model, tok, prompts_mmlu, args,
        B_ont, facet_mask, n_kv, head_dim, L, n_facets, tag="mmlu",
    )
    print(f"  MMLU done: runtime={stats_mmlu['runtime_s']:.1f}s  "
          f"E[Σg]={stats_mmlu['sigma_mean_global']:.4f}  "
          f"E[g_f]={[f'{x:.3f}' for x in stats_mmlu['g_f_mean_global']]}",
          flush=True)

    print("\n[run] MetaTool forward pass (hook read-only) ...", flush=True)
    stats_meta = run_dataset(
        model, tok, prompts_meta, args,
        B_ont, facet_mask, n_kv, head_dim, L, n_facets, tag="metatool",
    )
    print(f"  MetaTool done: runtime={stats_meta['runtime_s']:.1f}s  "
          f"E[Σg]={stats_meta['sigma_mean_global']:.4f}  "
          f"E[g_f]={[f'{x:.3f}' for x in stats_meta['g_f_mean_global']]}",
          flush=True)

    # --- Summary + decision-tree hint ---
    ratio = [
        (stats_mmlu["g_f_mean_global"][f]
         / max(stats_meta["g_f_mean_global"][f], 1e-12))
        for f in range(n_facets)
    ]
    sigma_ratio = (stats_mmlu["sigma_mean_global"]
                   / max(stats_meta["sigma_mean_global"], 1e-12))
    print("\n=== SUMMARY ===")
    print(f"E[Σ_f g_f]  MMLU={stats_mmlu['sigma_mean_global']:.4f}  "
          f"MetaTool={stats_meta['sigma_mean_global']:.4f}  "
          f"ratio(MMLU/MetaTool)={sigma_ratio:.3f}")
    print(f"Per-facet MMLU/MetaTool ratio: "
          f"{[f'{r:.3f}' for r in ratio]}")
    sigma_mmlu = stats_mmlu["sigma_mean_global"]
    if sigma_mmlu < 0.1:
        hint = "< 0.1: code bug after all — re-audit (unlikely)"
    elif sigma_mmlu < 0.3:
        hint = "0.1-0.3: baseline-centered gate (Step 2.2) is promising"
    elif sigma_mmlu < 0.7:
        hint = ("0.3-0.7: hard threshold OR multiplicative gate "
                "(Step 2.1 or 2.4)")
    else:
        hint = ("> 0.7: phase-closure is structurally impossible → "
                "Path B or Path C")
    print(f"Decision-tree hint: {hint}")

    payload_out = {
        "model": args.model,
        "b_ont_path": args.b_ont,
        "seed": args.seed,
        "n_mmlu": args.n_mmlu,
        "n_metatool": args.n_metatool,
        "max_ctx": args.max_ctx,
        "facet_order": payload.get("facet_order"),
        "cols_per_facet": facet_mask.sum(dim=(0, 1, 3)).tolist(),
        "mmlu": stats_mmlu,
        "metatool": stats_meta,
        "per_facet_mmlu_over_metatool_ratio": ratio,
        "sigma_mmlu_over_metatool_ratio": sigma_ratio,
        "decision_tree_hint": hint,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload_out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
