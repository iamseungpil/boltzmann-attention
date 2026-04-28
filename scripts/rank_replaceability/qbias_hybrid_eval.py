#!/usr/bin/env python3
"""qbias_hybrid_eval.py — Theorem 2 verification (E3 follow-up).

Tests whether adding a Q-bias steering term (β · V_k V_k^T · Q at every layer's
q_proj output, per-head) closes the residual gap left by static rank-k injection.

For each query, we run:
  full       := model([prefix, user]).logits[-1]
  noprompt   := model([user]).logits[-1]
  static     := model([user]) + V_k V_k^T phi_mean injection at o_proj input
  hybrid(β)  := static + β · V_k V_k^T Q at q_proj output

Reports: KL(full ‖ method) and top-1 agreement, swept over β.

Theorem 2 predicts:
  - Sign of optimal β matches the sign of ∂_q λ_P(q_centroid)
  - Adding β · V_k V_k^T Q with the right sign reduces the residual KL
  - Magnitude of residual reduction = first-order Taylor term

Usage:
  python qbias_hybrid_eval.py --model Qwen/Qwen2.5-7B-Instruct \
      --task metatool_st4 \
      --e1-json reports/rank_replaceability_2026_04/qwen_metatool_n256.json \
      --max-samples 128 --k-static 1 \
      --beta-list -0.3,-0.1,-0.05,0,0.05,0.1,0.3 \
      --device cuda:0 \
      --out reports/rank_replaceability_2026_04/qwen_qbias_n128.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from measure_phi_rank import (  # type: ignore
    DEFAULT_TOOL_SYSTEM_PROMPT,
    build_messages,
    detect_model_family,
    load_metatool_st4,
    load_tau2,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", default="metatool_st4")
    p.add_argument("--metatool-path", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--e1-json", required=True)
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--k-static", type=int, default=1, help="rank of static injection")
    p.add_argument("--k-qbias", type=int, default=4,
                   help="rank of Q-bias projector V_k V_k^T")
    p.add_argument("--beta-list", default="-0.3,-0.1,-0.05,0,0.05,0.1,0.3")
    p.add_argument(
        "--basis-mode",
        choices=["phi_p", "q_self"],
        default="phi_p",
        help="phi_p: V_k from E1 npz (Phi_P SVD). "
             "q_self: measure Q activations on noprompt inputs and SVD per (layer,head).",
    )
    p.add_argument(
        "--basis-measure-n", type=int, default=128,
        help="If basis-mode=q_self, # queries used to compute Q-SVD basis.",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


# =============================================================================
# Hooks
# =============================================================================

class StaticInjection:
    """Adds vec[ell, h, d_h] to o_proj input at last position, per (layer, head)."""

    def __init__(self, model, injection: torch.Tensor):
        self.model = model
        self.injection = injection  # (L, H_q, d_h) on device
        self.handles = []
        self.enabled = False

    def _hook(self, ell: int):
        def fn(module, inputs):
            if not self.enabled:
                return inputs
            x = inputs[0]
            B, T, D = x.shape
            H_q = self.injection.shape[1]
            d_h = self.injection.shape[2]
            assert D == H_q * d_h
            x_r = x.view(B, T, H_q, d_h).clone()
            inj = self.injection[ell].to(x.dtype)
            x_r[:, -1, :, :] = x_r[:, -1, :, :] + inj
            return (x_r.view(B, T, D),) + tuple(inputs[1:])
        return fn

    def install(self):
        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._hook(i)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class QBiasHook:
    """Q-bias steering: Q' = Q + β · (V_k V_k^T) Q at every layer's q_proj output, per-head.

    V_k is per (layer, head). Apply at all positions (canonical Q-bias semantics).
    """

    def __init__(self, model, P_kk: torch.Tensor, beta: float):
        """P_kk: (L, H_q, d_h, d_h) projector V_k V_k^T per (layer, head). beta: scalar."""
        self.model = model
        self.P_kk = P_kk
        self.beta = beta
        self.handles = []
        self.enabled = False

    def _hook(self, ell: int):
        def fn(module, inputs, output):
            if not self.enabled or self.beta == 0:
                return output
            # output: (B, T, H_q*d_h) — q_proj output
            x = output
            B, T, D = x.shape
            H_q = self.P_kk.shape[1]
            d_h = self.P_kk.shape[2]
            assert D == H_q * d_h
            x_r = x.view(B, T, H_q, d_h)
            # Per-head: Q' = Q + β · (P_kk[ell, h] @ Q)
            P = self.P_kk[ell].to(x.dtype)  # (H_q, d_h, d_h)
            # (B, T, H_q, d_h) @ (H_q, d_h, d_h)^T  -> (B, T, H_q, d_h)
            Q_proj = torch.einsum("bthi,hij->bthj", x_r, P)
            x_new = x_r + self.beta * Q_proj
            return x_new.view(B, T, D)
        return fn

    def install(self):
        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(
                layer.self_attn.q_proj.register_forward_hook(self._hook(i))
            )

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# =============================================================================
# Helpers
# =============================================================================

def build_static_injection(eigvecs: np.ndarray, phi_mean: np.ndarray, k: int) -> np.ndarray:
    L, H_q, K_save, d_h = eigvecs.shape
    k = min(k, K_save)
    inj = np.zeros((L, H_q, d_h), dtype=np.float32)
    if k <= 0:
        return inj
    for ell in range(L):
        for h in range(H_q):
            V = eigvecs[ell, h, :k]  # (k, d_h)
            mean = phi_mean[ell, h]
            inj[ell, h] = V.T @ (V @ mean)
    return inj


def build_qbias_projector(eigvecs: np.ndarray, k: int) -> np.ndarray:
    """Returns (L, H_q, d_h, d_h) per-head projector V_k V_k^T."""
    L, H_q, K_save, d_h = eigvecs.shape
    k = min(k, K_save)
    P = np.zeros((L, H_q, d_h, d_h), dtype=np.float32)
    if k <= 0:
        return P
    for ell in range(L):
        for h in range(H_q):
            V = eigvecs[ell, h, :k]  # (k, d_h)
            P[ell, h] = V.T @ V
    return P


def measure_q_self_basis(
    model, tokenizer, items, device, dtype,
    L: int, H_q: int, d_h: int, k_save: int = 32,
) -> np.ndarray:
    """E8: compute V_k from SVD of Q activations on noprompt inputs.

    For each query, run a noprompt (user-only) forward, capture q_proj output
    at last position, reshape to (H_q, d_h). Stack over queries → (N, H_q, d_h).
    SVD per (layer, head) yields eigvecs of shape (L, H_q, k_save, d_h).
    """
    q_buffers = [[] for _ in range(L)]
    handles = []

    def make_hook(ell):
        def fn(module, inputs, output):
            x = output  # (B, T, H_q*d_h)
            B, T, D = x.shape
            assert D == H_q * d_h, f"q_proj out D={D} != H_q*d_h"
            q_last = x[0, -1, :].view(H_q, d_h).detach().float().cpu().numpy()
            q_buffers[ell].append(q_last)
        return fn

    for ell in range(L):
        layer = model.model.layers[ell]
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_hook(ell)))

    print(f"[q_self] measuring Q activations on {len(items)} noprompt inputs", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for i, item in enumerate(items):
            msgs = [{"role": "user", "content": item["query"]}]
            try:
                txt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            except Exception:
                continue
            ids = tokenizer(txt, return_tensors="pt").input_ids.to(device)
            _ = model(input_ids=ids, use_cache=False, return_dict=True)
            if (i + 1) % 32 == 0:
                print(f"  q_self [{i+1}/{len(items)}] {time.time()-t0:.1f}s", flush=True)

    for h in handles:
        h.remove()

    # SVD per (layer, head)
    eigvecs = np.zeros((L, H_q, k_save, d_h), dtype=np.float32)
    for ell in range(L):
        Q = np.stack(q_buffers[ell])  # (N, H_q, d_h)
        for h in range(H_q):
            M = Q[:, h, :]  # (N, d_h)
            try:
                _, s, Vh = np.linalg.svd(M, full_matrices=False, compute_uv=True)
            except np.linalg.LinAlgError:
                continue
            kk = min(k_save, Vh.shape[0])
            eigvecs[ell, h, :kk] = Vh[:kk]
    print(f"[q_self] basis ready ({time.time()-t0:.1f}s)", flush=True)
    return eigvecs


@torch.no_grad()
def get_last_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False, return_dict=True)
    return out.logits[:, -1, :]


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"[e1] L={L} H_q={H_q} d_h={d_h} K_save={K_save}", flush=True)

    # Build static injection (k=k_static fixed) — always from Phi_P SVD
    static_inj_np = build_static_injection(eigvecs, phi_mean, args.k_static)
    # Build Q-bias projector (k=k_qbias) — basis source depends on --basis-mode
    if args.basis_mode == "phi_p":
        P_kk_np = build_qbias_projector(eigvecs, args.k_qbias)
        qbias_basis_source = "phi_p_svd_from_e1_npz"
    elif args.basis_mode == "q_self":
        # Need to measure Q-self basis. Defer until after model load.
        P_kk_np = None
        qbias_basis_source = "q_self_svd_on_noprompt"
    else:
        raise ValueError(args.basis_mode)

    # Data
    if args.task == "metatool_st4":
        items = load_metatool_st4(args.metatool_path, args.max_samples)
    elif args.task.startswith("tau2_"):
        items = load_tau2(args.task.split("_", 1)[1], args.max_samples)
    else:
        raise ValueError(args.task)
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

    # If q_self basis: measure Q activations BEFORE installing intervention hooks
    if args.basis_mode == "q_self":
        n_meas = min(args.basis_measure_n, len(items))
        q_eigvecs = measure_q_self_basis(
            model, tokenizer, items[:n_meas],
            args.device, dtype, L, H_q, d_h, k_save=max(args.k_qbias, 16),
        )
        P_kk_np = build_qbias_projector(q_eigvecs, args.k_qbias)

    # Move tensors to device
    static_inj_t = torch.from_numpy(static_inj_np).to(args.device, dtype=dtype)
    P_kk_t = torch.from_numpy(P_kk_np).to(args.device, dtype=dtype)

    inj = StaticInjection(model, static_inj_t)
    inj.install()
    qb = QBiasHook(model, P_kk_t, beta=0.0)
    qb.install()

    betas = sorted(set(float(b) for b in args.beta_list.split(",")))
    methods = ["full", "noprompt", "static_only"] + [f"hybrid_b{b:+.3f}" for b in betas]
    print(f"[methods] {methods}", flush=True)

    metrics = {m: {"kl": [], "top1": [], "logit_resid": []} for m in methods if m != "full"}

    N = len(items)
    t0 = time.time()
    for i, item in enumerate(items):
        msgs_full = build_messages(item, DEFAULT_TOOL_SYSTEM_PROMPT, prefix_mode="real")
        try:
            txt_full = tokenizer.apply_chat_template(msgs_full, add_generation_prompt=True, tokenize=False)
        except Exception:
            continue
        ids_full = tokenizer(txt_full, return_tensors="pt").input_ids.to(args.device)

        msgs_user = [{"role": "user", "content": item["query"]}]
        try:
            txt_user = tokenizer.apply_chat_template(msgs_user, add_generation_prompt=True, tokenize=False)
        except Exception:
            continue
        ids_user = tokenizer(txt_user, return_tensors="pt").input_ids.to(args.device)

        # full
        inj.enabled = False; qb.enabled = False
        L_full = get_last_logits(model, ids_full)[0].float()

        # noprompt
        inj.enabled = False; qb.enabled = False
        L_np = get_last_logits(model, ids_user)[0].float()

        def record(name, L_x):
            kl = F.kl_div(F.log_softmax(L_x, dim=-1), F.softmax(L_full, dim=-1), reduction="sum").item()
            ag = int(L_full.argmax().item() == L_x.argmax().item())
            res = (L_full - L_x).abs().mean().item()
            metrics[name]["kl"].append(kl)
            metrics[name]["top1"].append(ag)
            metrics[name]["logit_resid"].append(res)

        record("noprompt", L_np)

        # static_only
        inj.enabled = True; qb.enabled = False
        L_s = get_last_logits(model, ids_user)[0].float()
        record("static_only", L_s)

        # hybrid for each beta
        for b in betas:
            inj.enabled = True
            qb.beta = b
            qb.enabled = (b != 0)
            L_h = get_last_logits(model, ids_user)[0].float()
            record(f"hybrid_b{b:+.3f}", L_h)
        qb.enabled = False
        inj.enabled = False

        if (i + 1) % 8 == 0 or i == N - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"[{i+1}/{N}] elapsed={elapsed:.1f}s rate={rate:.2f}/s", flush=True)

    inj.remove()
    qb.remove()

    summary = {}
    for m in metrics:
        d = metrics[m]
        if not d["kl"]:
            continue
        summary[m] = {
            "kl_mean": float(np.mean(d["kl"])),
            "kl_median": float(np.median(d["kl"])),
            "top1": float(np.mean(d["top1"])),
            "logit_resid_mean": float(np.mean(d["logit_resid"])),
        }

    out = {
        "model": args.model,
        "task": args.task,
        "n_samples": N,
        "k_static": args.k_static,
        "k_qbias": args.k_qbias,
        "basis_mode": args.basis_mode,
        "qbias_basis_source": qbias_basis_source,
        "betas": betas,
        "summary": summary,
        "details": metrics,
        "wall_seconds": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] saved -> {out_path}", flush=True)
    print()
    print(f"{'method':<25} {'KL':>8} {'top1':>6} {'logit_resid':>12}")
    print("-" * 60)
    for m in ["noprompt", "static_only"] + [f"hybrid_b{b:+.3f}" for b in betas]:
        if m not in summary:
            continue
        s = summary[m]
        print(f"{m:<25} {s['kl_mean']:>8.4f} {s['top1']:>6.3f} {s['logit_resid_mean']:>12.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
