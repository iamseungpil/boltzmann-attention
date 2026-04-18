#!/usr/bin/env python3
"""P1 — Random-direction KL scaling (Proposition (c) test).

Proposition (c) [handoff_shared_basis_parallel_2026_04_19 §2.1] predicts:
    E_U[<U U^T q, d*(q)>] = (r/d) <q, d*(q)>
with variance ~ r(d-r)/d^2(d+2).  Under variance-dominant (generic d*) regime,
this yields:
    KL(p_perturbed || p_no_steer)  ~  r/d   (linear in r)
    ||logit_perturbed - logit_no_steer||_2  ~  sqrt(r/d)

For each rank r in --ranks we build per-(layer,head) Haar-orthonormal
U in R^{d x r} (via QR of iid Gaussian), form P = U U^T, apply the K-side
perturbation k_new = k + alpha * P k on marker-span positions of tau2-bench
Telecom prompts, then measure logit/probability shift at the prompt's last
token over the full vocabulary.  This mirrors Variant D's hook semantics
(canonical_adaseka_engine.install_hooks) but removes AdaSEKA routing so
that the random-direction null is clean.

Output:
    reports/shared_basis_proposition_2026_04_19/random_rank_scaling.json
Decision criterion: linear regression KL vs r/d, R^2 > 0.85 passes P1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from canonical_adaseka_engine import build_marker_steer_mask
from eval_tau2_bench import (
    build_chat_prompt_marked,
    build_tools_json,
    extract_domain_tools,
)


def parse_layers_spec(spec: str, total: int):
    if spec == "all":
        return list(range(total))
    if spec.startswith("last"):
        return list(range(total - int(spec[4:]), total))
    if spec.startswith("first"):
        return list(range(int(spec[5:])))
    return [int(x) for x in spec.split(",")]


def build_random_projection(L_sel: int, n_kv: int, d: int, r: int, seed: int,
                            device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Per-(L_sel, n_kv) Haar-orthonormal U of shape (d, r), returns P = U U^T."""
    g = torch.Generator(device="cpu").manual_seed(seed + 1000 * r)
    proj = torch.empty(L_sel, n_kv, d, d, dtype=torch.float32)
    for li in range(L_sel):
        for h in range(n_kv):
            A = torch.randn(d, d, generator=g)
            Q, _ = torch.linalg.qr(A)
            U_r = Q[:, :r].contiguous()
            proj[li, h] = U_r @ U_r.T
    return proj.to(device=device, dtype=dtype)


def make_kbias_hook(i_layer: int, proj: torch.Tensor,
                    n_kv: int, head_dim: int,
                    mask_b: torch.Tensor, amp: float):
    """Hook factory: applies k += amp * P_{L,h} @ k on marker-span positions."""
    def _hook(mod, inputs, output):
        k_in = output
        if k_in.dim() == 3:
            B, T, D = k_in.shape
            if D != n_kv * head_dim:
                return k_in
            k_view = k_in.view(B, T, n_kv, head_dim).clone()
            need_reshape = True
        elif k_in.dim() == 4:
            B, T, H, d = k_in.shape
            k_view = k_in.clone()
            need_reshape = False
        else:
            return k_in
        mb = mask_b.to(k_view.device)
        if mb.shape[0] != T or mb.sum() == 0:
            return k_in
        for h in range(n_kv):
            P = proj[i_layer, h]
            if P.device != k_view.device:
                P = P.to(k_view.device)
            k_sel = k_view[0, mb, h, :]
            delta = (P.to(k_sel.dtype) @ k_sel.T).T
            k_view[0, mb, h, :] = k_sel + amp * delta
        if need_reshape:
            return k_view.reshape(B, T, n_kv * head_dim).to(k_in.dtype)
        return k_view.to(k_in.dtype)
    return _hook


@torch.no_grad()
def last_token_logits(model, ids) -> torch.Tensor:
    out = model(input_ids=ids, use_cache=False)
    return out.logits[0, -1, :].float()


@torch.no_grad()
def run_one_task(model, tok, task, tools_json, ranks, sel_layers,
                 n_kv, head_dim, amp, seed):
    d = head_dim
    prompt_marked = build_chat_prompt_marked(tok, task, tools_json)
    ids, steer_mask = build_marker_steer_mask(prompt_marked, tok)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ids = ids.to(device)
    mask_b = steer_mask[0].to(device)
    if mask_b.sum().item() == 0:
        return None

    logit_base = last_token_logits(model, ids)
    logp_base = F.log_softmax(logit_base, dim=-1)
    argmax_base = int(logit_base.argmax().item())

    per_rank = {}
    for r in ranks:
        proj = build_random_projection(len(sel_layers), n_kv, d, r, seed, device, dtype)

        handles = []
        for i, L in enumerate(sel_layers):
            attn = model.model.layers[L].self_attn
            module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
            handles.append(module.register_forward_hook(
                make_kbias_hook(i, proj, n_kv, head_dim, mask_b, amp)
            ))
        try:
            logit_p = last_token_logits(model, ids)
        finally:
            for h in handles:
                h.remove()

        logp_p = F.log_softmax(logit_p, dim=-1)
        p_p = logp_p.exp()
        kl = float((p_p * (logp_p - logp_base)).sum().item())
        d_logit = float((logit_p - logit_base).norm().item())
        argmax_p = int(logit_p.argmax().item())
        per_rank[str(r)] = {
            "kl": kl,
            "d_logit": d_logit,
            "flipped": int(argmax_p != argmax_base),
        }

        del proj
    return {
        "prompt_len": int(ids.shape[1]),
        "mask_sum": int(mask_b.sum().item()),
        "per_rank": per_rank,
    }


def linear_fit(xs, ys):
    n = len(xs)
    if n < 2:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        return {"slope": 0.0, "intercept": my, "r_squared": 0.0}
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / syy if syy > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "r_squared": r2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", default="external/tau2-bench/data/tau2/domains/telecom/tasks.json")
    p.add_argument("--layers-spec", default="last10")
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 3, 6, 12, 24, 48, 96])
    p.add_argument("--amp", type=float, default=0.3)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/shared_basis_proposition_2026_04_19/random_rank_scaling.json")
    args = p.parse_args()

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()

    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_q
    d = head_dim
    L_total = len(model.model.layers)
    sel_layers = parse_layers_spec(args.layers_spec, L_total)
    print(f"[config] n_q={n_q} n_kv={n_kv} d={d} L_total={L_total}")
    print(f"[sel_layers] {sel_layers}")
    for r in args.ranks:
        if r > d:
            raise ValueError(f"rank {r} > head_dim {d}")

    tasks_all = json.load(open(args.dataset))
    tasks = tasks_all[: args.n]
    domain_tools = extract_domain_tools(tasks_all)
    tools_json = build_tools_json(domain_tools)
    print(f"[tasks] N={len(tasks)} (of {len(tasks_all)}); n_tools={len(domain_tools)}")
    print(f"[sweep] ranks={args.ranks} amp={args.amp}")

    per_task = []
    t0 = time.time()
    for idx, task in enumerate(tasks):
        try:
            out = run_one_task(model, tok, task, tools_json, args.ranks, sel_layers,
                               n_kv, head_dim, args.amp, args.seed)
        except Exception as exc:
            print(f"  [{idx+1}/{len(tasks)}] ERR {type(exc).__name__}: {exc}", flush=True)
            continue
        if out is None:
            print(f"  [{idx+1}/{len(tasks)}] empty marker mask; skip", flush=True)
            continue
        per_task.append({"idx": idx, "task_id": task.get("id", f"t{idx}"), **out})
        if (idx + 1) % 5 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(tasks) - idx - 1)
            means_kl = {
                str(r): statistics.mean(pt["per_rank"][str(r)]["kl"] for pt in per_task)
                for r in args.ranks
            }
            summary = "  ".join(f"r={r}:{means_kl[str(r)]:.4f}" for r in args.ranks)
            print(f"  [{idx+1}/{len(tasks)}] t={elapsed:.1f}s eta={eta:.1f}s  meanKL: {summary}",
                  flush=True)

    if not per_task:
        raise RuntimeError("no tasks produced results")

    # Aggregate
    agg = {}
    for r in args.ranks:
        kls = [pt["per_rank"][str(r)]["kl"] for pt in per_task]
        dls = [pt["per_rank"][str(r)]["d_logit"] for pt in per_task]
        flips = [pt["per_rank"][str(r)]["flipped"] for pt in per_task]
        agg[str(r)] = {
            "r": r,
            "r_over_d": r / d,
            "kl_mean": statistics.mean(kls),
            "kl_std": statistics.pstdev(kls),
            "dlogit_mean": statistics.mean(dls),
            "dlogit_std": statistics.pstdev(dls),
            "flip_rate": sum(flips) / len(flips),
            "n": len(kls),
        }

    # Linear fits
    xs_ratio = [agg[str(r)]["r_over_d"] for r in args.ranks]
    kl_means = [agg[str(r)]["kl_mean"] for r in args.ranks]
    dl_means = [agg[str(r)]["dlogit_mean"] for r in args.ranks]

    fit_kl = linear_fit(xs_ratio, kl_means)          # KL vs r/d
    fit_dl_sqrt = linear_fit([math.sqrt(x) for x in xs_ratio], dl_means)  # ||dl|| vs sqrt(r/d)

    # log-log (power law) fit
    try:
        log_x = [math.log(x) for x in xs_ratio]
        log_kl = [math.log(max(v, 1e-12)) for v in kl_means]
        log_dl = [math.log(max(v, 1e-12)) for v in dl_means]
        power_kl = linear_fit(log_x, log_kl)         # slope expected ~1.0
        power_dl = linear_fit(log_x, log_dl)         # slope expected ~0.5
    except Exception:
        power_kl = {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}
        power_dl = {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

    decision_kl = "PASS" if fit_kl["r_squared"] > 0.85 else "FAIL"

    print(f"\n[linear fit] KL vs r/d:        slope={fit_kl['slope']:.4f}  "
          f"intercept={fit_kl['intercept']:.4f}  R^2={fit_kl['r_squared']:.4f}")
    print(f"[linear fit] ||dlogit|| vs √(r/d): slope={fit_dl_sqrt['slope']:.4f}  "
          f"intercept={fit_dl_sqrt['intercept']:.4f}  R^2={fit_dl_sqrt['r_squared']:.4f}")
    print(f"[power law]  log KL vs log(r/d): slope={power_kl['slope']:.4f}  "
          f"R^2={power_kl['r_squared']:.4f}  (expect ≈ 1.0)")
    print(f"[power law]  log ||dl|| vs log(r/d): slope={power_dl['slope']:.4f}  "
          f"R^2={power_dl['r_squared']:.4f}  (expect ≈ 0.5)")
    print(f"[decision]   KL vs r/d R^2>0.85 → {decision_kl}")

    out = {
        "args": vars(args),
        "head_dim": d,
        "n_kv": n_kv,
        "sel_layers": sel_layers,
        "n_tasks_used": len(per_task),
        "aggregate": agg,
        "linear_fit_kl_vs_ratio": fit_kl,
        "linear_fit_dlogit_vs_sqrt_ratio": fit_dl_sqrt,
        "power_law_kl": power_kl,
        "power_law_dlogit": power_dl,
        "decision_kl_linear": decision_kl,
        "per_task": per_task,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
