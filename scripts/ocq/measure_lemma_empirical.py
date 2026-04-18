#!/usr/bin/env python3
"""Lemma-empirical sweep (a) + (b) + (c) for Argmax-Subspace Selectivity Lemma.

Three measurements in a single forward-pass loop:

(a) P1 re-run with per-task U resample
      Same 7-rank sweep as measure_random_rank_scaling.py, but
      ``build_random_projection`` receives a seed that depends on (rank, task_idx)
      so that each of the 100 queries sees an independently drawn Haar-orthonormal
      U.  The per-rank KL std across tasks now reflects the Haar-measure variance
      predicted by Ledoux concentration, not single-draw fluctuation.

(b) Prompt-end argmax margin m(q) = ell_(1)(q) - ell_(2)(q)
      Measured from the no-steer baseline forward pass.  Gives an empirical
      distribution of the "top-1 vs top-2 logit gap" that the Lemma needs as
      an empirical lower bound m_0.

(c) Attention-weight shift at the prompt-end query position
      ``attn_base = softmax(q_last k^T / sqrt(d))``, ``attn_pert`` analogously
      with the K-side perturbation hook installed.  Frobenius norm and L1
      distance over key positions, per (layer, head), captured for all 7 ranks.
      Dis-ambiguates whether the r=96 KL explosion originates at the attention
      logit (Lemma scope) or later in the layer stack (FFN/LM-head amplification).

Output:
    reports/shared_basis_proposition_2026_04_19/lemma_empirical.json
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
    """Per-(L_sel, n_kv) Haar-orthonormal U of shape (d, r); returns P = U U^T."""
    g = torch.Generator(device="cpu").manual_seed(seed)
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
    """Hook factory: k += amp * P_{L,h} @ k on marker-span positions."""
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
def forward_capture(model, ids, capture_attn: bool):
    """Forward pass returning (last-token logits [V], attentions [L][1,H,T,T] or None)."""
    out = model(
        input_ids=ids,
        use_cache=False,
        output_attentions=capture_attn,
    )
    logit_last = out.logits[0, -1, :].float()
    attns = None
    if capture_attn:
        # attentions is tuple of length L, each (1, n_q, T, T)
        # Keep only last-token query row for memory: list of (n_q, T)
        attns = [a[0, :, -1, :].float().cpu() for a in out.attentions]
    return logit_last, attns


def attn_shift_summary(attn_base, attn_pert, sel_layers):
    """Per-layer Frobenius + L1 norm of (pert - base) at last-token query row.

    Returns dict: per-layer stats (mean/max over heads) + overall stats.
    """
    if attn_base is None or attn_pert is None:
        return None
    per_layer = {}
    fro_all, l1_all = [], []
    for L in sel_layers:
        a_b = attn_base[L]  # (n_q, T)
        a_p = attn_pert[L]
        diff = a_p - a_b
        fro = diff.norm(dim=-1)       # (n_q,)
        l1 = diff.abs().sum(dim=-1)   # (n_q,)
        per_layer[str(L)] = {
            "fro_mean": float(fro.mean().item()),
            "fro_max": float(fro.max().item()),
            "l1_mean": float(l1.mean().item()),
            "l1_max": float(l1.max().item()),
        }
        fro_all.append(fro)
        l1_all.append(l1)
    fro_cat = torch.cat(fro_all)
    l1_cat = torch.cat(l1_all)
    return {
        "per_layer": per_layer,
        "overall_fro_mean": float(fro_cat.mean().item()),
        "overall_fro_max": float(fro_cat.max().item()),
        "overall_l1_mean": float(l1_cat.mean().item()),
        "overall_l1_max": float(l1_cat.max().item()),
    }


@torch.no_grad()
def run_one_task(model, tok, task, tools_json, ranks, sel_layers,
                 n_kv, head_dim, amp, base_seed, task_idx,
                 resample_per_task: bool, measure_attn: bool,
                 attn_ranks: set):
    d = head_dim
    prompt_marked = build_chat_prompt_marked(tok, task, tools_json)
    ids, steer_mask = build_marker_steer_mask(prompt_marked, tok)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ids = ids.to(device)
    mask_b = steer_mask[0].to(device)
    if mask_b.sum().item() == 0:
        return None

    # (b) baseline: full-vocab logits + margin + attention (if needed)
    logit_base, attn_base = forward_capture(model, ids, capture_attn=measure_attn)
    logp_base = F.log_softmax(logit_base, dim=-1)
    argmax_base = int(logit_base.argmax().item())
    top2 = torch.topk(logit_base, 2)
    m_q = float((top2.values[0] - top2.values[1]).item())

    per_rank = {}
    for r in ranks:
        # (a) per-task U resample: seed depends on (rank, task)
        seed_r = base_seed + 1000 * r
        if resample_per_task:
            seed_r += (task_idx + 1) * 7919  # coprime offset
        proj = build_random_projection(len(sel_layers), n_kv, d, r, seed_r, device, dtype)

        handles = []
        for i, L in enumerate(sel_layers):
            attn = model.model.layers[L].self_attn
            module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
            handles.append(module.register_forward_hook(
                make_kbias_hook(i, proj, n_kv, head_dim, mask_b, amp)
            ))
        capture_this = measure_attn and (r in attn_ranks)
        try:
            logit_p, attn_p = forward_capture(model, ids, capture_attn=capture_this)
        finally:
            for h in handles:
                h.remove()

        logp_p = F.log_softmax(logit_p, dim=-1)
        p_p = logp_p.exp()
        kl = float((p_p * (logp_p - logp_base)).sum().item())
        d_logit = float((logit_p - logit_base).norm().item())
        argmax_p = int(logit_p.argmax().item())
        shift = attn_shift_summary(attn_base, attn_p, sel_layers) if capture_this else None
        per_rank[str(r)] = {
            "kl": kl,
            "d_logit": d_logit,
            "flipped": int(argmax_p != argmax_base),
            "attn_shift": shift,
        }
        del proj

    return {
        "prompt_len": int(ids.shape[1]),
        "mask_sum": int(mask_b.sum().item()),
        "margin": m_q,          # (b)
        "top2_diff_tokens": [int(top2.indices[0].item()), int(top2.indices[1].item())],
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


def quantiles(vals, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not vals:
        return {}
    s = sorted(vals)
    out = {}
    for q in qs:
        idx = int(round(q * (len(s) - 1)))
        out[f"q{int(q*100):02d}"] = s[idx]
    return out


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
    p.add_argument("--resample-per-task", action="store_true", default=True)
    p.add_argument("--no-resample-per-task", dest="resample_per_task", action="store_false")
    p.add_argument("--measure-attn", action="store_true", default=True)
    p.add_argument("--no-measure-attn", dest="measure_attn", action="store_false")
    p.add_argument("--attn-ranks", nargs="+", type=int, default=None,
                   help="ranks at which to capture attention (default: all)")
    p.add_argument("--out", default="reports/shared_basis_proposition_2026_04_19/lemma_empirical.json")
    args = p.parse_args()

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation="eager",  # output_attentions requires eager
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

    attn_ranks = set(args.attn_ranks) if args.attn_ranks else set(args.ranks)
    print(f"[sweep] ranks={args.ranks} amp={args.amp} resample_per_task={args.resample_per_task}")
    print(f"[attn] measure_attn={args.measure_attn} at ranks={sorted(attn_ranks)}")

    per_task = []
    t0 = time.time()
    for idx, task in enumerate(tasks):
        try:
            out = run_one_task(
                model, tok, task, tools_json, args.ranks, sel_layers,
                n_kv, head_dim, args.amp, args.seed, idx,
                args.resample_per_task, args.measure_attn, attn_ranks,
            )
        except Exception as exc:
            print(f"  [{idx+1}/{len(tasks)}] ERR {type(exc).__name__}: {exc}", flush=True)
            import traceback; traceback.print_exc()
            continue
        if out is None:
            continue
        per_task.append({"idx": idx, "task_id": task.get("id", f"t{idx}"), **out})
        if (idx + 1) % 5 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(tasks) - idx - 1)
            means_kl = {
                str(r): statistics.mean(pt["per_rank"][str(r)]["kl"] for pt in per_task)
                for r in args.ranks
            }
            margins = [pt["margin"] for pt in per_task]
            mm = statistics.mean(margins)
            summary = "  ".join(f"r={r}:{means_kl[str(r)]:.4f}" for r in args.ranks)
            print(f"  [{idx+1}/{len(tasks)}] t={elapsed:.1f}s eta={eta:.1f}s  "
                  f"m̄={mm:.3f}  meanKL: {summary}", flush=True)

    if not per_task:
        raise RuntimeError("no tasks produced results")

    # Aggregate KL/dlogit/flip/attn per rank
    agg = {}
    for r in args.ranks:
        kls = [pt["per_rank"][str(r)]["kl"] for pt in per_task]
        dls = [pt["per_rank"][str(r)]["d_logit"] for pt in per_task]
        flips = [pt["per_rank"][str(r)]["flipped"] for pt in per_task]
        shifts = [pt["per_rank"][str(r)]["attn_shift"] for pt in per_task
                  if pt["per_rank"][str(r)]["attn_shift"] is not None]
        entry = {
            "r": r,
            "r_over_d": r / d,
            "kl_mean": statistics.mean(kls),
            "kl_std": statistics.pstdev(kls),
            "dlogit_mean": statistics.mean(dls),
            "dlogit_std": statistics.pstdev(dls),
            "flip_rate": sum(flips) / len(flips),
            "flip_count": sum(flips),
            "n": len(kls),
        }
        if shifts:
            entry["attn_shift"] = {
                "overall_fro_mean": statistics.mean(s["overall_fro_mean"] for s in shifts),
                "overall_fro_max_mean": statistics.mean(s["overall_fro_max"] for s in shifts),
                "overall_l1_mean": statistics.mean(s["overall_l1_mean"] for s in shifts),
                "overall_l1_max_mean": statistics.mean(s["overall_l1_max"] for s in shifts),
                "n_captured": len(shifts),
            }
        agg[str(r)] = entry

    # (b) margin distribution
    margins = [pt["margin"] for pt in per_task]
    margin_stats = {
        "n": len(margins),
        "mean": statistics.mean(margins),
        "std": statistics.pstdev(margins),
        "min": min(margins),
        "max": max(margins),
        "quantiles": quantiles(margins),
    }

    # (a) Linear fits with per-task resample
    xs_ratio = [agg[str(r)]["r_over_d"] for r in args.ranks]
    kl_means = [agg[str(r)]["kl_mean"] for r in args.ranks]
    dl_means = [agg[str(r)]["dlogit_mean"] for r in args.ranks]

    fit_kl = linear_fit(xs_ratio, kl_means)
    fit_dl_sqrt = linear_fit([math.sqrt(x) for x in xs_ratio], dl_means)

    try:
        log_x = [math.log(x) for x in xs_ratio]
        log_kl = [math.log(max(v, 1e-12)) for v in kl_means]
        log_dl = [math.log(max(v, 1e-12)) for v in dl_means]
        power_kl = linear_fit(log_x, log_kl)
        power_dl = linear_fit(log_x, log_dl)
    except Exception:
        power_kl = {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}
        power_dl = {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

    decision_kl = "PASS" if fit_kl["r_squared"] > 0.85 else "FAIL"

    print(f"\n========== (a) per-task-resampled rank sweep ==========")
    for r in args.ranks:
        e = agg[str(r)]
        print(f"  r={r:>3}  r/d={e['r_over_d']:.3f}  "
              f"KL={e['kl_mean']:.4f}±{e['kl_std']:.4f}  "
              f"||dl||={e['dlogit_mean']:.2f}±{e['dlogit_std']:.2f}  "
              f"flip={e['flip_count']}/{e['n']}"
              + (f"  attn_fro={e['attn_shift']['overall_fro_mean']:.4f}"
                 if "attn_shift" in e else ""))
    print(f"  [linear]  KL vs r/d:        slope={fit_kl['slope']:.4f}  R²={fit_kl['r_squared']:.4f}")
    print(f"  [linear]  ||dl|| vs √(r/d): slope={fit_dl_sqrt['slope']:.4f}  R²={fit_dl_sqrt['r_squared']:.4f}")
    print(f"  [power]   log KL vs log(r/d):    slope={power_kl['slope']:.4f}  R²={power_kl['r_squared']:.4f}")
    print(f"  [power]   log ||dl|| vs log(r/d): slope={power_dl['slope']:.4f}  R²={power_dl['r_squared']:.4f}")
    print(f"  [decision] R²>0.85 → {decision_kl}")

    print(f"\n========== (b) margin m(q) distribution ==========")
    print(f"  N={margin_stats['n']}  mean={margin_stats['mean']:.3f}  std={margin_stats['std']:.3f}")
    print(f"  min={margin_stats['min']:.3f}  max={margin_stats['max']:.3f}")
    q = margin_stats["quantiles"]
    print(f"  quantiles: q10={q['q10']:.3f}  q25={q['q25']:.3f}  q50={q['q50']:.3f}  "
          f"q75={q['q75']:.3f}  q90={q['q90']:.3f}")

    if args.measure_attn:
        print(f"\n========== (c) attention-weight shift at last-token query ==========")
        for r in args.ranks:
            e = agg[str(r)]
            if "attn_shift" in e:
                a = e["attn_shift"]
                print(f"  r={r:>3}  Fro_mean={a['overall_fro_mean']:.4f}  "
                      f"Fro_max={a['overall_fro_max_mean']:.4f}  "
                      f"L1_mean={a['overall_l1_mean']:.4f}  "
                      f"L1_max={a['overall_l1_max_mean']:.4f}  "
                      f"(n_captured={a['n_captured']})")

    out = {
        "args": vars(args),
        "head_dim": d,
        "n_kv": n_kv,
        "sel_layers": sel_layers,
        "n_tasks_used": len(per_task),
        "aggregate": agg,
        "margin_stats": margin_stats,
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
