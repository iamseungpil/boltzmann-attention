#!/usr/bin/env python3
"""Same measurements as measure_lemma_empirical.py but for MetaTool Subtask4.

Only difference is the prompt-building layer:
    - MetaTool ST4 entries have {action_prompt, tool, query, thought_prompt}
    - candidates are parsed from the numbered list inside action_prompt
    - the user-query segment is wrapped in `**...**` markers for the
      marker-gated steering mask (same as tau2 path)

All measurement logic (KL, margin, attention shift, Haar rank sweep,
per-task resample) is imported from measure_lemma_empirical.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
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
from measure_lemma_empirical import (
    parse_layers_spec,
    build_random_projection,
    make_kbias_hook,
    forward_capture,
    attn_shift_summary,
    linear_fit,
    quantiles,
)


CAND_RE = re.compile(r"\d+\.\s+tool name:\s+([A-Za-z0-9_\-\+\.]+)")
QUERY_RE = re.compile(r"\[User's Query Start\]\n(.*?)\n\[User's Query End\]",
                      re.DOTALL)


def parse_candidates(action_prompt: str):
    return CAND_RE.findall(action_prompt)


def parse_user_query(action_prompt: str):
    m = QUERY_RE.search(action_prompt)
    if m:
        return m.group(1).strip()
    # fallback: old format
    for line in action_prompt.splitlines():
        if line.strip().startswith("User query"):
            parts = line.split('"')
            if len(parts) >= 2:
                return parts[1]
    return action_prompt[:300]


def build_st4_prompt_marked(tokenizer, entry, marker: str = "**") -> str:
    action_prompt = entry["action_prompt"]
    candidates = parse_candidates(action_prompt)
    query = parse_user_query(action_prompt)
    tool_list = ", ".join(candidates)
    system_msg = (
        "You are a tool-selection agent. Given a user query, emit ONE OR MORE "
        f"<tool_call> blocks naming tools from this list: [{tool_list}]. "
        "Format each tool call exactly as:\n"
        '<tool_call>{"name": "ToolName", "arguments": {}}</tool_call>\n'
        "Emit MULTIPLE blocks if the query needs multiple tools. "
        "Do not include explanations. Output ONLY the <tool_call> blocks."
    )
    marked = f"{marker}{query}{marker}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": marked},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def run_one_task(model, tok, entry, ranks, sel_layers,
                 n_kv, head_dim, amp, base_seed, task_idx,
                 resample_per_task: bool, measure_attn: bool,
                 attn_ranks):
    d = head_dim
    prompt_marked = build_st4_prompt_marked(tok, entry)
    ids, steer_mask = build_marker_steer_mask(prompt_marked, tok)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ids = ids.to(device)
    mask_b = steer_mask[0].to(device)
    if mask_b.sum().item() == 0:
        return None

    logit_base, attn_base = forward_capture(model, ids, capture_attn=measure_attn)
    logp_base = F.log_softmax(logit_base, dim=-1)
    argmax_base = int(logit_base.argmax().item())
    top2 = torch.topk(logit_base, 2)
    m_q = float((top2.values[0] - top2.values[1]).item())

    per_rank = {}
    for r in ranks:
        seed_r = base_seed + 1000 * r
        if resample_per_task:
            seed_r += (task_idx + 1) * 7919
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
        "margin": m_q,
        "top2_diff_tokens": [int(top2.indices[0].item()), int(top2.indices[1].item())],
        "gt_tools": entry.get("tool") if isinstance(entry.get("tool"), list) else [entry.get("tool")],
        "per_rank": per_rank,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--layers-spec", default="last10")
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 3, 6, 12, 24, 48, 96])
    p.add_argument("--amp", type=float, default=0.3)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resample-per-task", action="store_true", default=True)
    p.add_argument("--no-resample-per-task", dest="resample_per_task", action="store_false")
    p.add_argument("--measure-attn", action="store_true", default=True)
    p.add_argument("--no-measure-attn", dest="measure_attn", action="store_false")
    p.add_argument("--attn-ranks", nargs="+", type=int, default=None)
    p.add_argument("--out", default="reports/new_theorem_test/phase_a_st4.json")
    args = p.parse_args()

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        attn_implementation="eager",
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

    data_all = json.load(open(args.dataset))
    data = data_all[: args.n]
    print(f"[tasks] N={len(data)} (of {len(data_all)})")

    attn_ranks = set(args.attn_ranks) if args.attn_ranks else set(args.ranks)
    print(f"[sweep] ranks={args.ranks} amp={args.amp} resample_per_task={args.resample_per_task}")

    per_task = []
    t0 = time.time()
    for idx, entry in enumerate(data):
        try:
            out = run_one_task(
                model, tok, entry, args.ranks, sel_layers,
                n_kv, head_dim, args.amp, args.seed, idx,
                args.resample_per_task, args.measure_attn, attn_ranks,
            )
        except Exception as exc:
            print(f"  [{idx+1}/{len(data)}] ERR {type(exc).__name__}: {exc}", flush=True)
            import traceback; traceback.print_exc()
            continue
        if out is None:
            continue
        per_task.append({"idx": idx, **out})
        if (idx + 1) % 5 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(data) - idx - 1)
            means_kl = {
                str(r): statistics.mean(pt["per_rank"][str(r)]["kl"] for pt in per_task)
                for r in args.ranks
            }
            mm = statistics.mean(pt["margin"] for pt in per_task)
            summary = "  ".join(f"r={r}:{means_kl[str(r)]:.4f}" for r in args.ranks)
            print(f"  [{idx+1}/{len(data)}] t={elapsed:.1f}s eta={eta:.1f}s  "
                  f"m̄={mm:.3f}  meanKL: {summary}", flush=True)

    if not per_task:
        raise RuntimeError("no tasks produced results")

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

    margins = [pt["margin"] for pt in per_task]
    margin_stats = {
        "n": len(margins),
        "mean": statistics.mean(margins),
        "std": statistics.pstdev(margins),
        "min": min(margins),
        "max": max(margins),
        "quantiles": quantiles(margins),
    }

    xs_ratio = [agg[str(r)]["r_over_d"] for r in args.ranks]
    kl_means = [agg[str(r)]["kl_mean"] for r in args.ranks]
    dl_means = [agg[str(r)]["dlogit_mean"] for r in args.ranks]
    attn_fro_means = [agg[str(r)].get("attn_shift", {}).get("overall_fro_mean", 0.0)
                      for r in args.ranks]

    fit_kl = linear_fit(xs_ratio, kl_means)
    fit_dl_sqrt = linear_fit([math.sqrt(x) for x in xs_ratio], dl_means)
    try:
        log_x = [math.log(x) for x in xs_ratio]
        power_kl = linear_fit(log_x, [math.log(max(v, 1e-12)) for v in kl_means])
        power_dl = linear_fit(log_x, [math.log(max(v, 1e-12)) for v in dl_means])
        power_attn = linear_fit(log_x, [math.log(max(v, 1e-12)) for v in attn_fro_means])
    except Exception:
        power_kl = power_dl = power_attn = {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

    decision_kl = "PASS" if fit_kl["r_squared"] > 0.85 else "FAIL"

    print(f"\n========== (a) per-task-resampled rank sweep ==========")
    for r in args.ranks:
        e = agg[str(r)]
        line = (f"  r={r:>3}  r/d={e['r_over_d']:.3f}  "
                f"KL={e['kl_mean']:.4f}±{e['kl_std']:.4f}  "
                f"||dl||={e['dlogit_mean']:.2f}±{e['dlogit_std']:.2f}  "
                f"flip={e['flip_count']}/{e['n']}")
        if "attn_shift" in e:
            line += f"  attn_fro={e['attn_shift']['overall_fro_mean']:.4f}"
        print(line)
    print(f"  [linear]  KL vs r/d:        slope={fit_kl['slope']:.4f}  R²={fit_kl['r_squared']:.4f}")
    print(f"  [linear]  ||dl|| vs √(r/d): slope={fit_dl_sqrt['slope']:.4f}  R²={fit_dl_sqrt['r_squared']:.4f}")
    print(f"  [power]   log KL vs log(r/d):    slope={power_kl['slope']:.4f}  R²={power_kl['r_squared']:.4f}")
    print(f"  [power]   log ||dl|| vs log(r/d): slope={power_dl['slope']:.4f}  R²={power_dl['r_squared']:.4f}")
    print(f"  [power]   log attn_fro vs log(r/d): slope={power_attn['slope']:.4f}  R²={power_attn['r_squared']:.4f}  (Ledoux expect 0.5)")
    print(f"  [decision] KL R²>0.85 → {decision_kl}")

    print(f"\n========== (b) margin m(q) distribution ==========")
    print(f"  N={margin_stats['n']}  mean={margin_stats['mean']:.3f}  std={margin_stats['std']:.3f}")
    print(f"  min={margin_stats['min']:.3f}  max={margin_stats['max']:.3f}")
    q = margin_stats["quantiles"]
    if q:
        print(f"  quantiles: q10={q['q10']:.3f}  q25={q['q25']:.3f}  q50={q['q50']:.3f}  "
              f"q75={q['q75']:.3f}  q90={q['q90']:.3f}")

    if args.measure_attn:
        print(f"\n========== (c) attention-weight shift ==========")
        for r in args.ranks:
            e = agg[str(r)]
            if "attn_shift" in e:
                a = e["attn_shift"]
                print(f"  r={r:>3}  Fro_mean={a['overall_fro_mean']:.4f}  "
                      f"L1_mean={a['overall_l1_mean']:.4f}  "
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
        "power_law_attn_fro": power_attn,
        "decision_kl_linear": decision_kl,
        "per_task": per_task,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
