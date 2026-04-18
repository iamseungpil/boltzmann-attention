#!/usr/bin/env python3
"""B1 — Cross-benchmark direction specificity (variant A vs D, matched rank).

For each (M, V) pair:
    - variant A: projection = B_ont @ B_ont^T at first r_bont cols of each
      per-(layer, head) basis (default r_bont = 12 for Qwen τ²)
    - variant D: Haar-random orthonormal (already measured per-benchmark in
      Phase A with the same α and sel_layers)

Cross-benchmark mode: --b-ont points to a foreign benchmark's B_ont
(e.g., Qwen Telecom B_ont applied to Retail prompts).  This tests whether
direction specificity transfers across task distributions.

For argmax flip rate at the prompt's last token + full-vocabulary KL we
reuse the same measurement logic as ``measure_lemma_empirical.py`` / ``_st4``.

Output:
    reports/new_theorem_test/b1_<label>.json
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
from eval_tau2_bench import (
    build_chat_prompt_marked,
    build_tools_json,
    extract_domain_tools,
)
from measure_lemma_empirical import (
    parse_layers_spec,
    make_kbias_hook,
    forward_capture,
    attn_shift_summary,
    quantiles,
)


# --- ST4 prompt helpers (mirror measure_lemma_empirical_st4) ---
CAND_RE = re.compile(r"\d+\.\s+tool name:\s+([A-Za-z0-9_\-\+\.]+)")
QUERY_RE = re.compile(r"\[User's Query Start\]\n(.*?)\n\[User's Query End\]", re.DOTALL)


def build_st4_prompt_marked(tokenizer, entry, marker="**"):
    action_prompt = entry["action_prompt"]
    cands = CAND_RE.findall(action_prompt)
    m = QUERY_RE.search(action_prompt)
    query = m.group(1).strip() if m else action_prompt[:200]
    tool_list = ", ".join(cands)
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


def load_bont_projection(b_ont_path: str, sel_layers, device, dtype, r_override=None):
    """Build per-(L_sel, head) projection P = B_ont @ B_ont^T.

    B_ont shape (L, H, d, R).  We project to rank r = min(R, r_override).
    """
    obj = torch.load(b_ont_path, map_location="cpu", weights_only=False)
    B = obj["B_ont"] if isinstance(obj, dict) else obj
    L, H, d, R = B.shape
    r = R if r_override is None else min(R, r_override)
    print(f"[b_ont] loaded {b_ont_path}: L={L} H={H} d={d} R={R}  using r={r}")
    B_sel = B[sel_layers, :, :, :r].to(device=device, dtype=torch.float32)
    # Project onto span
    P = torch.einsum("lhdr,lher->lhde", B_sel, B_sel)
    return P.to(dtype=dtype), r


def build_haar_projection(L_sel, n_kv, d, r, seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.empty(L_sel, n_kv, d, d, dtype=torch.float32)
    for li in range(L_sel):
        for h in range(n_kv):
            A = torch.randn(d, d, generator=g)
            Q, _ = torch.linalg.qr(A)
            Ur = Q[:, :r].contiguous()
            proj[li, h] = Ur @ Ur.T
    return proj.to(device=device, dtype=dtype)


# --- Main measurement: apply a given projection, compute flip / KL / margin ---

@torch.no_grad()
def measure_variant(model, tok, tasks_iter, prompt_builder, proj, sel_layers,
                    n_kv, head_dim, amp, capture_attn, capture_limit=None):
    """Iterate tasks, apply projection-based K-bias hook, collect metrics."""
    per_task = []
    device = next(model.parameters()).device
    t0 = time.time()
    for idx, entry in enumerate(tasks_iter):
        try:
            prompt_marked = prompt_builder(tok, entry)
            ids, steer_mask = build_marker_steer_mask(prompt_marked, tok)
            ids = ids.to(device)
            mask_b = steer_mask[0].to(device)
            if mask_b.sum().item() == 0:
                continue
            # baseline
            logit_base, attn_base = forward_capture(model, ids, capture_attn=capture_attn)
            logp_base = F.log_softmax(logit_base, dim=-1)
            argmax_base = int(logit_base.argmax().item())
            top2 = torch.topk(logit_base, 2)
            m_q = float((top2.values[0] - top2.values[1]).item())

            # install hook + forward
            handles = []
            for i, L in enumerate(sel_layers):
                attn = model.model.layers[L].self_attn
                module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
                handles.append(module.register_forward_hook(
                    make_kbias_hook(i, proj, n_kv, head_dim, mask_b, amp)
                ))
            try:
                logit_p, attn_p = forward_capture(model, ids, capture_attn=capture_attn)
            finally:
                for h in handles:
                    h.remove()

            logp_p = F.log_softmax(logit_p, dim=-1)
            p_p = logp_p.exp()
            kl = float((p_p * (logp_p - logp_base)).sum().item())
            d_logit = float((logit_p - logit_base).norm().item())
            argmax_p = int(logit_p.argmax().item())
            shift = attn_shift_summary(attn_base, attn_p, sel_layers) if capture_attn else None

            per_task.append({
                "idx": idx,
                "margin": m_q,
                "kl": kl,
                "d_logit": d_logit,
                "flipped": int(argmax_p != argmax_base),
                "attn_shift_fro_mean": shift["overall_fro_mean"] if shift else None,
                "argmax_base_token": argmax_base,
                "argmax_pert_token": argmax_p,
            })
            if capture_limit is not None and len(per_task) >= capture_limit:
                break
            if (len(per_task)) % 20 == 0:
                elapsed = time.time() - t0
                flips = sum(p["flipped"] for p in per_task)
                print(f"   [{len(per_task)}] t={elapsed:.1f}s  flips={flips}  mean_kl={statistics.mean(p['kl'] for p in per_task):.4f}", flush=True)
        except Exception as exc:
            print(f"   [{idx}] ERR {type(exc).__name__}: {exc}", flush=True)
    return per_task


def summarize(per_task):
    if not per_task:
        return None
    kls = [p["kl"] for p in per_task]
    dls = [p["d_logit"] for p in per_task]
    flips = [p["flipped"] for p in per_task]
    margins = [p["margin"] for p in per_task]
    fro = [p["attn_shift_fro_mean"] for p in per_task if p["attn_shift_fro_mean"] is not None]
    return {
        "n": len(per_task),
        "flip_count": sum(flips),
        "flip_rate": sum(flips) / len(flips),
        "kl_mean": statistics.mean(kls),
        "kl_std": statistics.pstdev(kls),
        "dlogit_mean": statistics.mean(dls),
        "margin_mean": statistics.mean(margins),
        "margin_min": min(margins),
        "attn_fro_mean": statistics.mean(fro) if fro else None,
    }


# --- Bench-specific loaders ---

def load_tau2_tasks(path, n):
    tasks_all = json.load(open(path))
    return tasks_all[:n], extract_domain_tools(tasks_all)


def tau2_prompt_builder_factory(tools_json):
    def builder(tok, task):
        return build_chat_prompt_marked(tok, task, tools_json)
    return builder


def st4_prompt_builder(tok, entry):
    return build_st4_prompt_marked(tok, entry)


def load_st4_entries(path, n):
    data = json.load(open(path))
    return data[:n], None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--b-ont", required=True, help="Path to B_ont.pt used for variant A")
    p.add_argument("--bench", required=True,
                   choices=["tau2_telecom", "tau2_retail", "tau2_airline",
                            "tau2_banking", "metatool_st4"],
                   help="Evaluation benchmark")
    p.add_argument("--layers-spec", default="last10")
    p.add_argument("--amp", type=float, default=0.3)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--r-override", type=int, default=None,
                   help="Override B_ont rank (default: use full R)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-random-control", action="store_true",
                   help="Skip variant D run (if already have Phase A data)")
    p.add_argument("--capture-attn", action="store_true", default=True)
    p.add_argument("--no-capture-attn", dest="capture_attn", action="store_false")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    BENCH_DATASETS = {
        "tau2_telecom": "external/tau2-bench/data/tau2/domains/telecom/tasks.json",
        "tau2_retail": "external/tau2-bench/data/tau2/domains/retail/tasks.json",
        "tau2_airline": "external/tau2-bench/data/tau2/domains/airline/tasks.json",
        "tau2_banking": "external/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json",
        "metatool_st4": "/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json",
    }

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager",
    ).to(args.device)
    model.eval()

    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_q
    L_total = len(model.model.layers)
    sel_layers = parse_layers_spec(args.layers_spec, L_total)
    print(f"[config] n_q={n_q} n_kv={n_kv} d={head_dim} L_total={L_total} sel_layers={sel_layers}")

    # Load dataset + prompt builder
    ds_path = BENCH_DATASETS[args.bench]
    if args.bench == "metatool_st4":
        tasks, _ = load_st4_entries(ds_path, args.n)
        prompt_builder = st4_prompt_builder
    else:
        tasks_all_for_tools = json.load(open(ds_path))
        tasks = tasks_all_for_tools[: args.n]
        domain_tools = extract_domain_tools(tasks_all_for_tools)
        tools_json = build_tools_json(domain_tools)
        prompt_builder = tau2_prompt_builder_factory(tools_json)
    print(f"[bench] {args.bench} N={len(tasks)}")

    # Variant A (B_ont projection)
    proj_A, r_used = load_bont_projection(
        args.b_ont, sel_layers,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
        r_override=args.r_override,
    )
    print(f"\n=== Variant A (B_ont) ===")
    A_per_task = measure_variant(
        model, tok, tasks, prompt_builder, proj_A, sel_layers,
        n_kv, head_dim, args.amp, capture_attn=args.capture_attn,
    )
    summary_A = summarize(A_per_task)
    print(f"[A] {summary_A}")

    # Variant D (random orthonormal, same rank)
    summary_D = None
    D_per_task = []
    if not args.no_random_control:
        print(f"\n=== Variant D (random orthonormal, r={r_used}) ===")
        proj_D = build_haar_projection(
            len(sel_layers), n_kv, head_dim, r_used, args.seed,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        D_per_task = measure_variant(
            model, tok, tasks, prompt_builder, proj_D, sel_layers,
            n_kv, head_dim, args.amp, capture_attn=args.capture_attn,
        )
        summary_D = summarize(D_per_task)
        print(f"[D] {summary_D}")

    # Aggregate + gap
    gap = None
    if summary_A is not None and summary_D is not None:
        gap = {
            "flip_rate_A_minus_D": summary_A["flip_rate"] - summary_D["flip_rate"],
            "kl_A_over_D_ratio": (summary_A["kl_mean"] / summary_D["kl_mean"]) if summary_D["kl_mean"] > 0 else float("inf"),
            "flip_count_A": summary_A["flip_count"],
            "flip_count_D": summary_D["flip_count"],
        }

    print("\n========== B1 Cross-bench summary ==========")
    print(f"Bench: {args.bench}  B_ont: {args.b_ont}  r={r_used}  amp={args.amp}")
    print(f"Variant A:  flip={summary_A['flip_count']}/{summary_A['n']} "
          f"({summary_A['flip_rate']*100:.2f}%)  "
          f"KL={summary_A['kl_mean']:.4f}  m̄={summary_A['margin_mean']:.3f}")
    if summary_D:
        print(f"Variant D:  flip={summary_D['flip_count']}/{summary_D['n']} "
              f"({summary_D['flip_rate']*100:.2f}%)  "
              f"KL={summary_D['kl_mean']:.4f}")
    if gap:
        print(f"Gap:        A-D flip={gap['flip_rate_A_minus_D']*100:+.2f}pp  "
              f"KL ratio A/D={gap['kl_A_over_D_ratio']:.2f}")

    out = {
        "args": vars(args),
        "head_dim": head_dim,
        "n_kv": n_kv,
        "sel_layers": sel_layers,
        "r_bont_used": r_used,
        "summary_A": summary_A,
        "summary_D": summary_D,
        "gap": gap,
        "per_task_A": A_per_task,
        "per_task_D": D_per_task,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
