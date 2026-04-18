"""BFCL canonical AdaSEKA evaluation with variant A (our B_ont) and variant D (random).

Purpose: C3 (B_ont direction load-bearing) cross-benchmark validation.
Tier 3 on Telecom showed: A +28.89pp, D +0.00pp under matched 20% ‖δ‖/‖K‖.
Here we test the same ablation on BFCL parallel_multiple (second facet-diverse multi-tool
benchmark).

Uses τ²-telecom-derived AdaSEKA experts as cross-domain proxy. If variant A still
beats variant D on BFCL, the B_ont direction specificity generalizes across domains
(stronger claim than within-domain specificity).

Usage:
  python eval_bfcl_canonical.py --max-samples 100 \
    --variant-a-path external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom/expert_paths.json \
    --variant-d-path external/SEKA/seka_projections/adaseka-qwen25-7b-tau2-telecom-variantD/expert_paths.json \
    --out reports/bfcl_2026_04_18/bfcl_canonical_tier3_N100.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from canonical_adaseka_engine import (
    CanonicalAdaSEKAEngine,
    build_marker_steer_mask,
)
from eval_bfcl import (
    download_bfcl,
    extract_function_names,
    get_gt_function_names,
    compute_f1,
)


def build_bfcl_prompt_marked(tokenizer, entry, marker="**"):
    """Build BFCL chat prompt with marker around user content for steer_mask localization."""
    question = entry["question"]
    functions = entry["function"]
    if isinstance(question, list) and isinstance(question[0], list):
        messages = question[0]
    elif isinstance(question, list):
        messages = question
    else:
        messages = [{"role": "user", "content": str(question)}]

    tools_desc = []
    for fn in functions:
        tools_desc.append(f"- {fn['name']}: {fn.get('description','')}")
    tools_str = "\n".join(tools_desc)

    # Wrap user content with markers (only first user message)
    wrapped = []
    for m in messages:
        if m.get("role") == "user" and len(wrapped) == 0:
            wrapped.append({"role": "user", "content": f"{marker}{m['content']}{marker}"})
        else:
            wrapped.append(m)

    system_msg = (f"You are a helpful assistant with access to the following tools:\n"
                  f"{tools_str}\nRespond with Python-style function calls, one per line.")
    full = [{"role": "system", "content": system_msg}] + wrapped
    return tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def run_no_steer(model, tok, data, answers, max_new_tokens=256, device="cuda:0"):
    per_sample = []
    t0 = time.time()
    for i, (entry, ans_entry) in enumerate(zip(data, answers)):
        # Use regular prompt (no marker) for no_steer
        q = entry["question"]
        fns = entry["function"]
        if isinstance(q, list) and isinstance(q[0], list):
            messages = q[0]
        elif isinstance(q, list):
            messages = q
        else:
            messages = [{"role": "user", "content": str(q)}]
        tools_desc = "\n".join(f"- {fn['name']}: {fn.get('description','')}" for fn in fns)
        system_msg = f"You are a helpful assistant with access to the following tools:\n{tools_desc}\nRespond with Python-style function calls, one per line."
        full = [{"role": "system", "content": system_msg}] + messages
        prompt = tok.apply_chat_template(full, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        pred = extract_function_names(gen)
        gt = get_gt_function_names(ans_entry)
        m = compute_f1(pred, gt)
        per_sample.append({"id": entry.get("id", i), "pred": pred, "gt": gt, "metrics": m,
                           "gen_head": gen[:300]})
        if (i + 1) % 10 == 0 or i == len(data) - 1:
            avg = sum(s["metrics"]["f1"] for s in per_sample) / len(per_sample)
            print(f"  [no_steer] {i+1}/{len(data)}  running_F1={avg:.4f}", flush=True)
    runtime = time.time() - t0
    macro = {k: sum(s["metrics"][k] for s in per_sample) / len(per_sample)
             for k in ["precision", "recall", "f1", "exact"]}
    return {"method": "no_steer", "n": len(per_sample), "macro": macro,
            "runtime_s": runtime, "per_sample": per_sample}


@torch.no_grad()
def run_canonical(model, tok, data, answers, engine, amp, method_name,
                  max_new_tokens=256, device="cuda:0"):
    per_sample = []
    t0 = time.time()
    for i, (entry, ans_entry) in enumerate(zip(data, answers)):
        prompt_marked = build_bfcl_prompt_marked(tok, entry)
        ids, steer_mask = build_marker_steer_mask(prompt_marked, tok)
        ids = ids.to(device)
        steer_mask = steer_mask.to(device)
        with engine.install_hooks(ids, steer_mask, amplify=amp):
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        pred = extract_function_names(gen)
        gt = get_gt_function_names(ans_entry)
        m = compute_f1(pred, gt)
        per_sample.append({"id": entry.get("id", i), "pred": pred, "gt": gt, "metrics": m,
                           "gen_head": gen[:300], "mask_sum": int(steer_mask.sum().item())})
        if (i + 1) % 10 == 0 or i == len(data) - 1:
            avg = sum(s["metrics"]["f1"] for s in per_sample) / len(per_sample)
            print(f"  [{method_name}] {i+1}/{len(data)}  running_F1={avg:.4f}", flush=True)
    runtime = time.time() - t0
    macro = {k: sum(s["metrics"][k] for s in per_sample) / len(per_sample)
             for k in ["precision", "recall", "f1", "exact"]}
    return {"method": method_name, "n": len(per_sample), "macro": macro,
            "runtime_s": runtime, "per_sample": per_sample}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--subset", default="parallel_multiple")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--variant-a-path", required=True)
    p.add_argument("--variant-d-path", required=True)
    p.add_argument("--adaseka-layers", default="last10")
    p.add_argument("--amp", type=float, default=0.3)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(args.device)
    model.eval()
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_q

    print(f"[bfcl] downloading subset={args.subset}", flush=True)
    data, answers = download_bfcl(args.subset)
    data = data[: args.max_samples]
    answers = answers[: args.max_samples]
    print(f"[bfcl] N={len(data)}", flush=True)

    results = []

    # 1. no_steer baseline
    print(f"\n=== [1/3] no_steer ===", flush=True)
    r = run_no_steer(model, tok, data, answers,
                     max_new_tokens=args.max_new_tokens, device=args.device)
    print(f"  no_steer: F1={r['macro']['f1']:.4f} Exact={r['macro']['exact']:.4f}", flush=True)
    results.append(r)

    # 2. Variant A (our τ²-telecom B_ont experts, cross-domain proxy)
    print(f"\n=== [2/3] canonical_adaseka variant A (τ²-telecom B_ont) ===", flush=True)
    engine_a = CanonicalAdaSEKAEngine(
        model=model, expert_paths_json=args.variant_a_path, layers=args.adaseka_layers,
        topk=args.topk, temperature=args.T,
        n_kv=n_kv, n_q=n_q, head_dim=head_dim,
    )
    r = run_canonical(model, tok, data, answers, engine_a, args.amp,
                       f"canonical_adaseka_amp{args.amp}_variantA",
                       max_new_tokens=args.max_new_tokens, device=args.device)
    print(f"  variantA: F1={r['macro']['f1']:.4f} Exact={r['macro']['exact']:.4f}", flush=True)
    results.append(r)
    del engine_a
    torch.cuda.empty_cache()

    # 3. Variant D (random orthonormal experts)
    print(f"\n=== [3/3] canonical_adaseka variant D (random) ===", flush=True)
    engine_d = CanonicalAdaSEKAEngine(
        model=model, expert_paths_json=args.variant_d_path, layers=args.adaseka_layers,
        topk=args.topk, temperature=args.T,
        n_kv=n_kv, n_q=n_q, head_dim=head_dim,
    )
    r = run_canonical(model, tok, data, answers, engine_d, args.amp,
                       f"canonical_adaseka_amp{args.amp}_variantD",
                       max_new_tokens=args.max_new_tokens, device=args.device)
    print(f"  variantD: F1={r['macro']['f1']:.4f} Exact={r['macro']['exact']:.4f}", flush=True)
    results.append(r)

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {
        "model": args.model,
        "subset": args.subset,
        "n_tasks": len(data),
        "variant_a_path": args.variant_a_path,
        "variant_d_path": args.variant_d_path,
        "amp": args.amp,
        "topk": args.topk,
        "T": args.T,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}", flush=True)

    # Summary
    print("\n" + "=" * 80)
    base = results[0]["macro"]["f1"]
    print(f"{'Method':<45} {'F1':<10} {'ΔF1':<10}")
    print("=" * 80)
    for r in results:
        f1 = r["macro"]["f1"]
        df = (f1 - base) * 100
        print(f"  {r['method']:<43} {f1:<10.4f} {df:<+9.2f}pp")


if __name__ == "__main__":
    main()
