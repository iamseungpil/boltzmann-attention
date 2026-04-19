"""MetaTool Subtask4 eval with canonical AdaptiveSEKALLM.

Uses AdaSEKA's original dynamic routing over per-facet experts built from our
B_ont (via build_adaseka_experts_from_bont.py). This is the Phase 2B canonical
comparison: AdaSEKA operator form with our per-facet decomposition as expert list.

Usage:
  python eval_subtask4_with_adaseka.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --expert-path external/SEKA/seka_projections/adaseka-qwen25-7b-metatool/expert_paths.json \
    --amplify 1.0 3.0 5.0 \
    --out reports/adaseka_full_eval_2026_04_16/qwen_st4_adaseka_full497.json
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
from typing import List, Dict

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "ocq"))
sys.path.insert(0, str(REPO / "external" / "SEKA"))

from src.model import AdaptiveSEKALLM
from eval_metatool_subtask1 import parse_candidates
from eval_metatool_subtask4 import (
    build_fc_prompt, extract_tool_names, compute_metrics,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--expert-path", required=True,
                   help="JSON mapping expert_name → svd.pt path")
    p.add_argument("--dataset", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--amplify", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    p.add_argument("--layers", default="last10")
    p.add_argument("--marker-start", default="**")
    p.add_argument("--marker-end", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--top-k-singular", type=int, default=5)
    p.add_argument("--combination-method", default="weighted_top_k")
    p.add_argument("--out", required=True)
    return p.parse_args()


def wrap_user(prompt: str, m_start="**", m_end=None) -> str:
    import re
    m_end = m_end or m_start
    for pat in [
        r'(<\|im_start\|>user\n)(.*?)(<\|im_end\|>)',
        r'(<\|start_header_id\|>user<\|end_header_id\|>\n\n)(.*?)(<\|eot_id\|>)',
        r'(\[INST\]\s*)(.*?)(\s*\[/INST\])',
    ]:
        new = re.sub(pat, lambda m: f"{m.group(1)}{m_start}{m.group(2)}{m_end}{m.group(3)}",
                     prompt, count=1, flags=re.DOTALL)
        if new != prompt:
            return new
    print(f"[WARN] no user-wrap matched: {prompt[:100]}", flush=True)
    return prompt


def run_eval(ada, data, amp_factor, max_new_tokens, m_start, m_end):
    ada.amplify_factor = amp_factor
    per_sample, agg = [], {k: 0.0 for k in
                           ["F1", "F_0.5", "EU", "Jaccard", "Exact", "precision", "recall"]}
    t0 = time.time()
    for i, entry in enumerate(data):
        ap = entry["action_prompt"]
        gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
        cands = parse_candidates(ap)
        fc = build_fc_prompt(ada.tok, ap, cands)
        wrapped = wrap_user(fc, m_start, m_end or m_start)
        try:
            out = ada.generate(wrapped, steer=True, return_raw=False,
                               max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=ada.tok.eos_token_id)
            gen = out if isinstance(out, str) else out[0]
        except Exception as e:
            gen = f"[ERROR] {e!r}"
        pred = extract_tool_names(gen, cands)
        m = compute_metrics(pred, gt)
        for k in agg:
            agg[k] += m[k]
        per_sample.append({"index": entry.get("index", i), "gt": gt, "pred": pred,
                           "generation_head": gen[:300], "metrics": m})
        if i % 20 == 0:
            print(f"  [amp={amp_factor} {i}/{len(data)}] gt={gt[:2]} pred={pred[:2]} F1={m['F1']:.3f}",
                  flush=True)
    N = len(data)
    for k in agg:
        agg[k] /= max(N, 1)
    print(f"[eval] amp={amp_factor} F1={agg['F1']:.4f} runtime={time.time()-t0:.0f}s", flush=True)
    return {"method": f"adaseka_amp{amp_factor}", "amplify_factor": amp_factor,
            "macro": agg, "per_sample": per_sample, "runtime_s": time.time() - t0, "n_queries": N}


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    expert_paths = json.load(open(args.expert_path))
    # AdaSEKA expects paths relative to external/SEKA/ or absolute
    # Resolve relative to this script's CWD
    print(f"[adaseka] expert_paths: {expert_paths}", flush=True)

    # Change to external/SEKA so relative paths resolve
    import os
    os.chdir(REPO / "external" / "SEKA")

    print(f"[adaseka] loading {args.model}", flush=True)
    t0 = time.time()
    ada = AdaptiveSEKALLM(
        args.model, expert_paths=expert_paths,
        marker_start=args.marker_start, marker_end=args.marker_end,
        layers=args.layers,
        top_k_singular=args.top_k_singular,
        combination_method=args.combination_method,
        amplify_factor=1.0,  # override per-eval
        torch_dtype=torch.bfloat16,
        device=args.device,
    )
    print(f"[adaseka] loaded in {time.time()-t0:.1f}s", flush=True)
    if ada.tok.pad_token is None:
        ada.tok.pad_token = ada.tok.eos_token
        print(f"[fix] tok.pad_token = eos_token", flush=True)

    os.chdir(REPO)  # Restore

    data_all = json.load(open(args.dataset))
    data = data_all[:args.max_samples] if args.max_samples > 0 else data_all
    print(f"[data] N={len(data)}", flush=True)

    # Baseline no_steer
    print("\n[eval] no_steer baseline", flush=True)
    ada.amplify_factor = 0.0
    baseline = run_eval(ada, data, 0.0, args.max_new_tokens, args.marker_start, args.marker_end)
    baseline["method"] = "no_steer"
    results = [baseline]

    for amp in args.amplify:
        print(f"\n[eval] AdaSEKA amplify_factor={amp}", flush=True)
        results.append(run_eval(ada, data, amp, args.max_new_tokens, args.marker_start, args.marker_end))

    blob = {"model": args.model, "expert_paths": expert_paths,
            "layers": args.layers, "n_queries": len(data), "results": results}
    with open(out, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
