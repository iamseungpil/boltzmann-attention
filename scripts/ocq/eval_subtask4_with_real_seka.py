#!/usr/bin/env python3
"""eval_subtask4_with_real_seka.py — Evaluate REAL SEKA / AdaSEKA from external/SEKA on MetaTool Subtask4.

Replaces the earlier mislabeled "AdaSEKA proxy" comparison in §5.5.3 with the
actual SEKA implementation. Steps:

1. Convert B_ont (L, H, d, r=24) → P_pos (L, H, d, d) via P = B B^T.
2. Wrap user query in `**...**` markers so SEKA's encode_with_markers selects
   query tokens for steering.
3. Run SEKALLM.generate(...) on each Subtask4 query with sweep of amplify_pos.
4. Parse <tool_call> blocks from output, compute F1/recall/exact metrics.

Usage:
  python scripts/ocq/eval_subtask4_with_real_seka.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --max-samples 20 \
    --amplify 1.0 2.0 5.0 \
    --out reports/baselines_2026_04_15/real_seka_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "external" / "SEKA"))
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from src.model.seka_llm import SEKALLM  # noqa: E402

from eval_metatool_subtask4 import (  # noqa: E402
    build_fc_prompt, extract_tool_names, compute_metrics,
)
from eval_metatool_subtask1 import parse_candidates  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--b-ont", default=None,
                   help="Our B_ont .pt; will be converted to P_pos = B B^T.")
    p.add_argument("--p-pos", default=None,
                   help="Direct path to pre-built SEKA P_pos .pt (skips B_ont conversion).")
    p.add_argument("--p-pos-out", default=None,
                   help="Path to save converted P_pos .pt (default temp).")
    p.add_argument("--dataset",
                   default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--amplify", type=float, nargs="+", default=[1.0, 2.0, 5.0],
                   help="amplify_pos sweep values for SEKA.")
    p.add_argument("--layers", default="last10",
                   help="SEKA layer selector (default: last10).")
    p.add_argument("--marker-start", default="**")
    p.add_argument("--marker-end", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--out", required=True)
    return p.parse_args()


def convert_bont_to_ppos(b_ont_path: str, out_path: Path) -> str:
    """Convert our B_ont (L, H, d, r) to SEKA-format P_pos (L, H, d, d).
    P = B B^T is the rank-r orthogonal projector embedded in d×d."""
    obj = torch.load(b_ont_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        B = obj["B_ont"]  # (L, H, d, r)
    else:
        B = obj
    L, H, d, r = B.shape
    print(f"[convert] B_ont shape {(L, H, d, r)} → P_pos {(L, H, d, d)}", flush=True)
    P_pos = torch.einsum("lhdr,lher->lhde", B.float(), B.float())  # (L, H, d, d)
    print(f"[convert] avg P_pos trace per head = {P_pos.diagonal(dim1=-2, dim2=-1).sum(-1).mean():.3f} (≈ rank r={r})",
          flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # SEKA's _load_proj expects either a tensor or dict with 'proj' key
    torch.save(P_pos, out_path)
    print(f"[convert] saved {out_path}", flush=True)
    return str(out_path)


def wrap_user_query_in_markers(prompt: str, m_start: str = "**", m_end: str = None) -> str:
    """SEKA's encode_with_markers selects tokens between m_start..m_end as the
    steer mask. We wrap the user query content with these markers.

    Supports multiple chat templates:
    - Qwen: <|im_start|>user\n...<|im_end|>
    - Llama-3: <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
    - Mistral: [INST]...[/INST]
    """
    if m_end is None:
        m_end = m_start
    def _wrap(m):
        return f"{m.group(1)}{m_start}{m.group(2)}{m_end}{m.group(3)}"

    # Qwen-style template
    pattern_qwen = r'(<\|im_start\|>user\n)(.*?)(<\|im_end\|>)'
    new_prompt = re.sub(pattern_qwen, _wrap, prompt, count=1, flags=re.DOTALL)
    if new_prompt != prompt:
        return new_prompt

    # Llama-3-style template
    pattern_llama = r'(<\|start_header_id\|>user<\|end_header_id\|>\n\n)(.*?)(<\|eot_id\|>)'
    new_prompt = re.sub(pattern_llama, _wrap, prompt, count=1, flags=re.DOTALL)
    if new_prompt != prompt:
        return new_prompt

    # Mistral-style template
    pattern_mistral = r'(\[INST\]\s*)(.*?)(\s*\[/INST\])'
    new_prompt = re.sub(pattern_mistral, _wrap, prompt, count=1, flags=re.DOTALL)
    if new_prompt != prompt:
        return new_prompt

    # No template matched — warning print
    print(f"[WARN] wrap_user_query_in_markers: no chat template matched! steer_mask will be empty. Prompt head: {prompt[:200]}", flush=True)
    return new_prompt


def run_seka_eval(seka_model: SEKALLM, data: List[dict], amplify_pos: float,
                  max_new_tokens: int, m_start: str, m_end: str) -> Dict:
    seka_model.amplify_pos = amplify_pos
    per_sample = []
    aggregated = {k: 0.0 for k in
                  ["F1", "F_0.5", "EU", "Jaccard", "Exact", "precision", "recall"]}
    t0 = time.time()
    for i, entry in enumerate(data):
        action_prompt = entry["action_prompt"]
        gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
        cands = parse_candidates(action_prompt)
        fc_prompt = build_fc_prompt(seka_model.tok, action_prompt, cands)
        marked_prompt = wrap_user_query_in_markers(fc_prompt, m_start, m_end)
        try:
            out = seka_model.generate(
                marked_prompt, steer=True, return_raw=False,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=seka_model.tok.eos_token_id,
            )
            gen_text = out if isinstance(out, str) else out[0]
        except Exception as e:
            gen_text = f"[ERROR] {e!r}"
        pred = extract_tool_names(gen_text, cands)
        metrics = compute_metrics(pred, gt)
        for k in aggregated:
            aggregated[k] += metrics[k]
        per_sample.append({
            "index": entry.get("index", i),
            "gt": gt, "pred": pred,
            "generation_head": gen_text[:300],
            "metrics": metrics,
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(data)}] gt={gt[:2]} pred={pred[:2]} F1={metrics['F1']:.3f}",
                  flush=True)
    runtime = time.time() - t0
    N = len(data)
    for k in aggregated:
        aggregated[k] = aggregated[k] / max(N, 1)
    return {"amplify_pos": amplify_pos, "n_queries": N,
            "macro": aggregated, "runtime_s": runtime,
            "per_sample": per_sample}


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Convert B_ont → P_pos
    p_pos_path = Path(args.p_pos_out) if args.p_pos_out else \
        Path("/tmp/seka_p_pos_qwen25_7b_metatool.pt")
    convert_bont_to_ppos(args.b_ont, p_pos_path)

    # 2) Load SEKA model
    print(f"[seka] loading model={args.model}", flush=True)
    t0 = time.time()
    seka = SEKALLM(
        args.model, device=args.device,
        marker_start=args.marker_start, marker_end=args.marker_end,
        pos_pt=str(p_pos_path), layers=args.layers,
        amplify_pos=1.0, amplify_neg=0.0,
        torch_dtype=torch.bfloat16,
        attn_implementation=getattr(args, "attn_impl", "sdpa"),
    )
    print(f"[seka] loaded {time.time() - t0:.1f}s", flush=True)

    # Fix tokenizer pad_token bug (SEKA encode_with_markers requires padding)
    if seka.tok.pad_token is None:
        seka.tok.pad_token = seka.tok.eos_token
        print(f"[fix] set tok.pad_token = eos_token ({seka.tok.eos_token})", flush=True)

    # 3) Load Subtask4 data
    data = json.load(open(args.dataset))
    if args.max_samples > 0:
        data = data[:args.max_samples]
    print(f"[data] N={len(data)} queries", flush=True)

    # 4) Run no_steer baseline once for reference
    print("\n[eval] no_steer baseline", flush=True)
    seka.amplify_pos = 0.0  # No steering effectively
    no_steer_per_sample = []
    no_steer_agg = {k: 0.0 for k in
                    ["F1", "F_0.5", "EU", "Jaccard", "Exact", "precision", "recall"]}
    t0 = time.time()
    for i, entry in enumerate(data):
        action_prompt = entry["action_prompt"]
        gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
        cands = parse_candidates(action_prompt)
        fc_prompt = build_fc_prompt(seka.tok, action_prompt, cands)
        ids = seka.tok(fc_prompt, return_tensors="pt").to(seka.device)
        out = seka.model.generate(
            **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=seka.tok.eos_token_id,
        )
        gen_text = seka.tok.decode(
            out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_tool_names(gen_text, cands)
        m = compute_metrics(pred, gt)
        for k in no_steer_agg: no_steer_agg[k] += m[k]
        no_steer_per_sample.append({
            "index": entry.get("index", i), "gt": gt, "pred": pred,
            "generation_head": gen_text[:300], "metrics": m,
        })
    N = len(data)
    for k in no_steer_agg: no_steer_agg[k] /= max(N, 1)
    print(f"[eval] no_steer F1={no_steer_agg['F1']:.3f} runtime={time.time()-t0:.1f}s",
          flush=True)

    # 5) Sweep amplify_pos
    results = [{"method": "no_steer", "amplify_pos": 0.0,
                "n_queries": N, "macro": no_steer_agg,
                "per_sample": no_steer_per_sample}]
    for amp in args.amplify:
        print(f"\n[eval] SEKA amplify_pos={amp}", flush=True)
        r = run_seka_eval(seka, data, amp, args.max_new_tokens,
                          args.marker_start, args.marker_end or args.marker_start)
        m = r["macro"]
        print(f"[eval] amp={amp}: F1={m['F1']:.3f} F0.5={m['F_0.5']:.3f} "
              f"Exact={m['Exact']:.3f} runtime={r['runtime_s']:.1f}s", flush=True)
        results.append({"method": f"real_seka_amp{amp}", **r})

    out_blob = {
        "model": args.model, "b_ont": args.b_ont,
        "p_pos_path": str(p_pos_path),
        "layers": args.layers, "n_queries": N,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out_blob, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
