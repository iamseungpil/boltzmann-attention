#!/usr/bin/env python3
"""
MetaTool Subtask1 evaluation: tool selection with similar choices.

Dataset: /tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json
  995 queries × 10 candidate tools each (199 unique tools)
  Each entry has:
    - action_prompt: full prompt ending with `tool:`
    - tool:         ground-truth tool name
    - query:        the user query (already inside action_prompt)
    - index:        sample id

Eval protocol
-------------
For each entry:
  1. Tokenize action_prompt
  2. Generate up to ``--max-new-tokens`` tokens (greedy default)
  3. Decode the *new* tokens only
  4. Extract the first candidate tool name appearing in the generation
     (case-insensitive substring match against the 10 candidates parsed
     from the prompt itself, longest-first to break "Sudoku" vs "sudoku2"
     style ambiguity)
  5. Compare to ground truth → top-1 accuracy

Methods supported
-----------------
  no_steer        baseline: pure model.generate(), no hook
  ocq_bias_a<α>   forward hook on layer.self_attn.k_proj that adds
                  α * (K @ B_ont) @ B_ont.T to K (boosts attention along
                  ontology directions, the Phase 1.x K-bias mechanism)
  ocq_quant       OCQ K-cache quantization (no bias) — uses
                  scripts/ocq/quantizer.ocq_kivi_quantize at hook time
  ocq_quant_bias  ocq_bias + ocq_quant combined

Phase B Week 1 kill-switch: ≥3pp top-1 lift over no_steer required to
keep Path B alive.

CLI example
-----------
    source /home/woori/workspace_common/CDP/poc/set.env && \\
    python scripts/ocq/eval_metatool_subtask1.py \\
        --model Qwen/Qwen2.5-7B \\
        --device cuda:0 \\
        --methods no_steer ocq_bias_a1 ocq_bias_a3 \\
        --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \\
        --max-samples 50 \\
        --out /tmp/metatool_subtask1_smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from quantizer import ocq_kivi_quantize  # noqa: E402


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument(
        "--dataset",
        default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json",
    )
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap on number of queries (0 = all 995).")
    p.add_argument("--start-idx", type=int, default=0,
                   help="Start index for slicing the dataset (default 0).")
    p.add_argument("--methods", nargs="+",
                   default=["no_steer", "ocq_bias_a1", "ocq_bias_a3"],
                   help="Methods to evaluate. no_steer / ocq_bias_a<α> / "
                        "ocq_quant / ocq_quant_bias.")
    p.add_argument("--b-ont", type=str, default="",
                   help="Path to B_ont .pt file for OCQ methods.")
    p.add_argument("--ocq-quant-bits", type=int, default=4,
                   help="Residual bits for ocq_quant variants.")
    p.add_argument("--ocq-ont-mode", default="1b", choices=["1a", "1b", "1c"])
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump-failures", action="store_true",
                   help="Save all failing samples (no_match / wrong choice) for dip analysis.")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def resolve_dtype(name: str, dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    lowered = name.lower()
    if "qwen" in lowered or "llama" in lowered:
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------
# Prompt parsing and tool extraction
# ---------------------------------------------------------------------

CAND_RE = re.compile(r"\d+\.\s+tool name:\s+([A-Za-z0-9_\-\+\.]+)\s*,")


def parse_candidates(prompt: str) -> List[str]:
    """Pull the 10 candidate tool names out of the action_prompt's
    numbered list. Returns names in their original order."""
    return CAND_RE.findall(prompt)


def extract_choice(generation: str, candidates: List[str]) -> Optional[str]:
    """Find the candidate that appears EARLIEST (smallest character position)
    in the generation. Ties broken by longer-name-first to avoid prefix
    collisions ("Sudoku" vs "Sudoku2" style).

    Also accepts "None" if the model declined to choose any tool.
    """
    if not generation:
        return None
    gen_low = generation.lower()
    earliest_pos = -1
    earliest_cand: Optional[str] = None
    for cand in candidates:
        pos = gen_low.find(cand.lower())
        if pos < 0:
            continue
        if earliest_pos < 0 or pos < earliest_pos or (
            pos == earliest_pos and len(cand) > len(earliest_cand or "")
        ):
            earliest_pos = pos
            earliest_cand = cand
    if earliest_cand is not None:
        return earliest_cand
    # Fall back: model said "None"?
    none_pos = gen_low.find("none")
    if none_pos >= 0:
        return "None"
    return None


# ---------------------------------------------------------------------
# K-bias hook (Phase 1.x style)
# ---------------------------------------------------------------------
#
# At every layer, the hook on k_proj adds α · (K @ B_ont) @ B_ont.T to
# the pre-RoPE K. Geometrically: this amplifies the projection of K
# onto span(B_ont), causing attention to concentrate along ontology
# directions during the resolution forward.
#
# B_ont is per-(layer, head): shape (L, H, d, r_ont).

@contextmanager
def install_kbias_hooks(
    model,
    B_ont: torch.Tensor,    # (L, H, d, r_ont)
    alpha: float,
    n_kv: int,
    head_dim: int,
):
    handles = []
    L, H, d, r = B_ont.shape
    assert H == n_kv and d == head_dim, f"B_ont shape mismatch"
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L:
                break
            k_proj = layer.self_attn.k_proj
            B_ont_layer = B_ont[layer_idx]  # (H, d, r_ont)

            def make_hook(li, B_ont_lh):
                def hook(module, inputs, output):
                    B, T, D = output.shape
                    if D != n_kv * head_dim:
                        return output
                    # (B, T, n_kv*d) -> (B, n_kv, T, d)
                    K = output.view(B, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                    orig_dtype = K.dtype
                    K_f = K.float()
                    # Per-head: K_modified = K + α * (K @ B_ont) @ B_ont.T
                    B_ont_dev = B_ont_lh.to(device=K.device, dtype=torch.float32)
                    # K_f: (B, n_kv, T, d), B_ont_dev: (n_kv, d, r_ont)
                    # einsum to apply per-head
                    coeffs = torch.einsum("bhtd,hdr->bhtr", K_f, B_ont_dev)
                    K_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_ont_dev)
                    K_modified = K_f + alpha * K_proj
                    out = K_modified.permute(0, 2, 1, 3).contiguous().view(B, T, D).to(orig_dtype)
                    return out
                return hook

            handles.append(k_proj.register_forward_hook(make_hook(layer_idx, B_ont_layer)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def install_quant_hooks(
    model,
    B_ont: torch.Tensor,    # (L, H, d, r_ont)
    bits_residual: int,
    ont_mode: str,
    n_kv: int,
    head_dim: int,
    plus_bias_alpha: float = 0.0,
):
    """OCQ quantization (categorical 1-bit on B_ont + KIVI on residual),
    optionally with K-bias addition (alpha > 0)."""
    handles = []
    L, H, d, r = B_ont.shape
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L:
                break
            k_proj = layer.self_attn.k_proj
            B_ont_layer = B_ont[layer_idx]

            def make_hook(li, B_ont_lh):
                def hook(module, inputs, output):
                    B, T, D = output.shape
                    if D != n_kv * head_dim:
                        return output
                    K = output.view(B, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                    orig_dtype = K.dtype
                    K_f = K.float()
                    B_ont_dev = B_ont_lh.to(device=K.device, dtype=torch.float32)

                    # Optional bias first
                    if plus_bias_alpha > 0:
                        coeffs = torch.einsum("bhtd,hdr->bhtr", K_f, B_ont_dev)
                        K_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_ont_dev)
                        K_f = K_f + plus_bias_alpha * K_proj

                    # OCQ quantization (categorical + kivi residual)
                    K_q = ocq_kivi_quantize(
                        K_f, B_ont_dev,
                        bits_residual=bits_residual, ont_mode=ont_mode,
                    )
                    out = K_q.permute(0, 2, 1, 3).contiguous().view(B, T, D).to(orig_dtype)
                    return out
                return hook

            handles.append(k_proj.register_forward_hook(make_hook(layer_idx, B_ont_layer)))
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------
# Method dispatcher
# ---------------------------------------------------------------------

def parse_method(method: str) -> Tuple[str, Dict]:
    """Parse method tag into (kind, params)."""
    if method == "no_steer":
        return "no_steer", {}
    if method.startswith("ocq_bias_a"):
        alpha = float(method[len("ocq_bias_a"):])
        return "bias", {"alpha": alpha}
    if method == "ocq_quant":
        return "quant", {"alpha": 0.0}
    if method.startswith("ocq_quant_bias_a"):
        alpha = float(method[len("ocq_quant_bias_a"):])
        return "quant", {"alpha": alpha}
    if method == "ocq_quant_bias":
        return "quant", {"alpha": 1.0}  # default α=1 for combined
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------

@torch.no_grad()
def run_method(
    model,
    tokenizer,
    data: List[dict],
    method: str,
    args,
    B_ont: Optional[torch.Tensor],
    n_kv: int,
    head_dim: int,
) -> Dict:
    kind, params = parse_method(method)

    def _generate_one(prompt: str) -> str:
        ids = tokenizer(prompt, return_tensors="pt").to(args.device)
        out = model.generate(
            **ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0, ids["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    correct = 0
    total = 0
    none_correct = 0
    none_total = 0
    pred_counts = {"matched": 0, "no_match": 0, "none_pred": 0}
    sample_log = []

    t0 = time.time()
    if kind == "no_steer":
        ctx = _nullcontext()
    elif kind == "bias":
        ctx = install_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
        )
    elif kind == "quant":
        ctx = install_quant_hooks(
            model, B_ont,
            bits_residual=args.ocq_quant_bits,
            ont_mode=args.ocq_ont_mode,
            n_kv=n_kv, head_dim=head_dim,
            plus_bias_alpha=params["alpha"],
        )
    else:
        raise ValueError(kind)

    with ctx:
        for entry in data:
            prompt = entry["action_prompt"]
            gt = entry["tool"]
            cands = parse_candidates(prompt)
            generation = _generate_one(prompt)
            choice = extract_choice(generation, cands)

            is_none_query = (gt == "None")
            if is_none_query:
                none_total += 1
                if choice == "None":
                    none_correct += 1
            else:
                total += 1
                if choice is not None and choice.lower() == gt.lower():
                    correct += 1

            if choice is None:
                pred_counts["no_match"] += 1
            elif choice == "None":
                pred_counts["none_pred"] += 1
            else:
                pred_counts["matched"] += 1

            if args.verbose and len(sample_log) < 5:
                sample_log.append({
                    "index": entry.get("index"),
                    "gt": gt,
                    "cands": cands,
                    "generation": generation[:200],
                    "choice": choice,
                })

    runtime = time.time() - t0
    n_total = total + none_total
    n_correct = correct + none_correct
    return {
        "method": method,
        "n_queries": n_total,
        "top1_correct": n_correct,
        "top1_accuracy": n_correct / max(n_total, 1),
        "tool_queries": total,
        "tool_accuracy": correct / max(total, 1),
        "none_queries": none_total,
        "none_accuracy": none_correct / max(none_total, 1) if none_total else None,
        "pred_counts": pred_counts,
        "runtime_s": runtime,
        "samples": sample_log,
    }


@contextmanager
def _nullcontext():
    yield


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[load] {args.model} on {args.device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = resolve_dtype(args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    cfg = model.config
    n_kv = cfg.num_key_value_heads
    n_q = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    L = cfg.num_hidden_layers
    print(f"[cfg] L={L} n_kv={n_kv} n_q={n_q} head_dim={head_dim}", flush=True)

    print(f"[data] {args.dataset}", flush=True)
    with open(args.dataset) as f:
        data = json.load(f)
    start = max(0, args.start_idx)
    end = start + args.max_samples if args.max_samples > 0 else len(data)
    data = data[start:end]
    print(f"[data] {len(data)} queries (slice [{start}:{start+len(data)}])",
          flush=True)

    # Load B_ont if needed
    needs_b_ont = any(m != "no_steer" for m in args.methods)
    B_ont = None
    if needs_b_ont:
        if not args.b_ont:
            raise ValueError("--b-ont is required for OCQ methods")
        print(f"[ocq] loading B_ont from {args.b_ont}", flush=True)
        payload = torch.load(args.b_ont, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "B_ont" in payload:
            B_ont = payload["B_ont"]
        else:
            B_ont = payload
        if B_ont.shape[:2] != (L, n_kv):
            raise ValueError(
                f"B_ont (L,H)=({B_ont.shape[0]},{B_ont.shape[1]}) "
                f"!= model (L,n_kv)=({L},{n_kv})"
            )
        if B_ont.shape[2] != head_dim:
            raise ValueError(
                f"B_ont head_dim={B_ont.shape[2]} != model head_dim={head_dim}"
            )
        print(f"[ocq] B_ont shape {tuple(B_ont.shape)}", flush=True)

    results = []
    for method in args.methods:
        print(f"\n[eval] {method}", flush=True)
        res = run_method(
            model, tok, data, method, args,
            B_ont=B_ont, n_kv=n_kv, head_dim=head_dim,
        )
        print(
            f"[eval] {method}: top1={res['top1_accuracy']*100:.2f}% "
            f"({res['top1_correct']}/{res['n_queries']}) "
            f"tool_acc={res['tool_accuracy']*100:.2f}% "
            f"runtime={res['runtime_s']:.1f}s "
            f"preds={res['pred_counts']}",
            flush=True,
        )
        results.append(res)

    # Comparison summary
    print("\n=== SUMMARY ===")
    no_steer = next((r for r in results if r["method"] == "no_steer"), None)
    base_acc = no_steer["top1_accuracy"] if no_steer else None
    for r in results:
        delta = ""
        if base_acc is not None and r["method"] != "no_steer":
            d = (r["top1_accuracy"] - base_acc) * 100
            delta = f"  Δ={d:+.2f}pp"
        print(f"  {r['method']:25s}  top1={r['top1_accuracy']*100:6.2f}%"
              f"  ({r['top1_correct']}/{r['n_queries']}){delta}")

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "n_queries": len(data),
        "methods": args.methods,
        "ocq": {
            "b_ont_path": args.b_ont,
            "ont_mode": args.ocq_ont_mode,
            "quant_bits": args.ocq_quant_bits,
        } if needs_b_ont else None,
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
