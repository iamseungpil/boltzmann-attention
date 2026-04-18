#!/usr/bin/env python3
"""
Polarity-flip predictor measurement (D1 verification).

Implements the cross-model polarity sign predictor from Lemma 5.A in
``reports/steering_paper/MODEL_SPECIFICITY_MATH_REINFORCEMENT_2026_04_18.md``:

    s := (1/sqrt(d)) * <BB^T q, k_bar_G - k_bar>

Per (layer, q-head, sample) we capture pre-RoPE q at the last prompt token
and pre-RoPE K at all prompt token positions, then compute s. Sample-level
aggregate = mean over (layer, q-head); per (model, domain) prediction =
median sign across samples.

Compares predicted sign to measured best-beta sign:
    Qwen telecom +0.10 ->  +    (under-focused regime)
    Qwen retail  -0.03 ->  -    (coverage regime)
    Llama telecom -0.05 -> -    (cross-model polarity flip on telecom)
    Llama retail  null  ->  ~0  (Llama retail boundary)

CLI:
    python scripts/ocq/measure_polarity_flip_predictor.py \\
        --model NousResearch/Meta-Llama-3.1-8B-Instruct \\
        --device cuda:1 \\
        --b-ont external/SEKA/seka_projections/ontology-llama31-8b-tau2-telecom/B_ont.pt \\
        --domain telecom \\
        --max-samples 50 \\
        --out reports/polarity_flip_2026_04_18/llama31_telecom.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

# Reuse prompt construction + GT extraction from eval_tau2_bench
from eval_tau2_bench import (  # type: ignore
    build_chat_prompt,
    build_tools_json,
    extract_gt_tools,
    extract_domain_tools,
)


# ---------------------------------------------------------------------------
# Capture hooks
# ---------------------------------------------------------------------------

class QKCapture:
    """Captures pre-RoPE q at the last prompt token and pre-RoPE K at all
    prompt token positions, per (layer, head). Holds tensors on CPU as
    float32 numpy arrays after each forward pass."""

    def __init__(self, n_layers: int, n_q: int, n_kv: int, head_dim: int):
        self.n_layers = n_layers
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        # Filled at __call__ time per-sample
        self.q_last: Dict[int, np.ndarray] = {}  # layer -> (n_q, d)
        self.K_all: Dict[int, np.ndarray] = {}   # layer -> (n_kv, T, d)

    def reset(self):
        self.q_last.clear()
        self.K_all.clear()

    def make_q_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # output shape (B, T, n_q*d). B==1 always here.
            if output.dim() != 3:
                return output
            B, T, D = output.shape
            if D != self.n_q * self.head_dim:
                return output
            Q = output.view(B, T, self.n_q, self.head_dim)[0]  # (T, n_q, d)
            q_last = Q[-1].detach().to(torch.float32).cpu().numpy()  # (n_q, d)
            self.q_last[layer_idx] = q_last
            return output
        return hook

    def make_k_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if output.dim() != 3:
                return output
            B, T, D = output.shape
            if D != self.n_kv * self.head_dim:
                return output
            K = output.view(B, T, self.n_kv, self.head_dim)[0]  # (T, n_kv, d)
            K_perm = K.permute(1, 0, 2).contiguous()  # (n_kv, T, d)
            self.K_all[layer_idx] = K_perm.detach().to(torch.float32).cpu().numpy()
            return output
        return hook


@contextmanager
def install_capture_hooks(model, capture: QKCapture):
    handles = []
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= capture.n_layers:
                break
            handles.append(
                layer.self_attn.q_proj.register_forward_hook(
                    capture.make_q_hook(layer_idx)
                )
            )
            handles.append(
                layer.self_attn.k_proj.register_forward_hook(
                    capture.make_k_hook(layer_idx)
                )
            )
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# GT mask construction
# ---------------------------------------------------------------------------

def find_subseq_positions(haystack: List[int], needle: List[int]) -> List[int]:
    """Return all start positions where ``needle`` occurs in ``haystack``."""
    if not needle:
        return []
    positions = []
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            positions.append(i)
    return positions


def build_gt_mask(
    tokenizer,
    input_ids: List[int],
    gt_tool_names: List[str],
) -> np.ndarray:
    """Mark token positions whose token-spans coincide with a GT tool name.

    A GT tool name typically appears once in the system message (tool catalog)
    and possibly multiple times. We mark every occurrence of every GT name.

    Returns a bool array of length ``len(input_ids)``.
    """
    T = len(input_ids)
    mask = np.zeros(T, dtype=bool)
    for name in gt_tool_names:
        # Try multiple candidate tokenisations (with/without leading space,
        # with/without quotes).
        candidates = [name, " " + name, '"' + name + '"', '"' + name]
        for cand in candidates:
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if not ids:
                continue
            for start in find_subseq_positions(input_ids, ids):
                mask[start : start + len(ids)] = True
    return mask


# ---------------------------------------------------------------------------
# Polarity score per sample
# ---------------------------------------------------------------------------

def polarity_score_per_sample(
    capture: QKCapture,
    B_ont: torch.Tensor,         # (L, n_kv, d, r)
    gt_mask: np.ndarray,         # (T,)
) -> Dict[str, float]:
    """Compute s = (1/sqrt(d)) * <BB^T q, k_bar_G - k_bar> per (layer, q-head)
    and aggregate to a single sample-level mean and median.

    Returns dict with sample-level stats and per-(layer, head) signed score
    flattened to a list (for downstream aggregation).
    """
    L, n_kv, d, r = B_ont.shape
    n_q = capture.n_q
    g = n_q // n_kv

    # Sanity: gt_mask length must match captured K length
    T = next(iter(capture.K_all.values())).shape[1]
    assert gt_mask.shape[0] == T, f"gt_mask length {gt_mask.shape[0]} != T {T}"

    if not gt_mask.any():
        return {
            "n_layers": L, "n_q": n_q, "n_kv": n_kv, "d": d, "T": T,
            "gt_token_count": 0,
            "scores": [],
            "sample_mean": float("nan"),
            "sample_median_sign": 0,
        }

    sqrt_d = math.sqrt(d)
    # B_ont as numpy float32 once
    B_np = B_ont.to(torch.float32).cpu().numpy()  # (L, n_kv, d, r)

    scores: List[float] = []
    for layer_idx in range(L):
        if layer_idx not in capture.q_last or layer_idx not in capture.K_all:
            continue
        Q = capture.q_last[layer_idx]   # (n_q, d)
        K = capture.K_all[layer_idx]    # (n_kv, T, d)
        for h_q in range(n_q):
            h_kv = h_q // g
            q = Q[h_q]                  # (d,)
            k = K[h_kv]                 # (T, d)
            B = B_np[layer_idx, h_kv]   # (d, r)

            # baseline softmax weights
            logits = (k @ q) / sqrt_d   # (T,)
            logits -= logits.max()      # numerical stability
            p0 = np.exp(logits)
            p0 /= p0.sum()

            pi_g = float(p0[gt_mask].sum())
            if pi_g <= 0.0:
                continue

            # k_bar and k_bar_g
            k_bar = (p0[:, None] * k).sum(axis=0)
            p_g = p0 * gt_mask
            k_bar_g = (p_g[:, None] * k).sum(axis=0) / pi_g
            delta_k = k_bar_g - k_bar    # (d,)

            # u = B B^T q
            u = B @ (B.T @ q)            # (d,)
            s = float(u @ delta_k) / sqrt_d
            scores.append(s)

    scores_arr = np.array(scores) if scores else np.array([0.0])
    return {
        "n_layers": L, "n_q": n_q, "n_kv": n_kv, "d": d, "T": T,
        "gt_token_count": int(gt_mask.sum()),
        "scores": scores,
        "sample_mean": float(scores_arr.mean()),
        "sample_median_sign": int(np.sign(np.median(scores_arr))),
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--b-ont", required=True, help="path to B_ont.pt (L, n_kv, d, r)")
    p.add_argument("--domain", required=True, choices=["retail", "airline", "telecom"])
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument(
        "--tau2-tasks",
        default="",
        help="path to tau2 tasks JSON; default uses external/tau2-bench/data/tau2/domains/<d>/tasks.json",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] model={args.model} device={args.device}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    print(f"[cfg] L={n_layers} n_q={n_q} n_kv={n_kv} head_dim={head_dim}", flush=True)

    print(f"[load] B_ont {args.b_ont}", flush=True)
    B_ont = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    if isinstance(B_ont, dict):
        B_ont = B_ont.get("B_ont", B_ont.get("B", B_ont))
    L, H_kv, d, r = B_ont.shape
    assert L == n_layers and H_kv == n_kv and d == head_dim, (
        f"B_ont {B_ont.shape} != model ({n_layers}, {n_kv}, {head_dim}, *)"
    )
    print(f"[B_ont] shape {B_ont.shape}", flush=True)

    tasks_path = Path(args.tau2_tasks) if args.tau2_tasks else (
        REPO / "external" / "tau2-bench" / "data" / "tau2"
        / "domains" / args.domain / "tasks.json"
    )
    print(f"[data] {tasks_path}", flush=True)
    with open(tasks_path) as f:
        tasks = json.load(f)
    tasks = tasks[: args.max_samples]
    domain_tools = extract_domain_tools(tasks)
    tools = build_tools_json(domain_tools=domain_tools)
    print(f"[data] {len(tasks)} tasks, {len(domain_tools)} domain tools", flush=True)

    capture = QKCapture(n_layers=n_layers, n_q=n_q, n_kv=n_kv, head_dim=head_dim)

    per_sample: List[dict] = []
    sample_means: List[float] = []
    sample_signs: List[int] = []

    t_start = time.time()
    with torch.inference_mode(), install_capture_hooks(model, capture):
        for idx, task in enumerate(tasks):
            capture.reset()
            try:
                prompt = build_chat_prompt(tokenizer, task, tools)
            except Exception as e:
                print(f"  [{idx}] prompt build failed: {e}", flush=True)
                continue
            input_ids = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(args.device)
            T = int(input_ids.shape[1])

            gt_tools = extract_gt_tools(task)
            gt_mask = build_gt_mask(tokenizer, input_ids[0].cpu().tolist(), gt_tools)

            try:
                model(input_ids=input_ids, use_cache=False)
            except Exception as e:
                print(f"  [{idx}] forward failed: {e}", flush=True)
                continue

            stats = polarity_score_per_sample(capture, B_ont, gt_mask)
            stats["task_id"] = task.get("id", idx)
            stats["n_gt_tools"] = len(gt_tools)
            stats["gt_tools"] = gt_tools
            stats["prompt_len"] = T
            per_sample.append({k: v for k, v in stats.items() if k != "scores"})

            if math.isfinite(stats["sample_mean"]):
                sample_means.append(stats["sample_mean"])
                sample_signs.append(stats["sample_median_sign"])

            if (idx + 1) % 10 == 0 or idx == len(tasks) - 1:
                cur = np.array(sample_means) if sample_means else np.array([0.0])
                cur_sign = (
                    int(np.sign(np.median(cur))) if sample_means else 0
                )
                print(
                    f"  [{idx+1}/{len(tasks)}] median_sign={cur_sign} "
                    f"mean_s={cur.mean():+.4e} "
                    f"frac_pos={(np.array(sample_signs) > 0).mean() if sample_signs else 0:.3f}",
                    flush=True,
                )

    elapsed = time.time() - t_start
    sample_means_arr = np.array(sample_means) if sample_means else np.array([0.0])
    sample_signs_arr = np.array(sample_signs) if sample_signs else np.array([0])

    aggregate = {
        "n_samples_used": len(sample_means),
        "median_s": float(np.median(sample_means_arr)),
        "mean_s": float(sample_means_arr.mean()),
        "predicted_sign": int(np.sign(np.median(sample_means_arr))),
        "frac_positive_samples": float((sample_signs_arr > 0).mean()),
        "frac_negative_samples": float((sample_signs_arr < 0).mean()),
        "frac_zero_samples": float((sample_signs_arr == 0).mean()),
    }

    out = {
        "model": args.model,
        "domain": args.domain,
        "b_ont": args.b_ont,
        "n_tasks_attempted": len(tasks),
        "elapsed_sec": elapsed,
        "config": {
            "n_layers": n_layers, "n_q": n_q, "n_kv": n_kv, "head_dim": head_dim,
            "B_ont_rank": int(B_ont.shape[-1]),
        },
        "aggregate": aggregate,
        "per_sample": per_sample,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] elapsed={elapsed:.1f}s  predicted_sign={aggregate['predicted_sign']}  median_s={aggregate['median_s']:+.4e}", flush=True)
    print(f"[out] {out_path}", flush=True)


if __name__ == "__main__":
    main()
