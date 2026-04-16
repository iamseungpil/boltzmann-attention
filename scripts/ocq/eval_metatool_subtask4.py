#!/usr/bin/env python3
"""eval_metatool_subtask4.py — Multi-tool FC evaluation on MetaTool Subtask4 (497 × 2-tool GT).

For each query:
  1. Build chat-template prompt with the 10 candidate tools as a JSON tool schema.
  2. Generate with `<tool_call>{"name": X, "arguments": {...}}</tool_call>` structured output.
  3. Extract ALL `name` fields from the generation (regex over <tool_call>...</tool_call>).
  4. Compare to ground truth (list of 2 tool names). Compute:
     F1, F_0.5, EU(α=1,β=2,γ=1), Jaccard, Exact-set.
  5. (Optional) Facet-graded metrics if `--facet-map` provided (JSON tool→facet-tuple).

Usage (smoke, Qwen-Instruct, N=20):
  python scripts/ocq/eval_metatool_subtask4.py \
    --model Qwen/Qwen2.5-7B-Instruct --device cuda:0 \
    --methods no_steer ocq_bias_a0.3 \
    --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \
    --max-samples 20 \
    --out reports/subtask4_smoke/qwen_real_n20.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Reuse K-bias hooks from Subtask1 driver
from eval_metatool_subtask1 import (
    install_kbias_hooks,
    install_facet_gated_hooks,
    install_vbias_hooks,
    install_kvbias_hooks,
    install_normalized_kbias_hooks,
    install_contrastive_kbias_hooks,
    install_q_bias_hooks,
    install_qkv_joint_hooks,
    install_layer_adaptive_qk_hooks,
    install_caa_hooks,
    install_adaseka_proxy_hooks,
    parse_candidates,
    parse_method,
    build_facet_masks,
    _nullcontext,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument(
        "--dataset",
        default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json",
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["no_steer", "ocq_bias_a0.3"],
        help=(
            "Methods to compare. Examples: no_steer, ocq_bias_a0.3, "
            "ocq_qbias_b-0.1, ocq_qkv_a0.025_v0_q-0.1, "
            "ocq_qk_layered_a0.3_q-0.1."
        ),
    )
    p.add_argument(
        "--b-ont",
        type=str,
        default="",
        help="B_ont .pt path for all ontology-steered methods.",
    )
    p.add_argument("--facet-map", type=str, default="",
                   help="Optional JSON: {tool_name: [intent, io_type, domain, tool_category]}")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-heads", type=str, default="",
                   help="Skip heads spec (e.g. 'L0-L21' skips all heads in layers 0..21)")
    return p.parse_args()


def build_fc_prompt(tokenizer, action_prompt: str, candidates: List[str]) -> str:
    """Build FC-style chat prompt.

    We inject a system message that instructs the model to emit one or more
    `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` blocks per
    query, using ONLY the listed candidate tool names. This bypasses the
    `tools` parameter (whose exact schema varies across tokenizer versions)
    while still eliciting structured output from Qwen-Instruct / Mistral-
    Instruct / Llama-Instruct chat templates.
    """
    # Extract the user query from action_prompt (last 'User query:' line)
    m = re.search(r'User query:\s*"([^"]+)"\s*tool:\s*$', action_prompt)
    user_query = m.group(1) if m else action_prompt.split("User query:")[-1].strip().split('"')[1]

    tool_list = ", ".join(candidates)
    system_msg = (
        "You are a tool-selection agent. Given a user query, emit ONE OR MORE "
        f"<tool_call> blocks naming tools from this list: [{tool_list}]. "
        "Format each tool call exactly as:\n"
        '<tool_call>{"name": "ToolName", "arguments": {}}</tool_call>\n'
        "Emit MULTIPLE blocks if the query needs multiple tools. "
        "Do not include explanations. Output ONLY the <tool_call> blocks."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*\{[^}]*?"name"\s*:\s*"([^"]+)"', re.DOTALL
)


def extract_tool_names(generation: str, candidates: List[str]) -> List[str]:
    """Extract tool names from FC-style generation. Filter to candidates only."""
    raw = TOOL_CALL_RE.findall(generation)
    cand_set = {c.lower(): c for c in candidates}
    out = []
    for name in raw:
        key = name.lower()
        if key in cand_set and cand_set[key] not in out:
            out.append(cand_set[key])
    return out


def compute_metrics(
    pred: List[str], gt: List[str],
    facet_map: Optional[Dict[str, List[str]]] = None,
    alpha: float = 1.0, beta_cost: float = 2.0, gamma: float = 1.0,
) -> Dict[str, float]:
    P = set(pred)
    G = set(gt)
    TP = len(P & G)
    FP = len(P - G)
    FN = len(G - P)

    precision = TP / len(P) if P else 0.0
    recall = TP / len(G) if G else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
    eu_num = alpha * TP - beta_cost * FP - gamma * FN
    eu = max(0.0, eu_num / (alpha * len(G))) if G else 0.0
    jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    exact = 1.0 if P == G else 0.0

    metrics = {
        "precision": precision, "recall": recall,
        "F1": f1, "F_0.5": f05, "EU": eu,
        "Jaccard": jaccard, "Exact": exact,
        "TP": TP, "FP": FP, "FN": FN,
    }

    if facet_map:
        # Facet-graded similarity
        def graded(p: str, g: str) -> float:
            if p == g: return 1.0
            phi_p = facet_map.get(p); phi_g = facet_map.get(g)
            if not phi_p or not phi_g: return 0.0
            return 0.5 if any(a == b for a, b in zip(phi_p, phi_g)) else 0.0

        # Bipartite-matching sum
        fg_p = sum(max((graded(p, g) for g in G), default=0.0) for p in P)
        fg_g = sum(max((graded(p, g) for p in P), default=0.0) for g in G)
        fg_f1 = (2 * fg_p * fg_g / (fg_p + fg_g)) if (fg_p + fg_g) > 0 else 0.0
        fg_f05_num = 1.25 * fg_p * fg_g
        fg_f05_den = 0.25 * fg_p + fg_g
        fg_f05 = fg_f05_num / fg_f05_den if fg_f05_den > 0 else 0.0
        # FG-EU
        cross_fp = sum(1 for p in P - G
                       if facet_map.get(p) and all(a != b for a, b in
                           zip(facet_map[p], facet_map.get(next(iter(G)), ['', '', '', '']))))
        fg_eu = max(0.0, (alpha * fg_p - beta_cost * cross_fp - gamma * FN) / (alpha * len(G))) if G else 0.0
        metrics.update({
            "FG-F1": fg_f1, "FG-F_0.5": fg_f05, "FG-EU": fg_eu,
            "fg_p": fg_p, "fg_g": fg_g,
        })

    return metrics


@torch.no_grad()
def run_method(
    model, tokenizer, data: List[dict], method: str, args,
    B_ont: Optional[torch.Tensor], n_kv: int, head_dim: int,
    facet_map: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    kind, params = parse_method(method)

    skip_heads = getattr(args, "_skip_heads_set", None)

    if kind == "no_steer":
        ctx = _nullcontext()
    elif kind == "bias":
        ctx = install_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=skip_heads,
        )
    elif kind == "normbias":
        ctx = install_normalized_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
        )
    elif kind == "cbias":
        ctx = install_contrastive_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            dominant_rank=params["dominant_rank"],
            n_kv=n_kv, head_dim=head_dim,
        )
    elif kind == "vbias":
        ctx = install_vbias_hooks(
            model, B_ont, alpha=params["alpha_v"],
            n_kv=n_kv, head_dim=head_dim,
        )
    elif kind == "kvbias":
        ctx = install_kvbias_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"], alpha_v=params["alpha_v"],
            n_kv=n_kv, head_dim=head_dim,
        )
    elif kind == "qbias":
        n_q = model.config.num_attention_heads
        ctx = install_q_bias_hooks(
            model, B_ont, beta=params["beta"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
        )
    elif kind == "qkv_joint":
        n_q = model.config.num_attention_heads
        ctx = install_qkv_joint_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"],
            gamma_v=params["gamma_v"],
            beta_q=params["beta_q"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
        )
    elif kind == "layer_adaptive_qk":
        n_q = model.config.num_attention_heads
        ctx = install_layer_adaptive_qk_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"],
            beta_q=params["beta_q"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
        )
    elif kind == "caa":
        ctx = install_caa_hooks(model, B_ont, alpha=params["alpha"])
    elif kind == "adaseka":
        n_q = model.config.num_attention_heads
        ctx = install_adaseka_proxy_hooks(
            model, B_ont, alpha=params["alpha"], M=params["M"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            temperature=params["T"],
        )
    else:
        raise ValueError(f"{kind} not supported in Subtask4 driver")

    per_sample = []
    aggregated = {k: 0.0 for k in ["F1", "F_0.5", "EU", "Jaccard", "Exact", "precision", "recall"]}
    if facet_map:
        for k in ["FG-F1", "FG-F_0.5", "FG-EU"]:
            aggregated[k] = 0.0

    t0 = time.time()
    with ctx:
        for i, entry in enumerate(data):
            action_prompt = entry["action_prompt"]
            gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
            cands = parse_candidates(action_prompt)

            fc_prompt = build_fc_prompt(tokenizer, action_prompt, cands)
            ids = tokenizer(fc_prompt, return_tensors="pt").to(args.device)
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = out[0, ids["input_ids"].shape[1]:]
            generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
            pred = extract_tool_names(generation, cands)
            metrics = compute_metrics(pred, gt, facet_map=facet_map)

            for k in aggregated:
                aggregated[k] += metrics[k]

            per_sample.append({
                "index": entry.get("index", i),
                "query": entry.get("query", "")[:150],
                "gt": gt,
                "pred": pred,
                "generation_head": generation[:300],
                "metrics": metrics,
            })
            if args.verbose and i < 3:
                print(f"[{i}] gt={gt} pred={pred} F1={metrics['F1']:.3f}", flush=True)

    runtime = time.time() - t0
    N = len(data)
    for k in aggregated:
        aggregated[k] = aggregated[k] / max(N, 1)

    return {
        "method": method,
        "n_queries": N,
        "macro": aggregated,
        "runtime_s": runtime,
        "per_sample": per_sample,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[load] {args.model}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = resolve_dtype(args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    L = cfg.num_hidden_layers

    # Parse skip-heads spec (uses parse_skip_heads imported from subtask1)
    from eval_metatool_subtask1 import parse_skip_heads
    if args.skip_heads:
        args._skip_heads_set = parse_skip_heads(args.skip_heads, n_kv)
        print(f"[skip-heads] spec='{args.skip_heads}' -> {len(args._skip_heads_set)} (layer,head) pairs skipped", flush=True)
    else:
        args._skip_heads_set = None

    with open(args.dataset) as f:
        data = json.load(f)
    start = max(0, args.start_idx)
    end = start + args.max_samples if args.max_samples > 0 else len(data)
    data = data[start:end]
    print(f"[data] {len(data)} queries (Subtask4 multi-tool)", flush=True)

    facet_map = None
    if args.facet_map:
        facet_map = json.load(open(args.facet_map))

    needs_b_ont = any(m != "no_steer" for m in args.methods)
    B_ont = None
    if needs_b_ont:
        if not args.b_ont:
            raise ValueError("--b-ont required for non-no_steer methods")
        payload = torch.load(args.b_ont, map_location="cpu", weights_only=False)
        B_ont = payload["B_ont"] if isinstance(payload, dict) else payload

    results = []
    for method in args.methods:
        print(f"\n[eval] {method}", flush=True)
        r = run_method(model, tok, data, method, args, B_ont, n_kv, head_dim, facet_map)
        print(
            f"[eval] {method}: F1={r['macro']['F1']:.3f} "
            f"F0.5={r['macro']['F_0.5']:.3f} EU={r['macro']['EU']:.3f} "
            f"Jac={r['macro']['Jaccard']:.3f} Exact={r['macro']['Exact']:.3f} "
            f"runtime={r['runtime_s']:.1f}s",
            flush=True,
        )
        results.append(r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "n_queries": len(data),
        "methods": args.methods,
        "facet_map": args.facet_map or None,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
