#!/usr/bin/env python3
"""
ToolBench (StableToolBench solvable_queries) evaluation with Q+K attention steering.

Dataset: external/StableToolBench/solvable_queries/test_instruction/{G1,G2,G3}*.json
  6 subsets × 61-163 queries. Each query has its own api_list (~4-13 APIs) plus
  relevant APIs (GT). 47 categories across the corpus; 814 unique tools used.

Eval protocol
-------------
For each query:
  1. Assemble tools_json from query's api_list PLUS N distractor APIs from the
     cross-query pool (--n-distractors; 0 = baseline, matching native catalog).
  2. Build chat prompt (system + user with query text).
  3. Generate up to --max-new-tokens (greedy).
  4. Parse predicted tool calls as (tool_name, api_name) pairs.
  5. Compute F1 / Exact / Recall / Precision / nDCG against relevant APIs.

Methods: reuses all hook infrastructure from eval_metatool_subtask1 via
eval_tau2_bench.get_steering_context (already proven on τ²).

CLI example
-----------
    python scripts/ocq/eval_toolbench.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --device cuda:0 \\
        --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-toolbench/B_ont.pt \\
        --subsets G1_instruction \\
        --n-distractors 0 \\
        --methods no_steer ocq_qbias_b-0.03 ocq_ladapt_k0.05_q-0.03 \\
        --max-samples 50 \\
        --out reports/toolbench_2026_04_17/g1inst_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_metatool_subtask1 import parse_method, parse_skip_heads, resolve_dtype  # noqa: E402
from eval_tau2_bench import get_steering_context  # noqa: E402


QUERIES_DIR = REPO / "external" / "StableToolBench" / "solvable_queries" / "test_instruction"
ALL_SUBSETS = [
    "G1_category", "G1_instruction", "G1_tool",
    "G2_category", "G2_instruction",
    "G3_instruction",
]


# =====================================================================
# Tool identifier normalization
# =====================================================================
# ToolBench api_names can be long/messy ("/offices/search/:service/...").
# We use a compact canonical id: "tool_name::api_name". For model prompting
# we sanitize the api_name to a valid function identifier; we keep both a
# "display id" (for the model) and a "gt id" (for scoring).

_ID_BAD = re.compile(r"[^A-Za-z0-9_]+")


def sanitize_name(s: str) -> str:
    """Produce a function-identifier-safe name while keeping it human-readable."""
    s = s.strip()
    s = _ID_BAD.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "unnamed"
    if s[0].isdigit():
        s = "_" + s
    return s[:64]


def canonical_pair(tool_name: str, api_name: str) -> Tuple[str, str]:
    """Return (tool_sanitized, api_sanitized). Used for both prompt and GT keys."""
    return sanitize_name(tool_name or ""), sanitize_name(api_name or "")


def display_id(tool: str, api: str) -> str:
    """The identifier that the model sees in the tool list and that we extract
    back from generation: 'tool::api'."""
    t, a = canonical_pair(tool, api)
    return f"{t}__{a}"


# =====================================================================
# Data loading
# =====================================================================

def load_queries(subsets: List[str]) -> List[dict]:
    """Load queries from listed subsets. Each query is annotated with subset name."""
    out = []
    for s in subsets:
        p = QUERIES_DIR / f"{s}.json"
        d = json.loads(p.read_text())
        for q in d:
            q["_subset"] = s
        out.extend(d)
    return out


def build_tool_pool(queries: List[dict]) -> List[dict]:
    """Collect all unique (tool_name, api_name) APIs across all queries.
    Returns a list of normalized api dicts (one per unique display_id)."""
    seen: Dict[str, dict] = {}
    for q in queries:
        for api in q.get("api_list", []):
            disp = display_id(api.get("tool_name", ""), api.get("api_name", ""))
            if disp in seen:
                continue
            seen[disp] = {
                "tool_name": api.get("tool_name", ""),
                "api_name": api.get("api_name", ""),
                "category_name": api.get("category_name", ""),
                "api_description": (api.get("api_description") or "").strip(),
                "required_parameters": api.get("required_parameters", []),
                "optional_parameters": api.get("optional_parameters", []),
                "display_id": disp,
            }
    return list(seen.values())


def make_distractor_sampler(pool: List[dict], seed: int = 0):
    rng = random.Random(seed)

    def sample(exclude_ids: set, n: int) -> List[dict]:
        """Pick n distractors from pool, excluding anything in exclude_ids."""
        candidates = [x for x in pool if x["display_id"] not in exclude_ids]
        if n <= 0:
            return []
        if n >= len(candidates):
            return candidates
        return rng.sample(candidates, n)

    return sample


# =====================================================================
# Prompt construction
# =====================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant that routes user requests to the right APIs. "
    "For each request, determine which tools/APIs you need to call and output "
    "them as function calls.\n\n"
    "IMPORTANT: Output ALL tool calls needed to fully satisfy the request. "
    'Output each call on its own line as JSON:\n'
    '{"name": "<display_id>", "arguments": {...}}\n\n'
    'Use the display_id shown in the tool list; it has the form "TOOLNAME__APINAME".'
)


def _api_params_schema(api: dict) -> dict:
    props: Dict[str, dict] = {}
    required: List[str] = []
    for param in (api.get("required_parameters") or [])[:8]:
        n = sanitize_name(param.get("name", ""))
        if not n:
            continue
        props[n] = {
            "type": "string",
            "description": (param.get("description") or "")[:120],
        }
        required.append(n)
    for param in (api.get("optional_parameters") or [])[:6]:
        n = sanitize_name(param.get("name", ""))
        if not n or n in props:
            continue
        props[n] = {
            "type": "string",
            "description": (param.get("description") or "")[:120],
        }
    return {"type": "object", "properties": props, "required": required}


def build_tools_json(apis: List[dict]) -> List[dict]:
    tools = []
    for api in apis:
        desc = api.get("api_description") or ""
        desc = (desc or f"{api['api_name']} of {api['tool_name']}")[:240]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": api["display_id"],
                    "description": f"[{api.get('category_name','?')}] {desc}",
                    "parameters": _api_params_schema(api),
                },
            }
        )
    return tools


def build_chat_prompt(tokenizer, user_query: str, tools: List[dict]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False,
        )
    except (TypeError, ValueError):
        tools_text = json.dumps([t["function"] for t in tools], indent=2)
        messages[0]["content"] = (
            SYSTEM_PROMPT + "\n\nAvailable tools:\n" + tools_text
        )
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    return prompt


# =====================================================================
# Parsing generation & ground truth
# =====================================================================

NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def extract_predicted_ids(generation: str, valid_ids: set) -> List[str]:
    """Extract predicted display_ids from generation text.
    Strict matches to valid_ids (the per-query tool list) first; then
    fallback substring search with longest-first to avoid prefix collisions.
    """
    found = NAME_RE.findall(generation)
    valid = []
    seen = set()
    for n in found:
        if n in valid_ids and n not in seen:
            valid.append(n)
            seen.add(n)
    if valid:
        return valid
    # Fallback: substring scan (some models omit proper JSON)
    gen_low = generation.lower()
    sorted_ids = sorted(valid_ids, key=len, reverse=True)
    for did in sorted_ids:
        if did.lower() in gen_low and did not in seen:
            valid.append(did)
            seen.add(did)
    return valid


def gt_ids_from_query(q: dict) -> List[str]:
    """Convert 'relevant APIs' field to display_id list (order preserved)."""
    out = []
    seen = set()
    for entry in q.get("relevant APIs", []):
        if len(entry) < 2:
            continue
        did = display_id(entry[0], entry[1])
        if did in seen:
            continue
        seen.add(did)
        out.append(did)
    return out


def compute_metrics(pred_ids: List[str], gt_ids: List[str]) -> Dict[str, float]:
    pred_set = set(pred_ids)
    gt_set = set(gt_ids)
    if not gt_set:
        return {
            "F1": 1.0 if not pred_set else 0.0,
            "Exact": 1 if pred_set == gt_set else 0,
            "Recall": 1.0,
            "Precision": 1.0 if not pred_set else 0.0,
            "GT_subset": 1,
            "nDCG": 1.0 if not pred_set else 0.0,
        }
    tp = len(pred_set & gt_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gt_set)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    exact = 1 if pred_set == gt_set else 0
    gt_sub = 1 if gt_set <= pred_set else 0
    dcg = 0.0
    for i, did in enumerate(pred_ids):
        if did in gt_set:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(gt_set)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {
        "F1": f1,
        "Exact": exact,
        "Recall": rec,
        "Precision": prec,
        "GT_subset": gt_sub,
        "nDCG": ndcg,
    }


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--subsets", nargs="+", default=["G1_instruction"], choices=ALL_SUBSETS + ["all"])
    p.add_argument("--max-samples", type=int, default=0, help="0 = use all queries in chosen subsets")
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--n-distractors", type=int, default=0,
                   help="Number of distractor APIs to inject into each prompt. 0 = native api_list only.")
    p.add_argument("--distractor-seed", type=int, default=0)
    p.add_argument("--methods", nargs="+",
                   default=["no_steer", "ocq_qbias_b-0.03", "ocq_ladapt_k0.05_q-0.03"])
    p.add_argument("--b-ont", type=str, default="")
    p.add_argument("--ocq-quant-bits", type=int, default=4)
    p.add_argument("--ocq-ont-mode", default="1b", choices=["1a", "1b", "1c"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-heads", type=str, default="")
    p.add_argument("--skip-sink-tokens", type=int, default=0)
    p.add_argument("--multipass-n", type=int, default=3)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# =====================================================================
# Eval loop
# =====================================================================

@torch.no_grad()
def run_method(
    model, tokenizer,
    queries: List[dict],
    pool: List[dict],
    method: str,
    args,
    B_ont: Optional[torch.Tensor],
    n_kv: int, n_q: int, head_dim: int, L: int,
    skip_heads=None,
):
    ctx = get_steering_context(
        model, method, args,
        B_ont=B_ont, n_kv=n_kv, n_q=n_q, head_dim=head_dim, L=L,
        facet_mask=None, skip_heads=skip_heads,
    )
    sample = make_distractor_sampler(pool, seed=args.distractor_seed)
    multipass = method.startswith("multipass_")
    max_passes = args.multipass_n if multipass else 1

    all_metrics = []
    per_sample = []
    t0 = time.time()

    with ctx:
        for i, q in enumerate(queries):
            # Build tools_json per query = native api_list + distractors
            native_apis = []
            native_ids = set()
            for api in q.get("api_list", []):
                did = display_id(api.get("tool_name", ""), api.get("api_name", ""))
                if did in native_ids:
                    continue
                native_ids.add(did)
                native_apis.append(
                    {
                        "tool_name": api.get("tool_name", ""),
                        "api_name": api.get("api_name", ""),
                        "category_name": api.get("category_name", ""),
                        "api_description": (api.get("api_description") or "").strip(),
                        "required_parameters": api.get("required_parameters", []),
                        "optional_parameters": api.get("optional_parameters", []),
                        "display_id": did,
                    }
                )
            distractors = sample(native_ids, args.n_distractors)
            apis_all = native_apis + distractors
            rng = random.Random(args.distractor_seed + i)
            rng.shuffle(apis_all)
            tools_json = build_tools_json(apis_all)
            valid_ids = {a["display_id"] for a in apis_all}

            user_query = q.get("query", "")
            prompt = build_chat_prompt(tokenizer, user_query, tools_json)
            gt_ids = gt_ids_from_query(q)

            all_tools_found: List[str] = []
            for pass_idx in range(max_passes):
                if pass_idx > 0 and all_tools_found:
                    extra = (
                        "\n\nYou have already listed: "
                        + ", ".join(all_tools_found)
                        + ". List any ADDITIONAL tools that are still needed, or reply 'none'."
                    )
                    pass_prompt = prompt + extra
                else:
                    pass_prompt = prompt
                ids = tokenizer(pass_prompt, return_tensors="pt", truncation=True, max_length=6000).to(args.device)
                out = model.generate(
                    **ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                new_tokens = out[0, ids["input_ids"].shape[1]:]
                generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
                pass_ids = extract_predicted_ids(generation, valid_ids)
                new_ones = [x for x in dict.fromkeys(pass_ids) if x not in all_tools_found]
                if not new_ones and pass_idx > 0:
                    break
                all_tools_found.extend(new_ones)
                if not multipass:
                    break

            metrics = compute_metrics(all_tools_found, gt_ids)
            all_metrics.append(metrics)
            per_sample.append(
                {
                    "query_id": q.get("query_id"),
                    "subset": q.get("_subset"),
                    "n_tools_shown": len(apis_all),
                    "gt_ids": gt_ids,
                    "pred_ids": all_tools_found,
                    "metrics": metrics,
                }
            )
            if args.verbose and i < 3:
                per_sample[-1]["generation"] = generation[:400]

            if (i + 1) % 10 == 0 or i == len(queries) - 1:
                avg_f1 = sum(m["F1"] for m in all_metrics) / len(all_metrics)
                print(f"  [{method}] {i+1}/{len(queries)}  running_F1={avg_f1:.4f}", flush=True)

    runtime = time.time() - t0
    n = len(all_metrics)
    agg = {
        k: (sum(m[k] for m in all_metrics) / n if n else 0.0)
        for k in ["F1", "Exact", "Recall", "Precision", "GT_subset", "nDCG"]
    }
    return {
        "method": method,
        "n_queries": n,
        "metrics": agg,
        "runtime_s": runtime,
        "per_sample": per_sample,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    subsets = ALL_SUBSETS if "all" in args.subsets else args.subsets
    print(f"[data] subsets: {subsets}", flush=True)
    queries_all = load_queries(subsets)
    start = max(0, args.start_idx)
    end = start + args.max_samples if args.max_samples > 0 else len(queries_all)
    queries = queries_all[start:end]
    pool = build_tool_pool(queries_all)
    print(f"[data] {len(queries_all)} queries total; using slice [{start}:{start+len(queries)}] = {len(queries)}")
    print(f"[pool] {len(pool)} unique (tool, api) pairs across loaded subsets")

    print(f"[load] {args.model} on {args.device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = resolve_dtype(args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    model.eval()
    cfg = model.config
    n_kv = cfg.num_key_value_heads
    n_q = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    L = cfg.num_hidden_layers
    print(f"[load] done in {time.time()-t0:.1f}s; L={L} n_kv={n_kv} n_q={n_q} head_dim={head_dim}", flush=True)

    # Load B_ont
    B_ont = None
    if args.b_ont:
        print(f"[ocq] loading B_ont from {args.b_ont}", flush=True)
        payload = torch.load(args.b_ont, map_location="cpu", weights_only=False)
        B_ont = payload["B_ont"] if isinstance(payload, dict) else payload
        print(f"[ocq] B_ont shape {tuple(B_ont.shape)}", flush=True)
    skip_heads = parse_skip_heads(args.skip_heads) if args.skip_heads else None

    # Evaluate each method
    results = []
    for method in args.methods:
        print(f"\n[eval] {method}", flush=True)
        res = run_method(
            model, tok, queries, pool, method, args,
            B_ont=B_ont, n_kv=n_kv, n_q=n_q, head_dim=head_dim, L=L,
            skip_heads=skip_heads,
        )
        m = res["metrics"]
        print(
            f"[eval] {method}: F1={m['F1']:.4f} Exact={m['Exact']:.4f} "
            f"Recall={m['Recall']:.4f} Precision={m['Precision']:.4f} "
            f"GT_sub={m['GT_subset']:.4f} nDCG={m['nDCG']:.4f} runtime={res['runtime_s']:.1f}s",
            flush=True,
        )
        results.append(res)

    payload = {
        "model": args.model,
        "subsets": subsets,
        "n_queries": len(queries),
        "n_distractors": args.n_distractors,
        "pool_size": len(pool),
        "max_new_tokens": args.max_new_tokens,
        "methods": args.methods,
        "skip_heads_spec": args.skip_heads,
        "skip_sink_tokens": args.skip_sink_tokens,
        "ocq": {"quant_bits": args.ocq_quant_bits, "ont_mode": args.ocq_ont_mode},
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
