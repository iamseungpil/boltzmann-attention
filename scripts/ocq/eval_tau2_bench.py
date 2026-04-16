#!/usr/bin/env python3
"""
τ²-bench evaluation: multi-turn tool-selection with Q+K attention steering.

Dataset: external/tau2-bench/data/tau2/domains/retail/tasks.json
  114 tasks × 0-13 ground-truth actions each (15 unique retail tools).
  Each task has user_scenario.instructions with task description and
  evaluation_criteria.actions with ground-truth tool calls.

Eval protocol
-------------
For each task:
  1. Build a chat prompt (system: available tools in function-calling JSON schema,
     user: composite task instruction from all instruction fields)
  2. Generate up to --max-new-tokens tokens (greedy)
  3. Extract predicted tool names from generation via regex
  4. Compare predicted tool SET vs ground truth tool SET (unique names only)
  5. Compute: F1, Exact Match, Recall, Precision, GT⊆Pred, nDCG@K

Methods supported
-----------------
  Reuses the hook infrastructure from eval_metatool_subtask1.py:
    no_steer           baseline: pure model.generate(), no hook
    ocq_qbias_b<β>     Q-only full B_ont subtraction
    ocq_ladapt_k<α>_q<β>  layer-adaptive K early + Q late
    ocq_bias_a<α>      K-only bias
    (and all other methods supported by parse_method)

CLI example
-----------
    python scripts/ocq/eval_tau2_bench.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --device cuda:0 \\
        --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-tau2-retail/B_ont.pt \\
        --methods no_steer ocq_qbias_b-0.03 ocq_ladapt_k0.05_q-0.03 \\
        --max-samples 50 \\
        --out reports/tau2_2026_04_17/retail_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_metatool_subtask1 import (
    install_kbias_hooks,
    install_q_bias_hooks,
    install_layer_adaptive_hooks,
    install_normalized_kbias_hooks,
    install_contrastive_kbias_hooks,
    install_vbias_hooks,
    install_kvbias_hooks,
    install_qkv_joint_hooks,
    install_caa_hooks,
    install_adaseka_proxy_hooks,
    install_quant_hooks,
    install_facet_gated_hooks,
    build_facet_masks,
    parse_method,
    parse_skip_heads,
    resolve_dtype,
)


# =====================================================================
# Retail domain tool definitions (from τ²-bench retail/tools.py)
# =====================================================================

RETAIL_TOOLS = [
    "calculate", "cancel_pending_order", "exchange_delivered_order_items",
    "find_user_id_by_email", "find_user_id_by_name_zip", "get_item_details",
    "get_order_details", "get_product_details", "get_user_details",
    "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items", "transfer_to_human_agents",
]

# Minimal descriptions extracted from tau2-bench retail/tools.py docstrings.
# These are sufficient for the model to understand tool purpose without
# needing the full parameter schemas for tool SELECTION evaluation.
TOOL_DESCRIPTIONS = {
    "calculate": "Calculate the result of a mathematical expression.",
    "cancel_pending_order": "Cancel a pending order. The order status will be changed to 'cancelled' and payment refunded.",
    "exchange_delivered_order_items": "Exchange items in a delivered order to new items of the same product type.",
    "find_user_id_by_email": "Find user id by email address.",
    "find_user_id_by_name_zip": "Find user id by first name, last name, and zip code.",
    "get_item_details": "Get the inventory details of an item by item id.",
    "get_order_details": "Get the status and details of an order by order id.",
    "get_product_details": "Get the inventory details of a product by product id.",
    "get_user_details": "Get the details of a user, including their orders.",
    "modify_pending_order_address": "Modify the shipping address of a pending order.",
    "modify_pending_order_items": "Modify items in a pending order to new items of the same product type.",
    "modify_pending_order_payment": "Modify the payment method of a pending order.",
    "modify_user_address": "Modify the default address of a user.",
    "return_delivered_order_items": "Return some items of a delivered order. Order status changed to 'return requested'.",
    "transfer_to_human_agents": "Transfer the user to a human agent with a summary of the issue.",
}

# Minimal parameter schemas for function-calling format (enough for
# the model to understand each tool's purpose and interface).
TOOL_PARAMS = {
    "calculate": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Mathematical expression"}},
        "required": ["expression"],
    },
    "cancel_pending_order": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "Order id like '#W0000000'"},
            "reason": {"type": "string", "description": "'no longer needed' or 'ordered by mistake'"},
        },
        "required": ["order_id", "reason"],
    },
    "exchange_delivered_order_items": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "new_item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    },
    "find_user_id_by_email": {
        "type": "object",
        "properties": {"email": {"type": "string"}},
        "required": ["email"],
    },
    "find_user_id_by_name_zip": {
        "type": "object",
        "properties": {
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "zip": {"type": "string"},
        },
        "required": ["first_name", "last_name", "zip"],
    },
    "get_item_details": {
        "type": "object",
        "properties": {"item_id": {"type": "string"}},
        "required": ["item_id"],
    },
    "get_order_details": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
    "get_product_details": {
        "type": "object",
        "properties": {"product_id": {"type": "string"}},
        "required": ["product_id"],
    },
    "get_user_details": {
        "type": "object",
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    },
    "modify_pending_order_address": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "address1": {"type": "string"}, "address2": {"type": "string"},
            "city": {"type": "string"}, "state": {"type": "string"},
            "country": {"type": "string"}, "zip": {"type": "string"},
        },
        "required": ["order_id", "address1", "address2", "city", "state", "country", "zip"],
    },
    "modify_pending_order_items": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "new_item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    },
    "modify_pending_order_payment": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "payment_method_id"],
    },
    "modify_user_address": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "address1": {"type": "string"}, "address2": {"type": "string"},
            "city": {"type": "string"}, "state": {"type": "string"},
            "country": {"type": "string"}, "zip": {"type": "string"},
        },
        "required": ["user_id", "address1", "address2", "city", "state", "country", "zip"],
    },
    "return_delivered_order_items": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"},
        },
        "required": ["order_id", "item_ids", "payment_method_id"],
    },
    "transfer_to_human_agents": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Q+K attention steering on tau2-bench (retail domain)"
    )
    p.add_argument("--model", required=True, help="HF model id (e.g. Qwen/Qwen2.5-7B-Instruct)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--domain", default="retail",
                   choices=["retail", "airline", "telecom", "banking_knowledge"],
                   help="tau2-bench domain")
    p.add_argument("--dataset", default="",
                   help="Override path to tasks.json (default: auto from --domain)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap on number of tasks (0 = all)")
    p.add_argument("--start-idx", type=int, default=0,
                   help="Start index for slicing the dataset")
    p.add_argument("--methods", nargs="+",
                   default=["no_steer", "ocq_qbias_b-0.03", "ocq_ladapt_k0.05_q-0.03"],
                   help="Methods to evaluate")
    p.add_argument("--b-ont", type=str, default="",
                   help="Path to B_ont .pt file")
    p.add_argument("--ocq-quant-bits", type=int, default=4)
    p.add_argument("--ocq-ont-mode", default="1b", choices=["1a", "1b", "1c"])
    p.add_argument("--max-new-tokens", type=int, default=300,
                   help="Max generation tokens (multi-tool tasks need longer output)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-heads", type=str, default="")
    p.add_argument("--skip-sink-tokens", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# =====================================================================
# Prompt construction
# =====================================================================

def build_tools_json() -> List[dict]:
    """Build the tools list in OpenAI-compatible function calling format."""
    tools = []
    for name in RETAIL_TOOLS:
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": TOOL_PARAMS[name],
            },
        })
    return tools


def build_user_message(task: dict) -> str:
    """Compose the user message from all instruction fields in the task."""
    instr = task["user_scenario"]["instructions"]
    parts = []
    if instr.get("reason_for_call"):
        parts.append(instr["reason_for_call"])
    if instr.get("known_info"):
        parts.append(instr["known_info"])
    if instr.get("unknown_info"):
        parts.append(instr["unknown_info"])
    if instr.get("task_instructions"):
        parts.append(instr["task_instructions"])
    return " ".join(parts)


SYSTEM_PROMPT = (
    "You are a retail customer service agent. You have access to the following tools "
    "to help resolve customer issues. For each customer request, determine which tools "
    "you need to call and output them as function calls.\n\n"
    "IMPORTANT: Output ALL tool calls you would need to make to fully resolve the "
    "customer's request. Output each tool call in this format:\n"
    '{"name": "<tool_name>", "arguments": {...}}\n\n'
    "List every tool call needed, one per line."
)


def build_chat_prompt(
    tokenizer,
    task: dict,
    tools: List[dict],
) -> str:
    """Build a chat prompt using the tokenizer's chat template with tools."""
    user_msg = build_user_message(task)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    # Try to use the tokenizer's native tool support if available
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
    except (TypeError, ValueError):
        # Fallback: embed tools in system message manually
        tools_text = json.dumps(tools, indent=2)
        messages[0]["content"] = SYSTEM_PROMPT + "\nAvailable tools:\n" + tools_text
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


# =====================================================================
# Tool extraction from generation
# =====================================================================

# Matches "name": "tool_name" patterns in JSON-like output
TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def extract_tool_names(generation: str) -> List[str]:
    """Extract all tool names from generation text.

    Handles both JSON tool_call format and plain text mentions.
    Returns list of tool names in order of appearance (may have duplicates).
    """
    # Primary: JSON "name": "tool_name" pattern
    found = TOOL_NAME_RE.findall(generation)
    # Filter to only known retail tools
    valid = [t for t in found if t in RETAIL_TOOLS]
    if valid:
        return valid

    # Fallback: scan for tool names as substrings (longest first to avoid
    # prefix collisions like "get_order" matching "get_order_details")
    sorted_tools = sorted(RETAIL_TOOLS, key=len, reverse=True)
    gen_lower = generation.lower()
    found_fallback = []
    for tool in sorted_tools:
        if tool.lower() in gen_lower:
            found_fallback.append(tool)
    return found_fallback


def extract_gt_tools(task: dict) -> List[str]:
    """Extract unique ground-truth tool names from evaluation_criteria.actions."""
    actions = task.get("evaluation_criteria", {}).get("actions", [])
    seen = set()
    unique = []
    for a in actions:
        name = a["name"]
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


# =====================================================================
# Metrics
# =====================================================================

def compute_tau2_metrics(
    pred_tools: List[str],
    gt_tools: List[str],
) -> Dict[str, float]:
    """Compute per-task metrics for tool selection.

    Args:
        pred_tools: unique predicted tool names
        gt_tools: unique ground-truth tool names

    Returns:
        Dict with F1, Exact, Recall, Precision, GT_subset, nDCG
    """
    pred_set = set(pred_tools)
    gt_set = set(gt_tools)

    # Handle empty GT (tasks with 0 actions)
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
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact = 1 if pred_set == gt_set else 0
    gt_subset = 1 if gt_set <= pred_set else 0

    # nDCG: treats pred list order as ranking, GT membership as relevance
    dcg = 0.0
    for i, tool in enumerate(pred_tools):
        if tool in gt_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1
    # Ideal DCG: all GT tools ranked first
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(gt_set)))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "F1": f1,
        "Exact": exact,
        "Recall": recall,
        "Precision": precision,
        "GT_subset": gt_subset,
        "nDCG": ndcg,
    }


# =====================================================================
# Method dispatch (reuses hooks from eval_metatool_subtask1)
# =====================================================================

@contextmanager
def _nullcontext():
    yield


def get_steering_context(
    model, method: str, args,
    B_ont, n_kv, n_q, head_dim, L,
    facet_mask=None, skip_heads=None,
):
    """Return the appropriate context manager for the given method."""
    kind, params = parse_method(method)
    effective_skip = skip_heads if params.get("use_skip") else None
    effective_sink = args.skip_sink_tokens if params.get("use_sinkskip") else 0

    if kind == "no_steer":
        return _nullcontext()
    elif kind == "bias":
        return install_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip, skip_sink=effective_sink,
        )
    elif kind == "qbias":
        return install_q_bias_hooks(
            model, B_ont, beta=params["beta"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            skip_heads=effective_skip, skip_sink=effective_sink,
        )
    elif kind == "layer_adaptive":
        return install_layer_adaptive_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"], beta_q=params["beta_q"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            k_boundary_frac=params.get("k_frac", 0.25),
            skip_heads=effective_skip, skip_sink=effective_sink,
        )
    elif kind == "normbias":
        return install_normalized_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip, skip_sink=effective_sink,
        )
    elif kind == "cbias":
        return install_contrastive_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            dominant_rank=params["dominant_rank"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip, skip_sink=effective_sink,
        )
    elif kind == "vbias":
        return install_vbias_hooks(
            model, B_ont, alpha=params["alpha_v"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip, skip_sink=effective_sink,
        )
    elif kind == "kvbias":
        return install_kvbias_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"], alpha_v=params["alpha_v"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip, skip_sink=effective_sink,
        )
    elif kind == "qkv_joint":
        return install_qkv_joint_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"],
            gamma_v=params["gamma_v"],
            beta_q=params["beta_q"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            skip_heads=effective_skip, skip_sink=effective_sink,
        )
    elif kind == "caa":
        return install_caa_hooks(
            model, B_ont, alpha=params["alpha"],
            skip_sink=effective_sink,
        )
    elif kind == "adaseka":
        return install_adaseka_proxy_hooks(
            model, B_ont, alpha=params["alpha"], M=params["M"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            temperature=params["T"], skip_sink=effective_sink,
        )
    elif kind == "quant":
        return install_quant_hooks(
            model, B_ont,
            bits_residual=args.ocq_quant_bits,
            ont_mode=args.ocq_ont_mode,
            n_kv=n_kv, head_dim=head_dim,
            plus_bias_alpha=params["alpha"],
        )
    elif kind == "facet_gated":
        if facet_mask is None:
            raise ValueError("facet_gated requires facet_mask")
        return install_facet_gated_hooks(
            model, B_ont, facet_mask,
            alpha_base=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip, skip_sink=effective_sink,
        )
    else:
        raise ValueError(f"Unsupported method kind: {kind}")


# =====================================================================
# Eval loop
# =====================================================================

@torch.no_grad()
def run_method(
    model,
    tokenizer,
    tasks: List[dict],
    tools_json: List[dict],
    method: str,
    args,
    B_ont: Optional[torch.Tensor],
    n_kv: int,
    n_q: int,
    head_dim: int,
    L: int,
    facet_mask=None,
    skip_heads=None,
) -> Dict:
    """Evaluate a single method across all tasks."""
    ctx = get_steering_context(
        model, method, args,
        B_ont=B_ont, n_kv=n_kv, n_q=n_q, head_dim=head_dim, L=L,
        facet_mask=facet_mask, skip_heads=skip_heads,
    )

    all_metrics = []
    per_sample = []
    t0 = time.time()

    with ctx:
        for i, task in enumerate(tasks):
            prompt = build_chat_prompt(tokenizer, task, tools_json)
            ids = tokenizer(prompt, return_tensors="pt").to(args.device)
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = out[0, ids["input_ids"].shape[1]:]
            generation = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Extract predictions and ground truth
            pred_raw = extract_tool_names(generation)
            pred_unique = list(dict.fromkeys(pred_raw))  # deduplicate preserving order
            gt_unique = extract_gt_tools(task)

            metrics = compute_tau2_metrics(pred_unique, gt_unique)
            all_metrics.append(metrics)

            per_sample.append({
                "task_id": task["id"],
                "gt_tools": gt_unique,
                "pred_tools": pred_unique,
                "metrics": metrics,
            })

            if args.verbose and i < 3:
                per_sample[-1]["generation"] = generation[:500]

            if (i + 1) % 10 == 0 or i == len(tasks) - 1:
                avg_f1 = sum(m["F1"] for m in all_metrics) / len(all_metrics)
                print(f"  [{method}] {i+1}/{len(tasks)}  running_F1={avg_f1:.4f}", flush=True)

    runtime = time.time() - t0
    n = len(all_metrics)

    # Aggregate metrics
    agg = {}
    for key in ["F1", "Exact", "Recall", "Precision", "GT_subset", "nDCG"]:
        vals = [m[key] for m in all_metrics]
        agg[key] = sum(vals) / n if n > 0 else 0.0

    return {
        "method": method,
        "n_tasks": n,
        "metrics": agg,
        "runtime_s": runtime,
        "per_sample": per_sample,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Resolve dataset path
    if not args.dataset:
        args.dataset = str(
            REPO / "external" / "tau2-bench" / "data" / "tau2"
            / "domains" / args.domain / "tasks.json"
        )

    print(f"[load] {args.model} on {args.device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = resolve_dtype(args.model, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
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

    # Load tasks
    print(f"[data] {args.dataset}", flush=True)
    with open(args.dataset) as f:
        tasks = json.load(f)
    start = max(0, args.start_idx)
    end = start + args.max_samples if args.max_samples > 0 else len(tasks)
    tasks = tasks[start:end]
    print(f"[data] {len(tasks)} tasks (slice [{start}:{start+len(tasks)}])", flush=True)

    # Build tools JSON once
    tools_json = build_tools_json()

    # Load B_ont if needed
    needs_b_ont = any(m != "no_steer" for m in args.methods)
    needs_facet_mask = any("facet_gated" in m for m in args.methods)
    B_ont = None
    facet_mask = None
    if needs_b_ont:
        if not args.b_ont:
            raise ValueError("--b-ont is required for steering methods")
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

        if needs_facet_mask:
            if not (isinstance(payload, dict) and "r_per_pair" in payload):
                raise ValueError("facet_gated requires B_ont payload with 'r_per_pair'")
            r_per_pair = payload["r_per_pair"]
            n_facets = len(payload.get("facet_order", ["f0", "f1", "f2", "f3"]))
            facet_mask = build_facet_masks(
                r_per_pair, L=L, H=n_kv, r_ont=B_ont.shape[-1], n_facets=n_facets
            )
            print(f"[ocq] facet_mask shape {tuple(facet_mask.shape)}", flush=True)

    # Parse skip-heads
    skip_heads = None
    if args.skip_heads:
        skip_heads = parse_skip_heads(args.skip_heads, n_kv)
        print(f"[skip] {len(skip_heads)} (layer,head) pairs skipped", flush=True)

    # Run all methods
    results = []
    for method in args.methods:
        print(f"\n[eval] {method}", flush=True)
        res = run_method(
            model, tok, tasks, tools_json,
            method=method, args=args,
            B_ont=B_ont, n_kv=n_kv, n_q=n_q,
            head_dim=head_dim, L=L,
            facet_mask=facet_mask, skip_heads=skip_heads,
        )
        m = res["metrics"]
        print(
            f"[eval] {method}: F1={m['F1']:.4f} Exact={m['Exact']:.4f} "
            f"Recall={m['Recall']:.4f} Precision={m['Precision']:.4f} "
            f"GT_sub={m['GT_subset']:.4f} nDCG={m['nDCG']:.4f} "
            f"runtime={res['runtime_s']:.1f}s",
            flush=True,
        )
        results.append(res)

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Method':35s} {'F1':>7s} {'Exact':>7s} {'Recall':>7s} {'Prec':>7s} {'GT_sub':>7s} {'nDCG':>7s}")
    print("-" * 90)
    no_steer = next((r for r in results if r["method"] == "no_steer"), None)
    base_f1 = no_steer["metrics"]["F1"] if no_steer else None
    for r in results:
        m = r["metrics"]
        delta = ""
        if base_f1 is not None and r["method"] != "no_steer":
            d = (m["F1"] - base_f1) * 100
            delta = f"  dF1={d:+.2f}pp"
        print(
            f"  {r['method']:33s} {m['F1']:7.4f} {m['Exact']:7.4f} "
            f"{m['Recall']:7.4f} {m['Precision']:7.4f} "
            f"{m['GT_subset']:7.4f} {m['nDCG']:7.4f}{delta}"
        )
    print("=" * 90)

    # Save results
    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "domain": args.domain,
        "n_tasks": len(tasks),
        "max_new_tokens": args.max_new_tokens,
        "methods": args.methods,
        "skip_heads_spec": args.skip_heads or None,
        "skip_sink_tokens": args.skip_sink_tokens,
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
