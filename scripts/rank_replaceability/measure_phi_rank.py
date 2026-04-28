#!/usr/bin/env python3
"""measure_phi_rank.py — E1 of EXPERIMENT_PLAN_v27 (rank-bounded prompt replaceability).

Measures the effective rank r*(τ) of the prefix-attention output function
    Φ_P^{(ℓ,h)}(q) := λ_P^{(ℓ,h)}(q) · attn(q; K_P^{(ℓ,h)}, V_P^{(ℓ,h)})
over a task query distribution Q, per (layer, head).

Outputs JSON with:
  - per (layer, head) singular spectrum of stacked Φ_P samples
  - r*(τ) at τ ∈ {0.90, 0.95, 0.99}
  - heatmap-friendly arrays

Usage (smoke):
  python measure_phi_rank.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --task metatool_st4 --max-samples 4 \
      --device cuda:0 \
      --out reports/rank_replaceability_2026_04/qwen_metatool_smoke.json

Usage (full run):
  python measure_phi_rank.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --task metatool_st4 --max-samples 256 \
      --device cuda:0 \
      --out reports/rank_replaceability_2026_04/qwen_metatool_n256.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Data loading
# =============================================================================

# MetaTool ST4: 497 records, each with (query, ground-truth tool list, candidates)
DEFAULT_METATOOL_PATH = "/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json"

# tau2-bench: per-domain tasks.json
DEFAULT_TAU2_BASE = (
    "/home/woori/workspace_common/boltzmann-attention/external/tau2-bench/data/tau2"
)


_METATOOL_TOOL_RE = __import__("re").compile(
    r"\d+\.\s*tool name:\s*([^,\n]+?),\s*tool description"
)


def load_metatool_st4(dataset_path: str, n: int, full_schema: bool = False) -> List[dict]:
    """Returns list of {query, candidates, gt, system_full?} dicts.

    MetaTool ST4 schema: {action_prompt, thought_prompt, tool, query}.
    Candidates parsed via regex from action_prompt. If full_schema=True,
    we attach the action_prompt itself (rich tool descriptions) as the
    system content, producing a substantially longer prefix.
    """
    with open(dataset_path) as f:
        raw = json.load(f)
    out = []
    for entry in raw[:n]:
        query = entry.get("query")
        action = entry.get("action_prompt") or ""
        gt = entry.get("tool") or []
        cands = [m.strip().strip('"').strip("'") for m in _METATOOL_TOOL_RE.findall(action)]
        if not query or not cands:
            continue
        item = {"query": str(query), "candidates": cands, "gt": gt}
        if full_schema:
            # action_prompt is a full prompt with tool descriptions embedded.
            # Strip surrounding quotes (the dataset wraps it in ").
            sys_full = action.strip().strip('"').strip()
            item["system_full"] = sys_full
        out.append(item)
    return out


def _extract_domain_tools(tasks: List[dict]) -> List[str]:
    """Replicates eval_tau2_bench.extract_domain_tools: collect unique tool names
    from each task's evaluation_criteria.actions.
    """
    names = set()
    for t in tasks:
        if not isinstance(t, dict):
            continue
        crit = t.get("evaluation_criteria") or {}
        for a in (crit.get("actions") or []):
            if isinstance(a, dict) and a.get("name"):
                names.add(a["name"])
    return sorted(names)


# Minimal tool descriptions for tau2 (retail set imported from eval_tau2_bench.py
# convention; non-retail domains fall back to generic descriptions). Used when
# full_schema=True to expand the prefix to a more production-realistic length.
_TAU2_TOOL_DESCRIPTIONS = {
    # Retail
    "calculate": "Calculate the result of a mathematical expression.",
    "cancel_pending_order": "Cancel a pending order. Status changes to 'cancelled' and payment refunded.",
    "exchange_delivered_order_items": "Exchange items in a delivered order to new items of same product type.",
    "find_user_id_by_email": "Find user id by email address.",
    "find_user_id_by_name_zip": "Find user id by first name, last name, and zip code.",
    "get_item_details": "Get inventory details of an item by item id.",
    "get_order_details": "Get status and details of an order by order id.",
    "get_product_details": "Get inventory details of a product by product id.",
    "get_user_details": "Get details of a user, including their orders.",
    "modify_pending_order_address": "Modify shipping address of a pending order.",
    "modify_pending_order_items": "Modify items in a pending order to new items of same product type.",
    "modify_pending_order_payment": "Modify payment method of a pending order.",
    "modify_user_address": "Modify default address of a user.",
    "return_delivered_order_items": "Return items of a delivered order. Status changes to 'return requested'.",
    "transfer_to_human_agents": "Transfer the user to a human agent with summary of the issue.",
    # Telecom (descriptive fallbacks)
    "toggle_roaming": "Enable or disable international roaming on a customer's line.",
    "check_data_balance": "Check the remaining data balance on a customer's plan.",
    "reset_pin": "Reset the customer's account PIN to a new value.",
    "verify_otp": "Verify a one-time password sent to the customer's device.",
    "change_plan": "Switch the customer to a new mobile plan.",
    "suspend_line": "Temporarily suspend service on a customer's line.",
    "activate_line": "Activate or reactivate a customer's mobile line.",
    "report_issue": "Open a service issue ticket for the customer.",
    "check_coverage": "Look up cellular coverage at a customer's address.",
    # Airline (descriptive fallbacks)
    "search_flights": "Search for available flights given origin, destination, and date.",
    "book_flight": "Book a specific flight for the customer.",
    "cancel_reservation": "Cancel an existing flight reservation.",
    "modify_reservation": "Change the flight or date on an existing reservation.",
    "get_baggage_policy": "Retrieve the baggage allowance policy for a fare class.",
    "check_in_passenger": "Perform online check-in for a passenger on a booked flight.",
    "select_seat": "Reserve a specific seat on a booked flight.",
    "request_refund": "Initiate a refund request for an eligible reservation.",
}


def _build_full_tau2_schema(tools: List[str]) -> str:
    """Build a JSON-array string with name + description per tool. Generic
    fallback for tools not in _TAU2_TOOL_DESCRIPTIONS."""
    schemas = []
    for name in tools:
        desc = _TAU2_TOOL_DESCRIPTIONS.get(
            name, f"Tool '{name}'. Operates on the customer service domain."
        )
        # Mimic OpenAI function-calling JSON shape (compact)
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "object",
                            "description": "Tool-specific arguments.",
                        }
                    },
                    "required": [],
                },
            },
        })
    return json.dumps(schemas, indent=2)


def load_tau2(domain: str, n: int, full_schema: bool = False) -> List[dict]:
    """Loads tau2 retail/telecom/airline tasks. Returns list of {query, candidates, gt, system_full?}."""
    path = f"{DEFAULT_TAU2_BASE}/domains/{domain}/tasks.json"
    with open(path) as f:
        tasks = json.load(f)
    domain_tools = _extract_domain_tools(tasks)
    schema_str = _build_full_tau2_schema(domain_tools) if full_schema else None
    out = []
    for t in tasks[:n]:
        if isinstance(t, dict):
            user_scn = t.get("user_scenario") or {}
            instr = user_scn.get("instructions") if isinstance(user_scn, dict) else None
            instr = (
                instr
                or t.get("instructions")
                or t.get("instruction")
                or t.get("query")
                or json.dumps(t)[:512]
            )
            gt_actions = (t.get("evaluation_criteria") or {}).get("actions") or []
            gt = [a.get("name") for a in gt_actions if isinstance(a, dict) and a.get("name")]
        else:
            instr = str(t)[:512]
            gt = []
        item = {
            "query": str(instr)[:1024],
            "candidates": domain_tools,
            "gt": gt,
        }
        if schema_str is not None:
            item["system_full"] = (
                "You are a customer service agent. Use the tools listed below "
                "to resolve the customer's issue. Output the tool calls needed "
                "in JSON format.\n\n"
                "Available tools (full schema):\n"
                + schema_str
            )
        out.append(item)
    return out


# =============================================================================
# Prompt construction
# =============================================================================

DEFAULT_TOOL_SYSTEM_PROMPT = (
    "You are a tool-selection agent. You will be given a user query, and a list "
    "of available tools. Your job is to identify ALL tools required to fulfil "
    "the user's request, and emit them as JSON-formatted tool calls. Only emit "
    "tools from the candidate list. Emit them in the order they should be "
    "executed. Output format per tool call: "
    '<tool_call>{{"name": "ToolName", "arguments": {{}}}}</tool_call>'
)


_RNG = __import__("random").Random(0)
_LOREM_BASE = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat duis aute irure reprehenderit voluptate velit esse cillum "
    "fugiat nulla pariatur excepteur sint occaecat cupidatat non proident "
    "sunt in culpa qui officia deserunt mollit anim id est laborum"
).split()


def _random_words(n_words: int, rng=None) -> str:
    rng = rng or _RNG
    return " ".join(rng.choice(_LOREM_BASE) for _ in range(n_words))


def build_messages(
    item: dict,
    tool_system_prompt: str,
    prefix_mode: str = "real",
    rng=None,
) -> List[dict]:
    """Builds [system, user] messages.

    prefix_mode:
      - real: real system prompt + real candidates + real user query
      - random_prefix: replace system content with random words (≈ same length)
      - random_query: real prefix, but user message replaced with random words
      - shuffled_prefix: real candidates list shuffled (control for ordering)

    If item has 'system_full' field (set when load_*(full_schema=True)),
    the real system uses that instead of the names-only catalog. This drives
    the production-scale prefix experiment (E9).
    """
    cands = item.get("candidates") or []
    rng = rng or _RNG

    if item.get("system_full"):
        real_system = item["system_full"]
    elif cands:
        cand_str = "\n".join(f"- {c}" for c in cands)
        real_system = f"{tool_system_prompt}\n\nAvailable tools:\n{cand_str}"
    else:
        real_system = tool_system_prompt

    if prefix_mode == "real":
        sys_content = real_system
        user_content = item["query"]
    elif prefix_mode == "random_prefix":
        # Token-count proxy: real_system word count
        n_words = max(20, len(real_system.split()))
        sys_content = _random_words(n_words, rng)
        user_content = item["query"]
    elif prefix_mode == "random_query":
        sys_content = real_system
        n_words = max(8, len(item["query"].split()))
        user_content = _random_words(n_words, rng)
    elif prefix_mode == "shuffled_prefix":
        # Shuffle candidate order only (keep system text + user query intact)
        if cands:
            shuffled = list(cands)
            rng.shuffle(shuffled)
            cand_str = "\n".join(f"- {c}" for c in shuffled)
            sys_content = f"{tool_system_prompt}\n\nAvailable tools:\n{cand_str}"
        else:
            sys_content = real_system
        user_content = item["query"]
    else:
        raise ValueError(f"unknown prefix_mode={prefix_mode}")

    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]


def find_user_block_start(
    tokenizer, prompt_ids: torch.Tensor, model_family: str
) -> int:
    """Returns the position index where the user message block begins.

    Tokens before this index are treated as 'prefix' (system + tools).
    """
    text = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
    if model_family == "qwen":
        marker = "<|im_start|>user"
    elif model_family == "llama":
        marker = "<|start_header_id|>user<|end_header_id|>"
    else:
        # Fallback: assume <|im_start|>user works (Qwen-style chatml)
        marker = "<|im_start|>user"
    idx_char = text.find(marker)
    if idx_char < 0:
        raise RuntimeError(
            f"Could not find user marker {marker!r} in prompt (model_family={model_family})"
        )
    # Tokenize the prefix (text before the marker) and use its length as boundary
    prefix_text = text[:idx_char]
    # Encode without adding special tokens — we want raw token count of the prefix span
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    return len(prefix_ids)


def detect_model_family(model_name: str) -> str:
    n = model_name.lower()
    if "qwen" in n:
        return "qwen"
    if "llama" in n or "meta-llama" in n:
        return "llama"
    return "qwen"  # default chatml


# =============================================================================
# Hook to capture attention weights and V at each layer
# =============================================================================

class AttnCapture:
    """Captures (attn_weights, V) per layer by forward-hooking attention modules.

    Stored per-layer: attn_weights of shape (B, H_q, T, T_k) and V_kv of shape
    (B, H_kv, T, d_h). For GQA we expand V to H_q on the fly when computing
    per-head Φ_P.
    """

    def __init__(self, model, model_family: str):
        self.model = model
        self.family = model_family
        self.handles = []
        self.attn_weights: List[torch.Tensor] = []  # one per layer
        self.values: List[torch.Tensor] = []  # one per layer

    def _hook(self, layer_idx: int):
        def fn(module, args, kwargs, output):
            # transformers >=4.40 returns (output, attn_weights, past_key_value)
            # We require output_attentions=True at forward call site.
            attn = None
            if isinstance(output, tuple) and len(output) >= 2:
                attn = output[1]
            if attn is None:
                return
            # Capture V from the kwargs via stored intermediate. For HF Qwen/Llama
            # attention, the V is computed inside; safest is to recompute from K_proj/V_proj
            # in a separate pass. As a pragmatic alternative we capture via a second hook
            # on the v_proj output (set up below).
            self.attn_weights[layer_idx] = attn.detach()
        return fn

    def _v_hook(self, layer_idx: int):
        def fn(module, args, kwargs, output):
            # output: (B, T, H_kv * d_h) — reshape later
            self.values[layer_idx] = output.detach()
        return fn

    def install(self):
        self.attn_weights = [None] * self.model.config.num_hidden_layers
        self.values = [None] * self.model.config.num_hidden_layers
        layers = _get_layers(self.model, self.family)
        for i, layer in enumerate(layers):
            attn_mod = _get_attn_module(layer, self.family)
            v_proj = _get_v_proj(layer, self.family)
            self.handles.append(
                attn_mod.register_forward_hook(self._hook(i), with_kwargs=True)
            )
            self.handles.append(
                v_proj.register_forward_hook(self._v_hook(i), with_kwargs=True)
            )

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _get_layers(model, family: str):
    if family == "qwen" or family == "llama":
        return model.model.layers
    raise ValueError(family)


def _get_attn_module(layer, family: str):
    if family == "qwen" or family == "llama":
        return layer.self_attn
    raise ValueError(family)


def _get_v_proj(layer, family: str):
    if family == "qwen" or family == "llama":
        return layer.self_attn.v_proj
    raise ValueError(family)


# =============================================================================
# Φ_P extraction
# =============================================================================

def compute_phi_p_per_layer_head(
    attn_weights_layer: torch.Tensor,  # (1, H_q, T, T_k)
    v_layer: torch.Tensor,             # (1, T, H_kv * d_h)
    prefix_end: int,
    last_query_pos: int,
    num_kv_heads: int,
    head_dim: int,
    num_q_heads: int,
) -> torch.Tensor:
    """Returns Φ_P per head for the given query position. Shape (H_q, d_h).

    Φ_P[h] = sum_{p in prefix} attn[h, last_q_pos, p] * V_kv[h//group, p]
    """
    # Reshape V: (1, T, H_kv * d_h) -> (1, H_kv, T, d_h)
    V = v_layer.view(1, v_layer.shape[1], num_kv_heads, head_dim).permute(0, 2, 1, 3)
    # GQA group size
    group = num_q_heads // num_kv_heads

    # Slice attention to last query position vs prefix positions
    # attn_weights_layer: (1, H_q, T, T_k); take row=last_query_pos, cols 0:prefix_end
    a = attn_weights_layer[0, :, last_query_pos, :prefix_end]  # (H_q, prefix_end)
    # V at prefix positions per kv-head: (H_kv, prefix_end, d_h)
    V_pref = V[0, :, :prefix_end, :]
    # Expand V to H_q via group repeat
    V_pref_expanded = V_pref.repeat_interleave(group, dim=0)  # (H_q, prefix_end, d_h)
    # Φ_P[h] = a[h] @ V_pref_expanded[h] -> (d_h,)
    phi = torch.einsum("hp,hpd->hd", a, V_pref_expanded)  # (H_q, d_h)
    return phi.float().cpu()


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--task",
        choices=["metatool_st4", "tau2_retail", "tau2_telecom", "tau2_airline"],
        default="metatool_st4",
    )
    p.add_argument("--metatool-path", default=DEFAULT_METATOOL_PATH)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--tau-list", default="0.90,0.95,0.99",
        help="comma-separated τ values for r*",
    )
    p.add_argument(
        "--prefix-mode",
        choices=["real", "random_prefix", "random_query", "shuffled_prefix"],
        default="real",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tool-schema-mode",
        choices=["names", "full"],
        default="names",
        help="'names' = current behavior (tool list as bullet names). "
             "'full' = production-scale schema with descriptions+params (E9).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    taus = [float(t) for t in args.tau_list.split(",")]

    # Load data
    full_schema = args.tool_schema_mode == "full"
    if args.task == "metatool_st4":
        items = load_metatool_st4(args.metatool_path, args.max_samples, full_schema=full_schema)
    elif args.task.startswith("tau2_"):
        domain = args.task.split("_", 1)[1]
        items = load_tau2(domain, args.max_samples, full_schema=full_schema)
    else:
        raise ValueError(args.task)
    print(f"[data] task={args.task} N={len(items)}", flush=True)
    if len(items) == 0:
        print("ERROR: zero items loaded", file=sys.stderr)
        return 2

    # Load model
    print(f"[model] loading {args.model} (dtype={args.dtype}, device={args.device})", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",  # required for output_attentions
    )
    model.eval()

    family = detect_model_family(args.model)
    cfg = model.config
    H_q = cfg.num_attention_heads
    H_kv = getattr(cfg, "num_key_value_heads", H_q)
    d_h = cfg.hidden_size // H_q
    L = cfg.num_hidden_layers
    print(f"[model] L={L} H_q={H_q} H_kv={H_kv} d_h={d_h} family={family}", flush=True)

    # Pre-allocate Φ_P buffers per (layer, head): list[layer] of np.array(N, H_q, d_h)
    N = len(items)
    phi_buffer = np.zeros((L, N, H_q, d_h), dtype=np.float32)
    prefix_lens = np.zeros(N, dtype=np.int32)
    seq_lens = np.zeros(N, dtype=np.int32)

    import random as _random_mod
    rng = _random_mod.Random(args.seed)

    cap = AttnCapture(model, family)
    cap.install()

    t0 = time.time()
    try:
        for i, item in enumerate(items):
            messages = build_messages(
                item, DEFAULT_TOOL_SYSTEM_PROMPT,
                prefix_mode=args.prefix_mode, rng=rng,
            )
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except Exception as e:
                print(f"[skip {i}] chat_template failed: {e}", flush=True)
                continue
            enc = tokenizer(prompt_text, return_tensors="pt").to(args.device)
            input_ids = enc.input_ids
            T = input_ids.shape[1]
            try:
                prefix_end = find_user_block_start(tokenizer, input_ids, family)
            except RuntimeError as e:
                print(f"[skip {i}] {e}", flush=True)
                continue
            prefix_lens[i] = prefix_end
            seq_lens[i] = T

            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=enc.attention_mask,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )

            last_q_pos = T - 1  # last position before next-token prediction
            # Extract Φ_P per layer
            for ell in range(L):
                aw = cap.attn_weights[ell]  # (1, H_q, T, T_k)
                vv = cap.values[ell]        # (1, T, H_kv*d_h)
                if aw is None or vv is None:
                    continue
                phi = compute_phi_p_per_layer_head(
                    aw, vv, prefix_end, last_q_pos,
                    num_kv_heads=H_kv, head_dim=d_h, num_q_heads=H_q,
                )
                phi_buffer[ell, i] = phi.numpy()

            if (i + 1) % 8 == 0 or i == N - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                remain = (N - i - 1) / max(rate, 1e-3)
                print(
                    f"[{i+1}/{N}] elapsed={elapsed:.1f}s rate={rate:.2f}/s "
                    f"eta={remain:.1f}s prefix_len_mean={prefix_lens[:i+1].mean():.0f}",
                    flush=True,
                )
    finally:
        cap.remove()

    # SVD per (layer, head)
    print("[svd] computing per-(layer, head) spectrum...", flush=True)
    K_save = 32  # top-K right singular vectors saved for E3 reuse
    K_save = min(K_save, d_h, N)
    spectra = np.zeros((L, H_q, min(N, d_h)), dtype=np.float32)  # singular values
    eigvecs = np.zeros((L, H_q, K_save, d_h), dtype=np.float32)
    phi_mean = np.zeros((L, H_q, d_h), dtype=np.float32)
    r_star = {f"{tau:.2f}": np.zeros((L, H_q), dtype=np.int32) for tau in taus}

    for ell in range(L):
        for h in range(H_q):
            M = phi_buffer[ell, :, h, :]  # (N, d_h)
            phi_mean[ell, h] = M.mean(axis=0)
            # Eckart-Young is on un-centered SVD. We use raw Φ_P samples.
            try:
                # full_matrices=False → Vh shape (min(N,d_h), d_h)
                _, s, Vh = np.linalg.svd(M, full_matrices=False, compute_uv=True)
            except np.linalg.LinAlgError:
                s = np.zeros(min(N, d_h))
                Vh = np.zeros((min(N, d_h), d_h))
            k_full = min(len(s), spectra.shape[2])
            spectra[ell, h, :k_full] = s[:k_full]
            k_save = min(K_save, Vh.shape[0])
            eigvecs[ell, h, :k_save] = Vh[:k_save]
            energy = np.cumsum(s ** 2)
            total = energy[-1] if len(energy) > 0 and energy[-1] > 0 else 1.0
            for tau in taus:
                idx = int(np.searchsorted(energy / total, tau)) + 1
                r_star[f"{tau:.2f}"][ell, h] = min(idx, len(s))

    # Save bulky arrays (eigvecs, phi_mean) as .npz alongside JSON
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        eigvecs=eigvecs,
        phi_mean=phi_mean,
        spectra_full=spectra,
    )

    out = {
        "model": args.model,
        "task": args.task,
        "prefix_mode": args.prefix_mode,
        "seed": int(args.seed),
        "n_samples": int(N),
        "n_layers": int(L),
        "n_heads_q": int(H_q),
        "n_heads_kv": int(H_kv),
        "head_dim": int(d_h),
        "k_save": int(K_save),
        "taus": taus,
        "r_star": {k: v.tolist() for k, v in r_star.items()},
        "spectra_top16": spectra[:, :, :16].tolist(),
        "prefix_lens": prefix_lens.tolist(),
        "seq_lens": seq_lens.tolist(),
        "wall_seconds": time.time() - t0,
        "npz_path": str(npz_path.name),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] saved -> {out_path} (+ {npz_path.name})", flush=True)
    # Quick summary stats to stdout
    for tau in taus:
        rr = np.array(r_star[f"{tau:.2f}"])
        print(
            f"  τ={tau:.2f}: r* mean={rr.mean():.2f} median={np.median(rr):.0f} "
            f"min={rr.min()} max={rr.max()} "
            f"layer-mean range=[{rr.mean(axis=1).min():.2f}, {rr.mean(axis=1).max():.2f}]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
