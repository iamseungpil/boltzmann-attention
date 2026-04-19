#!/usr/bin/env python3
"""F11 — MOFCISS evaluator on MetaTool Subtask4 (multi-tool) / Subtask1 (single-tool).

MOFCISS = Multi-step Ontology-indexed Facet Coverage via Sparse Subtraction.
Mechanism (NEW_THEOREM_TEST.md §5 Phase F11):

    At decode step t, for each (layer, head):
        q_t     = K activation at current token
        c_t     = OMP top-k active anchors (or dense weighted, --no-omp)
        decay_n = exp(-lambda * |{s < t : facet(atom_s) == facet(n)}|)
        delta_K_t = alpha * sum_{n in supp(c_t)} c_t[n] * K_n * decay_n
        K_t     = K_t - delta_K_t

The per-step state (already-emitted facet cells) is updated *between*
generation steps by parsing the accumulated generation for emitted tool
names and looking up each tool's (verb, domain) via the dictionary's
facet_labels.  This breaks both span-invariance (non-linear sparse
selection) and stationarity (history-dependent decay) — the two closure
properties that F10 could not escape.

Usage:
  source /home/woori/venvs/seka_env/bin/activate
  CUDA_VISIBLE_DEVICES=0 python3 scripts/new_theorem_test/eval_metatool_subtask4_mofciss.py \\
      --dictionary external/SEKA/seka_projections/f11-qwen25-7b-metatool-dictionary/dictionary.pt \\
      --alpha 0.3 --decay 0.5 --top-k 5 --benchmark subtask4 --max-samples 200 \\
      --baseline-also --out reports/f11_metatool/f11b_mofciss_lambda05.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Make sibling scripts/ocq importable for prompt / parsing helpers.
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_metatool_subtask1 import (  # noqa: E402
    parse_candidates,
    resolve_dtype,
)
from eval_metatool_subtask4 import (  # noqa: E402
    build_fc_prompt,
    extract_tool_name_sequence,
    compute_metrics,
    compute_stepwise_metrics,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_DATASETS = {
    "subtask4": "/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json",
    "subtask1": "/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json",
}
MAX_TOOLS = {"subtask4": 2, "subtask1": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--dictionary", required=True,
                   help=".pt produced by build_f11_dictionary.py")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="subtraction strength")
    p.add_argument("--decay", type=float, default=0.5,
                   help="facet-cell step decay lambda (0 disables history)")
    p.add_argument("--top-k", type=int, default=5,
                   help="OMP top-k per (B,H,T) position")
    p.add_argument("--no-omp", action="store_true",
                   help="disable OMP: use dense weighted subtraction (F11d)")
    p.add_argument("--benchmark", choices=["subtask4", "subtask1"],
                   default="subtask4")
    p.add_argument("--dataset", default="",
                   help="override dataset path (defaults per --benchmark)")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--baseline-also", action="store_true",
                   help="also run no_steer baseline in same script run")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MOFCISS hook
# ---------------------------------------------------------------------------

@contextmanager
def install_mofciss_hooks(
    model,
    atoms_layered_norm: List[torch.Tensor],   # per-layer (M, H, d), unit-norm
    state: Dict[str, object],
    alpha: float,
    top_k: int,
    use_omp: bool,
    n_kv: int,
    head_dim: int,
):
    """Register forward hooks on every `k_proj` implementing MOFCISS.

    `atoms_layered_norm[li]` has shape (M, H, d) pre-loaded to GPU, pre-
    normalized along the last dim so that inner products equal cosines.

    `state` MUST contain:
        - "decay_per_atom": Tensor (M,) on GPU, updated between steps
    The hook reads but does not write to `state`.
    """
    handles = []
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= len(atoms_layered_norm):
                break
            k_proj = layer.self_attn.k_proj
            atoms_l = atoms_layered_norm[layer_idx]   # (M, H, d)

            def make_hook(_atoms_l, _alpha=alpha, _top_k=top_k, _use_omp=use_omp):
                M_atoms, H, d = _atoms_l.shape

                def hook(module, inputs, output):
                    if output.dim() != 3:
                        return output
                    B, T, D_total = output.shape
                    if D_total != n_kv * head_dim:
                        return output
                    # (B, T, H, d) -> (B, H, T, d)
                    K = output.view(B, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                    orig_dtype = K.dtype
                    K_f = K.float()
                    atoms_dev = _atoms_l.to(dtype=torch.float32)  # no-op if already fp32

                    # inner[b,h,t,m] = <K_f[b,h,t,:], atoms_dev[m,h,:]>
                    inner = torch.einsum("bhtd,mhd->bhtm", K_f, atoms_dev)

                    decay_vec = state["decay_per_atom"].to(device=K.device,
                                                           dtype=torch.float32)

                    if _use_omp:
                        k_eff = min(_top_k, M_atoms)
                        topk_vals, topk_idx = inner.topk(k_eff, dim=-1)  # (B,H,T,k)
                        # Gather selected atoms: atoms_dev[topk_idx, h_idx, :]
                        h_idx = (
                            torch.arange(H, device=K.device)
                            .view(1, H, 1, 1)
                            .expand_as(topk_idx)
                        )
                        selected_atoms = atoms_dev[topk_idx, h_idx, :]  # (B,H,T,k,d)
                        selected_decay = decay_vec[topk_idx]             # (B,H,T,k)
                        weights = topk_vals * selected_decay             # (B,H,T,k)
                        delta = (weights.unsqueeze(-1) * selected_atoms).sum(dim=3)
                    else:
                        # Dense: sum over all atoms with decay-weighted inner products.
                        weighted = inner * decay_vec.view(1, 1, 1, -1)  # (B,H,T,M)
                        delta = torch.einsum("bhtm,mhd->bhtd", weighted, atoms_dev)

                    K_modified = K_f - _alpha * delta
                    out = (
                        K_modified.permute(0, 2, 1, 3)
                        .contiguous()
                        .view(B, T, D_total)
                        .to(orig_dtype)
                    )
                    return out

                return hook

            handles.append(k_proj.register_forward_hook(make_hook(atoms_l)))
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Facet / decay bookkeeping
# ---------------------------------------------------------------------------

def _build_decay_vector(
    emitted_cells: List[Tuple[str, str]],
    facet_labels: Dict[str, Tuple[str, str]],
    plugin_order: List[str],
    decay_lambda: float,
    device: torch.device,
) -> torch.Tensor:
    """decay[n] = exp(-lambda * count(facet_of_n in emitted_cells))."""
    if decay_lambda <= 0 or not emitted_cells:
        return torch.ones(len(plugin_order), device=device, dtype=torch.float32)
    counter = Counter(emitted_cells)
    vec = torch.empty(len(plugin_order), dtype=torch.float32)
    for i, n in enumerate(plugin_order):
        f = facet_labels.get(n)
        c = counter.get(f, 0) if f is not None else 0
        vec[i] = float(torch.exp(torch.tensor(-decay_lambda * c)))
    return vec.to(device)


# ---------------------------------------------------------------------------
# MOFCISS decode loop (manual greedy w/ KV cache)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mofciss_generate(
    model,
    tokenizer,
    fc_prompt: str,
    candidates: List[str],
    facet_labels: Dict[str, Tuple[str, str]],
    plugin_order: List[str],
    state: Dict[str, object],
    args,
    device: torch.device,
    max_tools: int,
    missing_facet_warned: set,
) -> Tuple[List[str], str]:
    """Greedy decode with per-step MOFCISS state updates.

    Returns (pred_sequence, generation_text).  The hook must already be
    installed; this function just mutates `state` between forward passes.
    """
    # Reset per-query state.
    state["emitted_cells"] = []
    state["processed_tools"] = set()
    state["decay_per_atom"] = torch.ones(
        len(plugin_order), device=device, dtype=torch.float32
    )

    enc = tokenizer(fc_prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    past_kv = None
    generated: List[int] = []
    cur_ids = input_ids

    eos_id = tokenizer.eos_token_id

    for _ in range(args.max_new_tokens):
        out = model(input_ids=cur_ids, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())
        generated.append(next_tok)
        if eos_id is not None and next_tok == eos_id:
            break

        # Parse the generation so far for newly emitted tool names.
        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        emitted = extract_tool_name_sequence(gen_text, candidates)

        processed: set = state["processed_tools"]  # type: ignore
        state_changed = False
        for tool in emitted:
            if tool in processed:
                continue
            processed.add(tool)
            if tool in facet_labels:
                facet = facet_labels[tool]
                state["emitted_cells"].append(facet)  # type: ignore
                state_changed = True
            else:
                if tool not in missing_facet_warned:
                    missing_facet_warned.add(tool)
                    print(
                        f"[warn] candidate '{tool}' missing from dictionary "
                        f"facet_labels — emission ignored",
                        flush=True,
                    )

        if state_changed:
            state["decay_per_atom"] = _build_decay_vector(
                state["emitted_cells"],  # type: ignore
                facet_labels,
                plugin_order,
                args.decay,
                device,
            )

        # Stop once enough distinct tools have been emitted.
        if len(processed) >= max_tools:
            break

        cur_ids = torch.tensor([[next_tok]], device=device)

    generation = tokenizer.decode(generated, skip_special_tokens=True)
    pred_sequence = extract_tool_name_sequence(generation, candidates)
    return pred_sequence, generation


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def _aggregate_init(facet_map_present: bool) -> Tuple[Dict[str, float], Dict[str, float]]:
    macro_keys = ["F1", "F_0.5", "EU", "Jaccard", "Exact", "precision", "recall"]
    stepwise_keys = [
        "emitted_any_rate", "emitted_two_rate", "emitted_two_distinct_rate",
        "first_tool_hit_rate", "second_tool_hit_rate", "second_distinct_hit_rate",
        "second_recovery_given_first_hit_rate", "repeated_first_tool_rate",
    ]
    macro = {k: 0.0 for k in macro_keys}
    stepwise = {k: 0.0 for k in stepwise_keys}
    if facet_map_present:
        for k in ["FG-F1", "FG-F_0.5", "FG-EU"]:
            macro[k] = 0.0
        stepwise["repeated_first_facet_rate"] = 0.0
    return macro, stepwise


@torch.no_grad()
def run_mofciss(
    model,
    tokenizer,
    data: List[dict],
    args,
    payload: dict,
    device: torch.device,
    n_kv: int,
    head_dim: int,
) -> Dict:
    """Run MOFCISS over the dataset with step-adaptive K-subtraction hooks."""
    plugin_order: List[str] = payload["plugin_order"]
    facet_labels: Dict[str, Tuple[str, str]] = payload["facet_labels"]
    atoms_stacked: torch.Tensor = payload["atoms_stacked"]  # (M, L, H, d) fp16

    M, L, H, d = atoms_stacked.shape
    assert M == len(plugin_order), "atoms_stacked / plugin_order size mismatch"
    assert H == n_kv and d == head_dim, (
        f"dictionary shape mismatch: atoms (H={H},d={d}) vs model "
        f"(n_kv={n_kv},head_dim={head_dim})"
    )

    # Pre-load and unit-normalize atoms per layer on GPU (once per run).
    atoms_layered_norm: List[torch.Tensor] = []
    for li in range(L):
        a = atoms_stacked[:, li].to(device=device, dtype=torch.float32)  # (M,H,d)
        a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        atoms_layered_norm.append(a.contiguous())

    state: Dict[str, object] = {
        "emitted_cells": [],
        "processed_tools": set(),
        "decay_per_atom": torch.ones(M, device=device, dtype=torch.float32),
    }
    missing_facet_warned: set = set()

    max_tools = MAX_TOOLS[args.benchmark]

    macro, stepwise = _aggregate_init(False)
    per_sample: List[dict] = []

    t0 = time.time()
    with install_mofciss_hooks(
        model,
        atoms_layered_norm,
        state,
        alpha=args.alpha,
        top_k=args.top_k,
        use_omp=(not args.no_omp),
        n_kv=n_kv,
        head_dim=head_dim,
    ):
        for i, entry in enumerate(data):
            action_prompt = entry["action_prompt"]
            gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
            cands = parse_candidates(action_prompt)
            fc_prompt = build_fc_prompt(tokenizer, action_prompt, cands)

            pred_sequence, generation = _mofciss_generate(
                model=model,
                tokenizer=tokenizer,
                fc_prompt=fc_prompt,
                candidates=cands,
                facet_labels=facet_labels,
                plugin_order=plugin_order,
                state=state,
                args=args,
                device=device,
                max_tools=max_tools,
                missing_facet_warned=missing_facet_warned,
            )
            pred: List[str] = []
            for name in pred_sequence:
                if name not in pred:
                    pred.append(name)

            metrics = compute_metrics(pred, gt, facet_map=None)
            stepwise_m = compute_stepwise_metrics(pred_sequence, gt, facet_map=None)
            for k in macro:
                macro[k] += metrics[k]
            for k in stepwise:
                stepwise[k] += stepwise_m.get(k, 0.0)

            per_sample.append({
                "index": entry.get("index", i),
                "query": entry.get("query", "")[:150],
                "gt": gt,
                "pred": pred,
                "pred_sequence": pred_sequence,
                "generation_head": generation[:300],
                "emitted_cells": list(state["emitted_cells"]),  # type: ignore
                "metrics": metrics,
                "stepwise": stepwise_m,
            })
            if args.verbose and i < 3:
                print(
                    f"[{i}] gt={gt} pred={pred} cells={state['emitted_cells']} "
                    f"F1={metrics['F1']:.3f}",
                    flush=True,
                )

    runtime = time.time() - t0
    N = max(len(data), 1)
    for k in macro:
        macro[k] /= N
    for k in stepwise:
        stepwise[k] /= N

    return {
        "method": "mofciss" if not args.no_omp else "mofciss_no_omp",
        "n_queries": len(data),
        "macro": macro,
        "stepwise": stepwise,
        "runtime_s": runtime,
        "per_sample": per_sample,
        "missing_candidates": sorted(missing_facet_warned),
    }


@torch.no_grad()
def run_baseline(
    model,
    tokenizer,
    data: List[dict],
    args,
    device: torch.device,
) -> Dict:
    """No-steer baseline using `model.generate` (mirrors eval_metatool_subtask4)."""
    max_tools = MAX_TOOLS[args.benchmark]  # not enforced on baseline — matches subtask4
    macro, stepwise = _aggregate_init(False)
    per_sample: List[dict] = []

    t0 = time.time()
    for i, entry in enumerate(data):
        action_prompt = entry["action_prompt"]
        gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
        cands = parse_candidates(action_prompt)
        fc_prompt = build_fc_prompt(tokenizer, action_prompt, cands)
        ids = tokenizer(fc_prompt, return_tensors="pt").to(device)
        out = model.generate(
            **ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0, ids["input_ids"].shape[1]:]
        generation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_sequence = extract_tool_name_sequence(generation, cands)
        pred: List[str] = []
        for name in pred_sequence:
            if name not in pred:
                pred.append(name)

        metrics = compute_metrics(pred, gt, facet_map=None)
        stepwise_m = compute_stepwise_metrics(pred_sequence, gt, facet_map=None)
        for k in macro:
            macro[k] += metrics[k]
        for k in stepwise:
            stepwise[k] += stepwise_m.get(k, 0.0)

        per_sample.append({
            "index": entry.get("index", i),
            "query": entry.get("query", "")[:150],
            "gt": gt,
            "pred": pred,
            "pred_sequence": pred_sequence,
            "generation_head": generation[:300],
            "metrics": metrics,
            "stepwise": stepwise_m,
        })
        if args.verbose and i < 3:
            print(
                f"[baseline {i}] gt={gt} pred={pred} F1={metrics['F1']:.3f}",
                flush=True,
            )

    _ = max_tools  # keep reference for clarity; no hard stop on baseline greedy.
    runtime = time.time() - t0
    N = max(len(data), 1)
    for k in macro:
        macro[k] /= N
    for k in stepwise:
        stepwise[k] /= N

    return {
        "method": "no_steer",
        "n_queries": len(data),
        "macro": macro,
        "stepwise": stepwise,
        "runtime_s": runtime,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.dataset:
        args.dataset = DEFAULT_DATASETS[args.benchmark]

    # --- load dictionary (CPU-side; hook will move per-layer slices to GPU). ---
    print(f"[dict] {args.dictionary}", flush=True)
    payload = torch.load(args.dictionary, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict) and "atoms_stacked" in payload, (
        "dictionary .pt missing atoms_stacked — rebuild with build_f11_dictionary.py"
    )
    assert payload.get("n_atoms", 0) > 0, "empty atom dictionary"
    print(
        f"[dict] M={payload['n_atoms']} L={payload['n_layers']} "
        f"n_kv={payload['n_kv']} d={payload['head_dim']} "
        f"model={payload.get('model','?')}",
        flush=True,
    )

    # --- load model ---
    print(f"[load] {args.model}", flush=True)
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
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q)
    device = torch.device(args.device)

    # --- load data ---
    with open(args.dataset) as f:
        data = json.load(f)
    start = max(0, args.start_idx)
    end = start + args.max_samples if args.max_samples > 0 else len(data)
    data = data[start:end]
    print(
        f"[data] {len(data)} queries ({args.benchmark}, "
        f"max_tools={MAX_TOOLS[args.benchmark]})",
        flush=True,
    )

    # --- baseline (optional) ---
    baseline_result: Optional[Dict] = None
    if args.baseline_also:
        print("\n[eval] no_steer baseline", flush=True)
        baseline_result = run_baseline(model, tok, data, args, device)
        print(
            f"[eval] baseline: F1={baseline_result['macro']['F1']:.3f} "
            f"F0.5={baseline_result['macro']['F_0.5']:.3f} "
            f"EU={baseline_result['macro']['EU']:.3f} "
            f"Exact={baseline_result['macro']['Exact']:.3f} "
            f"runtime={baseline_result['runtime_s']:.1f}s",
            flush=True,
        )

    # --- MOFCISS ---
    print(
        f"\n[eval] mofciss alpha={args.alpha} decay={args.decay} "
        f"top_k={args.top_k} use_omp={not args.no_omp}",
        flush=True,
    )
    mofciss_result = run_mofciss(
        model=model,
        tokenizer=tok,
        data=data,
        args=args,
        payload=payload,
        device=device,
        n_kv=n_kv,
        head_dim=head_dim,
    )
    print(
        f"[eval] {mofciss_result['method']}: "
        f"F1={mofciss_result['macro']['F1']:.3f} "
        f"F0.5={mofciss_result['macro']['F_0.5']:.3f} "
        f"EU={mofciss_result['macro']['EU']:.3f} "
        f"Exact={mofciss_result['macro']['Exact']:.3f} "
        f"runtime={mofciss_result['runtime_s']:.1f}s",
        flush=True,
    )

    # --- save ---
    out_payload = {
        "model": args.model,
        "dataset": args.dataset,
        "benchmark": args.benchmark,
        "n_queries": len(data),
        "method": mofciss_result["method"],
        "hyperparams": {
            "alpha": args.alpha,
            "decay": args.decay,
            "top_k": args.top_k,
            "use_omp": not args.no_omp,
            "max_tools": MAX_TOOLS[args.benchmark],
        },
        "macro": mofciss_result["macro"],
        "stepwise": mofciss_result["stepwise"],
        "runtime_s": mofciss_result["runtime_s"],
        "baseline_macro": baseline_result["macro"] if baseline_result else None,
        "baseline_stepwise": baseline_result["stepwise"] if baseline_result else None,
        "baseline_runtime_s": baseline_result["runtime_s"] if baseline_result else None,
        "per_sample": mofciss_result["per_sample"],
        "baseline_per_sample": baseline_result["per_sample"] if baseline_result else None,
        "missing_candidates": mofciss_result.get("missing_candidates", []),
        "dictionary_path": args.dictionary,
        "decode_policy": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tok.eos_token_id,
            "with_kv_cache": True,
        },
        "runtime_config": {
            "device": args.device,
            "dtype": str(dtype),
            "seed": args.seed,
            "attn_implementation": "eager",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
