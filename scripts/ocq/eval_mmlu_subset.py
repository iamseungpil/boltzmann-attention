#!/usr/bin/env python3
"""
MMLU subset evaluation for Corollary 6.7 phase-closure empirical check.

Goal
----
Show that the *facet-gated* K-bias hook leaves non-tool (MMLU) accuracy
essentially unchanged, while the *flat* K-bias hook can degrade it. This
is the empirical face of Cor 6.7 (facet phase-closure): if the query
carries no energy in the ontology subspace, the per-facet energy gate
g_f = ||K · B_f||² / ||K||² is ≈ 0 and the facet-gated hook contributes
nothing; whereas the flat hook uniformly amplifies all 24 axes regardless
of query content.

Metric
------
Next-token logit over {" A", " B", " C", " D"}. Accuracy = fraction of
samples where argmax == ground-truth letter.

Methods
-------
Any subset of:
    no_steer                 baseline
    ocq_bias_a<α>            flat K-bias (prior-art-style additive steering)
    ocq_facet_gated_a<α>     facet-gated K-bias (ours, Cor 6.7)

The hook implementations, facet-mask builder, and dtype resolver are
imported from ``eval_metatool_subtask1`` — this file is a thin MMLU
driver, not a reimplementation.

Cross-checks
------------
  - no_steer on Qwen2.5-7B, 1000 samples → ~60-70% (published baseline)
  - ocq_bias_a0.3 → some degradation expected (flat doesn't phase-close)
  - ocq_facet_gated_a0.3 → within ±0.5pp of no_steer  (Cor 6.7 prediction)

CLI
---
    source /home/woori/workspace_common/CDP/poc/set.env && \\
    python scripts/ocq/eval_mmlu_subset.py \\
        --model Qwen/Qwen2.5-7B \\
        --device cuda:0 \\
        --methods no_steer ocq_bias_a0.3 ocq_facet_gated_a0.3 ocq_facet_gated_a1.0 \\
        --b-ont external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt \\
        --n-samples 1000 \\
        --n-shot 5 \\
        --seed 42 \\
        --out reports/axis2_theoretical_verification/exp_mmlu_phase_closure.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from eval_metatool_subtask1 import (  # noqa: E402
    build_facet_masks,
    install_facet_gated_hooks,
    install_kbias_hooks,
    parse_method,
    resolve_dtype,
)

LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--mmlu-config", default="all",
                   help="HF config for cais/mmlu ('all' or a subject name).")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="Number of MMLU test samples (0 = all).")
    p.add_argument("--n-shot", type=int, default=5,
                   help="In-context examples per subject from the dev split.")
    p.add_argument("--methods", nargs="+",
                   default=["no_steer", "ocq_bias_a0.3",
                            "ocq_facet_gated_a0.3", "ocq_facet_gated_a1.0"],
                   help="Methods to evaluate. no_steer / ocq_bias_a<α> / "
                        "ocq_facet_gated_a<α>.")
    p.add_argument("--b-ont", type=str, default="",
                   help="Path to B_ont .pt file (required for non-no_steer).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-ctx", type=int, default=3072,
                   help="Left-truncate prompt tokens to this length.")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--gate-mode", default="soft",
                   choices=["soft", "hard_thresh", "hard_argmax"],
                   help="Facet gate mode (for ocq_facet_gated methods). "
                        "'soft'=Lipschitz energy-ratio (Hypothesis R), "
                        "'hard_thresh'=indicator above threshold, "
                        "'hard_argmax'=one-hot over facets. "
                        "Last two VIOLATE Hypothesis (R) — see Remark 6.14.A.3.")
    p.add_argument("--gate-thresh", type=float, default=0.25,
                   help="Threshold for hard_thresh gate (default=1/n_facets).")
    return p.parse_args()


# ---------------------------------------------------------------------
# MMLU prompt formatting
# ---------------------------------------------------------------------

def format_subject(subject: str) -> str:
    return subject.replace("_", " ")


def format_example(row: dict, include_answer: bool) -> str:
    """One MMLU row → 'Q.\nA. ..\nB. ..\nC. ..\nD. ..\nAnswer: X' block."""
    lines = [row["question"].strip()]
    for letter, choice in zip(LETTERS, row["choices"]):
        lines.append(f"{letter}. {choice}")
    if include_answer:
        lines.append(f"Answer: {LETTERS[row['answer']]}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def build_prompt(
    target: dict,
    fewshot_by_subject: Dict[str, List[dict]],
    n_shot: int,
) -> str:
    subj = target["subject"]
    head = (f"The following are multiple choice questions (with answers) "
            f"about {format_subject(subj)}.\n\n")
    shots = fewshot_by_subject.get(subj, [])[:n_shot]
    shot_blocks = [format_example(s, include_answer=True) for s in shots]
    tail = format_example(target, include_answer=False)
    return head + "\n\n".join(shot_blocks + [tail])


# ---------------------------------------------------------------------
# MMLU loading
# ---------------------------------------------------------------------

def load_mmlu(
    config: str,
    n_samples: int,
    n_shot: int,
    seed: int,
) -> tuple[List[dict], Dict[str, List[dict]]]:
    from datasets import load_dataset

    print(f"[data] loading cais/mmlu config={config}", flush=True)
    test_ds = load_dataset("cais/mmlu", config, split="test")
    dev_ds = load_dataset("cais/mmlu", config, split="dev")

    test_rows = [dict(r) for r in test_ds]
    rng = random.Random(seed)
    rng.shuffle(test_rows)
    if n_samples and n_samples > 0:
        test_rows = test_rows[:n_samples]

    fewshot: Dict[str, List[dict]] = defaultdict(list)
    for r in dev_ds:
        fewshot[r["subject"]].append(dict(r))
    # Deterministic order (dev split is already 5 per subject, but keep
    # it stable just in case).
    for subj in fewshot:
        fewshot[subj].sort(key=lambda x: x["question"])
        fewshot[subj] = fewshot[subj][:n_shot]

    print(f"[data] {len(test_rows)} test samples | "
          f"{len(fewshot)} subjects with dev shots", flush=True)
    return test_rows, fewshot


# ---------------------------------------------------------------------
# Next-token MC scoring
# ---------------------------------------------------------------------

def build_letter_token_ids(tokenizer) -> List[int]:
    """Return [id(' A'), id(' B'), id(' C'), id(' D')]. For tokenizers
    that don't produce a space-prefixed letter as a single token, fall
    back to the no-space variant.
    """
    ids = []
    for letter in LETTERS:
        tok_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(tok_ids) != 1:
            tok_ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(tok_ids) != 1:
                # Last resort: take the last token (model will still rank
                # this as the "Answer:" continuation).
                tok_ids = tok_ids[-1:]
        ids.append(tok_ids[0])
    return ids


@torch.no_grad()
def score_one(
    model,
    tokenizer,
    prompt: str,
    letter_ids: List[int],
    device: str,
    max_ctx: int,
) -> int:
    ids = tokenizer(prompt, return_tensors="pt",
                    truncation=True, max_length=max_ctx).to(device)
    logits = model(**ids).logits  # (1, T, V)
    last = logits[0, -1, :]
    sub = torch.stack([last[i] for i in letter_ids])  # (4,)
    return int(torch.argmax(sub).item())


@contextmanager
def _nullcontext():
    yield


def run_method(
    model,
    tokenizer,
    samples: List[dict],
    fewshot: Dict[str, List[dict]],
    method: str,
    args,
    B_ont: Optional[torch.Tensor],
    facet_mask: Optional[torch.Tensor],
    n_kv: int,
    head_dim: int,
    letter_ids: List[int],
) -> Dict:
    kind, params = parse_method(method)

    if kind == "no_steer":
        ctx = _nullcontext()
    elif kind == "bias":
        ctx = install_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
        )
    elif kind == "facet_gated":
        if facet_mask is None:
            raise ValueError("facet_gated method requires facet_mask")
        ctx = install_facet_gated_hooks(
            model, B_ont, facet_mask,
            alpha_base=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
            gate_mode=args.gate_mode,
            gate_thresh=args.gate_thresh,
        )
    else:
        raise ValueError(f"{kind} not supported in MMLU driver")

    correct = 0
    per_subject = defaultdict(lambda: [0, 0])  # [correct, total]
    t0 = time.time()
    with ctx:
        for i, row in enumerate(samples):
            prompt = build_prompt(row, fewshot, args.n_shot)
            pred_idx = score_one(
                model, tokenizer, prompt, letter_ids,
                device=args.device, max_ctx=args.max_ctx,
            )
            is_correct = (pred_idx == row["answer"])
            if is_correct:
                correct += 1
            per_subject[row["subject"]][1] += 1
            if is_correct:
                per_subject[row["subject"]][0] += 1
            if args.verbose and i < 3:
                print(f"  [{i}] subj={row['subject']} "
                      f"pred={LETTERS[pred_idx]} gt={LETTERS[row['answer']]}",
                      flush=True)
    runtime = time.time() - t0
    total = len(samples)
    per_subj_acc = {
        s: (c / t if t else None) for s, (c, t) in per_subject.items()
    }
    return {
        "method": method,
        "n_samples": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "runtime_s": runtime,
        "per_subject_accuracy": per_subj_acc,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[load] {args.model} on {args.device}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Left-truncate so that any over-long prompt keeps its trailing
    # "Answer:" intact. Without this, a right-truncated prompt would
    # leave logits[0, -1, :] at an arbitrary mid-context token and the
    # MC scoring would silently break. (Latent bug discovered 2026-04-10
    # during Cor 6.7 audit; does not fire at max_ctx=3072 on MMLU-all,
    # max observed length is 3002, but keep the guard in place.)
    tok.truncation_side = "left"
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
    print(f"[cfg] L={L} n_kv={n_kv} n_q={n_q} head_dim={head_dim}",
          flush=True)

    letter_ids = build_letter_token_ids(tok)
    print(f"[cfg] letter token ids = {letter_ids} "
          f"({[tok.decode([i]) for i in letter_ids]})", flush=True)

    samples, fewshot = load_mmlu(
        args.mmlu_config, args.n_samples, args.n_shot, args.seed,
    )

    # Load B_ont + facet_mask if needed
    needs_b_ont = any(m != "no_steer" for m in args.methods)
    needs_facet_mask = any(m.startswith("ocq_facet_gated") for m in args.methods)
    B_ont = None
    facet_mask = None
    if needs_b_ont:
        if not args.b_ont:
            raise ValueError("--b-ont is required for non-no_steer methods")
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
                raise ValueError(
                    "ocq_facet_gated requires B_ont payload with 'r_per_pair'"
                )
            r_per_pair = payload["r_per_pair"]
            n_facets = len(payload.get("facet_order", ["f0", "f1", "f2", "f3"]))
            r_ont = B_ont.shape[-1]
            facet_mask = build_facet_masks(
                r_per_pair, L=L, H=n_kv, r_ont=r_ont, n_facets=n_facets,
            )
            total_per_facet = facet_mask.sum(dim=(0, 1, 3))
            print(f"[ocq] facet_mask: shape {tuple(facet_mask.shape)}, "
                  f"cols per facet = {total_per_facet.tolist()}", flush=True)

    results = []
    for method in args.methods:
        print(f"\n[eval] {method}", flush=True)
        res = run_method(
            model, tok, samples, fewshot, method, args,
            B_ont=B_ont, facet_mask=facet_mask,
            n_kv=n_kv, head_dim=head_dim, letter_ids=letter_ids,
        )
        print(f"[eval] {method}: acc={res['accuracy']*100:.2f}% "
              f"({res['correct']}/{res['n_samples']}) "
              f"runtime={res['runtime_s']:.1f}s", flush=True)
        results.append(res)

    # Summary + Cor 6.7 gate check
    print("\n=== SUMMARY ===")
    base = next((r for r in results if r["method"] == "no_steer"), None)
    base_acc = base["accuracy"] if base else None
    cor67_checks = []
    for r in results:
        if base_acc is not None and r["method"] != "no_steer":
            delta_pp = (r["accuracy"] - base_acc) * 100
            delta_str = f"  Δ={delta_pp:+.2f}pp"
            if r["method"].startswith("ocq_facet_gated"):
                within = abs(delta_pp) <= 0.5
                cor67_checks.append({
                    "method": r["method"],
                    "delta_pp": delta_pp,
                    "within_0.5pp": within,
                })
                delta_str += "  [Cor6.7 " + ("PASS" if within else "FAIL") + "]"
        else:
            delta_str = ""
        print(f"  {r['method']:28s}  acc={r['accuracy']*100:6.2f}%"
              f"  ({r['correct']}/{r['n_samples']}){delta_str}")

    payload = {
        "model": args.model,
        "mmlu_config": args.mmlu_config,
        "n_samples": len(samples),
        "n_shot": args.n_shot,
        "seed": args.seed,
        "methods": args.methods,
        "b_ont_path": args.b_ont if needs_b_ont else None,
        "results": results,
        "cor67_checks": cor67_checks,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
