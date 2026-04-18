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
    p.add_argument(
        "--methods",
        nargs="+",
        default=["no_steer", "ocq_bias_a1", "ocq_bias_a3"],
        help=(
            "Methods to evaluate. Examples: no_steer, ocq_bias_a0.3, "
            "ocq_qbias_b-0.1, ocq_qkv_a0.025_v0_q-0.1, "
            "ocq_qk_layered_a0.3_q-0.1, ocq_quant."
        ),
    )
    p.add_argument("--b-ont", type=str, default="",
                   help="Path to B_ont .pt file for ontology-steered methods.")
    p.add_argument("--ocq-quant-bits", type=int, default=4,
                   help="Residual bits for ocq_quant variants.")
    p.add_argument("--ocq-ont-mode", default="1b", choices=["1a", "1b", "1c"])
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump-failures", action="store_true",
                   help="Save all failing samples (no_match / wrong choice) for dip analysis.")
    p.add_argument("--skip-heads", type=str, default="",
                   help="Rank-1 heads to skip, e.g. 'L0-L3,L27H1'. "
                        "L0-L3 = all heads in layers 0..3; L27H1 = head 1 of layer 27.")
    p.add_argument("--skip-sink-tokens", type=int, default=0,
                   help="Number of leading token positions to exclude from K-bias "
                        "(0 = apply everywhere, 1 = skip BOS/sink).")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--scorer", default="generation",
                   choices=["generation", "label_logprob"],
                   help="'generation': legacy free-gen + first-line parse. "
                        "'label_logprob': teacher-forced closed-set logprob over "
                        "candidates + 'None' (artifact-free).")
    p.add_argument("--include-none-candidate", action="store_true", default=True,
                   help="Include 'None' as a closed-set candidate (label_logprob only).")
    p.add_argument("--per-sample-dump", type=str, default="",
                   help="If set, write per-sample scoring details (label_logprob only) to this path.")
    p.add_argument("--lp-normalize", default="sum",
                   choices=["sum", "mean"],
                   help="Aggregation for label_logprob: 'sum' (raw, length-biased) "
                        "or 'mean' (per-token length-normalized).")
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


def parse_skip_heads(spec: str, n_kv: int) -> set:
    """Parse a skip-heads spec string into a set of (layer, head) tuples.

    Syntax examples:
        "L0-L3"     → all heads (0..n_kv-1) for layers 0,1,2,3
        "L27H1"     → head 1 of layer 27
        "L0-L3,L27H1"  → union of both
    """
    if not spec.strip():
        return set()
    result = set()
    for part in spec.split(","):
        part = part.strip()
        # Range: L<a>-L<b>  → all heads in layers a..b inclusive
        m_range = re.match(r"^L(\d+)-L(\d+)$", part)
        if m_range:
            lo, hi = int(m_range.group(1)), int(m_range.group(2))
            for li in range(lo, hi + 1):
                for hi_idx in range(n_kv):
                    result.add((li, hi_idx))
            continue
        # Single head: L<a>H<b>
        m_single = re.match(r"^L(\d+)H(\d+)$", part)
        if m_single:
            result.add((int(m_single.group(1)), int(m_single.group(2))))
            continue
        raise ValueError(f"cannot parse skip-heads token: '{part}'")
    return result


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
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
):
    handles = []
    L, H, d, r = B_ont.shape
    assert H == n_kv and d == head_dim, f"B_ont shape mismatch"
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L:
                break
            k_proj = layer.self_attn.k_proj
            B_ont_layer = B_ont[layer_idx].clone()  # (H, d, r_ont)
            # Zero out B_ont for skipped heads so the hook becomes a no-op
            if skip_heads:
                for h in range(H):
                    if (layer_idx, h) in skip_heads:
                        B_ont_layer[h].zero_()

            def make_hook(li, B_ont_lh, _skip_sink=skip_sink):
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
                    # Zero out bias for sink/BOS positions to avoid distorting
                    # attention on Mode A models where position 0 dominates
                    if _skip_sink > 0 and T > _skip_sink:
                        K_proj[:, :, :_skip_sink, :] = 0
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
def install_normalized_kbias_hooks(
    model,
    B_ont: torch.Tensor,    # (L, H, d, r_ont)
    alpha: float,
    n_kv: int,
    head_dim: int,
    epsilon: float = 1e-3,
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
):
    """Projection-normalized K-bias (Thm 6.9.5 empirical test).

    e_t = alpha * (B B^T k_t) / max(||B^T k_t||^2 / ||k_t||^2, epsilon)

    Normalizes by the per-token facet-subspace energy ratio so that
    tools with high ontology overlap (general-purpose) don't receive
    disproportionate amplification — eliminates the 'over-generalization'
    failure mode observed on Subtask4 (§5.5 E2 full 497 Δ=-4.6pp).
    """
    handles = []
    L, H, d, r = B_ont.shape
    assert H == n_kv and d == head_dim
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L:
                break
            k_proj = layer.self_attn.k_proj
            B_ont_layer = B_ont[layer_idx].clone()
            if skip_heads:
                for h in range(H):
                    if (layer_idx, h) in skip_heads:
                        B_ont_layer[h].zero_()

            def make_hook(li, B_ont_lh, _skip_sink=skip_sink):
                def hook(module, inputs, output):
                    B, T, D = output.shape
                    if D != n_kv * head_dim:
                        return output
                    K = output.view(B, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                    orig_dtype = K.dtype
                    K_f = K.float()
                    B_dev = B_ont_lh.to(device=K.device, dtype=torch.float32)
                    # coeffs: (B, n_kv, T, r_ont)
                    coeffs = torch.einsum("bhtd,hdr->bhtr", K_f, B_dev)
                    # K_proj: (B, n_kv, T, d) — B B^T k_t
                    K_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_dev)
                    # energy ratio per token: ||B^T k_t||^2 / ||k_t||^2
                    proj_energy = (coeffs ** 2).sum(dim=-1, keepdim=True)  # (B, n_kv, T, 1)
                    k_energy = (K_f ** 2).sum(dim=-1, keepdim=True) + epsilon  # (B, n_kv, T, 1)
                    ratio = proj_energy / k_energy  # (B, n_kv, T, 1)
                    # Normalize: divide by ratio (clamped at epsilon)
                    normalizer = torch.clamp(ratio, min=epsilon)
                    K_proj_normalized = K_proj / normalizer
                    if _skip_sink > 0 and T > _skip_sink:
                        K_proj_normalized[:, :, :_skip_sink, :] = 0
                    K_modified = K_f + alpha * K_proj_normalized
                    return K_modified.permute(0, 2, 1, 3).contiguous().view(B, T, D).to(orig_dtype)
                return hook

            handles.append(k_proj.register_forward_hook(make_hook(layer_idx, B_ont_layer)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def install_contrastive_kbias_hooks(
    model,
    B_ont: torch.Tensor,
    alpha: float,
    n_kv: int,
    head_dim: int,
    dominant_rank: int = 1,    # subtract top-k shared direction
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
):
    """Contrastive K-bias — subtract dominant shared direction from B.

    B_contrastive = B - P_dominant B, where P_dominant projects onto the top-k
    singular directions of B^T B (the shared 'tool-ness' axes most common
    across all tools). Then apply K-bias on the residual contrastive B.

    This fixes the over-generalization failure mode where general-purpose
    tools (high facet overlap) dominate under uniform amplification.
    """
    handles = []
    L, H, d, r = B_ont.shape
    try:
        # Precompute contrastive B per (layer, head)
        B_contrastive = torch.zeros_like(B_ont)
        for li in range(L):
            for hi in range(H):
                B_fh = B_ont[li, hi].float()  # (d, r)
                if B_fh.norm() < 1e-8:
                    continue
                # SVD of B_fh: identify dominant direction
                U, S, Vt = torch.linalg.svd(B_fh, full_matrices=False)
                # Remove top dominant_rank components
                U_res = U.clone()
                U_res[:, :dominant_rank] = 0
                B_contrastive[li, hi] = (U_res @ torch.diag(S) @ Vt).to(B_ont.dtype)

        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L:
                break
            k_proj = layer.self_attn.k_proj
            B_ont_layer = B_contrastive[layer_idx].clone()
            if skip_heads:
                for h in range(H):
                    if (layer_idx, h) in skip_heads:
                        B_ont_layer[h].zero_()

            def make_hook(li, B_ont_lh, _skip_sink=skip_sink):
                def hook(module, inputs, output):
                    B, T, D = output.shape
                    if D != n_kv * head_dim:
                        return output
                    K = output.view(B, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                    orig_dtype = K.dtype
                    K_f = K.float()
                    B_dev = B_ont_lh.to(device=K.device, dtype=torch.float32)
                    coeffs = torch.einsum("bhtd,hdr->bhtr", K_f, B_dev)
                    K_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_dev)
                    if _skip_sink > 0 and T > _skip_sink:
                        K_proj[:, :, :_skip_sink, :] = 0
                    K_modified = K_f + alpha * K_proj
                    return K_modified.permute(0, 2, 1, 3).contiguous().view(B, T, D).to(orig_dtype)
                return hook

            handles.append(k_proj.register_forward_hook(make_hook(layer_idx, B_ont_layer)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def install_vbias_hooks(
    model,
    B_ont: torch.Tensor,     # (L, H, d, r_ont) — reused from K-space; approximate
    alpha: float,
    n_kv: int,
    head_dim: int,
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
):
    """V-side bias: v_t' = v_t + α · B · B^T · v_t.
    Amplifies value projection onto the facet subspace. Mirror of install_kbias_hooks
    but on v_proj. Useful to test whether V-side boost of in-ontology content
    (orthogonal to attention-selection mechanism) helps multi-tool emission.
    """
    handles = []
    L, H, d, r = B_ont.shape
    assert H == n_kv and d == head_dim, "B_ont shape mismatch"
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L:
                break
            v_proj = layer.self_attn.v_proj
            B_ont_layer = B_ont[layer_idx].clone()
            if skip_heads:
                for h in range(H):
                    if (layer_idx, h) in skip_heads:
                        B_ont_layer[h].zero_()

            def make_hook(li, B_ont_lh, _skip_sink=skip_sink):
                def hook(module, inputs, output):
                    B, T, D = output.shape
                    if D != n_kv * head_dim:
                        return output
                    V = output.view(B, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                    orig_dtype = V.dtype
                    V_f = V.float()
                    B_dev = B_ont_lh.to(device=V.device, dtype=torch.float32)
                    coeffs = torch.einsum("bhtd,hdr->bhtr", V_f, B_dev)
                    V_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_dev)
                    if _skip_sink > 0 and T > _skip_sink:
                        V_proj[:, :, :_skip_sink, :] = 0
                    V_modified = V_f + alpha * V_proj
                    return V_modified.permute(0, 2, 1, 3).contiguous().view(B, T, D).to(orig_dtype)
                return hook

            handles.append(v_proj.register_forward_hook(make_hook(layer_idx, B_ont_layer)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def install_q_bias_hooks(
    model,
    B_ont: torch.Tensor,     # (L, H_kv, d, r_ont) — per-kv-head, broadcast to q-heads
    beta: float,             # positive: q += β·BB^T q (boost on-ontology q); negative: subtract
    n_kv: int,
    n_q: int,
    head_dim: int,
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
):
    """Q-side ontology-projection bias: q_t' = q_t + β · B B^T q_t per q-head, with
    each q-head h mapped to kv-head h // (n_q // n_kv) under GQA. Use β > 0 to amplify
    in-ontology Q (Thm 6.17 V-side mirror), β < 0 to apply Q-coverage mask
    (Thm 6.17 first-order coverage subtraction)."""
    handles = []
    L, H_kv, d, r = B_ont.shape
    assert H_kv == n_kv and d == head_dim
    assert n_q % n_kv == 0
    g = n_q // n_kv  # GQA group size
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L:
                break
            q_proj = layer.self_attn.q_proj
            B_ont_layer = B_ont[layer_idx].clone()  # (H_kv, d, r)
            if skip_heads:
                for h in range(H_kv):
                    if (layer_idx, h) in skip_heads:
                        B_ont_layer[h].zero_()
            B_q = B_ont_layer.unsqueeze(1).expand(-1, g, -1, -1).reshape(n_q, d, r)

            def make_hook(li, B_q_layer, _skip_sink=skip_sink):
                def hook(module, inputs, output):
                    B, T, D = output.shape
                    if D != n_q * head_dim:
                        return output
                    Q = output.view(B, T, n_q, head_dim).permute(0, 2, 1, 3).contiguous()
                    orig_dtype = Q.dtype
                    Q_f = Q.float()
                    B_dev = B_q_layer.to(device=Q.device, dtype=torch.float32)
                    coeffs = torch.einsum("bhtd,hdr->bhtr", Q_f, B_dev)
                    Q_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_dev)
                    if _skip_sink > 0 and T > _skip_sink:
                        Q_proj[:, :, :_skip_sink, :] = 0
                    Q_modified = Q_f + beta * Q_proj
                    return Q_modified.permute(0, 2, 1, 3).contiguous().view(B, T, D).to(orig_dtype)
                return hook

            handles.append(q_proj.register_forward_hook(make_hook(layer_idx, B_q)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def install_layer_adaptive_hooks(
    model,
    B_ont: torch.Tensor,    # (L, H_kv, d, r_ont)
    alpha_k: float,          # K-bias strength, applied to early layers only
    beta_q: float,           # Q-coverage strength, applied to later layers only
    n_kv: int,
    n_q: int,
    head_dim: int,
    k_boundary_frac: float = 0.25,   # K applied to first 25% of layers
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
):
    """Layer-Adaptive K+Q. K-bias on the first k_boundary_frac of layers for
    information-encoding precision, Q-coverage on the remaining layers for
    exploration without disturbing output convergence. Motivated by the
    U-shape MSE observation: early layers encode, late layers converge.
    K on all layers breaks Subtask4; K on the first quarter plus Q everywhere
    lifts Subtask4 instead.
    """
    handles = []
    L, H, d, r = B_ont.shape
    assert H == n_kv and d == head_dim
    assert n_q % n_kv == 0
    g = n_q // n_kv
    k_boundary = max(1, int(L * k_boundary_frac))

    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L:
                break

            B_ont_layer = B_ont[layer_idx].clone()
            if skip_heads:
                for h in range(H):
                    if (layer_idx, h) in skip_heads:
                        B_ont_layer[h].zero_()

            if layer_idx < k_boundary and alpha_k != 0:
                k_proj = layer.self_attn.k_proj

                def make_k_hook(li, B_lh, _alpha=alpha_k, _skip_sink=skip_sink):
                    def hook(module, inputs, output):
                        B, T, D = output.shape
                        if D != n_kv * head_dim:
                            return output
                        K = output.view(B, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                        orig_dtype = K.dtype
                        K_f = K.float()
                        B_dev = B_lh.to(device=K.device, dtype=torch.float32)
                        coeffs = torch.einsum("bhtd,hdr->bhtr", K_f, B_dev)
                        K_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_dev)
                        if _skip_sink > 0 and T > _skip_sink:
                            K_proj[:, :, :_skip_sink, :] = 0
                        K_modified = K_f + _alpha * K_proj
                        return K_modified.permute(0, 2, 1, 3).contiguous().view(B, T, D).to(orig_dtype)
                    return hook

                handles.append(k_proj.register_forward_hook(make_k_hook(layer_idx, B_ont_layer)))

            # Q-coverage: applied to ALL layers per paper §4 definition
            # (α_ℓ = α·1{ℓ<L/4}, β_ℓ = β). Earlier code gated this by
            # `layer_idx >= k_boundary` which made K and Q mutually exclusive
            # across layers — a silent deviation from the paper formula.
            if beta_q != 0:
                q_proj = layer.self_attn.q_proj
                B_q = B_ont_layer.unsqueeze(1).expand(-1, g, -1, -1).reshape(n_q, d, r)

                def make_q_hook(li, B_q_layer, _beta=beta_q, _skip_sink=skip_sink):
                    def hook(module, inputs, output):
                        B, T, D = output.shape
                        if D != n_q * head_dim:
                            return output
                        Q = output.view(B, T, n_q, head_dim).permute(0, 2, 1, 3).contiguous()
                        orig_dtype = Q.dtype
                        Q_f = Q.float()
                        B_dev = B_q_layer.to(device=Q.device, dtype=torch.float32)
                        coeffs = torch.einsum("bhtd,hdr->bhtr", Q_f, B_dev)
                        Q_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_dev)
                        if _skip_sink > 0 and T > _skip_sink:
                            Q_proj[:, :, :_skip_sink, :] = 0
                        Q_modified = Q_f + _beta * Q_proj
                        return Q_modified.permute(0, 2, 1, 3).contiguous().view(B, T, D).to(orig_dtype)
                    return hook

                handles.append(q_proj.register_forward_hook(make_q_hook(layer_idx, B_q)))

        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def install_caa_hooks(
    model,
    B_ont: torch.Tensor,    # (L, H_kv, d, r_ont) — we use first column as rank-1 direction
    alpha: float,
    target_layers: Optional[list] = None,
    skip_sink: int = 0,
):
    """CAA-style baseline (Rimsky 2024): rank-1 residual-stream bias.
    Uses B_ont's first column as the "concept direction" — direct
    comparison of rank-1 Q-side bias vs our rank-24 K-side / Q-coverage.
    The bias is applied to the layer output (residual stream) at each
    target layer."""
    handles = []
    L, H_kv, d, r = B_ont.shape
    if target_layers is None:
        target_layers = list(range(L // 2 - 1, L // 2 + 2))  # mid-3 layers
    # First column averaged across heads gives a rank-1 layer direction
    # in head_dim space, then broadcast to full hidden dim
    try:
        for layer_idx in target_layers:
            if layer_idx >= L: continue
            v_per_head = B_ont[layer_idx, :, :, 0]  # (H_kv, d)
            v_layer = v_per_head.reshape(-1)  # (H_kv * d,)
            v_layer = v_layer / (v_layer.norm() + 1e-8)
            layer = model.model.layers[layer_idx]

            def make_hook(li, v_dir, _skip_sink=skip_sink):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    B, T, D = h.shape
                    if v_dir.shape[0] != D:
                        # Pad or truncate to hidden dim
                        if v_dir.shape[0] < D:
                            v = torch.zeros(D, device=h.device, dtype=torch.float32)
                            v[:v_dir.shape[0]] = v_dir.to(h.device).float()
                        else:
                            v = v_dir[:D].to(h.device).float()
                    else:
                        v = v_dir.to(h.device).float()
                    h_f = h.float()
                    add = alpha * v.view(1, 1, D)
                    if _skip_sink > 0 and T > _skip_sink:
                        add = add.repeat(B, T, 1)
                        add[:, :_skip_sink, :] = 0
                    h_modified = (h_f + add).to(h.dtype)
                    if isinstance(output, tuple):
                        return (h_modified,) + output[1:]
                    return h_modified
                return hook

            handles.append(layer.register_forward_hook(make_hook(layer_idx, v_layer)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def install_adaseka_proxy_hooks(
    model,
    B_ont: torch.Tensor,    # (L, H_kv, d, r_ont) used as M-expert basis
    alpha: float,
    M: int,
    n_kv: int,
    n_q: int,
    head_dim: int,
    temperature: float = 0.1,
    skip_sink: int = 0,
):
    """AdaSEKA proxy (Kim et al. 2026): Q-side M-expert max-normalized routing.
    Splits B_ont into M equal-rank facet sub-blocks; for each Q token computes
    soft-max routing over experts (temperature T) and applies the winning
    expert's projection to Q. This mirrors AdaSEKA's max-of-M architecture
    and tests Cor 6.9: should saturate at expert rank r = R/M, gap +17 vs
    our rank-R F-simultaneous."""
    handles = []
    L, H_kv, d, R_total = B_ont.shape
    r_per_expert = R_total // M  # equal split, e.g. R=24, M=3 → r=8
    g = n_q // n_kv
    try:
        for layer_idx in range(L):
            q_proj = model.model.layers[layer_idx].self_attn.q_proj
            B_layer = B_ont[layer_idx].clone()
            # Split into M experts: each is (H_kv, d, r_per_expert)
            experts = [B_layer[:, :, m * r_per_expert:(m + 1) * r_per_expert]
                       for m in range(M)]
            # Broadcast to q-heads
            experts_q = [e.unsqueeze(1).expand(-1, g, -1, -1).reshape(n_q, d, r_per_expert)
                         for e in experts]

            def make_hook(li, exps, _T=temperature, _skip_sink=skip_sink):
                def hook(module, inputs, output):
                    B, T, D = output.shape
                    if D != n_q * head_dim:
                        return output
                    Q = output.view(B, T, n_q, head_dim).permute(0, 2, 1, 3).contiguous()
                    Q_f = Q.float()
                    # Compute per-expert energy: ||B_m^T q||^2 per head
                    energies = []
                    for em in exps:
                        em_dev = em.to(device=Q.device, dtype=torch.float32)
                        coeffs = torch.einsum("bhtd,hdr->bhtr", Q_f, em_dev)
                        energies.append((coeffs ** 2).sum(dim=-1))
                    # Stack and softmax-route over experts
                    E = torch.stack(energies, dim=-1)  # (B, n_q, T, M)
                    routing = torch.softmax(E / _T, dim=-1)  # max-norm at low T
                    # Winning-expert projection (weighted by routing)
                    Q_proj = torch.zeros_like(Q_f)
                    for m, em in enumerate(exps):
                        em_dev = em.to(device=Q.device, dtype=torch.float32)
                        coeffs = torch.einsum("bhtd,hdr->bhtr", Q_f, em_dev)
                        proj_m = torch.einsum("bhtr,hdr->bhtd", coeffs, em_dev)
                        w = routing[..., m].unsqueeze(-1)  # (B, n_q, T, 1)
                        Q_proj = Q_proj + w * proj_m
                    if _skip_sink > 0 and T > _skip_sink:
                        Q_proj[:, :, :_skip_sink, :] = 0
                    Q_modified = Q_f + alpha * Q_proj
                    return Q_modified.permute(0, 2, 1, 3).contiguous().view(B, T, D).to(Q.dtype)
                return hook

            handles.append(q_proj.register_forward_hook(make_hook(layer_idx, experts_q)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def install_qkv_joint_hooks(
    model,
    B_ont: torch.Tensor,
    alpha_k: float,           # K-side facet marker (Thm 6.17 (a))
    gamma_v: float,           # V-side facet amplifier (Thm 6.17 (c))
    beta_q: float,            # Q-side bias; positive=boost, negative=coverage-subtract
    n_kv: int,
    n_q: int,
    head_dim: int,
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
):
    """Thm 6.17 QKV-joint operator (stationary approximation; per-step coverage variant
    requires a generation-loop hook, deferred). Combines K-marker + V-amplifier + Q-bias
    via nested context managers, all over the same B_ont."""
    with install_kbias_hooks(model, B_ont, alpha=alpha_k, n_kv=n_kv,
                             head_dim=head_dim, skip_heads=skip_heads,
                             skip_sink=skip_sink):
        with install_vbias_hooks(model, B_ont, alpha=gamma_v, n_kv=n_kv,
                                 head_dim=head_dim, skip_heads=skip_heads,
                                 skip_sink=skip_sink):
            with install_q_bias_hooks(model, B_ont, beta=beta_q, n_kv=n_kv,
                                      n_q=n_q, head_dim=head_dim,
                                      skip_heads=skip_heads, skip_sink=skip_sink):
                yield


def _build_layer_adaptive_qk_schedule(
    num_layers: int,
    alpha_k: float,
    beta_q: float,
    early_end: int = 5,
    mid_end: int = 18,
    mid_alpha_scale: float = 0.3,
) -> Tuple[List[float], List[float]]:
    """Return per-layer K/Q coefficients for the staged Q+K schedule.

    Stage 1 (layers <= early_end): K-only strong bias.
    Stage 2 (early_end < layers <= mid_end): weak K + active Q coverage.
    Stage 3 (layers > mid_end): Q-only.
    """
    alpha_schedule: List[float] = []
    beta_schedule: List[float] = []
    for layer_idx in range(num_layers):
        if layer_idx <= early_end:
            alpha_schedule.append(alpha_k)
            beta_schedule.append(0.0)
        elif layer_idx <= mid_end:
            alpha_schedule.append(alpha_k * mid_alpha_scale)
            beta_schedule.append(beta_q)
        else:
            alpha_schedule.append(0.0)
            beta_schedule.append(beta_q)
    return alpha_schedule, beta_schedule


@contextmanager
def install_layer_adaptive_qk_hooks(
    model,
    B_ont: torch.Tensor,
    alpha_k: float,
    beta_q: float,
    n_kv: int,
    n_q: int,
    head_dim: int,
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
    early_end: int = 5,
    mid_end: int = 18,
    mid_alpha_scale: float = 0.3,
):
    """Layer-adaptive Q+K steering.

    Early layers apply only K-bias, mid layers use weak K plus Q coverage,
    and late layers use only Q coverage. This directly implements the staged
    schedule proposed after observing early-layer rank-1 amplification and
    mid/late-layer ontology-specific Q effects.
    """
    handles = []
    L, H_kv, d, r = B_ont.shape
    assert H_kv == n_kv and d == head_dim
    assert n_q % n_kv == 0
    g = n_q // n_kv
    alpha_schedule, beta_schedule = _build_layer_adaptive_qk_schedule(
        num_layers=min(L, len(model.model.layers)),
        alpha_k=alpha_k,
        beta_q=beta_q,
        early_end=early_end,
        mid_end=mid_end,
        mid_alpha_scale=mid_alpha_scale,
    )

    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L or layer_idx >= len(alpha_schedule):
                break
            alpha_l = alpha_schedule[layer_idx]
            beta_l = beta_schedule[layer_idx]
            if abs(alpha_l) < 1e-12 and abs(beta_l) < 1e-12:
                continue

            B_ont_layer = B_ont[layer_idx].clone()
            if skip_heads:
                for h in range(H_kv):
                    if (layer_idx, h) in skip_heads:
                        B_ont_layer[h].zero_()

            if abs(alpha_l) >= 1e-12:
                k_proj = layer.self_attn.k_proj

                def make_k_hook(B_ont_lh, alpha_scale, _skip_sink=skip_sink):
                    def hook(module, inputs, output):
                        Bsz, T, D = output.shape
                        if D != n_kv * head_dim:
                            return output
                        K = output.view(Bsz, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                        orig_dtype = K.dtype
                        K_f = K.float()
                        B_dev = B_ont_lh.to(device=K.device, dtype=torch.float32)
                        coeffs = torch.einsum("bhtd,hdr->bhtr", K_f, B_dev)
                        K_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_dev)
                        if _skip_sink > 0 and T > _skip_sink:
                            K_proj[:, :, :_skip_sink, :] = 0
                        K_modified = K_f + alpha_scale * K_proj
                        return K_modified.permute(0, 2, 1, 3).contiguous().view(Bsz, T, D).to(orig_dtype)
                    return hook

                handles.append(k_proj.register_forward_hook(make_k_hook(B_ont_layer, alpha_l)))

            if abs(beta_l) >= 1e-12:
                q_proj = layer.self_attn.q_proj
                B_q = B_ont_layer.unsqueeze(1).expand(-1, g, -1, -1).reshape(n_q, d, r)

                def make_q_hook(B_q_layer, beta_scale, _skip_sink=skip_sink):
                    def hook(module, inputs, output):
                        Bsz, T, D = output.shape
                        if D != n_q * head_dim:
                            return output
                        Q = output.view(Bsz, T, n_q, head_dim).permute(0, 2, 1, 3).contiguous()
                        orig_dtype = Q.dtype
                        Q_f = Q.float()
                        B_dev = B_q_layer.to(device=Q.device, dtype=torch.float32)
                        coeffs = torch.einsum("bhtd,hdr->bhtr", Q_f, B_dev)
                        Q_proj = torch.einsum("bhtr,hdr->bhtd", coeffs, B_dev)
                        if _skip_sink > 0 and T > _skip_sink:
                            Q_proj[:, :, :_skip_sink, :] = 0
                        Q_modified = Q_f + beta_scale * Q_proj
                        return Q_modified.permute(0, 2, 1, 3).contiguous().view(Bsz, T, D).to(orig_dtype)
                    return hook

                handles.append(q_proj.register_forward_hook(make_q_hook(B_q, beta_l)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def install_kvbias_hooks(
    model,
    B_ont: torch.Tensor,
    alpha_k: float,
    alpha_v: float,
    n_kv: int,
    head_dim: int,
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
):
    """Combined K-bias + V-bias (both applied from the same B_ont).
    Uses nested context managers internally.
    """
    with install_kbias_hooks(model, B_ont, alpha=alpha_k, n_kv=n_kv,
                             head_dim=head_dim, skip_heads=skip_heads,
                             skip_sink=skip_sink):
        with install_vbias_hooks(model, B_ont, alpha=alpha_v, n_kv=n_kv,
                                 head_dim=head_dim, skip_heads=skip_heads,
                                 skip_sink=skip_sink):
            yield


def build_facet_masks(
    r_per_pair: Dict[str, List[int]],
    L: int,
    H: int,
    r_ont: int,
    n_facets: int = 4,
) -> torch.Tensor:
    """Build per-(layer, head) facet column mask of shape (L, H, n_facets, r_ont).

    For head (l, h) with Gram-Schmidt basis constructed facet-by-facet in facet_order,
    the first r_per_facet[0] columns belong to facet 0, next r_per_facet[1] to facet 1,
    etc. Truncation to r_ont drops the tail of the last non-empty facet.
    """
    mask = torch.zeros(L, H, n_facets, r_ont)
    for key, r_list in r_per_pair.items():
        # key format: "L{layer}_H{head}"
        try:
            layer = int(key.split("L", 1)[1].split("_", 1)[0])
            head = int(key.split("H", 1)[1])
        except (ValueError, IndexError):
            continue
        if layer >= L or head >= H:
            continue
        cum = 0
        for f_idx, r_f in enumerate(r_list[:n_facets]):
            start = cum
            end = min(cum + int(r_f), r_ont)
            if start < r_ont and end > start:
                mask[layer, head, f_idx, start:end] = 1.0
            cum = end
            if cum >= r_ont:
                break
    return mask


# ---------------------------------------------------------------------
# Per-facet gated K-bias hook (OISA-inspired multi-facet routing)
# ---------------------------------------------------------------------
#
# Instead of a single uniform α on all 24 ontology axes, split the basis
# into n_facets=4 disjoint column groups (function_action, io_type, domain,
# tool_category) and apply an independent energy-fraction gate per facet:
#
#   K' = K + alpha_base * Σ_f g_f(K_bh) · (K_bh · B_f) · B_f^T
#
# where B_f is the column subset of B_ont belonging to facet f for that
# (layer, head), and the gate is computed per (batch, head, token):
#
#   g_f(K_bh) = ||K_bh · B_f||² / (||K_bh||² + eps)
#
# This gives phase-closure automatically: if K has no energy in facet f's
# subspace, g_f ≈ 0 and that facet contributes nothing. Non-tool queries
# that live outside the ontology subspace get approximately zero total
# intervention, whereas tool queries get selective per-facet amplification.
#
# Distinct from AdaSEKA (which uses a single max-normalized mixture over M
# experts and can never fully close) and from flat K-bias (which uniformly
# amplifies all 24 axes regardless of query). Structurally different.

@contextmanager
def install_facet_gated_hooks(
    model,
    B_ont: torch.Tensor,    # (L, H, d, r_ont)
    facet_mask: torch.Tensor,  # (L, H, n_facets, r_ont), 0/1
    alpha_base: float,
    n_kv: int,
    head_dim: int,
    gate_eps: float = 1e-6,
    skip_heads: Optional[set] = None,
    skip_sink: int = 0,
    gate_mode: str = "soft",      # "soft" | "hard_thresh" | "hard_argmax"
    gate_thresh: float = 0.25,    # threshold for "hard_thresh"
):
    handles = []
    L, H, d, r = B_ont.shape
    assert facet_mask.shape == (L, H, facet_mask.shape[2], r)
    n_facets = facet_mask.shape[2]
    try:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx >= L:
                break
            k_proj = layer.self_attn.k_proj
            B_ont_layer = B_ont[layer_idx].clone()  # (H, d, r_ont)
            mask_layer = facet_mask[layer_idx]  # (H, n_facets, r_ont)
            # Zero out B_ont for skipped heads
            if skip_heads:
                for h in range(H):
                    if (layer_idx, h) in skip_heads:
                        B_ont_layer[h].zero_()

            def make_hook(li, B_ont_lh, mask_lh, _skip_sink=skip_sink):
                def hook(module, inputs, output):
                    B_sz, T, D = output.shape
                    if D != n_kv * head_dim:
                        return output
                    K = output.view(B_sz, T, n_kv, head_dim).permute(0, 2, 1, 3).contiguous()
                    orig_dtype = K.dtype
                    K_f = K.float()
                    B_dev = B_ont_lh.to(device=K.device, dtype=torch.float32)  # (H, d, r)
                    M_dev = mask_lh.to(device=K.device, dtype=torch.float32)   # (H, n_facets, r)

                    # Full coeffs: (B, H, T, r_ont)
                    coeffs = torch.einsum("bhtd,hdr->bhtr", K_f, B_dev)

                    # K norm^2 per (batch, head, token): (B, H, T, 1)
                    K_norm_sq = (K_f ** 2).sum(dim=-1, keepdim=True) + gate_eps

                    # Apply each facet mask and accumulate the gated projection.
                    K_increment = torch.zeros_like(K_f)
                    # First pass: compute all g_f (soft energy-ratio)
                    all_g = []
                    all_masked_coeffs = []
                    for f in range(n_facets):
                        mask_f = M_dev[:, f, :]  # (H, r)
                        masked_coeffs = coeffs * mask_f.unsqueeze(0).unsqueeze(2)  # (B, H, T, r)
                        gate_num = (masked_coeffs ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
                        g_f = gate_num / K_norm_sq  # (B, H, T, 1), soft energy-ratio
                        all_g.append(g_f)
                        all_masked_coeffs.append(masked_coeffs)
                    # Transform gates according to gate_mode (R-violation experiment)
                    if gate_mode == "hard_thresh":
                        all_g = [(g > gate_thresh).float() for g in all_g]
                    elif gate_mode == "hard_argmax":
                        stacked = torch.cat(all_g, dim=-1)  # (B, H, T, n_facets)
                        winner = stacked.argmax(dim=-1, keepdim=True)
                        onehot = torch.zeros_like(stacked).scatter_(-1, winner, 1.0)
                        all_g = [onehot[..., f:f+1] for f in range(n_facets)]
                    # Second pass: apply gated projections
                    for f in range(n_facets):
                        K_proj_f = torch.einsum("bhtr,hdr->bhtd", all_masked_coeffs[f], B_dev)
                        K_increment = K_increment + all_g[f] * K_proj_f

                    # Zero out bias for sink/BOS positions
                    if _skip_sink > 0 and T > _skip_sink:
                        K_increment[:, :, :_skip_sink, :] = 0
                    K_modified = K_f + alpha_base * K_increment
                    out = K_modified.permute(0, 2, 1, 3).contiguous().view(B_sz, T, D).to(orig_dtype)
                    return out
                return hook

            handles.append(k_proj.register_forward_hook(make_hook(layer_idx, B_ont_layer, mask_layer)))
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
    """Parse method tag into (kind, params).

    The ``_rank1skip`` suffix on any bias/facet_gated method activates
    head skipping via --skip-heads.  Example: ``ocq_bias_a0.3_rank1skip``.

    The ``_sinkskip`` suffix activates position-aware sink skipping via
    --skip-sink-tokens.  Example: ``ocq_bias_a0.3_sinkskip``.

    Both can be combined: ``ocq_bias_a0.3_rank1skip_sinkskip``.
    """
    # Strip optional suffixes (order-independent)
    use_sinkskip = "_sinkskip" in method
    tag = method.replace("_sinkskip", "")
    use_skip = tag.endswith("_rank1skip")
    tag = tag[: -len("_rank1skip")] if use_skip else tag

    if tag == "no_steer":
        return "no_steer", {}
    if tag.startswith("ocq_bias_a"):
        alpha = float(tag[len("ocq_bias_a"):])
        return "bias", {"alpha": alpha, "use_skip": use_skip, "use_sinkskip": use_sinkskip}
    if tag == "ocq_quant":
        return "quant", {"alpha": 0.0}
    if tag.startswith("ocq_quant_bias_a"):
        alpha = float(tag[len("ocq_quant_bias_a"):])
        return "quant", {"alpha": alpha}
    if tag == "ocq_quant_bias":
        return "quant", {"alpha": 1.0}  # default α=1 for combined
    if tag.startswith("ocq_facet_gated_a"):
        alpha = float(tag[len("ocq_facet_gated_a"):])
        return "facet_gated", {"alpha": alpha, "use_skip": use_skip, "use_sinkskip": use_sinkskip}
    # Normalized K-bias (Thm 6.9.5): ocq_normbias_a<α>
    if tag.startswith("ocq_normbias_a"):
        alpha = float(tag[len("ocq_normbias_a"):])
        return "normbias", {"alpha": alpha, "use_skip": use_skip, "use_sinkskip": use_sinkskip}
    # Contrastive K-bias: ocq_cbias_a<α>_d<dominant_rank> (default d=1)
    if tag.startswith("ocq_cbias_a"):
        rest = tag[len("ocq_cbias_a"):]
        if "_d" in rest:
            a_str, d_str = rest.split("_d", 1)
            return "cbias", {"alpha": float(a_str), "dominant_rank": int(d_str),
                             "use_skip": use_skip, "use_sinkskip": use_sinkskip}
        return "cbias", {"alpha": float(rest), "dominant_rank": 1,
                         "use_skip": use_skip, "use_sinkskip": use_sinkskip}
    # V-side only: ocq_vbias_a<α_V>
    if tag.startswith("ocq_vbias_a"):
        alpha_v = float(tag[len("ocq_vbias_a"):])
        return "vbias", {"alpha_v": alpha_v, "use_skip": use_skip, "use_sinkskip": use_sinkskip}
    # CAA proxy: caa_a<α> — rank-1 residual-stream bias on B_ont's first column
    if tag.startswith("caa_a"):
        alpha = float(tag[len("caa_a"):])
        return "caa", {"alpha": alpha, "use_skip": use_skip, "use_sinkskip": use_sinkskip}
    # AdaSEKA proxy: adaseka_M<m>_a<α>_T<temp> — M-of-1 expert routing
    if tag.startswith("adaseka_M"):
        rest = tag[len("adaseka_M"):]
        m_str, after_M = rest.split("_a", 1)
        a_str, t_str = after_M.split("_T", 1)
        return "adaseka", {
            "M": int(m_str), "alpha": float(a_str), "T": float(t_str),
            "use_skip": use_skip, "use_sinkskip": use_sinkskip,
        }
    # Q-side bias: ocq_qbias_b<β> (positive: boost in-ontology Q; negative: subtract)
    if tag.startswith("ocq_qbias_b"):
        beta = float(tag[len("ocq_qbias_b"):])
        return "qbias", {"beta": beta, "use_skip": use_skip, "use_sinkskip": use_sinkskip}
    # QKV-joint (Thm 6.17 stationary approx): ocq_qkv_a<αk>_v<γv>_q<βq>
    if tag.startswith("ocq_qkv_a"):
        rest = tag[len("ocq_qkv_a"):]
        # rest = "<αk>_v<γv>_q<βq>"
        if "_v" not in rest or "_q" not in rest:
            raise ValueError(f"qkv tag needs _v<γv>_q<βq>: {method}")
        ak_str, after_v = rest.split("_v", 1)
        gv_str, bq_str = after_v.split("_q", 1)
        return "qkv_joint", {
            "alpha_k": float(ak_str),
            "gamma_v": float(gv_str),
            "beta_q": float(bq_str),
            "use_skip": use_skip, "use_sinkskip": use_sinkskip,
        }
    # Layer-adaptive Q+K: ocq_qk_layered_a<αk>_q<βq>
    if tag.startswith("ocq_qk_layered_a"):
        rest = tag[len("ocq_qk_layered_a"):]
        if "_q" not in rest:
            raise ValueError(f"layered qk tag needs _q<beta_q>: {method}")
        ak_str, bq_str = rest.split("_q", 1)
        return "layer_adaptive_qk", {
            "alpha_k": float(ak_str),
            "beta_q": float(bq_str),
            "use_skip": use_skip, "use_sinkskip": use_sinkskip,
        }
    # Combined K+V: ocq_kvbias_a<α_K>_v<α_V>
    if tag.startswith("ocq_kvbias_a"):
        rest = tag[len("ocq_kvbias_a"):]
        if "_v" not in rest:
            raise ValueError(f"kvbias tag needs _v<alpha_v>: {method}")
        ak_str, av_str = rest.split("_v", 1)
        return "kvbias", {
            "alpha_k": float(ak_str),
            "alpha_v": float(av_str),
            "use_skip": use_skip, "use_sinkskip": use_sinkskip,
        }
    # Layer-Adaptive K+Q: ocq_ladapt_k<αK>_q<βQ> or ocq_ladapt_k<αK>_q<βQ>_f<frac>
    if tag.startswith("ocq_ladapt_k"):
        rest = tag[len("ocq_ladapt_k"):]
        if "_q" not in rest:
            raise ValueError(f"layer_adaptive tag needs _q<betaQ>: {method}")
        ak_str, after_q = rest.split("_q", 1)
        if "_f" in after_q:
            bq_str, frac_str = after_q.split("_f", 1)
            k_frac = float(frac_str)
        else:
            bq_str = after_q
            k_frac = 0.25
        return "layer_adaptive", {
            "alpha_k": float(ak_str),
            "beta_q": float(bq_str),
            "k_frac": k_frac,
            "use_skip": use_skip, "use_sinkskip": use_sinkskip,
        }
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
    facet_mask: Optional[torch.Tensor] = None,
    skip_heads: Optional[set] = None,
) -> Dict:
    kind, params = parse_method(method)
    # Only pass skip_heads when the method tag requests it
    effective_skip = skip_heads if params.get("use_skip") else None
    # Only apply sink skipping when the method tag requests it
    effective_sink = args.skip_sink_tokens if params.get("use_sinkskip") else 0

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

    def _score_label_logprob(prompt: str, candidates: List[str]):
        """Teacher-forced closed-set scorer.

        Returns (top1_name, margin, all_scores_dict) where scores is
        {name: sum_logprob} over candidates. The prompt is assumed to end with
        an immediate answer slot (MetaTool ends with 'tool: ').
        Uses raw sum logprob (not length-normalized), the standard
        label-logprob classification objective.
        """
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        prompt_len = prompt_ids.shape[1]
        scores: Dict[str, float] = {}
        # 'None' is represented as literal token "None" appended after prompt.
        cand_list = list(candidates)
        if args.include_none_candidate and "None" not in cand_list:
            cand_list.append("None")
        # One forward per candidate (short continuations, cheap).
        for name in cand_list:
            # Tokenize the candidate string alone; MetaTool prompt ends with
            # 'tool: ' (trailing space), so no extra leading space needed.
            cand_ids = tokenizer(name, return_tensors="pt",
                                 add_special_tokens=False).input_ids.to(args.device)
            if cand_ids.shape[1] == 0:
                scores[name] = float("-inf")
                continue
            full = torch.cat([prompt_ids, cand_ids], dim=1)
            out = model(input_ids=full)
            logits = out.logits  # (1, T, V)
            # logits[t] predicts token at position t+1.
            # We want logP of cand_ids[0..C-1] given prompt_ids.
            # That's logits at positions [prompt_len-1 .. prompt_len+C-2].
            C = cand_ids.shape[1]
            slice_logits = logits[0, prompt_len - 1: prompt_len - 1 + C, :]
            log_probs = torch.log_softmax(slice_logits.float(), dim=-1)
            tgt = cand_ids[0]  # (C,)
            tok_lp = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)  # (C,)
            if args.lp_normalize == "mean":
                scores[name] = float(tok_lp.mean().item())
            else:
                scores[name] = float(tok_lp.sum().item())
        # argmax
        top1 = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0])
        return top1, margin, scores

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
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    elif kind == "quant":
        ctx = install_quant_hooks(
            model, B_ont,
            bits_residual=args.ocq_quant_bits,
            ont_mode=args.ocq_ont_mode,
            n_kv=n_kv, head_dim=head_dim,
            plus_bias_alpha=params["alpha"],
        )
    elif kind == "facet_gated":
        if facet_mask is None:
            raise ValueError("facet_gated method requires facet_mask (from r_per_pair)")
        ctx = install_facet_gated_hooks(
            model, B_ont, facet_mask,
            alpha_base=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    elif kind == "normbias":
        ctx = install_normalized_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    elif kind == "cbias":
        ctx = install_contrastive_kbias_hooks(
            model, B_ont, alpha=params["alpha"],
            dominant_rank=params["dominant_rank"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    elif kind == "vbias":
        ctx = install_vbias_hooks(
            model, B_ont, alpha=params["alpha_v"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    elif kind == "kvbias":
        ctx = install_kvbias_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"], alpha_v=params["alpha_v"],
            n_kv=n_kv, head_dim=head_dim,
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    elif kind == "qbias":
        n_q = model.config.num_attention_heads
        ctx = install_q_bias_hooks(
            model, B_ont, beta=params["beta"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    elif kind == "caa":
        ctx = install_caa_hooks(
            model, B_ont, alpha=params["alpha"],
            skip_sink=effective_sink,
        )
    elif kind == "adaseka":
        n_q = model.config.num_attention_heads
        ctx = install_adaseka_proxy_hooks(
            model, B_ont, alpha=params["alpha"], M=params["M"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            temperature=params["T"],
            skip_sink=effective_sink,
        )
    elif kind == "qkv_joint":
        n_q = model.config.num_attention_heads
        ctx = install_qkv_joint_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"],
            gamma_v=params["gamma_v"],
            beta_q=params["beta_q"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    elif kind == "layer_adaptive_qk":
        n_q = model.config.num_attention_heads
        ctx = install_layer_adaptive_qk_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"],
            beta_q=params["beta_q"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    elif kind == "layer_adaptive":
        n_q = model.config.num_attention_heads
        ctx = install_layer_adaptive_hooks(
            model, B_ont,
            alpha_k=params["alpha_k"],
            beta_q=params["beta_q"],
            n_kv=n_kv, n_q=n_q, head_dim=head_dim,
            k_boundary_frac=params.get("k_frac", 0.25),
            skip_heads=effective_skip,
            skip_sink=effective_sink,
        )
    else:
        raise ValueError(kind)

    per_sample_scores = []
    with ctx:
        for entry in data:
            prompt = entry["action_prompt"]
            gt = entry["tool"]
            cands = parse_candidates(prompt)
            if args.scorer == "label_logprob":
                choice, margin, scores = _score_label_logprob(prompt, cands)
                generation = ""
                if args.per_sample_dump:
                    per_sample_scores.append({
                        "index": entry.get("index"),
                        "gt": gt,
                        "choice": choice,
                        "margin": margin,
                        "scores": scores,
                    })
            else:
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
            elif args.dump_failures:
                is_wrong = (not is_none_query) and (
                    choice is None or choice.lower() != gt.lower()
                )
                if is_wrong:
                    sample_log.append({
                        "index": entry.get("index"),
                        "gt": gt,
                        "cands": cands,
                        "generation": generation[:300],
                        "choice": choice,
                        "reason": "no_match" if choice is None else ("none_pred" if choice == "None" else "wrong_tool"),
                    })

    runtime = time.time() - t0
    if args.per_sample_dump and per_sample_scores:
        dump_path = Path(args.per_sample_dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        method_path = dump_path.parent / f"{dump_path.stem}__{method}.jsonl"
        with method_path.open("w") as f:
            for s in per_sample_scores:
                f.write(json.dumps(s) + "\n")
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
    needs_facet_mask = any(m.startswith("ocq_facet_gated") for m in args.methods)
    B_ont = None
    facet_mask = None
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

        if needs_facet_mask:
            if not (isinstance(payload, dict) and "r_per_pair" in payload):
                raise ValueError(
                    "ocq_facet_gated requires B_ont payload with 'r_per_pair' dict "
                    "(rebuild B_ont with scripts/ocq/build_qwen_metatool_b_ont.py)"
                )
            r_per_pair = payload["r_per_pair"]
            n_facets = len(payload.get("facet_order", ["f0", "f1", "f2", "f3"]))
            r_ont = B_ont.shape[-1]
            facet_mask = build_facet_masks(
                r_per_pair, L=L, H=n_kv, r_ont=r_ont, n_facets=n_facets
            )
            total_per_facet = facet_mask.sum(dim=(0, 1, 3))  # (n_facets,)
            print(
                f"[ocq] facet_mask built: shape {tuple(facet_mask.shape)}, "
                f"total cols per facet = {total_per_facet.tolist()}",
                flush=True,
            )

    # Parse skip-heads spec
    skip_heads = None
    if args.skip_heads:
        skip_heads = parse_skip_heads(args.skip_heads, n_kv)
        print(f"[skip] {len(skip_heads)} (layer,head) pairs skipped from: {args.skip_heads}",
              flush=True)

    results = []
    for method in args.methods:
        print(f"\n[eval] {method}", flush=True)
        res = run_method(
            model, tok, data, method, args,
            B_ont=B_ont, n_kv=n_kv, head_dim=head_dim,
            facet_mask=facet_mask,
            skip_heads=skip_heads,
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
        "skip_heads_spec": args.skip_heads or None,
        "skip_heads_count": len(skip_heads) if skip_heads else 0,
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
