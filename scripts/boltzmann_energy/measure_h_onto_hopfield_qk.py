#!/usr/bin/env python3
"""H-Onto-Hopfield-QK — Corrected Path C using proper Q·K attention scores.

The prior script (measure_h_onto_hopfield.py) extracted K-vectors for BOTH
queries and tools, computing K·K similarity. This is NOT the Ramsauer-Hopfield
retrieval identity, which requires:
    patterns Ξ ↔ K = W_K @ h   (tool context)
    state   ξ ↔ Q = W_Q @ h   (query)
    score     ↔ softmax(Q·K^T / sqrt(d_k))

Prior baseline top-1 = 0/100 confirmed the K·K formulation is broken.

This script:
  - dual-hooks q_proj + k_proj in a single forward pass (no RoPE applied —
    raw semantic signal, position-free, consistent with H-Wells)
  - builds **per-head** onto subspaces (Q: 28 heads × 128, K: 4 KV-heads × 128)
    via W_Q / W_K pullback of residual-stream concept directions
  - 6-arm GQA retrieval: F / O_Q / O_K / O_QK / Ortho_QK / R_QK

Architecture (Qwen2.5-7B-Instruct):
    d_model=3584, n_q_heads=28, n_kv_heads=4, d_head=128, n_per_kv=7.
    W_Q: (3584, 3584), W_K: (512, 3584).
    Attention: Q-head h ∈ 0..27 uses K-head h//7 ∈ 0..3 (GQA).

Output: reports/boltzmann_energy/onto_hopfield_qk_L18.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Callable

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "new_theorem_test"))
sys.path.insert(0, str(HERE))

from afod_probe_metatool_toolbench import extract_verb, extract_domain  # noqa: E402

# Reuse utilities from prior Path C script (pure math, not model-dependent)
from measure_h_onto_hopfield import (  # noqa: E402
    orthonormal_basis,
    random_orthonormal_subspace,
    softmax_rows,
    retrieval_metrics,
    bootstrap_top1_ci,
    capture_residual_prompt_end,
)

METATOOL_INFO = "/tmp/MetaTool/dataset/plugin_info.json"
METATOOL_ST1 = "/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json"
DEFAULT_OUT = REPO / "reports" / "boltzmann_energy" / "onto_hopfield_qk_L18.json"


# ---------------------------------------------------------------
# Step 1: dual Q + K capture in a single forward pass
# ---------------------------------------------------------------

@torch.no_grad()
def capture_qk_pooled(model, tok, text: str, layer: int,
                     n_q_heads: int, n_kv_heads: int, d_head: int,
                     pooling: str = "prompt_end",
                     chat_template: bool = False) -> Dict[str, np.ndarray] | None:
    """Capture Q and K at `layer` in one forward pass.

    Hooks registered on `layers[layer].self_attn.q_proj` and `.k_proj`.
    Outputs are raw linear projections (pre-RoPE) — pure semantic signal.

    Returns:
        {"Q": (n_q_heads, d_head) float32, "K": (n_kv_heads, d_head) float32}
        (both prompt_end pooled, on CPU, numpy)
    """
    if chat_template:
        msgs = [{"role": "user", "content": text}]
        enc_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        enc_text = text
    enc = tok(enc_text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(next(model.parameters()).device)
    cap: dict = {}

    def _hook_q(mod, inputs, output):
        # output: (B, T, n_q_heads * d_head)
        if output.dim() == 3:
            B, T, D = output.shape
            if D == n_q_heads * d_head:
                cap["Q"] = output[0].detach().float().cpu().view(T, n_q_heads, d_head)

    def _hook_k(mod, inputs, output):
        if output.dim() == 3:
            B, T, D = output.shape
            if D == n_kv_heads * d_head:
                cap["K"] = output[0].detach().float().cpu().view(T, n_kv_heads, d_head)

    attn = model.model.layers[layer].self_attn
    hq = attn.q_proj.register_forward_hook(_hook_q)
    hk = attn.k_proj.register_forward_hook(_hook_k)
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        hq.remove()
        hk.remove()

    if "Q" not in cap or "K" not in cap:
        return None

    Q_seq = cap["Q"]  # (T, n_q_heads, d_head)
    K_seq = cap["K"]  # (T, n_kv_heads, d_head)

    if pooling == "prompt_end":
        Q_pool = Q_seq[-1]  # (n_q_heads, d_head)
        K_pool = K_seq[-1]  # (n_kv_heads, d_head)
    elif pooling == "mean_all":
        Q_pool = Q_seq.mean(dim=0)
        K_pool = K_seq.mean(dim=0)
    else:
        raise ValueError(f"unknown pooling {pooling}")

    return {"Q": Q_pool.numpy().astype(np.float32),
            "K": K_pool.numpy().astype(np.float32)}


# ---------------------------------------------------------------
# Step 3-4: per-head pullback + SVD orthonormalization
# ---------------------------------------------------------------

def pullback_per_head(U_resid: np.ndarray, W: torch.Tensor,
                     n_heads: int, d_head: int) -> np.ndarray:
    """Pull residual-space direction matrix U_resid (n_cat, d_model) through W ∈ (n_heads*d_head, d_model).

    Returns V ∈ (n_cat, n_heads, d_head): per-head concept directions.
    Computed on CPU (W cast to float32).
    """
    U_t = torch.from_numpy(U_resid).to(dtype=W.dtype, device=W.device)  # (n_cat, d_model)
    V_all = U_t @ W.T  # (n_cat, n_heads*d_head)
    V = V_all.detach().cpu().float().numpy()  # (n_cat, n_heads*d_head)
    return V.reshape(V.shape[0], n_heads, d_head)


def per_head_subspace(V: np.ndarray, tol: float = 1e-5) -> List[np.ndarray]:
    """V: (n_cat, n_heads, d_head). For each head h, SVD of (d_head, n_cat) stacking columns.

    Returns list[n_heads] of U^(h) ∈ R^(d_head, r_h).
    """
    n_cat, n_heads, d_head = V.shape
    bases: List[np.ndarray] = []
    for h in range(n_heads):
        D_h = V[:, h, :].T  # (d_head, n_cat)
        U_h = orthonormal_basis(D_h, tol=tol)
        bases.append(U_h)
    return bases


def projectors_from_bases(bases: List[np.ndarray], d_head: int) -> np.ndarray:
    """Build per-head projector tensor (n_heads, d_head, d_head) from list of (d_head, r_h) bases."""
    n_heads = len(bases)
    P = np.zeros((n_heads, d_head, d_head), dtype=np.float32)
    for h, U_h in enumerate(bases):
        if U_h.shape[1] == 0:
            # empty subspace → zero projector
            continue
        P[h] = (U_h @ U_h.T).astype(np.float32)
    return P


# ---------------------------------------------------------------
# Step 5: Park Thm 8 cross-projector norm (per-head)
# ---------------------------------------------------------------

def park_thm8_fro(bases_A: List[np.ndarray], bases_B: List[np.ndarray]) -> Tuple[float, List[float]]:
    """For each head h: compute ||P_A^(h) @ P_B^(h)||_F / sqrt(min(r_A^h, r_B^h)).

    Returns (mean_fro, per_head_fros).
    """
    assert len(bases_A) == len(bases_B)
    per_head = []
    for U_A, U_B in zip(bases_A, bases_B):
        r_A, r_B = U_A.shape[1], U_B.shape[1]
        if r_A == 0 or r_B == 0:
            per_head.append(0.0)
            continue
        P_A = U_A @ U_A.T
        P_B = U_B @ U_B.T
        cross = P_A @ P_B
        fro = float(np.linalg.norm(cross, ord="fro"))
        normalizer = float(np.sqrt(min(r_A, r_B)))
        per_head.append(fro / normalizer if normalizer > 0 else 0.0)
    return float(np.mean(per_head)), per_head


# ---------------------------------------------------------------
# Step 6: GQA-aware retrieval scoring
# ---------------------------------------------------------------

def compute_scores_gqa(Q: np.ndarray, K: np.ndarray,
                      P_Q: np.ndarray | None, P_K: np.ndarray | None,
                      n_per_kv: int, use_ortho_complement: bool = False,
                      d_head: int = 128) -> np.ndarray:
    """GQA retrieval score matrix (n_q_queries, n_tools).

    Q: (n_q_queries, n_q_heads, d_head)
    K: (n_tools, n_kv_heads, d_head)
    P_Q: (n_q_heads, d_head, d_head) or None (Identity)
    P_K: (n_kv_heads, d_head, d_head) or None (Identity)
    n_per_kv: heads per KV group (h → h // n_per_kv mapping)
    use_ortho_complement: if True, use (I - P_*) instead of P_*

    score(q, t) = (1/sqrt(d_head)) · Σ_h (ProjQ_h Q_q^h) · (ProjK_{h//n_per_kv} K_t^{h//n_per_kv})
    """
    n_q, n_q_heads, dh = Q.shape
    n_t, n_kv_heads, _ = K.shape
    beta = 1.0 / np.sqrt(dh)

    # Effective per-head transforms
    def _effective(P, n_heads):
        if P is None:
            # Identity per head
            I = np.eye(dh, dtype=np.float32)
            return np.broadcast_to(I, (n_heads, dh, dh)).copy()
        if use_ortho_complement:
            I = np.eye(dh, dtype=np.float32)
            return (I[None] - P).astype(np.float32)
        return P.astype(np.float32)

    TQ = _effective(P_Q, n_q_heads)  # (n_q_heads, dh, dh)
    TK = _effective(P_K, n_kv_heads)  # (n_kv_heads, dh, dh)

    # Transformed Q: Q'[q, h, :] = Q[q, h, :] @ TQ[h].T
    # (projection is symmetric, but be explicit)
    Q_proj = np.einsum("qhd,hde->qhe", Q, TQ)  # (n_q, n_q_heads, dh)
    K_proj = np.einsum("tgd,gde->tge", K, TK)  # (n_tools, n_kv_heads, dh)

    # For each Q head h, pair with K head h // n_per_kv
    # score = Σ_h (Q_proj[:, h] · K_proj[:, h//g]) summed over heads
    # Use head-expansion on K: K_expanded[:, h] = K_proj[:, h // n_per_kv]
    head_to_kv = np.arange(n_q_heads) // n_per_kv  # (n_q_heads,)
    K_expanded = K_proj[:, head_to_kv, :]  # (n_tools, n_q_heads, dh)

    # score[q, t] = Σ_h Σ_d Q_proj[q,h,d] * K_expanded[t,h,d]
    # = einsum "qhd,thd->qt"
    logits = np.einsum("qhd,thd->qt", Q_proj, K_expanded)
    scores = beta * logits
    # Return raw logits (softmax-ready); downstream retrieval_metrics uses argmax so
    # the softmax transform is unnecessary for ranking but we keep it for basin_depth
    # interpretability on the probability scale.
    return softmax_rows(scores)


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--pooling", default="prompt_end")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    t_start = time.time()

    partial: Dict = {
        "variant": "onto_hopfield_qk_L18",
        "model": args.model,
        "dataset": "MetaTool-Subtask1",
        "layer": args.layer,
        "pooling": args.pooling,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
    }

    try:
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)

        # -- Step 0: catalog + query pool (reused from H-Wells) --
        print("[data] loading plugin_info + Subtask1 ...", flush=True)
        info = json.load(open(METATOOL_INFO))
        st1 = json.load(open(METATOOL_ST1))

        plugins: Dict[str, dict] = {}
        for p in info:
            n = p.get("name_for_model") or p.get("name_for_human")
            if not n:
                continue
            desc = (p.get("description_for_model") or "").strip()
            if not desc:
                continue
            plugins[n] = {
                "desc": desc[:400],
                "verb": extract_verb(desc),
                "domain": extract_domain(desc),
            }
        plugin_names = sorted(plugins.keys())
        plugin_idx = {n: i for i, n in enumerate(plugin_names)}
        n_tools = len(plugin_names)
        verb_labels = [plugins[n]["verb"] for n in plugin_names]
        domain_labels = [plugins[n]["domain"] for n in plugin_names]
        verb_uniq = sorted(set(verb_labels))
        domain_uniq = sorted(set(domain_labels))
        n_verb = len(verb_uniq)
        n_domain = len(domain_uniq)
        print(f"[catalog] n_tools={n_tools}  n_verb={n_verb}  n_domain={n_domain}", flush=True)
        partial.update({
            "n_tools": n_tools,
            "n_verb_directions": n_verb,
            "n_domain_directions": n_domain,
        })

        pool = [e for e in st1 if e.get("tool") in plugins]
        print(f"[queries pool] {len(pool)} / {len(st1)} GT ∈ catalog", flush=True)
        assert len(pool) >= args.n_queries
        idx_perm = rng.permutation(len(pool))[:args.n_queries]
        queries = [pool[i] for i in idx_perm]
        n_q = len(queries)
        partial["n_queries"] = n_q
        print(f"[queries] using N={n_q}", flush=True)

        # -- Load model --
        print(f"[model] loading {args.model} fp16 on {args.device}", flush=True)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16,
        ).to(args.device)
        model.eval()
        cfg = model.config
        n_q_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        d_model = cfg.hidden_size
        d_head = d_model // n_q_heads
        n_per_kv = n_q_heads // n_kv_heads
        L_total = len(model.model.layers)
        print(f"[config] L_total={L_total}  n_q={n_q_heads}  n_kv={n_kv_heads}  "
              f"d_model={d_model}  d_head={d_head}  n_per_kv={n_per_kv}  layer={args.layer}",
              flush=True)
        assert 0 <= args.layer < L_total
        assert n_q_heads % n_kv_heads == 0, "GQA group size must divide n_q_heads"
        partial.update({
            "n_q_heads": n_q_heads,
            "n_kv_heads": n_kv_heads,
            "gqa_group_size": n_per_kv,
            "d_head": d_head,
            "d_model": d_model,
        })

        layer_mod = model.model.layers[args.layer]
        attn = layer_mod.self_attn
        # W_Q shape = (n_q_heads*d_head, d_model); W_K shape = (n_kv_heads*d_head, d_model)
        W_Q = attn.q_proj.weight.detach().float()  # on device, fp32
        W_K = attn.k_proj.weight.detach().float()
        assert W_Q.shape == (n_q_heads * d_head, d_model)
        assert W_K.shape == (n_kv_heads * d_head, d_model)
        print(f"[weights] W_Q {tuple(W_Q.shape)}  W_K {tuple(W_K.shape)}", flush=True)

        # -- Step 2: residual-stream directions at L=args.layer input --
        print(f"[Step 2] extracting residuals for {n_tools} tools at L={args.layer} input", flush=True)
        H = np.zeros((n_tools, d_model), dtype=np.float32)
        t0 = time.time()
        for i, name in enumerate(plugin_names):
            h = capture_residual_prompt_end(model, tok, plugins[name]["desc"], args.layer)
            if h is None:
                raise RuntimeError(f"residual capture failed for {name}")
            H[i] = h.numpy()
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_tools - i - 1)
                print(f"    [resid {i+1}/{n_tools}] t={elapsed:.1f}s eta={eta:.1f}s", flush=True)

        # Park directions per category
        verb_arr = np.array(verb_labels)
        domain_arr = np.array(domain_labels)

        def compute_dirs(arr, uniq):
            D = np.zeros((len(uniq), d_model), dtype=np.float32)
            for ci, c in enumerate(uniq):
                pos = (arr == c)
                neg = ~pos
                if pos.sum() == 0 or neg.sum() == 0:
                    continue
                d = H[pos].mean(axis=0) - H[neg].mean(axis=0)
                nrm = np.linalg.norm(d)
                if nrm > 1e-8:
                    D[ci] = d / nrm
            return D

        U_verb = compute_dirs(verb_arr, verb_uniq)   # (n_verb, d_model)
        U_dom = compute_dirs(domain_arr, domain_uniq)  # (n_domain, d_model)
        U_all = np.concatenate([U_verb, U_dom], axis=0)  # (n_verb+n_domain, d_model)
        print(f"[Step 2] U_verb {U_verb.shape} U_dom {U_dom.shape} U_all {U_all.shape}", flush=True)

        # -- Step 3: W_Q / W_K pullback per-head --
        print("[Step 3] W_Q / W_K pullback → per-head concept directions", flush=True)
        V_verb_Q = pullback_per_head(U_verb, W_Q, n_q_heads, d_head)   # (n_verb, n_q_heads, d_head)
        V_dom_Q = pullback_per_head(U_dom, W_Q, n_q_heads, d_head)
        V_all_Q = pullback_per_head(U_all, W_Q, n_q_heads, d_head)

        V_verb_K = pullback_per_head(U_verb, W_K, n_kv_heads, d_head)
        V_dom_K = pullback_per_head(U_dom, W_K, n_kv_heads, d_head)
        V_all_K = pullback_per_head(U_all, W_K, n_kv_heads, d_head)
        print(f"[Step 3] Q-pull verb={V_verb_Q.shape} dom={V_dom_Q.shape} all={V_all_Q.shape}",
              flush=True)
        print(f"[Step 3] K-pull verb={V_verb_K.shape} dom={V_dom_K.shape} all={V_all_K.shape}",
              flush=True)

        # -- Step 4: per-head SVD orthonormalization --
        print("[Step 4] per-head SVD subspaces", flush=True)
        bases_Q_verb = per_head_subspace(V_verb_Q)
        bases_Q_dom = per_head_subspace(V_dom_Q)
        bases_Q_all = per_head_subspace(V_all_Q)  # used for O_Q / O_QK / Ortho_QK

        bases_K_verb = per_head_subspace(V_verb_K)
        bases_K_dom = per_head_subspace(V_dom_K)
        bases_K_all = per_head_subspace(V_all_K)

        ranks_Q_all = [b.shape[1] for b in bases_Q_all]
        ranks_K_all = [b.shape[1] for b in bases_K_all]
        print(f"[Step 4] Q-all per-head ranks min={min(ranks_Q_all)} max={max(ranks_Q_all)} "
              f"mean={np.mean(ranks_Q_all):.1f}", flush=True)
        print(f"[Step 4] K-all per-head ranks min={min(ranks_K_all)} max={max(ranks_K_all)} "
              f"mean={np.mean(ranks_K_all):.1f}", flush=True)

        # Random controls: 49 Gaussian dirs → pullback → per-head SVD (same rank budget)
        rng_r = np.random.default_rng(args.seed)
        R_resid = rng_r.standard_normal(size=(n_verb + n_domain, d_model)).astype(np.float32)
        R_resid /= (np.linalg.norm(R_resid, axis=1, keepdims=True) + 1e-12)
        V_R_Q = pullback_per_head(R_resid, W_Q, n_q_heads, d_head)
        V_R_K = pullback_per_head(R_resid, W_K, n_kv_heads, d_head)
        bases_R_Q = per_head_subspace(V_R_Q)
        bases_R_K = per_head_subspace(V_R_K)

        P_Q_onto = projectors_from_bases(bases_Q_all, d_head)   # (n_q_heads, dh, dh)
        P_K_onto = projectors_from_bases(bases_K_all, d_head)   # (n_kv_heads, dh, dh)
        P_Q_rand = projectors_from_bases(bases_R_Q, d_head)
        P_K_rand = projectors_from_bases(bases_R_K, d_head)

        # -- Step 5: Park Thm 8 in Q and K spaces --
        mean_fro_Q, per_head_fros_Q = park_thm8_fro(bases_Q_verb, bases_Q_dom)
        mean_fro_K, per_head_fros_K = park_thm8_fro(bases_K_verb, bases_K_dom)
        g_park_Q = bool(mean_fro_Q < 0.3)
        g_park_K = bool(mean_fro_K < 0.3)
        print(f"[Step 5] Park8 Q-space: mean_fro={mean_fro_Q:.4f}  pass={g_park_Q}", flush=True)
        print(f"[Step 5] Park8 K-space: mean_fro={mean_fro_K:.4f}  pass={g_park_K}", flush=True)

        # -- Step 6: extract Q for queries, K for tools --
        print(f"[Step 6a] extracting Q for {n_q} queries", flush=True)
        Q_mat = np.zeros((n_q, n_q_heads, d_head), dtype=np.float32)
        gt_idx = np.zeros(n_q, dtype=np.int64)
        t0 = time.time()
        for qi, entry in enumerate(queries):
            res = capture_qk_pooled(model, tok, entry["query"], args.layer,
                                    n_q_heads, n_kv_heads, d_head,
                                    pooling=args.pooling, chat_template=False)
            if res is None:
                raise RuntimeError(f"query Q capture failed at qi={qi}")
            Q_mat[qi] = res["Q"]
            gt_idx[qi] = plugin_idx[entry["tool"]]
            if (qi + 1) % 50 == 0 or qi == 0:
                elapsed = time.time() - t0
                eta = elapsed / (qi + 1) * (n_q - qi - 1)
                print(f"    [Q {qi+1}/{n_q}] t={elapsed:.1f}s eta={eta:.1f}s", flush=True)

        print(f"[Step 6b] extracting K for {n_tools} tools", flush=True)
        K_mat = np.zeros((n_tools, n_kv_heads, d_head), dtype=np.float32)
        t0 = time.time()
        for i, name in enumerate(plugin_names):
            res = capture_qk_pooled(model, tok, plugins[name]["desc"], args.layer,
                                    n_q_heads, n_kv_heads, d_head,
                                    pooling=args.pooling, chat_template=False)
            if res is None:
                raise RuntimeError(f"tool K capture failed for {name}")
            K_mat[i] = res["K"]
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_tools - i - 1)
                print(f"    [K {i+1}/{n_tools}] t={elapsed:.1f}s eta={eta:.1f}s", flush=True)

        # Free GPU memory before scoring
        del model, W_Q, W_K
        torch.cuda.empty_cache()

        # -- Step 6c: 6-arm scoring --
        print("[Step 6c] 6-arm retrieval", flush=True)

        # Arm F: no projection
        print("    [arm F] ...", flush=True)
        scores_F = compute_scores_gqa(Q_mat, K_mat, None, None, n_per_kv,
                                       use_ortho_complement=False, d_head=d_head)
        # Arm O_Q: Q projected onto onto subspace, K identity
        print("    [arm O_Q] ...", flush=True)
        scores_O_Q = compute_scores_gqa(Q_mat, K_mat, P_Q_onto, None, n_per_kv,
                                         use_ortho_complement=False, d_head=d_head)
        # Arm O_K: Q identity, K projected
        print("    [arm O_K] ...", flush=True)
        scores_O_K = compute_scores_gqa(Q_mat, K_mat, None, P_K_onto, n_per_kv,
                                         use_ortho_complement=False, d_head=d_head)
        # Arm O_QK: both projected
        print("    [arm O_QK] ...", flush=True)
        scores_O_QK = compute_scores_gqa(Q_mat, K_mat, P_Q_onto, P_K_onto, n_per_kv,
                                          use_ortho_complement=False, d_head=d_head)
        # Arm Ortho_QK: orthogonal complement on both
        print("    [arm Ortho_QK] ...", flush=True)
        scores_Ortho_QK = compute_scores_gqa(Q_mat, K_mat, P_Q_onto, P_K_onto, n_per_kv,
                                              use_ortho_complement=True, d_head=d_head)
        # Arm R_QK: random projectors
        print("    [arm R_QK] ...", flush=True)
        scores_R_QK = compute_scores_gqa(Q_mat, K_mat, P_Q_rand, P_K_rand, n_per_kv,
                                          use_ortho_complement=False, d_head=d_head)

        m_F = retrieval_metrics(scores_F, gt_idx)
        m_O_Q = retrieval_metrics(scores_O_Q, gt_idx)
        m_O_K = retrieval_metrics(scores_O_K, gt_idx)
        m_O_QK = retrieval_metrics(scores_O_QK, gt_idx)
        m_Ortho_QK = retrieval_metrics(scores_Ortho_QK, gt_idx)
        m_R_QK = retrieval_metrics(scores_R_QK, gt_idx)

        # -- Step 7: bootstrap CIs --
        print(f"[Step 7] bootstrap CI (n={args.n_bootstrap})", flush=True)
        ci_F = bootstrap_top1_ci(m_F["_top1_correct"], args.n_bootstrap, args.seed)
        ci_O_Q = bootstrap_top1_ci(m_O_Q["_top1_correct"], args.n_bootstrap, args.seed + 1)
        ci_O_K = bootstrap_top1_ci(m_O_K["_top1_correct"], args.n_bootstrap, args.seed + 2)
        ci_O_QK = bootstrap_top1_ci(m_O_QK["_top1_correct"], args.n_bootstrap, args.seed + 3)
        ci_Ortho_QK = bootstrap_top1_ci(m_Ortho_QK["_top1_correct"], args.n_bootstrap, args.seed + 4)
        ci_R_QK = bootstrap_top1_ci(m_R_QK["_top1_correct"], args.n_bootstrap, args.seed + 5)

        def _fmt(name, m, ci):
            print(f"[arm {name:>8s}] top1={m['top1']:.4f}  top5={m['top5']:.4f}  "
                  f"mrr={m['mrr']:.4f}  CI95=[{ci[0]:.4f}, {ci[2]:.4f}]", flush=True)
        _fmt("F", m_F, ci_F)
        _fmt("O_Q", m_O_Q, ci_O_Q)
        _fmt("O_K", m_O_K, ci_O_K)
        _fmt("O_QK", m_O_QK, ci_O_QK)
        _fmt("Ortho_QK", m_Ortho_QK, ci_Ortho_QK)
        _fmt("R_QK", m_R_QK, ci_R_QK)

        # -- Step 8: pre-reg gates --
        t1_F = m_F["top1"]
        t1_O_Q = m_O_Q["top1"]
        t1_O_K = m_O_K["top1"]
        t1_O_QK = m_O_QK["top1"]
        t1_Ortho = m_Ortho_QK["top1"]
        t1_R = m_R_QK["top1"]

        baseline_dead = bool(t1_F < 0.10)
        eps = 1e-9
        if t1_F < eps:
            g_qk_1 = g_qk_2 = g_qk_3 = False
            g_q_only = g_k_only = False
        else:
            g_qk_1 = bool((t1_O_QK / t1_F >= 0.8) and (ci_O_QK[0] / t1_F >= 0.7))
            g_qk_2 = bool((t1_Ortho / t1_F) <= 0.6)
            g_qk_3 = bool(abs(t1_R - t1_Ortho) / t1_F <= 0.15)
            g_q_only = bool(0.7 <= (t1_O_Q / t1_F) <= 1.1)
            g_k_only = bool(0.7 <= (t1_O_K / t1_F) <= 1.1)

        gates = {
            "G_OntoHopf_QK_1_pass": g_qk_1,
            "G_OntoHopf_QK_2_pass": g_qk_2,
            "G_OntoHopf_QK_3_pass": g_qk_3,
            "G_OntoHopf_Q_only_pass": g_q_only,
            "G_OntoHopf_K_only_pass": g_k_only,
            "G_Park_Thm8_Q_pass": g_park_Q,
            "G_Park_Thm8_K_pass": g_park_K,
        }
        print(f"[gates] {gates}", flush=True)

        # -- Joint outcome --
        if baseline_dead:
            joint = "BASELINE_DEAD_QK"
        elif g_qk_1 and g_qk_2 and g_qk_3:
            joint = "STRONG_POSITIVE"
        elif g_qk_1 and not g_qk_2:
            joint = "PARTIAL_POSITIVE_ONTOLOGY"
        elif g_q_only and (t1_O_Q - t1_O_K) > 0.05:
            joint = "Q_ONLY_POSITIVE"
        elif g_k_only and (t1_O_K - t1_O_Q) > 0.05:
            joint = "K_ONLY_POSITIVE"
        elif (max(t1_F, t1_O_QK, t1_Ortho, t1_R) - min(t1_F, t1_O_QK, t1_Ortho, t1_R)) <= 0.10:
            joint = "PROJECTION_NEUTRAL"
        elif (t1_O_QK < t1_R) and (t1_R < t1_F):
            joint = "PROJECTION_DETRIMENTAL"
        else:
            joint = "PARTIAL"
        print(f"[joint] {joint}", flush=True)

        # -- Git SHA --
        try:
            import subprocess
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=str(REPO)).decode().strip()
        except Exception:
            sha = "unknown"

        out = {
            "variant": "onto_hopfield_qk_L18",
            "model": args.model,
            "dataset": "MetaTool-Subtask1",
            "n_queries": n_q,
            "n_tools": n_tools,
            "layer": args.layer,
            "pooling": args.pooling,
            "direction_source": f"residual-L{args.layer}-prompt-end",
            "n_verb_directions": n_verb,
            "n_domain_directions": n_domain,
            "d_model": d_model,
            "d_head": d_head,
            "n_q_heads": n_q_heads,
            "n_kv_heads": n_kv_heads,
            "gqa_group_size": n_per_kv,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "per_head_ranks": {
                "Q_all": ranks_Q_all,
                "K_all": ranks_K_all,
                "Q_verb": [b.shape[1] for b in bases_Q_verb],
                "Q_dom": [b.shape[1] for b in bases_Q_dom],
                "K_verb": [b.shape[1] for b in bases_K_verb],
                "K_dom": [b.shape[1] for b in bases_K_dom],
            },
            "park_thm8_Q": {
                "mean_fro": mean_fro_Q,
                "per_head_fros": per_head_fros_Q,
                "pass": g_park_Q,
            },
            "park_thm8_K": {
                "mean_fro": mean_fro_K,
                "per_head_fros": per_head_fros_K,
                "pass": g_park_K,
            },
            "arm_F": {
                "top1": m_F["top1"], "top5": m_F["top5"], "mrr": m_F["mrr"],
                "basin_depth_mean": m_F["basin_depth_mean"],
                "top1_ci_95": [ci_F[0], ci_F[2]], "top1_ci_mean": ci_F[1],
            },
            "arm_O_Q": {
                "top1": m_O_Q["top1"], "top5": m_O_Q["top5"], "mrr": m_O_Q["mrr"],
                "basin_depth_mean": m_O_Q["basin_depth_mean"],
                "top1_ci_95": [ci_O_Q[0], ci_O_Q[2]], "top1_ci_mean": ci_O_Q[1],
            },
            "arm_O_K": {
                "top1": m_O_K["top1"], "top5": m_O_K["top5"], "mrr": m_O_K["mrr"],
                "basin_depth_mean": m_O_K["basin_depth_mean"],
                "top1_ci_95": [ci_O_K[0], ci_O_K[2]], "top1_ci_mean": ci_O_K[1],
            },
            "arm_O_QK": {
                "top1": m_O_QK["top1"], "top5": m_O_QK["top5"], "mrr": m_O_QK["mrr"],
                "basin_depth_mean": m_O_QK["basin_depth_mean"],
                "top1_ci_95": [ci_O_QK[0], ci_O_QK[2]], "top1_ci_mean": ci_O_QK[1],
            },
            "arm_Ortho_QK": {
                "top1": m_Ortho_QK["top1"], "top5": m_Ortho_QK["top5"], "mrr": m_Ortho_QK["mrr"],
                "basin_depth_mean": m_Ortho_QK["basin_depth_mean"],
                "top1_ci_95": [ci_Ortho_QK[0], ci_Ortho_QK[2]], "top1_ci_mean": ci_Ortho_QK[1],
            },
            "arm_R_QK": {
                "top1": m_R_QK["top1"], "top5": m_R_QK["top5"], "mrr": m_R_QK["mrr"],
                "basin_depth_mean": m_R_QK["basin_depth_mean"],
                "top1_ci_95": [ci_R_QK[0], ci_R_QK[2]], "top1_ci_mean": ci_R_QK[1],
            },
            "gates": gates,
            "joint_outcome": joint,
            "runtime_s": float(time.time() - t_start),
            "git_sha": sha,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {out_path}", flush=True)
        print(f"[summary] F={t1_F:.3f}  O_Q={t1_O_Q:.3f}  O_K={t1_O_K:.3f}  "
              f"O_QK={t1_O_QK:.3f}  Ortho={t1_Ortho:.3f}  R={t1_R:.3f}  "
              f"→ {joint}  runtime={out['runtime_s']:.1f}s", flush=True)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] {e}\n{tb}", flush=True)
        partial["error"] = str(e)
        partial["traceback"] = tb
        partial["runtime_s"] = float(time.time() - t_start)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(partial, f, indent=2)
        print(f"[saved partial] {out_path}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
