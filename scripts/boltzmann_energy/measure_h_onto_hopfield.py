#!/usr/bin/env python3
"""H-Onto-Hopfield — Path C: Park unembedding × W_K pullback × K-space Hopfield retrieval.

Spec: memory/handoff_2026_04_19_path_c_onto_hopfield.md §4
Pre-reg gates (LOCKED before run, §5):
  G-OntoHopf-1: top1(O)/top1(F) >= 0.9  AND  CI-lower/top1(F) >= 0.85
  G-OntoHopf-2: top1(ortho)/top1(F) <= 0.6
  G-OntoHopf-3: |top1(R) - top1(ortho)|/top1(F) <= 0.1
  G-Park-Thm8:  ||P_verb P_domain||_F / sqrt(min(r_v, r_d)) < 0.3

Reuses measure_h_energy_wells.py infra:
  - capture_k_pooled()  — K at L=18 prompt_end head-avg
  - MetaTool Subtask1 + plugin_info loaders
  - afod_probe extract_verb / extract_domain

New in this script:
  - Residual-stream extraction at INPUT of layer L (via forward_pre_hook on layers[L])
  - Park concept directions (mean_pos - mean_neg, L2-normalize)
  - W_K pullback per-head + per-head k_norm + head-avg → K-space direction
  - SVD orthonormalization of verb / domain / combined onto subspaces
  - Random orthonormal subspace control (seed=0)
  - 4-arm retrieval: Full / Onto / Ortho / Random
  - Bootstrap 1000x CI for top-1 accuracy

Output: reports/boltzmann_energy/onto_hopfield_L18.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "new_theorem_test"))
sys.path.insert(0, str(HERE))

from afod_probe_metatool_toolbench import extract_verb, extract_domain  # noqa: E402
from measure_h_energy_wells import capture_k_pooled  # noqa: E402

METATOOL_INFO = "/tmp/MetaTool/dataset/plugin_info.json"
METATOOL_ST1 = "/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json"
DEFAULT_OUT = REPO / "reports" / "boltzmann_energy" / "onto_hopfield_L18.json"


# ---------------------------------------------------------------
# Residual stream capture at INPUT of a given layer
# ---------------------------------------------------------------

@torch.no_grad()
def capture_residual_prompt_end(model, tok, text: str, layer: int) -> torch.Tensor | None:
    """Capture the residual stream that will be the INPUT to model.model.layers[layer].

    Uses a forward_pre_hook on layers[layer]; inputs[0] has shape (B, T, d_model).
    Returns the prompt_end (last token) vector as a CPU float tensor of shape (d_model,).
    """
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(next(model.parameters()).device)
    cap: dict = {}

    def _pre_hook(mod, inputs):
        h = inputs[0]  # (B, T, d_model)
        if h.dim() == 3:
            cap["h"] = h[0, -1, :].detach().float().cpu()  # (d_model,)

    h_mod = model.model.layers[layer]
    hh = h_mod.register_forward_pre_hook(_pre_hook)
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        hh.remove()

    return cap.get("h")


# ---------------------------------------------------------------
# W_K pullback with per-head k_norm application
# ---------------------------------------------------------------

@torch.no_grad()
def wk_pullback_with_knorm(u_vec: torch.Tensor, W_K: torch.Tensor,
                           k_norm_module, n_kv: int, head_dim: int) -> torch.Tensor:
    """Pull a residual-space direction u (d_model,) to K-space (head_dim,).

    Steps:
      1. v_all = W_K @ u → (n_kv * head_dim,)
      2. reshape to (n_kv, head_dim)
      3. apply k_norm per-head (RMSNorm is per-last-dim; broadcast over n_kv)
      4. head-average → (head_dim,)

    u_vec: (d_model,) float on same device as k_norm
    W_K:   (n_kv*head_dim, d_model) float  (already on device)
    k_norm_module: RMSNorm normalizing last dim = head_dim
    """
    v_all = W_K @ u_vec  # (n_kv*head_dim,)
    v_heads = v_all.view(n_kv, head_dim)  # (n_kv, head_dim)
    # k_norm operates on last dim = head_dim, works fine on (n_kv, head_dim) batch
    v_heads_normed = k_norm_module(v_heads)
    v_avg = v_heads_normed.mean(dim=0)  # (head_dim,)
    return v_avg


# ---------------------------------------------------------------
# SVD orthonormalization
# ---------------------------------------------------------------

def orthonormal_basis(D: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """SVD-based orthonormal basis of column span of D.

    D: (d, n). Returns U of shape (d, r) with orthonormal columns, r = rank.
    """
    U, s, _ = np.linalg.svd(D, full_matrices=False)
    max_s = s.max() if len(s) > 0 else 0.0
    thresh = tol * max_s if max_s > 0 else tol
    r = int((s > thresh).sum())
    return U[:, :r]


def random_orthonormal_subspace(d: int, r: int, seed: int) -> np.ndarray:
    """Random orthonormal (d, r) matrix via QR on Gaussian."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(size=(d, r))
    Q, _ = np.linalg.qr(G)
    return Q[:, :r]


# ---------------------------------------------------------------
# Retrieval arms
# ---------------------------------------------------------------

def softmax_rows(M: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(axis=1, keepdims=True)


def retrieval_metrics(scores: np.ndarray, gt_idx: np.ndarray) -> Dict[str, float]:
    """scores: (n_q, n_tools), gt_idx: (n_q,).

    Returns dict with top1, top5, mrr, basin_depth_mean, and top1_correct (bool array).
    """
    n_q = scores.shape[0]
    ranked = np.argsort(-scores, axis=1)  # descending
    top1_correct = (ranked[:, 0] == gt_idx).astype(np.int64)
    top1 = float(top1_correct.mean())
    top5_correct = np.any(ranked[:, :5] == gt_idx[:, None], axis=1).astype(np.int64)
    top5 = float(top5_correct.mean())

    # MRR
    rr = np.zeros(n_q, dtype=np.float64)
    for q in range(n_q):
        pos = int(np.where(ranked[q] == gt_idx[q])[0][0])
        rr[q] = 1.0 / (pos + 1)
    mrr = float(rr.mean())

    # Basin depth = score_GT - score_second_best
    score_gt = scores[np.arange(n_q), gt_idx]
    # second-best = max excluding GT
    tmp = scores.copy()
    tmp[np.arange(n_q), gt_idx] = -np.inf
    score_second = tmp.max(axis=1)
    basin_depth = float((score_gt - score_second).mean())

    return {
        "top1": top1,
        "top5": top5,
        "mrr": mrr,
        "basin_depth_mean": basin_depth,
        "_top1_correct": top1_correct,
    }


def bootstrap_top1_ci(top1_correct: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    """Return (lo_2.5, mean, hi_97.5) of top1 accuracy via bootstrap over queries."""
    rng = np.random.default_rng(seed)
    n = len(top1_correct)
    boots = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = top1_correct[idx].mean()
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    mean = float(boots.mean())
    return lo, mean, hi


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

    # Partial state we can dump on failure
    partial = {
        "variant": "onto_hopfield_L18",
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

        # --- Load catalog + queries (same filter as H-Wells) ---
        print(f"[data] loading plugin_info + Subtask1 …", flush=True)
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
        partial["n_tools"] = n_tools
        partial["n_verb_directions"] = n_verb
        partial["n_domain_directions"] = n_domain

        pool = [e for e in st1 if e.get("tool") in plugins]
        print(f"[queries pool] {len(pool)} / {len(st1)} have GT ∈ plugin catalog", flush=True)
        assert len(pool) >= args.n_queries, f"need {args.n_queries} queries, have {len(pool)}"

        idx_perm = rng.permutation(len(pool))[:args.n_queries]
        queries = [pool[i] for i in idx_perm]
        n_q = len(queries)
        partial["n_queries"] = n_q
        print(f"[queries] using N={n_q}", flush=True)

        # --- Load model ---
        print(f"[model] loading {args.model} fp16 on {args.device}", flush=True)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16,
        ).to(args.device)
        model.eval()
        cfg = model.config
        n_q_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        d_model = cfg.hidden_size
        head_dim = d_model // n_q_heads
        L_total = len(model.model.layers)
        print(f"[config] L_total={L_total}  n_q={n_q_heads}  n_kv={n_kv}  "
              f"d_model={d_model}  d_head={head_dim}  layer={args.layer}", flush=True)
        assert 0 <= args.layer < L_total

        layer_mod = model.model.layers[args.layer]
        attn = layer_mod.self_attn
        # k_proj.weight shape = (n_kv*head_dim, d_model)
        W_K = attn.k_proj.weight.detach().float()  # keep on device, float32 for stable SVD
        print(f"[W_K] shape={tuple(W_K.shape)}  device={W_K.device}", flush=True)
        assert W_K.shape == (n_kv * head_dim, d_model), \
            f"unexpected W_K shape {W_K.shape} vs ({n_kv*head_dim}, {d_model})"
        k_norm_module = attn.k_norm if hasattr(attn, "k_norm") else None
        if k_norm_module is None:
            print("[WARN] model has no k_norm; pullback will skip normalization", flush=True)

        # --- Step 1: extract residual-stream directions for each label ---
        # For each verb category, mean over tools with verb==c minus mean over tools with verb!=c
        # residual extracted at INPUT of layer L (the state layer-L attention will process).
        print(f"[Step 1] extracting residual streams for {n_tools} tools at L={args.layer} input", flush=True)
        H = np.zeros((n_tools, d_model), dtype=np.float32)
        t0 = time.time()
        for i, name in enumerate(plugin_names):
            h = capture_residual_prompt_end(model, tok, plugins[name]["desc"], args.layer)
            if h is None:
                raise RuntimeError(f"residual capture failed for tool {name}")
            H[i] = h.numpy()
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_tools - i - 1)
                print(f"    [resid {i+1}/{n_tools}] t={elapsed:.1f}s eta={eta:.1f}s", flush=True)
        print(f"[Step 1] done, H_norm_mean={np.linalg.norm(H, axis=1).mean():.3f}", flush=True)

        # Compute Park directions: mean(pos) - mean(neg), L2-normalize
        print(f"[Step 1] computing Park directions: {n_verb} verb + {n_domain} domain", flush=True)
        verb_arr = np.array(verb_labels)
        domain_arr = np.array(domain_labels)

        def compute_directions(arr: np.ndarray, uniq: List[str]) -> np.ndarray:
            """Returns (n_cat, d_model) L2-normalized direction matrix."""
            dirs = np.zeros((len(uniq), d_model), dtype=np.float32)
            for ci, c in enumerate(uniq):
                pos_mask = (arr == c)
                neg_mask = ~pos_mask
                if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                    # degenerate: leave as zeros (will be filtered by SVD thresholding)
                    continue
                mu_pos = H[pos_mask].mean(axis=0)
                mu_neg = H[neg_mask].mean(axis=0)
                d = mu_pos - mu_neg
                nrm = np.linalg.norm(d)
                if nrm > 1e-8:
                    dirs[ci] = d / nrm
            return dirs

        U_verb_resid = compute_directions(verb_arr, verb_uniq)   # (n_verb, d_model)
        U_dom_resid = compute_directions(domain_arr, domain_uniq)  # (n_domain, d_model)
        print(f"[Step 1] verb_dirs norm mean={np.linalg.norm(U_verb_resid, axis=1).mean():.4f}  "
              f"domain_dirs norm mean={np.linalg.norm(U_dom_resid, axis=1).mean():.4f}", flush=True)

        # --- Step 2: W_K pullback (per-head k_norm, head-avg) ---
        print(f"[Step 2] W_K pullback with per-head k_norm to K-space", flush=True)
        device = next(model.parameters()).device

        def pull_all(U_res: np.ndarray) -> np.ndarray:
            """(n_cat, d_model) residual-space directions → (n_cat, head_dim) K-space."""
            out = np.zeros((U_res.shape[0], head_dim), dtype=np.float32)
            for ci in range(U_res.shape[0]):
                u_dev = torch.from_numpy(U_res[ci]).to(device).to(W_K.dtype)
                if k_norm_module is not None:
                    # k_norm stays on fp16 by default; cast module call path
                    v_all = W_K @ u_dev
                    v_heads = v_all.view(n_kv, head_dim)
                    v_heads_normed = k_norm_module(v_heads.to(torch.float16)).float()
                    v = v_heads_normed.mean(dim=0)
                else:
                    v_all = W_K @ u_dev
                    v = v_all.view(n_kv, head_dim).mean(dim=0)
                out[ci] = v.detach().cpu().numpy()
            return out

        V_verb_K = pull_all(U_verb_resid)   # (n_verb, head_dim)
        V_dom_K = pull_all(U_dom_resid)      # (n_domain, head_dim)
        print(f"[Step 2] V_verb_K shape={V_verb_K.shape} norm_mean={np.linalg.norm(V_verb_K, axis=1).mean():.4f}", flush=True)
        print(f"[Step 2] V_dom_K  shape={V_dom_K.shape} norm_mean={np.linalg.norm(V_dom_K, axis=1).mean():.4f}", flush=True)

        # --- Step 3: SVD orthonormalization ---
        # Stack as columns: (head_dim, n_cat) then SVD → U basis (head_dim, r)
        print("[Step 3] SVD orthonormalization of verb / domain / combined subspaces", flush=True)
        D_verb = V_verb_K.T  # (head_dim, n_verb)
        D_dom = V_dom_K.T    # (head_dim, n_domain)
        U_verb = orthonormal_basis(D_verb)  # (head_dim, r_v)
        U_dom = orthonormal_basis(D_dom)    # (head_dim, r_d)
        r_v = U_verb.shape[1]
        r_d = U_dom.shape[1]
        # Combined = SVD of [U_verb | U_dom]
        D_combined = np.concatenate([U_verb, U_dom], axis=1)  # (head_dim, r_v+r_d)
        U_onto = orthonormal_basis(D_combined)  # (head_dim, r_onto)
        r_onto = U_onto.shape[1]
        print(f"[Step 3] r_verb={r_v}  r_domain={r_d}  r_onto={r_onto}  (max={head_dim})", flush=True)

        # --- Step 4: projection operators ---
        P_onto = U_onto @ U_onto.T  # (head_dim, head_dim)
        P_perp = np.eye(head_dim, dtype=np.float32) - P_onto
        # random subspace of same dim as onto
        U_rand = random_orthonormal_subspace(head_dim, r_onto, args.seed).astype(np.float32)
        P_rand = U_rand @ U_rand.T
        print(f"[Step 4] projectors built (dim=onto→{r_onto}, perp→{head_dim - r_onto}, rand→{r_onto})", flush=True)

        # --- Step 5: Park Thm 8 structural check on K-space subspaces ---
        P_verb = U_verb @ U_verb.T
        P_dom = U_dom @ U_dom.T
        cross = P_verb @ P_dom
        cross_fro = float(np.linalg.norm(cross, ord="fro"))
        normalizer = float(np.sqrt(min(r_v, r_d))) if min(r_v, r_d) > 0 else 1.0
        verb_dom_fro = cross_fro / normalizer
        g_park_pass = bool(verb_dom_fro < 0.3)
        print(f"[Step 5] ||P_verb P_dom||_F = {cross_fro:.4f}  "
              f"/ sqrt(min(r_v,r_d)) = {normalizer:.4f}  "
              f"→ {verb_dom_fro:.4f}  pass={g_park_pass}", flush=True)

        # --- Step 6: tool K-vectors at L=18 prompt_end head-avg (reuse capture_k_pooled) ---
        print(f"[Step 6a] extracting tool K-vectors (n={n_tools}) at L={args.layer} prompt_end head-avg", flush=True)
        K_tools = np.zeros((n_tools, head_dim), dtype=np.float32)
        t0 = time.time()
        for i, name in enumerate(plugin_names):
            v = capture_k_pooled(model, tok, plugins[name]["desc"], args.layer,
                                  n_kv, head_dim, pooling=args.pooling,
                                  prepend_name=None, chat_template=False)
            if v is None:
                raise RuntimeError(f"K capture failed for tool {name}")
            K_tools[i] = v.numpy().astype(np.float32)
            if (i + 1) % 100 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_tools - i - 1)
                print(f"    [K {i+1}/{n_tools}] t={elapsed:.1f}s eta={eta:.1f}s", flush=True)

        # --- Query K-vectors ---
        print(f"[Step 6b] extracting query K-vectors (N={n_q})", flush=True)
        Q_queries = np.zeros((n_q, head_dim), dtype=np.float32)
        gt_idx = np.zeros(n_q, dtype=np.int64)
        t0 = time.time()
        for qi, entry in enumerate(queries):
            v = capture_k_pooled(model, tok, entry["query"], args.layer, n_kv, head_dim,
                                  pooling=args.pooling, prepend_name=None, chat_template=False)
            if v is None:
                raise RuntimeError(f"query K capture failed at qi={qi}")
            Q_queries[qi] = v.numpy().astype(np.float32)
            gt_idx[qi] = plugin_idx[entry["tool"]]
            if (qi + 1) % 50 == 0 or qi == 0:
                elapsed = time.time() - t0
                eta = elapsed / (qi + 1) * (n_q - qi - 1)
                print(f"    [Q {qi+1}/{n_q}] t={elapsed:.1f}s eta={eta:.1f}s", flush=True)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        # --- Step 7: 4-arm retrieval ---
        print("[Step 7] running 4-arm retrieval: F / O / ortho / R", flush=True)
        beta = 1.0 / np.sqrt(head_dim)

        def compute_scores(Q: np.ndarray, K: np.ndarray, P: np.ndarray | None) -> np.ndarray:
            """Softmax over tools of β (PQ)·(PK). If P is None, Full arm (no projection)."""
            if P is None:
                Qp = Q
                Kp = K
            else:
                Qp = Q @ P.T  # P is symmetric but be explicit
                Kp = K @ P.T
            logits = beta * (Qp @ Kp.T)  # (n_q, n_tools)
            return softmax_rows(logits)

        scores_F = compute_scores(Q_queries, K_tools, None)
        scores_O = compute_scores(Q_queries, K_tools, P_onto)
        scores_perp = compute_scores(Q_queries, K_tools, P_perp)
        scores_R = compute_scores(Q_queries, K_tools, P_rand)

        m_F = retrieval_metrics(scores_F, gt_idx)
        m_O = retrieval_metrics(scores_O, gt_idx)
        m_perp = retrieval_metrics(scores_perp, gt_idx)
        m_R = retrieval_metrics(scores_R, gt_idx)

        # Bootstrap CIs
        print(f"[Step 7b] bootstrap CI (n={args.n_bootstrap})", flush=True)
        ci_F = bootstrap_top1_ci(m_F["_top1_correct"], args.n_bootstrap, args.seed)
        ci_O = bootstrap_top1_ci(m_O["_top1_correct"], args.n_bootstrap, args.seed + 1)
        ci_perp = bootstrap_top1_ci(m_perp["_top1_correct"], args.n_bootstrap, args.seed + 2)
        ci_R = bootstrap_top1_ci(m_R["_top1_correct"], args.n_bootstrap, args.seed + 3)

        print(f"[arm F]    top1={m_F['top1']:.4f}  top5={m_F['top5']:.4f}  mrr={m_F['mrr']:.4f}  "
              f"basin={m_F['basin_depth_mean']:.4f}  CI95=[{ci_F[0]:.4f}, {ci_F[2]:.4f}]", flush=True)
        print(f"[arm O]    top1={m_O['top1']:.4f}  top5={m_O['top5']:.4f}  mrr={m_O['mrr']:.4f}  "
              f"basin={m_O['basin_depth_mean']:.4f}  CI95=[{ci_O[0]:.4f}, {ci_O[2]:.4f}]", flush=True)
        print(f"[arm ⊥]    top1={m_perp['top1']:.4f}  top5={m_perp['top5']:.4f}  mrr={m_perp['mrr']:.4f}  "
              f"basin={m_perp['basin_depth_mean']:.4f}  CI95=[{ci_perp[0]:.4f}, {ci_perp[2]:.4f}]", flush=True)
        print(f"[arm R]    top1={m_R['top1']:.4f}  top5={m_R['top5']:.4f}  mrr={m_R['mrr']:.4f}  "
              f"basin={m_R['basin_depth_mean']:.4f}  CI95=[{ci_R[0]:.4f}, {ci_R[2]:.4f}]", flush=True)

        # --- Step 8: pre-reg gates ---
        t1_F = m_F["top1"]
        t1_O = m_O["top1"]
        t1_perp = m_perp["top1"]
        t1_R = m_R["top1"]

        eps = 1e-9
        if t1_F < eps:
            # baseline weak → special flag
            ohopf1 = False
            ohopf2 = False
            ohopf3 = False
        else:
            # G-OntoHopf-1: top1(O)/top1(F) >= 0.9 AND CI-lower(O)/top1(F) >= 0.85
            ohopf1 = bool((t1_O / t1_F >= 0.9) and (ci_O[0] / t1_F >= 0.85))
            # G-OntoHopf-2: top1(⊥)/top1(F) <= 0.6
            ohopf2 = bool((t1_perp / t1_F) <= 0.6)
            # G-OntoHopf-3: |top1(R) - top1(⊥)|/top1(F) <= 0.1
            ohopf3 = bool(abs(t1_R - t1_perp) / t1_F <= 0.1)

        gates = {
            "G_OntoHopf_1_pass": ohopf1,
            "G_OntoHopf_2_pass": ohopf2,
            "G_OntoHopf_3_pass": ohopf3,
            "G_Park_Thm8_pass": g_park_pass,
        }
        print(f"[gates] G1={ohopf1}  G2={ohopf2}  G3={ohopf3}  G-Park8={g_park_pass}", flush=True)

        # --- Joint outcome (§5 decision matrix) ---
        if t1_F < 0.3:
            joint = "BASELINE_WEAK"
        elif ohopf1 and ohopf2 and ohopf3 and g_park_pass:
            joint = "STRONG_POSITIVE"
        elif ohopf1 and ohopf2 and ohopf3 and not g_park_pass:
            joint = "POSITIVE_WEAK_HIERARCHY"
        elif ohopf1 and not ohopf2:
            joint = "ONTOLOGY_NON_SPECIFIC"
        elif (not ohopf1) and ohopf2 and ohopf3:
            joint = "ONTOLOGY_DETRIMENTAL"
        elif (not ohopf1) and (not ohopf2):
            joint = "PROJECTION_DEAD"
        elif t1_perp > t1_F:
            joint = "BIZARRE_ORTHO_BETTER"
        else:
            joint = "PARTIAL"

        print(f"[joint] → {joint}", flush=True)

        # --- Git sha ---
        try:
            import subprocess
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
        except Exception:
            sha = "unknown"

        # --- Output ---
        out = {
            "variant": "onto_hopfield_L18",
            "model": args.model,
            "dataset": "MetaTool-Subtask1",
            "n_queries": n_q,
            "n_tools": n_tools,
            "layer": args.layer,
            "pooling": args.pooling,
            "direction_source": "residual-L18-prompt-end",
            "n_verb_directions": n_verb,
            "n_domain_directions": n_domain,
            "r_verb": r_v,
            "r_domain": r_d,
            "ontology_subspace_dim_after_orth": r_onto,
            "d_head": head_dim,
            "d_model": d_model,
            "n_kv": n_kv,
            "beta": float(beta),
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "park_thm8_check": {
                "verb_dom_cross_fro": cross_fro,
                "normalizer_sqrt_min_r": normalizer,
                "verb_dom_projector_norm": verb_dom_fro,
                "pass": g_park_pass,
            },
            "arm_F": {
                "top1": m_F["top1"], "top5": m_F["top5"], "mrr": m_F["mrr"],
                "basin_depth_mean": m_F["basin_depth_mean"],
                "top1_ci_95": [ci_F[0], ci_F[2]],
                "top1_ci_mean": ci_F[1],
            },
            "arm_O": {
                "top1": m_O["top1"], "top5": m_O["top5"], "mrr": m_O["mrr"],
                "basin_depth_mean": m_O["basin_depth_mean"],
                "top1_ci_95": [ci_O[0], ci_O[2]],
                "top1_ci_mean": ci_O[1],
            },
            "arm_ortho": {
                "top1": m_perp["top1"], "top5": m_perp["top5"], "mrr": m_perp["mrr"],
                "basin_depth_mean": m_perp["basin_depth_mean"],
                "top1_ci_95": [ci_perp[0], ci_perp[2]],
                "top1_ci_mean": ci_perp[1],
            },
            "arm_R": {
                "top1": m_R["top1"], "top5": m_R["top5"], "mrr": m_R["mrr"],
                "basin_depth_mean": m_R["basin_depth_mean"],
                "top1_ci_95": [ci_R[0], ci_R[2]],
                "top1_ci_mean": ci_R[1],
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
        print(f"[summary] F={t1_F:.3f}  O={t1_O:.3f}  ⊥={t1_perp:.3f}  R={t1_R:.3f}  "
              f"Park8={verb_dom_fro:.3f}  → {joint}  "
              f"runtime={out['runtime_s']:.1f}s", flush=True)

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
