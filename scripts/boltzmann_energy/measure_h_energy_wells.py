#!/usr/bin/env python3
"""H-Energy-Wells — Boltzmann Energy Framework v1 first experiment.

Spec: math/paper/iclr2027/BOLTZMANN_ENERGY_FRAMEWORK_v1.md §4.1 + §5
Brainstorm handoff: handoff_2026_04_19_b2_brainstorm_d_centroid_layer.md
Experiment handoff: handoff_2026_04_19_b2_experiment_h_energy_wells.md

Decisions (D-Centroid=A prompt-end, D-Layer=A L=27 single, D-Cluster-Source=A F8d reuse):
  - Model: Qwen2.5-7B-Instruct, fp16
  - Layer: L=27 (last attention), single
  - K capture: attn.k_norm (or k_proj) last-token, averaged over n_kv=4 heads → d_head=128 vec
  - Clusters: verb × domain via afod_probe_metatool_toolbench extract_verb/extract_domain
  - Dataset: MetaTool Subtask1 N=100 queries, filtered to GT ∈ name_for_model (deviation
    from spec §5.1 "Subtask4" — Subtask4 GT is meta-category not individual plugin, so
    facet distance ill-defined. Subtask1 is the natural fit for H-Energy-Wells.)
  - Seed: 0 fixed for query subsampling + shuffle control

Pre-reg gates (LOCKED before measurement):
  G-Wells-1: Ē(0) < Ē(1) < Ē(2) strict, Ē(2)-Ē(0) ≥ 0.5·std(E)
  G-Wells-2: median per-query Spearman ρ(d_total, E) ≥ 0.4
  G-Wells-3: same under shuffled labels, median ρ < 0.1

H-V-Form fits (regression on ΔE = E[q,j] − E[q,t*(q)] across all (q,j) pairs):
  Additive: ΔE ≈ β₀ + β_v·δ_v + β_d·δ_d
  Hopfield: ΔE ≈ β₀ + β_H·Φ_H(q,t) with Φ_H(q,t) = Σ_{f,μ} <q,ξ^{f,μ}><ξ^{f,μ},k_t>
  Sparse:   ΔE ≈ β₀ + β·δ_v  (verb only)  or  ΔE ≈ β₀ + β·δ_d  (domain only), pick best

Output: reports/boltzmann_energy/h_wells_qwen25_metatool_n100.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "new_theorem_test"))

from afod_probe_metatool_toolbench import extract_verb, extract_domain  # noqa

METATOOL_INFO = "/tmp/MetaTool/dataset/plugin_info.json"
METATOOL_ST1 = "/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json"
DEFAULT_OUT = REPO / "reports" / "boltzmann_energy" / "h_wells_qwen25_metatool_n100.json"


# ---------------------------------------------------------------
# 3.1 / 3.2 — K-vector capture helpers
# ---------------------------------------------------------------

@torch.no_grad()
def capture_last_token_k_single_layer(model, tok, text: str, layer: int,
                                       n_kv: int, head_dim: int) -> torch.Tensor | None:
    """Capture K at given layer, last token, averaged over n_kv heads → (head_dim,).

    Uses k_norm hook (Qwen-style) if present, else k_proj.
    """
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(next(model.parameters()).device)
    cap: dict = {}

    def _hook(mod, inputs, output):
        out = output
        if out.dim() == 3:
            B, T, D = out.shape
            if D != n_kv * head_dim:
                return
            out_view = out.view(B, T, n_kv, head_dim)
        elif out.dim() == 4:
            out_view = out
        else:
            return
        cap["k"] = out_view[0, -1, :, :].detach().float().cpu()  # (n_kv, head_dim)

    attn = model.model.layers[layer].self_attn
    module = attn.k_norm if hasattr(attn, "k_norm") else attn.k_proj
    h = module.register_forward_hook(_hook)
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        h.remove()

    if "k" not in cap:
        return None
    # Average across n_kv heads → (head_dim,)
    return cap["k"].mean(dim=0)


# ---------------------------------------------------------------
# 3.3 — Centroid computation per facet cluster
# ---------------------------------------------------------------

def compute_cluster_centroids(K: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """K:(N,d), labels len N → (n_clusters, d) centroid matrix + label-list order."""
    uniq = sorted(set(labels))
    lbl2idx = {l: i for i, l in enumerate(uniq)}
    centroids = np.zeros((len(uniq), K.shape[1]), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.int64)
    for k_vec, l in zip(K, labels):
        i = lbl2idx[l]
        centroids[i] += k_vec
        counts[i] += 1
    centroids /= np.maximum(counts[:, None], 1)
    return centroids, uniq


# ---------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------

def _r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _aic(y: np.ndarray, y_pred: np.ndarray, k_params: int) -> float:
    """Gaussian AIC. k_params includes intercept + slopes + implicit σ."""
    n = len(y)
    rss = float(((y - y_pred) ** 2).sum())
    if rss < 1e-12:
        rss = 1e-12
    # AIC = 2k + n*log(rss/n)
    return 2.0 * k_params + n * float(np.log(rss / n))


def fit_linear(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Least squares; X has explicit intercept column. Returns coefs, y_pred, R², AIC."""
    coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coefs
    r2 = _r2(y, y_pred)
    aic = _aic(y, y_pred, X.shape[1] + 1)  # +1 for σ
    return coefs, y_pred, r2, aic


# ---------------------------------------------------------------
# 3.4 — V-form regressions
# ---------------------------------------------------------------

def fit_v_hypotheses(dE: np.ndarray, dv: np.ndarray, dd: np.ndarray,
                     phi_hop: np.ndarray) -> Dict:
    """Fit Additive, Hopfield, Sparse on the same ΔE, return R² + AIC + coefs."""
    n = len(dE)
    ones = np.ones(n)

    X_add = np.stack([ones, dv.astype(np.float64), dd.astype(np.float64)], axis=1)
    c_add, _, r2_add, aic_add = fit_linear(X_add, dE)

    X_hop = np.stack([ones, phi_hop], axis=1)
    c_hop, _, r2_hop, aic_hop = fit_linear(X_hop, dE)

    X_sv = np.stack([ones, dv.astype(np.float64)], axis=1)
    _, _, r2_sv, aic_sv = fit_linear(X_sv, dE)
    X_sd = np.stack([ones, dd.astype(np.float64)], axis=1)
    _, _, r2_sd, aic_sd = fit_linear(X_sd, dE)

    return {
        "additive": {
            "R2": float(r2_add),
            "AIC": float(aic_add),
            "intercept": float(c_add[0]),
            "a_v": float(c_add[1]),
            "a_d": float(c_add[2]),
        },
        "hopfield": {
            "R2": float(r2_hop),
            "AIC": float(aic_hop),
            "intercept": float(c_hop[0]),
            "beta_H": float(c_hop[1]),
        },
        "sparse_verb": {
            "R2": float(r2_sv),
            "AIC": float(aic_sv),
        },
        "sparse_domain": {
            "R2": float(r2_sd),
            "AIC": float(aic_sd),
        },
    }


def compute_hopfield_feature(Q: np.ndarray, K: np.ndarray,
                              Xi_verb: np.ndarray, Xi_domain: np.ndarray) -> np.ndarray:
    """Φ_H[q, t] = Σ_{f,μ} <q, ξ^{f,μ}> <ξ^{f,μ}, k_t>.

    Q:(Nq, d), K:(Nt, d), Xi_*:(Mf, d) → Φ:(Nq, Nt)
    """
    # Per facet: Φ_f = (Q @ Xi^T) @ (Xi @ K^T) = Q @ (Xi^T @ Xi) @ K^T
    phi = np.zeros((Q.shape[0], K.shape[0]), dtype=np.float64)
    for Xi in (Xi_verb, Xi_domain):
        # (Nq, Mf) @ (Mf, Nt) via (Q @ Xi.T) @ (Xi @ K.T)
        qx = Q @ Xi.T  # (Nq, Mf)
        xk = Xi @ K.T  # (Mf, Nt)
        phi += qx @ xk
    return phi


# ---------------------------------------------------------------
# Spearman helper (per-row, handles ties)
# ---------------------------------------------------------------

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation between two 1-D arrays. Uses average-rank tie-handling."""
    def _ranks(a):
        # average ranks for ties
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a))
        # resolve ties
        a_sorted = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
                j += 1
            if j > i:
                avg = 0.5 * (i + j)
                ranks[order[i:j + 1]] = avg
            i = j + 1
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((rx * ry).sum() / denom)


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer", type=int, default=27, help="D-Layer=A default (last attn)")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- Load catalog + queries ---
    print(f"[data] loading plugin_info + Subtask1 …")
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
    print(f"[catalog] {n_tools} plugins with non-empty desc")

    verb_labels = [plugins[n]["verb"] for n in plugin_names]
    domain_labels = [plugins[n]["domain"] for n in plugin_names]
    n_verb = len(set(verb_labels))
    n_domain = len(set(domain_labels))
    print(f"[clusters] verb={n_verb}, domain={n_domain}")

    # Filter ST1 to queries whose GT is an individual plugin (name_for_model)
    pool = [e for e in st1 if e.get("tool") in plugins]
    print(f"[queries pool] {len(pool)} / {len(st1)} have GT ∈ plugin catalog")
    assert len(pool) >= args.n_queries, f"not enough queries ({len(pool)} < {args.n_queries})"

    # Stable subsample
    idx_perm = rng.permutation(len(pool))[:args.n_queries]
    queries = [pool[i] for i in idx_perm]
    n_q = len(queries)
    print(f"[queries] using N={n_q}")

    # --- Load model ---
    print(f"[model] loading {args.model} fp16 on {args.device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).to(args.device)
    model.eval()
    cfg = model.config
    n_q_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_q_heads
    L_total = len(model.model.layers)
    assert 0 <= args.layer < L_total, f"layer {args.layer} out of range [0,{L_total})"
    print(f"[config] L_total={L_total} n_q={n_q_heads} n_kv={n_kv} d_head={head_dim} layer={args.layer}")

    # --- T1.a: catalog K extraction ---
    print(f"[T1.a] extracting K[{n_tools}, {head_dim}] at L={args.layer} (prompt-end pooling, mean over kv heads)")
    K = np.zeros((n_tools, head_dim), dtype=np.float64)
    t0 = time.time()
    for i, name in enumerate(plugin_names):
        v = capture_last_token_k_single_layer(model, tok, plugins[name]["desc"],
                                               args.layer, n_kv, head_dim)
        if v is None:
            print(f"  [WARN] capture failed for {name}")
            continue
        K[i] = v.numpy().astype(np.float64)
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_tools - i - 1)
            print(f"    [{i+1}/{n_tools}] t={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    print(f"[T1.a] done, K_norm_mean={np.linalg.norm(K, axis=1).mean():.3f}")

    # --- T1.b: query K extraction ---
    print(f"[T1.b] extracting Q[{n_q}, {head_dim}]")
    Q = np.zeros((n_q, head_dim), dtype=np.float64)
    gt_plugin_idx = np.zeros(n_q, dtype=np.int64)
    t0 = time.time()
    for qi, entry in enumerate(queries):
        # Query text: raw user query (no chat template — same convention as tool desc)
        qtext = entry["query"]
        v = capture_last_token_k_single_layer(model, tok, qtext,
                                               args.layer, n_kv, head_dim)
        if v is None:
            raise RuntimeError(f"query capture failed at qi={qi}")
        Q[qi] = v.numpy().astype(np.float64)
        gt_plugin_idx[qi] = plugin_idx[entry["tool"]]
        if (qi + 1) % 20 == 0 or qi == 0:
            elapsed = time.time() - t0
            eta = elapsed / (qi + 1) * (n_q - qi - 1)
            print(f"    [{qi+1}/{n_q}] t={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    print(f"[T1.b] done")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # --- Compute energy E[q, t] = -<q, k_t> ---
    E = -(Q @ K.T)  # (n_q, n_tools)
    E_std = float(E.std())
    print(f"[energy] E shape={E.shape}, E_std={E_std:.4f}")

    # --- T2: cluster distance ---
    print("[T2] computing per-query facet distances to GT …")
    verb_arr = np.array(verb_labels)
    domain_arr = np.array(domain_labels)

    def compute_deltas(v_lab, d_lab):
        gt_v = v_lab[gt_plugin_idx]  # (n_q,)
        gt_d = d_lab[gt_plugin_idx]
        dv = (v_lab[None, :] != gt_v[:, None]).astype(np.int64)  # (n_q, n_tools)
        dd = (d_lab[None, :] != gt_d[:, None]).astype(np.int64)
        dt = dv + dd
        return dv, dd, dt

    dv, dd, dt = compute_deltas(verb_arr, domain_arr)

    # --- G-Wells-1 aggregated wells ---
    wells = {}
    for d in (0, 1, 2):
        mask = dt == d
        if mask.sum() == 0:
            wells[d] = float("nan")
        else:
            wells[d] = float(E[mask].mean())
    g1_strict = wells[0] < wells[1] < wells[2]
    if not np.isnan(wells[0]) and not np.isnan(wells[2]):
        delta_norm = (wells[2] - wells[0]) / (E_std + 1e-12)
    else:
        delta_norm = float("nan")
    g1_pass = bool(g1_strict and delta_norm >= 0.5)
    print(f"[G-Wells-1] Ē(0)={wells[0]:.4f} Ē(1)={wells[1]:.4f} Ē(2)={wells[2]:.4f}  Δ_norm={delta_norm:.3f}  pass={g1_pass}")

    # --- G-Wells-2 per-query Spearman ---
    rhos = np.array([spearman(dt[q].astype(np.float64), E[q]) for q in range(n_q)])
    median_rho = float(np.median(rhos))
    g2_pass = bool(median_rho >= 0.4)
    print(f"[G-Wells-2] median ρ={median_rho:.3f}  pass={g2_pass}")

    # --- T4 / G-Wells-3: shuffled labels control ---
    print("[T4] random-cluster control (shuffle labels, seed=0)")
    rng2 = np.random.default_rng(args.seed)
    verb_shuf = verb_arr[rng2.permutation(n_tools)]
    domain_shuf = domain_arr[rng2.permutation(n_tools)]
    dv_s, dd_s, dt_s = compute_deltas(verb_shuf, domain_shuf)
    rhos_s = np.array([spearman(dt_s[q].astype(np.float64), E[q]) for q in range(n_q)])
    median_rho_s = float(np.median(rhos_s))
    g3_pass = bool(median_rho_s < 0.1)
    print(f"[G-Wells-3] median ρ_shuf={median_rho_s:.3f}  pass={g3_pass}")

    # Wells on shuffled
    wells_s = {}
    for d in (0, 1, 2):
        mask = dt_s == d
        wells_s[d] = float(E[mask].mean()) if mask.sum() > 0 else float("nan")

    # --- T3: H-V-Form regressions on ΔE ---
    print("[T3] fitting V-forms …")
    E_gt_per_query = E[np.arange(n_q), gt_plugin_idx]  # (n_q,)
    dE = E - E_gt_per_query[:, None]  # (n_q, n_tools)
    dE_flat = dE.flatten()
    dv_flat = dv.flatten()
    dd_flat = dd.flatten()

    # Hopfield feature
    Xi_verb, verb_uniq = compute_cluster_centroids(K, verb_labels)
    Xi_domain, domain_uniq = compute_cluster_centroids(K, domain_labels)
    phi_hop = compute_hopfield_feature(Q, K, Xi_verb, Xi_domain)  # (n_q, n_tools)
    phi_flat = phi_hop.flatten()

    v_fits = fit_v_hypotheses(dE_flat, dv_flat, dd_flat, phi_flat)
    # Also fit on shuffled for reporting
    dv_s_flat = dv_s.flatten()
    dd_s_flat = dd_s.flatten()
    Xi_verb_s, _ = compute_cluster_centroids(K, list(verb_shuf))
    Xi_domain_s, _ = compute_cluster_centroids(K, list(domain_shuf))
    phi_hop_s = compute_hopfield_feature(Q, K, Xi_verb_s, Xi_domain_s)
    v_fits_shuf = fit_v_hypotheses(dE_flat, dv_s_flat, dd_s_flat, phi_hop_s.flatten())

    print(f"[T3] Additive R²={v_fits['additive']['R2']:.3f}  Hopfield R²={v_fits['hopfield']['R2']:.3f}  "
          f"Sparse(v)={v_fits['sparse_verb']['R2']:.3f}  Sparse(d)={v_fits['sparse_domain']['R2']:.3f}")

    # --- Joint outcome classification ---
    add_r2 = v_fits["additive"]["R2"]
    hop_r2 = v_fits["hopfield"]["R2"]
    sv_r2 = v_fits["sparse_verb"]["R2"]
    sd_r2 = v_fits["sparse_domain"]["R2"]
    sparse_r2 = max(sv_r2, sd_r2)

    gates_all_pass = g1_pass and g2_pass and g3_pass
    g1g2_pass_g3_fail = g1_pass and g2_pass and not g3_pass

    if gates_all_pass and hop_r2 >= 0.5 and (hop_r2 - add_r2) >= 0.1:
        joint = "STRONG_HOPFIELD"
    elif gates_all_pass and add_r2 >= 0.5 and hop_r2 < add_r2 and sparse_r2 < 0.7 * add_r2:
        joint = "STRONG_ADDITIVE"
    elif gates_all_pass and add_r2 >= 0.5 and sparse_r2 >= 0.7 * add_r2:
        joint = "MIXED"
    elif g1g2_pass_g3_fail:
        joint = "RANDOM_CONTAMINATION"
    elif g1_pass and not g2_pass:
        joint = "WEAK"
    elif not g1_pass and not g2_pass:
        joint = "FALSIFIED"
    else:
        joint = "PARTIAL"
    print(f"[joint] outcome={joint}")

    # --- Git SHA ---
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()
    except Exception:
        sha = "unknown"

    # --- Output ---
    out = {
        "model": args.model,
        "dataset": "MetaTool-Subtask1",  # deviation from spec §5.1 "Subtask4" (see script header)
        "dataset_note": "Subtask1 used instead of Subtask4; Subtask4 GT = meta-category (15 unique), "
                        "not individual plugin. H-Energy-Wells requires per-query GT plugin for "
                        "facet-distance. Subtask1 filtered to GT ∈ name_for_model: 765/995 eligible.",
        "n_queries": n_q,
        "n_tools": n_tools,
        "layer": args.layer,
        "pooling": "prompt_end_k_norm_avg_over_n_kv",
        "facets": ["verb", "domain"],
        "n_clusters_verb": n_verb,
        "n_clusters_domain": n_domain,
        "seed": args.seed,
        "E_std": E_std,
        "wells_aggregated": {
            "E_mean_per_distance": {"0": wells[0], "1": wells[1], "2": wells[2]},
            "G_Wells_1_pass": g1_pass,
            "G_Wells_1_strict_monotone": bool(g1_strict),
            "delta_E_normalized": float(delta_norm),
        },
        "wells_per_query": {
            "spearman_rhos": [float(r) for r in rhos],
            "median_rho": median_rho,
            "G_Wells_2_pass": g2_pass,
        },
        "random_control": {
            "spearman_rhos": [float(r) for r in rhos_s],
            "median_rho": median_rho_s,
            "G_Wells_3_pass": g3_pass,
            "wells_shuffled": {"0": wells_s[0], "1": wells_s[1], "2": wells_s[2]},
            "v_form_fits_shuffled": v_fits_shuf,
        },
        "v_form_fits": v_fits,
        "joint_outcome": joint,
        "runtime_s": float(time.time() - t_start),
        "git_sha": sha,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}")
    print(f"[summary] G1={g1_pass} G2={g2_pass} G3={g3_pass}  "
          f"Add R²={add_r2:.3f} Hop R²={hop_r2:.3f} SpV R²={sv_r2:.3f} SpD R²={sd_r2:.3f}  "
          f"→ {joint}")


if __name__ == "__main__":
    main()
