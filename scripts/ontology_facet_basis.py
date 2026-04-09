#!/usr/bin/env python3
"""
ontology_facet_basis — Phase 0 infrastructure for the paper:
  "Ontology-Guided K-Side Attention Bias: Focus Shifting without Fabrication"

Role in the pipeline: build the ontology-derived facet basis B ∈ ℝ^{d×r} in
per-head K-space that later phases use as the intervention direction for
K-bias injection.  This file produces *directions*, not interventions — the
injection hook lives in a separate module (kbias_injection.py, Phase 1).

What this script does:
  1. Run each sentence of a priority-ordered ontology through the LLM and
     extract per-category mean K vectors per (layer, head).  Only content
     tokens participate (BOS position 0 is a sink, excluded).
  2. Per (layer, head), Gram-Schmidt residualize the facet embedding matrices
     in priority order, SVD each residual, and concatenate into a single
     orthonormal basis B with block structure B = [B_domain | B_mfr | ...].
  3. Calibrate a per-head K covariance Σ_K from WikiText-2 using the **same
     token-selection rule as extraction** (content tokens only, BOS excluded)
     so that B and Σ_K live on the same support.
  4. Measure a diagnostic Hadamard ratio
         η_facet = det(B^T Σ_K B) / Π diag(B^T Σ_K B)   ∈ (0, 1]
     and a PCA sanity check (should be ≈ 1.0 by construction).

Note on η_facet's role (changed from the original framing):

  In the previous quant+steering unification framing, η_facet was the scalar
  that decided whether quantization was viable.  In the current standalone
  paper framing ("Ontology-Guided K-Side Attention Bias"), quantization is
  out of scope — the facet basis is used only as a direction source for
  K-bias injection.  η_facet is kept here as a diagnostic: values near 0
  indicate that the basis is poorly aligned with the K-space statistics,
  which would make downstream bias injection either ineffective (if the
  basis misses the variance-carrying directions) or destructive (if it
  clobbers the dominant directions).  A healthy range is roughly
  η_facet ∈ [0.1, 0.9].  This script does NOT gate the paper on η_facet;
  it only reports the diagnostic.

Padding fix (2026-04-09):

  A previous run of this pipeline (exp_facet_basis.py, pre-rename) reported
  η_facet ≈ 0 for 56/56 heads.  Root cause: K-vector extraction excluded BOS
  (correct), but the Σ_K calibration pipeline did NOT, so B was constructed
  on content tokens while Σ_K lived on content∪{BOS}.  The two tensors
  occupied different subspaces and their alignment collapsed.  This file
  excludes BOS in both places (see compute_per_head_sigma below).

Output: reports/axis2_theoretical_verification/ontology_facet_basis.json
"""

import os
import json
import time

os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU 0 in use elsewhere

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
MODEL_ID = 'mistralai/Mistral-7B-v0.3'
SHORT_NAME = 'mistral-7b'
TARGET_LAYERS = [0, 1, 2, 7, 15, 23, 31]
N_CALIB_TOKENS = 2048
ENERGY_THRESHOLD = 0.95
MIN_SINGVAL_RATIO = 1e-3
OUT = Path(
    '/home/woori/workspace_common/boltzmann-attention/reports/'
    'axis2_theoretical_verification/ontology_facet_basis.json'
)


# =====================================================================
# 1. Ontology — 4 facets, ordered most → least general
# =====================================================================

ONTOLOGY = {
    'domain': {
        'consumer':   [
            'Most households use these products daily for personal needs.',
            'They are designed for everyday personal use at home.',
            'Casual users typically buy them at retail stores or online.',
            'These appliances appear in nearly every modern home.',
        ],
        'gaming':     [
            'High frame rates and low latency matter most for competitive play.',
            'Esports tournaments use this category of hardware.',
            'Streamers and competitive players choose them for performance.',
            'Cooling and refresh rate dominate the spec sheet.',
        ],
        'enterprise': [
            'Companies deploy them in data centers at very large scale.',
            'IT departments procure these for office workflows and security.',
            'They emphasize reliability and security and uptime guarantees.',
            'Service-level agreements define their support contracts.',
        ],
        'automotive': [
            'Car manufacturers integrate these into modern vehicles.',
            'Drivers rely on them for safety and navigation features.',
            'They must meet strict automotive certification standards.',
            'These components ship in modern electric and hybrid cars.',
        ],
        'medical':    [
            'Hospitals use them for diagnosis and treatment of patients.',
            'Clinical trials validate their accuracy and safety.',
            'Doctors operate them under strict regulatory approval.',
            'These devices help patients in critical care situations.',
        ],
    },
    'manufacturer': {
        'Apple':   [
            'Apple released a new product at its California keynote event.',
            'Tim Cook announced the latest from Cupertino.',
            'Apple engineers designed the device entirely in California.',
            'The Apple logo is recognized worldwide as a premium brand.',
        ],
        'Samsung': [
            'Samsung unveiled a new lineup at the international trade show.',
            'The Korean conglomerate Samsung leads the display market.',
            'Samsung Electronics manufactures both displays and chips.',
            'A Samsung device offers unique foldable form factor features.',
        ],
        'Google':  [
            'Google launched a new product at its annual developer conference.',
            'The Alphabet subsidiary Google develops AI hardware in house.',
            'Google engineers built the device in Mountain View, California.',
            'Google services integrate seamlessly across all their devices.',
        ],
        'Nvidia':  [
            'Nvidia announced new GPU architectures for AI training workloads.',
            'Jensen Huang presented the latest from Nvidia at the keynote.',
            'Nvidia GPUs accelerate deep learning research worldwide.',
            'The Nvidia software stack dominates machine learning research.',
        ],
        'Tesla':   [
            'Tesla revealed its next vehicle at the company event yesterday.',
            'Elon Musk demonstrated the Tesla autopilot driving system.',
            'Tesla operates gigafactories around the world for batteries.',
            'A Tesla car runs entirely on advanced battery technology.',
        ],
    },
    'product_type': {
        'phone':       [
            'The phone fits comfortably in one hand for daily use.',
            'Users make calls and send messages with their phone.',
            'A modern phone has multiple cameras on the back.',
            'The phone runs apps from a curated application store.',
        ],
        'laptop':      [
            'The laptop folds shut for easy transport in a bag.',
            'Workers carry their laptop to meetings every day.',
            'A laptop has a keyboard and trackpad built into it.',
            'The laptop battery lasts a full workday on a charge.',
        ],
        'gpu':         [
            'The GPU accelerates graphics and matrix multiplication tasks.',
            'Researchers train large neural networks on GPU clusters.',
            'A GPU plugs into the PCIe slot of a desktop computer.',
            'The GPU has thousands of parallel cores for compute.',
        ],
        'car':         [
            'The car drives smoothly on the highway at high speed.',
            'Drivers steer the car with the steering wheel and pedals.',
            'A modern car has lane-keeping assistance built in.',
            'The car accelerates from zero to sixty in a few seconds.',
        ],
        'watch':       [
            'The watch tracks heart rate and sleep cycles continuously.',
            'A wearer fastens the watch around the wrist with a strap.',
            'The smart watch displays notifications synced from a phone.',
            'A watch with GPS tracks running and cycling routes outdoors.',
        ],
    },
    'price_tier': {
        'budget':  [
            'It is the most affordable option in the entire lineup.',
            'Buyers on a tight budget pick this entry-level model.',
            'The basic version costs less than two hundred dollars total.',
            'Cost-conscious customers choose this affordable variant.',
        ],
        'mid':     [
            'It balances features and price for the average buyer.',
            'The mid-range tier offers solid performance per dollar spent.',
            'Most consumers select the standard mid-tier configuration.',
            'A mid-priced model is the best value for most use cases.',
        ],
        'premium': [
            'The premium model includes the best materials and components.',
            'Enthusiasts pay extra for the top-tier flagship version.',
            'A luxury buyer chooses the most expensive variant available.',
            'The premium tier targets professional and high-end users.',
        ],
    },
}

# Facet ordering: most general → most specific.
# Higher priority facets keep their full subspace; later facets are
# residualized against everything above them.
FACET_ORDER = ['domain', 'manufacturer', 'product_type', 'price_tier']


# =====================================================================
# 2. Per-category K vector extraction
# =====================================================================

@torch.no_grad()
def extract_category_K(model, tok, ontology, target_layers, n_kv, head_dim):
    """For each (layer, head, facet, category), compute the mean K vector.

    Run each example sentence through the LLM, capture k_proj output, average
    over content tokens (excluding position 0 BOS sink), and average across
    sentences for each category.

    Returns dict: (facet, category, layer, head) → np.ndarray (head_dim,).
    """
    n_layers = model.config.num_hidden_layers
    pl_buf = [None] * n_layers

    handles = []
    for li in target_layers:
        layer = model.model.layers[li]

        def make_hook(li_):
            def h(m, inp, out):
                pl_buf[li_] = out.detach().cpu().float().numpy()
            return h

        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li)))

    out = {}
    for facet_name, cats in ontology.items():
        for cat_name, sentences in cats.items():
            sentence_K = {(li, h): [] for li in target_layers for h in range(n_kv)}
            for sent in sentences:
                ids = tok(sent, return_tensors='pt')['input_ids'].to('cuda:0')
                model(ids, use_cache=False)
                for li in target_layers:
                    K_li = pl_buf[li]  # (1, T, n_kv*head_dim)
                    if K_li is None:
                        continue
                    K_li = K_li[0]  # (T, n_kv*head_dim)
                    K_li = K_li.reshape(K_li.shape[0], n_kv, head_dim)
                    if K_li.shape[0] > 1:
                        K_li = K_li[1:]  # exclude BOS
                    for h in range(n_kv):
                        sentence_K[(li, h)].append(K_li[:, h, :].mean(0))
            for (li, h), v_list in sentence_K.items():
                if v_list:
                    out[(facet_name, cat_name, li, h)] = np.stack(v_list).mean(0)

    for hd in handles:
        hd.remove()
    return out


# =====================================================================
# 3. Gram-Schmidt residualization (Frisch-Waugh-Lovell on subspaces)
# =====================================================================

def build_facet_basis(E_list, energy_threshold=ENERGY_THRESHOLD,
                      min_singval_ratio=MIN_SINGVAL_RATIO):
    """Build a facet-orthogonal basis from a priority-ordered list of
    embedding matrices.

    Args:
        E_list: list of E^(f), each (d, K_f). First is highest priority.
        energy_threshold: cumulative singular-value-energy at which to truncate.
        min_singval_ratio: drop singular values below this fraction of σ_max.

    Returns:
        B: (d, r_tot) facet basis with B^T B = I (up to numerical noise).
        r_per_facet: list of int, dimension allocated to each facet.
    """
    d = E_list[0].shape[0]
    P_so_far = np.zeros((d, d))
    bases = []
    r_per_facet = []

    for f, E in enumerate(E_list):
        E_res = E - P_so_far @ E

        if E_res.size == 0 or np.allclose(E_res, 0, atol=1e-10):
            r_per_facet.append(0)
            bases.append(np.zeros((d, 0)))
            continue

        U, S, Vt = np.linalg.svd(E_res, full_matrices=False)

        # cumulative energy truncation
        cum_energy = np.cumsum(S ** 2) / max(np.sum(S ** 2), 1e-30)
        r_energy = int(np.searchsorted(cum_energy, energy_threshold) + 1)

        # drop tiny singular values
        if S[0] > 0:
            r_nonzero = int(np.sum(S > min_singval_ratio * S[0]))
        else:
            r_nonzero = 0

        r_f = max(0, min(r_energy, r_nonzero, len(S)))
        r_per_facet.append(r_f)

        if r_f == 0:
            bases.append(np.zeros((d, 0)))
            continue

        B_f = U[:, :r_f]
        bases.append(B_f)
        P_so_far = P_so_far + B_f @ B_f.T

    nonempty = [Bf for Bf in bases if Bf.shape[1] > 0]
    if not nonempty:
        return np.zeros((d, 0)), r_per_facet
    B = np.concatenate(nonempty, axis=1)
    return B, r_per_facet


# =====================================================================
# 4. η_facet measurement
# =====================================================================

def measure_eta_facet(B, Sigma):
    """Hadamard ratio η = det(M) / Π diag(M) where M = B_full^T Σ B_full.

    If B is rank-deficient (r_tot < d), pad with PCA of the residual subspace
    so that the determinant is well-defined relative to the total variance.
    The PCA padding is optimal in the residual, so any inefficiency in the
    final η comes from the facet block alone.
    """
    d = Sigma.shape[0]
    if B.shape[1] == 0:
        return 0.0

    if B.shape[1] < d:
        # Pad with PCA of (I - BB^T) Σ (I - BB^T)
        P_ortho = np.eye(d) - B @ B.T
        Sigma_res = P_ortho @ Sigma @ P_ortho
        Sigma_res = 0.5 * (Sigma_res + Sigma_res.T)
        w, V = np.linalg.eigh(Sigma_res)
        order = np.argsort(-w)
        V_sorted = V[:, order]

        n_extra = d - B.shape[1]
        Q, _ = np.linalg.qr(np.concatenate([B, V_sorted], axis=1))
        B_full = Q[:, :d]
    else:
        B_full = B

    M = B_full.T @ Sigma @ B_full
    M = 0.5 * (M + M.T)
    diag = np.maximum(np.diag(M), 1e-30)
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return 0.0
    log_eta = logdet - np.sum(np.log(diag))
    return float(np.exp(min(log_eta, 0.0)))  # cap at 1.0


# =====================================================================
# 5. Calibration Σ_K from WikiText-2
# =====================================================================

@torch.no_grad()
def compute_per_head_sigma(model, tok, target_layers, n_kv, head_dim,
                           n_tokens=N_CALIB_TOKENS):
    """Compute per-(layer, head) key covariance Σ_K from WT2 train tokens.

    BOS handling: position 0 is excluded to match extract_category_K.  This
    is the critical fix relative to the pre-2026-04-09 pipeline, which
    computed Σ_K on content∪{BOS} while the facet basis B was built on
    content only — producing η_facet ≈ 0 for every head (padding artifact).
    """
    from datasets import load_dataset

    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    ids = tok(calib_text, return_tensors='pt', truncation=True,
              max_length=n_tokens)['input_ids'].to('cuda:0')

    n_layers = model.config.num_hidden_layers
    pl_buf = [None] * n_layers
    handles = []
    for li in target_layers:
        layer = model.model.layers[li]

        def make_hook(li_):
            def h(m, inp, out):
                pl_buf[li_] = out.detach().cpu().float().numpy()
            return h

        handles.append(layer.self_attn.k_proj.register_forward_hook(make_hook(li)))

    model(ids, use_cache=False)

    for hd in handles:
        hd.remove()

    sigmas = {}
    for li in target_layers:
        K_li = pl_buf[li]
        if K_li is None:
            continue
        K_li = K_li[0]                                  # (T, n_kv*head_dim)
        T = K_li.shape[0]
        K_li = K_li.reshape(T, n_kv, head_dim)
        # PADDING FIX (2026-04-09): exclude BOS sink token to match the token
        # selection rule in extract_category_K.  Without this line the basis B
        # and the covariance Σ_K live on different token supports and η_facet
        # collapses to ~0 for every head.
        if K_li.shape[0] > 1:
            K_li = K_li[1:]
        for h in range(n_kv):
            X = K_li[:, h, :]
            X = X - X.mean(0)
            Sigma = (X.T @ X) / max(X.shape[0] - 1, 1)
            sigmas[(li, h)] = Sigma.astype(np.float64)
    return sigmas


# =====================================================================
# 6. Main
# =====================================================================

def main():
    print("=" * 72)
    print("exp_facet_basis — Day 1 prototype")
    print("=" * 72)
    print(f"  model           : {MODEL_ID}")
    print(f"  facet order     : {FACET_ORDER}")
    print(f"  target layers   : {TARGET_LAYERS}")
    print(f"  energy threshold: {ENERGY_THRESHOLD}")
    print()
    for f, cats in ONTOLOGY.items():
        print(f"  {f}: {len(cats)} categories — {list(cats.keys())}")
    print()

    print("Loading model ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = (
        getattr(model.config, 'head_dim', None)
        or (model.config.hidden_size // n_q)
    )
    print(f"  loaded in {time.time()-t0:.1f}s: "
          f"n_layers={n_layers}  n_kv={n_kv}  head_dim={head_dim}",
          flush=True)

    print("\n[1/4] Extracting per-category K vectors ...", flush=True)
    t0 = time.time()
    cat_K = extract_category_K(model, tok, ONTOLOGY, TARGET_LAYERS, n_kv, head_dim)
    print(f"  extracted {len(cat_K)} (facet, cat, layer, head) entries "
          f"in {time.time()-t0:.1f}s",
          flush=True)

    print("\n[2/4] Computing per-head Σ_K (WT2 calibration) ...", flush=True)
    t0 = time.time()
    sigmas = compute_per_head_sigma(model, tok, TARGET_LAYERS, n_kv, head_dim)
    print(f"  computed Σ for {len(sigmas)} (layer, head) pairs "
          f"in {time.time()-t0:.1f}s",
          flush=True)

    del model
    torch.cuda.empty_cache()

    print("\n[3/4] Building facet bases per (layer, head) ...", flush=True)
    results_per_pair = {}
    eta_values = []

    for (li, h), Sigma in sigmas.items():
        E_list = []
        for facet_name in FACET_ORDER:
            cats = list(ONTOLOGY[facet_name].keys())
            cols = []
            for c in cats:
                key = (facet_name, c, li, h)
                if key not in cat_K:
                    cols = None
                    break
                cols.append(cat_K[key])
            if cols is None:
                E_list = None
                break
            E_f = np.stack(cols, axis=1).astype(np.float64)
            E_list.append(E_f)
        if E_list is None:
            continue

        B, r_per_facet = build_facet_basis(E_list)
        if B.shape[1] == 0:
            continue

        # orthogonality sanity
        BtB = B.T @ B
        ortho_err = float(np.max(np.abs(BtB - np.eye(B.shape[1]))))

        # inter-facet block-cross-term sanity
        offsets = np.cumsum([0] + r_per_facet)
        max_cross = 0.0
        for f1 in range(len(r_per_facet)):
            for f2 in range(f1 + 1, len(r_per_facet)):
                if r_per_facet[f1] == 0 or r_per_facet[f2] == 0:
                    continue
                block = BtB[
                    offsets[f1]:offsets[f1] + r_per_facet[f1],
                    offsets[f2]:offsets[f2] + r_per_facet[f2],
                ]
                max_cross = max(max_cross, float(np.max(np.abs(block))))

        eta = measure_eta_facet(B, Sigma)

        # PCA reference (should be ≈ 1.0 by construction)
        w_pca, V_pca = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
        order = np.argsort(-w_pca)
        V_pca = V_pca[:, order]
        eta_pca = measure_eta_facet(V_pca, Sigma)

        M = B.T @ Sigma @ B
        diag_vals = np.diag(M).tolist()

        results_per_pair[f"L{li}_H{h}"] = {
            'layer': li,
            'head': h,
            'r_per_facet': r_per_facet,
            'r_tot': int(B.shape[1]),
            'orthogonality_max_err': ortho_err,
            'inter_facet_max_err': max_cross,
            'eta_facet': eta,
            'eta_pca_check': eta_pca,
            'diag_first8': diag_vals[:8],
            'diag_sum': float(np.sum(diag_vals)),
            'sigma_trace': float(np.trace(Sigma)),
        }
        eta_values.append(eta)

        print(
            f"  L{li:2d} H{h}: r_per_facet={r_per_facet}  r_tot={B.shape[1]:3d}  "
            f"η_facet={eta:.4f}  η_pca={eta_pca:.4f}  "
            f"orth={ortho_err:.1e}  cross={max_cross:.1e}",
            flush=True,
        )

    if not eta_values:
        print("\nNo valid (layer, head) pairs.", flush=True)
        return

    print(f"\n[4/4] Summary across {len(eta_values)} (layer, head) pairs:",
          flush=True)
    eta_arr = np.array(eta_values)
    summary = {
        'n_pairs': int(len(eta_arr)),
        'eta_median': float(np.median(eta_arr)),
        'eta_mean': float(np.mean(eta_arr)),
        'eta_q25': float(np.percentile(eta_arr, 25)),
        'eta_q75': float(np.percentile(eta_arr, 75)),
        'eta_min': float(np.min(eta_arr)),
        'eta_max': float(np.max(eta_arr)),
    }
    print(f"  η_facet median  : {summary['eta_median']:.4f}")
    print(f"  η_facet mean    : {summary['eta_mean']:.4f}")
    print(f"  η_facet IQR     : "
          f"[{summary['eta_q25']:.4f}, {summary['eta_q75']:.4f}]")
    print(f"  η_facet range   : "
          f"[{summary['eta_min']:.4f}, {summary['eta_max']:.4f}]")
    print()

    if summary['eta_median'] >= 0.3:
        verdict = "PROCEED — quantization+steering combination is viable"
    elif summary['eta_median'] >= 0.1:
        verdict = "MARGINAL — proceed with trade-off framing"
    else:
        verdict = "PIVOT — quantization side untenable, focus on steering-only"
    print(f"  → verdict: {verdict}")

    out = {
        'model': MODEL_ID,
        'short_name': SHORT_NAME,
        'ontology_summary': {f: list(c.keys()) for f, c in ONTOLOGY.items()},
        'facet_order': FACET_ORDER,
        'energy_threshold': ENERGY_THRESHOLD,
        'min_singval_ratio': MIN_SINGVAL_RATIO,
        'target_layers': TARGET_LAYERS,
        'n_calib_tokens': N_CALIB_TOKENS,
        'per_pair_results': results_per_pair,
        'summary': summary,
        'verdict': verdict,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
