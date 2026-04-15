#!/usr/bin/env python3
"""
Next-12: Two-level WF comparison at matched budget (avg_bits=2.0)
==================================================================

Objective: Fair comparison against v3 WF(floor=2) at equal budget.
Test whether CWF contributes anything at avg=2.0 (per dim).

Configs (all on Mistral-7B, Pre-RoPE PCA + L² Lloyd):
  A. Uniform 2-bit per dim (baseline, every dim = 2)
  B. Intra-head WF skip-floor=2 (v3 reproduction attempt):
       - per head, allocate dims with skip-or-floor=2 semantic
       - budget per head = 2 × d = 256 bits
       - high-variance dims get 2+, low-variance get 0 (replaced by mean)
  C. Inter-head CWF + uniform within head (our Next-10 config):
       - heads get different total budgets (some > 2d, others < 2d)
       - within each head, dims get uniform bits (floor=2)
  D. Two-level WF (CWF inter-head + WF-floor=2 intra-head):
       - heads get different total budgets via Theorem B
       - within each head, dims get skip-or-floor=2 WF
  E. Pure intra-head Shannon WF (no floor):
       - per head, optimal continuous WF with b≥0
       - some dims get 0, some get many bits

All configs at SAME total budget = 2.0 × d × n_layers × n_kv = 65536 bits.

Success criteria:
  - Config D < Config B: Two-level beats intra-only (validates CWF contribution)
  - Config D < 5.82: Beats v3 WF(floor=2) at matched budget (full SOTA)
  - Config C < Config A: CWF at same budget shows any gain
"""
import json
import time
import gc
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
DEVICE = 'cuda:1'  # Use GPU 1 since GPU 0 has MMLU running
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048
TARGET_AVG_BITS = 2.0  # per dim
B_MAX = 8

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mistral Exp4 per-layer sensitivity
EXP4_MISTRAL_DELTA_PPL = {
    0: 0.005, 1: 0.120, 2: 0.555, 3: 0.287, 4: 0.521, 5: 0.206,
    6: 0.304, 7: 0.166, 8: 0.152, 9: 0.160, 10: 0.070, 11: 0.079,
    12: 0.037, 13: 0.039, 14: 0.034, 15: 0.047, 16: 0.030, 17: 0.050,
    18: 0.025, 19: 0.024, 20: 0.067, 21: 0.067, 22: 0.155, 23: 0.122,
    24: 0.046, 25: 0.010, 26: -0.004, 27: 0.103, 28: 0.032, 29: 0.096,
    30: 0.116, 31: 0.028,
}


# ----------------------------------------------------------------------
# Per-dim Lloyd-Max
# ----------------------------------------------------------------------

def lloyd_max_1d_fit(col, bits, n_iter=30):
    n_levels = 2 ** bits
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.percentile(col, pcts)
    centroids = np.sort(centroids)
    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(boundaries, col)
        new_c = centroids.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - centroids)) < 1e-6:
            break
        centroids = new_c
    return centroids


# ----------------------------------------------------------------------
# Intra-head WF (per dim with floor + skip)
# ----------------------------------------------------------------------

def intra_head_wf_skip_floor(sigma2_per_dim, total_budget, b_floor=2, b_max=8):
    """
    Per-head intra-dim WF with skip-or-floor semantic.

    Args:
        sigma2_per_dim: (d,) variance per PCA dimension
        total_budget: total bits for this head
        b_floor: active dims get at least this many bits; inactive get 0

    Returns: bits per dim (d,) with values in {0, b_floor, b_floor+1, ..., b_max}
    """
    d = len(sigma2_per_dim)
    sigma2 = np.maximum(sigma2_per_dim, 1e-12)
    bits = np.zeros(d, dtype=int)
    spent = 0

    # Greedy marginal allocation
    while spent < total_budget:
        best_gain_per_bit = -np.inf
        best_action = None
        for j in range(d):
            if bits[j] == 0:
                if spent + b_floor > total_budget:
                    continue
                # Activation gain: σ² * (1 - 4^(-b_floor))
                gain = sigma2[j] * (1.0 - 4.0 ** (-b_floor))
                gpb = gain / b_floor
                if gpb > best_gain_per_bit:
                    best_gain_per_bit = gpb
                    best_action = ('activate', j)
            elif bits[j] < b_max:
                gain = sigma2[j] * (4.0 ** (-bits[j]) - 4.0 ** (-(bits[j] + 1)))
                if gain > best_gain_per_bit:
                    best_gain_per_bit = gain
                    best_action = ('add', j)
        if best_action is None:
            break
        action, j = best_action
        if action == 'activate':
            bits[j] = b_floor
            spent += b_floor
        else:
            bits[j] += 1
            spent += 1
    return bits


def intra_head_wf_continuous(sigma2_per_dim, total_budget, b_max=8):
    """
    Per-head intra-dim continuous Shannon WF (no floor, integer clipped).
    Each dim gets b_j = max(0, round(0.5 log4(σ²_j / μ))).
    """
    d = len(sigma2_per_dim)
    sigma2 = np.maximum(sigma2_per_dim, 1e-12)
    bits = np.zeros(d, dtype=int)
    spent = 0

    # Greedy: add 1 bit to dim with highest marginal gain, starting from 0
    while spent < total_budget:
        valid = bits < b_max
        if not valid.any():
            break
        gains = np.where(
            valid,
            sigma2 * (4.0 ** (-bits.astype(float)) - 4.0 ** (-(bits + 1).astype(float))),
            -np.inf
        )
        j_best = int(np.argmax(gains))
        bits[j_best] += 1
        spent += 1
    return bits


# ----------------------------------------------------------------------
# Inter-head Theorem B allocation (CWF)
# ----------------------------------------------------------------------

def inter_head_cwf_allocate(importance, total_budget, floor_per_head, b_max_per_head):
    """
    Allocate total budget across heads via Theorem B (greedy WF).
    Each head must get at least floor_per_head bits (total per head).

    Returns: array of budgets per head
    """
    n = len(importance)
    imp = np.maximum(np.array(importance, dtype=np.float64), 1e-12)
    budgets = np.full(n, floor_per_head, dtype=int)
    spent = n * floor_per_head
    if spent > total_budget:
        return budgets

    # Add 1 bit at a time to head with highest marginal gain
    while spent < total_budget:
        valid = budgets < b_max_per_head
        if not valid.any():
            break
        # Marginal gain ∝ imp[j] * 4^(-budget[j]/d_head)
        # Simplified: use imp as constant importance
        gains = np.where(valid, imp / (budgets + 1.0), -np.inf)
        j_best = int(np.argmax(gains))
        budgets[j_best] += 1
        spent += 1
    return budgets


# ----------------------------------------------------------------------
# Quantizer hook (handles per-dim bits)
# ----------------------------------------------------------------------

def fit_pca_and_per_dim_lloyd(K, bits_per_dim, use_mean_for_zero=True):
    """
    Fit PCA basis + per-dim Lloyd centroids.
    bits_per_dim[j]: bits for dim j (0 means skip, use mean)
    """
    K = K.astype(np.float32)
    K_mean = K.mean(axis=0)
    K_c = K - K_mean
    cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order]
    K_pca = K_c @ V

    d = K.shape[1]
    max_bits = max(int(bits_per_dim.max()), 2)
    centroids = np.zeros((d, 2 ** max_bits), dtype=np.float32)
    pca_mean = K_pca.mean(axis=0)

    for j in range(d):
        b = int(bits_per_dim[j])
        if b == 0:
            # Skip: reconstruct as 0 (PCA mean)
            centroids[j, 0] = 0.0
        else:
            c = lloyd_max_1d_fit(K_pca[:, j], b, n_iter=20).astype(np.float32)
            # Pad with zeros to max_bits
            centroids[j, :len(c)] = c

    return {
        'K_mean': K_mean,
        'V': V.astype(np.float32),
        'centroids': centroids,
        'bits_per_dim': bits_per_dim.astype(np.int32),
        'max_bits': max_bits,
    }


class PerDimLloydHook:
    """Hook applying per-dim Lloyd quantization with variable bits per dim."""
    def __init__(self, head_quantizers, n_kv, head_dim):
        self.hq = head_quantizers
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x_bf = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x_bf.float().cpu().numpy()
        x_q = np.zeros_like(x_np)

        for hk in range(self.n_kv):
            q = self.hq[hk]
            data = x_np[:, :, hk, :]
            shape = data.shape
            K_flat = data.reshape(-1, self.head_dim).astype(np.float32)
            K_c = K_flat - q['K_mean']
            K_pca = K_c @ q['V']

            K_pca_q = np.zeros_like(K_pca)
            c = q['centroids']
            bits_per_dim = q['bits_per_dim']
            for j in range(self.head_dim):
                b = int(bits_per_dim[j])
                if b == 0:
                    K_pca_q[:, j] = 0.0  # skip, use PCA mean
                else:
                    n_levels = 2 ** b
                    cj = c[j, :n_levels]
                    boundaries = (cj[:-1] + cj[1:]) / 2
                    idx = np.searchsorted(boundaries, K_pca[:, j])
                    K_pca_q[:, j] = cj[idx]

            K_recon = K_pca_q @ q['V'].T + q['K_mean']
            x_q[:, :, hk, :] = K_recon.reshape(shape)

        result = torch.from_numpy(x_q).to(output.device).to(output.dtype)
        return result.view(B, T, self.n_kv * self.head_dim)


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def collect_k_and_fisher(model, input_ids, n_layers, n_kv, n_q, head_dim):
    captured = {}
    handles = []
    def kh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['k'] = o.detach().cpu().float().numpy()
        return h
    def qh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['q'] = o.detach().cpu().float().numpy()
        return h
    def ah(li):
        def h(m, i, o):
            if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                captured.setdefault(li, {})['attn'] = o[1].detach().cpu().float().numpy()
        return h

    for li in range(n_layers):
        mod = model.model.layers[li].self_attn
        handles.append(mod.k_proj.register_forward_hook(kh(li)))
        handles.append(mod.q_proj.register_forward_hook(qh(li)))
        handles.append(mod.register_forward_hook(ah(li)))

    with torch.no_grad():
        _ = model(input_ids, output_attentions=True, use_cache=False)
    for h in handles:
        h.remove()

    n_q_per_kv = n_q // n_kv
    K_table = {}           # (li, hk) -> raw K (T, d)
    K_pca_table = {}       # (li, hk) -> K in PCA basis
    sigma2_table = {}      # (li, hk) -> per-dim variance in PCA basis
    V_table = {}           # (li, hk) -> PCA V
    K_mean_table = {}      # (li, hk) -> mean
    trace_table = np.zeros((n_layers, n_kv), dtype=np.float32)

    for li in range(n_layers):
        data = captured.get(li, {})
        if not all(k in data for k in ['k', 'q', 'attn']):
            continue
        T_c = data['k'].shape[1]
        K_all = data['k'].reshape(T_c, n_kv, head_dim).astype(np.float32)
        Q_all = data['q'].reshape(T_c, n_q, head_dim).astype(np.float32)
        attn_all = data['attn'][0].astype(np.float32)

        for hk in range(n_kv):
            K = K_all[:, hk, :]
            K_mean = K.mean(axis=0)
            K_c = K - K_mean
            cov = (K_c.T @ K_c) / max(T_c - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            V = eigvecs[:, order]
            K_pca = K_c @ V
            sigma2 = np.maximum(eigvals[order], 1e-12)

            K_table[(li, hk)] = K
            K_pca_table[(li, hk)] = K_pca
            sigma2_table[(li, hk)] = sigma2
            V_table[(li, hk)] = V
            K_mean_table[(li, hk)] = K_mean

            # Fisher trace
            q_heads = list(range(hk * n_q_per_kv, (hk+1) * n_q_per_kv))
            Q = Q_all[:, q_heads, :].mean(axis=1)
            attn_mean = attn_all[q_heads, :, :].mean(axis=0)
            s_t = (attn_mean * (1.0 - attn_mean)).sum(axis=1)
            M = ((Q * s_t[:, None]).T @ Q) / max(T_c, 1)
            trace_table[li, hk] = float(np.trace(M))

    return {
        'K': K_table,
        'K_pca': K_pca_table,
        'sigma2': sigma2_table,
        'V': V_table,
        'K_mean': K_mean_table,
        'trace': trace_table,
    }


def compute_ppl(model, input_ids):
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
            reduction='mean'
        )
        return float(torch.exp(loss).item()), float(loss.item())


# ----------------------------------------------------------------------
# Config builders
# ----------------------------------------------------------------------

def build_config_A_uniform(calib_data, n_layers, n_kv, head_dim, bits_per_dim=2):
    """Uniform bits per dim (baseline)."""
    per_layer = {li: [None] * n_kv for li in range(n_layers)}
    for (li, hk), K in calib_data['K'].items():
        bpd = np.full(head_dim, bits_per_dim, dtype=int)
        hd = fit_pca_and_per_dim_lloyd(K, bpd)
        per_layer[li][hk] = hd
    return per_layer, 'A_uniform_2b_per_dim'


def build_config_B_intra_skip_floor(calib_data, n_layers, n_kv, head_dim, avg_bits):
    """Per-head skip-or-floor=2 WF (v3 reproduction attempt)."""
    per_layer = {li: [None] * n_kv for li in range(n_layers)}
    per_head_budget = int(avg_bits * head_dim)
    for (li, hk), K in calib_data['K'].items():
        sigma2 = calib_data['sigma2'][(li, hk)]
        bpd = intra_head_wf_skip_floor(sigma2, per_head_budget, b_floor=2, b_max=B_MAX)
        hd = fit_pca_and_per_dim_lloyd(K, bpd)
        per_layer[li][hk] = hd
    return per_layer, 'B_intra_skip_floor2'


def build_config_B2_intra_continuous(calib_data, n_layers, n_kv, head_dim, avg_bits):
    """Per-head continuous Shannon WF (no floor, b≥0)."""
    per_layer = {li: [None] * n_kv for li in range(n_layers)}
    per_head_budget = int(avg_bits * head_dim)
    for (li, hk), K in calib_data['K'].items():
        sigma2 = calib_data['sigma2'][(li, hk)]
        bpd = intra_head_wf_continuous(sigma2, per_head_budget, b_max=B_MAX)
        hd = fit_pca_and_per_dim_lloyd(K, bpd)
        per_layer[li][hk] = hd
    return per_layer, 'B2_intra_continuous_WF'


def build_config_C_inter_uniform(calib_data, n_layers, n_kv, head_dim, avg_bits):
    """Inter-head CWF, uniform within head (Next-10 style)."""
    # Compute importance per head
    sens = np.zeros(n_layers, dtype=np.float32)
    for li in range(n_layers):
        sens[li] = max(0.0, EXP4_MISTRAL_DELTA_PPL.get(li, 0.0)) + 1e-6

    trace_table = calib_data['trace']
    importance = []
    index_map = []
    for li in range(n_layers):
        for hk in range(n_kv):
            imp = float(sens[li]) * float(trace_table[li, hk])
            importance.append(imp)
            index_map.append((li, hk))

    # Total budget in per-head bit units (not per-dim)
    total_head_bits = int(avg_bits * n_layers * n_kv)
    head_bits = inter_head_cwf_allocate(
        importance, total_head_bits,
        floor_per_head=2, b_max_per_head=B_MAX
    )

    per_layer = {li: [None] * n_kv for li in range(n_layers)}
    for k, (li, hk) in enumerate(index_map):
        K = calib_data['K'][(li, hk)]
        b = int(head_bits[k])
        bpd = np.full(head_dim, b, dtype=int)
        hd = fit_pca_and_per_dim_lloyd(K, bpd)
        per_layer[li][hk] = hd
    return per_layer, 'C_inter_CWF_uniform_intra'


def build_config_D_two_level(calib_data, n_layers, n_kv, head_dim, avg_bits):
    """
    Two-level WF: CWF inter-head total budget + per-head skip-floor=2 intra-head.
    """
    sens = np.zeros(n_layers, dtype=np.float32)
    for li in range(n_layers):
        sens[li] = max(0.0, EXP4_MISTRAL_DELTA_PPL.get(li, 0.0)) + 1e-6

    trace_table = calib_data['trace']
    # importance per head in total-budget units
    importance = []
    index_map = []
    for li in range(n_layers):
        for hk in range(n_kv):
            imp = float(sens[li]) * float(trace_table[li, hk])
            importance.append(imp)
            index_map.append((li, hk))

    # Total budget in per-dim × d × n_heads
    total_bits = int(avg_bits * head_dim * n_layers * n_kv)
    # Allocate to heads using proportion-based WF
    # Give each head at least head_dim bits (= 1 bit/dim), max 4×head_dim (= 4 bit/dim)
    min_per_head = 2 * head_dim     # at least 2 bits/dim on average
    max_per_head = 5 * head_dim

    # Greedy allocation of total_bits across heads using importance
    imp_arr = np.array(importance, dtype=np.float64)
    imp_arr = np.maximum(imp_arr, 1e-12)
    head_totals = np.full(len(index_map), min_per_head, dtype=int)
    spent = head_totals.sum()
    if spent > total_bits:
        # Over-allocated via floor — reduce min
        min_per_head = total_bits // len(index_map)
        head_totals = np.full(len(index_map), min_per_head, dtype=int)
        spent = head_totals.sum()

    # Add head_dim bits (= 1 full bit per dim per head) at a time greedily
    # Approximate marginal gain: imp / (current_bits / head_dim)
    step = max(1, head_dim // 4)  # 32 bits per step = 0.25 bits/dim
    while spent + step <= total_bits:
        valid = head_totals < max_per_head
        if not valid.any():
            break
        # Marginal value of adding 1 bit/dim to head j: imp * 4^(-b_j/d)
        avg_b = head_totals / head_dim
        gains = np.where(valid, imp_arr * (4.0 ** (-avg_b)), -np.inf)
        j_best = int(np.argmax(gains))
        head_totals[j_best] += step
        spent += step

    # Now allocate intra-head per dim using skip-floor=2 WF
    per_layer = {li: [None] * n_kv for li in range(n_layers)}
    for k, (li, hk) in enumerate(index_map):
        K = calib_data['K'][(li, hk)]
        sigma2 = calib_data['sigma2'][(li, hk)]
        head_budget = int(head_totals[k])
        bpd = intra_head_wf_skip_floor(sigma2, head_budget, b_floor=2, b_max=B_MAX)
        hd = fit_pca_and_per_dim_lloyd(K, bpd)
        per_layer[li][hk] = hd
    return per_layer, 'D_two_level_CWF_plus_intra_skip_floor2'


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def install_hooks(model, per_layer_data, n_kv, head_dim):
    handles = []
    for li, head_list in per_layer_data.items():
        # Fill None with passthrough
        for hk in range(n_kv):
            if head_list[hk] is None:
                head_list[hk] = {
                    'K_mean': np.zeros(head_dim, dtype=np.float32),
                    'V': np.eye(head_dim, dtype=np.float32),
                    'centroids': np.zeros((head_dim, 4), dtype=np.float32),
                    'bits_per_dim': np.full(head_dim, 2, dtype=np.int32),
                    'max_bits': 2,
                }
        hook = PerDimLloydHook(head_list, n_kv, head_dim)
        h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
        handles.append(h)
    return handles


def measure_config(model, eval_ids, builder_fn, calib_data, n_layers, n_kv, head_dim):
    t_fit = time.time()
    per_layer_data, cfg_name = builder_fn(calib_data, n_layers, n_kv, head_dim, TARGET_AVG_BITS)
    fit_time = time.time() - t_fit

    # Compute actual avg bits
    total_bits = 0
    n_dims = 0
    for li, hdlist in per_layer_data.items():
        for hd in hdlist:
            if hd is None:
                continue
            total_bits += int(hd['bits_per_dim'].sum())
            n_dims += head_dim
    actual_avg = total_bits / max(n_dims, 1)

    handles = install_hooks(model, per_layer_data, n_kv, head_dim)
    t_eval = time.time()
    try:
        ppl, loss = compute_ppl(model, eval_ids)
    except Exception as e:
        ppl, loss = float('inf'), float('inf')
        print(f"    PPL FAILED: {e}", flush=True)
    eval_time = time.time() - t_eval
    for h in handles:
        h.remove()

    return {
        'config_name': cfg_name,
        'avg_bits_actual': actual_avg,
        'ppl': ppl,
        'loss': loss,
        'fit_time': fit_time,
        'eval_time': eval_time,
    }


def main():
    print("=" * 70)
    print("Next-12: Two-level WF at matched budget (avg=2.0 per dim)")
    print("=" * 70, flush=True)
    t_start = time.time()

    print("\nLoading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s on {DEVICE}", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}", flush=True)

    # Data
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in ds['text'] if len(t.strip()) > 100]
        calib_text = '\n\n'.join(texts[:300])
        eval_text = '\n\n'.join(texts[300:600])
    except Exception:
        calib_text = " ".join(["Calib."] * 5000)
        eval_text = " ".join(["Eval."] * 5000)

    calib_enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    calib_ids = calib_enc['input_ids'].to(DEVICE)
    eval_enc = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL_TOKENS)
    eval_ids = eval_enc['input_ids'].to(DEVICE)

    # Baseline
    print("\n[Baseline] FP16 PPL...", flush=True)
    ppl_fp16, _ = compute_ppl(model, eval_ids)
    print(f"  FP16 PPL: {ppl_fp16:.4f}", flush=True)

    # Collect calibration
    print("\n[Phase 1] Collecting K + PCA + Fisher...", flush=True)
    t0 = time.time()
    calib_data = collect_k_and_fisher(model, calib_ids, n_layers, n_kv, n_q, head_dim)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Test all configs
    configs_to_run = [
        ('A_uniform',       lambda cd, nl, nk, hd, ab: build_config_A_uniform(cd, nl, nk, hd, 2)),
        ('B_intra_skip',    lambda cd, nl, nk, hd, ab: build_config_B_intra_skip_floor(cd, nl, nk, hd, ab)),
        ('B2_intra_cont',   lambda cd, nl, nk, hd, ab: build_config_B2_intra_continuous(cd, nl, nk, hd, ab)),
        ('C_inter_cwf',     lambda cd, nl, nk, hd, ab: build_config_C_inter_uniform(cd, nl, nk, hd, ab)),
        ('D_two_level',     lambda cd, nl, nk, hd, ab: build_config_D_two_level(cd, nl, nk, hd, ab)),
    ]

    results = {'ppl_fp16': ppl_fp16, 'target_avg_bits': TARGET_AVG_BITS, 'configs': {}}

    for cfg_key, builder in configs_to_run:
        print(f"\n[{cfg_key}] Fitting + measuring...", flush=True)
        try:
            res = measure_config(model, eval_ids, builder, calib_data, n_layers, n_kv, head_dim)
            delta = (res['ppl'] - ppl_fp16) / ppl_fp16 * 100
            print(f"  {res['config_name']}: "
                  f"avg_bits={res['avg_bits_actual']:.3f}, "
                  f"PPL={res['ppl']:.4f} (Δ {delta:+.2f}%) "
                  f"[fit: {res['fit_time']:.0f}s, eval: {res['eval_time']:.1f}s]",
                  flush=True)
            results['configs'][cfg_key] = {
                **res,
                'delta_vs_fp16_pct': delta,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            results['configs'][cfg_key] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Fair comparison at avg=2.0 bits/dim")
    print("=" * 70)
    print(f"  FP16 baseline: {ppl_fp16:.4f}")
    print(f"  Target: beat v3 WF(floor=2) = 5.82 PPL")
    print()
    print(f"{'Config':<32} | {'avg_bits':>10} | {'PPL':>10} | {'vs FP16':>10}")
    print('-' * 75)
    for cfg_key, cfg in results['configs'].items():
        if 'ppl' in cfg:
            print(f"  {cfg['config_name']:<30} | {cfg['avg_bits_actual']:>10.3f} | "
                  f"{cfg['ppl']:>10.4f} | {cfg['delta_vs_fp16_pct']:>+9.2f}%")

    # Verdict
    print("\n=== Verdict ===")
    best = None
    best_ppl = float('inf')
    for cfg_key, cfg in results['configs'].items():
        if 'ppl' in cfg and cfg['ppl'] < best_ppl:
            best_ppl = cfg['ppl']
            best = cfg_key
    print(f"  Best config: {best} → PPL {best_ppl:.4f}")
    print(f"  vs v3 WF(floor=2) (5.82):  {(best_ppl - 5.82) / 5.82 * 100:+.2f}%")
    print(f"  vs v3 Uniform 2b (6.4614): {(best_ppl - 6.4614) / 6.4614 * 100:+.2f}%")

    if best_ppl < 5.82:
        print(f"\n  🏆 SUCCESS: beats v3 WF(floor=2) at matched budget")
    elif best_ppl < 6.4614:
        print(f"\n  ⚠️  PARTIAL: beats v3 Uniform 2b but not WF(floor=2)")
    else:
        print(f"\n  ❌ FAIL: worse than v3 Uniform 2b at matched budget")

    results['runtime_sec'] = time.time() - t_start
    out_file = OUT_DIR / 'exp_next12_two_level_wf.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out_file}")
    print(f"Total: {results['runtime_sec']:.0f}s ({results['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
