"""
FOKVQ Experiment 3-1: Continuous Rotation Optimization (Lie Group)
=================================================================
Finds the optimal rotation matrix R in SO(d) that minimizes
quantization error of R @ K, using Givens rotations.

Method:
1. Load GPT-2 Medium
2. For each layer, parameterize rotation R as a product of Givens rotations
   in the SO(d) Lie group
3. Optimize rotation angles to minimize quantization MSE of R @ K
4. Compare:
   - Identity (no rotation, direct quantization)
   - PCA rotation (principal component axes)
   - Random rotation (random orthogonal matrix)
   - Optimized rotation (Givens-optimized)
5. Use scipy.optimize with gradient-free methods (Nelder-Mead / Powell)

Hypothesis: Optimized rotation achieves lower quantization error than
            fixed axes (PCA, random), demonstrating the value of
            continuous optimization on SO(d).
"""

import json
import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats, optimize
from sklearn.decomposition import PCA

# ── Configuration ──────────────────────────────────────────────────
MODEL_NAME = "openai-community/gpt2-medium"
DEVICE = "cuda:2"
MAX_SEQ_LEN = 128
NUM_SAMPLES = 40
QUANT_BITS = 3  # Use 3-bit quantization (where rotation matters most)
# Number of Givens rotation planes to optimize per layer
# Full SO(d) has d*(d-1)/2 parameters; we optimize a subset
NUM_GIVENS_PLANES = 64
MAX_OPT_ITERS = 200
LAYERS_TO_OPTIMIZE = list(range(0, 24, 3))  # Every 3rd layer for speed
OUTPUT_DIR = Path("/mnt/input/fokvq/results/exp3_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Calibration texts ─────────────────────────────────────────────
TEXTS = [
    "The Boltzmann distribution assigns probability proportional to exp(-E/kT) where E is energy and T is temperature.",
    "Gradient descent minimizes a loss function by iteratively updating parameters in the direction of steepest descent.",
    "The transformer architecture uses multi-head self-attention to compute weighted sums of value vectors.",
    "Bayesian inference updates prior beliefs using likelihood functions to obtain posterior distributions over parameters.",
    "Principal component analysis finds orthogonal directions of maximum variance in high-dimensional datasets.",
    "def forward(self, x): return self.linear(self.relu(self.norm(x)))",
    "for i in range(len(data)): result.append(process(data[i]))",
    "class DataLoader: def __init__(self, dataset, batch_size=32, shuffle=True): self.dataset = dataset",
    "import torch; model = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.ReLU())",
    "Hey, did you see the game last night? It was absolutely incredible, they came back from a huge deficit.",
    "I'm thinking about getting a new phone. The latest models have such amazing cameras and battery life.",
    "So my friend told me about this restaurant downtown and we should definitely check it out sometime.",
    "The weather has been so unpredictable lately. One day it's sunny and the next it's pouring rain.",
    "The Fourier transform decomposes a signal into constituent frequencies revealing its spectral content.",
    "Variational autoencoders learn latent representations by maximizing a lower bound on the data log-likelihood.",
    "Quantum entanglement creates correlations between particles that cannot be explained by classical physics alone.",
    "The central limit theorem states that the sum of independent random variables converges to a normal distribution.",
    "Convolutional neural networks apply learned spatial filters to detect hierarchical patterns in image data.",
    "I just finished reading this book and honestly it completely changed my perspective on life.",
    "The second law of thermodynamics states that entropy of an isolated system never decreases over time.",
]


def extract_k_per_layer(model, tokenizer, texts, device, max_seq_len):
    """
    Extract K (key) vectors per layer.

    Returns:
        dict: layer_idx -> np.ndarray (total_tokens, d)
    """
    num_layers = model.config.n_layer
    k_vectors = {l: [] for l in range(num_layers)}

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        for l in range(num_layers):
            h = hidden_states[l]
            attn = model.transformer.h[l].attn
            qkv = attn.c_attn(h)
            _, k, _ = qkv.split(model.config.n_embd, dim=-1)
            k_np = k.squeeze(0).detach().cpu().float().numpy()
            k_vectors[l].append(k_np)

        del inputs, outputs, hidden_states
        torch.cuda.empty_cache()

    for l in range(num_layers):
        k_vectors[l] = np.concatenate(k_vectors[l], axis=0)

    return k_vectors


def quantize_vector(v, n_bits):
    """Simulate uniform quantization to n_bits."""
    if n_bits >= 16:
        return v.copy()
    n_levels = 2 ** n_bits
    v_min = v.min(axis=0, keepdims=True)
    v_max = v.max(axis=0, keepdims=True)
    v_range = v_max - v_min
    v_range = np.where(v_range < 1e-10, 1.0, v_range)

    v_norm = (v - v_min) / v_range
    v_quant = np.round(v_norm * (n_levels - 1)) / (n_levels - 1)
    v_deq = v_quant * v_range + v_min
    return v_deq


def quantize_mse(k_data, rotation_matrix, n_bits):
    """
    Compute MSE of quantizing K after applying rotation.

    MSE = ||K - R^T Q(R K)||^2 / n
    where Q is quantization, R is rotation.
    """
    # Rotate
    k_rotated = k_data @ rotation_matrix  # (n, d)

    # Quantize in rotated space
    k_rotated_q = quantize_vector(k_rotated, n_bits)

    # Inverse rotate
    k_reconstructed = k_rotated_q @ rotation_matrix.T

    # MSE in original space
    mse = np.mean((k_data - k_reconstructed) ** 2)
    return mse


def select_givens_planes(d, num_planes, k_data, seed=42):
    """
    Select which (i, j) planes to use for Givens rotations.

    Strategy: pick planes where the coordinate-wise variance ratio is most
    uneven, as these benefit most from rotation.

    Args:
        d: dimension
        num_planes: number of planes to select
        k_data: (n, d) data for variance-based selection
        seed: random seed

    Returns:
        list of (i, j) tuples
    """
    rng = np.random.RandomState(seed)

    # Compute per-coordinate variance
    coord_var = np.var(k_data, axis=0)  # (d,)

    # Score each pair by variance ratio (higher = more benefit from rotation)
    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            ratio = max(coord_var[i], coord_var[j]) / (min(coord_var[i], coord_var[j]) + 1e-10)
            candidates.append((ratio, i, j))

    # Sort by ratio (highest first) and take top planes
    candidates.sort(reverse=True)
    planes = [(c[1], c[2]) for c in candidates[:num_planes]]

    return planes


def givens_rotation_matrix(d, planes, angles):
    """
    Construct rotation matrix R in SO(d) as product of Givens rotations.

    R = G_1(theta_1) @ G_2(theta_2) @ ... @ G_k(theta_k)

    where G_m(theta) is identity except for the 2x2 block at (i_m, j_m).

    Args:
        d: dimension
        planes: list of (i, j) tuples specifying rotation planes
        angles: array of rotation angles (one per plane)

    Returns:
        R: (d, d) orthogonal matrix
    """
    R = np.eye(d)
    for (i, j), theta in zip(planes, angles):
        c = math.cos(theta)
        s = math.sin(theta)
        # Apply Givens rotation to R
        # G @ R: affects rows i and j of R
        R_i = R[i, :].copy()
        R_j = R[j, :].copy()
        R[i, :] = c * R_i - s * R_j
        R[j, :] = s * R_i + c * R_j

    return R


def optimize_rotation(k_data, d, n_bits, planes, max_iters=200):
    """
    Optimize Givens rotation angles to minimize quantization MSE.

    Uses Powell's method (gradient-free, good for moderate dimensionality).

    Args:
        k_data: (n, d) key vectors
        d: dimension
        n_bits: quantization bits
        planes: list of (i, j) rotation planes
        max_iters: maximum optimization iterations

    Returns:
        best_angles: optimized angle vector
        best_R: optimized rotation matrix
        best_mse: resulting MSE
        opt_history: list of MSE values during optimization
    """
    opt_history = []

    def objective(angles):
        R = givens_rotation_matrix(d, planes, angles)
        mse = quantize_mse(k_data, R, n_bits)
        opt_history.append(mse)
        return mse

    # Initialize angles to 0 (identity rotation)
    x0 = np.zeros(len(planes))

    result = optimize.minimize(
        objective,
        x0,
        method="Powell",
        options={"maxiter": max_iters, "maxfev": max_iters * len(planes)},
    )

    best_angles = result.x
    best_R = givens_rotation_matrix(d, planes, best_angles)
    best_mse = result.fun

    return best_angles, best_R, best_mse, opt_history


def run_experiment():
    print(f"{'=' * 60}")
    print(f"FOKVQ Experiment 3-1: Continuous Rotation Optimization")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Bits: {QUANT_BITS}, Givens planes: {NUM_GIVENS_PLANES}")
    print(f"Layers to optimize: {LAYERS_TO_OPTIMIZE}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # ── Load model ─────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.n_layer
    n_embd = model.config.n_embd
    print(f"   Loaded in {time.time() - t0:.1f}s. L={num_layers}, d={n_embd}")

    # ── Extract K vectors ──────────────────────────────────────────
    print(f"\n[2/4] Extracting K vectors from {NUM_SAMPLES} samples...")
    expanded_texts = (TEXTS * ((NUM_SAMPLES // len(TEXTS)) + 1))[:NUM_SAMPLES]
    k_vectors = extract_k_per_layer(model, tokenizer, expanded_texts, DEVICE, MAX_SEQ_LEN)
    print(f"   Extracted. Tokens per layer: {k_vectors[0].shape[0]}")

    # Free model memory
    del model
    torch.cuda.empty_cache()
    print("   Model freed from GPU")

    # ── Compute baselines and optimize ─────────────────────────────
    print(f"\n[3/4] Computing baselines and optimizing rotations...")

    layer_results = {}
    all_improvements = []  # (improvement_pct, layer) for optimized vs PCA

    for l in LAYERS_TO_OPTIMIZE:
        print(f"\n   --- Layer {l} ---")
        k_data = k_vectors[l]

        # Subsample if too many tokens (for optimization speed)
        if k_data.shape[0] > 2000:
            idx = np.random.RandomState(42 + l).choice(
                k_data.shape[0], 2000, replace=False
            )
            k_sub = k_data[idx]
        else:
            k_sub = k_data

        # Baseline 1: Identity (no rotation)
        mse_identity = quantize_mse(k_sub, np.eye(n_embd), QUANT_BITS)
        print(f"   Identity MSE:     {mse_identity:.8f}")

        # Baseline 2: PCA rotation
        n_comp = min(n_embd, k_sub.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        pca.fit(k_sub)
        pca_basis = np.eye(n_embd)
        pca_basis[:n_comp, :] = pca.components_
        pca_R, _ = np.linalg.qr(pca_basis.T)  # (d, d) orthogonal
        mse_pca = quantize_mse(k_sub, pca_R, QUANT_BITS)
        print(f"   PCA MSE:          {mse_pca:.8f}")

        # Baseline 3: Random rotation
        random_R = np.linalg.qr(
            np.random.RandomState(42 + l).randn(n_embd, n_embd)
        )[0]
        mse_random = quantize_mse(k_sub, random_R, QUANT_BITS)
        print(f"   Random MSE:       {mse_random:.8f}")

        # Optimized rotation via Givens
        print(f"   Optimizing Givens rotation ({NUM_GIVENS_PLANES} planes, "
              f"max {MAX_OPT_ITERS} iters)...")
        planes = select_givens_planes(n_embd, NUM_GIVENS_PLANES, k_sub, seed=42 + l)

        opt_t0 = time.time()
        best_angles, best_R, mse_optimized, opt_history = optimize_rotation(
            k_sub, n_embd, QUANT_BITS, planes, max_iters=MAX_OPT_ITERS
        )
        opt_time = time.time() - opt_t0
        print(f"   Optimized MSE:    {mse_optimized:.8f} ({opt_time:.1f}s, "
              f"{len(opt_history)} evals)")

        # Improvement metrics
        imp_vs_identity = (mse_identity - mse_optimized) / (mse_identity + 1e-10) * 100
        imp_vs_pca = (mse_pca - mse_optimized) / (mse_pca + 1e-10) * 100
        imp_vs_random = (mse_random - mse_optimized) / (mse_random + 1e-10) * 100

        print(f"   Improvement vs identity: {imp_vs_identity:.2f}%")
        print(f"   Improvement vs PCA:      {imp_vs_pca:.2f}%")
        print(f"   Improvement vs random:   {imp_vs_random:.2f}%")

        layer_results[l] = {
            "mse_identity": float(mse_identity),
            "mse_pca": float(mse_pca),
            "mse_random": float(mse_random),
            "mse_optimized": float(mse_optimized),
            "improvement_vs_identity_pct": float(imp_vs_identity),
            "improvement_vs_pca_pct": float(imp_vs_pca),
            "improvement_vs_random_pct": float(imp_vs_random),
            "optimization_time_s": float(opt_time),
            "num_function_evals": len(opt_history),
            "opt_history_start": float(opt_history[0]) if opt_history else None,
            "opt_history_end": float(opt_history[-1]) if opt_history else None,
            "num_givens_planes": len(planes),
        }

        all_improvements.append(imp_vs_pca)

    # ── Aggregate results ──────────────────────────────────────────
    print(f"\n[4/4] Aggregating results...")

    layers_tested = list(layer_results.keys())
    mse_identity_all = [layer_results[l]["mse_identity"] for l in layers_tested]
    mse_pca_all = [layer_results[l]["mse_pca"] for l in layers_tested]
    mse_random_all = [layer_results[l]["mse_random"] for l in layers_tested]
    mse_optimized_all = [layer_results[l]["mse_optimized"] for l in layers_tested]

    # Paired t-tests
    t_opt_vs_pca, p_opt_vs_pca = stats.ttest_rel(mse_pca_all, mse_optimized_all)
    t_opt_vs_id, p_opt_vs_id = stats.ttest_rel(mse_identity_all, mse_optimized_all)
    t_pca_vs_rand, p_pca_vs_rand = stats.ttest_rel(mse_random_all, mse_pca_all)
    t_opt_vs_rand, p_opt_vs_rand = stats.ttest_rel(mse_random_all, mse_optimized_all)

    # Wilcoxon signed-rank (non-parametric)
    w_opt_vs_pca, wp_opt_vs_pca = stats.wilcoxon(
        mse_pca_all, mse_optimized_all, alternative="greater"
    )

    mean_imp = float(np.mean(all_improvements))

    print(f"\n{'=' * 60}")
    print(f"ROTATION OPTIMIZATION RESULTS (Lie Group / Givens)")
    print(f"{'=' * 60}")
    print(f"  Quantization: {QUANT_BITS}-bit, {NUM_GIVENS_PLANES} Givens planes")
    print(f"  Layers tested: {layers_tested}")
    print(f"\n  Method averages across layers:")
    print(f"    Identity MSE:   {np.mean(mse_identity_all):.8f}")
    print(f"    Random MSE:     {np.mean(mse_random_all):.8f}")
    print(f"    PCA MSE:        {np.mean(mse_pca_all):.8f}")
    print(f"    Optimized MSE:  {np.mean(mse_optimized_all):.8f}")
    print(f"\n  Mean improvement (optimized vs PCA): {mean_imp:.2f}%")
    print(f"\n  Statistical tests:")
    print(f"    Optimized < PCA:    t={t_opt_vs_pca:.3f}, p={p_opt_vs_pca:.6f}")
    print(f"    Optimized < Identity: t={t_opt_vs_id:.3f}, p={p_opt_vs_id:.6f}")
    print(f"    PCA < Random:       t={t_pca_vs_rand:.3f}, p={p_pca_vs_rand:.6f}")
    print(f"    Optimized < Random: t={t_opt_vs_rand:.3f}, p={p_opt_vs_rand:.6f}")
    print(f"    Wilcoxon (opt<PCA): W={w_opt_vs_pca:.1f}, p={wp_opt_vs_pca:.6f}")

    # Rank methods
    method_means = {
        "identity": np.mean(mse_identity_all),
        "random": np.mean(mse_random_all),
        "pca": np.mean(mse_pca_all),
        "optimized": np.mean(mse_optimized_all),
    }
    ranking = sorted(method_means.items(), key=lambda x: x[1])
    print(f"\n  Ranking (best to worst):")
    for i, (name, mse) in enumerate(ranking, 1):
        print(f"    {i}. {name}: {mse:.8f}")

    opt_is_best = ranking[0][0] == "optimized"
    opt_beats_pca = p_opt_vs_pca < 0.05 and np.mean(mse_optimized_all) < np.mean(mse_pca_all)

    print(f"\n  Optimized is best method: {'YES' if opt_is_best else 'NO'}")
    print(f"  Optimized significantly beats PCA: {'YES' if opt_beats_pca else 'NO'}")
    print(f"{'=' * 60}")

    # Per-layer table
    print(f"\n{'Layer':>6} {'Identity':>12} {'Random':>12} {'PCA':>12} {'Optimized':>12} {'Imp%':>8}")
    for l in layers_tested:
        lr = layer_results[l]
        print(f"{l:>6d} {lr['mse_identity']:>12.8f} {lr['mse_random']:>12.8f} "
              f"{lr['mse_pca']:>12.8f} {lr['mse_optimized']:>12.8f} "
              f"{lr['improvement_vs_pca_pct']:>8.2f}")

    # ── Save results ───────────────────────────────────────────────
    results = {
        "experiment": "3-1_continuous_rotation_optimization_lie_group",
        "model": MODEL_NAME,
        "device": DEVICE,
        "quant_bits": QUANT_BITS,
        "num_givens_planes": NUM_GIVENS_PLANES,
        "max_opt_iters": MAX_OPT_ITERS,
        "layers_tested": layers_tested,
        "num_layers_model": int(num_layers),
        "num_samples": NUM_SAMPLES,
        "per_layer": {str(l): layer_results[l] for l in layers_tested},
        "overall": {
            "mean_mse_identity": float(np.mean(mse_identity_all)),
            "mean_mse_random": float(np.mean(mse_random_all)),
            "mean_mse_pca": float(np.mean(mse_pca_all)),
            "mean_mse_optimized": float(np.mean(mse_optimized_all)),
            "mean_improvement_vs_pca_pct": mean_imp,
        },
        "statistical_tests": {
            "optimized_vs_pca": {
                "ttest_t": float(t_opt_vs_pca),
                "ttest_p": float(p_opt_vs_pca),
                "wilcoxon_W": float(w_opt_vs_pca),
                "wilcoxon_p": float(wp_opt_vs_pca),
            },
            "optimized_vs_identity": {
                "ttest_t": float(t_opt_vs_id),
                "ttest_p": float(p_opt_vs_id),
            },
            "pca_vs_random": {
                "ttest_t": float(t_pca_vs_rand),
                "ttest_p": float(p_pca_vs_rand),
            },
            "optimized_vs_random": {
                "ttest_t": float(t_opt_vs_rand),
                "ttest_p": float(p_opt_vs_rand),
            },
        },
        "ranking": [{"rank": i + 1, "method": name, "mse": float(mse)}
                     for i, (name, mse) in enumerate(ranking)],
        "optimized_is_best": bool(opt_is_best),
        "optimized_significantly_beats_pca": bool(opt_beats_pca),
        "runtime_seconds": time.time() - t0,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")
    print(f"Total runtime: {time.time() - t0:.1f}s")
    return results


if __name__ == "__main__":
    r = run_experiment()
    sys.exit(0)
