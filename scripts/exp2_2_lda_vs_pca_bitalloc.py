"""
FOKVQ Experiment 2-2: LDA vs PCA Bit Allocation
================================================
Compares non-uniform bit allocation quality for KV cache quantization
using PCA-based vs LDA-based axis selection.

Method:
1. Load GPT-2 Medium
2. Extract KV vectors from diverse text samples
3. Apply PCA and LDA decompositions to K and V vectors
4. Simulate non-uniform bit allocation:
   - Higher bits for top principal/discriminant axes
   - Lower bits for remaining axes
5. Measure reconstruction quality: MSE and KL divergence on attention output
6. Compare: does LDA-based axis selection outperform PCA for quantization?

Hypothesis: LDA-based axis selection preserves more task-relevant information,
            resulting in lower MSE and KL divergence after quantization.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ── Configuration ──────────────────────────────────────────────────
MODEL_NAME = "openai-community/gpt2-medium"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 256
NUM_SAMPLES = 80
BIT_CONFIGS = [
    {"name": "uniform_4bit", "high_bits": 4, "low_bits": 4, "top_k": 0},
    {"name": "pca_6_2", "high_bits": 6, "low_bits": 2, "top_k": 16},
    {"name": "pca_8_2", "high_bits": 8, "low_bits": 2, "top_k": 8},
    {"name": "lda_6_2", "high_bits": 6, "low_bits": 2, "top_k": 16},
    {"name": "lda_8_2", "high_bits": 8, "low_bits": 2, "top_k": 8},
]
OUTPUT_DIR = Path("/mnt/input/fokvq/results/exp2_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Diverse text samples ──────────────────────────────────────────
TEXTS = [
    "The Boltzmann distribution assigns probability proportional to exp(-E/kT) where E is energy.",
    "def forward(self, x): return self.linear(self.relu(self.norm(x)))",
    "Hey, did you see the game last night? It was absolutely incredible.",
    "Gradient descent minimizes a loss function by iteratively updating parameters.",
    "for i in range(len(data)): result.append(process(data[i]))",
    "I'm thinking about getting a new phone. The latest models are amazing.",
    "The transformer uses multi-head self-attention to compute weighted sums of value vectors.",
    "class DataLoader: def __init__(self, dataset, batch_size=32): self.dataset = dataset",
    "The weather has been so unpredictable lately. One day sunny, next it's raining.",
    "Principal component analysis finds orthogonal directions of maximum variance.",
    "import torch; model = torch.nn.Sequential(torch.nn.Linear(512, 256))",
    "My dog did the funniest thing today. She chased her tail and fell over.",
    "Quantum entanglement creates correlations that cannot be explained classically.",
    "try: result = model.generate(input_ids) except RuntimeError as e: log(e)",
    "Can you believe it's already March? This year is flying by so fast.",
    "The Fourier transform decomposes a signal into constituent frequencies.",
    "SELECT name, COUNT(id) FROM users LEFT JOIN orders GROUP BY name",
    "I just finished reading this book and it completely changed my perspective.",
    "Variational autoencoders learn latent representations by maximizing ELBO.",
    "docker run --gpus all -v /data:/mnt model-server:latest serve --model gpt2",
]

# Labels for LDA: 0=technical, 1=code, 2=conversational
TEXT_LABELS = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]


def quantize_vector(v, n_bits):
    """
    Simulate uniform quantization to n_bits.
    Maps values to 2^n_bits levels within [min, max] range.
    """
    if n_bits >= 16:
        return v.copy()
    n_levels = 2 ** n_bits
    v_min = v.min(axis=0, keepdims=True)
    v_max = v.max(axis=0, keepdims=True)
    v_range = v_max - v_min
    v_range = np.where(v_range < 1e-10, 1.0, v_range)

    # Normalize to [0, 1]
    v_norm = (v - v_min) / v_range
    # Quantize
    v_quant = np.round(v_norm * (n_levels - 1)) / (n_levels - 1)
    # De-normalize
    v_deq = v_quant * v_range + v_min
    return v_deq


def nonuniform_quantize(vectors, transform_matrix, high_bits, low_bits, top_k):
    """
    Non-uniform bit allocation using a given transform.

    1. Project vectors into transform space
    2. Quantize top-k axes with high_bits, rest with low_bits
    3. Inverse-project back to original space

    Args:
        vectors: (n, d) array of vectors
        transform_matrix: (d, d) orthogonal transform (columns = axes)
        high_bits: bits for top-k axes
        low_bits: bits for remaining axes
        top_k: number of axes to allocate high bits

    Returns:
        reconstructed: (n, d) quantized and reconstructed vectors
    """
    if top_k == 0:
        # Uniform quantization: no transform needed
        return quantize_vector(vectors, high_bits)

    # Project to transform space
    coeffs = vectors @ transform_matrix  # (n, d)

    # Split into high-importance and low-importance
    coeffs_high = coeffs[:, :top_k]
    coeffs_low = coeffs[:, top_k:]

    # Quantize separately
    coeffs_high_q = quantize_vector(coeffs_high, high_bits)
    coeffs_low_q = quantize_vector(coeffs_low, low_bits)

    # Recombine
    coeffs_q = np.concatenate([coeffs_high_q, coeffs_low_q], axis=1)

    # Inverse project
    reconstructed = coeffs_q @ transform_matrix.T
    return reconstructed


def compute_attention_kl(q, k_orig, k_quant, v_orig, v_quant):
    """
    Compute KL divergence between attention distributions
    using original vs quantized KV.

    Args:
        q, k_orig, k_quant, v_orig, v_quant: torch tensors (seq, d)

    Returns:
        kl_div: scalar KL divergence
        output_mse: MSE of attention output
    """
    d = q.shape[-1]
    scale = 1.0 / (d ** 0.5)

    # Original attention
    attn_orig = F.softmax(q @ k_orig.T * scale, dim=-1)
    out_orig = attn_orig @ v_orig

    # Quantized attention
    attn_quant = F.softmax(q @ k_quant.T * scale, dim=-1)
    out_quant = attn_quant @ v_quant

    # KL divergence: KL(orig || quant)
    log_orig = torch.log(attn_orig.clamp(min=1e-10))
    log_quant = torch.log(attn_quant.clamp(min=1e-10))
    kl = (attn_orig * (log_orig - log_quant)).sum(dim=-1).mean().item()

    # Output MSE
    mse = ((out_orig - out_quant) ** 2).mean().item()

    return kl, mse


def extract_qkv_per_layer(model, tokenizer, texts, device, max_seq_len):
    """
    Extract Q, K, V vectors per layer.

    Returns:
        dict: layer_idx -> (q_list, k_list, v_list) each list of np.ndarray (seq, d)
    """
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads

    layer_data = {l: {"q": [], "k": [], "v": []} for l in range(num_layers)}

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        for l in range(num_layers):
            h = hidden_states[l]  # (1, seq, n_embd)
            attn = model.transformer.h[l].attn
            qkv = attn.c_attn(h)
            q, k, v = qkv.split(model.config.n_embd, dim=-1)

            layer_data[l]["q"].append(q.squeeze(0).detach().cpu().float().numpy())
            layer_data[l]["k"].append(k.squeeze(0).detach().cpu().float().numpy())
            layer_data[l]["v"].append(v.squeeze(0).detach().cpu().float().numpy())

        del inputs, outputs, hidden_states
        torch.cuda.empty_cache()

    return layer_data


def run_experiment():
    print(f"{'=' * 60}")
    print(f"FOKVQ Experiment 2-2: LDA vs PCA Bit Allocation")
    print(f"Model: {MODEL_NAME}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # ── Load model ─────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
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

    # ── Extract QKV ────────────────────────────────────────────────
    print(f"\n[2/5] Extracting QKV from {NUM_SAMPLES} samples...")
    expanded_texts = (TEXTS * ((NUM_SAMPLES // len(TEXTS)) + 1))[:NUM_SAMPLES]
    expanded_labels = (TEXT_LABELS * ((NUM_SAMPLES // len(TEXT_LABELS)) + 1))[:NUM_SAMPLES]

    layer_data = extract_qkv_per_layer(model, tokenizer, expanded_texts, DEVICE, MAX_SEQ_LEN)

    # ── Compute PCA and LDA transforms per layer ──────────────────
    print("\n[3/5] Computing PCA and LDA transforms...")

    pca_transforms = {}
    lda_transforms = {}

    for l in range(num_layers):
        # Concatenate all K vectors for this layer
        k_all = np.concatenate(layer_data[l]["k"], axis=0)  # (total_tokens, d)
        n_tokens = k_all.shape[0]

        # Use SVD directly to get a full orthonormal basis
        # Center the data first (PCA = SVD of centered data)
        k_mean = k_all.mean(axis=0, keepdims=True)
        k_centered = k_all - k_mean

        # SVD: k_centered = U @ diag(S) @ Vt, columns of V are PCs
        # Use min(n_tokens, n_embd) components
        n_pca = min(n_tokens, n_embd)
        U_svd, S_svd, Vt_svd = np.linalg.svd(k_centered, full_matrices=False)
        # Vt_svd: (n_pca, d), rows are principal components
        pca_components = Vt_svd[:n_pca]  # (n_pca, d)
        explained_var_ratio = (S_svd[:n_pca] ** 2) / (S_svd ** 2).sum()

        # Build full orthonormal transform matrix (d, d)
        # If n_pca < d, fill remaining with random orthogonal complement
        if n_pca < n_embd:
            # Start with PCA components and complete the basis
            Q, _ = np.linalg.qr(pca_components.T)  # (d, n_pca)
            # Generate complement via QR of random matrix projected to null space
            null_proj = np.eye(n_embd) - Q @ Q.T
            rand_mat = np.random.randn(n_embd, n_embd - n_pca)
            complement = null_proj @ rand_mat
            Q_comp, _ = np.linalg.qr(complement)
            pca_full = np.concatenate([Q, Q_comp], axis=1)  # (d, d)
        else:
            pca_full = pca_components.T  # (d, d)

        pca_transforms[l] = pca_full

        # LDA transform:
        # Create per-token labels based on which sample they came from
        token_labels = []
        for i, k_arr in enumerate(layer_data[l]["k"]):
            label = expanded_labels[i]
            token_labels.extend([label] * k_arr.shape[0])
        token_labels = np.array(token_labels)

        # LDA gives at most (n_classes - 1) discriminant axes
        # We combine LDA axes (top discriminant) + PCA axes (remaining)
        n_classes = len(set(token_labels))
        n_lda_components = min(n_classes - 1, n_embd)

        lda = LinearDiscriminantAnalysis(n_components=n_lda_components)
        lda.fit(k_all, token_labels)

        # LDA scalings: (d, n_lda_components)
        lda_axes = lda.scalings_[:, :n_lda_components]
        # Orthonormalize LDA axes
        lda_axes_orth, _ = np.linalg.qr(lda_axes)  # (d, n_lda_components)

        # Fill remaining axes with PCA components orthogonal to LDA
        # Project out LDA subspace from PCA components
        remaining_pca = pca_full.copy()  # (d, d)
        proj = lda_axes_orth @ lda_axes_orth.T
        remaining_pca = remaining_pca - proj @ remaining_pca
        # Re-orthonormalize
        U_rem, S_rem, Vt_rem = np.linalg.svd(remaining_pca, full_matrices=False)
        # Take top (d - n_lda) axes
        n_remaining = n_embd - n_lda_components
        remaining_axes = U_rem[:, :n_remaining]

        # Combine: [LDA axes | PCA remainder]
        lda_full = np.concatenate([lda_axes_orth, remaining_axes], axis=1)  # (d, d)
        lda_transforms[l] = lda_full

        if l % 6 == 0:
            print(f"   Layer {l}/{num_layers}: n_tokens={n_tokens}, "
                  f"PCA var explained (top-8)={sum(explained_var_ratio[:8]):.3f}, "
                  f"LDA components={n_lda_components}")

    # ── Quantize and evaluate ──────────────────────────────────────
    print("\n[4/5] Evaluating bit allocation strategies...")

    results_per_config = {}

    for config in BIT_CONFIGS:
        name = config["name"]
        high_bits = config["high_bits"]
        low_bits = config["low_bits"]
        top_k = config["top_k"]

        is_lda = name.startswith("lda")
        is_pca = name.startswith("pca")

        layer_mse_k = []
        layer_mse_v = []
        layer_kl = []
        layer_attn_mse = []

        for l in range(num_layers):
            k_all = np.concatenate(layer_data[l]["k"], axis=0)
            v_all = np.concatenate(layer_data[l]["v"], axis=0)
            q_all = np.concatenate(layer_data[l]["q"], axis=0)

            # Select transform
            if is_lda:
                transform = lda_transforms[l]
            elif is_pca:
                transform = pca_transforms[l]
            else:
                transform = np.eye(n_embd)

            # Quantize K and V
            k_quant = nonuniform_quantize(k_all, transform, high_bits, low_bits, top_k)
            v_quant = nonuniform_quantize(v_all, transform, high_bits, low_bits, top_k)

            # K reconstruction MSE
            k_mse = float(np.mean((k_all - k_quant) ** 2))
            v_mse = float(np.mean((v_all - v_quant) ** 2))
            layer_mse_k.append(k_mse)
            layer_mse_v.append(v_mse)

            # Attention-level metrics (sample a subset for efficiency)
            sample_size = min(128, q_all.shape[0])
            idx = np.random.choice(q_all.shape[0], sample_size, replace=False)

            q_t = torch.from_numpy(q_all[idx]).float()
            k_orig_t = torch.from_numpy(k_all[idx]).float()
            k_quant_t = torch.from_numpy(k_quant[idx]).float()
            v_orig_t = torch.from_numpy(v_all[idx]).float()
            v_quant_t = torch.from_numpy(v_quant[idx]).float()

            kl, attn_mse = compute_attention_kl(q_t, k_orig_t, k_quant_t, v_orig_t, v_quant_t)
            layer_kl.append(kl)
            layer_attn_mse.append(attn_mse)

        results_per_config[name] = {
            "config": config,
            "k_mse_per_layer": layer_mse_k,
            "v_mse_per_layer": layer_mse_v,
            "kl_per_layer": layer_kl,
            "attn_output_mse_per_layer": layer_attn_mse,
            "mean_k_mse": float(np.mean(layer_mse_k)),
            "mean_v_mse": float(np.mean(layer_mse_v)),
            "mean_kl": float(np.mean(layer_kl)),
            "mean_attn_mse": float(np.mean(layer_attn_mse)),
        }

        print(f"   {name}: K_MSE={np.mean(layer_mse_k):.6f}, "
              f"V_MSE={np.mean(layer_mse_v):.6f}, "
              f"KL={np.mean(layer_kl):.6f}, "
              f"AttnMSE={np.mean(layer_attn_mse):.6f}")

    # ── Analysis ───────────────────────────────────────────────────
    print("\n[5/5] Analyzing LDA vs PCA comparison...")

    # Compare LDA vs PCA at same bit budget
    comparisons = [
        ("pca_6_2", "lda_6_2", "6/2 bit"),
        ("pca_8_2", "lda_8_2", "8/2 bit"),
    ]

    comparison_results = []
    for pca_name, lda_name, label in comparisons:
        pca_r = results_per_config[pca_name]
        lda_r = results_per_config[lda_name]

        # Paired t-test per layer: is LDA KL lower than PCA KL?
        t_kl, p_kl = stats.ttest_rel(pca_r["kl_per_layer"], lda_r["kl_per_layer"])
        t_mse, p_mse = stats.ttest_rel(pca_r["k_mse_per_layer"], lda_r["k_mse_per_layer"])

        kl_improvement = (pca_r["mean_kl"] - lda_r["mean_kl"]) / max(pca_r["mean_kl"], 1e-10)
        mse_improvement = (pca_r["mean_k_mse"] - lda_r["mean_k_mse"]) / max(pca_r["mean_k_mse"], 1e-10)

        lda_wins_kl = lda_r["mean_kl"] < pca_r["mean_kl"] and p_kl < 0.05
        lda_wins_mse = lda_r["mean_k_mse"] < pca_r["mean_k_mse"] and p_mse < 0.05

        comp = {
            "label": label,
            "pca_config": pca_name,
            "lda_config": lda_name,
            "pca_mean_kl": pca_r["mean_kl"],
            "lda_mean_kl": lda_r["mean_kl"],
            "kl_improvement_pct": float(kl_improvement * 100),
            "kl_ttest_t": float(t_kl),
            "kl_ttest_p": float(p_kl),
            "lda_wins_kl": bool(lda_wins_kl),
            "pca_mean_k_mse": pca_r["mean_k_mse"],
            "lda_mean_k_mse": lda_r["mean_k_mse"],
            "mse_improvement_pct": float(mse_improvement * 100),
            "mse_ttest_t": float(t_mse),
            "mse_ttest_p": float(p_mse),
            "lda_wins_mse": bool(lda_wins_mse),
        }
        comparison_results.append(comp)

        print(f"\n  {label} comparison:")
        print(f"    KL:  PCA={pca_r['mean_kl']:.6f}, LDA={lda_r['mean_kl']:.6f}, "
              f"improvement={kl_improvement*100:.2f}%, p={p_kl:.4f}")
        print(f"    MSE: PCA={pca_r['mean_k_mse']:.6f}, LDA={lda_r['mean_k_mse']:.6f}, "
              f"improvement={mse_improvement*100:.2f}%, p={p_mse:.4f}")
        print(f"    LDA wins: KL={'YES' if lda_wins_kl else 'NO'}, "
              f"MSE={'YES' if lda_wins_mse else 'NO'}")

    # Overall verdict
    lda_overall_better = any(c["lda_wins_kl"] or c["lda_wins_mse"] for c in comparison_results)

    print(f"\n{'=' * 60}")
    print(f"OVERALL: LDA-based allocation {'OUTPERFORMS' if lda_overall_better else 'does NOT outperform'} PCA")
    print(f"{'=' * 60}")

    # ── Save results ───────────────────────────────────────────────
    results = {
        "experiment": "2-2_lda_vs_pca_bit_allocation",
        "model": MODEL_NAME,
        "num_layers": int(num_layers),
        "n_embd": int(n_embd),
        "num_samples": NUM_SAMPLES,
        "bit_configs": BIT_CONFIGS,
        "per_config_results": {
            name: {
                "config": r["config"],
                "mean_k_mse": r["mean_k_mse"],
                "mean_v_mse": r["mean_v_mse"],
                "mean_kl": r["mean_kl"],
                "mean_attn_mse": r["mean_attn_mse"],
                "k_mse_per_layer": r["k_mse_per_layer"],
                "v_mse_per_layer": r["v_mse_per_layer"],
                "kl_per_layer": r["kl_per_layer"],
                "attn_output_mse_per_layer": r["attn_output_mse_per_layer"],
            }
            for name, r in results_per_config.items()
        },
        "comparisons": comparison_results,
        "lda_outperforms_pca": bool(lda_overall_better),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")
    return results


if __name__ == "__main__":
    r = run_experiment()
    sys.exit(0)
