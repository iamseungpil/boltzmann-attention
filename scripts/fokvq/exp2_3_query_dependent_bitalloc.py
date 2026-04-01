"""
FOKVQ Experiment 2-3: Query-Dependent vs Static Bit Allocation
==============================================================
Tests whether optimal bit allocation for KV cache quantization
depends on the query being processed.

Method:
1. Load GPT-2 Medium
2. Extract Q, K per layer from calibration data
3. For each query q_i, compute per-DIMENSION importance:
   - Query-dependent: sensitivity of attention output to K perturbation
     along each dimension d, weighted by the specific query
     I_d(q) = sum_j alpha_j * (q_d * k_{j,d})^2 / d
   - This captures which dimensions of K matter most for THIS query
4. Static criterion: per-dimension variance of K (Var_j[k_{j,d}])
5. Compare: does the per-dimension importance ranking change across queries?
   Measured via Spearman rank correlation and L1 allocation distance.

Hypothesis: Different queries emphasize different K dimensions,
            so optimal per-dimension bit allocation is query-dependent.
"""

import json
import sys
import time
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

# ── Configuration ──────────────────────────────────────────────────
MODEL_NAME = "openai-community/gpt2-medium"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 256
NUM_SAMPLES = 60
OUTPUT_DIR = Path("/mnt/input/fokvq/results/exp2_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Calibration texts (longer for more positions) ─────────────────
TEXTS = [
    "The Boltzmann distribution assigns probability proportional to exp(-E/kT) where E is energy and T is temperature. This fundamental relation from statistical mechanics governs the behavior of particles in thermal equilibrium. At higher temperatures, particles can access higher energy states more readily.",
    "Gradient descent minimizes a loss function by iteratively updating parameters in the direction of steepest descent. The learning rate controls the step size. Too large a rate causes divergence while too small a rate leads to slow convergence. Adaptive methods like Adam adjust rates per parameter.",
    "The transformer architecture uses multi-head self-attention to compute weighted sums of value vectors. Each head operates on a different learned projection of the input, allowing the model to attend to information from different representation subspaces at different positions.",
    "Bayesian inference updates prior beliefs using likelihood functions to obtain posterior distributions over parameters. The posterior is proportional to the likelihood times the prior. Markov chain Monte Carlo methods approximate the posterior when exact computation is intractable.",
    "Principal component analysis finds orthogonal directions of maximum variance in high-dimensional datasets. The first principal component captures the most variance. Subsequent components are orthogonal to all previous ones. PCA is equivalent to eigendecomposition of the covariance matrix.",
    "def forward(self, x): out = self.norm(x); out = self.relu(out); out = self.linear(out); return out + x  # residual connection ensures gradient flow through the network layers",
    "for i in range(len(data)): result.append(process(data[i])); if i % 100 == 0: logger.info(f'Processed {i}/{len(data)} items'); checkpoint(result, i)",
    "class DataLoader: def __init__(self, dataset, batch_size=32, shuffle=True): self.dataset = dataset; self.batch_size = batch_size; self.shuffle = shuffle; self.index = 0",
    "import torch; model = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))",
    "Hey, did you see the game last night? It was absolutely incredible, they came back from a huge deficit in the fourth quarter. The crowd was going wild and even the commentators were speechless. I still cannot believe that final shot.",
    "I'm thinking about getting a new phone. The latest models have such amazing cameras and battery life. But I can't decide between the two brands. My friend says one is better for photos while the other has a better screen and faster processor.",
    "So my friend told me about this restaurant downtown and we should definitely check it out sometime. They have this amazing fusion menu that combines Japanese and Italian cuisine. The reviews say the pasta is incredible and the sushi is fresh.",
    "The weather has been so unpredictable lately. One day it's sunny and the next it's pouring rain. I had to cancel my outdoor plans three times this week. Maybe we should just plan indoor activities for now until spring fully arrives.",
    "The Fourier transform decomposes a signal into constituent frequencies revealing its spectral content. The discrete version enables efficient computation via FFT. Applications range from signal processing to solving differential equations and image compression.",
    "Variational autoencoders learn latent representations by maximizing a lower bound on the data log-likelihood. The encoder maps data to a distribution in latent space while the decoder reconstructs data from latent samples. The KL term regularizes the latent space.",
    "Quantum entanglement creates correlations between particles that cannot be explained by classical physics alone. When two particles are entangled, measuring one instantly determines the state of the other regardless of distance. This has applications in quantum computing and cryptography.",
    "SELECT users.name, COUNT(orders.id) as order_count, SUM(orders.total) as total_spent FROM users LEFT JOIN orders ON users.id = orders.user_id WHERE orders.date > '2024-01-01' GROUP BY users.name HAVING order_count > 5 ORDER BY total_spent DESC",
    "docker run --gpus all -v /data:/mnt/data -p 8080:8080 --env CUDA_VISIBLE_DEVICES=0 --env MODEL_PATH=/mnt/data/weights model-server:latest serve --model gpt2-medium --batch-size 32 --max-length 2048",
    "I just finished reading this book and honestly it completely changed my perspective on life. The author argues that happiness comes not from external achievements but from cultivating inner peace. The chapter on mindfulness was particularly enlightening and practical.",
    "The central limit theorem states that the sum of independent random variables converges to a normal distribution. This holds regardless of the underlying distribution provided it has finite variance. The theorem is the foundation for many statistical inference procedures.",
]


def extract_qk_per_layer(model, tokenizer, texts, device, max_seq_len):
    """
    Extract Q, K vectors per layer for all texts.

    Returns:
        list of dicts: [{layer_idx: {"q": (seq, d), "k": (seq, d)}}, ...]
    """
    num_layers = model.config.n_layer
    samples = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        sample = {}

        for l in range(num_layers):
            h = hidden_states[l]
            attn = model.transformer.h[l].attn
            qkv = attn.c_attn(h)
            q, k, _ = qkv.split(model.config.n_embd, dim=-1)

            sample[l] = {
                "q": q.squeeze(0).detach().cpu().float().numpy(),
                "k": k.squeeze(0).detach().cpu().float().numpy(),
            }

        del inputs, outputs, hidden_states
        torch.cuda.empty_cache()
        samples.append(sample)

    return samples


def compute_query_dependent_dim_importance(q_vec, k_matrix):
    """
    Compute query-dependent per-DIMENSION importance for the key cache.

    For query q and keys K, the attention output is:
        o = sum_j alpha_j * v_j    where alpha_j = softmax(q^T k_j / sqrt(d))

    The sensitivity of attention scores to perturbation of dimension d is:
        I_d(q) = sum_j alpha_j * (q_d * k_{j,d})^2

    This captures how important each dimension of K is for forming the
    attention distribution given this specific query.

    Args:
        q_vec: (d,) single query vector
        k_matrix: (n_keys, d) key matrix

    Returns:
        dim_importance: (d,) importance per dimension
    """
    d = q_vec.shape[0]
    scale = 1.0 / math.sqrt(d)

    # Attention weights
    scores = k_matrix @ q_vec * scale  # (n_keys,)
    scores = scores - scores.max()  # numerical stability
    exp_scores = np.exp(scores)
    alpha = exp_scores / (exp_scores.sum() + 1e-10)  # (n_keys,)

    # Per-dimension importance: attention-weighted squared contribution
    # For each dim d: I_d = sum_j alpha_j * q_d^2 * k_{j,d}^2
    # Simplify: I_d = q_d^2 * sum_j alpha_j * k_{j,d}^2
    q_sq = q_vec ** 2  # (d,)
    k_sq_weighted = alpha[:, None] * (k_matrix ** 2)  # (n_keys, d)
    k_sq_mean = k_sq_weighted.sum(axis=0)  # (d,)

    dim_importance = q_sq * k_sq_mean  # (d,)

    return dim_importance


def compute_static_dim_importance(k_matrix):
    """
    Compute static (query-independent) per-dimension importance.
    Uses variance of K along each dimension.

    Args:
        k_matrix: (n_keys, d)

    Returns:
        dim_importance: (d,)
    """
    return np.var(k_matrix, axis=0)  # (d,)


def compute_allocation_l1(imp_a, imp_b):
    """
    Compute normalized L1 distance between two importance-based allocations.

    Returns value in [0, 2] where 0 = identical allocation, 2 = maximally different.
    """
    # Normalize to probability distributions
    p_a = imp_a / (imp_a.sum() + 1e-10)
    p_b = imp_b / (imp_b.sum() + 1e-10)
    return float(np.abs(p_a - p_b).sum())


def run_experiment():
    print(f"{'=' * 60}")
    print(f"FOKVQ Experiment 2-3: Query-Dependent vs Static Bit Allocation")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
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

    # ── Extract QK data ────────────────────────────────────────────
    print(f"\n[2/4] Extracting Q, K from {NUM_SAMPLES} samples...")
    expanded_texts = (TEXTS * ((NUM_SAMPLES // len(TEXTS)) + 1))[:NUM_SAMPLES]
    samples = extract_qk_per_layer(model, tokenizer, expanded_texts, DEVICE, MAX_SEQ_LEN)
    print(f"   Extracted {len(samples)} samples")

    # Free model
    del model
    torch.cuda.empty_cache()

    # ── Per-layer analysis ─────────────────────────────────────────
    print("\n[3/4] Computing query-dependent vs static dimension importance...")

    layer_results = {}
    all_static_vs_qdep_rhos = []
    all_alloc_l1s = []
    all_query_vs_query_rhos = []
    all_query_vs_query_l1s = []

    for l in range(num_layers):
        print(f"   Layer {l}/{num_layers-1}...", end="", flush=True)

        static_vs_qdep_rhos = []
        alloc_l1s = []
        query_vs_query_rhos = []
        query_vs_query_l1s = []

        for s_idx in range(min(len(samples), 20)):
            q_data = samples[s_idx][l]["q"]  # (seq, d)
            k_data = samples[s_idx][l]["k"]  # (seq, d)
            seq_len = q_data.shape[0]

            if seq_len < 16:
                continue

            # Static dimension importance (same for all queries in this sample)
            static_imp = compute_static_dim_importance(k_data)  # (d,)

            # Pick query positions spread across the sequence
            query_positions = np.linspace(
                seq_len // 4, seq_len - 1, min(8, seq_len // 4), dtype=int
            )
            qdep_imps = []

            for q_pos in query_positions:
                q_vec = q_data[q_pos]
                # Causal: only keys up to q_pos
                k_causal = k_data[:q_pos + 1]
                if k_causal.shape[0] < 4:
                    continue

                qdep_imp = compute_query_dependent_dim_importance(q_vec, k_causal)

                # Check for degenerate cases
                if np.std(qdep_imp) < 1e-12 or np.std(static_imp) < 1e-12:
                    continue

                # Spearman rank correlation between static and query-dependent
                rho, _ = stats.spearmanr(static_imp, qdep_imp)
                if not np.isnan(rho):
                    static_vs_qdep_rhos.append(rho)

                # Allocation L1 distance
                l1 = compute_allocation_l1(qdep_imp, static_imp)
                alloc_l1s.append(l1)

                qdep_imps.append(qdep_imp)

            # Compare query-dependent allocations across DIFFERENT queries
            # (same key cache, different queries)
            if len(qdep_imps) >= 2:
                for i in range(len(qdep_imps)):
                    for j in range(i + 1, len(qdep_imps)):
                        if (np.std(qdep_imps[i]) < 1e-12 or
                                np.std(qdep_imps[j]) < 1e-12):
                            continue
                        rho_qq, _ = stats.spearmanr(qdep_imps[i], qdep_imps[j])
                        if not np.isnan(rho_qq):
                            query_vs_query_rhos.append(rho_qq)
                        l1_qq = compute_allocation_l1(qdep_imps[i], qdep_imps[j])
                        query_vs_query_l1s.append(l1_qq)

        # Aggregate for this layer
        def safe_stats(arr):
            if len(arr) == 0:
                return 0.0, 0.0, 0
            return float(np.mean(arr)), float(np.std(arr)), len(arr)

        rho_mean, rho_std, rho_n = safe_stats(static_vs_qdep_rhos)
        l1_mean, l1_std, l1_n = safe_stats(alloc_l1s)
        qq_rho_mean, qq_rho_std, qq_n = safe_stats(query_vs_query_rhos)
        qq_l1_mean, qq_l1_std, qq_l1_n = safe_stats(query_vs_query_l1s)

        layer_results[l] = {
            "static_vs_qdep_rank_corr_mean": rho_mean,
            "static_vs_qdep_rank_corr_std": rho_std,
            "alloc_l1_distance_mean": l1_mean,
            "alloc_l1_distance_std": l1_std,
            "query_vs_query_rank_corr_mean": qq_rho_mean,
            "query_vs_query_rank_corr_std": qq_rho_std,
            "query_vs_query_l1_mean": qq_l1_mean,
            "query_vs_query_l1_std": qq_l1_std,
            "n_static_vs_qdep": rho_n,
            "n_query_vs_query": qq_n,
        }

        all_static_vs_qdep_rhos.extend(static_vs_qdep_rhos)
        all_alloc_l1s.extend(alloc_l1s)
        all_query_vs_query_rhos.extend(query_vs_query_rhos)
        all_query_vs_query_l1s.extend(query_vs_query_l1s)

        print(f" rho={rho_mean:.3f}(n={rho_n}), L1={l1_mean:.4f}, "
              f"qq_rho={qq_rho_mean:.3f}(n={qq_n})")

    # ── Statistical tests ──────────────────────────────────────────
    print("\n[4/4] Statistical analysis...")

    rho_arr = np.array(all_static_vs_qdep_rhos)
    l1_arr = np.array(all_alloc_l1s)
    qq_arr = np.array(all_query_vs_query_rhos)
    qq_l1_arr = np.array(all_query_vs_query_l1s)

    # Test 1: Static vs query-dependent rank correlation < 1.0?
    if len(rho_arr) > 1:
        t_rho, p_rho = stats.ttest_1samp(rho_arr, 1.0)
    else:
        t_rho, p_rho = 0.0, 1.0

    # Test 2: Allocation L1 > 0?
    if len(l1_arr) > 1:
        t_l1, p_l1 = stats.ttest_1samp(l1_arr, 0.0)
    else:
        t_l1, p_l1 = 0.0, 1.0

    # Test 3: Query-vs-query correlation < 1.0?
    if len(qq_arr) > 1:
        t_qq, p_qq = stats.ttest_1samp(qq_arr, 1.0)
    else:
        t_qq, p_qq = 0.0, 1.0

    # Test 4: Query-vs-query L1 > 0?
    if len(qq_l1_arr) > 1:
        t_qq_l1, p_qq_l1 = stats.ttest_1samp(qq_l1_arr, 0.0)
    else:
        t_qq_l1, p_qq_l1 = 0.0, 1.0

    overall_rho = float(np.mean(rho_arr)) if len(rho_arr) > 0 else 0.0
    overall_rho_std = float(np.std(rho_arr)) if len(rho_arr) > 0 else 0.0
    overall_l1 = float(np.mean(l1_arr)) if len(l1_arr) > 0 else 0.0
    overall_l1_std = float(np.std(l1_arr)) if len(l1_arr) > 0 else 0.0
    overall_qq = float(np.mean(qq_arr)) if len(qq_arr) > 0 else 0.0
    overall_qq_std = float(np.std(qq_arr)) if len(qq_arr) > 0 else 0.0
    overall_qq_l1 = float(np.mean(qq_l1_arr)) if len(qq_l1_arr) > 0 else 0.0
    overall_qq_l1_std = float(np.std(qq_l1_arr)) if len(qq_l1_arr) > 0 else 0.0

    # Hypothesis tests
    qdep_differs_from_static = (
        overall_rho < 0.95
        and p_rho < 0.05
        and overall_l1 > 0.01
        and p_l1 < 0.05
    )

    alloc_varies_across_queries = (
        overall_qq < 0.95 and p_qq < 0.05
    )

    print(f"\n{'=' * 70}")
    print(f"QUERY-DEPENDENT vs STATIC BIT ALLOCATION RESULTS")
    print(f"(Per-dimension importance analysis)")
    print(f"{'=' * 70}")
    print(f"  Static vs Q-dep rank correlation: {overall_rho:.4f} +/- {overall_rho_std:.4f} (n={len(rho_arr)})")
    print(f"  Alloc L1 distance (static vs Qdep): {overall_l1:.4f} +/- {overall_l1_std:.4f}")
    print(f"  Query-vs-query rank correlation:     {overall_qq:.4f} +/- {overall_qq_std:.4f} (n={len(qq_arr)})")
    print(f"  Query-vs-query L1 distance:          {overall_qq_l1:.4f} +/- {overall_qq_l1_std:.4f}")
    print(f"\n  t-test (rho < 1):    t={t_rho:.3f}, p={p_rho:.2e}")
    print(f"  t-test (L1 > 0):     t={t_l1:.3f}, p={p_l1:.2e}")
    print(f"  t-test (qq rho < 1): t={t_qq:.3f}, p={p_qq:.2e}")
    print(f"  t-test (qq L1 > 0):  t={t_qq_l1:.3f}, p={p_qq_l1:.2e}")
    print(f"\n  Q-dep allocation differs from static:     {'YES' if qdep_differs_from_static else 'NO'}")
    print(f"  Allocation varies across queries:          {'YES' if alloc_varies_across_queries else 'NO'}")
    print(f"{'=' * 70}")

    # Per-layer table
    print(f"\n{'Layer':>6} {'StatVsQdep':>12} {'AllocL1':>10} {'QvsQ_rho':>10} {'QvsQ_L1':>10} {'N':>5}")
    for l in range(num_layers):
        lr = layer_results[l]
        print(f"{l:>6d} {lr['static_vs_qdep_rank_corr_mean']:>12.4f} "
              f"{lr['alloc_l1_distance_mean']:>10.4f} "
              f"{lr['query_vs_query_rank_corr_mean']:>10.4f} "
              f"{lr['query_vs_query_l1_mean']:>10.4f} "
              f"{lr['n_static_vs_qdep']:>5d}")

    # ── Save results ───────────────────────────────────────────────
    results = {
        "experiment": "2-3_query_dependent_vs_static_bit_allocation",
        "model": MODEL_NAME,
        "device": DEVICE,
        "num_layers": int(num_layers),
        "n_embd": int(n_embd),
        "num_samples": NUM_SAMPLES,
        "max_seq_len": MAX_SEQ_LEN,
        "analysis_type": "per_dimension_importance",
        "overall": {
            "static_vs_qdep_rank_corr_mean": overall_rho,
            "static_vs_qdep_rank_corr_std": overall_rho_std,
            "alloc_l1_distance_mean": overall_l1,
            "alloc_l1_distance_std": overall_l1_std,
            "query_vs_query_rank_corr_mean": overall_qq,
            "query_vs_query_rank_corr_std": overall_qq_std,
            "query_vs_query_l1_mean": overall_qq_l1,
            "query_vs_query_l1_std": overall_qq_l1_std,
        },
        "statistical_tests": {
            "rho_lt_1": {
                "t_stat": float(t_rho),
                "p_value": float(p_rho),
                "n": len(all_static_vs_qdep_rhos),
            },
            "l1_gt_0": {
                "t_stat": float(t_l1),
                "p_value": float(p_l1),
                "n": len(all_alloc_l1s),
            },
            "qq_rho_lt_1": {
                "t_stat": float(t_qq),
                "p_value": float(p_qq),
                "n": len(all_query_vs_query_rhos),
            },
            "qq_l1_gt_0": {
                "t_stat": float(t_qq_l1),
                "p_value": float(p_qq_l1),
                "n": len(all_query_vs_query_l1s),
            },
        },
        "query_dependent_allocation_differs": bool(qdep_differs_from_static),
        "allocation_varies_across_queries": bool(alloc_varies_across_queries),
        "per_layer": {
            str(l): layer_results[l] for l in range(num_layers)
        },
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
