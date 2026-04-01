"""
FOKVQ Experiment 2-4: AFOD vs PCA vs Random Initialization
==========================================================
Compares three decomposition methods for selecting quantization axes
when performing non-uniform bit allocation on KV cache.

Methods:
1. PCA: Standard principal components from pooled calibration data
2. AFOD (Adaptive Facet-Oriented Decomposition):
   - Run PCA separately on multiple sub-corpora
   - Average the resulting subspaces via SVD of stacked bases
   - Intent: capture corpus-specific facets while generalizing
3. Random: Random orthogonal matrix (baseline)

For each method, quantize K at 2, 3, 4 bits using the decomposition axes
as the quantization coordinate system, then measure:
- MSE (reconstruction error of K vectors)
- KL divergence (attention distribution shift from original)

Hypothesis: AFOD provides better quantization axes than PCA alone,
            especially at very low bit widths (2-3 bits).
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

# ── Configuration ──────────────────────────────────────────────────
MODEL_NAME = "openai-community/gpt2-medium"
DEVICE = "cuda:1"
MAX_SEQ_LEN = 256
NUM_SAMPLES_PER_CORPUS = 30
BIT_WIDTHS = [2, 3, 4]
OUTPUT_DIR = Path("/mnt/input/fokvq/results/exp2_4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Three sub-corpora for AFOD facet estimation ───────────────────
TECHNICAL_TEXTS = [
    "The Boltzmann distribution assigns probability proportional to exp(-E/kT) where E is energy and T is temperature.",
    "Gradient descent minimizes a loss function by iteratively updating parameters in the direction of steepest descent.",
    "The transformer architecture uses multi-head self-attention to compute weighted sums of value vectors.",
    "Bayesian inference updates prior beliefs using likelihood functions to obtain posterior distributions over parameters.",
    "The Navier-Stokes equations describe the motion of viscous fluid substances through partial differential equations.",
    "Principal component analysis finds orthogonal directions of maximum variance in high-dimensional datasets.",
    "Quantum entanglement creates correlations between particles that cannot be explained by classical physics alone.",
    "The central limit theorem states that the sum of independent random variables converges to a normal distribution.",
    "Convolutional neural networks apply learned spatial filters to detect hierarchical patterns in image data.",
    "The Fourier transform decomposes a signal into constituent frequencies revealing its spectral content.",
    "Markov chain Monte Carlo methods generate samples from complex probability distributions using random walks.",
    "The attention mechanism computes compatibility scores between query and key vectors to weight value aggregation.",
    "Variational autoencoders learn latent representations by maximizing a lower bound on the data log-likelihood.",
    "The second law of thermodynamics states that entropy of an isolated system never decreases over time.",
    "Stochastic gradient descent approximates the true gradient using mini-batches of training examples.",
]

CODE_TEXTS = [
    "def forward(self, x): return self.linear(self.relu(self.norm(x)))",
    "for i in range(len(data)): result.append(process(data[i]))",
    "class DataLoader: def __init__(self, dataset, batch_size=32, shuffle=True): self.dataset = dataset",
    "import torch; model = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.ReLU())",
    "if __name__ == '__main__': parser = argparse.ArgumentParser(); args = parser.parse_args()",
    "async def fetch_data(url): async with aiohttp.ClientSession() as session: return await session.get(url)",
    "SELECT users.name, COUNT(orders.id) FROM users LEFT JOIN orders ON users.id = orders.user_id GROUP BY users.name",
    "git commit -m 'fix: resolve memory leak in attention cache by clearing hooks after forward pass'",
    "docker run --gpus all -v /data:/mnt/data -p 8080:8080 model-server:latest serve --model gpt2",
    "try: result = model.generate(input_ids, max_length=512) except RuntimeError as e: logger.error(e)",
    "def train_step(batch): loss = criterion(model(batch.input), batch.target); loss.backward(); optimizer.step()",
    "with torch.no_grad(): outputs = model(**inputs); logits = outputs.logits[:, -1, :]",
    "config = yaml.safe_load(open('config.yaml')); wandb.init(project=config['project'], config=config)",
    "np.random.seed(42); X_train, X_test = train_test_split(data, test_size=0.2, stratify=labels)",
    "lambda x: torch.cat([self.head(x) for head in self.heads], dim=-1)",
]

CONVERSATIONAL_TEXTS = [
    "Hey, did you see the game last night? It was absolutely incredible, they came back from a huge deficit.",
    "I'm thinking about getting a new phone. The latest models have such amazing cameras and battery life.",
    "So my friend told me about this restaurant downtown and we should definitely check it out sometime.",
    "The weather has been so unpredictable lately. One day it's sunny and the next it's pouring rain.",
    "I just finished reading this book and honestly it completely changed my perspective on life.",
    "Can you believe it's already March? This year is flying by so fast, I can barely keep up.",
    "My dog did the funniest thing today. She started chasing her tail and then fell over dizzy.",
    "I've been trying to cook more at home. Last night I made pasta from scratch and it was delicious.",
    "Do you have any plans for the weekend? I was thinking we could go hiking if the weather is nice.",
    "I saw this hilarious video online where a cat was trying to fit into a tiny box. So cute!",
    "My neighbor just got a new car and honestly I'm a little jealous. It looks really sleek.",
    "Remember when we used to stay up late playing video games? Those were the good old days.",
    "I need to start exercising more. Maybe I'll sign up for that gym that just opened nearby.",
    "The traffic this morning was absolutely terrible. It took me almost two hours to get to work.",
    "I love how coffee shops always smell so good. There's something comforting about that aroma.",
]

CORPORA = {
    "technical": TECHNICAL_TEXTS,
    "code": CODE_TEXTS,
    "conversational": CONVERSATIONAL_TEXTS,
}


def extract_qkv_per_layer(model, tokenizer, texts, device, max_seq_len):
    """
    Extract Q, K, V vectors per layer.

    Returns:
        dict: layer_idx -> {"q": np.ndarray (total_tokens, d),
                           "k": np.ndarray (total_tokens, d),
                           "v": np.ndarray (total_tokens, d)}
    """
    num_layers = model.config.n_layer
    layer_data = {l: {"q": [], "k": [], "v": []} for l in range(num_layers)}

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
            q, k, v = qkv.split(model.config.n_embd, dim=-1)

            layer_data[l]["q"].append(q.squeeze(0).detach().cpu().float().numpy())
            layer_data[l]["k"].append(k.squeeze(0).detach().cpu().float().numpy())
            layer_data[l]["v"].append(v.squeeze(0).detach().cpu().float().numpy())

        del inputs, outputs, hidden_states
        torch.cuda.empty_cache()

    # Concatenate
    for l in range(num_layers):
        for key in ["q", "k", "v"]:
            layer_data[l][key] = np.concatenate(layer_data[l][key], axis=0)

    return layer_data


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


def quantize_with_basis(k_vectors, basis_matrix, n_bits):
    """
    Quantize K vectors using a given orthogonal basis.

    1. Project to basis coordinates: coeffs = K @ basis
    2. Quantize coefficients
    3. Reconstruct: K_hat = coeffs_q @ basis^T

    Args:
        k_vectors: (n, d)
        basis_matrix: (d, d) orthogonal matrix (columns are basis vectors)
        n_bits: quantization bit width

    Returns:
        k_reconstructed: (n, d)
    """
    # Project to basis space
    coeffs = k_vectors @ basis_matrix  # (n, d)

    # Quantize in basis space
    coeffs_q = quantize_vector(coeffs, n_bits)

    # Reconstruct
    k_reconstructed = coeffs_q @ basis_matrix.T  # (n, d)
    return k_reconstructed


def compute_pca_basis(k_vectors):
    """
    Compute PCA basis from K vectors.

    Returns:
        basis: (d, d) orthogonal matrix, columns sorted by explained variance
    """
    d = k_vectors.shape[1]
    n_components = min(d, k_vectors.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(k_vectors)

    # Components are (n_components, d), each row is a direction
    # Pad to full d if needed
    basis = np.eye(d)
    basis[:n_components, :] = pca.components_
    # Ensure orthogonality via QR
    basis, _ = np.linalg.qr(basis.T)
    return basis  # (d, d)


def compute_afod_basis(corpus_k_vectors):
    """
    Adaptive Facet-Oriented Decomposition (AFOD).

    1. Run PCA on each sub-corpus separately
    2. Stack top-r principal components from each corpus
    3. SVD of stacked matrix to get averaged subspace
    4. Extend to full orthogonal basis via QR

    Args:
        corpus_k_vectors: dict of corpus_name -> (n, d) K vectors

    Returns:
        basis: (d, d) orthogonal matrix
    """
    d = list(corpus_k_vectors.values())[0].shape[1]
    r = min(32, d // 4)  # top-r components per corpus

    stacked_components = []
    for name, k_data in corpus_k_vectors.items():
        n_comp = min(r, k_data.shape[0] - 1, d)
        if n_comp < 1:
            continue
        pca = PCA(n_components=n_comp)
        pca.fit(k_data)
        stacked_components.append(pca.components_)  # (r, d)

    # Stack: (num_corpora * r, d)
    stacked = np.vstack(stacked_components)

    # SVD to find averaged subspace directions
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)

    # Vt rows are the averaged principal directions
    # Take all available directions and extend to full basis
    basis_partial = Vt[:min(d, Vt.shape[0]), :]  # (k, d) where k <= d

    # Extend to full orthogonal basis
    full_basis = np.eye(d)
    full_basis[:basis_partial.shape[0], :] = basis_partial
    full_basis, _ = np.linalg.qr(full_basis.T)
    return full_basis  # (d, d)


def compute_random_basis(d, seed=42):
    """
    Generate a random orthogonal matrix.

    Returns:
        basis: (d, d) orthogonal matrix
    """
    rng = np.random.RandomState(seed)
    A = rng.randn(d, d)
    Q, _ = np.linalg.qr(A)
    return Q  # (d, d)


def compute_kl_divergence(q_data, k_orig, k_quant):
    """
    Compute KL divergence between attention distributions.

    Args:
        q_data: (n_q, d) query vectors
        k_orig: (n_k, d) original keys
        k_quant: (n_k, d) quantized keys

    Returns:
        kl_mean: mean KL divergence across query positions
    """
    d = q_data.shape[1]
    scale = 1.0 / (d ** 0.5)

    # Sample queries for efficiency
    n_q = min(q_data.shape[0], 200)
    n_k = min(k_orig.shape[0], 200)

    q_sample = torch.from_numpy(q_data[:n_q]).float()
    k_orig_t = torch.from_numpy(k_orig[:n_k]).float()
    k_quant_t = torch.from_numpy(k_quant[:n_k]).float()

    # Attention distributions
    attn_orig = F.softmax(q_sample @ k_orig_t.T * scale, dim=-1)
    attn_quant = F.softmax(q_sample @ k_quant_t.T * scale, dim=-1)

    # KL(orig || quant)
    log_orig = torch.log(attn_orig.clamp(min=1e-10))
    log_quant = torch.log(attn_quant.clamp(min=1e-10))
    kl = (attn_orig * (log_orig - log_quant)).sum(dim=-1)

    return float(kl.mean().item())


def run_experiment():
    print(f"{'=' * 60}")
    print(f"FOKVQ Experiment 2-4: AFOD vs PCA vs Random Initialization")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
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

    # ── Extract per-corpus K vectors ───────────────────────────────
    print(f"\n[2/5] Extracting QKV from 3 corpora ({NUM_SAMPLES_PER_CORPUS} each)...")
    corpus_layer_data = {}
    for name, texts in CORPORA.items():
        expanded = (texts * ((NUM_SAMPLES_PER_CORPUS // len(texts)) + 1))[:NUM_SAMPLES_PER_CORPUS]
        print(f"   Extracting '{name}' ({len(expanded)} samples)...")
        corpus_layer_data[name] = extract_qkv_per_layer(
            model, tokenizer, expanded, DEVICE, MAX_SEQ_LEN
        )

    # ── Build pooled data ──────────────────────────────────────────
    print("\n[3/5] Building decomposition bases...")
    pooled_layer_data = {}
    for l in range(num_layers):
        pooled_layer_data[l] = {
            "q": np.concatenate([corpus_layer_data[c][l]["q"] for c in CORPORA], axis=0),
            "k": np.concatenate([corpus_layer_data[c][l]["k"] for c in CORPORA], axis=0),
            "v": np.concatenate([corpus_layer_data[c][l]["v"] for c in CORPORA], axis=0),
        }

    # ── Compute bases per layer ────────────────────────────────────
    print("   Computing PCA, AFOD, and Random bases per layer...")
    bases = {}  # layer -> {"pca": (d,d), "afod": (d,d), "random": (d,d)}
    for l in range(num_layers):
        k_pooled = pooled_layer_data[l]["k"]

        # PCA on pooled data
        pca_basis = compute_pca_basis(k_pooled)

        # AFOD: PCA per corpus, then average
        corpus_k = {
            name: corpus_layer_data[name][l]["k"] for name in CORPORA
        }
        afod_basis = compute_afod_basis(corpus_k)

        # Random orthogonal baseline
        random_basis = compute_random_basis(n_embd, seed=42 + l)

        bases[l] = {
            "pca": pca_basis,
            "afod": afod_basis,
            "random": random_basis,
        }

    # ── Quantize and measure ───────────────────────────────────────
    print(f"\n[4/5] Quantizing at {BIT_WIDTHS} bits and measuring quality...")
    methods = ["pca", "afod", "random", "identity"]

    results_table = {}  # method -> bits -> {"mse": [...], "kl": [...]}

    for method in methods:
        results_table[method] = {}
        for bits in BIT_WIDTHS:
            results_table[method][bits] = {"mse": [], "kl": []}

    for l in range(num_layers):
        if l % 6 == 0:
            print(f"   Layer {l}/{num_layers-1}...")

        q_data = pooled_layer_data[l]["q"]
        k_data = pooled_layer_data[l]["k"]

        for bits in BIT_WIDTHS:
            for method in methods:
                if method == "identity":
                    # Direct quantization without coordinate transform
                    k_quant = quantize_vector(k_data, bits)
                else:
                    basis = bases[l][method]
                    k_quant = quantize_with_basis(k_data, basis, bits)

                # MSE
                mse = float(np.mean((k_data - k_quant) ** 2))
                results_table[method][bits]["mse"].append(mse)

                # KL divergence
                kl = compute_kl_divergence(q_data, k_data, k_quant)
                results_table[method][bits]["kl"].append(kl)

    # ── Aggregate and compare ──────────────────────────────────────
    print(f"\n[5/5] Aggregating results...")

    summary = {}
    for method in methods:
        summary[method] = {}
        for bits in BIT_WIDTHS:
            mse_arr = np.array(results_table[method][bits]["mse"])
            kl_arr = np.array(results_table[method][bits]["kl"])
            summary[method][bits] = {
                "mse_mean": float(np.mean(mse_arr)),
                "mse_std": float(np.std(mse_arr)),
                "kl_mean": float(np.mean(kl_arr)),
                "kl_std": float(np.std(kl_arr)),
                "mse_per_layer": mse_arr.tolist(),
                "kl_per_layer": kl_arr.tolist(),
            }

    # Statistical comparisons: AFOD vs PCA (paired t-test per bit width)
    stat_tests = {}
    for bits in BIT_WIDTHS:
        afod_mse = np.array(results_table["afod"][bits]["mse"])
        pca_mse = np.array(results_table["pca"][bits]["mse"])
        rand_mse = np.array(results_table["random"][bits]["mse"])
        ident_mse = np.array(results_table["identity"][bits]["mse"])

        afod_kl = np.array(results_table["afod"][bits]["kl"])
        pca_kl = np.array(results_table["pca"][bits]["kl"])
        rand_kl = np.array(results_table["random"][bits]["kl"])

        # AFOD vs PCA (paired, two-sided)
        t_mse_ap, p_mse_ap = stats.ttest_rel(afod_mse, pca_mse)
        t_kl_ap, p_kl_ap = stats.ttest_rel(afod_kl, pca_kl)

        # PCA vs Random
        t_mse_pr, p_mse_pr = stats.ttest_rel(pca_mse, rand_mse)
        t_kl_pr, p_kl_pr = stats.ttest_rel(pca_kl, rand_kl)

        # PCA vs Identity (no transform)
        t_mse_pi, p_mse_pi = stats.ttest_rel(pca_mse, ident_mse)

        # AFOD relative improvement over PCA
        afod_improvement_mse = float(
            ((pca_mse - afod_mse) / (pca_mse + 1e-10)).mean() * 100
        )
        afod_improvement_kl = float(
            ((pca_kl - afod_kl) / (pca_kl + 1e-10)).mean() * 100
        )

        stat_tests[bits] = {
            "afod_vs_pca": {
                "mse_t": float(t_mse_ap), "mse_p": float(p_mse_ap),
                "kl_t": float(t_kl_ap), "kl_p": float(p_kl_ap),
                "afod_improvement_mse_pct": afod_improvement_mse,
                "afod_improvement_kl_pct": afod_improvement_kl,
            },
            "pca_vs_random": {
                "mse_t": float(t_mse_pr), "mse_p": float(p_mse_pr),
                "kl_t": float(t_kl_pr), "kl_p": float(p_kl_pr),
            },
            "pca_vs_identity": {
                "mse_t": float(t_mse_pi), "mse_p": float(p_mse_pi),
            },
        }

    # ── Print summary ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"AFOD vs PCA vs RANDOM vs IDENTITY RESULTS")
    print(f"{'=' * 70}")

    for bits in BIT_WIDTHS:
        print(f"\n--- {bits}-bit quantization ---")
        print(f"  {'Method':>12}  {'MSE':>12}  {'KL':>12}")
        for method in methods:
            s = summary[method][bits]
            print(f"  {method:>12}  {s['mse_mean']:>12.6f}  {s['kl_mean']:>12.6f}")

        st = stat_tests[bits]
        print(f"  AFOD vs PCA:    MSE p={st['afod_vs_pca']['mse_p']:.4f}, "
              f"KL p={st['afod_vs_pca']['kl_p']:.4f}, "
              f"MSE improvement={st['afod_vs_pca']['afod_improvement_mse_pct']:.2f}%")
        print(f"  PCA vs Random:  MSE p={st['pca_vs_random']['mse_p']:.4f}, "
              f"KL p={st['pca_vs_random']['kl_p']:.4f}")

    # Best method per bit width
    print(f"\n--- Best method by MSE ---")
    for bits in BIT_WIDTHS:
        best = min(methods, key=lambda m: summary[m][bits]["mse_mean"])
        print(f"  {bits}-bit: {best} (MSE={summary[best][bits]['mse_mean']:.6f})")

    # Determine overall findings
    afod_wins = sum(
        1 for bits in BIT_WIDTHS
        if summary["afod"][bits]["mse_mean"] < summary["pca"][bits]["mse_mean"]
    )
    pca_beats_random = sum(
        1 for bits in BIT_WIDTHS
        if summary["pca"][bits]["mse_mean"] < summary["random"][bits]["mse_mean"]
    )

    print(f"\n  AFOD beats PCA in {afod_wins}/{len(BIT_WIDTHS)} bit widths (MSE)")
    print(f"  PCA beats Random in {pca_beats_random}/{len(BIT_WIDTHS)} bit widths (MSE)")
    print(f"{'=' * 70}")

    # ── Save results ───────────────────────────────────────────────
    results = {
        "experiment": "2-4_afod_vs_pca_vs_random_initialization",
        "model": MODEL_NAME,
        "device": DEVICE,
        "num_layers": int(num_layers),
        "num_samples_per_corpus": NUM_SAMPLES_PER_CORPUS,
        "bit_widths": BIT_WIDTHS,
        "methods": methods,
        "summary": summary,
        "statistical_tests": stat_tests,
        "afod_wins_mse": afod_wins,
        "pca_beats_random_mse": pca_beats_random,
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
