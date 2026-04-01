"""
FOKVQ Experiment 2-1: Facet K-space Separability
=================================================
Tests whether K-space in attention has corpus-dependent facets.

Method:
1. Load GPT-2 Medium
2. Extract KV cache vectors from 3 text corpora (technical, code, conversational)
3. Compute PCA on K vectors per layer, get top-r principal components
4. Measure cosine similarity between PCA subspaces across corpora pairs
5. If subspaces differ significantly, K-space has corpus-dependent facets

Hypothesis: K-space principal subspaces differ across corpora (low cross-corpus
            subspace similarity), indicating faceted structure.
"""

import json
import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.decomposition import PCA

# ── Configuration ──────────────────────────────────────────────────
MODEL_NAME = "openai-community/gpt2-medium"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 256
TOP_R = 8           # Number of principal components to compare
NUM_SAMPLES = 30    # Samples per corpus
OUTPUT_DIR = Path("/mnt/input/fokvq/results/exp2_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Corpora ────────────────────────────────────────────────────────
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


def extract_k_vectors(model, tokenizer, texts, device, max_seq_len):
    """
    Extract K (key) vectors from all layers for a list of texts.

    Returns:
        dict: layer_idx -> np.ndarray of shape (total_tokens, hidden_dim)
    """
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads

    k_vectors = {l: [] for l in range(num_layers)}

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract K from each layer's attention
        # GPT-2: hidden_states[l] -> attn.c_attn -> split into Q,K,V
        hidden_states = outputs.hidden_states  # (num_layers+1,) tuple

        for l in range(num_layers):
            h = hidden_states[l]  # (1, seq_len, n_embd) — input to layer l
            # Compute K using the layer's attention projection
            attn = model.transformer.h[l].attn
            qkv = attn.c_attn(h)
            _, k, _ = qkv.split(model.config.n_embd, dim=-1)
            # k: (1, seq_len, n_embd)
            k_np = k.squeeze(0).detach().cpu().float().numpy()  # (seq_len, n_embd)
            k_vectors[l].append(k_np)

        del inputs, outputs, hidden_states
        torch.cuda.empty_cache()

    # Concatenate all tokens per layer
    for l in range(num_layers):
        k_vectors[l] = np.concatenate(k_vectors[l], axis=0)

    return k_vectors


def subspace_similarity(U1, U2):
    """
    Compute subspace similarity between two orthonormal bases.

    Uses the principal angles between subspaces:
        sim = (1/r) * sum(sigma_i^2)
    where sigma_i are singular values of U1^T @ U2.

    Returns a value in [0, 1]: 1 = identical subspaces, 0 = orthogonal.
    """
    # U1, U2: (d, r) orthonormal columns
    M = U1.T @ U2  # (r, r)
    sigmas = np.linalg.svd(M, compute_uv=False)
    # Clamp to [0,1] for numerical stability
    sigmas = np.clip(sigmas, 0.0, 1.0)
    return float(np.mean(sigmas ** 2))


def run_experiment():
    print(f"{'=' * 60}")
    print(f"FOKVQ Experiment 2-1: Facet K-space Separability")
    print(f"Model: {MODEL_NAME}")
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
    print(f"   Loaded in {time.time() - t0:.1f}s. L={num_layers}")

    # ── Extract K vectors per corpus ───────────────────────────────
    print("\n[2/4] Extracting K vectors from 3 corpora...")
    corpus_k = {}
    for name, texts in CORPORA.items():
        # Repeat texts to get enough samples
        expanded = (texts * ((NUM_SAMPLES // len(texts)) + 1))[:NUM_SAMPLES]
        print(f"   Extracting from '{name}' ({len(expanded)} samples)...")
        corpus_k[name] = extract_k_vectors(model, tokenizer, expanded, DEVICE, MAX_SEQ_LEN)

    # ── PCA per layer per corpus ───────────────────────────────────
    print("\n[3/4] Computing PCA subspaces and cross-corpus similarity...")
    corpus_names = list(CORPORA.keys())
    pairs = [
        (corpus_names[0], corpus_names[1]),
        (corpus_names[0], corpus_names[2]),
        (corpus_names[1], corpus_names[2]),
    ]

    # Store PCA components per corpus per layer
    pca_bases = {name: {} for name in corpus_names}
    explained_var = {name: {} for name in corpus_names}

    for name in corpus_names:
        for l in range(num_layers):
            k_data = corpus_k[name][l]  # (n_tokens, d)
            pca = PCA(n_components=TOP_R)
            pca.fit(k_data)
            pca_bases[name][l] = pca.components_.T  # (d, r) — columns are PCs
            explained_var[name][l] = pca.explained_variance_ratio_.tolist()

    # ── Subspace similarity ────────────────────────────────────────
    pair_similarities = {f"{a}_vs_{b}": [] for a, b in pairs}

    for l in range(num_layers):
        for a, b in pairs:
            U_a = pca_bases[a][l]  # (d, r)
            U_b = pca_bases[b][l]
            sim = subspace_similarity(U_a, U_b)
            pair_similarities[f"{a}_vs_{b}"].append(sim)

    # ── Within-corpus baseline (split-half) ────────────────────────
    print("   Computing within-corpus split-half baseline...")
    within_similarities = []
    for name in corpus_names:
        layer_sims = []
        for l in range(num_layers):
            k_data = corpus_k[name][l]
            n = len(k_data)
            idx = np.random.permutation(n)
            half = n // 2
            pca_a = PCA(n_components=TOP_R).fit(k_data[idx[:half]])
            pca_b = PCA(n_components=TOP_R).fit(k_data[idx[half:2*half]])
            sim = subspace_similarity(pca_a.components_.T, pca_b.components_.T)
            layer_sims.append(sim)
        within_similarities.append(layer_sims)

    within_mean = np.mean(within_similarities, axis=0).tolist()  # per-layer average

    # ── Analysis ───────────────────────────────────────────────────
    print("\n[4/4] Analyzing results...")

    # Average cross-corpus similarity per layer
    cross_sims = np.array([pair_similarities[k] for k in pair_similarities])  # (3, L)
    cross_mean = cross_sims.mean(axis=0).tolist()  # per-layer

    # Statistical test: are cross-corpus similarities lower than within-corpus?
    cross_flat = cross_sims.flatten()
    within_flat = np.array(within_similarities).flatten()

    t_stat, p_value = stats.ttest_ind(within_flat, cross_flat, alternative="greater")
    mw_stat, mw_p = stats.mannwhitneyu(within_flat, cross_flat, alternative="greater")

    overall_cross_mean = float(np.mean(cross_flat))
    overall_within_mean = float(np.mean(within_flat))
    separability = overall_within_mean - overall_cross_mean
    hypothesis_supported = separability > 0.02 and p_value < 0.05

    print(f"\n{'=' * 60}")
    print(f"FACET K-SPACE SEPARABILITY RESULTS")
    print(f"  Mean within-corpus subspace sim:  {overall_within_mean:.4f}")
    print(f"  Mean cross-corpus subspace sim:   {overall_cross_mean:.4f}")
    print(f"  Separability gap:                 {separability:.4f}")
    print(f"  t-test (within > cross):  t={t_stat:.3f}, p={p_value:.4f}")
    print(f"  Mann-Whitney:             U={mw_stat:.1f}, p={mw_p:.4f}")
    print(f"  Hypothesis supported:     {'YES' if hypothesis_supported else 'NO'}")
    print(f"{'=' * 60}")

    # Per-pair breakdown
    for pair_name, sims in pair_similarities.items():
        print(f"  {pair_name}: mean={np.mean(sims):.4f}, "
              f"min={np.min(sims):.4f}, max={np.max(sims):.4f}")

    # Per-layer details
    print(f"\n{'Layer':>6} {'Within':>10} {'Cross':>10} {'Gap':>10}")
    for l in range(num_layers):
        w = within_mean[l]
        c = cross_mean[l]
        print(f"{l:>6d} {w:>10.4f} {c:>10.4f} {w - c:>10.4f}")

    # ── Save results ───────────────────────────────────────────────
    results = {
        "experiment": "2-1_facet_k_space_separability",
        "model": MODEL_NAME,
        "num_layers": int(num_layers),
        "top_r": TOP_R,
        "num_samples_per_corpus": NUM_SAMPLES,
        "corpora": corpus_names,
        "pair_similarities": {k: v for k, v in pair_similarities.items()},
        "within_corpus_similarity": within_mean,
        "cross_corpus_similarity_mean": cross_mean,
        "overall_within_mean": overall_within_mean,
        "overall_cross_mean": overall_cross_mean,
        "separability_gap": float(separability),
        "ttest_stat": float(t_stat),
        "ttest_p_value": float(p_value),
        "mannwhitney_stat": float(mw_stat),
        "mannwhitney_p_value": float(mw_p),
        "hypothesis_supported": bool(hypothesis_supported),
        "explained_variance": {
            name: {str(l): explained_var[name][l] for l in range(num_layers)}
            for name in corpus_names
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")
    return results


if __name__ == "__main__":
    r = run_experiment()
    sys.exit(0)
