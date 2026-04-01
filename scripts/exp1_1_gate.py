"""
Experiment 1-1: Boltzmann T_eff Gate Test
=========================================
Uses forward hooks to capture Q, K directly from attention layers,
computing Boltzmann quantities manually (avoids sliding window issues).

Gate criteria:
  Strong success: ρ < -0.7, p < 0.01
  Weak success:   -0.7 < ρ < -0.3
  Failure:        ρ > -0.3
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 512
NUM_SAMPLES = 64
BATCH_SIZE = 4
OUTPUT_DIR = Path("/mnt/input/boltzmann/results/exp1_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_TEXTS = [
    "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow models to attend to all positions simultaneously.",
    "In statistical mechanics, the Boltzmann distribution describes the probability of a system being in a certain state as a function of energy and temperature.",
    "Quantum computing leverages superposition and entanglement to perform computations that would be intractable for classical computers.",
    "The human genome contains approximately three billion base pairs encoding around twenty thousand protein-coding genes across twenty-three chromosome pairs.",
    "Climate change models predict rising sea levels and more frequent extreme weather events as global temperatures continue to increase.",
    "Deep learning has achieved remarkable success in computer vision, speech recognition, natural language understanding, and strategic game playing.",
    "The standard model of particle physics describes electromagnetic, weak, and strong nuclear forces but does not incorporate gravity.",
    "Protein folding is one of the grand challenges in computational biology, with AlphaFold representing a breakthrough in structure prediction.",
    "Reinforcement learning from human feedback has become the dominant paradigm for aligning large language models with human preferences.",
    "The attention mechanism computes a weighted sum of values where weights are determined by the compatibility of queries with keys.",
    "Information theory quantifies the fundamental limits of data compression and reliable communication over noisy channels.",
    "Bayesian inference provides a principled framework for updating beliefs about model parameters given observed data evidence.",
    "Graph neural networks extend deep learning to non-Euclidean domains by operating on graph-structured data representations.",
    "The central dogma of molecular biology describes the flow of genetic information from DNA to RNA to protein synthesis.",
    "Variational autoencoders combine neural networks with variational inference to learn latent representations of complex data.",
    "Monte Carlo methods use random sampling to obtain numerical results for problems that may be deterministic in principle.",
]


class AttentionHook:
    """Hooks into attention layers to capture Q, K, V after projection and RoPE."""

    def __init__(self, model):
        self.captured = {}  # layer_idx -> (q, k)
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        # Qwen2: model.model.layers[i].self_attn
        for layer_idx, layer in enumerate(model.model.layers):
            attn_module = layer.self_attn
            hook = attn_module.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # For Qwen2Attention, we hook the forward and compute attention ourselves
            # We intercept the output tuple: (attn_output, attn_weights, past_key_value)
            # But attn_weights might be None. Instead, re-derive from hidden states.
            pass
        return hook_fn

    def remove(self):
        for h in self.hooks:
            h.remove()


def compute_boltzmann_quantities(model, inputs, num_layers, num_heads, head_dim):
    """Compute attention-based Boltzmann quantities by manually extracting Q,K."""
    device = inputs["input_ids"].device
    batch_size, seq_len = inputs["input_ids"].shape

    # Storage for per-layer results
    layer_entropy = []
    layer_teff = []

    # Use hooks to capture Q and K from each layer
    qk_store = {}

    def make_qk_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            hidden = args[0] if args else kwargs.get('hidden_states')
            if hidden is None:
                return

            # Get Q, K projections
            q = module.q_proj(hidden)
            k = module.k_proj(hidden)

            # Reshape to (batch, heads, seq, head_dim)
            bsz, slen, _ = q.shape
            q = q.view(bsz, slen, num_heads, head_dim).transpose(1, 2)
            k_heads = module.config.num_key_value_heads if hasattr(module, 'config') else getattr(module, 'num_key_value_heads', num_heads)
            k = k.view(bsz, slen, k_heads, head_dim).transpose(1, 2)

            # Repeat KV heads if GQA
            if k_heads != num_heads:
                repeat_factor = num_heads // k_heads
                k = k.repeat_interleave(repeat_factor, dim=1)

            qk_store[layer_idx] = (q.detach(), k.detach())

        return hook_fn

    # Register hooks
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        h = attn.register_forward_hook(make_qk_hook(layer_idx), with_kwargs=True)
        hooks.append(h)

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute Boltzmann quantities from captured Q, K
    results_per_layer = []
    sqrt_d = np.sqrt(head_dim)

    for layer_idx in range(num_layers):
        if layer_idx not in qk_store:
            results_per_layer.append({
                'entropy': np.full((batch_size, num_heads), np.nan),
                'teff': np.full((batch_size, num_heads), np.nan),
            })
            continue

        q, k = qk_store[layer_idx]  # (batch, heads, seq, head_dim)

        # Compute energy: E_j = q^T k_j / sqrt(d)
        # attention scores: (batch, heads, seq_q, seq_k)
        energy = torch.matmul(q, k.transpose(-2, -1)) / sqrt_d

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        energy = energy.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Boltzmann weights (softmax = Boltzmann distribution)
        alpha = F.softmax(energy.float(), dim=-1)  # (batch, heads, seq, seq)

        # Shannon entropy: S = -Σ α_j log α_j
        log_alpha = torch.where(
            alpha > 1e-10,
            alpha.log(),
            torch.zeros_like(alpha)
        )
        entropy = -(alpha * log_alpha).sum(dim=-1)  # (batch, heads, seq)

        # Effective temperature: T_eff = S / log(n)
        # For causal attention, effective n varies per position
        position_indices = torch.arange(1, seq_len + 1, device=device).float()
        log_n = position_indices.log().unsqueeze(0).unsqueeze(0)  # (1, 1, seq)
        log_n = log_n.clamp(min=1e-6)  # avoid div by 0 for position 0

        teff = entropy / log_n  # (batch, heads, seq)

        # Average T_eff over valid token positions (using attention mask)
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].unsqueeze(1).float()  # (batch, 1, seq)
            # Skip first position (log(1)=0 causes issues)
            mask[:, :, 0] = 0
            teff_avg = (teff * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # (batch, heads)
            entropy_avg = (entropy * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        else:
            teff_avg = teff[:, :, 1:].mean(dim=-1)
            entropy_avg = entropy[:, :, 1:].mean(dim=-1)

        results_per_layer.append({
            'entropy': entropy_avg.cpu().numpy(),  # (batch, heads)
            'teff': teff_avg.cpu().numpy(),
        })

        del q, k, energy, alpha, entropy, teff
        torch.cuda.empty_cache()

    del qk_store
    return results_per_layer


def run_experiment():
    print(f"{'='*60}")
    print(f"Experiment 1-1: Boltzmann T_eff Gate Test")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    # ── Load model ──────────────────────────────────────────────────────
    t0 = time.time()
    print(f"\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    print(f"   Loaded in {time.time()-t0:.1f}s")
    print(f"   Layers={num_layers}, Heads={num_heads}, KV_Heads={num_kv_heads}, HeadDim={head_dim}")

    # ── Prepare calibration data ────────────────────────────────────────
    print(f"\n[2/4] Preparing calibration data ({NUM_SAMPLES} samples)...")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = (CALIBRATION_TEXTS * ((NUM_SAMPLES // len(CALIBRATION_TEXTS)) + 1))[:NUM_SAMPLES]

    # ── Forward pass and collect Boltzmann quantities ───────────────────
    print(f"\n[3/4] Running forward passes with Q,K hooks...")

    # Accumulate per-layer, per-head T_eff
    layer_head_teff_accum = np.zeros((num_layers, num_heads))
    layer_head_entropy_accum = np.zeros((num_layers, num_heads))
    sample_count = 0

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(DEVICE)

        results_per_layer = compute_boltzmann_quantities(
            model, inputs, num_layers, num_heads, head_dim
        )

        for layer_idx, res in enumerate(results_per_layer):
            # res['teff']: (batch, heads)
            batch_mean_teff = np.nanmean(res['teff'], axis=0)  # (heads,)
            batch_mean_entropy = np.nanmean(res['entropy'], axis=0)
            layer_head_teff_accum[layer_idx] += batch_mean_teff * len(batch_texts)
            layer_head_entropy_accum[layer_idx] += batch_mean_entropy * len(batch_texts)

        sample_count += len(batch_texts)
        del inputs, results_per_layer
        torch.cuda.empty_cache()

        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        if batch_num % 4 == 1 or batch_num == total_batches:
            print(f"   Batch {batch_num}/{total_batches}")

    layer_head_teff = layer_head_teff_accum / sample_count
    layer_head_entropy = layer_head_entropy_accum / sample_count

    # ── Analysis ────────────────────────────────────────────────────────
    print(f"\n[4/4] Analyzing results...")

    layer_teff_mean = np.nanmean(layer_head_teff, axis=1)
    layer_teff_std = np.nanstd(layer_head_teff, axis=1)
    layer_entropy_mean = np.nanmean(layer_head_entropy, axis=1)

    # Sanity check
    valid_layers = ~np.isnan(layer_teff_mean)
    print(f"   Valid layers: {valid_layers.sum()}/{num_layers}")
    if valid_layers.sum() > 0:
        print(f"   T_eff range: [{np.nanmin(layer_teff_mean):.4f}, {np.nanmax(layer_teff_mean):.4f}]")
        print(f"   Entropy range: [{np.nanmin(layer_entropy_mean):.4f}, {np.nanmax(layer_entropy_mean):.4f}]")

    # Spearman correlation on valid layers
    layer_indices = np.arange(num_layers)
    valid_idx = layer_indices[valid_layers]
    valid_teff = layer_teff_mean[valid_layers]

    if len(valid_teff) >= 3:
        rho, p_value = stats.spearmanr(valid_idx, valid_teff)
    else:
        rho, p_value = np.nan, np.nan

    head_var_per_layer = np.nanvar(layer_head_teff, axis=1)

    # ── Gate Decision ───────────────────────────────────────────────────
    if np.isnan(rho):
        gate_decision = "ERROR"
        gate_msg = "Insufficient valid data for correlation."
    elif rho < -0.7 and p_value < 0.01:
        gate_decision = "STRONG_SUCCESS"
        gate_msg = f"Strong success: ρ={rho:.4f}, p={p_value:.2e}. Proceed to L1 full + L2."
    elif -0.7 <= rho < -0.3:
        gate_decision = "WEAK_SUCCESS"
        gate_msg = f"Weak success: ρ={rho:.4f}, p={p_value:.2e}. Proceed to 1-2, 1-3 only."
    else:
        gate_decision = "FAILURE"
        gate_msg = f"Failure: ρ={rho:.4f}, p={p_value:.2e}. Framework needs reassessment."

    print(f"\n{'='*60}")
    print(f"GATE DECISION: {gate_decision}")
    print(f"  Spearman ρ(layer, T_eff) = {rho:.4f}" if not np.isnan(rho) else "  Spearman ρ = NaN")
    print(f"  p-value = {p_value:.2e}" if not np.isnan(p_value) else "  p-value = NaN")
    print(f"  {gate_msg}")
    print(f"{'='*60}")

    # ── Save Results ────────────────────────────────────────────────────
    results = {
        "experiment": "1-1_gate_teff",
        "model": MODEL_NAME,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "num_kv_heads": int(num_kv_heads),
        "num_samples": int(sample_count),
        "max_seq_len": MAX_SEQ_LEN,
        "spearman_rho": float(rho) if not np.isnan(rho) else None,
        "spearman_p_value": float(p_value) if not np.isnan(p_value) else None,
        "gate_decision": gate_decision,
        "gate_message": gate_msg,
        "layer_teff_mean": [float(x) if not np.isnan(x) else None for x in layer_teff_mean],
        "layer_teff_std": [float(x) if not np.isnan(x) else None for x in layer_teff_std],
        "layer_entropy_mean": [float(x) if not np.isnan(x) else None for x in layer_entropy_mean],
        "head_var_per_layer": [float(x) if not np.isnan(x) else None for x in head_var_per_layer],
        "layer_head_teff": [[float(x) if not np.isnan(x) else None for x in row] for row in layer_head_teff],
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Visualization ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp 1-1: Boltzmann T_eff Gate Test ({MODEL_NAME})\n"
                 f"ρ={rho:.4f}, p={p_value:.2e} → {gate_decision}" if not np.isnan(rho)
                 else f"Exp 1-1: Boltzmann T_eff Gate Test ({MODEL_NAME})",
                 fontsize=13, fontweight='bold')

    # (a) T_eff vs layer index
    ax = axes[0, 0]
    valid = ~np.isnan(layer_teff_mean)
    ax.errorbar(layer_indices[valid], layer_teff_mean[valid],
                yerr=layer_teff_std[valid], fmt='o-', capsize=3,
                color='steelblue', markersize=4)
    if not np.isnan(rho):
        # Add trend line
        z = np.polyfit(valid_idx, valid_teff, 1)
        ax.plot(layer_indices, np.polyval(z, layer_indices), '--', color='red', alpha=0.5,
                label=f'Linear fit (slope={z[0]:.4f})')
        ax.legend(fontsize=8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("T_eff (mean ± std over heads)")
    ax.set_title(f"Progressive Cooling: T_eff vs Layer")

    # (b) Heatmap: layer × head
    ax = axes[0, 1]
    heatmap_data = np.where(np.isnan(layer_head_teff), 0, layer_head_teff)
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title("T_eff Heatmap (Layer × Head)")
    plt.colorbar(im, ax=ax, label="T_eff")

    # (c) Head variance per layer (functional differentiation)
    ax = axes[1, 0]
    valid_var = ~np.isnan(head_var_per_layer)
    ax.bar(layer_indices[valid_var], head_var_per_layer[valid_var], color='coral', alpha=0.7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Var_h[T_eff]")
    ax.set_title("Head Temperature Variance (Functional Differentiation)")

    # (d) Entropy vs layer
    ax = axes[1, 1]
    valid_ent = ~np.isnan(layer_entropy_mean)
    ax.plot(layer_indices[valid_ent], layer_entropy_mean[valid_ent], 'o-',
            color='darkgreen', markersize=4)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Mean Entropy S")
    ax.set_title("Attention Entropy vs Layer")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "exp1_1_teff_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figures saved to {fig_path}")

    # Print summary table
    print(f"\n--- Layer T_eff Summary ---")
    print(f"{'Layer':>6} {'T_eff':>10} {'Std':>10} {'Entropy':>10} {'HeadVar':>10}")
    for l in range(num_layers):
        te = f"{layer_teff_mean[l]:.4f}" if not np.isnan(layer_teff_mean[l]) else "NaN"
        ts = f"{layer_teff_std[l]:.4f}" if not np.isnan(layer_teff_std[l]) else "NaN"
        en = f"{layer_entropy_mean[l]:.4f}" if not np.isnan(layer_entropy_mean[l]) else "NaN"
        hv = f"{head_var_per_layer[l]:.6f}" if not np.isnan(head_var_per_layer[l]) else "NaN"
        print(f"{l:>6d} {te:>10} {ts:>10} {en:>10} {hv:>10}")

    return results


if __name__ == "__main__":
    results = run_experiment()
    gate = results["gate_decision"]
    sys.exit(0 if gate in ("STRONG_SUCCESS", "WEAK_SUCCESS") else 1)
