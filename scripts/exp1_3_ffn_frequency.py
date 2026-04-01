"""
Experiment 1-3: FFN Frequency Selectivity
==========================================
Tests whether FFN enhances spectral frequency selectivity in attention.
Compares spectral weighting entropy with and without FFN contribution.

Hypothesis: FFN-included condition has lower spectral entropy (sharper frequency selection).
"""

import json, sys, time, math
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2-medium"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 256
NUM_SAMPLES = 48
BATCH_SIZE = 2
OUTPUT_DIR = Path("/mnt/input/boltzmann/results/exp1_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXTS = [
    "The transformer architecture revolutionized natural language processing by introducing self-attention.",
    "In statistical mechanics the Boltzmann distribution describes state probabilities as a function of energy.",
    "Quantum computing leverages superposition and entanglement for exponentially faster computation.",
    "The human genome contains approximately three billion base pairs encoding twenty thousand genes.",
    "Climate change models predict rising sea levels and more frequent extreme weather events worldwide.",
    "Deep learning achieved remarkable success in vision speech recognition and strategic game playing.",
    "The standard model describes electromagnetic weak and strong nuclear forces but not gravity.",
    "Protein folding is a grand challenge in computational biology solved partially by AlphaFold.",
    "Reinforcement learning from human feedback aligns large language models with human preferences.",
    "Information theory quantifies fundamental limits of data compression and reliable communication.",
    "Bayesian inference provides principled belief updating given observed data evidence.",
    "Graph neural networks extend deep learning to non-Euclidean domains with graph structure.",
]


def get_attention_spectral_weights(model, inputs, use_ffn=True):
    """
    Extract attention energy contributions per frequency component.

    GPT-2 uses learned position embeddings (not RoPE), so we decompose
    the Q,K inner products in the frequency domain using DFT.

    Returns per-layer spectral entropy.
    """
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    device = inputs["input_ids"].device
    batch_size, seq_len = inputs["input_ids"].shape

    # Hook to optionally zero out FFN
    ffn_hooks = []
    if not use_ffn:
        for block in model.transformer.h:
            def zero_ffn(module, input, output, block=block):
                # Zero out FFN contribution: return input unchanged
                return torch.zeros_like(output)
            h = block.mlp.register_forward_hook(zero_ffn)
            ffn_hooks.append(h)

    # Hook to capture Q, K from each layer
    qk_store = {}
    attn_hooks = []
    for idx, block in enumerate(model.transformer.h):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden = input[0]
                qkv = module.c_attn(hidden)
                q, k, v = qkv.split(model.config.n_embd, dim=-1)
                bsz, slen, _ = q.shape
                q = q.view(bsz, slen, num_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, slen, num_heads, head_dim).transpose(1, 2)
                qk_store[layer_idx] = (q.detach().float(), k.detach().float())
            return hook_fn
        h = block.attn.register_forward_hook(make_hook(idx))
        attn_hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in attn_hooks + ffn_hooks:
        h.remove()

    # Compute spectral weights per layer
    layer_spectral_entropy = []
    layer_spectral_weights = []

    for layer_idx in range(num_layers):
        if layer_idx not in qk_store:
            layer_spectral_entropy.append(np.nan)
            layer_spectral_weights.append(None)
            continue

        q, k = qk_store[layer_idx]  # (batch, heads, seq, head_dim)

        # DFT of Q and K along head_dim axis
        # This decomposes the inner product into frequency contributions
        q_fft = torch.fft.rfft(q, dim=-1)  # (batch, heads, seq, head_dim//2+1)
        k_fft = torch.fft.rfft(k, dim=-1)

        # Spectral energy per frequency: |Q(f)|*|K(f)| averaged over positions
        # This measures how much each frequency contributes to attention energy
        spectral_energy = (q_fft.abs() * k_fft.abs()).mean(dim=2)  # (batch, heads, freq)

        # Normalize to get spectral weight distribution
        spectral_sum = spectral_energy.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        p_omega = spectral_energy / spectral_sum  # (batch, heads, freq)

        # Spectral entropy: H = -Σ p(ω) log p(ω)
        log_p = torch.where(p_omega > 1e-10, p_omega.log(), torch.zeros_like(p_omega))
        H = -(p_omega * log_p).sum(dim=-1)  # (batch, heads)

        # Average over batch and heads
        H_mean = H.mean().item()
        layer_spectral_entropy.append(H_mean)

        # Store spectral weights for analysis (averaged over batch)
        layer_spectral_weights.append(p_omega.mean(dim=0).cpu().numpy())  # (heads, freq)

    del qk_store
    return layer_spectral_entropy, layer_spectral_weights


def run_experiment():
    print(f"{'='*60}")
    print(f"Experiment 1-3: FFN Frequency Selectivity")
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*60}")

    t0 = time.time()
    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    num_freq = head_dim // 2 + 1
    print(f"   Loaded in {time.time()-t0:.1f}s. L={num_layers}, H={num_heads}, d_h={head_dim}, freq={num_freq}")

    print(f"\n[2/4] Preparing data ({NUM_SAMPLES} samples)...")
    texts = (TEXTS * ((NUM_SAMPLES // len(TEXTS)) + 1))[:NUM_SAMPLES]

    # Accumulate spectral entropy per layer for both conditions
    entropy_with_ffn = np.zeros(num_layers)
    entropy_without_ffn = np.zeros(num_layers)
    spectral_w_ffn = [np.zeros((num_heads, num_freq)) for _ in range(num_layers)]
    spectral_wo_ffn = [np.zeros((num_heads, num_freq)) for _ in range(num_layers)]
    n_batches = 0

    print(f"\n[3/4] Running forward passes...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)

        # With FFN
        ent_w, sw_w = get_attention_spectral_weights(model, inputs, use_ffn=True)
        # Without FFN
        ent_wo, sw_wo = get_attention_spectral_weights(model, inputs, use_ffn=False)

        for l in range(num_layers):
            entropy_with_ffn[l] += ent_w[l] if not np.isnan(ent_w[l]) else 0
            entropy_without_ffn[l] += ent_wo[l] if not np.isnan(ent_wo[l]) else 0
            if sw_w[l] is not None:
                spectral_w_ffn[l] += sw_w[l]
            if sw_wo[l] is not None:
                spectral_wo_ffn[l] += sw_wo[l]

        n_batches += 1
        del inputs
        torch.cuda.empty_cache()

        if (i // BATCH_SIZE) % 6 == 0:
            print(f"   Batch {i//BATCH_SIZE+1}/{(len(texts)+BATCH_SIZE-1)//BATCH_SIZE}")

    entropy_with_ffn /= n_batches
    entropy_without_ffn /= n_batches
    for l in range(num_layers):
        spectral_w_ffn[l] /= n_batches
        spectral_wo_ffn[l] /= n_batches

    # Analysis
    print(f"\n[4/4] Analyzing...")
    entropy_diff = entropy_without_ffn - entropy_with_ffn  # positive = FFN reduces entropy
    layers = np.arange(num_layers)

    # Paired t-test: is entropy significantly lower with FFN?
    t_stat, p_value = stats.ttest_rel(entropy_without_ffn, entropy_with_ffn)
    mean_diff = entropy_diff.mean()
    ffn_reduces_entropy = mean_diff > 0 and p_value < 0.05

    # Wilcoxon signed-rank (non-parametric)
    w_stat, w_pvalue = stats.wilcoxon(entropy_diff)

    # Per-layer sign test
    n_ffn_lower = (entropy_diff > 0).sum()

    print(f"\n{'='*60}")
    print(f"FFN FREQUENCY SELECTIVITY RESULTS")
    print(f"  Mean spectral entropy WITH FFN:    {entropy_with_ffn.mean():.4f}")
    print(f"  Mean spectral entropy WITHOUT FFN: {entropy_without_ffn.mean():.4f}")
    print(f"  Mean difference (wo - w):          {mean_diff:.4f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"  Wilcoxon: W={w_stat:.1f}, p={w_pvalue:.4f}")
    print(f"  Layers where FFN reduces entropy:  {n_ffn_lower}/{num_layers}")
    print(f"  FFN enhances selectivity: {'YES' if ffn_reduces_entropy else 'NO'}")
    print(f"{'='*60}")

    print(f"\n{'Layer':>6} {'With_FFN':>10} {'Without':>10} {'Diff':>10}")
    for l in range(num_layers):
        print(f"{l:>6d} {entropy_with_ffn[l]:>10.4f} {entropy_without_ffn[l]:>10.4f} {entropy_diff[l]:>10.4f}")

    results = {
        "experiment": "1-3_ffn_frequency_selectivity",
        "model": MODEL_NAME,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "num_samples": NUM_SAMPLES,
        "entropy_with_ffn": entropy_with_ffn.tolist(),
        "entropy_without_ffn": entropy_without_ffn.tolist(),
        "entropy_diff": entropy_diff.tolist(),
        "mean_diff": float(mean_diff),
        "paired_t_stat": float(t_stat),
        "paired_p_value": float(p_value),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(w_pvalue),
        "n_ffn_reduces_entropy": int(n_ffn_lower),
        "ffn_enhances_selectivity": bool(ffn_reduces_entropy),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp 1-3: FFN Frequency Selectivity ({MODEL_NAME})\n"
                 f"Δ={mean_diff:.4f}, p={p_value:.4f}, "
                 f"FFN {'enhances' if ffn_reduces_entropy else 'does NOT enhance'} selectivity",
                 fontsize=12, fontweight='bold')

    # (a) Entropy comparison per layer
    ax = axes[0, 0]
    ax.plot(layers, entropy_with_ffn, 'o-', color='steelblue', label='With FFN', markersize=4)
    ax.plot(layers, entropy_without_ffn, 's--', color='coral', label='Without FFN', markersize=4)
    ax.set_xlabel("Layer"); ax.set_ylabel("Spectral Entropy H(ω)")
    ax.set_title("Spectral Entropy: With vs Without FFN"); ax.legend()

    # (b) Entropy difference
    ax = axes[0, 1]
    colors = ['green' if d > 0 else 'red' for d in entropy_diff]
    ax.bar(layers, entropy_diff, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("ΔH (without - with)")
    ax.set_title(f"FFN Effect on Spectral Entropy ({n_ffn_lower}/{num_layers} layers reduced)")

    # (c) Spectral weight heatmap WITH FFN (sample layer)
    ax = axes[1, 0]
    mid = num_layers // 2
    im = ax.imshow(spectral_w_ffn[mid], aspect='auto', cmap='viridis')
    ax.set_xlabel("Frequency Index"); ax.set_ylabel("Head")
    ax.set_title(f"Spectral Weights WITH FFN (Layer {mid})")
    plt.colorbar(im, ax=ax)

    # (d) Average spectral profile comparison
    ax = axes[1, 1]
    avg_w = np.mean([spectral_w_ffn[l].mean(axis=0) for l in range(num_layers)], axis=0)
    avg_wo = np.mean([spectral_wo_ffn[l].mean(axis=0) for l in range(num_layers)], axis=0)
    freqs = np.arange(len(avg_w))
    ax.plot(freqs, avg_w, '-', color='steelblue', label='With FFN')
    ax.plot(freqs, avg_wo, '--', color='coral', label='Without FFN')
    ax.set_xlabel("Frequency Index"); ax.set_ylabel("Weight p(ω)")
    ax.set_title("Average Spectral Profile"); ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp1_3_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    r = run_experiment()
    sys.exit(0)
