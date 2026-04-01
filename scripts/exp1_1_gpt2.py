"""
Experiment 1-1b: Boltzmann T_eff Gate Test — GPT-2 Medium (no GQA)
Cross-validation with standard multi-head attention model.
"""
import json, sys, time
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2-medium"
DEVICE = "cuda:1"
MAX_SEQ_LEN = 512
NUM_SAMPLES = 64
BATCH_SIZE = 4
OUTPUT_DIR = Path("/mnt/input/boltzmann/results/exp1_1_gpt2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_TEXTS = [
    "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms.",
    "In statistical mechanics, the Boltzmann distribution describes the probability of a system being in a certain state.",
    "Quantum computing leverages superposition and entanglement to perform computations exponentially faster.",
    "The human genome contains approximately three billion base pairs encoding around twenty thousand genes.",
    "Climate change models predict rising sea levels and more frequent extreme weather events worldwide.",
    "Deep learning has achieved remarkable success in computer vision, speech recognition, and game playing.",
    "The standard model of particle physics describes three of the four known fundamental forces.",
    "Protein folding is one of the grand challenges in computational biology and biophysics research.",
    "Reinforcement learning from human feedback has become the dominant paradigm for aligning large language models.",
    "Information theory quantifies the fundamental limits of data compression and reliable communication.",
    "Bayesian inference provides a principled framework for updating beliefs about model parameters.",
    "Graph neural networks extend deep learning to non-Euclidean domains by operating on graph-structured data.",
    "The central dogma of molecular biology describes the flow of genetic information from DNA to RNA to protein.",
    "Variational autoencoders combine neural networks with variational inference to learn latent representations.",
    "Monte Carlo methods use random sampling to obtain numerical results for problems that may be deterministic.",
    "Natural language understanding requires capturing semantic meaning beyond surface-level pattern matching.",
]


def run_experiment():
    print(f"{'='*60}")
    print(f"Experiment 1-1b: T_eff Gate (GPT-2 Medium, no GQA)")
    print(f"{'='*60}")

    t0 = time.time()
    print("\n[1/4] Loading GPT-2 medium...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map=DEVICE
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    print(f"   Loaded in {time.time()-t0:.1f}s. L={num_layers}, H={num_heads}, d_h={head_dim}")
    print(f"   GQA: No (standard MHA)")

    print(f"\n[2/4] Preparing data...")
    texts = (CALIBRATION_TEXTS * ((NUM_SAMPLES // len(CALIBRATION_TEXTS)) + 1))[:NUM_SAMPLES]

    print(f"\n[3/4] Forward passes with output_attentions=True...")
    layer_head_teff = np.zeros((num_layers, num_heads))
    layer_head_entropy = np.zeros((num_layers, num_heads))
    sample_count = 0

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        for layer_idx, attn in enumerate(outputs.attentions):
            # attn: (batch, heads, seq, seq) — real attention weights for GPT-2
            alpha = attn.float()
            log_alpha = torch.where(alpha > 1e-10, alpha.log(), torch.zeros_like(alpha))
            entropy = -(alpha * log_alpha).sum(dim=-1)  # (batch, heads, seq)

            # T_eff per position
            pos_idx = torch.arange(1, seq_len + 1, device=DEVICE).float()
            log_n = pos_idx.log().clamp(min=1e-6).unsqueeze(0).unsqueeze(0)
            teff = entropy / log_n

            mask = inputs["attention_mask"].unsqueeze(1).float()
            mask[:, :, 0] = 0  # skip pos 0
            teff_avg = (teff * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            entropy_avg = (entropy * mask).sum(-1) / mask.sum(-1).clamp(min=1)

            layer_head_teff[layer_idx] += teff_avg.mean(0).cpu().numpy() * len(batch_texts)
            layer_head_entropy[layer_idx] += entropy_avg.mean(0).cpu().numpy() * len(batch_texts)

        sample_count += len(batch_texts)
        del outputs, inputs
        torch.cuda.empty_cache()
        if (batch_start // BATCH_SIZE) % 4 == 0:
            print(f"   Batch {batch_start//BATCH_SIZE+1}/{(len(texts)+BATCH_SIZE-1)//BATCH_SIZE}")

    layer_head_teff /= sample_count
    layer_head_entropy /= sample_count

    print(f"\n[4/4] Analysis...")
    layer_teff_mean = layer_head_teff.mean(axis=1)
    layer_teff_std = layer_head_teff.std(axis=1)
    layer_entropy_mean = layer_head_entropy.mean(axis=1)
    head_var = layer_head_teff.var(axis=1)

    layer_indices = np.arange(num_layers)
    rho, p_value = stats.spearmanr(layer_indices, layer_teff_mean)

    if rho < -0.7 and p_value < 0.01:
        gate = "STRONG_SUCCESS"
    elif -0.7 <= rho < -0.3:
        gate = "WEAK_SUCCESS"
    else:
        gate = "FAILURE"

    print(f"\n{'='*60}")
    print(f"GATE: {gate} | ρ={rho:.4f}, p={p_value:.2e}")
    print(f"{'='*60}")

    # Summary table
    print(f"\n{'Layer':>6} {'T_eff':>10} {'Std':>10} {'Entropy':>10} {'HeadVar':>10}")
    for l in range(num_layers):
        print(f"{l:>6d} {layer_teff_mean[l]:>10.4f} {layer_teff_std[l]:>10.4f} "
              f"{layer_entropy_mean[l]:>10.4f} {head_var[l]:>10.6f}")

    results = {
        "experiment": "1-1b_gate_teff_gpt2",
        "model": MODEL_NAME,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "gqa": False,
        "num_samples": sample_count,
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
        "gate_decision": gate,
        "layer_teff_mean": layer_teff_mean.tolist(),
        "layer_teff_std": layer_teff_std.tolist(),
        "layer_entropy_mean": layer_entropy_mean.tolist(),
        "head_var_per_layer": head_var.tolist(),
        "layer_head_teff": layer_head_teff.tolist(),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp 1-1b: GPT-2 Medium (no GQA) — ρ={rho:.4f}, {gate}", fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.errorbar(layer_indices, layer_teff_mean, yerr=layer_teff_std,
                fmt='o-', capsize=3, color='steelblue', markersize=5)
    z = np.polyfit(layer_indices, layer_teff_mean, 1)
    ax.plot(layer_indices, np.polyval(z, layer_indices), '--r', alpha=0.5, label=f'slope={z[0]:.4f}')
    ax.set_xlabel("Layer"); ax.set_ylabel("T_eff"); ax.set_title("T_eff vs Layer"); ax.legend()

    ax = axes[0, 1]
    im = ax.imshow(layer_head_teff, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title("T_eff Heatmap")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    ax.bar(layer_indices, head_var, color='coral', alpha=0.7)
    ax.set_xlabel("Layer"); ax.set_ylabel("Var_h[T_eff]"); ax.set_title("Head Variance")

    ax = axes[1, 1]
    ax.plot(layer_indices, layer_entropy_mean, 'o-', color='darkgreen', markersize=5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Entropy S"); ax.set_title("Attention Entropy vs Layer")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp1_1b_gpt2_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    r = run_experiment()
    sys.exit(0 if r["gate_decision"] != "FAILURE" else 1)
