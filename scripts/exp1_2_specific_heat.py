"""
Experiment 1-2: Specific Heat-Based Adaptive KV Cache Quantization
==================================================================
Compares three quantization strategies on GPT-2 Medium:
  1. Uniform: all tokens get same bit width
  2. Variance-based: ||k||^2 determines bit allocation
  3. Specific heat-based: C(q) determines bit allocation

Key metric: disagreement rate + PPL comparison
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
from datasets import load_dataset

MODEL_NAME = "openai-community/gpt2-medium"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 256
EVAL_SAMPLES = 64
CALIB_SAMPLES = 32
BATCH_SIZE = 2
BIT_WIDTHS = [2, 3, 4]
OUTPUT_DIR = Path("/mnt/input/boltzmann/results/exp1_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def symmetric_quantize(tensor, bits):
    """Symmetric uniform quantization."""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / qmax
    quantized = (tensor / scale).round().clamp(qmin, qmax)
    return quantized * scale


def extract_qk(model, inputs):
    """Extract Q, K from all layers via hooks on GPT-2."""
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    qk_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = input[0]  # (batch, seq, n_embd)
            # GPT-2: c_attn projects to 3*n_embd, then split
            qkv = module.c_attn(hidden)  # (batch, seq, 3*n_embd)
            q, k, v = qkv.split(model.config.n_embd, dim=-1)
            bsz, slen, _ = q.shape
            q = q.view(bsz, slen, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, slen, num_heads, head_dim).transpose(1, 2)
            qk_store[layer_idx] = (q.detach().float(), k.detach().float())
        return hook_fn

    hooks = []
    for idx, block in enumerate(model.transformer.h):
        h = block.attn.register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()
    return qk_store


def compute_boltzmann_quantities(qk_store, seq_len, device):
    """Compute C(q), ||k||^2, F from Q,K pairs."""
    results = {}
    for layer_idx, (q, k) in qk_store.items():
        head_dim = q.shape[-1]
        sqrt_d = math.sqrt(head_dim)

        energy = torch.matmul(q, k.transpose(-2, -1)) / sqrt_d
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        energy = energy.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        alpha = F.softmax(energy, dim=-1)
        energy_safe = energy.masked_fill(causal.unsqueeze(0).unsqueeze(0), 0.0)

        mean_E = (alpha * energy_safe).sum(-1)
        mean_E2 = (alpha * energy_safe ** 2).sum(-1)
        C_q = mean_E2 - mean_E ** 2  # (batch, heads, seq)

        k_norm_sq = (k ** 2).sum(-1)  # (batch, heads, seq)

        log_Z = torch.logsumexp(energy.masked_fill(
            causal.unsqueeze(0).unsqueeze(0), float('-inf')), dim=-1)
        F_val = -sqrt_d * log_Z

        results[layer_idx] = {
            'C_q': C_q.cpu(),
            'k_norm_sq': k_norm_sq.cpu(),
            'F_val': F_val.cpu(),
        }
    return results


def quantize_k_adaptive(k, criterion, bits, seq_len):
    """Quantize K with adaptive bit allocation based on criterion per position."""
    # Move criterion to same device as k
    crit = criterion.to(k.device)
    if crit.dim() == 3 and k.dim() == 4:
        crit = crit.unsqueeze(-1)  # (batch, heads, seq, 1)
    elif crit.dim() == 2:
        crit = crit.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq, 1)

    # Handle size mismatches by truncating
    if crit.shape[2] > k.shape[2]:
        crit = crit[:, :, :k.shape[2]]
    elif crit.shape[2] < k.shape[2]:
        pad = torch.zeros(*crit.shape[:2], k.shape[2] - crit.shape[2], *crit.shape[3:], device=k.device)
        crit = torch.cat([crit, pad], dim=2)

    median_c = crit.median()
    hi_bits = min(bits + 1, 8)
    lo_bits = max(bits - 1, 1)
    k_hi = symmetric_quantize(k, hi_bits)
    k_lo = symmetric_quantize(k, lo_bits)
    mask = (crit > median_c).float()
    return k_hi * mask + k_lo * (1 - mask)


def eval_ppl_with_quantized_k(model, tokenizer, texts, quant_fn):
    """Evaluate PPL by replacing K with quantized version in attention."""
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)
        seq_len = inputs["input_ids"].shape[1]

        # First pass: get Q, K for C(q) computation
        qk_store = extract_qk(model, inputs)
        boltz = compute_boltzmann_quantities(qk_store, seq_len, DEVICE)

        # Second pass: with quantized K
        def make_quant_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is (attn_output, attn_weights) or just attn_output
                # We need to intercept and modify K before attention computation
                hidden = input[0]
                qkv = module.c_attn(hidden)
                q_proj, k_proj, v_proj = qkv.split(model.config.n_embd, dim=-1)
                bsz, slen, _ = k_proj.shape
                k_reshaped = k_proj.view(bsz, slen, num_heads, head_dim).transpose(1, 2)

                # Apply quantization
                k_quantized = quant_fn(k_reshaped, layer_idx, boltz)

                # Recompute attention with quantized K
                q = q_proj.view(bsz, slen, num_heads, head_dim).transpose(1, 2)
                v = v_proj.view(bsz, slen, num_heads, head_dim).transpose(1, 2)
                attn_w = torch.matmul(q, k_quantized.transpose(-2, -1)) / math.sqrt(head_dim)
                causal = torch.triu(torch.ones(slen, slen, device=attn_w.device, dtype=torch.bool), diagonal=1)
                attn_w = attn_w.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_w = F.softmax(attn_w, dim=-1)
                attn_out = torch.matmul(attn_w, v)
                attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, slen, -1)
                attn_out = module.c_proj(attn_out)
                attn_out = module.resid_dropout(attn_out)

                # Return modified output (replacing the original attention output)
                if isinstance(output, tuple):
                    return (attn_out,) + output[1:]
                return attn_out
            return hook_fn

        hooks = []
        for idx, block in enumerate(model.transformer.h):
            h = block.attn.register_forward_hook(make_quant_hook(idx))
            hooks.append(h)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            nll = outputs.loss.item()

        for h in hooks:
            h.remove()

        if math.isnan(nll) or math.isinf(nll):
            nll = 20.0  # cap extreme values

        mask = inputs["attention_mask"]
        n_tokens = mask.sum().item()
        total_nll += nll * n_tokens
        total_tokens += n_tokens

        del inputs, qk_store, boltz
        torch.cuda.empty_cache()

    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')


def run_experiment():
    print(f"{'='*60}")
    print(f"Experiment 1-2: Specific Heat Quantization")
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*60}")

    t0 = time.time()
    print("\n[1/5] Loading model...")
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
    print(f"   Loaded in {time.time()-t0:.1f}s. L={num_layers}, H={num_heads}, d_h={head_dim}")

    print("\n[2/5] Loading WikiText-2...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        eval_texts = [t for t in dataset["text"] if len(t.strip()) > 100][:EVAL_SAMPLES]
    except Exception:
        eval_texts = ["The study of thermodynamics reveals structural analogies. " * 20] * EVAL_SAMPLES
    print(f"   {len(eval_texts)} samples")

    # ── Step 3: Compute C(q) and ||k||² for disagreement analysis ──
    print("\n[3/5] Computing Boltzmann quantities on calibration set...")
    calib = eval_texts[:CALIB_SAMPLES]
    all_C_q = {l: [] for l in range(num_layers)}
    all_k_norm = {l: [] for l in range(num_layers)}

    for i in range(0, len(calib), BATCH_SIZE):
        batch = calib[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)
        seq_len = inputs["input_ids"].shape[1]
        qk = extract_qk(model, inputs)
        boltz = compute_boltzmann_quantities(qk, seq_len, DEVICE)
        mask = inputs["attention_mask"].cpu()

        for l in boltz:
            C = boltz[l]['C_q'].mean(dim=1)  # avg over heads → (batch, seq)
            K = boltz[l]['k_norm_sq'].mean(dim=1)
            for b in range(C.shape[0]):
                v = mask[b].bool()
                all_C_q[l].extend(C[b][v].numpy().tolist())
                all_k_norm[l].extend(K[b][v].numpy().tolist())
        del inputs, qk, boltz
        torch.cuda.empty_cache()
        if i % 8 == 0:
            print(f"   Batch {i//BATCH_SIZE+1}/{len(calib)//BATCH_SIZE}")

    # ── Step 4: Disagreement analysis ──
    print("\n[4/5] Analyzing bit allocation disagreement...")
    disagreement_rates = []
    correlations = []
    layer_stats = []

    for l in range(num_layers):
        c = np.array(all_C_q[l])
        k = np.array(all_k_norm[l])
        c_alloc = (c > np.median(c)).astype(int)
        k_alloc = (k > np.median(k)).astype(int)
        disagree = (c_alloc != k_alloc).mean()
        rho_ck, _ = stats.spearmanr(c, k)
        disagreement_rates.append(disagree)
        correlations.append(rho_ck)
        layer_stats.append({
            'layer': l, 'disagreement': float(disagree),
            'rho_C_k': float(rho_ck),
            'C_q_mean': float(c.mean()), 'k_norm_mean': float(k.mean()),
        })
        print(f"   L{l:2d}: disagree={disagree:.3f} ρ(C,k²)={rho_ck:.3f}")

    avg_disagree = np.mean(disagreement_rates)
    avg_corr = np.mean(correlations)
    print(f"\n   AVG disagreement: {avg_disagree:.3f}")
    print(f"   AVG ρ(C,||k||²): {avg_corr:.3f}")

    # ── Step 5: PPL evaluation ──
    print("\n[5/5] Evaluating PPL with quantized KV cache...")
    eval_sub = eval_texts[:32]  # subset for speed
    ppl_results = {}

    # Baseline PPL (no quantization)
    print("   Computing baseline PPL...")
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(eval_sub), BATCH_SIZE):
        batch = eval_sub[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        n = inputs["attention_mask"].sum().item()
        total_nll += out.loss.item() * n
        total_tok += n
    ppl_baseline = math.exp(total_nll / total_tok)
    ppl_results["baseline"] = ppl_baseline
    print(f"   Baseline PPL: {ppl_baseline:.2f}")

    for bits in BIT_WIDTHS:
        print(f"\n   --- {bits}-bit ---")

        # Uniform
        def uniform_quant(k, layer_idx, boltz, bits=bits):
            return symmetric_quantize(k, bits)

        ppl = eval_ppl_with_quantized_k(model, tokenizer, eval_sub, uniform_quant)
        ppl_results[f"uniform_{bits}bit"] = ppl
        print(f"   uniform:       PPL={ppl:.2f} (Δ={ppl-ppl_baseline:+.2f})")

        # Variance-based
        def variance_quant(k, layer_idx, boltz, bits=bits):
            k_norm = (k ** 2).sum(-1).clamp(min=1e-8)  # (batch, heads, seq)
            return quantize_k_adaptive(k, k_norm, bits, k.shape[2])

        ppl = eval_ppl_with_quantized_k(model, tokenizer, eval_sub, variance_quant)
        ppl_results[f"variance_{bits}bit"] = ppl
        print(f"   variance:      PPL={ppl:.2f} (Δ={ppl-ppl_baseline:+.2f})")

        # Specific heat-based
        def heat_quant(k, layer_idx, boltz, bits=bits):
            if layer_idx in boltz:
                C_q = boltz[layer_idx]['C_q'][:k.shape[0], :, :k.shape[2]]
                return quantize_k_adaptive(k, C_q, bits, k.shape[2])
            return symmetric_quantize(k, bits)

        ppl = eval_ppl_with_quantized_k(model, tokenizer, eval_sub, heat_quant)
        ppl_results[f"specific_heat_{bits}bit"] = ppl
        print(f"   specific_heat: PPL={ppl:.2f} (Δ={ppl-ppl_baseline:+.2f})")

    # ── Decision ──
    heat_wins = 0
    var_wins = 0
    for bits in BIT_WIDTHS:
        h_ppl = ppl_results.get(f"specific_heat_{bits}bit", float('inf'))
        v_ppl = ppl_results.get(f"variance_{bits}bit", float('inf'))
        if h_ppl < v_ppl:
            heat_wins += 1
        elif v_ppl < h_ppl:
            var_wins += 1

    if avg_disagree > 0.2 and heat_wins > var_wins:
        decision = "USEFUL_PERSPECTIVE"
        msg = f"C(q) differs from Var ({avg_disagree:.1%}) AND outperforms ({heat_wins}/{len(BIT_WIDTHS)} wins)"
    elif avg_disagree > 0.2:
        decision = "DIFFERENT_BUT_NOT_BETTER"
        msg = f"C(q) differs ({avg_disagree:.1%}) but doesn't consistently outperform"
    else:
        decision = "REPACKAGING"
        msg = f"C(q) ≈ Var ({avg_disagree:.1%} disagreement)"

    print(f"\n{'='*60}")
    print(f"DECISION: {decision}")
    print(f"  {msg}")
    print(f"  Heat wins: {heat_wins}, Var wins: {var_wins}")
    print(f"{'='*60}")

    results = {
        "experiment": "1-2_specific_heat_quantization",
        "model": MODEL_NAME,
        "decision": decision,
        "avg_disagreement": float(avg_disagree),
        "avg_correlation": float(avg_corr),
        "ppl_results": {k: float(v) for k, v in ppl_results.items()},
        "heat_wins": heat_wins,
        "var_wins": var_wins,
        "layer_stats": layer_stats,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Exp 1-2: Specific Heat Quantization — {decision}", fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    ax.bar(range(num_layers), disagreement_rates, color='steelblue', alpha=0.7)
    ax.axhline(y=0.2, color='red', linestyle='--', label='20% threshold')
    ax.axhline(y=avg_disagree, color='green', linestyle='-', label=f'avg={avg_disagree:.3f}')
    ax.set_xlabel("Layer"); ax.set_ylabel("Disagreement"); ax.set_title("C(q) vs ||k||² Allocation Disagreement"); ax.legend()

    ax = axes[0, 1]
    ax.bar(range(num_layers), correlations, color='coral', alpha=0.7)
    ax.set_xlabel("Layer"); ax.set_ylabel("ρ"); ax.set_title("Spearman ρ(C(q), ||k||²)")

    ax = axes[1, 0]
    methods = ["uniform", "variance", "specific_heat"]
    colors = ['gray', 'steelblue', 'coral']
    x = np.arange(len(BIT_WIDTHS))
    w = 0.25
    for j, (m, c) in enumerate(zip(methods, colors)):
        ppls = [ppl_results.get(f"{m}_{b}bit", 0) for b in BIT_WIDTHS]
        bars = ax.bar(x + j*w, ppls, w, label=m, color=c, alpha=0.8)
    ax.axhline(y=ppl_baseline, color='black', linestyle='--', label=f'baseline={ppl_baseline:.1f}')
    ax.set_xticks(x + w); ax.set_xticklabels([f"{b}bit" for b in BIT_WIDTHS])
    ax.set_ylabel("PPL"); ax.set_title("Perplexity Comparison"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    mid = num_layers // 2
    c_arr = np.array(all_C_q[mid])[:500]
    k_arr = np.array(all_k_norm[mid])[:500]
    ax.scatter(k_arr, c_arr, alpha=0.3, s=10, c='purple')
    ax.set_xlabel("||k||²"); ax.set_ylabel("C(q)")
    ax.set_title(f"C(q) vs ||k||² Scatter (Layer {mid})")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp1_2_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    r = run_experiment()
    sys.exit(0)
