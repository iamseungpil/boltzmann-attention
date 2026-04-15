#!/usr/bin/env python3
"""
V2ai: Direct Jacobian Norm Measurement per Layer
===================================================

v2ah measured layer-isolated cascade by quantizing one layer at a time.
This experiment measures the underlying *forward Jacobian norm* directly,
independent of the quantization scheme:

    J_{L←ℓ} ≈ ‖h_L^{perturbed} - h_L^FP‖ / ‖δ‖

for small random perturbations δ injected at layer ℓ's residual stream.
This isolates the model's structural amplification factor from the
quantizer-specific error magnitude.

Protocol per model:
  1. FP16 forward, capture h_ℓ for selected layers + h_L (final).
  2. For each sampled ℓ ∈ {0, 1, 2, n/4, n/2, 3n/4, n-1}:
       For each trial t ∈ 1..K:
         a. Sample δ_t ~ N(0, I) and rescale to ‖δ_t‖ = ε * ‖h_ℓ‖
         b. Hook layer ℓ to add δ_t to its output, forward, capture h_L^t
         c. Compute amplification a_t = ‖h_L^t - h_L^FP‖ / ‖δ_t‖
       Average a over trials → estimated ‖J_{L←ℓ}‖ (random-direction).
  3. Optionally repeat with δ aligned to the top PCA eigenvector of K at
     layer ℓ (the "Lloyd error" direction) for a "directed" Jacobian norm.

Outputs: per-layer Jacobian operator-norm estimate. Compare across modes.

Predictions:
  - Mistral L2 should have largest Jacobian norm (matches v2ah dominant).
  - Nemo L0/L1 should have largest.
  - Qwen-7B should be relatively flat across layers (no spike).
  - Qwen-1.5B should have late-layer spikes (matches v2ah L22).
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
N_EVAL = 1024
N_TRIALS = 4
EPS = 0.01
N_LAYERS_SAMPLE = 8
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

MODELS = [
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b'),
    ('mistralai/Mistral-Nemo-Base-2407', 'nemo-12b'),
    ('Qwen/Qwen2.5-7B', 'qwen-7b'),
    ('Qwen/Qwen2.5-1.5B', 'qwen-1.5b'),
]


class AddPerturbHook:
    """Forward hook that adds a fixed perturbation to a decoder layer's output."""
    def __init__(self, delta):
        self.delta = delta
    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            new_h = output[0] + self.delta.to(output[0].dtype).to(output[0].device)
            return (new_h,) + output[1:]
        else:
            return output + self.delta.to(output.dtype).to(output.device)


def capture_residuals(model, ids, n_layers, layers_to_capture):
    """Run forward, capture residual stream at specified layer indices.
    Returns dict {layer_idx: tensor (1, T, hidden)}."""
    captured = {}
    handles = []
    for li in layers_to_capture:
        def mk(li=li):
            def h(m, i, o):
                a = o[0] if isinstance(o, tuple) else o
                captured[li] = a.detach().clone()
            return h
        handles.append(model.model.layers[li].register_forward_hook(mk()))
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    for h in handles: h.remove()
    return captured


def forward_with_perturb(model, ids, n_layers, perturb_layer, delta, capture_layer):
    """Forward with delta added at perturb_layer's output, capture residual at capture_layer."""
    captured = {}
    perturb_handle = model.model.layers[perturb_layer].register_forward_hook(AddPerturbHook(delta))
    def cap_hook(m, i, o):
        a = o[0] if isinstance(o, tuple) else o
        captured['h'] = a.detach().clone()
    cap_handle = model.model.layers[capture_layer].register_forward_hook(cap_hook)
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    perturb_handle.remove()
    cap_handle.remove()
    return captured['h']


def analyze_model(model_id, sn):
    print(f"\n{'='*70}\n  {sn}: {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='sdpa', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    print(f"  n_layers={n_layers} hidden={hidden} loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    eval_text = '\n\n'.join(texts[300:600])
    eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=N_EVAL)['input_ids'].to('cuda:0')
    T = eval_ids.shape[1]

    # Sample layers (force include 0, 1, 2, last; spread rest)
    if n_layers <= N_LAYERS_SAMPLE + 3:
        sampled = list(range(n_layers))
    else:
        sampled = [0, 1, 2]
        step = (n_layers - 4) // (N_LAYERS_SAMPLE - 4)
        for i in range(N_LAYERS_SAMPLE - 4):
            sampled.append(3 + i * step)
        sampled.append(n_layers - 1)
        sampled = sorted(set(sampled))
    L_final = n_layers - 1
    print(f"  Sampled layers: {sampled}", flush=True)

    # Capture FP16 residuals at all sampled layers + final
    capture_set = sorted(set(sampled + [L_final]))
    h_fp = capture_residuals(model, eval_ids, n_layers, capture_set)
    h_L_fp = h_fp[L_final]
    print(f"  FP16 captured at {len(capture_set)} layers, final norm² (per-token avg) = "
          f"{float(h_L_fp.float().pow(2).sum(-1).mean().item()):.3e}", flush=True)

    # For each sampled layer, do K trials of random perturbation
    per_layer = []
    for li in sampled:
        h_li = h_fp[li]
        h_li_norm_per_token = h_li.float().pow(2).sum(-1).sqrt()  # (1, T)
        avg_norm = float(h_li_norm_per_token.mean().item())
        sigma = EPS * avg_norm   # perturbation magnitude per token

        amps = []
        for trial in range(N_TRIALS):
            # Random unit-direction perturbation, scaled to sigma per token
            torch.manual_seed(1000 * li + trial)
            delta = torch.randn_like(h_li.float())  # (1, T, hidden)
            delta_norms = delta.pow(2).sum(-1, keepdim=True).sqrt()  # (1, T, 1)
            delta = delta / delta_norms.clamp_min(1e-12) * sigma
            # delta is now norm sigma per token

            h_perturbed = forward_with_perturb(model, eval_ids, n_layers,
                                               perturb_layer=li, delta=delta,
                                               capture_layer=L_final)

            d_out = (h_perturbed - h_L_fp).float()
            d_in_norm_sq = (delta.float()).pow(2).sum(-1).mean().item()  # avg per-token ‖δ‖²
            d_out_norm_sq = d_out.pow(2).sum(-1).mean().item()           # avg per-token ‖J δ‖²
            amp = (d_out_norm_sq / max(d_in_norm_sq, 1e-12)) ** 0.5      # ratio of norms
            amps.append(amp)

        amp_mean = float(np.mean(amps))
        amp_std = float(np.std(amps))
        per_layer.append({
            'layer': li,
            'avg_residual_norm': avg_norm,
            'jacobian_norm_mean': amp_mean,
            'jacobian_norm_std': amp_std,
            'jacobian_norm_sq_mean': amp_mean ** 2,
            'trials': amps,
        })

    print(f"\n  Per-layer Jacobian norm estimates (random direction):")
    print(f"  {'layer':<6}|{'‖h_ℓ‖':>12}|{'‖J_{{L←ℓ}}‖':>14}|{'std':>10}|{'sq':>10}")
    for p in per_layer:
        print(f"  {p['layer']:<6}|{p['avg_residual_norm']:>12.3e}|"
              f"{p['jacobian_norm_mean']:>14.4f}|{p['jacobian_norm_std']:>10.4f}|"
              f"{p['jacobian_norm_sq_mean']:>10.4f}", flush=True)

    # Find dominant Jacobian layer
    dom = max(per_layer, key=lambda p: p['jacobian_norm_mean'])
    print(f"\n  Dominant Jacobian: L{dom['layer']} (norm={dom['jacobian_norm_mean']:.4f})", flush=True)

    del model, tok
    gc.collect(); torch.cuda.empty_cache()

    return {
        'model': model_id, 'short_name': sn,
        'n_layers': n_layers, 'sampled_layers': sampled,
        'eps': EPS, 'n_trials': N_TRIALS,
        'per_layer': per_layer,
        'dominant_layer': dom['layer'],
    }


def main():
    print("="*70)
    print("V2ai: Direct Jacobian Norm Measurement")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid, sn in MODELS:
        try:
            results[sn] = analyze_model(mid, sn)
        except Exception as e:
            print(f"ERROR on {sn}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*70)
    print("JACOBIAN NORM PROFILE")
    print("="*70)
    for sn, r in results.items():
        print(f"\n  {sn} (n_layers={r['n_layers']}, dominant L{r['dominant_layer']}):")
        per = r['per_layer']
        for p in per:
            print(f"    L{p['layer']:<3} ‖J‖ = {p['jacobian_norm_mean']:.4f} ± {p['jacobian_norm_std']:.4f}")
        # Summary: max / median / total norm² (proxy for total cascade strength)
        norms = [p['jacobian_norm_mean'] for p in per]
        norms_sq = [n**2 for n in norms]
        print(f"    --> max={max(norms):.4f}, median={float(np.median(norms)):.4f}, "
              f"sum_sq={sum(norms_sq):.3f}")

    out = OUT_DIR / 'exp_v2ai_jacobian_norm.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
