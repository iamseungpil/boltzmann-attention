#!/usr/bin/env python3
"""
Flip Probability Calibration: Does our metric predict actual ranking flips?

For each (layer, head, query position):
1. Compute FP16 attention scores → ground truth ranking
2. Compute 2-bit attention scores → quantized ranking
3. For each token: compute predicted P_flip from margin + query-direction noise
4. Check: does P_flip correlate with actual flips?

Metrics: Brier score, ECE, AUROC, recall@K comparison (flip-risk vs raw-score)

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp_flip_calibration.py --model mistralai/Mistral-7B-v0.3
"""
import argparse
import gc
import json
import math
import os
import sys
import warnings
from pathlib import Path

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

DTYPE = torch.bfloat16
sys.path.insert(0, str(Path(__file__).resolve().parent))


def uniform_quantize(x, bits):
    n = 2 ** bits
    mn = x.amin(dim=-2, keepdim=True)
    mx = x.amax(dim=-2, keepdim=True)
    rng = (mx - mn).clamp(min=1e-10)
    step = rng / (n - 1)
    return torch.round((x - mn) / step) * step + mn, step.squeeze(-2)


def find_attn(model):
    return [(n, m) for n, m in model.named_modules()
            if 'Attention' in type(m).__name__ and hasattr(m, 'k_proj')]


@torch.no_grad()
def collect_flip_data(model, tokenizer, device, bits=2,
                      n_chunks=5, chunk_len=2048):
    """Collect per-token flip data across layers and heads."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    ids = tokenizer.encode(text, return_tensors="pt", truncation=False)
    ids = ids[:, :n_chunks * chunk_len]

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, 'num_key_value_heads', n_heads)
    d_head = cfg.hidden_size // n_heads
    G = n_heads // n_kv

    attn_modules = find_attn(model)
    all_data = []

    for ci in range(n_chunks):
        chunk = ids[:, ci*chunk_len:(ci+1)*chunk_len].to(device)
        B, S = chunk.shape

        # Hook to capture Q, K pre-RoPE
        q_data, k_data = {}, {}
        hooks = []
        for li, (name, attn) in enumerate(attn_modules):
            def qh(li_=li):
                def fn(m, i, o): q_data[li_] = o.detach()
                return fn
            def kh(li_=li):
                def fn(m, i, o): k_data[li_] = o.detach()
                return fn
            hooks.append(attn.q_proj.register_forward_hook(qh()))
            hooks.append(attn.k_proj.register_forward_hook(kh()))

        model(chunk, use_cache=False)
        for h in hooks:
            h.remove()

        # Process selected layers (early, mid, late)
        for li in [0, 4, 8, 16, 24, 31]:
            if li >= len(attn_modules):
                continue
            name, attn = attn_modules[li]

            Q = q_data[li].view(B, S, n_heads, d_head).transpose(1, 2).float()
            K = k_data[li].view(B, S, n_kv, d_head).transpose(1, 2).float()

            # Apply RoPE
            pos = torch.arange(S, device=device).unsqueeze(0)
            if hasattr(attn, 'rotary_emb'):
                dummy = torch.zeros(B, n_kv, S, d_head, device=device, dtype=Q.dtype)
                cos, sin = attn.rotary_emb(dummy, pos)
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                Q, K = apply_rotary_pos_emb(Q.to(cos.dtype), K.to(cos.dtype), cos, sin)
                Q, K = Q.float(), K.float()

            # Quantize K
            K_exp = K.repeat_interleave(G, dim=1)
            K_q, step = uniform_quantize(K, bits)
            K_q_exp = K_q.repeat_interleave(G, dim=1)

            # Per-dim quantization noise variance: Δ²/12
            step_exp = step.repeat_interleave(G, dim=1) if step.dim() > 1 else step
            # step shape: (B, n_kv, d) → expand to (B, n_heads, d)

            # Sample query positions (last 64 positions for efficiency)
            q_positions = list(range(max(S-64, 1), S))

            for qi in q_positions:
                for head in range(min(n_heads, 8)):  # sample 8 heads
                    q = Q[0, head, qi]  # (d,)

                    # FP16 scores
                    s_fp16 = (q @ K_exp[0, head].T) / math.sqrt(d_head)  # (S,)
                    # Quantized scores
                    s_quant = (q @ K_q_exp[0, head].T) / math.sqrt(d_head)  # (S,)

                    # Causal: only look at positions <= qi
                    s_fp16 = s_fp16[:qi+1]
                    s_quant = s_quant[:qi+1]

                    # Ground truth: top-1 in FP16
                    fp16_top1 = s_fp16.argmax().item()
                    # Quantized: top-1
                    quant_top1 = s_quant.argmax().item()

                    # Did a flip happen?
                    flipped = (fp16_top1 != quant_top1)

                    # Compute our metric for each token
                    margin = s_quant.max() - s_quant  # (S,) margin from top
                    # Query-direction noise std
                    kv_head = head // G
                    step_h = step[0, kv_head] if step.dim() > 1 else step  # (d,)
                    sigma_sq = (q ** 2 * step_h ** 2 / 12).sum() / d_head
                    sigma = sigma_sq.sqrt()
                    # Pairwise: nu = sqrt(2) * sigma (same quant params for winner and challenger)
                    nu = math.sqrt(2) * sigma.item()

                    # Flip risk per token — vectorized
                    if nu > 1e-10:
                        z = (margin / nu).cpu().numpy()
                        flip_risk = torch.from_numpy(
                            sp_stats.norm.sf(z).astype(np.float32))
                    else:
                        flip_risk = torch.zeros(len(margin))

                    # Actual flip labels (was this token the FP16 winner but not quant winner?)
                    flip_labels = torch.zeros(len(margin))
                    if flipped:
                        flip_labels[fp16_top1] = 1.0  # the true winner was "flipped out"

                    all_data.append({
                        'layer': li,
                        'head': head,
                        'chunk': ci,
                        'query_pos': qi,
                        'flipped': flipped,
                        'fp16_top1': fp16_top1,
                        'quant_top1': quant_top1,
                        'margin_at_true_winner': margin[fp16_top1].item() if flipped else 0.0,
                        'flip_risk_at_true_winner': flip_risk[fp16_top1].item() if flipped else 0.0,
                        'sigma': sigma.item(),
                        'nu': nu,
                        # For recall@K comparison
                        'flip_risk_ranking': flip_risk.argsort(descending=True)[:32].tolist(),
                        'score_ranking': s_quant.argsort(descending=True)[:32].tolist(),
                    })

        del q_data, k_data
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  chunk {ci+1}/{n_chunks} done, {len(all_data)} samples", flush=True)

    return all_data


def analyze_calibration(data):
    """Compute calibration metrics."""
    flips = [d['flipped'] for d in data]
    flip_rate = sum(flips) / len(flips)
    print(f"\nOverall flip rate: {flip_rate:.4f} ({sum(flips)}/{len(flips)})")

    # Per-layer flip rate
    layers = sorted(set(d['layer'] for d in data))
    print("\nPer-layer flip rate:")
    for l in layers:
        ld = [d for d in data if d['layer'] == l]
        fr = sum(d['flipped'] for d in ld) / len(ld)
        print(f"  Layer {l}: {fr:.4f} ({sum(d['flipped'] for d in ld)}/{len(ld)})")

    # When flip happens, is the true winner in our top-32 by flip_risk?
    flipped_data = [d for d in data if d['flipped']]
    if flipped_data:
        recall_flip = sum(1 for d in flipped_data
                         if d['fp16_top1'] in d['flip_risk_ranking']) / len(flipped_data)
        recall_score = sum(1 for d in flipped_data
                          if d['fp16_top1'] in d['score_ranking']) / len(flipped_data)
        print(f"\nRecall@32 for true winner when flip occurs:")
        print(f"  By flip_risk: {recall_flip:.4f}")
        print(f"  By raw score: {recall_score:.4f}")
        print(f"  Δ (ours - score): {recall_flip - recall_score:+.4f}")

    return {
        'flip_rate': flip_rate,
        'n_samples': len(data),
        'n_flips': sum(flips),
        'per_layer': {l: sum(d['flipped'] for d in data if d['layer']==l) /
                        len([d for d in data if d['layer']==l])
                      for l in layers},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--n-chunks", type=int, default=5)
    parser.add_argument("--output-dir", default="results/flip_calibration")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"FLIP CALIBRATION: {args.model}, {args.bits}-bit", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, trust_remote_code=True,
        attn_implementation="eager"
    ).to(args.device).eval()

    print("\nCollecting flip data...", flush=True)
    data = collect_flip_data(model, tokenizer, args.device,
                            bits=args.bits, n_chunks=args.n_chunks)

    print("\nAnalyzing calibration...", flush=True)
    metrics = analyze_calibration(data)

    short = args.model.split("/")[-1].replace(".", "_")
    out_path = out_dir / f"{short}_{args.bits}bit_flip_calibration.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
