#!/usr/bin/env python3
"""
CLTQ Diagnostic: Cross-Layer Token importance for KV quantization.

Measures per-token importance = Σ_l α_l(t) × ||v_l(t) − v̄_l||
across all layers. Tests whether this metric predicts NIAH sensitivity.

Also measures: cumulative attention, value uniqueness, attention sink patterns.

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp_cltq_diagnostic.py --model mistralai/Mistral-7B-v0.3
"""
import argparse, gc, json, math, os, sys, warnings
from pathlib import Path

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16
sys.path.insert(0, str(Path(__file__).resolve().parent))


def find_attn(model):
    return [(n, m) for n, m in model.named_modules()
            if 'Attention' in type(m).__name__ and hasattr(m, 'k_proj')]


def make_niah_prompt(tokenizer, context_len, depth):
    needle = "The secret number is 7392."
    query = "\n\nBased on the text above, what is the secret number? Answer:"
    filler = "The city of Darworth has a population of 8,234 and is known for its annual cherry blossom festival. " * 80
    needle_toks = tokenizer.encode(needle, add_special_tokens=False)
    query_toks = tokenizer.encode(query, add_special_tokens=False)
    filler_toks = tokenizer.encode(filler, add_special_tokens=False)
    avail = context_len - len(needle_toks) - len(query_toks) - 5
    pos = int(avail * depth)
    tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
    tokens += filler_toks[:pos] + needle_toks + filler_toks[:avail - pos] + query_toks
    tokens = tokens[:context_len]
    needle_start = len([tokenizer.bos_token_id] if tokenizer.bos_token_id else []) + pos
    needle_end = needle_start + len(needle_toks)
    return torch.tensor([tokens]), needle_start, needle_end


@torch.no_grad()
def compute_cross_layer_importance(model, tokenizer, device, context_len=4096, depth=0.5):
    """Compute per-token cross-layer importance scores."""

    input_ids, needle_start, needle_end = make_niah_prompt(tokenizer, context_len, depth)
    input_ids = input_ids.to(device)
    B, S = input_ids.shape

    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, 'num_key_value_heads', n_heads)
    d_head = cfg.hidden_size // n_heads
    G = n_heads // n_kv
    n_layers = cfg.num_hidden_layers

    # Extract Q, K, V at all layers
    q_data, k_data, v_data = {}, {}, {}
    hooks = []
    attn_modules = find_attn(model)

    for li, (name, attn) in enumerate(attn_modules):
        def qh(li_=li):
            def fn(m, i, o): q_data[li_] = o.detach()
            return fn
        def kh(li_=li):
            def fn(m, i, o): k_data[li_] = o.detach()
            return fn
        def vh(li_=li):
            def fn(m, i, o): v_data[li_] = o.detach()
            return fn
        hooks.append(attn.q_proj.register_forward_hook(qh()))
        hooks.append(attn.k_proj.register_forward_hook(kh()))
        hooks.append(attn.v_proj.register_forward_hook(vh()))

    model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()

    # Compute per-token importance across layers
    cumul_attn = torch.zeros(S, device=device)  # Σ_l Σ_h α_l,h(t) (attention received)
    value_weighted_attn = torch.zeros(S, device=device)  # Σ_l α_l(t) × ||v_l(t) - v̄||
    per_layer_needle_attn = []  # attention on needle per layer

    query_pos = S - 1  # last token = retrieval query

    for li in range(n_layers):
        name, attn = attn_modules[li]

        Q = q_data[li].view(B, S, n_heads, d_head).transpose(1, 2).float()
        K = k_data[li].view(B, S, n_kv, d_head).transpose(1, 2).float()
        V = v_data[li].view(B, S, n_kv, d_head).transpose(1, 2).float()

        # Apply RoPE
        pos_ids = torch.arange(S, device=device).unsqueeze(0)
        if hasattr(attn, 'rotary_emb'):
            dummy = torch.zeros(B, n_kv, S, d_head, device=device, dtype=Q.dtype)
            cos, sin = attn.rotary_emb(dummy, pos_ids)
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            Q, K = apply_rotary_pos_emb(Q.to(cos.dtype), K.to(cos.dtype), cos, sin)
            Q, K = Q.float(), K.float()

        # Expand K for GQA
        K_exp = K.repeat_interleave(G, dim=1)

        # Attention weights for the LAST query position (retrieval query)
        q_last = Q[0, :, query_pos:query_pos+1, :]  # (n_heads, 1, d)
        scores = (q_last @ K_exp[0].transpose(-1, -2)) / math.sqrt(d_head)  # (n_heads, 1, S)
        # Causal mask
        scores[:, :, query_pos+1:] = float('-inf')
        attn_weights = F.softmax(scores, dim=-1).squeeze(1)  # (n_heads, S)

        # Mean over heads
        mean_attn = attn_weights.mean(0)  # (S,)
        cumul_attn += mean_attn

        # Value uniqueness: ||v_t - v̄|| per token
        V_exp = V.repeat_interleave(G, dim=1)
        v_mean = V_exp[0].mean(dim=1, keepdim=True).mean(dim=0, keepdim=True)  # mean over heads and seq
        v_dev = (V_exp[0] - v_mean).norm(dim=-1).mean(0)  # (S,) mean over heads

        # Importance: α_l(t) × ||v_l(t) - v̄||
        value_weighted_attn += mean_attn * v_dev

        # Needle attention
        needle_attn = mean_attn[needle_start:needle_end].sum().item()
        per_layer_needle_attn.append(needle_attn)

    # Normalize
    cumul_attn_np = cumul_attn.cpu().numpy()
    vw_attn_np = value_weighted_attn.cpu().numpy()

    # Needle stats
    needle_cumul = float(cumul_attn_np[needle_start:needle_end].sum())
    needle_vw = float(vw_attn_np[needle_start:needle_end].sum())
    needle_rank_cumul = int((cumul_attn_np > cumul_attn_np[needle_start:needle_end].max()).sum()) + 1
    needle_rank_vw = int((vw_attn_np > vw_attn_np[needle_start:needle_end].max()).sum()) + 1

    # Top-K analysis: are needles in top-K by each metric?
    topk_sizes = [16, 32, 64, 128, 256]
    needle_in_topk = {}
    for k in topk_sizes:
        top_cumul = set(np.argsort(cumul_attn_np)[-k:])
        top_vw = set(np.argsort(vw_attn_np)[-k:])
        needle_tokens = set(range(needle_start, needle_end))
        needle_in_topk[k] = {
            'cumul': bool(needle_tokens & top_cumul),
            'value_weighted': bool(needle_tokens & top_vw),
        }

    # BOS/sink analysis
    bos_cumul = float(cumul_attn_np[0])
    bos_rank = int((cumul_attn_np > cumul_attn_np[0]).sum()) + 1

    results = {
        'context_len': context_len,
        'depth': depth,
        'needle_range': [needle_start, needle_end],
        'seq_len': S,
        'needle_cumul_attn': round(needle_cumul, 4),
        'needle_value_weighted': round(needle_vw, 4),
        'needle_rank_cumul': needle_rank_cumul,
        'needle_rank_vw': needle_rank_vw,
        'needle_in_topk': needle_in_topk,
        'bos_cumul_attn': round(bos_cumul, 4),
        'bos_rank': bos_rank,
        'per_layer_needle_attn': [round(x, 6) for x in per_layer_needle_attn],
        'total_tokens': S,
    }

    del q_data, k_data, v_data
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--output-dir", default="results/cltq_diagnostic")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"CLTQ DIAGNOSTIC: {args.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, trust_remote_code=True,
        attn_implementation="eager"
    ).to(args.device).eval()

    all_results = []
    for depth in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"\n[depth={depth}]", flush=True)
        r = compute_cross_layer_importance(model, tokenizer, args.device,
                                            context_len=args.context_len, depth=depth)
        all_results.append(r)
        print(f"  Needle cumul_attn rank: {r['needle_rank_cumul']}/{r['total_tokens']}")
        print(f"  Needle value_weighted rank: {r['needle_rank_vw']}/{r['total_tokens']}")
        print(f"  Needle in top-32 (cumul): {r['needle_in_topk'][32]['cumul']}")
        print(f"  Needle in top-32 (vw): {r['needle_in_topk'][32]['value_weighted']}")
        print(f"  BOS rank: {r['bos_rank']}")
        # Per-layer needle attention
        nl_attn = r['per_layer_needle_attn']
        early = sum(nl_attn[:8]) / 8
        mid = sum(nl_attn[8:24]) / 16
        late = sum(nl_attn[24:]) / 8
        print(f"  Needle attn: early(0-7)={early:.4f}, mid(8-23)={mid:.4f}, late(24-31)={late:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Depth':>6s} {'Needle rank (cumul)':>20s} {'Needle rank (vw)':>18s} {'In top-32?':>10s}")
    for r in all_results:
        in32 = "Y" if r['needle_in_topk'][32]['value_weighted'] else "N"
        print(f"{r['depth']:>6.1f} {r['needle_rank_cumul']:>20d} {r['needle_rank_vw']:>18d} {in32:>10s}")

    short = args.model.split("/")[-1].replace(".", "_")
    out_path = out_dir / f"{short}_cltq_diagnostic.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
