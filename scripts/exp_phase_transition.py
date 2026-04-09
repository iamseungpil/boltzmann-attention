#!/usr/bin/env python3
"""
Phase Transition: At how many 2-bit layers does NIAH collapse?

Sweep N = {4, 8, 12, 16, 20, 24, 28, 32} layers quantized to 2-bit.
Rest stays FP16. Test which layers from {first-N, last-N, evenly-spaced-N}.

Key hypothesis: NIAH collapses at some critical N due to cumulative
residual stream corruption, not individual token errors.

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp_phase_transition.py \
    --model mistralai/Mistral-7B-v0.3
"""
import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))


def find_attn_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if 'Attention' in type(m).__name__ and hasattr(m, 'k_proj')]


def uniform_quantize_2d(x, bits):
    if bits >= 16:
        return x
    n_lev = 2 ** bits
    c_min = x.amin(dim=-2, keepdim=True)
    c_max = x.amax(dim=-2, keepdim=True)
    rng = (c_max - c_min).clamp(min=1e-10)
    step = rng / (n_lev - 1)
    return torch.round((x - c_min) / step) * step + c_min


class SelectiveLayerQuantPatcher:
    """Quantize K only for selected layers."""

    def __init__(self, model, quant_layer_indices: list, bits: int = 2):
        self.model = model
        self.quant_layers = set(quant_layer_indices)
        self.bits = bits
        self.active = False
        self.original_forwards = {}
        self._patched = False

        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, 'num_key_value_heads', self.n_heads)
        self.d_head = cfg.hidden_size // self.n_heads
        self.G = self.n_heads // self.n_kv

    def patch(self):
        if self._patched:
            return
        attn_modules = find_attn_modules(self.model)
        for i, (name, attn) in enumerate(attn_modules):
            if i not in self.quant_layers:
                continue
            orig = attn.forward
            self.original_forwards[name] = orig
            attn.forward = self._make_patched(attn, orig, i)
        self._patched = True

    def unpatch(self):
        if not self._patched:
            return
        for name, mod in self.model.named_modules():
            if name in self.original_forwards:
                mod.forward = self.original_forwards[name]
        self.original_forwards.clear()
        self._patched = False

    def _make_patched(self, attn_module, orig_forward, layer_idx):
        patcher = self

        def patched_forward(hidden_states, attention_mask=None,
                            position_ids=None, past_key_value=None,
                            output_attentions=False, use_cache=False,
                            cache_position=None, position_embeddings=None,
                            **kwargs):
            if not patcher.active:
                return orig_forward(
                    hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    output_attentions=output_attentions, use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs)

            bsz, q_len, _ = hidden_states.size()
            num_heads = getattr(attn_module, 'num_heads',
                                attn_module.config.num_attention_heads)
            num_kv_heads = getattr(attn_module, 'num_key_value_heads',
                                   attn_module.config.num_key_value_heads)
            head_dim = attn_module.head_dim
            G = num_heads // num_kv_heads

            query_states = attn_module.q_proj(hidden_states)
            key_states = attn_module.k_proj(hidden_states)
            value_states = attn_module.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            if position_embeddings is not None:
                cos, sin = position_embeddings
            elif hasattr(attn_module, 'rotary_emb'):
                if position_ids is not None:
                    cos, sin = attn_module.rotary_emb(value_states, position_ids)
                else:
                    cos, sin = attn_module.rotary_emb(value_states, seq_len=q_len)
            else:
                cos, sin = None, None

            if cos is not None:
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

            # QUANTIZE K for this layer
            for hk in range(num_kv_heads):
                key_states[:, hk] = uniform_quantize_2d(
                    key_states[:, hk].float(), patcher.bits
                ).to(key_states.dtype)

            # GQA expand
            if num_kv_heads != num_heads:
                key_states = key_states[:, :, None, :, :].expand(
                    bsz, num_kv_heads, G, -1, head_dim
                ).reshape(bsz, num_heads, -1, head_dim)
                value_states = value_states[:, :, None, :, :].expand(
                    bsz, num_kv_heads, G, -1, head_dim
                ).reshape(bsz, num_heads, -1, head_dim)

            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

            if q_len > 1:
                causal = torch.triu(
                    torch.full((q_len, q_len), float('-inf'),
                               device=hidden_states.device), diagonal=1
                ).unsqueeze(0).unsqueeze(0)
                attn_weights = attn_weights + causal

            if attention_mask is not None:
                am = attention_mask
                if am.dim() == 4:
                    if am.shape[-1] > key_states.shape[2]:
                        am = am[:, :, :, -key_states.shape[2]:]
                    if am.shape[-2] > q_len:
                        am = am[:, :, -q_len:, :]
                attn_weights = attn_weights + am

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32
                                     ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = attn_module.o_proj(attn_output.to(hidden_states.dtype))

            return attn_output, attn_weights if output_attentions else None

        return patched_forward


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
    return torch.tensor([tokens])


@torch.no_grad()
def run_niah(model, tokenizer, device, quant_layer_indices, bits=2,
             context_len=4096, depths=[0.3, 0.5, 0.7], repeats=3):
    """Run NIAH with selective layer quantization."""
    if len(quant_layer_indices) == 0:
        # FP16
        scores = []
        for d in depths:
            for _ in range(repeats):
                ids = make_niah_prompt(tokenizer, context_len, d).to(device)
                S = ids.shape[1]
                gen = model.generate(ids, max_new_tokens=20, do_sample=False)
                text = tokenizer.decode(gen[0, S:], skip_special_tokens=True)
                scores.append(1.0 if "7392" in text else 0.0)
                del gen; torch.cuda.empty_cache()
        return sum(scores) / len(scores)

    patcher = SelectiveLayerQuantPatcher(model, quant_layer_indices, bits)
    patcher.patch()
    patcher.active = True

    scores = []
    for d in depths:
        for _ in range(repeats):
            ids = make_niah_prompt(tokenizer, context_len, d).to(device)
            S = ids.shape[1]
            gen = model.generate(ids, max_new_tokens=20, do_sample=False)
            text = tokenizer.decode(gen[0, S:], skip_special_tokens=True)
            scores.append(1.0 if "7392" in text else 0.0)
            del gen; torch.cuda.empty_cache()

    patcher.active = False
    patcher.unpatch()
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--output-dir", default="results/phase_transition")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"{'='*60}")
    print(f"PHASE TRANSITION: {args.model}, {args.bits}-bit")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, trust_remote_code=True,
        attn_implementation="eager"
    ).to(args.device).eval()

    n_layers = model.config.num_hidden_layers
    results = {}

    # FP16 baseline
    print("\n[FP16 baseline]", flush=True)
    score = run_niah(model, tokenizer, args.device, [], context_len=args.context_len)
    results['fp16'] = {"n_quant_layers": 0, "niah": round(score, 3), "layers": []}
    print(f"  NIAH = {score:.3f}", flush=True)

    # Sweep: first-N layers quantized
    for N in [4, 8, 12, 16, 20, 24, 28, 32]:
        if N > n_layers:
            continue
        layers = list(range(N))
        tag = f"first_{N}"
        print(f"\n[{tag}] ({N}/{n_layers} layers at {args.bits}-bit)", flush=True)
        score = run_niah(model, tokenizer, args.device, layers,
                         bits=args.bits, context_len=args.context_len)
        results[tag] = {"n_quant_layers": N, "niah": round(score, 3),
                        "layers": layers, "pattern": "first_N"}
        print(f"  NIAH = {score:.3f}", flush=True)

    # Sweep: last-N layers quantized
    for N in [4, 8, 12, 16, 20, 24, 28, 32]:
        if N > n_layers:
            continue
        layers = list(range(n_layers - N, n_layers))
        tag = f"last_{N}"
        print(f"\n[{tag}] ({N}/{n_layers} layers at {args.bits}-bit)", flush=True)
        score = run_niah(model, tokenizer, args.device, layers,
                         bits=args.bits, context_len=args.context_len)
        results[tag] = {"n_quant_layers": N, "niah": round(score, 3),
                        "layers": layers, "pattern": "last_N"}
        print(f"  NIAH = {score:.3f}", flush=True)

    # Sweep: evenly-spaced N layers
    for N in [4, 8, 16]:
        if N > n_layers:
            continue
        step = n_layers // N
        layers = [i * step for i in range(N)]
        tag = f"even_{N}"
        print(f"\n[{tag}] ({N}/{n_layers} layers at {args.bits}-bit)", flush=True)
        score = run_niah(model, tokenizer, args.device, layers,
                         bits=args.bits, context_len=args.context_len)
        results[tag] = {"n_quant_layers": N, "niah": round(score, 3),
                        "layers": layers, "pattern": "even_N"}
        print(f"  NIAH = {score:.3f}", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Config':<15s} {'N layers':>9s} {'NIAH':>6s}")
    for tag in sorted(results.keys(), key=lambda k: results[k]['n_quant_layers']):
        r = results[tag]
        print(f"{tag:<15s} {r['n_quant_layers']:>9d} {r['niah']:>6.3f}")

    short = args.model.split("/")[-1].replace(".", "_")
    out_path = out_dir / f"{short}_phase_transition_{args.bits}bit.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
