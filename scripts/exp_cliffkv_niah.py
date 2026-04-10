#!/usr/bin/env python3
"""
CliffKV-style NIAH diagnostic: does selective promotion on the attention path
recover retrieval under a low-bit key approximation?

Core experiment in the current diagnostic wrapper:
1. Quantize all K to 2-bit for the attention computation
2. Compute proxy scores from the low-bit K
3. Select top-K tokens by proxy score
4. Reconstruct those K tokens at higher precision for the attention path
5. Run attention with mixed-precision K
6. Test whether bounded NIAH recovers

Important limitation:
the underlying cache object remains FP16 in this script. The experiment is
useful for selector/path diagnostics, but it is not yet a storage-valid
compressed-cache benchmark.
"""
import argparse
import gc
import json
import math
import os
import random
import socket
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.bfloat16


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_metadata() -> dict:
    return {
        "intent": "test whether score-guided selective promotion produces a retrieval-recovery signal on the current attention-path proxy harness",
        "hypothesis": "promoting a small top-k subset of low-bit attention-path keys should improve bounded NIAH over the uniform_2bit proxy baseline",
        "verification": "bounded proxy-harness NIAH sweep over promote_k and promote_bits against fp16, uniform_2bit, and uniform_3bit references",
        "kill_criterion": "drop if selective promotion fails to improve over the uniform_2bit proxy baseline on the same prompt grid",
        "limitations": {
            "claim_scope": "attention_path_proxy_diagnostic",
            "evaluation_mode": "attention_path_diagnostic",
            "cache_storage_validity": (
                "Current implementation keeps the underlying KV cache in FP16 and applies "
                "promotion only on the attention path. These results are attention-path proxies, "
                "not compressed-cache storage measurements."
            ),
            "budget_note": (
                "Effective average bits in this script are upper-bound proxies for accessed "
                "precision, not realized stored KV-cache budgets."
            ),
        },
    }


def make_run_metadata(args) -> dict:
    return {
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "limitations": build_metadata()["limitations"],
    }


def make_output_paths(out_dir: Path, model_name: str, experiment_tag: str):
    short = model_name.split("/")[-1].replace(".", "_")
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return (
        out_dir / f"{short}_{experiment_tag}_{stamp}.json",
        out_dir / f"{short}_{experiment_tag}_latest.json",
    )


def result_bits_for_method(method: str) -> int:
    if method == "fp16":
        return 16
    if method == "uniform_3bit":
        return 3
    return 2


def uniform_quantize_2d(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-dim uniform quantization for (seq, d) or (batch, seq, d)."""
    if bits >= 16:
        return x
    n_lev = 2 ** bits
    c_min = x.amin(dim=-2, keepdim=True)
    c_max = x.amax(dim=-2, keepdim=True)
    rng = (c_max - c_min).clamp(min=1e-10)
    step = rng / (n_lev - 1)
    return torch.round((x - c_min) / step) * step + c_min


def run_self_tests() -> None:
    x = torch.tensor([[0.0, 2.0], [1.0, 4.0], [2.0, 6.0]], dtype=torch.float32)
    q = uniform_quantize_2d(x, bits=2)
    assert q.shape == x.shape
    assert torch.allclose(q[0], x[0])
    assert torch.allclose(q[-1], x[-1])
    meta = build_metadata()
    assert "attention-path proxies" in meta["limitations"]["cache_storage_validity"]
    print("[PASS] exp_cliffkv_niah self-tests")


def align_attention_mask(attention_mask, q_len: int, kv_len: int):
    """Trim a broadcastable attention mask to the active query/key lengths."""
    if attention_mask is None:
        return None
    aligned = attention_mask
    if aligned.dim() == 4:
        if aligned.shape[-1] > kv_len:
            aligned = aligned[..., -kv_len:]
        if aligned.shape[-2] > q_len:
            aligned = aligned[..., -q_len:, :]
    elif aligned.dim() == 3:
        if aligned.shape[-1] > kv_len:
            aligned = aligned[..., -kv_len:]
        if aligned.shape[-2] > q_len:
            aligned = aligned[..., -q_len:, :]
    elif aligned.dim() == 2 and aligned.shape[-1] > kv_len:
        aligned = aligned[..., -kv_len:]
    return aligned


def find_attn_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if 'Attention' in type(m).__name__ and hasattr(m, 'k_proj')]


class CliffKVPatcher:
    """Patches attention to test CliffKV: 2-bit base + selective promotion."""

    def __init__(self, model, promote_k: int = 64, promote_bits: int = 16):
        """
        Args:
            model: the LLM
            promote_k: number of tokens to promote per head
            promote_bits: precision for promoted tokens (3, 4, or 16=FP16)
        """
        self.model = model
        self.promote_k = promote_k
        self.promote_bits = promote_bits
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
        for i, (name, attn) in enumerate(find_attn_modules(self.model)):
            orig = attn.forward
            self.original_forwards[name] = orig
            attn.forward = self._make_patched(attn, orig)
        self._patched = True

    def unpatch(self):
        if not self._patched:
            return
        for name, mod in self.model.named_modules():
            if name in self.original_forwards:
                mod.forward = self.original_forwards[name]
        self.original_forwards.clear()
        self._patched = False

    def _make_patched(self, attn_module, orig_forward):
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

            # Apply RoPE
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

            if past_key_value is not None:
                if hasattr(past_key_value, 'update'):
                    cache_kwargs = (
                        {"cache_position": cache_position}
                        if cache_position is not None else None
                    )
                    layer_idx = getattr(attn_module, "layer_idx", 0)
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, layer_idx, cache_kwargs
                    )
                else:
                    past_key, past_value = past_key_value
                    key_states = torch.cat([past_key, key_states], dim=2)
                    value_states = torch.cat([past_value, value_states], dim=2)

            # === CLIFFKV CORE ===
            # Operate on the full cache, not just the current token.
            key_fp16 = key_states.clone()
            kv_len = key_fp16.shape[2]

            # Step 1: Quantize ALL keys to 2-bit
            key_2bit = torch.zeros_like(key_states)
            for hk in range(num_kv_heads):
                key_2bit[:, hk] = uniform_quantize_2d(key_states[:, hk].float(), 2).to(key_states.dtype)

            # Step 2: For each query head, compute proxy scores from 2-bit K
            # and select top-K tokens for promotion
            key_mixed = key_2bit.clone()

            # Expand to match query heads for scoring
            key_2bit_exp = key_2bit.repeat_interleave(G, dim=1)

            # Proxy scores: Q @ K_2bit^T (all query positions vs all key positions)
            proxy_scores = torch.matmul(
                query_states.float(), key_2bit_exp.float().transpose(-1, -2)
            ) / math.sqrt(head_dim)  # (bsz, n_heads, q_len, kv_len)

            # For generation (q_len=1), proxy_scores is (bsz, n_heads, 1, kv_len)
            # For prefill (q_len>1), use the last query position for selection
            if q_len > 1:
                # Use max over query positions for each KV head
                proxy_for_select = proxy_scores[:, :, -1, :]  # (bsz, n_heads, kv_len)
            else:
                proxy_for_select = proxy_scores.squeeze(-2)  # (bsz, n_heads, kv_len)

            # Aggregate across query heads in each GQA group
            # -> (bsz, n_kv_heads, kv_len)
            proxy_per_kv = proxy_for_select.view(bsz, num_kv_heads, G, -1).max(dim=2).values

            # Select top-K tokens per KV head
            K_promote = min(patcher.promote_k, kv_len)
            _, topk_idx = proxy_per_kv.topk(K_promote, dim=-1)  # (bsz, n_kv, K)

            # Step 3: Promote selected tokens to higher precision
            hi_precision = []
            for hk in range(num_kv_heads):
                if patcher.promote_bits >= 16:
                    hi_precision.append(key_fp16[:, hk])
                else:
                    hi_precision.append(
                        uniform_quantize_2d(
                            key_fp16[:, hk].float(), patcher.promote_bits
                        ).to(key_states.dtype)
                    )
            for b in range(bsz):
                for hk in range(num_kv_heads):
                    idx = topk_idx[b, hk]  # (K,)
                    key_mixed[b, hk, idx] = hi_precision[hk][b, idx]

            # Step 4: Standard attention with mixed-precision K
            if num_kv_heads != num_heads:
                n_rep = G
                key_mixed = key_mixed[:, :, None, :, :].expand(
                    bsz, num_kv_heads, n_rep, -1, head_dim
                ).reshape(bsz, num_heads, -1, head_dim)
                value_states = value_states[:, :, None, :, :].expand(
                    bsz, num_kv_heads, n_rep, -1, head_dim
                ).reshape(bsz, num_heads, -1, head_dim)

            attn_weights = torch.matmul(
                query_states, key_mixed.transpose(2, 3)
            ) / math.sqrt(head_dim)

            am = align_attention_mask(attention_mask, q_len=q_len, kv_len=key_mixed.shape[2])
            if am is not None:
                if am.dim() == 2:
                    am = am[:, None, None, :]
                elif am.dim() == 3:
                    am = am[:, None, :, :]
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

    tokens = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    tokens += filler_toks[:pos] + needle_toks + filler_toks[:avail - pos] + query_toks
    tokens = tokens[:context_len]
    return torch.tensor([tokens])


@torch.no_grad()
def run_niah_test(model, tokenizer, device, method, promote_k=64, promote_bits=16,
                  context_len=4096, depths=[0.1, 0.3, 0.5, 0.7, 0.9], repeats=3,
                  max_new_tokens=20):
    """Run NIAH test with given method."""
    scores = []

    for depth in depths:
        for rep in range(repeats):
            input_ids = make_niah_prompt(tokenizer, context_len, depth).to(device)
            S = input_ids.shape[1]

            if method == 'fp16':
                gen = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            elif method == 'uniform_2bit':
                from exp4_2_v3_full_quant_ppl import AttentionKQuantPatcher
                p = AttentionKQuantPatcher(model, "uniform", 2)
                try:
                    p.patch(); p.active = True
                    gen = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
                finally:
                    p.active = False; p.unpatch()
            elif method == 'uniform_3bit':
                from exp4_2_v3_full_quant_ppl import AttentionKQuantPatcher
                p = AttentionKQuantPatcher(model, "uniform", 3)
                try:
                    p.patch(); p.active = True
                    gen = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
                finally:
                    p.active = False; p.unpatch()
            elif method.startswith('cliffkv'):
                p = CliffKVPatcher(model, promote_k=promote_k, promote_bits=promote_bits)
                try:
                    p.patch(); p.active = True
                    gen = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
                finally:
                    p.active = False; p.unpatch()
            else:
                raise ValueError(f"Unknown method: {method}")

            text = tokenizer.decode(gen[0, S:], skip_special_tokens=True)
            hit = 1.0 if "7392" in text else 0.0
            scores.append(hit)

            del gen
            torch.cuda.empty_cache()

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "method": method,
        "avg_score": round(avg, 3),
        "n_trials": len(scores),
        "bits": result_bits_for_method(method),
        "base_bits": 2,
        "promote_k": promote_k,
        "promote_bits": promote_bits,
        **build_metadata(),
    }


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--self-test", action="store_true")
    pre, _ = bootstrap.parse_known_args()

    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--model", required=not pre.self_test)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--output-dir", default="results/cliffkv")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--promote-ks", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--promote-bits", nargs="+", type=int, default=[16, 3])
    parser.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-tag", default="cliffkv_niah")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"{'='*60}")
    print(f"CLIFFKV NIAH TEST: {args.model}")
    print(f"Context: {args.context_len}")
    print(f"Seed={args.seed}, repeats={args.repeats}, depths={args.depths}")
    meta = build_metadata()
    print(f"Intent: {meta['intent']}")
    print(f"Hypothesis: {meta['hypothesis']}")
    print(f"Verification: {meta['verification']}")
    print(f"Kill criterion: {meta['kill_criterion']}")
    print("Claim scope: attention-path proxy diagnostic only")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, trust_remote_code=True,
        attn_implementation="eager"
    ).to(args.device).eval()

    results = {}

    # Baselines
    for method in ['fp16', 'uniform_2bit', 'uniform_3bit']:
        print(f"\n[{method}]", flush=True)
        r = run_niah_test(model, tokenizer, args.device, method,
                          context_len=args.context_len, depths=args.depths,
                          repeats=args.repeats,
                          max_new_tokens=args.max_new_tokens)
        results[method] = r
        print(f"  NIAH = {r['avg_score']}", flush=True)

    # CliffKV: sweep promote_k and promote_bits
    for promote_k in args.promote_ks:
        for promote_bits in args.promote_bits:
            tag = f"cliffkv_k{promote_k}_b{promote_bits}"
            print(f"\n[{tag}]", flush=True)
            r = run_niah_test(model, tokenizer, args.device, 'cliffkv',
                              promote_k=promote_k, promote_bits=promote_bits,
                              context_len=args.context_len, depths=args.depths,
                              repeats=args.repeats,
                              max_new_tokens=args.max_new_tokens)
            results[tag] = r
            avg_bits = 2.0 + promote_k / args.context_len * (promote_bits - 2)
            r['effective_avg_bits_upper_bound'] = round(avg_bits, 3)
            print(f"  NIAH = {r['avg_score']}, avg_bits_upper_bound = {avg_bits:.3f}", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Method':<30s} {'NIAH':>6s} {'Avg bits':>9s}")
    for tag, r in results.items():
        bits = r.get('effective_avg_bits_upper_bound', 16.0 if 'fp16' in tag else
                      2.0 if '2bit' in tag else 3.0)
        print(f"{tag:<30s} {r['avg_score']:>6.3f} {bits:>9.3f}")

    payload = {
        "experiment_tag": args.experiment_tag,
        "model": args.model,
        "context_len": args.context_len,
        "depths": args.depths,
        "repeats": args.repeats,
        "seed": args.seed,
        "claim_scope": "attention_path_proxy_diagnostic",
        "results": results,
        "metadata": build_metadata(),
        "run_metadata": make_run_metadata(args),
    }
    out_path, latest_path = make_output_paths(out_dir, args.model, args.experiment_tag)
    tmp_path = out_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(out_path)
    latest_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
