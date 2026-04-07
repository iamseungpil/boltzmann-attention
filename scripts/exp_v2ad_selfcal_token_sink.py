#!/usr/bin/env python3
"""
V2ad: Self-Calibrated Token-Based Sink
=========================================

v2ab: manually-specified sink tokens (newlines, BOS) work for Mistral and
help Nemo significantly but only partially.
v2ac: same approach FAILS on Qwen — the hardcoded newline token IDs don't
appear in eval text because Qwen's BPE merges newlines with surrounding text.

Solution: self-calibrating sink set.
  1. Run calibration forward pass with attention output
  2. For each top-32 high-κ head, find top-10 most-attended positions
  3. Extract TOKEN IDs at those positions (not positions themselves)
  4. Deduplicate → sink token ID set
  5. At eval time, keep K in FP16 for any position whose token ID is in set

This is model-agnostic and handles BPE merging correctly.

Runs on Mistral, Nemo, Qwen2.5-7B, Qwen2.5-1.5B for full universal validation.
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from collections import Counter

DTYPE = torch.bfloat16
N_CALIB = 2048
EVAL_LENGTHS = [2048, 8192, 32768]
TOP_K_HEADS = 32
TOP_K_POSITIONS_PER_HEAD = 10
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')

MODELS = [
    'mistralai/Mistral-7B-v0.3',
    'mistralai/Mistral-Nemo-Base-2407',
    'Qwen/Qwen2.5-7B',
    'Qwen/Qwen2.5-1.5B',
]


def lloyd_1d(col, bits, n_iter=15):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    c = np.sort(np.percentile(col, pcts)).astype(np.float64)
    for _ in range(n_iter):
        b = (c[:-1] + c[1:]) / 2
        idx = np.searchsorted(b, col)
        new_c = c.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - c)) < 1e-6:
            break
        c = new_c
    return c.astype(np.float32)


def compute_ppl(model, ids):
    with torch.no_grad():
        out = model(ids, use_cache=False)
        logits = out.logits[:, :-1].contiguous()
        tgt = ids[:, 1:].contiguous()
        B, Tm1, V = logits.shape
        lf = logits.reshape(-1, V); tf = tgt.reshape(-1)
        total_loss = 0.0; total_n = 0
        CHUNK = 4096
        for s in range(0, lf.size(0), CHUNK):
            e = min(s + CHUNK, lf.size(0))
            l = lf[s:e].float()
            loss = F.cross_entropy(l, tf[s:e], reduction='sum')
            total_loss += float(loss.item())
            total_n += (e - s)
            del l
        return float(np.exp(total_loss / total_n))


class TokenSinkHook:
    def __init__(self, n_kv, head_dim, V_list, mean_list, cents_list, sink_mask):
        self.n_kv = n_kv; self.head_dim = head_dim
        self.V_list = V_list; self.mean_list = mean_list
        self.cents_list = cents_list; self.sink_mask = sink_mask
    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x_orig = output.view(B, T, self.n_kv, self.head_dim).float().cpu().numpy()
        x = x_orig.copy()
        for hk in range(self.n_kv):
            V = self.V_list[hk]; m = self.mean_list[hk]; cents = self.cents_list[hk]
            data = x[:, :, hk, :].reshape(-1, self.head_dim)
            K_c = data - m; K_pca = K_c @ V
            K_q = K_pca.copy()
            for j in range(self.head_dim):
                cj = cents[j]
                if cj is None or len(cj) == 1:
                    K_q[:, j] = cj[0] if cj is not None else 0.0
                else:
                    bnd = (cj[:-1] + cj[1:]) / 2
                    idx = np.searchsorted(bnd, K_pca[:, j])
                    K_q[:, j] = cj[idx]
            K_rec = K_q @ V.T + m
            x[:, :, hk, :] = K_rec.reshape(B, T, self.head_dim)
        if self.sink_mask is not None:
            mask_t = self.sink_mask[:T] if len(self.sink_mask) >= T else np.pad(self.sink_mask, (0, T-len(self.sink_mask)))
            sink_positions = np.where(mask_t)[0]
            for p in sink_positions:
                x[:, p, :, :] = x_orig[:, p, :, :]
        return torch.from_numpy(x).to(output.device).to(output.dtype).view(B, T, self.n_kv * self.head_dim)


def calibrate_and_find_sink_tokens(model, tok, calib_ids, n_layers, n_kv, n_q, head_dim):
    """
    Calibration pass: capture k_proj + attention weights.
    Identify sink token IDs from top-attended positions in top-κ heads.
    """
    q_per_kv = n_q // n_kv
    captured_k = {}
    def mk(li):
        def h(m, i, o): captured_k[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        out = model(calib_ids, use_cache=False, output_attentions=True)
    attn_weights = out.attentions
    for h in handles: h.remove()

    # Compute κ for each head
    basis = {}
    head_stats = []
    for li in range(n_layers):
        K_all = captured_k[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        ph = []
        for hk in range(n_kv):
            K = K_all[:, hk, :]; mean = K.mean(axis=0); Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev, vv = np.linalg.eigh(cov)
            order = np.argsort(ev)[::-1]
            V = vv[:, order].astype(np.float32)
            ev = ev[order]
            kappa = float(ev[0] / max(ev[-1], 1e-12))
            head_stats.append({'layer': li, 'kv_head': hk, 'kappa': kappa})
            ph.append({'V': V, 'mean': mean.astype(np.float32), 'eigvals': ev, 'K_pca': Kc @ V})
        basis[li] = ph

    head_stats.sort(key=lambda x: -x['kappa'])
    top_heads = head_stats[:TOP_K_HEADS]

    # For each top head, find top-K attended positions
    sink_token_counter = Counter()
    calib_ids_np = calib_ids[0].cpu().numpy()

    # Also track pos0 attention mass for mode detection
    pos0_masses = []

    for rec in top_heads:
        li = rec['layer']; hk = rec['kv_head']
        attn = attn_weights[li][0].float().cpu().numpy()
        q_start = hk * q_per_kv
        q_end = q_start + q_per_kv
        attn_avg = attn[q_start:q_end].mean(axis=0).mean(axis=0)
        pos0_masses.append(float(attn_avg[0]))
        top_pos = np.argsort(-attn_avg)[:TOP_K_POSITIONS_PER_HEAD]
        for p in top_pos:
            token_id = int(calib_ids_np[p])
            sink_token_counter[token_id] += 1

    # Sink token set: tokens appearing in ≥2 heads' top-K (robust)
    MIN_COUNT = 2
    sink_token_ids = set(tid for tid, cnt in sink_token_counter.items() if cnt >= MIN_COUNT)
    # Also ensure at least top-10 by frequency regardless
    top_by_freq = [tid for tid, _ in sink_token_counter.most_common(10)]
    sink_token_ids.update(top_by_freq)

    mean_pos0 = float(np.mean(pos0_masses))

    # Decode for reporting
    sink_decoded = [(tid, tok.decode([tid]), sink_token_counter[tid]) for tid in sorted(sink_token_ids, key=lambda t: -sink_token_counter[t])]

    return basis, sink_token_ids, sink_decoded, mean_pos0, head_stats


def fit_cents(basis, n_layers, n_kv, head_dim, bits=2):
    out = {}
    for li in range(n_layers):
        per = []
        for hk in range(n_kv):
            K_pca = basis[li][hk]['K_pca']
            cents = []
            for j in range(head_dim):
                cents.append(lloyd_1d(K_pca[:, j], bits, 15))
            per.append(cents)
        out[li] = per
    return out


def install(model, basis, cents, sink_mask, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        hook = TokenSinkHook(n_kv, head_dim, V_list, mean_list, cents[li], sink_mask)
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
    return handles


def build_sink_mask(ids_tensor, sink_ids):
    ids_np = ids_tensor[0].cpu().numpy()
    return np.array([int(t) in sink_ids for t in ids_np], dtype=bool)


def run_model(model_id):
    sn = model_id.split('/')[-1].lower()
    print(f"\n{'='*70}\n  {sn}: {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
        output_attentions=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // n_q)
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_q={n_q}, head_dim={head_dim}, loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text = '\n\n'.join(texts[300:3000])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')

    # Self-calibrated sink discovery
    basis, sink_ids, sink_decoded, mean_pos0, head_stats = calibrate_and_find_sink_tokens(
        model, tok, calib_ids, n_layers, n_kv, n_q, head_dim
    )
    print(f"  Top-32 pos0 attention mass: {mean_pos0*100:.1f}%", flush=True)
    print(f"  Self-calibrated sink set ({len(sink_ids)} IDs):")
    for tid, decoded, cnt in sink_decoded[:15]:
        ds_short = decoded.replace('\n', '\\n').replace('\t', '\\t')[:15]
        print(f"    tid={tid:>6} cnt={cnt:>3} {ds_short!r}")

    cents = fit_cents(basis, n_layers, n_kv, head_dim)
    print(f"  Calibrated in {time.time()-t0:.1f}s", flush=True)

    result = {
        'model': model_id,
        'n_layers': n_layers, 'n_kv': n_kv,
        'mean_pos0_top32': mean_pos0,
        'sink_ids': sorted(sink_ids),
        'sink_decoded_top15': [(int(tid), d, int(cnt)) for tid, d, cnt in sink_decoded[:15]],
        'configs': {},
    }

    # Switch to sdpa for eval (more memory-efficient at long L)
    del model
    gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='sdpa', low_cpu_mem_usage=True,
    )
    model.eval()

    for L in EVAL_LENGTHS:
        try:
            eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
            actual_T = eval_ids.shape[1]
            sink_mask = build_sink_mask(eval_ids, sink_ids)
            n_sinks = int(sink_mask.sum())

            ppl_fp16 = compute_ppl(model, eval_ids)
            print(f"\n  L={L} (T={actual_T}) FP16={ppl_fp16:.4f}, {n_sinks} sink tokens ({n_sinks/actual_T*100:.1f}%)")

            handles = install(model, basis, cents, None, n_layers, n_kv, head_dim)
            p_ns = compute_ppl(model, eval_ids)
            for h in handles: h.remove()
            handles = install(model, basis, cents, sink_mask, n_layers, n_kv, head_dim)
            p_tok = compute_ppl(model, eval_ids)
            for h in handles: h.remove()

            print(f"    Lloyd no sink    : {p_ns:.4f}  Δ={p_ns-ppl_fp16:+.4f}")
            print(f"    Lloyd selfcal-tok: {p_tok:.4f}  Δ={p_tok-ppl_fp16:+.4f}", flush=True)

            result['configs'][L] = {
                'actual_T': actual_T, 'n_sinks': n_sinks, 'frac_sinks': n_sinks/actual_T,
                'ppl_fp16': ppl_fp16, 'ppl_no_sink': p_ns, 'ppl_tok_sink': p_tok,
            }
        except Exception as e:
            print(f"  ERROR at L={L}: {e}")

    del model, tok, basis, cents
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    print("="*70)
    print("V2ad: Self-Calibrated Token Sink (Universal)")
    print("="*70, flush=True)
    t_start = time.time()

    results = {}
    for mid in MODELS:
        try:
            sn = mid.split('/')[-1].lower()
            results[sn] = run_model(mid)
        except Exception as e:
            print(f"ERROR on {mid}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*70)
    print("SUMMARY — Self-Calibrated Token Sink Universal Test")
    print("="*70)
    for sn, r in results.items():
        print(f"\n  {sn} (pos0_mass={r['mean_pos0_top32']*100:.1f}%, {len(r['sink_ids'])} sink ids)")
        for L in EVAL_LENGTHS:
            if L not in r['configs']:
                continue
            c = r['configs'][L]
            delta_tok = c['ppl_tok_sink'] - c['ppl_fp16']
            delta_ns = c['ppl_no_sink'] - c['ppl_fp16']
            closed = (delta_ns - delta_tok) / max(delta_ns, 1e-6) * 100
            print(f"    L={L:<5} FP16={c['ppl_fp16']:.3f} nosink={c['ppl_no_sink']:.3f} "
                  f"tok={c['ppl_tok_sink']:.3f} ({c['frac_sinks']*100:.1f}% sinks, {closed:.0f}% closed)")

    out = OUT_DIR / 'exp_v2ad_selfcal_token_sink.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
