#!/usr/bin/env python3
"""
exp_appendix_c4_crosseval — Cross-dataset mode prediction verification
=======================================================================

Reviewer W4 critique (NeurIPS): "PPL only on WikiText-2; the main claim is
which method is correct for which model — does this hold beyond WT2?"

This script tests three Mode-representative models on a non-WikiText-2
evaluation set (C4 web text), using the *same* WikiText-2 calibration as
in the main paper. If the mode classification framework generalizes, the
optimal method for each mode should still win on C4:

  Mistral-7B (Mode A) → Lloyd + sink_k=1 should be best
  Mistral-Nemo-12B (Mode B) → Grid (no sink) should be best
  Qwen2.5-7B (Mode C) → Lloyd + sink_k=1 should be best

The Pre-RoPE per-head PCA basis is fit once on WT2 (as in the main paper);
only the evaluation distribution changes. Same quantizers, same hooks.

Output: reports/axis2_theoretical_verification/exp_appendix_c4_crosseval.json
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # GPU 0 is in use

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
N_CALIB = 2048
EVAL_LENGTH = 2048
N_C4_DOCS = 200  # use first 200 C4 validation docs concatenated
OUT = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification/exp_appendix_c4_crosseval.json')

MODELS = [
    ('mistralai/Mistral-7B-v0.3', 'mistral-7b', 'A'),
    ('mistralai/Mistral-Nemo-Base-2407', 'nemo-12b', 'B'),
    ('Qwen/Qwen2.5-7B', 'qwen-7b', 'C'),
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


def uniform_grid_1d(col, bits):
    n_levels = 2 ** bits
    r = float(np.max(np.abs(col)))
    if r < 1e-12:
        return np.array([0.0] * n_levels, dtype=np.float32)
    return np.linspace(-r, r, n_levels).astype(np.float32)


def compute_ppl(model, ids):
    with torch.no_grad():
        out = model(ids, use_cache=False)
        logits = out.logits[:, :-1].contiguous()
        tgt = ids[:, 1:].contiguous()
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_flat = tgt.reshape(-1)
        total_loss = 0.0
        total_n = 0
        CHUNK = 4096
        for s in range(0, logits_flat.size(0), CHUNK):
            e = min(s + CHUNK, logits_flat.size(0))
            l = logits_flat[s:e].float()
            t = tgt_flat[s:e]
            loss = F.cross_entropy(l, t, reduction='sum')
            total_loss += float(loss.item())
            total_n += (e - s)
            del l
        del logits, logits_flat
        return float(np.exp(total_loss / total_n))


class QuantHook:
    def __init__(self, n_kv, head_dim, V, mean, cents, sink_k):
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.V = V          # (n_kv, head_dim, head_dim) PCA basis
        self.mean = mean    # (n_kv, head_dim)
        self.cents = cents  # (n_kv, head_dim, n_levels)
        self.sink_k = sink_k

    def __call__(self, module, inputs, output):
        # output: (B, T, n_kv * head_dim) for K projection
        x = output
        B, T, D = x.shape
        nk, hd = self.n_kv, self.head_dim
        x4 = x.reshape(B, T, nk, hd).contiguous()
        # center, project, quantize, project back, uncentre
        mean_t = torch.from_numpy(self.mean).to(x.device, x.dtype)
        V_t = torch.from_numpy(self.V).to(x.device, x.dtype)  # (nk, hd, hd)
        cents_t = torch.from_numpy(self.cents).to(x.device, x.dtype)  # (nk,hd,L)

        x_centered = x4 - mean_t[None, None]
        # rotate: (B,T,nk,hd) @ (nk,hd,hd) -> (B,T,nk,hd)
        x_rot = torch.einsum('btnh,nhd->btnd', x_centered, V_t)
        # quantize each (n,d) column to nearest centroid
        diff = (x_rot.unsqueeze(-1) - cents_t[None, None]).abs()
        idx = diff.argmin(dim=-1)  # (B,T,nk,hd)
        x_q = torch.gather(
            cents_t[None, None].expand(B, T, -1, -1, -1), -1, idx.unsqueeze(-1)
        ).squeeze(-1)
        # rotate back
        x_back = torch.einsum('btnd,nhd->btnh', x_q, V_t)
        x_back = x_back + mean_t[None, None]
        # protect first sink_k positions in FP16
        if self.sink_k > 0:
            x_back[:, : self.sink_k] = x4[:, : self.sink_k]
        return x_back.reshape(B, T, D).contiguous()


def calibrate(model, ids, n_layers, n_kv, head_dim):
    """Per-head Pre-RoPE PCA on calibration tokens."""
    pl = [None] * n_layers
    handles = []
    for li in range(n_layers):
        layer = model.model.layers[li]
        def mk(li):
            def h(m, inp, out):
                pl[li] = out.detach().cpu().float().numpy()
            return h
        handles.append(layer.self_attn.k_proj.register_forward_hook(mk(li)))
    with torch.no_grad():
        model(ids, use_cache=False)
    for h in handles: h.remove()
    basis = []
    for li in range(n_layers):
        x = pl[li]  # (1, T, n_kv*head_dim)
        x = x.reshape(x.shape[1], n_kv, head_dim)
        per_head = []
        for h in range(n_kv):
            cols = x[:, h, :]
            mean = cols.mean(0)
            centred = cols - mean
            cov = centred.T @ centred / max(centred.shape[0] - 1, 1)
            w, V = np.linalg.eigh(cov)
            order = np.argsort(-w)
            V = V[:, order]
            per_head.append({'V': V.astype(np.float32), 'mean': mean.astype(np.float32),
                             'data': (centred @ V).astype(np.float32)})
        basis.append(per_head)
    return basis


def fit_cents(basis, qtype, n_layers, n_kv, head_dim, bits=2):
    out = []
    for li in range(n_layers):
        V_arr = np.stack([basis[li][h]['V'] for h in range(n_kv)])
        mean_arr = np.stack([basis[li][h]['mean'] for h in range(n_kv)])
        cents_arr = np.zeros((n_kv, head_dim, 2 ** bits), dtype=np.float32)
        for h in range(n_kv):
            data = basis[li][h]['data']
            for d in range(head_dim):
                col = data[:, d]
                if qtype == 'lloyd':
                    cents_arr[h, d] = lloyd_1d(col, bits)
                else:
                    cents_arr[h, d] = uniform_grid_1d(col, bits)
        out.append({'V': V_arr, 'mean': mean_arr, 'cents': cents_arr})
    return out


def install(model, layer_packs, sink_k, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        p = layer_packs[li]
        hook = QuantHook(n_kv, head_dim, p['V'], p['mean'], p['cents'], sink_k)
        h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
        handles.append(h)
    return handles


def load_c4_text(tokenizer, n_docs=N_C4_DOCS):
    """Load first N C4 validation docs as a single text block."""
    from datasets import load_dataset
    ds = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
    parts = []
    for i, doc in enumerate(ds):
        if i >= n_docs:
            break
        parts.append(doc['text'])
    return '\n\n'.join(parts)


def load_wt2_calib(tokenizer):
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    return '\n\n'.join(texts[:300])


def run_one_model(model_id, short_name, expected_mode, c4_text):
    print(f"\n{'=' * 70}\n[{short_name}] {model_id}  (expected Mode {expected_mode})\n{'=' * 70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // n_q)
    print(f"  loaded n_layers={n_layers} n_kv={n_kv} head_dim={head_dim} in {time.time()-t0:.1f}s", flush=True)

    calib_text = load_wt2_calib(tok)
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')
    print(f"  WT2 calib tokens: {calib_ids.shape[1]}", flush=True)

    print("  [1/3] Calibrating Pre-RoPE per-head PCA on WT2 ...", flush=True)
    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)
    print("  [2/3] Fitting Lloyd + Grid centroids ...", flush=True)
    cents_lloyd = fit_cents(basis, 'lloyd', n_layers, n_kv, head_dim, bits=2)
    cents_grid = fit_cents(basis, 'grid', n_layers, n_kv, head_dim, bits=2)

    print("  [3/3] PPL on C4 (out-of-distribution evaluation) ...", flush=True)
    eval_ids = tok(c4_text, return_tensors='pt', truncation=True, max_length=EVAL_LENGTH)['input_ids'].to('cuda:0')
    print(f"    C4 eval tokens: {eval_ids.shape[1]}", flush=True)

    results = {}
    fp16 = compute_ppl(model, eval_ids)
    results['fp16'] = fp16
    print(f"    FP16 PPL                  = {fp16:.4f}", flush=True)
    for qtype, packs in [('lloyd', cents_lloyd), ('grid', cents_grid)]:
        for sink_k in [0, 1]:
            handles = install(model, packs, sink_k, n_layers, n_kv, head_dim)
            ppl = compute_ppl(model, eval_ids)
            for h in handles: h.remove()
            key = f'{qtype}_sink{sink_k}'
            results[key] = ppl
            print(f"    {qtype:<5} sink={sink_k}  PPL = {ppl:.4f}  (Δ = {ppl - fp16:+.4f})", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        'model': model_id,
        'short_name': short_name,
        'expected_mode': expected_mode,
        'n_layers': n_layers,
        'n_kv': n_kv,
        'head_dim': head_dim,
        'calib_dataset': 'wikitext-2-raw-v1',
        'eval_dataset': 'allenai/c4-en-validation',
        'eval_n_c4_docs': N_C4_DOCS,
        'eval_length': EVAL_LENGTH,
        'results': results,
    }


def determine_winner(results, expected_mode):
    """Mode prediction: A and C → Lloyd+sink, B → Grid+sink0."""
    cfgs = [(k, v) for k, v in results.items() if k != 'fp16']
    cfgs.sort(key=lambda x: x[1])
    winner = cfgs[0][0]
    if expected_mode in ('A', 'C'):
        predicted = 'lloyd_sink1'
    else:  # B
        predicted = 'grid_sink0'
    return winner, predicted, winner == predicted


def main():
    print("=" * 70)
    print("exp_appendix_c4_crosseval — Cross-dataset mode verification")
    print("=" * 70)
    print(f"  calibration: WikiText-2 train (300 paragraphs, 2048 tokens)")
    print(f"  evaluation:  allenai/c4 en validation, first {N_C4_DOCS} docs, L={EVAL_LENGTH}")
    print(f"  models: {len(MODELS)}", flush=True)

    print("\nLoading C4 once (streaming) ...", flush=True)
    # Need a tokenizer just to access load_c4_text; tokenization is per-model
    tmp_tok = AutoTokenizer.from_pretrained(MODELS[0][0], use_fast=True)
    c4_text = load_c4_text(tmp_tok, n_docs=N_C4_DOCS)
    print(f"  loaded {len(c4_text)} chars from C4 validation", flush=True)
    del tmp_tok

    out = {
        'description': 'Cross-dataset W4 verification: WT2-calibrated Pre-RoPE PCA evaluated on C4',
        'protocol': 'Calibrate per-head PCA basis + Lloyd/Grid centroids on WT2 train; '
                    'evaluate PPL on first 200 C4 en validation docs at L=2048.',
        'models': [],
    }

    for model_id, short, mode in MODELS:
        try:
            r = run_one_model(model_id, short, mode, c4_text)
            winner, predicted, match = determine_winner(r['results'], mode)
            r['actual_winner'] = winner
            r['predicted_winner'] = predicted
            r['mode_prediction_correct'] = match
            out['models'].append(r)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
            out['models'].append({'short_name': short, 'error': str(e)})

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}", flush=True)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Cross-dataset (C4) PPL with WT2 calibration")
    print("=" * 70)
    print(f"  {'model':<14} {'mode':<5} {'fp16':>8} {'L':>8} {'L+s1':>8} {'G':>8} {'G+s1':>8}  winner    pred?")
    n_correct = 0
    for r in out['models']:
        if 'error' in r:
            print(f"  {r['short_name']:<14}  ERROR: {r['error'][:50]}")
            continue
        res = r['results']
        marker = '✓' if r['mode_prediction_correct'] else '✗'
        print(f"  {r['short_name']:<14} {r['expected_mode']:<5} "
              f"{res['fp16']:>8.3f} {res['lloyd_sink0']:>8.3f} {res['lloyd_sink1']:>8.3f} "
              f"{res['grid_sink0']:>8.3f} {res['grid_sink1']:>8.3f}  "
              f"{r['actual_winner']:<11} {marker}")
        if r['mode_prediction_correct']:
            n_correct += 1
    n_total = sum(1 for r in out['models'] if 'error' not in r)
    print(f"\n  Mode prediction holds on {n_correct}/{n_total} models when evaluated on C4.")
    print("  (Calibration set unchanged: WikiText-2 train.)")


if __name__ == '__main__':
    main()
