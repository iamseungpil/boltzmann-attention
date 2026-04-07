#!/usr/bin/env python3
"""
V2ab: Token-Based Sink Protection
===================================

v2y revealed that Nemo's high-κ heads attend heavily to NEWLINE and BOS
tokens *across the sequence*, not just pos 0. This suggests that position-
based sink protection (sink_k=1 = protect pos 0) is the wrong abstraction
for Nemo — we should protect SPECIFIC TOKEN IDs wherever they appear.

Token-based sink:
  1. Identify sink token IDs from calibration: most-attended tokens by top-κ heads
  2. At eval, for each position whose token is in the sink set, keep K in FP16
  3. Test if this closes Nemo's Lloyd catastrophe at L=32768

Compared to:
  - Lloyd sink_k=1 (position-based): closes Mistral but not Nemo
  - Grid: closes Nemo but model-specific
  - Lloyd + token-based sink: hopefully universal

We identify sink tokens empirically from the v2y output (top-5 most frequent
attended tokens across top-32 high-κ heads on Nemo):
  \\n\\n\\n, <s>, ' and', ' the', ' ' (space)

This script:
  1. Loads Nemo, calibrates
  2. Constructs sink token set (configurable)
  3. Runs PPL at L ∈ {2048, 8192, 32768} with Lloyd + token-sink
  4. Also tests on Mistral (expected: matches or beats pos-0 sink)
"""
import json, os, time, gc
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

DTYPE = torch.bfloat16
N_CALIB = 2048
EVAL_LENGTHS = [2048, 8192, 32768]
OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')


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
    """
    Per-head PCA + per-dim Lloyd + TOKEN-BASED sink protection.

    sink_mask: boolean tensor (T,) — True where token is a sink (FP16 kept)
    position_sink_k: int — additional pos 0..k-1 protection (default 0)
    """
    def __init__(self, n_kv, head_dim, V_list, mean_list, cents_list, sink_mask, position_sink_k=0):
        self.n_kv = n_kv; self.head_dim = head_dim
        self.V_list = V_list; self.mean_list = mean_list
        self.cents_list = cents_list
        self.sink_mask = sink_mask  # np.bool_ array of length T, or None
        self.position_sink_k = position_sink_k

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
        # Position-based sink (legacy)
        if self.position_sink_k > 0:
            k = min(self.position_sink_k, T)
            x[:, :k, :, :] = x_orig[:, :k, :, :]
        # Token-based sink: restore FP16 where mask is True
        if self.sink_mask is not None:
            mask_t = self.sink_mask[:T] if len(self.sink_mask) >= T else np.pad(self.sink_mask, (0, T-len(self.sink_mask)))
            sink_positions = np.where(mask_t)[0]
            for p in sink_positions:
                x[:, p, :, :] = x_orig[:, p, :, :]
        return torch.from_numpy(x).to(output.device).to(output.dtype).view(B, T, self.n_kv * self.head_dim)


def calibrate(model, ids, n_layers, n_kv, head_dim):
    pl = {}
    def mk(li):
        def h(m, i, o): pl[li] = o.detach().cpu().float().numpy()
        return h
    handles = [model.model.layers[li].self_attn.k_proj.register_forward_hook(mk(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(ids, use_cache=False)
    for h in handles: h.remove()
    basis = {}
    for li in range(n_layers):
        K_all = pl[li].reshape(-1, n_kv, head_dim).astype(np.float32)
        ph = []
        for hk in range(n_kv):
            K = K_all[:, hk, :]; mean = K.mean(axis=0); Kc = K - mean
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1)
            ev, vv = np.linalg.eigh(cov)
            order = np.argsort(ev)[::-1]
            V = vv[:, order].astype(np.float32)
            ph.append({'V': V, 'mean': mean.astype(np.float32), 'eigvals': ev[order], 'K_pca': Kc @ V})
        basis[li] = ph
    return basis


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


def install(model, basis, cents, sink_mask, position_sink_k, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        hook = TokenSinkHook(n_kv, head_dim, V_list, mean_list, cents[li], sink_mask, position_sink_k)
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
    return handles


def identify_sink_tokens(tok, sink_strs):
    """Given list of token strings, return list of token IDs."""
    sink_ids = set()
    for s in sink_strs:
        enc = tok(s, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        for t in enc:
            sink_ids.add(int(t.item()))
        # Also try with leading space
        enc2 = tok(' ' + s, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        for t in enc2:
            sink_ids.add(int(t.item()))
    return sink_ids


def build_sink_mask(ids_tensor, sink_ids):
    """Boolean mask over positions where token is a sink."""
    ids_np = ids_tensor[0].cpu().numpy()
    mask = np.array([int(t) in sink_ids for t in ids_np], dtype=bool)
    return mask


def run_model(model_id, sn, sink_strs):
    print(f"\n{'='*70}\n  {sn}: {model_id}\n{'='*70}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=DTYPE, device_map='cuda:0',
        attn_implementation='sdpa', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // model.config.num_attention_heads)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text = '\n\n'.join(texts[300:3000])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')

    # Identify sink token IDs
    sink_ids = identify_sink_tokens(tok, sink_strs)
    # Always include BOS
    if tok.bos_token_id is not None:
        sink_ids.add(int(tok.bos_token_id))
    print(f"  Sink token set ({len(sink_ids)} ids): {sorted(sink_ids)[:20]}", flush=True)

    # Calibrate
    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)
    cents = fit_cents(basis, n_layers, n_kv, head_dim)
    print(f"  Calibrated in {time.time()-t0:.1f}s", flush=True)

    result = {'model': model_id, 'sink_strs': sink_strs, 'sink_ids': sorted(sink_ids), 'configs': {}}
    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        actual_T = eval_ids.shape[1]
        # Build token mask for this eval sequence
        sink_mask = build_sink_mask(eval_ids, sink_ids)
        n_sinks = int(sink_mask.sum())

        ppl_fp16 = compute_ppl(model, eval_ids)
        print(f"\n  L={L} (T={actual_T}) FP16={ppl_fp16:.4f}, {n_sinks} sink tokens ({n_sinks/actual_T*100:.1f}%)")

        # Config 1: Lloyd no sink
        handles = install(model, basis, cents, None, 0, n_layers, n_kv, head_dim)
        p_ns = compute_ppl(model, eval_ids)
        for h in handles: h.remove()
        # Config 2: Lloyd + position sink_k=1
        handles = install(model, basis, cents, None, 1, n_layers, n_kv, head_dim)
        p_pos = compute_ppl(model, eval_ids)
        for h in handles: h.remove()
        # Config 3: Lloyd + token sink
        handles = install(model, basis, cents, sink_mask, 0, n_layers, n_kv, head_dim)
        p_tok = compute_ppl(model, eval_ids)
        for h in handles: h.remove()
        # Config 4: Lloyd + both
        handles = install(model, basis, cents, sink_mask, 1, n_layers, n_kv, head_dim)
        p_both = compute_ppl(model, eval_ids)
        for h in handles: h.remove()

        print(f"    Lloyd no sink    : {p_ns:.4f}  Δ={p_ns-ppl_fp16:+.4f}")
        print(f"    Lloyd pos-sink=1 : {p_pos:.4f}  Δ={p_pos-ppl_fp16:+.4f}")
        print(f"    Lloyd tok-sink   : {p_tok:.4f}  Δ={p_tok-ppl_fp16:+.4f}")
        print(f"    Lloyd pos+tok    : {p_both:.4f}  Δ={p_both-ppl_fp16:+.4f}", flush=True)

        result['configs'][L] = {
            'actual_T': actual_T,
            'n_sink_tokens': n_sinks,
            'frac_sinks': n_sinks / actual_T,
            'ppl_fp16': ppl_fp16,
            'ppl_no_sink': p_ns,
            'ppl_pos_sink_1': p_pos,
            'ppl_tok_sink': p_tok,
            'ppl_both': p_both,
        }

    del model, tok, basis, cents
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    print("="*70)
    print("V2ab: Token-Based Sink Protection")
    print("="*70, flush=True)
    t_start = time.time()

    # Sink token strings identified from v2y Nemo analysis
    NEMO_SINKS = ['\n\n\n', '\n\n', '\n', '<s>']
    MISTRAL_SINKS = ['\n\n\n', '\n\n', '\n', '<s>']

    results = {}
    for model_id, sn, sinks in [
        ('mistralai/Mistral-7B-v0.3', 'mistral-7b', MISTRAL_SINKS),
        ('mistralai/Mistral-Nemo-Base-2407', 'nemo-12b', NEMO_SINKS),
    ]:
        try:
            results[sn] = run_model(model_id, sn, sinks)
        except Exception as e:
            print(f"ERROR on {sn}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for sn, r in results.items():
        print(f"\n  {sn}:")
        for L in EVAL_LENGTHS:
            c = r['configs'][L]
            print(f"    L={L:<5} FP16={c['ppl_fp16']:.3f} | "
                  f"nosink={c['ppl_no_sink']:.3f} pos1={c['ppl_pos_sink_1']:.3f} "
                  f"tok={c['ppl_tok_sink']:.3f} both={c['ppl_both']:.3f} | "
                  f"{c['n_sink_tokens']}/{c['actual_T']} sinks ({c['frac_sinks']*100:.1f}%)")

    out = OUT_DIR / 'exp_v2ab_token_sink.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
