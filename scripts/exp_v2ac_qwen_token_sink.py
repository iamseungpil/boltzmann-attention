#!/usr/bin/env python3
"""
V2ac: Token-Based Sink on Qwen
================================

v2ab verified that token-based sink beats position-based sink on Mistral and
is essential for Nemo. This extends the test to Qwen2.5-7B (and 1.5B) to
complete the universal claim across all tested architectures.
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

MODELS = [
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
    def __init__(self, n_kv, head_dim, V_list, mean_list, cents_list, sink_mask, position_sink_k=0):
        self.n_kv = n_kv; self.head_dim = head_dim
        self.V_list = V_list; self.mean_list = mean_list
        self.cents_list = cents_list
        self.sink_mask = sink_mask
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
        if self.position_sink_k > 0:
            k = min(self.position_sink_k, T)
            x[:, :k, :, :] = x_orig[:, :k, :, :]
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


def install(model, basis, cents, sink_mask, pos_k, n_layers, n_kv, head_dim):
    handles = []
    for li in range(n_layers):
        V_list = [basis[li][hk]['V'] for hk in range(n_kv)]
        mean_list = [basis[li][hk]['mean'] for hk in range(n_kv)]
        hook = TokenSinkHook(n_kv, head_dim, V_list, mean_list, cents[li], sink_mask, pos_k)
        handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
    return handles


def identify_sink_tokens(tok):
    """Sink tokens: newlines + BOS/EOS variants + common whitespace-heavy delimiters."""
    sink_ids = set()
    for s in ['\n', '\n\n', '\n\n\n']:
        for t in tok(s, add_special_tokens=False, return_tensors='pt')['input_ids'][0]:
            sink_ids.add(int(t.item()))
    # Qwen has no BOS but has EOS and <|im_start|>, <|im_end|>, <|endoftext|>
    if tok.bos_token_id is not None:
        sink_ids.add(int(tok.bos_token_id))
    if tok.eos_token_id is not None:
        sink_ids.add(int(tok.eos_token_id))
    # Qwen special
    for sp in ['<|endoftext|>', '<|im_start|>', '<|im_end|>']:
        try:
            ids = tok(sp, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
            for t in ids:
                sink_ids.add(int(t.item()))
        except Exception:
            pass
    return sink_ids


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
        attn_implementation='sdpa', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', None) or (model.config.hidden_size // model.config.num_attention_heads)
    print(f"  n_layers={n_layers}, n_kv={n_kv}, head_dim={head_dim}, loaded in {time.time()-t0:.1f}s", flush=True)

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]
    calib_text = '\n\n'.join(texts[:300])
    eval_text = '\n\n'.join(texts[300:3000])
    calib_ids = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB)['input_ids'].to('cuda:0')

    sink_ids = identify_sink_tokens(tok)
    print(f"  Sink token set ({len(sink_ids)} ids): {sorted(sink_ids)[:20]}", flush=True)

    basis = calibrate(model, calib_ids, n_layers, n_kv, head_dim)
    cents = fit_cents(basis, n_layers, n_kv, head_dim)
    print(f"  Calibrated in {time.time()-t0:.1f}s", flush=True)

    result = {'model': model_id, 'sink_ids': sorted(sink_ids), 'configs': {}}
    for L in EVAL_LENGTHS:
        eval_ids = tok(eval_text, return_tensors='pt', truncation=True, max_length=L)['input_ids'].to('cuda:0')
        actual_T = eval_ids.shape[1]
        sink_mask = build_sink_mask(eval_ids, sink_ids)
        n_sinks = int(sink_mask.sum())

        ppl_fp16 = compute_ppl(model, eval_ids)
        print(f"\n  L={L} (T={actual_T}) FP16={ppl_fp16:.4f}, {n_sinks} sink tokens ({n_sinks/actual_T*100:.1f}%)")

        handles = install(model, basis, cents, None, 0, n_layers, n_kv, head_dim)
        p_ns = compute_ppl(model, eval_ids)
        for h in handles: h.remove()
        handles = install(model, basis, cents, None, 1, n_layers, n_kv, head_dim)
        p_pos = compute_ppl(model, eval_ids)
        for h in handles: h.remove()
        handles = install(model, basis, cents, sink_mask, 0, n_layers, n_kv, head_dim)
        p_tok = compute_ppl(model, eval_ids)
        for h in handles: h.remove()
        handles = install(model, basis, cents, sink_mask, 1, n_layers, n_kv, head_dim)
        p_both = compute_ppl(model, eval_ids)
        for h in handles: h.remove()

        print(f"    Lloyd no sink    : {p_ns:.4f}  Δ={p_ns-ppl_fp16:+.4f}")
        print(f"    Lloyd pos-sink=1 : {p_pos:.4f}  Δ={p_pos-ppl_fp16:+.4f}")
        print(f"    Lloyd tok-sink   : {p_tok:.4f}  Δ={p_tok-ppl_fp16:+.4f}")
        print(f"    Lloyd pos+tok    : {p_both:.4f}  Δ={p_both-ppl_fp16:+.4f}", flush=True)

        result['configs'][L] = {
            'actual_T': actual_T, 'n_sink_tokens': n_sinks, 'frac_sinks': n_sinks/actual_T,
            'ppl_fp16': ppl_fp16, 'ppl_no_sink': p_ns, 'ppl_pos_sink_1': p_pos,
            'ppl_tok_sink': p_tok, 'ppl_both': p_both,
        }

    del model, tok, basis, cents
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    print("="*70)
    print("V2ac: Qwen Token-Based Sink Test")
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
    print("SUMMARY")
    print("="*70)
    for sn, r in results.items():
        print(f"\n  {sn}:")
        for L in EVAL_LENGTHS:
            c = r['configs'][L]
            print(f"    L={L:<5} FP16={c['ppl_fp16']:.3f} | "
                  f"nosink={c['ppl_no_sink']:.3f} pos1={c['ppl_pos_sink_1']:.3f} "
                  f"tok={c['ppl_tok_sink']:.3f} both={c['ppl_both']:.3f} | "
                  f"{c['n_sink_tokens']}/{c['actual_T']} ({c['frac_sinks']*100:.1f}%)")

    out = OUT_DIR / 'exp_v2ac_qwen_token_sink.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Runtime: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}m)")


if __name__ == '__main__':
    main()
