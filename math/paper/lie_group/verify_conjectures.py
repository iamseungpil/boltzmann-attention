#!/usr/bin/env python3
"""
Conjecture Verification for Lie Group Unification Paper
GPU 0: GPT-2 Medium, GPU 1: Qwen2.5-7B (parallel)
Usage: source set.env && python3 verify_conjectures.py --gpu0 gpt2 --gpu1 qwen
"""
import argparse, json, time, os, sys, traceback
import numpy as np
from scipy import stats
from pathlib import Path
# Single model run (Qwen only, RoPE model)

# Existing PPL data
GPT2_PPL = {
    'fp16': 21.3606,
    'uniform_2': 99.2370, 'uniform_3': 24.9662, 'uniform_4': 21.7656,
    'kivi_2': 138.5630, 'kivi_3': 27.1509, 'kivi_4': 21.9616,
    'fokvq_2': 25.6516, 'fokvq_3': 21.6762, 'fokvq_4': 21.4124,
    'fokvq_qw_2': 28.0034, 'fokvq_qw_3': 21.8133, 'fokvq_qw_4': 21.4282,
    'quip_2': 94.5450, 'quip_3': 24.1853, 'quip_4': 21.7251,
}
QWEN_PPL = {
    'fp16': 7.6803,
    'kivi_2': 72.6203, 'kivi_3': 8.8739, 'kivi_4': 7.8417,
    'quip_2': 157.5566, 'quip_3': 9.9678, 'quip_4': 7.9972,
    'fokvq_2': 56.9985, 'fokvq_3': 10.3131, 'fokvq_4': 8.2108,
    'fokvq_e2_2': 10.5054,
}

def extract_kv_cache(model_name, device, max_tokens=2000):
    """Forward pass, collect K (post-RoPE from past_kv) and Q (from projection)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    # WikiText-2
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    ids = tokenizer.encode(text, return_tensors="pt", truncation=False)[:, :max_tokens].to(device)

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_q = cfg.num_attention_heads
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // n_q
    G = n_q // n_kv

    print(f"  layers={n_layers}, q_heads={n_q}, kv_heads={n_kv}, d_head={d_head}, G={G}, tokens={ids.shape[1]}")

    keys, queries = {}, {}

    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=True)
        past_kv = out.past_key_values
        hidden = out.hidden_states  # (n_layers+1,)

        for l, item in enumerate(past_kv):
            # DynamicCache yields (K, V, ...) tuples per layer
            K_tensor = item[0] if isinstance(item, tuple) else item.keys
            K = K_tensor[0].cpu().float().numpy()  # (n_kv, seq, d_head)
            for h in range(n_kv):
                keys[(l, h)] = K[h]

            # Q from projection
            hs = hidden[l]  # input to layer l
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'layers'):  # Qwen/Llama
                    Q_all = model.model.layers[l].self_attn.q_proj(hs)[0].cpu().float().numpy()
                elif hasattr(model, 'transformer'):  # GPT-2
                    qkv = model.transformer.h[l].attn.c_attn(hs)
                    Q_all = qkv[0, :, :cfg.hidden_size].cpu().float().numpy()
                else:
                    continue
                Q_r = Q_all.reshape(-1, n_q, d_head)
                for g in range(n_q):
                    queries[(l, g)] = Q_r[:, g, :]
            except Exception as e:
                print(f"  Warning: Q extraction failed at layer {l}: {e}")

    del model
    torch.cuda.empty_cache()

    info = {'n_layers': n_layers, 'n_heads': n_q, 'n_kv_heads': n_kv,
            'd_head': d_head, 'G': G, 'model_name': model_name, 'n_tokens': ids.shape[1]}
    return keys, queries, info


def experiment_A(keys, info):
    """Verify G_PCA <= G_block4 <= G_block2 <= G_norot <= G_uniform for all heads."""
    print("\n" + "="*70)
    print("EXPERIMENT A: MSE Order Verification")
    print("="*70)
    d = info['d_head']
    results = []

    for (l, h), K in keys.items():
        n = K.shape[0]
        if n < d: continue
        Kc = K - K.mean(0)
        S = Kc.T @ Kc / n
        eig = np.maximum(np.linalg.eigvalsh(S), 1e-10)

        G_PCA = np.exp(np.mean(np.log(eig)))
        G_uniform = np.mean(eig)
        diag = np.maximum(np.diag(S), 1e-10)
        G_norot = np.exp(np.mean(np.log(diag)))

        ld2 = sum(np.log(max(np.linalg.det(S[2*j:2*j+2, 2*j:2*j+2]), 1e-20)) for j in range(d//2))
        G_b2 = np.exp(ld2 / d)

        if d % 4 == 0:
            ld4 = sum(np.log(max(np.linalg.det(S[4*j:4*j+4, 4*j:4*j+4]), 1e-20)) for j in range(d//4))
            G_b4 = np.exp(ld4 / d)
        else:
            G_b4 = G_b2

        ok = (G_PCA <= G_b4*1.001 and G_b4 <= G_b2*1.001 and
              G_b2 <= G_norot*1.001 and G_norot <= G_uniform*1.001)

        results.append({
            'l': l, 'h': h, 'G_PCA': float(G_PCA), 'G_b4': float(G_b4),
            'G_b2': float(G_b2), 'G_norot': float(G_norot), 'G_uni': float(G_uniform),
            'R_aniso': float(G_uniform/G_PCA), 'eta2': float(G_PCA/G_b2),
            'eta4': float(G_PCA/G_b4), 'ok': ok
        })

    n_ok = sum(r['ok'] for r in results)
    R = [r['R_aniso'] for r in results]
    e2 = [r['eta2'] for r in results]
    e4 = [r['eta4'] for r in results]

    print(f"  Order holds: {n_ok}/{len(results)} heads")
    print(f"  R_aniso: mean={np.mean(R):.2f}, min={np.min(R):.2f}, max={np.max(R):.2f}")
    print(f"  eta(2):  mean={np.mean(e2):.4f}, min={np.min(e2):.4f}, max={np.max(e2):.4f}")
    print(f"  eta(4):  mean={np.mean(e4):.4f}, min={np.min(e4):.4f}, max={np.max(e4):.4f}")

    verdict = "VERIFIED" if n_ok == len(results) else f"VIOLATED in {len(results)-n_ok} heads"
    print(f"  => CONJECTURE A: {verdict}")
    if n_ok < len(results):
        for v in [r for r in results if not r['ok']][:3]:
            print(f"     L{v['l']}H{v['h']}: PCA={v['G_PCA']:.6f} b4={v['G_b4']:.6f} b2={v['G_b2']:.6f} norot={v['G_norot']:.6f} uni={v['G_uni']:.6f}")
    return results


def experiment_B(keys, info, ppl_data):
    """Verify ln(PPL/fp16) ~ c * G(M). Test 1-point calibration."""
    print("\n" + "="*70)
    print("EXPERIMENT B: PPL-MSE Relationship")
    print("="*70)
    d = info['d_head']
    fp16 = ppl_data['fp16']

    # Average covariance across all heads
    Sigmas = []
    for K in keys.values():
        Kc = K - K.mean(0)
        Sigmas.append(Kc.T @ Kc / K.shape[0])
    S_avg = np.mean(Sigmas, axis=0)
    eig = np.maximum(np.linalg.eigvalsh(S_avg), 1e-10)

    G_PCA = np.exp(np.mean(np.log(eig)))
    G_uni = np.mean(eig)
    G_norot = np.exp(np.mean(np.log(np.maximum(np.diag(S_avg), 1e-10))))

    method_G = {}
    for k in ppl_data:
        if k == 'fp16': continue
        m = k.rsplit('_', 1)[0]
        if m in ('uniform',): method_G[k] = G_uni
        elif m in ('kivi',): method_G[k] = G_norot
        elif m in ('fokvq', 'fokvq_qw', 'fokvq_e2'): method_G[k] = G_PCA
        elif m in ('quip',): method_G[k] = (G_PCA * G_uni) ** 0.5
        else: method_G[k] = G_norot

    for bits in [2, 3, 4]:
        print(f"\n  --- {bits}-bit ---")
        xs, ys, labels = [], [], []
        for k in ppl_data:
            if k == 'fp16' or not k.endswith(f'_{bits}'): continue
            p = ppl_data[k]
            if p == float('inf') or p <= fp16: continue
            xs.append(method_G.get(k, G_norot))
            ys.append(np.log(p / fp16))
            labels.append(k)

        if len(xs) < 3:
            print(f"  Insufficient data ({len(xs)} points)")
            continue

        x, y = np.array(xs), np.array(ys)
        slope, intercept, r, p_val, _ = stats.linregress(x, y)
        R2 = r**2

        for lb, xi, yi in zip(labels, xs, ys):
            print(f"    {lb:20s}: G={xi:.6f}, ln(PPL/fp16)={yi:.4f}, PPL={ppl_data[lb]:.2f}")
        print(f"  R² = {R2:.4f}")
        print(f"  => {'STRONG' if R2>0.8 else 'MODERATE' if R2>0.5 else 'WEAK'} (R²={R2:.3f})")

    # 1-point calibration (3-bit)
    print(f"\n  --- 1-Point Calibration (3-bit) ---")
    uni_key = 'uniform_3'
    if uni_key in ppl_data and ppl_data[uni_key] < float('inf'):
        ppl_uni = ppl_data[uni_key]
        print(f"  Calibration: {uni_key} PPL={ppl_uni:.2f}")
        for k in ppl_data:
            if k == 'fp16' or k == uni_key or not k.endswith('_3'): continue
            p_actual = ppl_data[k]
            if p_actual == float('inf'): continue
            G_m = method_G.get(k, G_norot)
            ratio = G_m / G_uni
            p_pred = fp16 * (ppl_uni / fp16) ** ratio
            err = abs(p_pred - p_actual) / p_actual * 100
            print(f"    {k:20s}: pred={p_pred:.2f}, actual={p_actual:.2f}, err={err:.1f}%")


def experiment_C(keys, info):
    """Test Gaussianity of KV cache via kurtosis, skewness, Shapiro-Wilk."""
    print("\n" + "="*70)
    print("EXPERIMENT C: Gaussianity Test")
    print("="*70)
    d = info['d_head']

    all_kurt, all_skew, all_shapiro_p = [], [], []
    test_keys = list(keys.keys())[:40]

    for (l, h) in test_keys:
        K = keys[(l, h)]
        for j in range(d):
            ch = K[:, j]
            all_kurt.append(stats.kurtosis(ch))
            all_skew.append(stats.skew(ch))
            try:
                _, p = stats.shapiro(ch[:5000])
                all_shapiro_p.append(p)
            except: pass

    kurt = np.array(all_kurt)
    skew = np.array(all_skew)
    shap = np.array(all_shapiro_p)

    ht_frac = np.mean(np.abs(kurt) > 3)
    normal_frac = np.mean(shap > 0.05) if len(shap) > 0 else 0

    print(f"  Channels tested: {len(kurt)}")
    print(f"  Kurtosis: mean={np.mean(kurt):.3f}, median={np.median(kurt):.3f}")
    print(f"    |kurt|>1: {np.sum(np.abs(kurt)>1)}/{len(kurt)} ({100*np.mean(np.abs(kurt)>1):.1f}%)")
    print(f"    |kurt|>3: {np.sum(np.abs(kurt)>3)}/{len(kurt)} ({100*ht_frac:.1f}%) [heavy-tailed]")
    print(f"  Skewness: mean={np.mean(skew):.3f}, |skew|>1: {100*np.mean(np.abs(skew)>1):.1f}%")
    print(f"  Shapiro-Wilk normal (p>0.05): {100*normal_frac:.1f}%")

    # Layer-wise
    print(f"\n  Layer-wise kurtosis:")
    for l in range(0, info['n_layers'], max(1, info['n_layers']//6)):
        lk = [kurt[i] for i, (li, _) in enumerate(test_keys) if li == l for _ in range(d) if i*d + _ < len(kurt)]
        # Simpler: compute per layer
        layer_kurts = []
        for idx, (li, hi) in enumerate(test_keys):
            if li == l:
                start = idx * d
                end = start + d
                if end <= len(kurt):
                    layer_kurts.extend(kurt[start:end].tolist())
        if layer_kurts:
            print(f"    L{l:2d}: mean={np.mean(layer_kurts):.3f}, max|kurt|={np.max(np.abs(layer_kurts)):.3f}")

    verdict = "HOLDS" if ht_frac < 0.05 else "MODERATE" if ht_frac < 0.20 else "VIOLATED"
    print(f"\n  => GAUSSIANITY: {verdict} ({ht_frac*100:.1f}% heavy-tailed)")
    return {'kurtosis_mean': float(np.mean(kurt)), 'heavy_tail_frac': float(ht_frac),
            'shapiro_normal_frac': float(normal_frac)}


def experiment_D(keys, queries, info):
    """Compare PCA vs weighted-PCA rotation distortion."""
    print("\n" + "="*70)
    print("EXPERIMENT D: Weighted Hadamard (PCA vs Weighted-PCA)")
    print("="*70)
    d, G = info['d_head'], info['G']
    diffs = []
    test_keys = list(keys.keys())[:20]

    for (l, h) in test_keys:
        K = keys[(l, h)]
        Kc = K - K.mean(0)
        SK = Kc.T @ Kc / K.shape[0]

        q_heads = [(l, g) for g in range(h*G, (h+1)*G) if (l, g) in queries]
        if not q_heads: continue

        SQ_list = []
        for ql, qg in q_heads:
            Q = queries[(ql, qg)]
            Qc = Q - Q.mean(0)
            SQ_list.append(Qc.T @ Qc / Q.shape[0])
        SQ_eff = np.mean(SQ_list, axis=0)

        w = np.maximum(np.diag(SQ_eff), 1e-10) / np.trace(SQ_eff)

        # PCA
        _, V = np.linalg.eigh(SK)
        diag_pca = np.diag(V.T @ SK @ V)
        obj_pca = np.sum(w * np.log(np.maximum(diag_pca, 1e-10)))

        # Weighted PCA
        Wsq = np.diag(np.sqrt(w))
        _, Vw = np.linalg.eigh(Wsq @ SK @ Wsq)
        diag_w = np.diag(Vw.T @ SK @ Vw)
        obj_w = np.sum(w * np.log(np.maximum(diag_w, 1e-10)))

        diff = (obj_pca - obj_w) / abs(obj_w) * 100 if obj_w != 0 else 0
        diffs.append(diff)

    diffs = np.array(diffs)
    print(f"  Heads tested: {len(diffs)}")
    print(f"  PCA vs Weighted-PCA diff: mean={np.mean(np.abs(diffs)):.3f}%, max={np.max(np.abs(diffs)):.3f}%")
    verdict = "PCA sufficient" if np.mean(np.abs(diffs)) < 5 else "PCA insufficient"
    print(f"  => {verdict}")
    return {'mean_diff': float(np.mean(diffs)), 'max_diff': float(np.max(np.abs(diffs)))}


def experiment_E(keys, info):
    """Test diagnostic statistic convergence with sample size."""
    print("\n" + "="*70)
    print("EXPERIMENT E: Sample Size Convergence")
    print("="*70)
    d = info['d_head']

    mid = info['n_layers'] // 2
    tk = next(((l,h) for l,h in keys if l == mid and h == 0), list(keys.keys())[0])
    K = keys[tk]
    n_full = K.shape[0]

    Kc = K - K.mean(0)
    S_ref = Kc.T @ Kc / n_full
    eig_ref = np.maximum(np.linalg.eigvalsh(S_ref), 1e-10)
    R_ref = np.mean(eig_ref) / np.exp(np.mean(np.log(eig_ref)))

    ld2 = sum(np.log(max(np.linalg.det(S_ref[2*j:2*j+2, 2*j:2*j+2]), 1e-20)) for j in range(d//2))
    G_b2_ref = np.exp(ld2 / d)
    G_pca_ref = np.exp(np.mean(np.log(eig_ref)))
    eta2_ref = G_pca_ref / G_b2_ref

    print(f"  Reference (n={n_full}): R_aniso={R_ref:.4f}, eta(2)={eta2_ref:.4f}")
    print(f"\n  {'n':>6s} | {'R_aniso':>10s} {'±std':>8s} {'err%':>7s} | {'eta2':>8s} {'±std':>8s} {'err%':>7s} | rank_ok")
    print("  " + "-"*75)

    for ns in [50, 100, 200, 500, 1000]:
        if ns >= n_full: continue
        Rs, Es = [], []
        for _ in range(10):
            idx = np.random.choice(n_full, ns, replace=False)
            Ks = K[idx]
            Ksc = Ks - Ks.mean(0)
            Ss = Ksc.T @ Ksc / ns
            alpha = min(1.0, d / (ns * 2))
            Ss = (1-alpha)*Ss + alpha*(np.trace(Ss)/d)*np.eye(d)  # shrinkage
            es = np.maximum(np.linalg.eigvalsh(Ss), 1e-10)
            Rs.append(np.mean(es) / np.exp(np.mean(np.log(es))))
            ld = sum(np.log(max(np.linalg.det(Ss[2*j:2*j+2, 2*j:2*j+2]), 1e-20)) for j in range(d//2))
            gb2 = np.exp(ld / d)
            gpca = np.exp(np.mean(np.log(es)))
            Es.append(gpca / gb2)

        Ra, Ea = np.array(Rs), np.array(Es)
        r_err = abs(np.mean(Ra)-R_ref)/R_ref*100
        e_err = abs(np.mean(Ea)-eta2_ref)/max(abs(eta2_ref),1e-10)*100
        rok = sum((r>2)==(R_ref>2) and (e>0.95)==(eta2_ref>0.95) for r,e in zip(Rs,Es))
        print(f"  {ns:6d} | {np.mean(Ra):10.4f} {np.std(Ra):8.4f} {r_err:6.1f}% | {np.mean(Ea):8.4f} {np.std(Ea):8.4f} {e_err:6.1f}% | {rok}/10")


def run_model(model_name, device, ppl_data, output_dir):
    """Run all 5 experiments for one model."""
    tag = model_name.split('/')[-1]
    print(f"\n{'#'*70}")
    print(f"# {tag} on {device}")
    print(f"{'#'*70}")

    t0 = time.time()
    keys, queries, info = extract_kv_cache(model_name, device, max_tokens=2000)
    print(f"  KV extraction: {time.time()-t0:.1f}s")

    rA = experiment_A(keys, info)
    experiment_B(keys, info, ppl_data)
    rC = experiment_C(keys, info)
    rD = experiment_D(keys, queries, info)
    experiment_E(keys, info)

    print(f"\n  Total: {time.time()-t0:.1f}s")

    # Save
    out = output_dir / f"{tag}_verification.json"
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, dict): return {k: conv(v) for k,v in o.items()}
        if isinstance(o, list): return [conv(v) for v in o]
        return o
    with open(out, 'w') as f:
        json.dump(conv({'A': rA, 'C': rC, 'D': rD, 'info': info}), f, indent=2)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1', help='GPU device')
    args = parser.parse_args()

    outdir = Path(__file__).parent / "verification_results"
    outdir.mkdir(exist_ok=True)

    t0 = time.time()
    run_model('Qwen/Qwen2.5-7B', args.device, QWEN_PPL, outdir)

    print(f"\n{'='*70}")
    print(f"ALL DONE in {time.time()-t0:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
