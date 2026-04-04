#!/usr/bin/env python3
"""
NIAH: Pre-RoPE PCA + Gaussian Lloyd-Max + QJL  vs  TurboQuant + Lloyd-Max + QJL
GPU 0: 2-bit experiments
GPU 1: 3-bit experiments
"""
import json, time, traceback
import numpy as np
from pathlib import Path
from multiprocessing import Process

OUTDIR = Path(__file__).parent / "verification_results"

# ============================================================
# Lloyd-Max codebooks for N(0,1)
# ============================================================
LM_CB = {
    2: (np.array([-0.9816, 0.0, 0.9816]),
        np.array([-1.5104, -0.4528, 0.4528, 1.5104])),
    3: (np.array([-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480]),
        np.array([-2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519])),
    4: (np.array([-2.4008, -1.8435, -1.4370, -1.0993, -0.7996, -0.5224, -0.2582, 0.0,
                   0.2582, 0.5224, 0.7996, 1.0993, 1.4370, 1.8435, 2.4008]),
        np.array([-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
                   0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326])),
}

def lloyd_max_quant(x, bits, mean=None, scale=None):
    """Lloyd-Max for N(mean, scale^2). Center → normalize → codebook → denormalize."""
    if mean is None: mean = np.mean(x)
    if scale is None: scale = max(np.std(x - mean), 1e-10)
    x_norm = (x - mean) / scale
    x_clip = np.clip(x_norm, -3.5, 3.5)
    th, ct = LM_CB[bits]
    idx = np.searchsorted(th, x_clip)
    return ct[idx] * scale + mean

def uniform_quant(x, bits):
    nl = 2 ** bits
    vmin, vmax = x.min(), x.max()
    if vmax - vmin < 1e-10: return x.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((x - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin

def qjl_residual(K_orig, K_quant, d_proj=32, bits_res=8):
    """QJL: project residual with random JL matrix, quantize, add back."""
    n, d = K_orig.shape
    residual = K_orig - K_quant
    # Random JL projection matrix (fixed seed per layer for reproducibility)
    np.random.seed(12345)
    P = np.random.randn(d, d_proj) / np.sqrt(d_proj)  # (d, d_proj)
    # Project residual
    res_proj = residual @ P  # (n, d_proj)
    # Quantize projected residual
    res_proj_q = np.zeros_like(res_proj)
    for j in range(d_proj):
        res_proj_q[:, j] = uniform_quant(res_proj[:, j], bits_res)
    # Reconstruct: pseudo-inverse of P
    K_corrected = K_quant + res_proj_q @ P.T  # (n, d)
    return K_corrected


def run_niah_experiment(device, bits_list, tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'#'*70}")
    print(f"# NIAH Final: {tag} on {device}, bits={bits_list}")
    print(f"{'#'*70}")

    model_name = 'Qwen/Qwen2.5-7B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads

    # ============================================================
    # Calibration: collect pre-RoPE PCA bases
    # ============================================================
    print("  [Calibration] Collecting pre-RoPE covariance...")
    filler_text = "The grass is green and the sky is blue. Science is wonderful. Numbers are abstract. "
    cal_ids = tokenizer.encode(filler_text * 80, return_tensors="pt", truncation=False)[:, :2048].to(device)

    pca_bases = {}     # (l,h) -> eigenvectors
    pca_eigenvals = {} # (l,h) -> eigenvalues
    pca_means = {}     # (l,h) -> mean vector
    capture = {}
    hooks = []

    def make_cal_hook(li):
        def fn(mod, args, kwargs):
            hs = args[0] if args else kwargs.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    capture[(li, h)] = k[:, h, :]
        return fn

    for l in range(n_layers):
        hooks.append(model.model.layers[l].self_attn.register_forward_pre_hook(
            make_cal_hook(l), with_kwargs=True))
    with torch.no_grad():
        model(cal_ids, use_cache=False)
    for hook in hooks:
        hook.remove()

    for (l, h), K in capture.items():
        mean = K.mean(0)
        Kc = K - mean
        S = Kc.T @ Kc / K.shape[0]
        evals, evecs = np.linalg.eigh(S)
        pca_bases[(l, h)] = evecs
        pca_eigenvals[(l, h)] = np.maximum(evals, 1e-10)
        pca_means[(l, h)] = mean

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    # Global variance for TurboQuant scale
    all_vars = [np.mean(pca_eigenvals[(l, h)]) for l, h in pca_eigenvals]
    turbo_scale = np.sqrt(np.mean(all_vars))

    print(f"  Calibration done: {len(pca_bases)} heads, turbo_scale={turbo_scale:.4f}")

    # ============================================================
    # Quantization hooks
    # ============================================================
    def make_quant_hook(li, bits, method):
        """Hook on k_proj output (pre-RoPE K). Quantize and return."""
        def fn(mod, inp, out):
            k = out[0] if isinstance(out, tuple) else out
            k_np = k.detach().cpu().float().numpy().reshape(-1, n_kv, d_head)

            for h in range(n_kv):
                K_h = k_np[:, h, :]  # (seq, d_head) — this is PRE-RoPE

                if method == 'pre_pca_lloyd_qjl':
                    # Pre-RoPE PCA + Gaussian Lloyd-Max + QJL
                    V = pca_bases[(li, h)]
                    evals = pca_eigenvals[(li, h)]
                    mu = pca_means[(li, h)]
                    K_pca = (K_h - mu) @ V  # PCA transform
                    K_pca_q = np.zeros_like(K_pca)
                    for j in range(d_head):
                        K_pca_q[:, j] = lloyd_max_quant(
                            K_pca[:, j], bits, mean=0.0, scale=np.sqrt(evals[j]))
                    K_h_q = K_pca_q @ V.T + mu  # inverse PCA
                    # QJL residual correction
                    K_h_final = qjl_residual(K_h, K_h_q)
                    k_np[:, h, :] = K_h_final

                elif method == 'turbo_lloyd_qjl':
                    # TurboQuant: random rotation + Lloyd-Max + QJL
                    K_rot = K_h @ R_random
                    K_rot_q = np.zeros_like(K_rot)
                    for j in range(d_head):
                        # After random rotation, each dim ≈ same variance
                        K_rot_q[:, j] = lloyd_max_quant(K_rot[:, j], bits)
                    K_h_q = K_rot_q @ R_random.T
                    K_h_final = qjl_residual(K_h, K_h_q)
                    k_np[:, h, :] = K_h_final

                elif method == 'turbo_uni':
                    # TurboQuant with uniform quantizer (no Lloyd-Max, no QJL)
                    K_rot = K_h @ R_random
                    K_rot_q = np.zeros_like(K_rot)
                    for j in range(d_head):
                        K_rot_q[:, j] = uniform_quant(K_rot[:, j], bits)
                    k_np[:, h, :] = K_rot_q @ R_random.T

                elif method == 'pre_pca_uni':
                    # Pre-RoPE PCA + uniform (no Lloyd-Max, no QJL)
                    V = pca_bases[(li, h)]
                    mu = pca_means[(li, h)]
                    K_pca = (K_h - mu) @ V
                    K_pca_q = np.zeros_like(K_pca)
                    for j in range(d_head):
                        K_pca_q[:, j] = uniform_quant(K_pca[:, j], bits)
                    k_np[:, h, :] = K_pca_q @ V.T + mu

            return torch.tensor(k_np.reshape(-1, n_kv * d_head),
                              dtype=k.dtype, device=k.device).unsqueeze(0)
        return fn

    # ============================================================
    # NIAH test
    # ============================================================
    ctx_lengths = [4096, 8192]
    depths = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    n_trials = 10
    filler = "The grass is green. The sky is blue. Science advances every day. Numbers tell stories. "

    methods = ['fp16', 'turbo_uni', 'pre_pca_uni', 'turbo_lloyd_qjl', 'pre_pca_lloyd_qjl']

    results = {}

    for ctx_len in ctx_lengths:
        for bits in bits_list:
            for method in methods:
                key = f"{method}_{bits}b_{ctx_len}" if method != 'fp16' else f"fp16_{ctx_len}"
                if key in results:
                    continue

                print(f"\n  [{method} {bits}b ctx={ctx_len}]")

                hks = []
                if method != 'fp16':
                    for l in range(n_layers):
                        kp = model.model.layers[l].self_attn.k_proj
                        hks.append(kp.register_forward_hook(make_quant_hook(l, bits, method)))

                depth_results = {}
                for depth in depths:
                    correct = 0
                    for trial in range(n_trials):
                        passkey = str(np.random.randint(10000, 99999))  # 5-digit for harder task
                        needle = f"CRITICAL INFORMATION: The access code is {passkey}. This is very important."
                        question = f"\nBased on the document above, what is the access code? The code is"

                        needle_ids = tokenizer.encode(needle, add_special_tokens=False)
                        q_ids = tokenizer.encode(question, add_special_tokens=False)
                        filler_ids = tokenizer.encode(filler, add_special_tokens=False)

                        avail = ctx_len - len(needle_ids) - len(q_ids) - 5
                        if avail < 50: continue
                        ff = (filler_ids * (avail // len(filler_ids) + 2))[:avail]
                        pos = max(0, min(int(len(ff) * depth), len(ff)))
                        toks = ff[:pos] + needle_ids + ff[pos:] + q_ids
                        toks = toks[:ctx_len]

                        input_ids = torch.tensor([toks], device=device)
                        with torch.no_grad():
                            out = model(input_ids, use_cache=False)
                            logits = out.logits[0, -1, :]
                            # Generate 5 tokens greedily to check passkey
                            generated = []
                            cur_ids = input_ids
                            for _ in range(5):
                                out2 = model(cur_ids, use_cache=False)
                                next_id = out2.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                                generated.append(next_id.item())
                                cur_ids = torch.cat([cur_ids, next_id], dim=1)
                                if cur_ids.shape[1] > ctx_len + 10:
                                    break

                            gen_text = tokenizer.decode(generated).strip()
                            if passkey in gen_text:
                                correct += 1

                    acc = correct / n_trials if n_trials > 0 else 0
                    depth_results[depth] = acc
                    print(f"    d={depth:.2f}: {correct}/{n_trials}={acc:.0%}")

                for hk in hks:
                    hk.remove()

                avg_acc = np.mean(list(depth_results.values()))
                results[key] = {'depths': {str(d): a for d, a in depth_results.items()},
                               'avg': float(avg_acc)}
                print(f"    avg={avg_acc:.0%}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"NIAH Summary ({tag})")
    print(f"{'='*70}")

    for ctx_len in ctx_lengths:
        print(f"\n  === Context {ctx_len} ===")
        for bits in bits_list:
            print(f"\n  --- {bits}-bit ---")
            print(f"  {'Method':30s} " + ' '.join(f'd={d:.2f}' for d in depths) + '  Avg')

            for method in methods:
                key = f"{method}_{bits}b_{ctx_len}" if method != 'fp16' else f"fp16_{ctx_len}"
                if key not in results: continue
                r = results[key]
                accs = [r['depths'].get(str(d), 0) for d in depths]
                print(f"  {method:30s} " + ' '.join(f'{a:5.0%}' for a in accs) + f'  {r["avg"]:.0%}')

        # Key comparison at each bit level
        for bits in bits_list:
            turbo_key = f"turbo_lloyd_qjl_{bits}b_{ctx_len}"
            pre_key = f"pre_pca_lloyd_qjl_{bits}b_{ctx_len}"
            if turbo_key in results and pre_key in results:
                t_avg = results[turbo_key]['avg']
                p_avg = results[pre_key]['avg']
                winner = "PrePCA" if p_avg >= t_avg else "TurboQ"
                print(f"\n  KEY {bits}b ctx={ctx_len}: PrePCA+Lloyd+QJL={p_avg:.0%} vs TurboQ+Lloyd+QJL={t_avg:.0%} -> {winner}")

    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / f"niah_final_{tag}.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTDIR / f'niah_final_{tag}.json'}")

    del model; torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    t0 = time.time()
    OUTDIR.mkdir(exist_ok=True)

    # GPU 0: 2-bit (hardest, most differentiation)
    # GPU 1: 3-bit
    p0 = Process(target=run_niah_experiment, args=('cuda:0', [2], '2bit'))
    p1 = Process(target=run_niah_experiment, args=('cuda:1', [3], '3bit'))

    p0.start(); p1.start()
    p0.join(); p1.join()

    print(f"\nALL DONE in {time.time()-t0:.1f}s")
