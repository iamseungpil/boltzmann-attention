#!/usr/bin/env python3
"""
Lloyd-Max + NIAH verification.
GPU 0: Lloyd-Max MSE/PPL comparison
GPU 1: NIAH passkey retrieval
"""
import json, time, os, sys, traceback
import numpy as np
from pathlib import Path
from multiprocessing import Process

def gaussian_lloyd_max_codebook(bits):
    """Return Lloyd-Max codebook for N(0,1). Pre-computed optimal values."""
    # Optimal centroids and thresholds for N(0,1)
    # Source: Max (1960), Lloyd (1982)
    codebooks = {
        1: {'centroids': np.array([-0.7979, 0.7979]),
            'thresholds': np.array([0.0])},
        2: {'centroids': np.array([-1.5104, -0.4528, 0.4528, 1.5104]),
            'thresholds': np.array([-0.9816, 0.0, 0.9816])},
        3: {'centroids': np.array([-2.1519, -1.3440, -0.7560, -0.2451,
                                    0.2451, 0.7560, 1.3440, 2.1519]),
            'thresholds': np.array([-1.7480, -1.0500, -0.5006, 0.0,
                                     0.5006, 1.0500, 1.7480])},
        4: {'centroids': np.array([-2.7326, -2.0690, -1.6180, -1.2562,
                                   -0.9423, -0.6568, -0.3881, -0.1284,
                                    0.1284, 0.3881, 0.6568, 0.9423,
                                    1.2562, 1.6180, 2.0690, 2.7326]),
            'thresholds': np.array([-2.4008, -1.8435, -1.4370, -1.0993,
                                    -0.7996, -0.5224, -0.2582, 0.0,
                                     0.2582, 0.5224, 0.7996, 1.0993,
                                     1.4370, 1.8435, 2.4008])},
    }
    return codebooks.get(bits, codebooks[4])

def lloyd_max_quantize(x, bits, scale=1.0):
    """Quantize x ~ N(0, scale^2) using Gaussian Lloyd-Max."""
    cb = gaussian_lloyd_max_codebook(bits)
    x_norm = x / scale  # normalize to N(0,1)
    # Find nearest centroid
    thresholds = cb['thresholds']
    centroids = cb['centroids']
    indices = np.searchsorted(thresholds, x_norm)
    x_q_norm = centroids[indices]
    return x_q_norm * scale  # scale back

def uniform_quantize(x, bits, vmin=None, vmax=None):
    """Standard uniform scalar quantization."""
    n_levels = 2 ** bits
    if vmin is None: vmin = x.min()
    if vmax is None: vmax = x.max()
    if vmax - vmin < 1e-10: return x.copy()
    scale = (vmax - vmin) / (n_levels - 1)
    q = np.round((x - vmin) / scale).astype(int)
    q = np.clip(q, 0, n_levels - 1)
    return q * scale + vmin

def run_lloydmax_experiment(device='cuda:0'):
    """Compare Uniform vs Lloyd-Max quantizers with different rotations."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'#'*70}")
    print(f"# Lloyd-Max Experiment on {device}")
    print(f"{'#'*70}")

    model_name = 'Qwen/Qwen2.5-7B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_ids = tokenizer.encode(text, return_tensors="pt", truncation=False)
    cal_ids = all_ids[:, :2048].to(device)

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads

    # Step 1: Collect pre-RoPE keys for calibration
    print("  [Step 1] Collecting pre-RoPE keys...")
    pre_keys_all = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            hs = args[0] if len(args) > 0 else kwargs.get('hidden_states')
            if hs is not None:
                k = module.k_proj(hs)[0].detach().cpu().float().numpy()
                kr = k.reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    pre_keys_all[(layer_idx, h)] = kr[:, h, :]
        return hook_fn

    for l in range(n_layers):
        attn = model.model.layers[l].self_attn
        hooks.append(attn.register_forward_pre_hook(make_hook(l), with_kwargs=True))

    post_keys_all = {}
    with torch.no_grad():
        out = model(cal_ids, use_cache=True)
    for l, item in enumerate(out.past_key_values):
        K = (item[0] if isinstance(item, tuple) else item.keys)[0].cpu().float().numpy()
        for h in range(n_kv):
            post_keys_all[(l, h)] = K[h]
    for hook in hooks:
        hook.remove()

    # Step 2: PCA
    pca_pre = {}
    eigenvalues_pre = {}
    for (l, h), K in pre_keys_all.items():
        Kc = K - K.mean(0)
        S = Kc.T @ Kc / K.shape[0]
        evals, evecs = np.linalg.eigh(S)
        pca_pre[(l, h)] = evecs
        eigenvalues_pre[(l, h)] = np.maximum(evals, 1e-10)

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    print(f"  Collected {len(pre_keys_all)} heads")

    # Step 3: MSE comparison - 6 methods
    print(f"\n{'='*70}")
    print("MSE: 6 methods (rotation x quantizer)")
    print(f"{'='*70}")

    results = []
    bits_list = [2, 3, 4]

    for (l, h) in sorted(pre_keys_all.keys()):
        K_pre = pre_keys_all[(l, h)]
        K_post = post_keys_all[(l, h)]
        V = pca_pre[(l, h)]
        evals = eigenvalues_pre[(l, h)]
        n, d = K_pre.shape

        for bits in bits_list:
            def mse_method(K_orig, rotation=None, use_lloyd=False, eigenvals=None):
                if rotation is not None:
                    K_rot = K_orig @ rotation
                else:
                    K_rot = K_orig.copy()

                K_recon = np.zeros_like(K_rot)
                for j in range(d):
                    col = K_rot[:, j]
                    if use_lloyd and eigenvals is not None:
                        scale = np.sqrt(eigenvals[j]) if eigenvals[j] > 1e-10 else 1.0
                        K_recon[:, j] = lloyd_max_quantize(col, bits, scale=scale)
                    elif use_lloyd:
                        scale = np.std(col) if np.std(col) > 1e-10 else 1.0
                        K_recon[:, j] = lloyd_max_quantize(col, bits, scale=scale)
                    else:
                        K_recon[:, j] = uniform_quantize(col, bits)

                if rotation is not None:
                    K_recon = K_recon @ rotation.T
                return np.mean((K_orig - K_recon) ** 2)

            # 1. TurboQuant: random rot + uniform
            mse_turbo_uni = mse_method(K_post, R_random, use_lloyd=False)
            # 2. TurboQuant: random rot + Lloyd-Max (actual TurboQuant)
            mse_turbo_lloyd = mse_method(K_post, R_random, use_lloyd=True)
            # 3. Pre-RoPE PCA + uniform (our previous experiment)
            mse_pre_uni = mse_method(K_pre, V, use_lloyd=False)
            # 4. Pre-RoPE PCA + Gaussian Lloyd-Max (OPTIMAL)
            mse_pre_lloyd = mse_method(K_pre, V, use_lloyd=True, eigenvals=evals)
            # 5. No rotation + uniform (baseline)
            mse_identity = mse_method(K_post, None, use_lloyd=False)
            # 6. No rotation + Lloyd-Max
            mse_identity_lloyd = mse_method(K_post, None, use_lloyd=True)

            results.append({
                'l': l, 'h': h, 'bits': bits,
                'turbo_uni': float(mse_turbo_uni),
                'turbo_lloyd': float(mse_turbo_lloyd),
                'pre_pca_uni': float(mse_pre_uni),
                'pre_pca_lloyd': float(mse_pre_lloyd),
                'identity_uni': float(mse_identity),
                'identity_lloyd': float(mse_identity_lloyd),
            })

    # Summary
    for bits in bits_list:
        br = [r for r in results if r['bits'] == bits]
        methods = ['identity_uni', 'identity_lloyd', 'turbo_uni', 'turbo_lloyd',
                   'pre_pca_uni', 'pre_pca_lloyd']
        labels = ['NoRot+Uniform', 'NoRot+LloydMax', 'TurboQ+Uniform', 'TurboQ+LloydMax',
                  'PrePCA+Uniform', 'PrePCA+LloydMax']

        vals = {m: np.mean([r[m] for r in br]) for m in methods}
        best = min(vals.values())

        print(f"\n--- {bits}-bit ---")
        for m, label in zip(methods, labels):
            v = vals[m]
            ratio = v / best if best > 0 else 0
            marker = ' *** BEST ***' if abs(v - best) < 1e-10 else ''
            print(f"  {label:22s}: MSE={v:.6f} ({ratio:.2f}x){marker}")

        # Key comparison
        turbo_best = vals['turbo_lloyd']
        pre_best = vals['pre_pca_lloyd']
        print(f"\n  TurboQ+LloydMax vs PrePCA+LloydMax: {turbo_best:.6f} vs {pre_best:.6f}")
        if pre_best < turbo_best:
            print(f"  => PrePCA+LloydMax WINS by {turbo_best/pre_best:.2f}x")
        else:
            print(f"  => TurboQ+LloydMax WINS by {pre_best/turbo_best:.2f}x")

        # Lloyd-Max gain over uniform (same rotation)
        turbo_gain = vals['turbo_uni'] / vals['turbo_lloyd']
        pre_gain = vals['pre_pca_uni'] / vals['pre_pca_lloyd']
        print(f"  Lloyd-Max gain (TurboQ): {turbo_gain:.2f}x")
        print(f"  Lloyd-Max gain (PrePCA): {pre_gain:.2f}x")

    # Save
    outdir = Path(__file__).parent / "verification_results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / "lloydmax_mse_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outdir / 'lloydmax_mse_results.json'}")

    del model; torch.cuda.empty_cache()
    return results


def run_niah_simple(device='cuda:1'):
    """Simple NIAH: passkey retrieval with pre-RoPE PCA vs TurboQuant vs identity."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'#'*70}")
    print(f"# NIAH Experiment on {device}")
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

    # Calibration
    print("  [Calibration]...")
    filler = "The grass is green and the sky is blue. Today is a wonderful day for science. "
    cal_ids = tokenizer.encode(filler * 60, return_tensors="pt", truncation=False)[:, :2048].to(device)

    pca_pre = {}
    hooks = []
    capture = {}

    def make_hook(li):
        def fn(mod, args, kwargs):
            hs = args[0] if args else kwargs.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    K = k[:, h, :]; Kc = K - K.mean(0)
                    capture[(li, h)] = Kc.T @ Kc / K.shape[0]
        return fn

    for l in range(n_layers):
        hooks.append(model.model.layers[l].self_attn.register_forward_pre_hook(make_hook(l), with_kwargs=True))
    with torch.no_grad():
        model(cal_ids, use_cache=False)
    for hook in hooks:
        hook.remove()

    for (l, h), cov in capture.items():
        _, V = np.linalg.eigh(cov)
        pca_pre[(l, h)] = V

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]
    print(f"  Calibration done: {len(pca_pre)} heads")

    # NIAH test
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_trials = 10
    ctx_len = 4096
    bits_list = [2, 4]

    def make_quant_hook(li, bits, rot_dict=None, rand_rot=None):
        n_levels = 2 ** bits
        def fn(mod, inp, out):
            k = (out[0] if isinstance(out, tuple) else out)
            k_np = k.detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
            for h in range(n_kv):
                Kh = k_np[:, h, :]
                if rot_dict and (li, h) in rot_dict:
                    R = rot_dict[(li, h)]; Kr = Kh @ R
                elif rand_rot is not None:
                    R = rand_rot; Kr = Kh @ R
                else:
                    R = None; Kr = Kh.copy()
                for j in range(d_head):
                    Kr[:, j] = uniform_quantize(Kr[:, j], bits)
                k_np[:, h, :] = Kr @ R.T if R is not None else Kr
            return torch.tensor(k_np.reshape(-1, n_kv*d_head), dtype=k.dtype, device=k.device).unsqueeze(0)
        return fn

    def test_niah(method_name, bits, rot_dict=None, rand_rot=None):
        hks = []
        if method_name != 'fp16':
            for l in range(n_layers):
                kp = model.model.layers[l].self_attn.k_proj
                hks.append(kp.register_forward_hook(make_quant_hook(l, bits, rot_dict, rand_rot)))

        depth_acc = {}
        for depth in depths:
            correct = 0
            for trial in range(n_trials):
                passkey = str(np.random.randint(1000, 9999))
                needle = f"IMPORTANT: The secret passkey is {passkey}. Remember it."
                question = f"\nWhat is the secret passkey mentioned above? Answer: The passkey is"

                needle_ids = tokenizer.encode(needle, add_special_tokens=False)
                question_ids = tokenizer.encode(question, add_special_tokens=False)
                filler_ids = tokenizer.encode(filler, add_special_tokens=False)

                avail = ctx_len - len(needle_ids) - len(question_ids) - 5
                full_filler = (filler_ids * (avail // len(filler_ids) + 1))[:avail]

                pos = int(len(full_filler) * depth)
                all_ids = full_filler[:pos] + needle_ids + full_filler[pos:] + question_ids
                input_ids = torch.tensor([all_ids[:ctx_len]], device=device)

                with torch.no_grad():
                    out = model(input_ids, use_cache=False)
                    logits = out.logits[0, -1, :]
                    top5 = torch.topk(logits, 10).indices
                    top5_text = [tokenizer.decode(t.item()).strip() for t in top5]
                    # Check if passkey or its first digits appear
                    if any(passkey[:2] in t for t in top5_text[:5]):
                        correct += 1

            acc = correct / n_trials
            depth_acc[depth] = acc
            print(f"    [{method_name} {bits}bit] depth={depth:.2f}: {correct}/{n_trials} = {acc:.0%}")

        for hk in hks:
            hk.remove()
        return depth_acc

    results = {}

    # FP16 baseline
    print("\n  [FP16 baseline]")
    results['fp16'] = test_niah('fp16', 0)

    for bits in bits_list:
        print(f"\n  [{bits}-bit experiments]")
        results[f'identity_{bits}'] = test_niah('identity', bits)
        results[f'random_{bits}'] = test_niah('random', bits, rand_rot=R_random)
        results[f'pre_pca_{bits}'] = test_niah('pre_pca', bits, rot_dict=pca_pre)

    # Summary
    print(f"\n{'='*70}")
    print("NIAH Summary")
    print(f"{'='*70}")
    print(f"  {'Method':25s}", ' '.join(f'd={d:.2f}' for d in depths), ' Avg')
    for key in ['fp16'] + [f'{m}_{b}' for b in bits_list for m in ['identity', 'random', 'pre_pca']]:
        if key not in results: continue
        accs = [results[key].get(d, 0) for d in depths]
        avg = np.mean(accs)
        print(f"  {key:25s}", ' '.join(f'{a:5.0%}' for a in accs), f' {avg:.0%}')

    # Ranking check
    print(f"\n  MSE ranking predicts NIAH ranking?")
    for bits in bits_list:
        accs = {m: np.mean(list(results[f'{m}_{bits}'].values()))
                for m in ['identity', 'random', 'pre_pca'] if f'{m}_{bits}' in results}
        mse_order = ['pre_pca', 'random', 'identity']
        niah_order = sorted(accs, key=lambda m: accs[m], reverse=True)
        match = mse_order == niah_order
        print(f"    {bits}-bit: MSE pred={mse_order}, NIAH={niah_order} -> {'MATCH' if match else 'MISMATCH'}")
        for m in niah_order:
            print(f"      {m}: {accs[m]:.0%}")

    outdir = Path(__file__).parent / "verification_results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / "niah_results.json", 'w') as f:
        json.dump({k: {str(dk): dv for dk, dv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nSaved: {outdir / 'niah_results.json'}")

    del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    t0 = time.time()
    p0 = Process(target=run_lloydmax_experiment, args=('cuda:0',))
    p1 = Process(target=run_niah_simple, args=('cuda:1',))
    p0.start(); p1.start()
    p0.join(); p1.join()
    print(f"\nALL DONE in {time.time()-t0:.1f}s")
