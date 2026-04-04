#!/usr/bin/env python3
"""
Combined verification: Lloyd-Max MSE + NIAH (both GPUs)
GPU 0: Lloyd-Max MSE experiment
GPU 1: NIAH passkey retrieval
Fixes: Lloyd-Max scale from actual std, NIAH hook uses forward_hook correctly
"""
import json, time, traceback
import numpy as np
from pathlib import Path
from multiprocessing import Process

OUTDIR = Path(__file__).parent / "verification_results"

def gaussian_lloyd_max(x, bits, scale=None):
    """Lloyd-Max quantize x assumed ~ N(0, scale^2). If scale=None, estimate from data."""
    CB = {
        2: (np.array([-0.9816, 0.0, 0.9816]), np.array([-1.5104, -0.4528, 0.4528, 1.5104])),
        3: (np.array([-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480]),
            np.array([-2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519])),
        4: (np.array([-2.4008, -1.8435, -1.4370, -1.0993, -0.7996, -0.5224, -0.2582, 0.0,
                       0.2582, 0.5224, 0.7996, 1.0993, 1.4370, 1.8435, 2.4008]),
            np.array([-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3881, -0.1284,
                       0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326])),
    }
    if scale is None:
        scale = max(np.std(x), 1e-10)
    th, ct = CB[bits]
    idx = np.searchsorted(th * scale, x)
    return ct[idx] * scale

def uniform_quant(x, bits):
    nl = 2 ** bits
    vmin, vmax = x.min(), x.max()
    if vmax - vmin < 1e-10: return x.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((x - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin

def load_model_and_keys(device, max_tokens=2048):
    """Load Qwen, collect pre/post RoPE keys."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_name = 'Qwen/Qwen2.5-7B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    all_ids = tokenizer.encode(text, return_tensors="pt", truncation=False)

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads

    cal_ids = all_ids[:, :max_tokens].to(device)

    # Hooks for pre-RoPE K
    pre_capture = {}
    hooks = []
    def make_hook(li):
        def fn(mod, args, kwargs):
            hs = args[0] if args else kwargs.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    pre_capture[(li, h)] = k[:, h, :]
        return fn

    for l in range(n_layers):
        hooks.append(model.model.layers[l].self_attn.register_forward_pre_hook(make_hook(l), with_kwargs=True))

    post_keys = {}
    with torch.no_grad():
        out = model(cal_ids, use_cache=True)
    for l, item in enumerate(out.past_key_values):
        K = (item[0] if isinstance(item, tuple) else item.keys)[0].cpu().float().numpy()
        for h in range(n_kv):
            post_keys[(l, h)] = K[h]
    for hook in hooks:
        hook.remove()

    pre_keys = dict(pre_capture)

    # PCA
    pca_pre = {}
    for (l, h), K in pre_keys.items():
        Kc = K - K.mean(0)
        _, V = np.linalg.eigh(Kc.T @ Kc / K.shape[0])
        pca_pre[(l, h)] = V

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    info = {'n_layers': n_layers, 'n_kv': n_kv, 'd_head': d_head}
    return model, tokenizer, all_ids, pre_keys, post_keys, pca_pre, R_random, info


def run_lloydmax_mse(device='cuda:0'):
    """GPU 0: Lloyd-Max vs Uniform MSE comparison."""
    import torch
    print(f"\n{'#'*70}")
    print(f"# Lloyd-Max MSE Experiment on {device}")
    print(f"{'#'*70}")
    t0 = time.time()

    model, tokenizer, all_ids, pre_keys, post_keys, pca_pre, R_random, info = load_model_and_keys(device)
    d = info['d_head']

    del model; torch.cuda.empty_cache()  # free GPU, only need numpy now

    results = []
    bits_list = [2, 3, 4]

    for (l, h) in sorted(pre_keys.keys()):
        K_pre = pre_keys[(l, h)]
        K_post = post_keys[(l, h)]
        V = pca_pre[(l, h)]

        for bits in bits_list:
            def compute_mse(K, rot=None, lloyd=False):
                Kr = K @ rot if rot is not None else K.copy()
                Kq = np.zeros_like(Kr)
                for j in range(d):
                    if lloyd:
                        Kq[:, j] = gaussian_lloyd_max(Kr[:, j], bits)
                    else:
                        Kq[:, j] = uniform_quant(Kr[:, j], bits)
                if rot is not None:
                    Kq = Kq @ rot.T
                return float(np.mean((K - Kq)**2))

            results.append({
                'l': l, 'h': h, 'bits': bits,
                'norot_uni':      compute_mse(K_post),
                'norot_lloyd':    compute_mse(K_post, lloyd=True),
                'turbo_uni':      compute_mse(K_post, R_random),
                'turbo_lloyd':    compute_mse(K_post, R_random, lloyd=True),
                'pre_pca_uni':    compute_mse(K_pre, V),
                'pre_pca_lloyd':  compute_mse(K_pre, V, lloyd=True),
            })

    # Summary
    print(f"\n{'='*70}")
    print(f"Lloyd-Max MSE Results (avg over {len(pre_keys)} heads)")
    print(f"{'='*70}")

    methods = ['norot_uni', 'norot_lloyd', 'turbo_uni', 'turbo_lloyd', 'pre_pca_uni', 'pre_pca_lloyd']
    labels = ['NoRot+Uni', 'NoRot+Lloyd', 'Turbo+Uni', 'Turbo+Lloyd', 'PrePCA+Uni', 'PrePCA+Lloyd']

    for bits in bits_list:
        br = [r for r in results if r['bits'] == bits]
        vals = {m: np.mean([r[m] for r in br]) for m in methods}
        best = min(vals.values())

        print(f"\n--- {bits}-bit ---")
        print(f"  {'Method':22s} {'MSE':>10s} {'Ratio':>8s}")
        for m, label in zip(methods, labels):
            v = vals[m]
            r = v/best
            mark = ' ***' if abs(v-best) < 1e-10 else ''
            print(f"  {label:22s} {v:10.6f} {r:7.2f}x{mark}")

        # Key comparisons
        print(f"\n  Lloyd-Max gain (same rotation):")
        print(f"    NoRot:  {vals['norot_uni']/vals['norot_lloyd']:.2f}x")
        print(f"    Turbo:  {vals['turbo_uni']/vals['turbo_lloyd']:.2f}x")
        print(f"    PrePCA: {vals['pre_pca_uni']/vals['pre_pca_lloyd']:.2f}x")

        print(f"  Rotation gain (same quantizer):")
        print(f"    Uniform:  Turbo/PrePCA = {vals['turbo_uni']/vals['pre_pca_uni']:.2f}x")
        print(f"    Lloyd:    Turbo/PrePCA = {vals['turbo_lloyd']/vals['pre_pca_lloyd']:.2f}x")

        print(f"  BEST vs TurboQuant paper (Turbo+Lloyd):")
        print(f"    PrePCA+Lloyd vs Turbo+Lloyd = {vals['turbo_lloyd']/vals['pre_pca_lloyd']:.2f}x")

    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / "lloydmax_v2_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s. Saved: {OUTDIR / 'lloydmax_v2_results.json'}")


def run_niah(device='cuda:1'):
    """GPU 1: NIAH passkey retrieval."""
    import torch
    print(f"\n{'#'*70}")
    print(f"# NIAH Experiment on {device}")
    print(f"{'#'*70}")
    t0 = time.time()

    model, tokenizer, all_ids, pre_keys, post_keys, pca_pre, R_random, info = load_model_and_keys(device)
    n_kv, n_layers, d = info['n_kv'], info['n_layers'], info['d_head']

    filler = "The grass is green and the sky is blue. Today is a wonderful day. "
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_trials = 10
    ctx_len = 4096
    bits_list = [2, 4]

    def make_hook(li, bits, rot_dict=None, rand_rot=None):
        nl = 2 ** bits
        def fn(mod, inp, out):
            k = out[0] if isinstance(out, tuple) else out
            k_np = k.detach().cpu().float().numpy().reshape(-1, n_kv, d)
            for h in range(n_kv):
                Kh = k_np[:, h, :]
                R = rot_dict.get((li, h)) if rot_dict else rand_rot
                Kr = Kh @ R if R is not None else Kh.copy()
                for j in range(d):
                    Kr[:, j] = uniform_quant(Kr[:, j], bits)
                k_np[:, h, :] = Kr @ R.T if R is not None else Kr
            return torch.tensor(k_np.reshape(-1, n_kv*d), dtype=k.dtype, device=k.device).unsqueeze(0)
        return fn

    def test_method(name, bits, rot_dict=None, rand_rot=None):
        hks = []
        if name != 'fp16':
            for l in range(n_layers):
                hks.append(model.model.layers[l].self_attn.k_proj.register_forward_hook(
                    make_hook(l, bits, rot_dict, rand_rot)))

        depth_acc = {}
        for depth in depths:
            correct = 0
            for trial in range(n_trials):
                pk = str(np.random.randint(1000, 9999))
                needle_ids = tokenizer.encode(f"SECRET: The passkey is {pk}. Remember it.", add_special_tokens=False)
                q_ids = tokenizer.encode(f"\nWhat is the passkey? Answer: {pk[0]}", add_special_tokens=False)
                # We check if the model predicts the SECOND digit
                filler_ids = tokenizer.encode(filler, add_special_tokens=False)

                avail = ctx_len - len(needle_ids) - len(q_ids) - 2
                if avail < 10: continue
                ff = (filler_ids * (avail // len(filler_ids) + 1))[:avail]
                pos = int(len(ff) * depth)
                toks = ff[:pos] + needle_ids + ff[pos:] + q_ids

                input_ids = torch.tensor([toks[:ctx_len]], device=device)
                with torch.no_grad():
                    out = model(input_ids, use_cache=False)
                    logits = out.logits[0, -1, :]
                    pred_tok = tokenizer.decode(logits.argmax().item()).strip()
                    # Check if predicted token matches 2nd digit of passkey
                    if pred_tok == pk[1]:
                        correct += 1

            acc = correct / n_trials
            depth_acc[depth] = acc
            print(f"    [{name} {bits}b] d={depth:.2f}: {correct}/{n_trials}={acc:.0%}")

        for hk in hks:
            hk.remove()
        return depth_acc

    results = {}
    print("\n  [FP16]")
    results['fp16'] = test_method('fp16', 0)

    for bits in bits_list:
        print(f"\n  [{bits}-bit]")
        results[f'identity_{bits}'] = test_method('identity', bits)
        results[f'random_{bits}'] = test_method('random', bits, rand_rot=R_random)
        results[f'pre_pca_{bits}'] = test_method('pre_pca', bits, rot_dict=pca_pre)

    # Summary
    print(f"\n{'='*70}")
    print("NIAH Summary")
    print(f"{'='*70}")
    print(f"  {'Method':25s} " + ' '.join(f'd={d:.2f}' for d in depths) + '  Avg')
    for key in sorted(results.keys()):
        accs = [results[key].get(d, 0) for d in depths]
        print(f"  {key:25s} " + ' '.join(f'{a:5.0%}' for a in accs) + f'  {np.mean(accs):.0%}')

    print(f"\n  MSE rank = NIAH rank?")
    for bits in bits_list:
        accs = {m: np.mean(list(results[f'{m}_{bits}'].values()))
                for m in ['identity', 'random', 'pre_pca'] if f'{m}_{bits}' in results}
        mse_pred = ['pre_pca', 'random', 'identity']
        niah_ord = sorted(accs, key=lambda m: accs[m], reverse=True)
        print(f"    {bits}b: MSE={mse_pred}, NIAH={niah_ord} -> {'MATCH' if mse_pred==niah_ord else 'MISMATCH'}")

    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / "niah_v2_results.json", 'w') as f:
        json.dump({k: {str(dk): dv for dk, dv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s")

    del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    t0 = time.time()
    OUTDIR.mkdir(exist_ok=True)
    p0 = Process(target=run_lloydmax_mse, args=('cuda:0',))
    p1 = Process(target=run_niah, args=('cuda:1',))
    p0.start(); p1.start()
    p0.join(); p1.join()
    print(f"\nALL DONE in {time.time()-t0:.1f}s")
