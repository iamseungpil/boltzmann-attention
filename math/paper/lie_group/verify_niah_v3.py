#!/usr/bin/env python3
"""
NIAH Final v3: Sequential execution (no multiprocessing CUDA fork issue)
Pre-RoPE PCA + Lloyd-Max + QJL  vs  TurboQuant + Lloyd-Max + QJL
Single GPU, sequential. Run two instances manually for parallelism.
Usage: python3 verify_niah_v3.py --device cuda:0 --bits 2
       python3 verify_niah_v3.py --device cuda:1 --bits 3
"""
import argparse, json, time, sys
import numpy as np
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTDIR = Path(__file__).parent / "verification_results"

LM_CB = {
    2: (np.array([-0.9816, 0.0, 0.9816]),
        np.array([-1.5104, -0.4528, 0.4528, 1.5104])),
    3: (np.array([-1.7480, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7480]),
        np.array([-2.1519, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1519])),
}

def lloyd_max_q(x, bits):
    mu, sig = np.mean(x), max(np.std(x), 1e-10)
    xn = np.clip((x - mu) / sig, -3.5, 3.5)
    th, ct = LM_CB[bits]
    return ct[np.searchsorted(th, xn)] * sig + mu

def uniform_q(x, bits):
    nl = 2**bits; vmin, vmax = x.min(), x.max()
    if vmax - vmin < 1e-10: return x.copy()
    s = (vmax - vmin)/(nl-1)
    return np.clip(np.round((x-vmin)/s).astype(int), 0, nl-1) * s + vmin

def qjl_correct(K_orig, K_quant, d_proj=32):
    n, d = K_orig.shape
    res = K_orig - K_quant
    np.random.seed(12345)
    P = np.random.randn(d, d_proj) / np.sqrt(d_proj)
    rp = res @ P
    rpq = np.stack([uniform_q(rp[:, j], 8) for j in range(d_proj)], axis=1)
    return K_quant + rpq @ P.T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--bits', type=int, default=2)
    args = parser.parse_args()

    device = args.device
    bits = args.bits
    print(f"NIAH v3: device={device}, bits={bits}")

    model_name = 'Qwen/Qwen2.5-7B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    cfg = model.config
    n_kv = cfg.num_key_value_heads
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  layers={n_layers}, kv={n_kv}, d={d_head}")

    # Calibration
    print("[Calibration]...")
    filler = "The grass is green and the sky is blue. Today is a wonderful day. "
    cal_ids = tokenizer.encode(filler * 80, return_tensors="pt", truncation=False)[:, :2048].to(device)

    pca_bases = {}
    capture = {}

    def make_cal_hook(li):
        def fn(mod, args, kwargs):
            hs = args[0] if args else kwargs.get('hidden_states')
            if hs is not None:
                k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    Kh = k[:, h, :]
                    Kc = Kh - Kh.mean(0)
                    capture[(li, h)] = Kc.T @ Kc / Kh.shape[0]
        return fn

    hooks = []
    for l in range(n_layers):
        hooks.append(model.model.layers[l].self_attn.register_forward_pre_hook(
            make_cal_hook(l), with_kwargs=True))

    with torch.no_grad():
        model(cal_ids, use_cache=False)

    for hook in hooks:
        hook.remove()

    for (l, h), cov in capture.items():
        _, V = np.linalg.eigh(cov)
        pca_bases[(l, h)] = V

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]
    print(f"  Calibration done: {len(pca_bases)} heads")

    # Quantization hook
    def make_quant_hook(li, method):
        def fn(mod, inp, out):
            k = out
            k_np = k[0].detach().cpu().float().numpy().reshape(-1, n_kv, d_head)

            for h in range(n_kv):
                Kh = k_np[:, h, :]

                if 'pre_pca' in method:
                    V = pca_bases.get((li, h))
                    if V is None: continue
                    mu = Kh.mean(0)
                    Kr = (Kh - mu) @ V
                elif 'turbo' in method:
                    mu = np.zeros(d_head)
                    Kr = Kh @ R_random
                else:
                    mu = np.zeros(d_head)
                    Kr = Kh.copy()

                # Quantize
                Kq = np.zeros_like(Kr)
                if 'lloyd' in method:
                    for j in range(d_head):
                        Kq[:, j] = lloyd_max_q(Kr[:, j], bits)
                else:
                    for j in range(d_head):
                        Kq[:, j] = uniform_q(Kr[:, j], bits)

                # Inverse rotation
                if 'pre_pca' in method:
                    Kh_q = Kq @ V.T + mu
                elif 'turbo' in method:
                    Kh_q = Kq @ R_random.T
                else:
                    Kh_q = Kq

                # QJL correction
                if 'qjl' in method:
                    Kh_q = qjl_correct(Kh, Kh_q)

                k_np[:, h, :] = Kh_q

            return torch.tensor(k_np.reshape(-1, n_kv*d_head), dtype=k.dtype, device=k.device).unsqueeze(0)
        return fn

    # NIAH
    depths = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    n_trials = 10
    ctx_lengths = [4096, 8192]
    methods = ['fp16', 'turbo_lloyd_qjl', 'pre_pca_lloyd_qjl', 'turbo_uni', 'pre_pca_uni']

    results = {}

    for ctx_len in ctx_lengths:
        for method in methods:
            key = f"{method}_{bits}b_{ctx_len}" if method != 'fp16' else f"fp16_{ctx_len}"
            if key in results: continue

            print(f"\n[{method} {bits}b ctx={ctx_len}]")

            hks = []
            if method != 'fp16':
                for l in range(n_layers):
                    hks.append(model.model.layers[l].self_attn.k_proj.register_forward_hook(
                        make_quant_hook(l, method)))

            depth_acc = {}
            for depth in depths:
                correct = 0
                for trial in range(n_trials):
                    pk = str(np.random.randint(10000, 99999))
                    needle = f"CRITICAL: The access code is {pk}."
                    question = f"\nWhat is the access code? Answer: "

                    nid = tokenizer.encode(needle, add_special_tokens=False)
                    qid = tokenizer.encode(question, add_special_tokens=False)
                    fid = tokenizer.encode(filler, add_special_tokens=False)

                    avail = ctx_len - len(nid) - len(qid) - 5
                    if avail < 50: continue
                    ff = (fid * (avail//len(fid)+2))[:avail]
                    pos = max(0, min(int(len(ff)*depth), len(ff)))
                    toks = ff[:pos] + nid + ff[pos:] + qid
                    toks = toks[:ctx_len]

                    input_ids = torch.tensor([toks], device=device)
                    with torch.no_grad():
                        # Generate 6 tokens
                        gen = []
                        cur = input_ids
                        for _ in range(6):
                            out = model(cur, use_cache=False)
                            nxt = out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                            gen.append(nxt.item())
                            cur = torch.cat([cur, nxt], dim=1)

                        gen_text = tokenizer.decode(gen).strip()
                        if pk in gen_text:
                            correct += 1

                acc = correct / n_trials
                depth_acc[depth] = acc
                print(f"  d={depth:.2f}: {correct}/{n_trials}={acc:.0%}")

            for hk in hks:
                hk.remove()

            avg = np.mean(list(depth_acc.values()))
            results[key] = {'depths': {str(d): a for d, a in depth_acc.items()}, 'avg': float(avg)}
            print(f"  avg={avg:.0%}")

    # Summary
    print(f"\n{'='*70}")
    print(f"NIAH Summary (bits={bits})")
    print(f"{'='*70}")
    for ctx_len in ctx_lengths:
        print(f"\n  ctx={ctx_len}")
        print(f"  {'Method':30s} " + ' '.join(f'd={d:.2f}' for d in depths) + '  Avg')
        for method in methods:
            key = f"{method}_{bits}b_{ctx_len}" if method != 'fp16' else f"fp16_{ctx_len}"
            if key not in results: continue
            r = results[key]
            accs = [r['depths'].get(str(d), 0) for d in depths]
            print(f"  {method:30s} " + ' '.join(f'{a:5.0%}' for a in accs) + f'  {r["avg"]:.0%}')

    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / f"niah_final_{bits}b.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTDIR / f'niah_final_{bits}b.json'}")

    del model; torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
