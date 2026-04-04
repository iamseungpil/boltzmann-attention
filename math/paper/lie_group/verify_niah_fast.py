#!/usr/bin/env python3
"""
Fast NIAH: Pre-RoPE PCA+Lloyd+QJL vs TurboQuant+Lloyd+QJL only.
Single token prediction (no generation). Unbuffered output.
Usage: PYTHONUNBUFFERED=1 python3 verify_niah_fast.py --device cuda:0
"""
import sys, json, time, argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.stdout.reconfigure(line_buffering=True)  # force line buffering

OUTDIR = Path(__file__).parent / "verification_results"
LM_CB = {
    2: (np.array([-0.9816, 0.0, 0.9816]), np.array([-1.5104, -0.4528, 0.4528, 1.5104])),
    3: (np.array([-1.748, -1.05, -0.5006, 0.0, 0.5006, 1.05, 1.748]),
        np.array([-2.1519, -1.344, -0.756, -0.2451, 0.2451, 0.756, 1.344, 2.1519])),
}

def lm_q(x, bits):
    mu, sig = float(np.mean(x)), max(float(np.std(x)), 1e-10)
    xn = np.clip((x - mu) / sig, -3.5, 3.5)
    th, ct = LM_CB[bits]
    return ct[np.searchsorted(th, xn)] * sig + mu

def uni_q(x, bits):
    nl = 2**bits; lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-10: return x.copy()
    s = (hi-lo)/(nl-1)
    return np.clip(np.round((x-lo)/s).astype(int), 0, nl-1) * s + lo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 3])
    args = parser.parse_args()
    device = args.device

    print(f"NIAH Fast: device={device}, bits={args.bits}", flush=True)

    model_name = 'Qwen/Qwen2.5-7B'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    print("Model loaded", flush=True)

    cfg = model.config
    n_kv = cfg.num_key_value_heads
    n_layers = cfg.num_hidden_layers
    d = cfg.hidden_size // cfg.num_attention_heads

    # Calibration
    print("[Cal] start", flush=True)
    filler = "The grass is green and the sky is blue. Today is wonderful. "
    cal_ids = tokenizer.encode(filler*80, return_tensors="pt", truncation=False)[:, :2048].to(device)

    pca = {}; capture = {}
    hooks = []
    for l in range(n_layers):
        def mk(li):
            def fn(mod, args, kwargs):
                hs = args[0] if args else kwargs.get('hidden_states')
                if hs is not None:
                    k = mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1, n_kv, d)
                    for h in range(n_kv):
                        K = k[:,h,:]; Kc = K - K.mean(0)
                        capture[(li,h)] = Kc.T @ Kc / K.shape[0]
            return fn
        hooks.append(model.model.layers[l].self_attn.register_forward_pre_hook(mk(l), with_kwargs=True))

    with torch.no_grad():
        model(cal_ids, use_cache=False)
    for h in hooks: h.remove()

    for (l,h), cov in capture.items():
        _, V = np.linalg.eigh(cov)
        pca[(l,h)] = V

    np.random.seed(42)
    R_rand = np.linalg.qr(np.random.randn(d,d))[0]
    print(f"[Cal] done: {len(pca)} heads", flush=True)

    # Hook factory
    active_method = [None]
    active_bits = [2]

    def quant_hook(li):
        def fn(mod, inp, out):
            method = active_method[0]
            if method is None: return out
            bits = active_bits[0]
            k = out
            k_np = k[0].detach().cpu().float().numpy().reshape(-1, n_kv, d)
            for h in range(n_kv):
                Kh = k_np[:,h,:]
                if method == 'pre_pca':
                    V = pca.get((li,h))
                    if V is None: continue
                    mu = Kh.mean(0); Kr = (Kh-mu) @ V
                elif method == 'turbo':
                    mu = np.zeros(d); Kr = Kh @ R_rand
                else:
                    continue
                for j in range(d):
                    Kr[:,j] = lm_q(Kr[:,j], bits)
                if method == 'pre_pca':
                    k_np[:,h,:] = Kr @ V.T + mu
                else:
                    k_np[:,h,:] = Kr @ R_rand.T
            return torch.tensor(k_np.reshape(-1, n_kv*d), dtype=k.dtype, device=k.device).unsqueeze(0)
        return fn

    # Install hooks once
    hook_handles = []
    for l in range(n_layers):
        hook_handles.append(
            model.model.layers[l].self_attn.k_proj.register_forward_hook(quant_hook(l)))

    # NIAH
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_trials = 5
    ctx_len = 4096
    methods = [('fp16', None), ('turbo', 'turbo'), ('pre_pca', 'pre_pca')]

    results = {}

    for bits in args.bits:
        active_bits[0] = bits
        for method_name, method_key in methods:
            key = f"{method_name}_{bits}b"
            if method_name == 'fp16':
                key = 'fp16'
                if key in results: continue

            active_method[0] = method_key  # None for fp16
            print(f"\n[{key}]", flush=True)

            depth_acc = {}
            for depth in depths:
                correct = 0
                for trial in range(n_trials):
                    pk = str(np.random.randint(10000, 99999))
                    needle = f"IMPORTANT: The secret code is {pk}."
                    question = f"\nWhat is the secret code mentioned above? The code is "

                    nid = tokenizer.encode(needle, add_special_tokens=False)
                    qid = tokenizer.encode(question, add_special_tokens=False)
                    fid = tokenizer.encode(filler, add_special_tokens=False)

                    avail = ctx_len - len(nid) - len(qid) - 5
                    ff = (fid * (avail//len(fid)+2))[:avail]
                    pos = max(0, min(int(len(ff)*depth), len(ff)))
                    toks = ff[:pos] + nid + ff[pos:] + qid
                    toks = toks[:ctx_len]

                    input_ids = torch.tensor([toks], device=device)
                    with torch.no_grad():
                        out = model(input_ids, use_cache=False)
                        logits = out.logits[0, -1, :]
                        top10 = torch.topk(logits, 10).indices.tolist()
                        top10_text = [tokenizer.decode(t).strip() for t in top10]
                        # Check if first digit of passkey appears in top-3
                        if any(pk[:1] in t for t in top10_text[:3]):
                            correct += 1

                acc = correct / n_trials
                depth_acc[depth] = acc
                print(f"  d={depth:.2f}: {correct}/{n_trials}={acc:.0%}", flush=True)

            avg = np.mean(list(depth_acc.values()))
            results[key] = {'depths': {str(d): a for d,a in depth_acc.items()}, 'avg': float(avg)}
            print(f"  avg={avg:.0%}", flush=True)

    # Remove hooks
    for h in hook_handles: h.remove()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("NIAH Summary", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Method':25s} " + ' '.join(f'd={d:.2f}' for d in depths) + '  Avg', flush=True)
    for key in sorted(results.keys()):
        r = results[key]
        accs = [r['depths'].get(str(d), 0) for d in depths]
        print(f"  {key:25s} " + ' '.join(f'{a:5.0%}' for a in accs) + f'  {r["avg"]:.0%}', flush=True)

    # Ranking check
    for bits in args.bits:
        accs = {}
        for mn, mk in methods:
            k = f"{mn}_{bits}b" if mn != 'fp16' else 'fp16'
            if k in results: accs[mn] = results[k]['avg']
        if 'turbo' in accs and 'pre_pca' in accs:
            winner = 'pre_pca' if accs['pre_pca'] >= accs['turbo'] else 'turbo'
            print(f"\n  {bits}b: pre_pca={accs.get('pre_pca',0):.0%} vs turbo={accs.get('turbo',0):.0%} -> {winner}", flush=True)

    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / f"niah_fast.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved. Total: {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
if __name__ == '__main__':
    main()
