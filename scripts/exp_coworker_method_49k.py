#!/usr/bin/env python3
"""
CRITICAL: Reproduce coworker's best method on 49K test protocol.

Coworker's method: PCA + Lloyd-Max + WF(floor=1) + Sink(k=1)
Their result (2K eval): Mistral +0.14 PPL vs FP16

We verify on 49K test (our protocol) to get the TRUE headline number.

Also compare all combinations:
  A. PCA + Uniform 2-bit (our best so far: +0.802)
  B. PCA + Uniform 2-bit + Sink (our best: +0.741)
  C. PCA + Lloyd 2-bit (uniform bits, catastrophic expected)
  D. PCA + Lloyd + WF(floor=1) (coworker's WF method)
  E. PCA + Lloyd + WF(floor=1) + Sink (coworker's full method) ★
  F. PCA + Lloyd + WF(floor=2) (our WF, for comparison)

Usage:
  CUDA_VISIBLE_DEVICES=3 python exp_coworker_method_49k.py --model mistralai/Mistral-7B-v0.3
"""
import argparse, json, time, gc, os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from pathlib import Path

DTYPE = torch.bfloat16
CALIB_TOKENS = 2048
EVAL_CHUNK = 2048
MAX_EVAL = 50000
N_SINK = 1


def lloyd_1d(col, bits, n_iter=25):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([float(col.mean())], dtype=np.float32)
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.sort(np.percentile(col, pcts)).astype(np.float64)
    for _ in range(n_iter):
        b = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(b, col)
        new_c = centroids.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - centroids)) < 1e-6:
            break
        centroids = new_c
    return centroids.astype(np.float32)


def uniform_quant(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def wf_alloc(sigma2, total_budget, b_floor=1, b_max=8):
    """Coworker's WF: greedy, floor=1, allows 0-bit (inactive) dims."""
    n = len(sigma2)
    s = np.maximum(sigma2, 1e-12)
    bits = np.zeros(n, dtype=int)
    spent = 0
    while spent < total_budget:
        best_g = -np.inf
        best = None
        for j in range(n):
            if bits[j] == 0:
                if spent + b_floor > total_budget:
                    continue
                g = s[j] * (1.0 - 4.0 ** (-b_floor)) / b_floor
                if g > best_g:
                    best_g, best = g, ('act', j)
            elif bits[j] < b_max:
                g = s[j] * (4.0 ** (-bits[j]) - 4.0 ** (-(bits[j] + 1)))
                if g > best_g:
                    best_g, best = g, ('add', j)
        if best is None:
            break
        op, j = best
        if op == 'act':
            bits[j] = b_floor
            spent += b_floor
        else:
            bits[j] += 1
            spent += 1
    return bits


class MethodHook:
    def __init__(self, n_kv, d_head, pca_bases, lloyd_centroids, bit_allocs, use_sink, n_sink=1):
        """
        pca_bases: dict h → V (d,d)
        lloyd_centroids: dict h → dict dim → centroids array
        bit_allocs: dict h → array of bits per dim
        """
        self.n_kv = n_kv
        self.d_head = d_head
        self.V = pca_bases
        self.centroids = lloyd_centroids
        self.bits = bit_allocs
        self.use_sink = use_sink
        self.n_sink = n_sink

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.d_head).float().cpu().numpy()
        out = x.copy()

        for h in range(self.n_kv):
            V = self.V[h]
            data = x[:, :, h, :]  # (B, T, d)

            for b_idx in range(B):
                block = data[b_idx]  # (T, d)
                start_t = self.n_sink if self.use_sink else 0

                if start_t >= T:
                    continue

                tokens = block[start_t:]  # (T-start, d)
                rotated = tokens @ V

                quantized = np.zeros_like(rotated)
                ba = self.bits[h]
                for j in range(self.d_head):
                    b = int(ba[j])
                    if b <= 0:
                        quantized[:, j] = 0.0  # zeroed out
                    elif self.centroids is not None and h in self.centroids and j in self.centroids[h]:
                        c = self.centroids[h][j]
                        if len(c) > 1:
                            boundaries = (c[:-1] + c[1:]) / 2
                            idx = np.searchsorted(boundaries, rotated[:, j])
                            quantized[:, j] = c[idx]
                        else:
                            quantized[:, j] = c[0]
                    else:
                        quantized[:, j] = uniform_quant(rotated[:, j], b)

                out[b_idx, start_t:, h, :] = quantized @ V.T
                out[b_idx, :start_t, h, :] = block[:start_t]

        return torch.from_numpy(out).to(output.device).to(output.dtype).view(B, T, -1)


def eval_ppl(model, eval_ids, device):
    model.eval()
    total_len = min(eval_ids.shape[1], MAX_EVAL)
    n_chunks = total_len // EVAL_CHUNK
    total_nll, total_scored = 0.0, 0
    for i in range(n_chunks):
        chunk = eval_ids[:, i*EVAL_CHUNK:(i+1)*EVAL_CHUNK].to(device)
        with torch.no_grad():
            out = model(chunk, labels=chunk, use_cache=False)
        total_nll += out.loss.item() * (EVAL_CHUNK - 1)
        total_scored += EVAL_CHUNK - 1
    return float(np.exp(total_nll / total_scored))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="results/coworker_49k")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    short = args.model.split("/")[-1].replace(".", "_")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"{'='*60}")
    print(f"COWORKER METHOD VERIFICATION: {args.model}")
    print(f"Protocol: 49K test, Lloyd + WF(floor=1) + PCA + Sink")
    print(f"{'='*60}")

    tok_kw = {"trust_remote_code": True}
    mdl_kw = {"torch_dtype": DTYPE, "trust_remote_code": True}
    if args.hf_token:
        tok_kw["token"] = args.hf_token
        mdl_kw["token"] = args.hf_token

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(args.model, **mdl_kw).to(device).eval()

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads

    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)

    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_ids = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)[:, :CALIB_TOKENS].to(device)

    # FP16
    print("\n[FP16]")
    ppl_fp16 = eval_ppl(model, eval_ids, device)
    print(f"  PPL = {ppl_fp16:.4f}")

    # Calibrate PCA + eigenvalues
    print("\n[Calibration]")
    k_data = {}
    hooks = []
    for li in range(n_layers):
        def kh(li_=li):
            def fn(mod, inp, out):
                k_data[li_] = out.detach().cpu().float().numpy()
            return fn
        hooks.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(kh()))
    with torch.no_grad():
        model(calib_ids, use_cache=False)
    for h in hooks:
        h.remove()

    pca_all = {}
    eigvals_all = {}
    calib_pca_data = {}
    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        for hk in range(n_kv):
            K = k_np[:, hk, :].astype(np.float32)
            Kc = K - K.mean(0)
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1) + 1e-6 * np.eye(d_head)
            ev, V = np.linalg.eigh(cov)
            V = V[:, ::-1].astype(np.float32)
            ev = ev[::-1].astype(np.float32)
            pca_all[(li, hk)] = V
            eigvals_all[(li, hk)] = ev
            calib_pca_data[(li, hk)] = Kc @ V  # PCA'd calibration data
    del k_data
    gc.collect()
    print(f"  Calibrated {n_layers}×{n_kv} heads")

    # Configs to test
    configs = []

    # A: PCA + Uniform 2-bit (no sink)
    configs.append(("A_pca_uni2", False, False, "PCA+Uniform 2-bit"))
    # B: PCA + Uniform 2-bit + Sink
    configs.append(("B_pca_uni2_sink", True, False, "PCA+Uniform 2-bit+Sink"))
    # C: PCA + Lloyd uniform 2-bit (all dims 2-bit Lloyd)
    configs.append(("C_pca_lloyd2", False, True, "PCA+Lloyd 2-bit (uniform)"))
    # D: PCA + Lloyd + WF(floor=1)
    configs.append(("D_pca_lloyd_wf1", False, "wf1", "PCA+Lloyd+WF(floor=1)"))
    # E: PCA + Lloyd + WF(floor=1) + Sink ★
    configs.append(("E_pca_lloyd_wf1_sink", True, "wf1", "PCA+Lloyd+WF(floor=1)+Sink ★"))
    # F: PCA + Lloyd + WF(floor=2)
    configs.append(("F_pca_lloyd_wf2", False, "wf2", "PCA+Lloyd+WF(floor=2)"))

    results = {"model": args.model, "fp16": round(ppl_fp16, 4), "configs": {}}

    for cfg_name, use_sink, lloyd_mode, desc in configs:
        out_path = out_dir / f"{short}_{cfg_name}.json"
        if out_path.exists():
            cached = json.loads(out_path.read_text())
            print(f"\n[{cfg_name}] {desc} → PPL={cached['ppl']} (cached)")
            results["configs"][cfg_name] = cached
            continue

        print(f"\n[{cfg_name}] {desc}")
        t0 = time.time()

        hook_handles = []
        for li in range(n_layers):
            pca_b = {hk: pca_all[(li, hk)] for hk in range(n_kv)}
            lloyd_c = {}
            bit_allocs = {}

            for hk in range(n_kv):
                ev = eigvals_all[(li, hk)]
                cal_data = calib_pca_data[(li, hk)]

                if lloyd_mode == "wf1":
                    ba = wf_alloc(ev, d_head * 2, b_floor=1, b_max=8)
                elif lloyd_mode == "wf2":
                    ba = wf_alloc(ev, d_head * 2, b_floor=2, b_max=8)
                elif lloyd_mode is True:
                    ba = np.full(d_head, 2, dtype=int)
                else:  # False = uniform quantization
                    ba = np.full(d_head, 2, dtype=int)

                bit_allocs[hk] = ba

                if lloyd_mode is not False:
                    # Fit Lloyd per dim
                    lloyd_c[hk] = {}
                    for j in range(d_head):
                        b = int(ba[j])
                        if b > 0:
                            lloyd_c[hk][j] = lloyd_1d(cal_data[:, j], b)
                else:
                    lloyd_c = None

            hook = MethodHook(n_kv, d_head, pca_b,
                             lloyd_c if lloyd_mode is not False else None,
                             bit_allocs, use_sink, N_SINK)
            hook_handles.append(
                model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))

        ppl = eval_ppl(model, eval_ids, device)
        for h in hook_handles:
            h.remove()
        elapsed = time.time() - t0

        delta = ppl - ppl_fp16
        avg_bits = np.mean([bit_allocs[hk].mean() for hk in range(n_kv)])
        result = {
            "ppl": round(ppl, 4), "delta": round(delta, 4),
            "desc": desc, "runtime_sec": round(elapsed, 1),
            "avg_bits": round(float(avg_bits), 3),
        }
        out_path.write_text(json.dumps(result, indent=2))
        results["configs"][cfg_name] = result
        print(f"  PPL = {ppl:.4f} (Δ={delta:+.4f}, avg_bits={avg_bits:.2f}, {elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    print(f"FP16: {ppl_fp16:.4f}")
    for cfg_name, use_sink, lloyd_mode, desc in configs:
        c = results["configs"][cfg_name]
        print(f"  {cfg_name}: PPL={c['ppl']:.4f} (Δ={c['delta']:+.4f}) {desc}")

    full_path = out_dir / f"{short}_coworker_49k_full.json"
    full_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {full_path}")


if __name__ == "__main__":
    main()
