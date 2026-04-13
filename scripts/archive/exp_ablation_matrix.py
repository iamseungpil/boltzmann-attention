#!/usr/bin/env python3
"""
CRITICAL: 3-factor ablation matrix on 49K test.
Tests all combinations of PCA / Sink / WF independently.

9 configs per model:
  1. Baseline: no PCA, no sink, no WF (raw Uniform 2-bit)
  2. +PCA only
  3. +Sink only (BOS FP16)
  4. +WF only (floor=2)
  5. +PCA +Sink
  6. +PCA +WF
  7. +Sink +WF
  8. +PCA +Sink +WF  ← HEADLINE RESULT
  9. FP16 baseline

Protocol: WikiText-2 TEST split, 49K tokens, 2048 chunks.
Calibration: WikiText-2 TRAIN, 2048 tokens.

Usage:
  CUDA_VISIBLE_DEVICES=3 python exp_ablation_matrix.py --model mistralai/Mistral-7B-v0.3
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
BITS = 2
N_SINK = 1  # BOS only
WF_FLOOR = 2


def uniform_quant(col, bits):
    nl = 2 ** bits
    vmin, vmax = col.min(), col.max()
    if vmax - vmin < 1e-10:
        return col.copy()
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


def water_filling_bits(eigvals, avg_bits, floor=2):
    """Greedy WF: start at floor, add +1 to highest-importance dims."""
    d = len(eigvals)
    total = d * avg_bits
    imp = np.maximum(eigvals, 1e-12).astype(np.float64)
    bits = np.full(d, floor, dtype=int)
    spent = d * floor
    if spent > total:
        return bits
    while spent < total:
        # marginal gain of +1 bit: importance × (4^(-b) - 4^(-(b+1)))
        gains = imp * (4.0 ** (-bits.astype(float)) - 4.0 ** (-(bits + 1).astype(float)))
        gains[bits >= 8] = -np.inf  # cap at 8 bits
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            break
        bits[j] += 1
        spent += 1
    return bits


class AblationHook:
    def __init__(self, n_kv, d_head, use_pca, use_wf, use_sink,
                 pca_bases=None, eigvals=None, n_sink=1):
        self.n_kv = n_kv
        self.d_head = d_head
        self.use_pca = use_pca
        self.use_wf = use_wf
        self.use_sink = use_sink
        self.pca_bases = pca_bases  # dict h → V
        self.eigvals = eigvals      # dict h → eigenvalues
        self.n_sink = n_sink
        # precompute WF bits
        self.wf_bits = {}
        if use_wf and eigvals:
            for h in range(n_kv):
                self.wf_bits[h] = water_filling_bits(eigvals[h], BITS, WF_FLOOR)

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.d_head).float().cpu().numpy()
        out = x.copy()

        for h in range(self.n_kv):
            data = x[:, :, h, :]  # (B, T, d)

            for b_idx in range(B):
                block = data[b_idx]  # (T, d)

                # Sink protection: keep first n_sink tokens as-is
                start_t = self.n_sink if self.use_sink else 0

                if start_t >= T:
                    out[b_idx, :, h, :] = block
                    continue

                tokens_to_quant = block[start_t:]  # (T-start, d)

                if self.use_pca and self.pca_bases:
                    V = self.pca_bases[h]
                    rotated = tokens_to_quant @ V
                else:
                    rotated = tokens_to_quant.copy()

                # Quantize
                quantized = np.zeros_like(rotated)
                if self.use_wf and h in self.wf_bits:
                    bits_alloc = self.wf_bits[h]
                    for j in range(self.d_head):
                        b = max(int(bits_alloc[j]), 1)
                        quantized[:, j] = uniform_quant(rotated[:, j], b)
                else:
                    for j in range(self.d_head):
                        quantized[:, j] = uniform_quant(rotated[:, j], BITS)

                if self.use_pca and self.pca_bases:
                    V = self.pca_bases[h]
                    reconstructed = quantized @ V.T
                else:
                    reconstructed = quantized

                out[b_idx, start_t:, h, :] = reconstructed
                # sink tokens stay as original
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
        scored = EVAL_CHUNK - 1
        total_nll += out.loss.item() * scored
        total_scored += scored
    return float(np.exp(total_nll / total_scored)), n_chunks, total_scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="results/ablation_matrix")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    short = args.model.split("/")[-1].replace(".", "_")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"{'='*60}")
    print(f"ABLATION MATRIX: {args.model}")
    print(f"Protocol: WikiText-2 TEST, 49K tokens, 2048 chunks")
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

    # Eval data (TEST)
    eval_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in eval_ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)
    print(f"  Eval: {eval_ids.shape[1]} tokens → {min(eval_ids.shape[1], MAX_EVAL)} used")

    # Calib data (TRAIN)
    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_ids = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)[:, :CALIB_TOKENS].to(device)

    # FP16 baseline
    print("\n[FP16]")
    ppl_fp16, nc, ns = eval_ppl(model, eval_ids, device)
    print(f"  PPL = {ppl_fp16:.4f} ({nc} chunks, {ns} tokens)")

    # Calibrate PCA
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

    pca_bases_all = {}  # (layer, head) → V
    eigvals_all = {}    # (layer, head) → eigenvalues
    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        for hk in range(n_kv):
            K = k_np[:, hk, :].astype(np.float32)
            Kc = K - K.mean(0)
            cov = (Kc.T @ Kc) / max(K.shape[0]-1, 1) + 1e-6 * np.eye(d_head)
            ev, V = np.linalg.eigh(cov)
            V = V[:, ::-1].astype(np.float32)
            ev = ev[::-1]
            pca_bases_all[(li, hk)] = V
            eigvals_all[(li, hk)] = ev.astype(np.float32)
    del k_data
    gc.collect()
    print(f"  Calibrated {n_layers}×{n_kv} heads")

    # 9 configs: all combinations of PCA/Sink/WF
    configs = [
        ("baseline",       False, False, False, "No PCA, No Sink, No WF"),
        ("pca",            True,  False, False, "+PCA only"),
        ("sink",           False, True,  False, "+Sink only"),
        ("wf",             False, False, True,  "+WF only"),
        ("pca_sink",       True,  True,  False, "+PCA +Sink"),
        ("pca_wf",         True,  False, True,  "+PCA +WF"),
        ("sink_wf",        False, True,  True,  "+Sink +WF"),
        ("pca_sink_wf",    True,  True,  True,  "+PCA +Sink +WF ★"),
    ]

    results = {"model": args.model, "fp16": round(ppl_fp16, 4), "protocol": "49K_test", "configs": {}}

    for cfg_name, use_pca, use_sink, use_wf, desc in configs:
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
            pca_b = {hk: pca_bases_all[(li, hk)] for hk in range(n_kv)} if use_pca else None
            ev = {hk: eigvals_all[(li, hk)] for hk in range(n_kv)} if use_wf else None
            hook = AblationHook(n_kv, d_head, use_pca, use_wf, use_sink,
                               pca_bases=pca_b, eigvals=ev, n_sink=N_SINK)
            hook_handles.append(
                model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))

        ppl, nc, ns = eval_ppl(model, eval_ids, device)
        for h in hook_handles:
            h.remove()
        elapsed = time.time() - t0

        delta = ppl - ppl_fp16
        result = {
            "ppl": round(ppl, 4), "delta": round(delta, 4),
            "pca": use_pca, "sink": use_sink, "wf": use_wf,
            "desc": desc, "runtime_sec": round(elapsed, 1),
        }
        out_path.write_text(json.dumps(result, indent=2))
        results["configs"][cfg_name] = result
        print(f"  PPL = {ppl:.4f} (Δ={delta:+.4f}, {elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*60}")
    print(f"ABLATION MATRIX: {args.model}")
    print(f"{'='*60}")
    print(f"{'Config':<20s} {'PCA':>4s} {'Sink':>5s} {'WF':>4s} {'PPL':>8s} {'Δ':>8s}")
    print(f"{'-'*50}")
    print(f"{'FP16':<20s} {'':>4s} {'':>5s} {'':>4s} {ppl_fp16:>8.4f} {0:>+8.4f}")
    for cfg_name, use_pca, use_sink, use_wf, desc in configs:
        c = results["configs"][cfg_name]
        p = "✓" if use_pca else ""
        s = "✓" if use_sink else ""
        w = "✓" if use_wf else ""
        print(f"{cfg_name:<20s} {p:>4s} {s:>5s} {w:>4s} {c['ppl']:>8.4f} {c['delta']:>+8.4f}")

    # Save full
    full_path = out_dir / f"{short}_ablation_full.json"
    full_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {full_path}")


if __name__ == "__main__":
    main()
