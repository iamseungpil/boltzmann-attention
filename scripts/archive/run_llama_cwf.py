#!/usr/bin/env python3
"""
Llama-3.1-8B CWF cross-verification (Delegation Request 1).
Phase 1: Per-layer sensitivity measurement (Exp4-style)
Phase 2: CWF sweep at [2.0, 2.156, 2.5, 3.0, 3.5] avg_bits

Usage:
  CUDA_VISIBLE_DEVICES=2 python run_llama_cwf.py
"""
import json, time, gc, os, sys
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'meta-llama/Llama-3.1-8B'
SHORT = 'llama-8b'
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024
N_EVAL_TOKENS = 2048
B_FLOOR = 2
B_MAX = 6
HF_TOKEN = os.environ.get('HF_TOKEN', '')

OUT_DIR = Path('results/cwf_llama')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CWF_BUDGETS = [2.0, 2.156, 2.5, 3.0, 3.5]


def lloyd_max_1d_fit(col, bits, n_iter=30):
    n_levels = 2 ** bits
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.percentile(col, pcts)
    centroids = np.sort(centroids)
    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(boundaries, col)
        new_c = centroids.copy()
        for k in range(n_levels):
            m = idx == k
            if m.sum() > 0:
                new_c[k] = col[m].mean()
        if np.max(np.abs(new_c - centroids)) < 1e-6:
            break
        centroids = new_c
    return centroids


def water_filling_global(importance, total_budget, b_floor=2, b_max=6):
    """Greedy WF: allocate bits to highest-importance heads first."""
    n = len(importance)
    imp = np.maximum(np.array(importance, dtype=np.float64), 1e-12)
    bits = np.full(n, b_floor, dtype=int)
    spent = n * b_floor
    if spent > total_budget:
        return bits
    while spent < total_budget:
        valid = bits < b_max
        if not valid.any():
            break
        gains = np.where(
            valid,
            imp * (4.0 ** (-bits.astype(float)) - 4.0 ** (-(bits + 1).astype(float))),
            -np.inf
        )
        j_best = int(np.argmax(gains))
        bits[j_best] += 1
        spent += 1
    return bits


def fit_pca_l2_lloyd(K, bits):
    K = K.astype(np.float32)
    K_mean = K.mean(axis=0)
    K_c = K - K_mean
    cov = (K_c.T @ K_c) / max(K.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order]
    K_pca = K_c @ V
    d = K.shape[1]
    centroids = np.zeros((d, 2 ** bits), dtype=np.float32)
    for j in range(d):
        centroids[j] = lloyd_max_1d_fit(K_pca[:, j], bits, n_iter=20).astype(np.float32)
    return {'K_mean': K_mean, 'V': V.astype(np.float32), 'centroids': centroids, 'bits': bits}


class PCAL2LloydHook:
    def __init__(self, head_quantizers, n_kv, head_dim):
        self.hq = head_quantizers
        self.n_kv = n_kv
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x_bf = output.view(B, T, self.n_kv, self.head_dim)
        x_np = x_bf.float().cpu().numpy()
        x_q = np.zeros_like(x_np)
        for hk in range(self.n_kv):
            q = self.hq[hk]
            data = x_np[:, :, hk, :]
            shape = data.shape
            K_flat = data.reshape(-1, self.head_dim).astype(np.float32)
            K_c = K_flat - q['K_mean']
            K_pca = K_c @ q['V']
            K_pca_q = np.zeros_like(K_pca)
            c = q['centroids']
            for j in range(self.head_dim):
                boundaries = (c[j, :-1] + c[j, 1:]) / 2
                idx = np.searchsorted(boundaries, K_pca[:, j])
                K_pca_q[:, j] = c[j, idx]
            K_recon = K_pca_q @ q['V'].T + q['K_mean']
            x_q[:, :, hk, :] = K_recon.reshape(shape)
        result = torch.from_numpy(x_q).to(output.device).to(output.dtype)
        return result.view(B, T, self.n_kv * self.head_dim)


def eval_ppl(model, eval_ids, seq_len=N_EVAL_TOKENS):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    n_chunks = eval_ids.shape[1] // seq_len
    for i in range(n_chunks):
        chunk = eval_ids[:, i*seq_len:(i+1)*seq_len].to(DEVICE)
        with torch.no_grad():
            out = model(chunk, labels=chunk, use_cache=False)
        total_nll += out.loss.item() * (seq_len - 1)
        total_tokens += seq_len - 1
    return float(np.exp(total_nll / total_tokens))


def main():
    print(f"{'='*60}")
    print(f"Llama CWF Cross-Verification")
    print(f"{'='*60}")

    # Load model
    tok_kw = {"trust_remote_code": True}
    mdl_kw = {"torch_dtype": DTYPE, "trust_remote_code": True}
    if HF_TOKEN:
        tok_kw["token"] = HF_TOKEN
        mdl_kw["token"] = HF_TOKEN

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **mdl_kw).to(DEVICE).eval()

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_heads = cfg.num_attention_heads
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // n_heads
    n_q_per_kv = n_heads // n_kv
    print(f"  n_layers={n_layers}, n_kv={n_kv}, n_heads={n_heads}, d_head={d_head}")

    # Load data
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join([t for t in ds["text"] if t.strip()])
    eval_ids = tokenizer.encode(eval_text, return_tensors="pt", truncation=False)

    calib_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_text = "\n\n".join([t for t in calib_ds["text"] if t.strip()])
    calib_all = tokenizer.encode(calib_text, return_tensors="pt", truncation=False)
    calib_ids = calib_all[:, :N_CALIB_TOKENS].to(DEVICE)

    # ── Phase 0: FP16 baseline ──
    fp16_path = OUT_DIR / f"{SHORT}_fp16.json"
    if fp16_path.exists():
        ppl_fp16 = json.loads(fp16_path.read_text())["ppl"]
        print(f"FP16 baseline: {ppl_fp16} (cached)")
    else:
        print("Measuring FP16 baseline...")
        ppl_fp16 = eval_ppl(model, eval_ids)
        fp16_path.write_text(json.dumps({"ppl": ppl_fp16, "model": MODEL_NAME}))
        print(f"  FP16 PPL = {ppl_fp16:.4f}")

    # ── Phase 1: Per-layer sensitivity (Exp4-style) ──
    sens_path = OUT_DIR / f"{SHORT}_per_layer_sensitivity.json"
    if sens_path.exists():
        sens_data = json.loads(sens_path.read_text())
        delta_ppl = {int(k): v for k, v in sens_data["delta_ppl"].items()}
        print(f"Per-layer sensitivity: loaded from cache ({len(delta_ppl)} layers)")
    else:
        print("\nPhase 1: Per-layer sensitivity measurement...")
        delta_ppl = {}

        # Calibrate all layers with L² Lloyd 2-bit
        print("  Calibrating all layers...")
        k_data = {}
        hooks = []
        def make_calib_hook(li):
            def fn(mod, inp, out):
                k_data[li] = out.detach().cpu().float().numpy()
            return fn
        for li in range(n_layers):
            hooks.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(make_calib_hook(li)))
        with torch.no_grad():
            model(calib_ids, use_cache=False)
        for h in hooks:
            h.remove()

        # Fit Lloyd per layer per head
        lloyd_fits = {}
        for li in range(n_layers):
            k_np = k_data[li].reshape(-1, n_kv, d_head)
            for hk in range(n_kv):
                lloyd_fits[(li, hk)] = fit_pca_l2_lloyd(k_np[:, hk, :], bits=2)

        # Measure per-layer ΔPPL
        for li in range(n_layers):
            head_qs = {hk: lloyd_fits[(li, hk)] for hk in range(n_kv)}
            hook_obj = PCAL2LloydHook(head_qs, n_kv, d_head)
            handle = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook_obj)
            ppl_li = eval_ppl(model, eval_ids)
            handle.remove()
            delta_ppl[li] = round(ppl_li - ppl_fp16, 6)
            print(f"  Layer {li:2d}: ΔPPL = {delta_ppl[li]:+.4f} (PPL={ppl_li:.4f})")

        sens_data = {"model": MODEL_NAME, "ppl_fp16": ppl_fp16, "delta_ppl": delta_ppl}
        sens_path.write_text(json.dumps(sens_data, indent=2))
        print(f"  Saved to {sens_path}")

    # ── Phase 2: CWF sweep ──
    print(f"\nPhase 2: CWF sweep ({CWF_BUDGETS})...")

    # Collect K and Fisher trace for CWF importance
    print("  Collecting K + Q + attention for Fisher trace...")
    captured = {}
    hooks = []
    def kh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['k'] = o.detach().cpu().float().numpy()
        return h
    def qh(li):
        def h(m, i, o):
            captured.setdefault(li, {})['q'] = o.detach().cpu().float().numpy()
        return h
    def ah(li):
        def h(m, i, o):
            if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                captured.setdefault(li, {})['attn'] = o[1].detach().cpu().float().numpy()
        return h
    for li in range(n_layers):
        mod = model.model.layers[li].self_attn
        hooks.append(mod.k_proj.register_forward_hook(kh(li)))
        hooks.append(mod.q_proj.register_forward_hook(qh(li)))
        hooks.append(mod.register_forward_hook(ah(li)))
    with torch.no_grad():
        model(calib_ids, output_attentions=True, use_cache=False)
    for h in hooks:
        h.remove()

    # Compute Fisher trace per (layer, kv_head)
    T = calib_ids.shape[1]
    trace_table = np.zeros((n_layers, n_kv), dtype=np.float32)
    K_table = {}
    for li in range(n_layers):
        k_np = captured[li]['k'].reshape(-1, n_kv, d_head)
        q_np = captured[li]['q'].reshape(-1, n_heads, d_head)
        attn_np = captured[li].get('attn')
        for hk in range(n_kv):
            K_table[(li, hk)] = k_np[:, hk, :]
            q_group = q_np[:, hk*n_q_per_kv:(hk+1)*n_q_per_kv, :]
            q_avg = q_group.mean(axis=1)
            if attn_np is not None and attn_np.ndim == 4:
                s = attn_np[0, hk*n_q_per_kv:(hk+1)*n_q_per_kv, :, :].mean(axis=0)
                s_diag = np.diag(s.mean(axis=0))
            else:
                s_diag = np.eye(T) / T
            tr_M = np.sum(q_avg ** 2) / T
            trace_table[li, hk] = max(tr_M, 1e-10)

    # CWF importance = sensitivity[l] × trace(M[l,h])
    importance = np.zeros(n_layers * n_kv, dtype=np.float64)
    for li in range(n_layers):
        sens = max(delta_ppl.get(li, 0.001), 0.001)
        for hk in range(n_kv):
            importance[li * n_kv + hk] = sens * trace_table[li, hk]

    results = {"model": MODEL_NAME, "ppl_fp16": ppl_fp16, "configs": {}}

    for avg_bits in CWF_BUDGETS:
        config_name = f"cwf_avg{avg_bits}"
        cwf_path = OUT_DIR / f"{SHORT}_{config_name}.json"
        if cwf_path.exists():
            cached = json.loads(cwf_path.read_text())
            print(f"  {config_name}: PPL={cached['ppl']} (cached)")
            results["configs"][config_name] = cached
            continue

        total_budget = int(round(avg_bits * n_layers * n_kv * d_head))
        head_budget = int(round(avg_bits * d_head))

        # Global WF across (layer, head) pairs
        bits_per_head = water_filling_global(importance, int(round(avg_bits * n_layers * n_kv)),
                                              b_floor=B_FLOOR, b_max=B_MAX)
        actual_avg = bits_per_head.sum() / len(bits_per_head)

        # Fit Lloyd per head at allocated bits
        print(f"  {config_name} (actual avg={actual_avg:.3f})... ", end="", flush=True)
        t0 = time.time()

        head_quantizers = {}
        for li in range(n_layers):
            for hk in range(n_kv):
                idx = li * n_kv + hk
                b = int(bits_per_head[idx])
                head_quantizers[(li, hk)] = fit_pca_l2_lloyd(K_table[(li, hk)], b)

        # Install hooks
        hook_handles = []
        for li in range(n_layers):
            hq = {hk: head_quantizers[(li, hk)] for hk in range(n_kv)}
            hook_obj = PCAL2LloydHook(hq, n_kv, d_head)
            hook_handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook_obj))

        ppl = eval_ppl(model, eval_ids)
        for h in hook_handles:
            h.remove()
        elapsed = time.time() - t0

        result = {
            "ppl": round(ppl, 4),
            "avg_bits_target": avg_bits,
            "avg_bits_actual": round(actual_avg, 4),
            "runtime_sec": round(elapsed, 1),
        }
        cwf_path.write_text(json.dumps(result, indent=2))
        results["configs"][config_name] = result
        print(f"PPL={ppl:.4f} (avg_bits={actual_avg:.3f}, {elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {MODEL_NAME}")
    print(f"{'='*60}")
    print(f"FP16: {ppl_fp16:.4f}")
    for cfg_name, cfg_data in sorted(results["configs"].items()):
        print(f"  {cfg_name}: PPL={cfg_data['ppl']:.4f}")

    # Save full results
    full_path = OUT_DIR / f"{SHORT}_cwf_full.json"
    full_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {full_path}")


if __name__ == "__main__":
    main()
