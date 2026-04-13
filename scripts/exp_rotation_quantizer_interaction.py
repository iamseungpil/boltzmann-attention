#!/usr/bin/env python3
"""
3×3 Rotation-Quantizer Interaction Ablation.
Tests whether Lloyd's PPL failure is caused by PCA rotation or Lloyd itself.

Matrix:
          | Uniform    | Lloyd      | Clipped-Lloyd |
----------+------------+------------+---------------+
No Rot    | A          | B          | G             |
PCA       | C          | D          | H             |
Random    | E          | F          | I             |

All configs use identical preprocessing: K_mean subtraction + rotation + quantize + inverse.
NoRot = identity rotation (still subtracts K_mean for fair comparison).

Key comparisons:
  B vs A: Lloyd inherently bad? (without rotation)
  D vs B: PCA makes Lloyd worse?
  F vs D: Random rotation fixes Lloyd?
  H vs D: Clipped Lloyd rescues PCA+Lloyd?
  H vs C: Clipped Lloyd beats Uniform?

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp_rotation_quantizer_interaction.py --model mistralai/Mistral-7B-v0.3 --bits 2
"""
import argparse, json, time, gc, os, hashlib
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import numpy as np
import torch
from pathlib import Path
from scipy.stats import special_ortho_group

DTYPE = torch.bfloat16
CALIB_TOKENS = 2048
EVAL_CHUNK = 2048
MAX_EVAL = 50000
CLIP_PERCENTILE = 99.5  # for clipped lloyd


def lloyd_1d(col, bits, n_iter=30):
    n_levels = 2 ** bits
    if n_levels <= 1:
        return np.array([col.mean()], dtype=np.float32)
    pcts = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.sort(np.percentile(col, pcts)).astype(np.float64)
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
    return centroids.astype(np.float32)


def clipped_lloyd_1d(col, bits, clip_pct=CLIP_PERCENTILE, n_iter=30):
    """True clipped Lloyd: clip data to percentile bounds, then fit Lloyd on clipped data."""
    n_levels = 2 ** bits
    clip_lo = np.percentile(col, 100 - clip_pct)
    clip_hi = np.percentile(col, clip_pct)
    col_clipped = np.clip(col, clip_lo, clip_hi)
    if n_levels <= 1:
        return np.array([col_clipped.mean()], dtype=np.float32), clip_lo, clip_hi
    centroids = lloyd_1d(col_clipped, bits, n_iter)
    return centroids, clip_lo, clip_hi


def lloyd_quantize(col, centroids):
    if len(centroids) <= 1:
        return np.full_like(col, centroids[0])
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    idx = np.searchsorted(boundaries, col)
    return centroids[idx]


def uniform_quantize(col, vmin, vmax, bits):
    nl = 2 ** bits
    col_c = np.clip(col, vmin, vmax)
    if vmax - vmin < 1e-10:
        return col_c
    s = (vmax - vmin) / (nl - 1)
    q = np.clip(np.round((col_c - vmin) / s).astype(int), 0, nl - 1)
    return q * s + vmin


class InteractionHook:
    """Unified hook: center → rotate → quantize → inverse rotate → uncenter."""
    def __init__(self, n_kv, d_head, K_means, rotations, quant_fn, bits):
        self.n_kv = n_kv
        self.d_head = d_head
        self.K_means = K_means      # dict h → (d,)
        self.R = rotations          # dict h → (d,d) orthogonal matrix
        self.quant_fn = quant_fn    # dict h → dict j → callable(col) → quantized
        self.bits = bits
        self.mse_accum = []
        self.linf_accum = []
        self.attn_err_accum = []

    def __call__(self, module, inputs, output):
        B, T, _ = output.shape
        x = output.view(B, T, self.n_kv, self.d_head).float().cpu().numpy()
        out = x.copy()
        for h in range(self.n_kv):
            m = self.K_means[h]
            R = self.R[h]
            data = x[:, :, h, :]
            for b in range(B):
                block = data[b]  # (T, d)
                centered = block - m
                rotated = centered @ R
                quantized = np.zeros_like(rotated)
                for j in range(self.d_head):
                    quantized[:, j] = self.quant_fn[h][j](rotated[:, j])
                err = rotated - quantized
                self.mse_accum.append(float(np.mean(err**2)))
                self.linf_accum.append(float(np.max(np.abs(err))))
                reconstructed = quantized @ R.T + m
                out[b, :, h, :] = reconstructed
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
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--output-dir", default="results/interaction")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0"
    short = args.model.split("/")[-1].replace(".", "_")
    bits = args.bits

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"{'='*60}")
    print(f"ROTATION-QUANTIZER INTERACTION: {args.model}, {bits}-bit")
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

    # Data
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

    # Calibration
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

    # Per-head calibration
    calib_info = {}  # (li, hk) → {K_mean, K_centered, PCA_V, eigenvalues, kurtosis}
    for li in range(n_layers):
        k_np = k_data[li].reshape(-1, n_kv, d_head)
        for hk in range(n_kv):
            K = k_np[:, hk, :].astype(np.float32)
            K_mean = K.mean(0)
            K_c = K - K_mean
            cov = (K_c.T @ K_c) / max(K.shape[0]-1, 1) + 1e-6 * np.eye(d_head)
            ev, V = np.linalg.eigh(cov)
            V = V[:, ::-1].astype(np.float32)
            ev = ev[::-1].astype(np.float32)
            # Kurtosis of PCA-rotated dims
            K_pca = K_c @ V
            kurtosis = np.zeros(d_head)
            for j in range(d_head):
                col = K_pca[:, j]
                std = col.std()
                if std > 1e-10:
                    kurtosis[j] = np.mean((col/std)**4) - 3
            calib_info[(li, hk)] = {
                "K_mean": K_mean, "K_c": K_c, "V": V, "ev": ev,
                "kurtosis": kurtosis
            }
    del k_data
    gc.collect()
    print(f"  Calibrated {n_layers}×{n_kv} heads")

    # Print kurtosis summary
    all_kurt = np.concatenate([calib_info[(li,hk)]["kurtosis"] for li in range(n_layers) for hk in range(n_kv)])
    print(f"  Kurtosis: mean={all_kurt.mean():.2f}, max={all_kurt.max():.2f}, >3: {(all_kurt>3).sum()}/{len(all_kurt)}")

    # Build rotation matrices
    def get_rotations(rot_type, seed=42):
        rotations = {}
        for li in range(n_layers):
            for hk in range(n_kv):
                if rot_type == "norot":
                    rotations[(li, hk)] = np.eye(d_head, dtype=np.float32)
                elif rot_type == "pca":
                    rotations[(li, hk)] = calib_info[(li, hk)]["V"]
                elif rot_type == "random":
                    np.random.seed(seed * 10000 + li * 100 + hk)
                    rotations[(li, hk)] = special_ortho_group.rvs(d_head).astype(np.float32)
        return rotations

    # Build quantizer functions
    def get_quant_fns(rot_type, quant_type, rotations):
        fns = {}
        for li in range(n_layers):
            for hk in range(n_kv):
                R = rotations[(li, hk)]
                K_c = calib_info[(li, hk)]["K_c"]
                K_rot = K_c @ R  # rotated calibration data
                head_fns = {}
                for j in range(d_head):
                    col = K_rot[:, j]
                    if quant_type == "uniform":
                        vmin, vmax = col.min(), col.max()
                        head_fns[j] = lambda c, vm=vmin, vx=vmax: uniform_quantize(c, vm, vx, bits)
                    elif quant_type == "lloyd":
                        centroids = lloyd_1d(col, bits)
                        head_fns[j] = lambda c, ct=centroids: lloyd_quantize(c, ct)
                    elif quant_type == "clipped_lloyd":
                        centroids, clo, chi = clipped_lloyd_1d(col, bits)
                        head_fns[j] = lambda c, ct=centroids, lo=clo, hi=chi: lloyd_quantize(np.clip(c, lo, hi), ct)
                fns[(li, hk)] = head_fns
        return fns

    # Configs
    configs = [
        ("A_norot_uniform", "norot", "uniform"),
        ("B_norot_lloyd", "norot", "lloyd"),
        ("C_pca_uniform", "pca", "uniform"),
        ("D_pca_lloyd", "pca", "lloyd"),
        ("E_random_uniform", "random", "uniform"),
        ("F_random_lloyd", "random", "lloyd"),
        ("G_norot_clipped", "norot", "clipped_lloyd"),
        ("H_pca_clipped", "pca", "clipped_lloyd"),
        ("I_random_clipped", "random", "clipped_lloyd"),
    ]

    results = {"model": args.model, "bits": bits, "fp16": round(ppl_fp16, 4), "configs": {}}

    for cfg_name, rot_type, quant_type in configs:
        out_path = out_dir / f"{short}_{bits}b_{cfg_name}.json"
        if out_path.exists():
            cached = json.loads(out_path.read_text())
            print(f"\n  [{cfg_name}] → PPL={cached['ppl']} (cached)")
            results["configs"][cfg_name] = cached
            continue

        # For random, average over seeds
        if rot_type == "random":
            ppls = []
            for seed in args.random_seeds:
                rotations = get_rotations("random", seed=seed)
                quant_fns = get_quant_fns(rot_type, quant_type, rotations)
                hook_handles = []
                for li in range(n_layers):
                    K_means = {hk: calib_info[(li,hk)]["K_mean"] for hk in range(n_kv)}
                    R_dict = {hk: rotations[(li,hk)] for hk in range(n_kv)}
                    Q_dict = {hk: quant_fns[(li,hk)] for hk in range(n_kv)}
                    hook = InteractionHook(n_kv, d_head, K_means, R_dict, Q_dict, bits)
                    hook_handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
                ppl = eval_ppl(model, eval_ids, device)
                for h in hook_handles:
                    h.remove()
                ppls.append(ppl)
                print(f"    seed={seed}: PPL={ppl:.4f}")
            ppl_mean = float(np.mean(ppls))
            ppl_std = float(np.std(ppls))
            result = {"ppl": round(ppl_mean, 4), "ppl_std": round(ppl_std, 4),
                      "delta": round(ppl_mean - ppl_fp16, 4),
                      "seeds": args.random_seeds, "ppls": [round(p,4) for p in ppls]}
        else:
            print(f"\n  [{cfg_name}]...", end=" ", flush=True)
            t0 = time.time()
            rotations = get_rotations(rot_type)
            quant_fns = get_quant_fns(rot_type, quant_type, rotations)
            hook_handles = []
            all_hooks = []
            for li in range(n_layers):
                K_means = {hk: calib_info[(li,hk)]["K_mean"] for hk in range(n_kv)}
                R_dict = {hk: rotations[(li,hk)] for hk in range(n_kv)}
                Q_dict = {hk: quant_fns[(li,hk)] for hk in range(n_kv)}
                hook = InteractionHook(n_kv, d_head, K_means, R_dict, Q_dict, bits)
                all_hooks.append(hook)
                hook_handles.append(model.model.layers[li].self_attn.k_proj.register_forward_hook(hook))
            ppl = eval_ppl(model, eval_ids, device)
            for h in hook_handles:
                h.remove()
            elapsed = time.time() - t0
            mse = float(np.mean([h.mse_accum for h in all_hooks if h.mse_accum])) if all_hooks[0].mse_accum else 0
            linf = float(np.max([max(h.linf_accum) for h in all_hooks if h.linf_accum])) if all_hooks[0].linf_accum else 0
            result = {"ppl": round(ppl, 4), "delta": round(ppl - ppl_fp16, 4),
                      "mse": round(mse, 6), "linf": round(linf, 4),
                      "runtime_sec": round(elapsed, 1)}
            print(f"PPL={ppl:.4f} (Δ={ppl-ppl_fp16:+.4f}, MSE={mse:.4f}, L∞={linf:.2f}, {elapsed:.1f}s)")

        out_path.write_text(json.dumps(result, indent=2))
        results["configs"][cfg_name] = result

    # Summary
    print(f"\n{'='*60}")
    print(f"INTERACTION TABLE: {args.model} {bits}-bit")
    print(f"{'='*60}")
    print(f"FP16: {ppl_fp16:.4f}\n")
    print(f"{'':12s} {'Uniform':>10s} {'Lloyd':>10s} {'ClipLloyd':>10s}")
    for rot, prefix in [("NoRot", "norot"), ("PCA", "pca"), ("Random", "random")]:
        u = results["configs"].get(f"{prefix[0].upper()}_{prefix}_uniform", results["configs"].get(f"A_{prefix}_uniform", results["configs"].get(f"C_{prefix}_uniform", results["configs"].get(f"E_{prefix}_uniform", {}))))
        l = results["configs"].get(f"{prefix[0].upper()}_{prefix}_lloyd", results["configs"].get(f"B_{prefix}_lloyd", results["configs"].get(f"D_{prefix}_lloyd", results["configs"].get(f"F_{prefix}_lloyd", {}))))
        c = results["configs"].get(f"{prefix[0].upper()}_{prefix}_clipped", results["configs"].get(f"G_{prefix}_clipped", results["configs"].get(f"H_{prefix}_clipped", results["configs"].get(f"I_{prefix}_clipped", {}))))
        u_ppl = u.get("ppl", "—") if u else "—"
        l_ppl = l.get("ppl", "—") if l else "—"
        c_ppl = c.get("ppl", "—") if c else "—"
        print(f"{rot:12s} {u_ppl:>10} {l_ppl:>10} {c_ppl:>10}")

    # Key comparisons
    print(f"\nKey comparisons:")
    for name, c1, c2 in [
        ("B vs A (Lloyd inherent?)", "B_norot_lloyd", "A_norot_uniform"),
        ("D vs C (PCA+Lloyd vs PCA+Uni)", "D_pca_lloyd", "C_pca_uniform"),
        ("F vs E (Random+Lloyd vs Random+Uni)", "F_random_lloyd", "E_random_uniform"),
        ("H vs C (ClipLloyd vs Uniform)", "H_pca_clipped", "C_pca_uniform"),
        ("H vs D (ClipLloyd vs Lloyd)", "H_pca_clipped", "D_pca_lloyd"),
        ("D vs B (PCA makes Lloyd worse?)", "D_pca_lloyd", "B_norot_lloyd"),
        ("F vs D (Random fixes Lloyd?)", "F_random_lloyd", "D_pca_lloyd"),
    ]:
        r1 = results["configs"].get(c1, {})
        r2 = results["configs"].get(c2, {})
        if r1 and r2:
            p1, p2 = r1.get("ppl", 0), r2.get("ppl", 0)
            if p2 > 0:
                print(f"  {name}: {p1:.4f} vs {p2:.4f} (ratio={p1/p2:.3f})")

    full_path = out_dir / f"{short}_{bits}b_interaction.json"
    full_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {full_path}")


if __name__ == "__main__":
    main()
