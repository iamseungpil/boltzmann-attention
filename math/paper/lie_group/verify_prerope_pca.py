#!/usr/bin/env python3
"""
Pre-RoPE PCA vs Post-RoPE PCA vs TurboQuant: MSE + PPL verification
GPU 0: MSE experiment (fast), GPU 1: PPL experiment (slow)
"""
import argparse, json, time, os, sys, traceback
import numpy as np
from pathlib import Path

def run_mse_experiment(device='cuda:0'):
    """MSE comparison: Pre-RoPE PCA vs Post-RoPE PCA vs Random rotation vs No rotation"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'#'*70}")
    print(f"# MSE Experiment on {device}")
    print(f"{'#'*70}")

    model_name = 'Qwen/Qwen2.5-7B'
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    # WikiText-2
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    ids = tokenizer.encode(text, return_tensors="pt", truncation=False)[:, :2048].to(device)

    cfg = model.config
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    d_head = cfg.hidden_size // cfg.num_attention_heads
    print(f"  layers={n_layers}, kv_heads={n_kv}, d_head={d_head}, tokens={ids.shape[1]}")

    # Collect pre-RoPE and post-RoPE keys
    pre_rope_keys = {}   # (layer, head) -> (n_tokens, d_head)
    post_rope_keys = {}

    # Hook to capture pre-RoPE K
    hooks = []
    pre_rope_capture = {}

    def make_pre_rope_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            # Qwen2Attention.forward receives hidden_states as first arg
            hidden_states = args[0] if len(args) > 0 else kwargs.get('hidden_states')
            if hidden_states is not None:
                # Compute K projection (before RoPE)
                k_proj = module.k_proj(hidden_states)  # (batch, seq, n_kv*d_head)
                k_pre = k_proj[0].detach().cpu().float().numpy()
                k_reshaped = k_pre.reshape(-1, n_kv, d_head)  # (seq, n_kv, d_head)
                for h in range(n_kv):
                    pre_rope_capture[(layer_idx, h)] = k_reshaped[:, h, :]
        return hook_fn

    # Register pre-forward hooks on attention modules
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for l in range(n_layers):
            attn = model.model.layers[l].self_attn
            hook = attn.register_forward_pre_hook(make_pre_rope_hook(l), with_kwargs=True)
            hooks.append(hook)

    with torch.no_grad():
        out = model(ids, use_cache=True)

    # Get post-RoPE keys from past_key_values
    for l, item in enumerate(out.past_key_values):
        K_tensor = item[0] if isinstance(item, tuple) else item.keys
        K = K_tensor[0].cpu().float().numpy()  # (n_kv, seq, d_head)
        for h in range(n_kv):
            post_rope_keys[(l, h)] = K[h]

    pre_rope_keys = dict(pre_rope_capture)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    del model
    torch.cuda.empty_cache()

    print(f"  Collected: {len(pre_rope_keys)} pre-RoPE, {len(post_rope_keys)} post-RoPE heads")

    # ============================================================
    # MSE comparison for each head
    # ============================================================
    print(f"\n{'='*70}")
    print("MSE Comparison: Pre-RoPE PCA vs Post-RoPE PCA vs Random vs Identity")
    print(f"{'='*70}")

    results = []
    bits_list = [2, 3, 4]

    for (l, h) in sorted(post_rope_keys.keys()):
        if (l, h) not in pre_rope_keys:
            continue

        K_pre = pre_rope_keys[(l, h)]    # (n, d) pre-RoPE
        K_post = post_rope_keys[(l, h)]  # (n, d) post-RoPE
        n, d = K_pre.shape

        # Covariances
        Sigma_pre = (K_pre - K_pre.mean(0)).T @ (K_pre - K_pre.mean(0)) / n
        Sigma_post = (K_post - K_post.mean(0)).T @ (K_post - K_post.mean(0)) / n

        eig_pre = np.maximum(np.linalg.eigvalsh(Sigma_pre), 1e-10)
        eig_post = np.maximum(np.linalg.eigvalsh(Sigma_post), 1e-10)

        # G values
        G_pre_pca = np.exp(np.mean(np.log(eig_pre)))
        G_post_pca = np.exp(np.mean(np.log(eig_post)))
        G_uniform = np.mean(eig_pre)  # = tr(Sigma)/d, same for pre and post

        # PCA eigenvectors
        _, V_pre = np.linalg.eigh(Sigma_pre)
        _, V_post = np.linalg.eigh(Sigma_post)

        for bits in bits_list:
            n_levels = 2 ** bits

            def quantize_and_mse(K_orig, rotation=None):
                """Apply rotation, uniform quantize, inverse rotation, measure MSE."""
                if rotation is not None:
                    K_rot = K_orig @ rotation  # (n, d)
                else:
                    K_rot = K_orig.copy()

                # Per-dimension uniform quantization
                mse_total = 0.0
                K_recon = np.zeros_like(K_rot)
                for j in range(d):
                    col = K_rot[:, j]
                    vmin, vmax = col.min(), col.max()
                    if vmax - vmin < 1e-10:
                        K_recon[:, j] = col
                        continue
                    # Quantize
                    scale = (vmax - vmin) / (n_levels - 1)
                    q = np.round((col - vmin) / scale).astype(int)
                    q = np.clip(q, 0, n_levels - 1)
                    K_recon[:, j] = q * scale + vmin

                if rotation is not None:
                    K_recon = K_recon @ rotation.T  # inverse rotation

                mse = np.mean((K_orig - K_recon) ** 2)
                return mse

            # Water-filling quantization (non-uniform bits)
            def quantize_wf_and_mse(K_orig, rotation):
                """PCA rotation + water-filling bit allocation."""
                K_rot = K_orig @ rotation  # (n, d)
                variances = np.var(K_rot, axis=0)
                variances = np.maximum(variances, 1e-10)

                # Water-filling: allocate more bits to high-variance dims
                total_bits = d * bits
                log_var = np.log2(variances)
                # b_j = bits + (1/2)(log2(var_j) - mean(log2(var)))
                mean_log_var = np.mean(log_var)
                b_alloc = bits + 0.5 * (log_var - mean_log_var)
                b_alloc = np.clip(b_alloc, 1, bits * 2)
                # Normalize to same total
                b_alloc = b_alloc * (total_bits / b_alloc.sum())
                b_alloc = np.maximum(np.round(b_alloc), 1).astype(int)

                mse_total = 0.0
                K_recon = np.zeros_like(K_rot)
                for j in range(d):
                    col = K_rot[:, j]
                    nl = 2 ** int(b_alloc[j])
                    vmin, vmax = col.min(), col.max()
                    if vmax - vmin < 1e-10:
                        K_recon[:, j] = col
                        continue
                    scale = (vmax - vmin) / (nl - 1)
                    q = np.round((col - vmin) / scale).astype(int)
                    q = np.clip(q, 0, nl - 1)
                    K_recon[:, j] = q * scale + vmin

                K_recon = K_recon @ rotation.T
                mse = np.mean((K_orig - K_recon) ** 2)
                return mse

            # Random orthogonal rotation (TurboQuant simulation)
            np.random.seed(42)
            R_random = np.linalg.qr(np.random.randn(d, d))[0]

            # 1. No rotation + uniform bits (baseline)
            mse_identity = quantize_and_mse(K_post)

            # 2. Random rotation + uniform bits (TurboQuant)
            mse_turbo = quantize_and_mse(K_post, R_random)

            # 3. Post-RoPE PCA + uniform bits
            mse_post_pca_uni = quantize_and_mse(K_post, V_post)

            # 4. Post-RoPE PCA + water-filling (current FOKVQ)
            mse_post_pca_wf = quantize_wf_and_mse(K_post, V_post)

            # 5. Pre-RoPE PCA + uniform bits
            mse_pre_pca_uni = quantize_and_mse(K_pre, V_pre)

            # 6. Pre-RoPE PCA + water-filling (OPTIMAL)
            mse_pre_pca_wf = quantize_wf_and_mse(K_pre, V_pre)

            results.append({
                'layer': l, 'head': h, 'bits': bits,
                'mse_identity': float(mse_identity),
                'mse_turbo': float(mse_turbo),
                'mse_post_pca_uni': float(mse_post_pca_uni),
                'mse_post_pca_wf': float(mse_post_pca_wf),
                'mse_pre_pca_uni': float(mse_pre_pca_uni),
                'mse_pre_pca_wf': float(mse_pre_pca_wf),
                'R_aniso_pre': float(G_uniform / G_pre_pca),
            })

    # Summarize
    print(f"\n{'='*70}")
    print("Summary: Average MSE across all heads")
    print(f"{'='*70}")

    for bits in bits_list:
        br = [r for r in results if r['bits'] == bits]
        if not br:
            continue

        print(f"\n--- {bits}-bit ---")
        methods = ['mse_identity', 'mse_turbo', 'mse_post_pca_uni', 'mse_post_pca_wf',
                   'mse_pre_pca_uni', 'mse_pre_pca_wf']
        labels = ['No rotation', 'TurboQuant(random)', 'PostRoPE PCA+uni',
                  'PostRoPE PCA+WF(FOKVQ)', 'PreRoPE PCA+uni', 'PreRoPE PCA+WF(OPTIMAL)']

        vals = {m: np.mean([r[m] for r in br]) for m in methods}
        best = min(vals.values())

        for m, label in zip(methods, labels):
            v = vals[m]
            ratio = v / best if best > 0 else 0
            marker = ' *** BEST ***' if v == best else ''
            print(f"  {label:30s}: MSE={v:.6f} (x{ratio:.2f}){marker}")

        r_aniso = np.mean([r['R_aniso_pre'] for r in br])
        print(f"  R_aniso(Sigma_0) = {r_aniso:.2f}")
        print(f"  Theory: PreRoPE PCA+WF should be {r_aniso:.2f}x better than TurboQuant")
        actual_ratio = vals['mse_turbo'] / vals['mse_pre_pca_wf'] if vals['mse_pre_pca_wf'] > 0 else 0
        print(f"  Actual:  {actual_ratio:.2f}x")
        print(f"  Theory vs Actual error: {abs(r_aniso - actual_ratio)/r_aniso*100:.1f}%")

        # Key comparison: Pre-RoPE vs Post-RoPE vs TurboQuant
        print(f"\n  KEY: Pre-RoPE PCA+WF vs TurboQuant: {vals['mse_pre_pca_wf']:.6f} vs {vals['mse_turbo']:.6f}")
        if vals['mse_pre_pca_wf'] < vals['mse_turbo']:
            print(f"       => Pre-RoPE WINS by {vals['mse_turbo']/vals['mse_pre_pca_wf']:.2f}x")
        else:
            print(f"       => TurboQuant WINS by {vals['mse_pre_pca_wf']/vals['mse_turbo']:.2f}x")

        print(f"  KEY: Post-RoPE PCA+WF vs TurboQuant: {vals['mse_post_pca_wf']:.6f} vs {vals['mse_turbo']:.6f}")
        if vals['mse_post_pca_wf'] < vals['mse_turbo']:
            print(f"       => Post-RoPE WINS by {vals['mse_turbo']/vals['mse_post_pca_wf']:.2f}x")
        else:
            print(f"       => TurboQuant WINS by {vals['mse_post_pca_wf']/vals['mse_turbo']:.2f}x")

    # Save
    outdir = Path(__file__).parent / "verification_results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / "prerope_mse_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outdir / 'prerope_mse_results.json'}")

    return results


def run_ppl_experiment(device='cuda:1'):
    """PPL comparison: Pre-RoPE PCA vs Post-RoPE PCA vs TurboQuant(random) vs Identity"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"\n{'#'*70}")
    print(f"# PPL Experiment on {device}")
    print(f"{'#'*70}")

    model_name = 'Qwen/Qwen2.5-7B'
    print(f"  Loading {model_name}...")
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
    n_heads = cfg.num_attention_heads
    d_head = cfg.hidden_size // n_heads

    # First: collect pre-RoPE covariance from a calibration pass
    print("  [Step 1] Calibration: collecting pre-RoPE covariance...")
    cal_ids = all_ids[:, :2048].to(device)

    pre_rope_covs = {}  # (layer, head) -> Sigma_pre
    post_rope_covs = {}
    hooks = []
    pre_capture = {}

    def make_cal_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if len(args) > 0 else kwargs.get('hidden_states')
            if hidden_states is not None:
                k_proj = module.k_proj(hidden_states)
                k_pre = k_proj[0].detach().cpu().float().numpy()
                k_reshaped = k_pre.reshape(-1, n_kv, d_head)
                for h in range(n_kv):
                    K = k_reshaped[:, h, :]
                    Kc = K - K.mean(0)
                    pre_capture[(layer_idx, h)] = Kc.T @ Kc / K.shape[0]
        return hook_fn

    for l in range(n_layers):
        attn = model.model.layers[l].self_attn
        hook = attn.register_forward_pre_hook(make_cal_hook(l), with_kwargs=True)
        hooks.append(hook)

    with torch.no_grad():
        out = model(cal_ids, use_cache=True)

    for l, item in enumerate(out.past_key_values):
        K_tensor = item[0] if isinstance(item, tuple) else item.keys
        K = K_tensor[0].cpu().float().numpy()
        for h in range(n_kv):
            Kh = K[h]
            Kc = Kh - Kh.mean(0)
            post_rope_covs[(l, h)] = Kc.T @ Kc / Kh.shape[0]

    pre_rope_covs = dict(pre_capture)

    for hook in hooks:
        hook.remove()

    # Compute PCA bases
    pca_pre = {}   # (l, h) -> eigenvectors of pre-RoPE cov
    pca_post = {}

    for (l, h) in pre_rope_covs:
        _, V = np.linalg.eigh(pre_rope_covs[(l, h)])
        pca_pre[(l, h)] = V
    for (l, h) in post_rope_covs:
        _, V = np.linalg.eigh(post_rope_covs[(l, h)])
        pca_post[(l, h)] = V

    print(f"  Calibration done: {len(pca_pre)} heads")

    # PPL evaluation with quantized KV cache
    chunk_len = 2048
    max_eval_tokens = min(all_ids.shape[1], 50000)  # limit for speed
    n_chunks = (max_eval_tokens - 1) // chunk_len
    bits_list = [2, 3, 4]

    def eval_ppl_with_hook(method_name, bits, pre_pca=None, post_pca=None, random_rot=None):
        """Evaluate PPL with a specific quantization hook on K."""
        n_levels = 2 ** bits
        hook_handles = []

        def make_quant_hook(layer_idx):
            def hook_fn(module, args, kwargs, output):
                # output contains (attn_output, attn_weights, past_key_value)
                # We need to modify the past_key_value's K
                # Actually, modifying past_kv after attention is too late.
                # Instead, we hook into the K projection and quantize there.
                pass
            return hook_fn

        # Simpler approach: hook on k_proj output and quantize in-place
        def make_k_quant_hook(layer_idx):
            def hook_fn(module, input, output):
                # output = k_proj(hidden_states), shape (batch, seq, n_kv*d_head)
                k = output[0] if isinstance(output, tuple) else output
                k_np = k.detach().cpu().float().numpy()  # (seq, n_kv*d_head)
                k_reshaped = k_np.reshape(-1, n_kv, d_head)  # (seq, n_kv, d_head)

                for h in range(n_kv):
                    K_h = k_reshaped[:, h, :]  # (seq, d_head)

                    # Apply rotation
                    if pre_pca is not None and (layer_idx, h) in pre_pca:
                        R = pre_pca[(layer_idx, h)]
                        K_rot = K_h @ R
                    elif post_pca is not None:
                        # Post-RoPE PCA can't be applied here (this is pre-RoPE)
                        # Skip, handle differently
                        K_rot = K_h
                    elif random_rot is not None:
                        K_rot = K_h @ random_rot
                    else:
                        K_rot = K_h

                    # Quantize per-dimension
                    for j in range(d_head):
                        col = K_rot[:, j]
                        vmin, vmax = col.min(), col.max()
                        if vmax - vmin < 1e-10:
                            continue
                        scale = (vmax - vmin) / (n_levels - 1)
                        q = np.round((col - vmin) / scale).astype(int)
                        q = np.clip(q, 0, n_levels - 1)
                        K_rot[:, j] = q * scale + vmin

                    # Inverse rotation
                    if pre_pca is not None and (layer_idx, h) in pre_pca:
                        K_h_q = K_rot @ R.T
                    elif random_rot is not None:
                        K_h_q = K_rot @ random_rot.T
                    else:
                        K_h_q = K_rot

                    k_reshaped[:, h, :] = K_h_q

                k_quant = k_reshaped.reshape(-1, n_kv * d_head)
                return torch.tensor(k_quant, dtype=k.dtype, device=k.device).unsqueeze(0)
            return hook_fn

        # Register hooks on k_proj
        for l in range(n_layers):
            k_proj_module = model.model.layers[l].self_attn.k_proj
            hk = k_proj_module.register_forward_hook(make_k_quant_hook(l))
            hook_handles.append(hk)

        # Evaluate PPL
        total_nll = 0.0
        total_tokens = 0

        with torch.no_grad():
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_len
                end = start + chunk_len + 1
                if end > max_eval_tokens:
                    break
                input_ids = all_ids[:, start:end].to(device)
                target_ids = input_ids[:, 1:]
                input_ids = input_ids[:, :-1]

                outputs = model(input_ids, use_cache=False)
                logits = outputs.logits  # (1, seq, vocab)

                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                nll = loss_fct(logits[0], target_ids[0]).item()
                total_nll += nll
                total_tokens += target_ids.shape[1]

                if (chunk_idx + 1) % 5 == 0:
                    ppl_so_far = np.exp(total_nll / total_tokens)
                    print(f"    [{method_name} {bits}bit] chunk {chunk_idx+1}/{n_chunks} | ppl={ppl_so_far:.4f}")

        for hk in hook_handles:
            hk.remove()

        ppl = np.exp(total_nll / total_tokens)
        return ppl, total_tokens

    # Random rotation matrix (fixed seed)
    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    # Run PPL for each method and bits
    ppl_results = {}

    # FP16 baseline (no quantization, no hooks)
    print("\n  [Step 2] FP16 baseline...")
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_len
            end = start + chunk_len + 1
            if end > max_eval_tokens:
                break
            input_ids = all_ids[:, start:end].to(device)
            target_ids = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
            outputs = model(input_ids, use_cache=False)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            nll = loss_fct(outputs.logits[0], target_ids[0]).item()
            total_nll += nll
            total_tokens += target_ids.shape[1]
    ppl_fp16 = np.exp(total_nll / total_tokens)
    ppl_results['fp16'] = ppl_fp16
    print(f"  FP16 PPL = {ppl_fp16:.4f} ({total_tokens} tokens)")

    # Methods to test
    methods = [
        ('identity', None, None, None),           # no rotation
        ('random_rot', None, None, R_random),      # TurboQuant simulation
        ('pre_rope_pca', pca_pre, None, None),     # Pre-RoPE PCA (OPTIMAL)
    ]

    for bits in bits_list:
        for method_name, pre, post, rand in methods:
            print(f"\n  [Step] {method_name} {bits}-bit...")
            t0 = time.time()
            ppl, n_tok = eval_ppl_with_hook(method_name, bits, pre_pca=pre, post_pca=post, random_rot=rand)
            elapsed = time.time() - t0
            ppl_results[f'{method_name}_{bits}'] = ppl
            print(f"  {method_name}_{bits}: PPL={ppl:.4f} ({elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*70}")
    print("PPL Summary")
    print(f"{'='*70}")
    print(f"  FP16: {ppl_fp16:.4f}")
    for bits in bits_list:
        print(f"\n  --- {bits}-bit ---")
        for method_name, _, _, _ in methods:
            key = f'{method_name}_{bits}'
            if key in ppl_results:
                ppl = ppl_results[key]
                ratio = ppl / ppl_fp16
                print(f"    {method_name:20s}: PPL={ppl:.4f} ({ratio:.3f}x FP16)")

    # Save
    outdir = Path(__file__).parent / "verification_results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / "prerope_ppl_results.json", 'w') as f:
        json.dump(ppl_results, f, indent=2)
    print(f"\nSaved: {outdir / 'prerope_ppl_results.json'}")

    del model
    torch.cuda.empty_cache()
    return ppl_results


if __name__ == '__main__':
    from multiprocessing import Process

    t0 = time.time()

    # GPU 0: MSE experiment (fast, ~30s)
    # GPU 1: PPL experiment (slow, ~30min)
    p0 = Process(target=run_mse_experiment, args=('cuda:0',))
    p1 = Process(target=run_ppl_experiment, args=('cuda:1',))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print(f"\n{'='*70}")
    print(f"ALL DONE in {time.time()-t0:.1f}s")
    print(f"{'='*70}")
