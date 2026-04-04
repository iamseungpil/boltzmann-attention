#!/usr/bin/env python3
"""
NIAH (Needle in a Haystack) verification for Pre-RoPE PCA theorem.
GPU 0: 2-bit experiments, GPU 1: 3/4-bit experiments (parallel)

Passkey retrieval: insert "The passkey is {N}" at various depths,
check if model can retrieve N from the end of context.
"""
import json, time, os, sys, traceback
import numpy as np
from pathlib import Path
from multiprocessing import Process

def run_niah(device, bits_list, tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'#'*70}")
    print(f"# NIAH Experiment: {tag} on {device}, bits={bits_list}")
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
    # Step 1: Calibration — collect pre-RoPE covariance
    # ============================================================
    print("  [Step 1] Calibration...")
    filler = "The grass is green. The sky is blue. The sun is yellow. Today is a beautiful day. "
    cal_text = filler * 50
    cal_ids = tokenizer.encode(cal_text, return_tensors="pt", truncation=False)[:, :2048].to(device)

    pre_rope_covs = {}
    pre_capture = {}
    hooks = []

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
        model(cal_ids, use_cache=False)

    for hook in hooks:
        hook.remove()

    # PCA bases
    pca_pre = {}
    for (l, h), cov in pre_capture.items():
        _, V = np.linalg.eigh(cov)
        pca_pre[(l, h)] = V

    np.random.seed(42)
    R_random = np.linalg.qr(np.random.randn(d_head, d_head))[0]

    print(f"  Calibration done: {len(pca_pre)} heads")

    # ============================================================
    # Step 2: NIAH passkey retrieval
    # ============================================================
    context_lengths = [4096]
    depths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_trials = 5  # per depth
    methods = ['fp16', 'identity', 'random_rot', 'pre_rope_pca']

    def make_k_quant_hook(layer_idx, bits, rotation_dict=None, random_rot=None):
        n_levels = 2 ** bits
        def hook_fn(module, input, output):
            k = output[0] if isinstance(output, tuple) else output
            k_np = k.detach().cpu().float().numpy()
            k_reshaped = k_np.reshape(-1, n_kv, d_head)

            for h in range(n_kv):
                K_h = k_reshaped[:, h, :]

                if rotation_dict is not None and (layer_idx, h) in rotation_dict:
                    R = rotation_dict[(layer_idx, h)]
                    K_rot = K_h @ R
                elif random_rot is not None:
                    K_rot = K_h @ random_rot
                else:
                    K_rot = K_h.copy()

                for j in range(d_head):
                    col = K_rot[:, j]
                    vmin, vmax = col.min(), col.max()
                    if vmax - vmin < 1e-10:
                        continue
                    scale = (vmax - vmin) / (n_levels - 1)
                    q = np.round((col - vmin) / scale).astype(int)
                    q = np.clip(q, 0, n_levels - 1)
                    K_rot[:, j] = q * scale + vmin

                if rotation_dict is not None and (layer_idx, h) in rotation_dict:
                    k_reshaped[:, h, :] = K_rot @ R.T
                elif random_rot is not None:
                    k_reshaped[:, h, :] = K_rot @ random_rot.T
                else:
                    k_reshaped[:, h, :] = K_rot

            return torch.tensor(k_reshaped.reshape(-1, n_kv * d_head),
                              dtype=k.dtype, device=k.device).unsqueeze(0)
        return hook_fn

    def create_passkey_prompt(context_len, depth, passkey):
        """Create a prompt with passkey hidden at given depth."""
        needle = f"The special passkey is {passkey}. Remember this number."
        question = f"\nBased on the text above, what is the special passkey? The passkey is"

        needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
        question_tokens = tokenizer.encode(question, add_special_tokens=False)

        filler_text = "The grass is green. The sky is blue. The sun shines bright. "
        filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)

        # Calculate sizes
        available = context_len - len(needle_tokens) - len(question_tokens) - 2
        if available < 10:
            return None

        # Repeat filler to fill context
        full_filler = []
        while len(full_filler) < available:
            full_filler.extend(filler_tokens)
        full_filler = full_filler[:available]

        # Insert needle at depth
        insert_pos = int(len(full_filler) * depth)
        insert_pos = max(0, min(insert_pos, len(full_filler)))

        all_tokens = full_filler[:insert_pos] + needle_tokens + full_filler[insert_pos:] + question_tokens
        return all_tokens[:context_len]

    results = {}

    for ctx_len in context_lengths:
        for bits in bits_list:
            for method in methods:
                key = f"{method}_{bits}bit_{ctx_len}"
                if method == 'fp16':
                    key = f"fp16_{ctx_len}"
                    if key in results:
                        continue

                print(f"\n  [{method} {bits}bit ctx={ctx_len}]")

                # Install hooks
                hook_handles = []
                if method != 'fp16':
                    for l in range(n_layers):
                        k_proj = model.model.layers[l].self_attn.k_proj
                        if method == 'identity':
                            hk = k_proj.register_forward_hook(
                                make_k_quant_hook(l, bits))
                        elif method == 'random_rot':
                            hk = k_proj.register_forward_hook(
                                make_k_quant_hook(l, bits, random_rot=R_random))
                        elif method == 'pre_rope_pca':
                            hk = k_proj.register_forward_hook(
                                make_k_quant_hook(l, bits, rotation_dict=pca_pre))
                        hook_handles.append(hk)

                depth_results = {}
                for depth in depths:
                    correct = 0
                    total = 0
                    for trial in range(n_trials):
                        passkey = str(np.random.randint(1000, 9999))
                        tokens = create_passkey_prompt(ctx_len, depth, passkey)
                        if tokens is None:
                            continue

                        input_ids = torch.tensor([tokens], device=device)

                        with torch.no_grad():
                            outputs = model(input_ids, use_cache=False)
                            logits = outputs.logits[0, -1, :]  # last token logits

                            # Check if top prediction starts with passkey
                            top_token = tokenizer.decode(logits.argmax().item()).strip()

                            # Also check if passkey appears in top-5
                            top5 = torch.topk(logits, 5).indices
                            top5_text = [tokenizer.decode(t.item()).strip() for t in top5]

                            # Success: passkey's first digit matches top prediction
                            if top_token.startswith(passkey[0]) or passkey in ' '.join(top5_text):
                                correct += 1
                            total += 1

                    acc = correct / total if total > 0 else 0
                    depth_results[depth] = {'accuracy': acc, 'correct': correct, 'total': total}
                    print(f"    depth={depth:.1f}: {correct}/{total} = {acc:.1%}")

                # Remove hooks
                for hk in hook_handles:
                    hk.remove()

                avg_acc = np.mean([d['accuracy'] for d in depth_results.values()])
                results[key] = {
                    'depths': depth_results,
                    'avg_accuracy': float(avg_acc),
                    'method': method,
                    'bits': bits,
                    'context_len': ctx_len
                }
                print(f"    Average accuracy: {avg_acc:.1%}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"NIAH Summary ({tag})")
    print(f"{'='*70}")

    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len}")
        fp16_key = f"fp16_{ctx_len}"

        if fp16_key in results:
            fp16_acc = results[fp16_key]['avg_accuracy']
            print(f"  FP16 avg accuracy: {fp16_acc:.1%}")

        for bits in bits_list:
            print(f"\n  --- {bits}-bit ---")
            print(f"  {'Method':20s} {'Avg Acc':>8s}  Depth: {' '.join(f'{d:.1f}' for d in depths)}")

            for method in methods:
                key = f"{method}_{bits}bit_{ctx_len}" if method != 'fp16' else fp16_key
                if key not in results:
                    continue
                r = results[key]
                avg = r['avg_accuracy']
                depth_accs = ' '.join(f"{r['depths'].get(d, {}).get('accuracy', 0):.0%}" for d in depths)
                print(f"  {method:20s} {avg:>7.1%}  {depth_accs}")

    # Ranking check: MSE order = NIAH order?
    print(f"\n  MSE ranking predicts NIAH ranking?")
    for bits in bits_list:
        for ctx_len in context_lengths:
            accs = {}
            for method in ['identity', 'random_rot', 'pre_rope_pca']:
                key = f"{method}_{bits}bit_{ctx_len}"
                if key in results:
                    accs[method] = results[key]['avg_accuracy']

            if len(accs) >= 3:
                mse_order = ['pre_rope_pca', 'random_rot', 'identity']  # predicted: best to worst MSE
                niah_order = sorted(accs, key=lambda m: accs[m], reverse=True)
                match = mse_order == niah_order
                print(f"    {bits}-bit ctx={ctx_len}: MSE pred={mse_order}, NIAH={niah_order} -> {'MATCH' if match else 'MISMATCH'}")

    # Save
    outdir = Path(__file__).parent / "verification_results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / f"niah_results_{tag}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {outdir / f'niah_results_{tag}.json'}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    t0 = time.time()

    # GPU 0: 2-bit (hardest, most differentiation)
    # GPU 1: 3,4-bit
    p0 = Process(target=run_niah, args=('cuda:0', [2], 'gpu0_2bit'))
    p1 = Process(target=run_niah, args=('cuda:1', [3, 4], 'gpu1_34bit'))

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print(f"\n{'='*70}")
    print(f"ALL NIAH DONE in {time.time()-t0:.1f}s")
    print(f"{'='*70}")
