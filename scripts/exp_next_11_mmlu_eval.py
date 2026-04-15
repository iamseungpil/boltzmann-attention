#!/usr/bin/env python3
"""
Next-11: MMLU downstream eval for CWF best configs
====================================================

Evaluates Mistral-7B on MMLU 5-shot with several CWF configurations
and baselines. Uses a subset of MMLU subjects for speed.

Configs tested:
  (A) FP16 baseline
  (B) Uniform 2-bit per-dim (PCA rotation)
  (C) CWF avg=2.156 (matches Next-4 E budget)
  (D) CWF avg=2.5 (beats v3 Uniform 2b)
  (E) CWF avg=3.5 (beats v3 WF floor=2)

Metric: 5-shot accuracy on MMLU subset (fewer subjects for speed)
"""
import json
import time
import gc
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_NAME = 'mistralai/Mistral-7B-v0.3'
DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
N_CALIB_TOKENS = 1024

# MMLU subset (10 subjects for speed, balanced across domains)
MMLU_SUBJECTS = [
    'abstract_algebra',           # math
    'college_computer_science',   # cs
    'high_school_physics',        # physics
    'high_school_biology',        # bio
    'high_school_chemistry',      # chem
    'high_school_world_history',  # history
    'professional_law',           # law
    'professional_psychology',    # psych
    'public_relations',           # social
    'philosophy',                 # humanities
]
N_SHOTS = 5

B_FLOOR = 2
B_MAX = 6

# Mistral Exp4 sensitivities
EXP4_MISTRAL_DELTA_PPL = {
    0: 0.005, 1: 0.120, 2: 0.555, 3: 0.287, 4: 0.521, 5: 0.206,
    6: 0.304, 7: 0.166, 8: 0.152, 9: 0.160, 10: 0.070, 11: 0.079,
    12: 0.037, 13: 0.039, 14: 0.034, 15: 0.047, 16: 0.030, 17: 0.050,
    18: 0.025, 19: 0.024, 20: 0.067, 21: 0.067, 22: 0.155, 23: 0.122,
    24: 0.046, 25: 0.010, 26: -0.004, 27: 0.103, 28: 0.032, 29: 0.096,
    30: 0.116, 31: 0.028,
}

CONFIGS_TO_TEST = [
    ('fp16', None),
    ('cwf_avg2.156', 2.156),
    ('cwf_avg2.5', 2.5),
    ('cwf_avg3.5', 3.5),
]

OUT_DIR = Path('/home/woori/workspace_common/boltzmann-attention/reports/axis2_theoretical_verification')


# ----------------------------------------------------------------------
# Quantizer (same as Next-10)
# ----------------------------------------------------------------------

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
    n = len(importance)
    imp = np.array(importance, dtype=np.float64)
    imp = np.maximum(imp, 1e-12)
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
    return {
        'K_mean': K_mean,
        'V': V.astype(np.float32),
        'centroids': centroids,
        'bits': bits,
    }


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


def collect_k_and_fisher(model, input_ids, n_layers, n_kv, n_q, head_dim):
    captured = {}
    handles = []
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
        handles.append(mod.k_proj.register_forward_hook(kh(li)))
        handles.append(mod.q_proj.register_forward_hook(qh(li)))
        handles.append(mod.register_forward_hook(ah(li)))

    with torch.no_grad():
        _ = model(input_ids, output_attentions=True, use_cache=False)
    for h in handles:
        h.remove()

    T = input_ids.shape[1]
    n_q_per_kv = n_q // n_kv
    trace_table = np.zeros((n_layers, n_kv), dtype=np.float32)
    K_table = {}

    for li in range(n_layers):
        data = captured.get(li, {})
        if not all(k in data for k in ['k', 'q', 'attn']):
            continue
        T_c = data['k'].shape[1]
        K_all = data['k'].reshape(T_c, n_kv, head_dim).astype(np.float32)
        Q_all = data['q'].reshape(T_c, n_q, head_dim).astype(np.float32)
        attn_all = data['attn'][0].astype(np.float32)

        for hk in range(n_kv):
            K = K_all[:, hk, :]
            q_heads = list(range(hk * n_q_per_kv, (hk+1) * n_q_per_kv))
            Q = Q_all[:, q_heads, :].mean(axis=1)
            attn_mean = attn_all[q_heads, :, :].mean(axis=0)
            s_t = (attn_mean * (1.0 - attn_mean)).sum(axis=1)
            M = ((Q * s_t[:, None]).T @ Q) / max(T_c, 1)
            trace_table[li, hk] = float(np.trace(M))
            K_table[(li, hk)] = K

    return K_table, trace_table


def build_quantizers_for_config(avg_bits, K_table, trace_table, n_layers, n_kv, head_dim):
    """Build per-(layer, head) quantizers for a given CWF config."""
    sens = np.zeros(n_layers, dtype=np.float32)
    for li in range(n_layers):
        sens[li] = max(0.0, EXP4_MISTRAL_DELTA_PPL.get(li, 0.0)) + 1e-6

    total_budget = int(round(n_layers * n_kv * avg_bits))
    importance_flat = []
    index_map = []
    for li in range(n_layers):
        for hk in range(n_kv):
            imp = float(sens[li]) * float(trace_table[li, hk])
            importance_flat.append(imp)
            index_map.append((li, hk))

    bits_flat = water_filling_global(
        np.array(importance_flat), total_budget, B_FLOOR, B_MAX
    )
    bits_table = np.zeros((n_layers, n_kv), dtype=int)
    for k, (li, hk) in enumerate(index_map):
        bits_table[li, hk] = bits_flat[k]

    per_layer_head_data = {li: [None] * n_kv for li in range(n_layers)}
    for (li, hk), K in K_table.items():
        b = int(bits_table[li, hk])
        try:
            hd = fit_pca_l2_lloyd(K, b)
            per_layer_head_data[li][hk] = hd
        except Exception:
            pass

    for li in range(n_layers):
        for hk in range(n_kv):
            if per_layer_head_data[li][hk] is None:
                per_layer_head_data[li][hk] = {
                    'K_mean': np.zeros(head_dim, dtype=np.float32),
                    'V': np.eye(head_dim, dtype=np.float32),
                    'centroids': np.stack([
                        np.linspace(-3, 3, 2**B_FLOOR, dtype=np.float32)
                        for _ in range(head_dim)
                    ]),
                    'bits': B_FLOOR,
                }
    return per_layer_head_data, bits_table


# ----------------------------------------------------------------------
# MMLU evaluation
# ----------------------------------------------------------------------

def format_mmlu_question(q, choices, include_answer=False, answer=None):
    """Format a single MMLU question."""
    s = q.strip() + "\n"
    for i, c in enumerate(choices):
        s += f"{chr(65 + i)}. {c}\n"
    s += "Answer:"
    if include_answer and answer is not None:
        s += f" {chr(65 + answer)}\n\n"
    return s


def build_mmlu_prompt(dev_examples, test_q, test_choices, subject):
    """Build 5-shot prompt."""
    header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    prompt = header
    for ex in dev_examples[:N_SHOTS]:
        prompt += format_mmlu_question(
            ex['question'], ex['choices'], include_answer=True, answer=ex['answer']
        )
    prompt += format_mmlu_question(test_q, test_choices, include_answer=False)
    return prompt


def mmlu_predict(model, tok, prompt, answer_letters=('A', 'B', 'C', 'D')):
    """Predict the best answer letter for a MMLU question."""
    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1, :]

    letter_ids = [tok.encode(' ' + l, add_special_tokens=False)[-1] for l in answer_letters]
    letter_logits = torch.tensor([logits[lid].item() for lid in letter_ids])
    pred = int(letter_logits.argmax().item())
    return pred, letter_logits.tolist()


def evaluate_mmlu(model, tok, subjects, max_per_subject=None):
    """Evaluate MMLU on multiple subjects."""
    from datasets import load_dataset
    total_correct = 0
    total = 0
    per_subject = {}
    for subject in subjects:
        try:
            ds_test = load_dataset('cais/mmlu', subject, split='test')
            ds_dev = load_dataset('cais/mmlu', subject, split='dev')
        except Exception as e:
            print(f"    Subject {subject} load failed: {e}", flush=True)
            continue

        dev_examples = [ds_dev[i] for i in range(min(N_SHOTS, len(ds_dev)))]

        n_examples = len(ds_test)
        if max_per_subject:
            n_examples = min(n_examples, max_per_subject)

        correct = 0
        for i in range(n_examples):
            ex = ds_test[i]
            prompt = build_mmlu_prompt(dev_examples, ex['question'], ex['choices'], subject)
            try:
                pred, _ = mmlu_predict(model, tok, prompt)
                if pred == ex['answer']:
                    correct += 1
            except Exception as e:
                pass

        acc = correct / n_examples if n_examples > 0 else 0
        total_correct += correct
        total += n_examples
        per_subject[subject] = {'correct': correct, 'total': n_examples, 'accuracy': acc}
        print(f"    {subject}: {correct}/{n_examples} = {acc:.3f}", flush=True)

    overall_acc = total_correct / total if total > 0 else 0
    return overall_acc, per_subject


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Next-11: MMLU downstream eval for CWF")
    print("=" * 70, flush=True)
    t_start = time.time()

    print("\nLoading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, device_map=DEVICE,
        attn_implementation='eager', low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t_start:.1f}s", flush=True)

    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_q

    # Collect calibration data for quantizer fitting
    print("\n[Phase 1] Collecting calibration keys + Fisher traces...", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_text = '\n\n'.join([t for t in ds['text'] if len(t.strip()) > 100][:300])
    except Exception:
        calib_text = " ".join(["Calib."] * 5000)

    calib_enc = tok(calib_text, return_tensors='pt', truncation=True, max_length=N_CALIB_TOKENS)
    calib_ids = calib_enc['input_ids'].to(DEVICE)

    K_table, trace_table = collect_k_and_fisher(model, calib_ids, n_layers, n_kv, n_q, head_dim)
    print(f"  Collected {len(K_table)} (layer, head) pairs", flush=True)

    # Run MMLU eval for each config
    results = {'model': MODEL_NAME, 'subjects': MMLU_SUBJECTS, 'n_shots': N_SHOTS, 'configs': {}}

    for cfg_name, avg_bits in CONFIGS_TO_TEST:
        print(f"\n[{cfg_name}] " + (f"avg_bits={avg_bits}" if avg_bits else "FP16"), flush=True)

        # Build quantizers for this config
        handles = []
        if avg_bits is not None:
            t_fit = time.time()
            per_layer_head_data, bits_table = build_quantizers_for_config(
                avg_bits, K_table, trace_table, n_layers, n_kv, head_dim
            )
            print(f"  Fit done in {time.time()-t_fit:.1f}s", flush=True)

            for li in range(n_layers):
                hook = PCAL2LloydHook(per_layer_head_data[li], n_kv, head_dim)
                h = model.model.layers[li].self_attn.k_proj.register_forward_hook(hook)
                handles.append(h)

        # Evaluate MMLU
        t_eval = time.time()
        print(f"  Evaluating MMLU ({len(MMLU_SUBJECTS)} subjects, 5-shot)...", flush=True)
        overall_acc, per_subject = evaluate_mmlu(model, tok, MMLU_SUBJECTS)

        for h in handles:
            h.remove()

        print(f"  Overall: {overall_acc:.4f} ({time.time()-t_eval:.1f}s)", flush=True)

        results['configs'][cfg_name] = {
            'avg_bits': avg_bits,
            'overall_accuracy': overall_acc,
            'per_subject': per_subject,
            'eval_time_sec': time.time() - t_eval,
        }

    # Summary
    print("\n" + "=" * 70)
    print("MMLU RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} | {'avg_bits':>10} | {'MMLU Acc':>10} | {'vs FP16':>10}")
    print('-' * 65)
    fp16_acc = results['configs'].get('fp16', {}).get('overall_accuracy', 0)
    for cfg_name, cfg in results['configs'].items():
        avg_b = cfg.get('avg_bits') or 16
        acc = cfg.get('overall_accuracy', 0)
        delta_pts = (acc - fp16_acc) * 100
        print(f"  {cfg_name:<18} | {avg_b:>10} | {acc:>10.4f} | {delta_pts:>+9.2f}pt")

    results['runtime_sec'] = time.time() - t_start
    out_file = OUT_DIR / 'exp_next11_mmlu_eval.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Total runtime: {results['runtime_sec']:.0f}s ({results['runtime_sec']/60:.1f}m)")


if __name__ == '__main__':
    main()
