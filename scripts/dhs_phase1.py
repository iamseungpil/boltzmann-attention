"""
DHS/HEAT Phase 1 experiments.

Implements the first theory-critical experiments from the integrated plan:
1. Dual-exponential superiority against simple baselines.
2. Sequence-length valley-depth scaling.
3. GQA multimodality via per-query-head and per-KV-group profiles.

The script reuses standard attention outputs and keeps outputs in JSON so later
reporting can consume them directly.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_TEXTS = [
    "The transformer architecture revolutionized natural language processing by introducing self attention mechanisms that compare every token with the rest of the sequence.",
    "In statistical mechanics, the Boltzmann distribution describes the probability of a system being in a certain state as a function of energy and temperature.",
    "Climate change models predict rising sea levels, more frequent extreme weather events, and long-term shifts in regional precipitation patterns.",
    "Deep learning has achieved strong performance in computer vision, speech recognition, natural language understanding, and strategic game playing.",
    "Information theory quantifies the limits of data compression and reliable communication over noisy channels using entropy and mutual information.",
    "Bayesian inference provides a principled framework for updating beliefs about model parameters using observed evidence and prior assumptions.",
    "Graph neural networks extend deep learning to non Euclidean domains by operating directly on graph structured data.",
    "Reinforcement learning from human feedback has become a dominant paradigm for aligning large language models with user preferences.",
]


MODEL_PRESETS = {
    "gpt2-medium": {
        "name": "openai-community/gpt2-medium",
        "attn_impl": "eager",
    },
    "qwen2.5-3b": {
        "name": "Qwen/Qwen2.5-3B",
        "attn_impl": "eager",
        "trust_remote_code": True,
    },
}


def cfg_attr(cfg, *names):
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    raise AttributeError(f"Missing config attribute among {names}")


def make_long_texts(tokenizer, max_seq_len, num_samples):
    texts = []
    idx = 0
    while len(texts) < num_samples:
        seed = BASE_TEXTS[idx % len(BASE_TEXTS)]
        chunks = [seed]
        while True:
            candidate = " ".join(chunks)
            tok_len = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
            if tok_len >= max_seq_len + 32:
                break
            chunks.append(BASE_TEXTS[(idx + len(chunks)) % len(BASE_TEXTS)])
        texts.append(" ".join(chunks))
        idx += 1
    return texts


def load_model(model_key, device):
    preset = MODEL_PRESETS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        preset["name"],
        trust_remote_code=preset.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        preset["name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        attn_implementation=preset.get("attn_impl", "eager"),
        trust_remote_code=preset.get("trust_remote_code", False),
    )
    model.eval()
    return model, tokenizer


def model_head_info(model):
    cfg = model.config
    num_heads = cfg_attr(cfg, "num_attention_heads", "n_head")
    hidden_size = cfg_attr(cfg, "hidden_size", "n_embd")
    kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    return {
        "num_layers": cfg_attr(cfg, "num_hidden_layers", "n_layer"),
        "num_heads": num_heads,
        "num_kv_heads": kv_heads,
        "hidden_size": hidden_size,
        "head_dim": hidden_size // num_heads,
        "max_positions": getattr(cfg, "max_position_embeddings", getattr(cfg, "n_positions", None)),
    }


def extract_mean_profiles(model, tokenizer, device, max_seq_len, num_samples, batch_size):
    info = model_head_info(model)
    texts = make_long_texts(tokenizer, max_seq_len, num_samples)
    num_layers = info["num_layers"]
    num_heads = info["num_heads"]

    layer_profiles = np.zeros((num_layers, num_heads, max_seq_len), dtype=np.float64)
    layer_counts = np.zeros((num_layers, num_heads), dtype=np.int64)

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        for b in range(attention_mask.shape[0]):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len < 16:
                continue
            for layer_idx, attn in enumerate(outputs.attentions):
                # Average how much each key position is attended to.
                profile = attn[b, :, :valid_len, :valid_len].float().mean(dim=1).cpu().numpy()
                layer_profiles[layer_idx, :, :valid_len] += profile
                layer_counts[layer_idx] += 1

        del outputs, inputs
        torch.cuda.empty_cache()

    valid = layer_counts > 0
    layer_profiles[valid] /= layer_counts[valid][..., None]
    return {
        "profiles": layer_profiles,
        "counts": layer_counts,
        "max_seq_len": max_seq_len,
        "num_samples": num_samples,
    }


def dual_exp_fn(x, a, b, k1, k2, c, seq_len):
    return a * np.exp(-k1 * x) + b * np.exp(-k2 * ((seq_len - 1) - x)) + c


def single_exp_fn(x, a, k, c):
    return a * np.exp(-k * x) + c


def compute_r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def compute_aic_bic(y, yhat, k_params):
    n = len(y)
    rss = float(np.sum((y - yhat) ** 2))
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k_params
    bic = n * np.log(rss / n) + k_params * np.log(n)
    return aic, bic


def fit_profile(profile):
    y = np.asarray(profile, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    eps = 1e-8

    # Dual exponential.
    dual_result = {"ok": False}
    try:
        p0 = [max(y[0] - np.median(y), eps), max(y[-1] - np.median(y), eps), 0.02, 0.02, max(np.min(y), eps)]
        bounds = ([0.0, 0.0, 1e-5, 1e-5, 0.0], [10.0, 10.0, 2.0, 2.0, 10.0])
        popt, _ = curve_fit(
            lambda xx, a, b, k1, k2, c: dual_exp_fn(xx, a, b, k1, k2, c, len(y)),
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        yhat = dual_exp_fn(x, *popt, seq_len=len(y))
        aic, bic = compute_aic_bic(y, yhat, 5)
        dual_result = {
            "ok": True,
            "params": [float(v) for v in popt],
            "r2": float(compute_r2(y, yhat)),
            "aic": float(aic),
            "bic": float(bic),
        }
    except Exception as exc:
        dual_result["error"] = str(exc)

    # Poly-2.
    p2 = np.polyfit(x, y, 2)
    yhat2 = np.polyval(p2, x)
    aic2, bic2 = compute_aic_bic(y, yhat2, 3)

    # Poly-4.
    p4 = np.polyfit(x, y, 4)
    yhat4 = np.polyval(p4, x)
    aic4, bic4 = compute_aic_bic(y, yhat4, 5)

    # Single exponential.
    single_result = {"ok": False}
    try:
        p0 = [max(y[0] - np.median(y), eps), 0.02, max(np.min(y), eps)]
        bounds = ([0.0, 1e-5, 0.0], [10.0, 2.0, 10.0])
        popt, _ = curve_fit(single_exp_fn, x, y, p0=p0, bounds=bounds, maxfev=20000)
        yhat = single_exp_fn(x, *popt)
        aic, bic = compute_aic_bic(y, yhat, 3)
        single_result = {
            "ok": True,
            "params": [float(v) for v in popt],
            "r2": float(compute_r2(y, yhat)),
            "aic": float(aic),
            "bic": float(bic),
        }
    except Exception as exc:
        single_result["error"] = str(exc)

    return {
        "dual_exp": dual_result,
        "poly2": {
            "params": [float(v) for v in p2],
            "r2": float(compute_r2(y, yhat2)),
            "aic": float(aic2),
            "bic": float(bic2),
        },
        "poly4": {
            "params": [float(v) for v in p4],
            "r2": float(compute_r2(y, yhat4)),
            "aic": float(aic4),
            "bic": float(bic4),
        },
        "single_exp": single_result,
    }


def valley_depth(profile):
    y = np.asarray(profile, dtype=np.float64)
    center = y[len(y) // 2]
    return float(((y[0] + y[-1]) / 2.0) / max(center, 1e-8))


def run_dual_exp(args):
    model, tokenizer = load_model(args.model, args.device)
    info = model_head_info(model)
    data = extract_mean_profiles(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_seq_len=args.max_seq_len,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    wins_r2 = 0
    wins_aic = 0
    wins_bic = 0
    total = 0
    per_head = []
    dual_r2 = []
    poly4_r2 = []

    for layer_idx in range(info["num_layers"]):
        for head_idx in range(info["num_heads"]):
            profile = data["profiles"][layer_idx, head_idx]
            fit = fit_profile(profile)
            if not fit["dual_exp"]["ok"]:
                continue
            total += 1
            dual = fit["dual_exp"]
            poly4 = fit["poly4"]
            wins_r2 += int(dual["r2"] > poly4["r2"])
            wins_aic += int(dual["aic"] < poly4["aic"])
            wins_bic += int(dual["bic"] < poly4["bic"])
            dual_r2.append(dual["r2"])
            poly4_r2.append(poly4["r2"])
            per_head.append({
                "layer": layer_idx,
                "head": head_idx,
                "dual_exp_r2": dual["r2"],
                "poly4_r2": poly4["r2"],
                "single_exp_r2": fit["single_exp"].get("r2"),
                "dual_exp_params": dual["params"],
            })

    result = {
        "experiment": "dhs_phase1_dual_exponential_superiority",
        "model_key": args.model,
        "model_name": MODEL_PRESETS[args.model]["name"],
        "num_layers": info["num_layers"],
        "num_heads": info["num_heads"],
        "num_kv_heads": info["num_kv_heads"],
        "max_seq_len": args.max_seq_len,
        "num_samples": args.num_samples,
        "total_fitted_heads": total,
        "dual_exp_beats_poly4_r2_fraction": wins_r2 / max(total, 1),
        "dual_exp_beats_poly4_aic_fraction": wins_aic / max(total, 1),
        "dual_exp_beats_poly4_bic_fraction": wins_bic / max(total, 1),
        "mean_dual_exp_r2": float(np.mean(dual_r2)) if dual_r2 else None,
        "mean_poly4_r2": float(np.mean(poly4_r2)) if poly4_r2 else None,
        "paired_t_r2_dual_vs_poly4": (
            {
                "t_stat": float(stats.ttest_rel(dual_r2, poly4_r2).statistic),
                "p_value": float(stats.ttest_rel(dual_r2, poly4_r2).pvalue),
            }
            if len(dual_r2) > 1
            else None
        ),
        "per_head": per_head,
        "runtime_s": time.time() - args.start_time,
    }
    return result


def run_length_scaling(args):
    model, tokenizer = load_model(args.model, args.device)
    info = model_head_info(model)
    max_positions = info["max_positions"]
    length_results = []
    all_depths = []
    used_lengths = []
    for seq_len in args.lengths:
        if max_positions is not None and seq_len > max_positions:
            continue
        data = extract_mean_profiles(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_seq_len=seq_len,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
        depths = []
        for layer_idx in range(info["num_layers"]):
            for head_idx in range(info["num_heads"]):
                depths.append(valley_depth(data["profiles"][layer_idx, head_idx, :seq_len]))
        length_results.append({
            "seq_len": seq_len,
            "median_valley_depth": float(np.median(depths)),
            "mean_valley_depth": float(np.mean(depths)),
            "std_valley_depth": float(np.std(depths)),
        })
        all_depths.append(float(np.median(depths)))
        used_lengths.append(seq_len)

    corr = stats.spearmanr(np.log(used_lengths), all_depths)
    return {
        "experiment": "dhs_phase1_sequence_length_scaling",
        "model_key": args.model,
        "model_name": MODEL_PRESETS[args.model]["name"],
        "length_results": length_results,
        "used_lengths": used_lengths,
        "skipped_lengths": [seq_len for seq_len in args.lengths if seq_len not in used_lengths],
        "spearman_log_length_vs_median_valley_depth": {
            "rho": float(corr.statistic),
            "p_value": float(corr.pvalue),
        },
        "runtime_s": time.time() - args.start_time,
    }


def run_gqa_multimodality(args):
    model, tokenizer = load_model(args.model, args.device)
    info = model_head_info(model)
    data = extract_mean_profiles(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_seq_len=args.max_seq_len,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    group_size = max(info["num_heads"] // max(info["num_kv_heads"], 1), 1)
    per_q_head_fit = []
    kv_group_profiles = []

    for layer_idx in range(info["num_layers"]):
        for head_idx in range(info["num_heads"]):
            fit = fit_profile(data["profiles"][layer_idx, head_idx])
            if fit["dual_exp"]["ok"]:
                per_q_head_fit.append(fit["dual_exp"]["r2"])

        for kv_idx in range(info["num_kv_heads"]):
            h0 = kv_idx * group_size
            h1 = min((kv_idx + 1) * group_size, info["num_heads"])
            group_profile = data["profiles"][layer_idx, h0:h1].mean(axis=0)
            peaks, _ = find_peaks(group_profile, prominence=np.max(group_profile) * 0.02)
            kv_group_profiles.append({
                "layer": layer_idx,
                "kv_group": kv_idx,
                "num_peaks": int(len(peaks)),
                "peak_positions": [int(p) for p in peaks.tolist()],
                "valley_depth": valley_depth(group_profile),
            })

    multimodal_groups = sum(1 for item in kv_group_profiles if item["num_peaks"] >= 2)
    return {
        "experiment": "dhs_phase1_gqa_multimodality",
        "model_key": args.model,
        "model_name": MODEL_PRESETS[args.model]["name"],
        "num_heads": info["num_heads"],
        "num_kv_heads": info["num_kv_heads"],
        "mean_per_q_head_dual_exp_r2": float(np.mean(per_q_head_fit)) if per_q_head_fit else None,
        "fraction_kv_groups_multimodal": multimodal_groups / max(len(kv_group_profiles), 1),
        "kv_group_profiles": kv_group_profiles,
        "runtime_s": time.time() - args.start_time,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["dual_exp", "length_scaling", "gqa_multimodality"])
    parser.add_argument("--model", required=True, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lengths", type=int, nargs="*", default=[256, 512, 1024, 2048])
    return parser.parse_args()


def main():
    args = parse_args()
    args.start_time = time.time()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.experiment == "dual_exp":
        result = run_dual_exp(args)
    elif args.experiment == "length_scaling":
        result = run_length_scaling(args)
    else:
        result = run_gqa_multimodality(args)

    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({
        "experiment": result["experiment"],
        "model": result["model_name"],
        "output": str(out_path),
        "runtime_s": result["runtime_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
