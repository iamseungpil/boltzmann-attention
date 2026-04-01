"""
DHS/HEAT Phase 2 experiments.

1. RoPE-kappa coupling on a RoPE model by sweeping rope_theta.
2. Residual-reheating coupling on GPT-2 by sweeping residual scale.
"""

import argparse
import json
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from scipy import stats
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dhs_phase1 import (
    MODEL_PRESETS,
    extract_mean_profiles,
    fit_profile,
    make_long_texts,
)


def load_model_with_config(model_name, device, trust_remote_code=False, attn_impl="eager", config=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        attn_implementation=attn_impl,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def run_rope_kappa(args):
    preset = MODEL_PRESETS[args.model]
    theta_results = []

    for rope_theta in args.rope_thetas:
        config = AutoConfig.from_pretrained(
            preset["name"],
            trust_remote_code=preset.get("trust_remote_code", False),
        )
        if not hasattr(config, "rope_theta"):
            raise ValueError(f"Model {preset['name']} does not expose rope_theta")
        config.rope_theta = float(rope_theta)

        model, tokenizer = load_model_with_config(
            preset["name"],
            device=args.device,
            trust_remote_code=preset.get("trust_remote_code", False),
            attn_impl=preset.get("attn_impl", "eager"),
            config=config,
        )

        data = extract_mean_profiles(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_seq_len=args.max_seq_len,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )

        k2_values = []
        dual_r2 = []
        profiles = data["profiles"]
        num_layers, num_heads, _ = profiles.shape
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                fit = fit_profile(profiles[layer_idx, head_idx])
                dual = fit["dual_exp"]
                if dual["ok"]:
                    k2_values.append(float(dual["params"][3]))
                    dual_r2.append(float(dual["r2"]))

        theta_results.append(
            {
                "rope_theta": float(rope_theta),
                "mean_kappa2": float(np.mean(k2_values)),
                "median_kappa2": float(np.median(k2_values)),
                "std_kappa2": float(np.std(k2_values)),
                "mean_dual_exp_r2": float(np.mean(dual_r2)),
                "num_fitted_heads": len(k2_values),
            }
        )

        del model
        torch.cuda.empty_cache()

    thetas = [item["rope_theta"] for item in theta_results]
    medians = [item["median_kappa2"] for item in theta_results]
    means = [item["mean_kappa2"] for item in theta_results]
    corr_med = stats.spearmanr(thetas, medians)
    corr_mean = stats.spearmanr(thetas, means)
    return {
        "experiment": "dhs_phase2_rope_kappa_coupling",
        "model_key": args.model,
        "model_name": preset["name"],
        "rope_theta_results": theta_results,
        "spearman_theta_vs_median_kappa2": {
            "rho": float(corr_med.statistic),
            "p_value": float(corr_med.pvalue),
        },
        "spearman_theta_vs_mean_kappa2": {
            "rho": float(corr_mean.statistic),
            "p_value": float(corr_mean.pvalue),
        },
        "runtime_s": time.time() - args.start_time,
    }


def patch_gpt2_residual_scale(model, residual_scale):
    for block in model.transformer.h:
        def forward_with_scale(
            self,
            hidden_states,
            past_key_value=None,
            cache_position=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            _scale=residual_scale,
            **kwargs,
        ):
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            attn_outputs = self.attn(
                hidden_states,
                past_key_value=past_key_value,
                cache_position=cache_position,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]
            hidden_states = attn_output + (_scale * residual)

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = feed_forward_hidden_states + (_scale * residual)

            if use_cache:
                outputs = (hidden_states,) + outputs
            else:
                outputs = (hidden_states,) + outputs[1:]
            return outputs

        block.forward = MethodType(forward_with_scale, block)


def compute_teff_from_attentions(outputs, attention_mask):
    num_layers = len(outputs.attentions)
    seq_len = attention_mask.shape[1]
    layer_head_teff = []
    for layer_idx in range(num_layers):
        attn = outputs.attentions[layer_idx].float()
        log_alpha = torch.where(attn > 1e-10, attn.log(), torch.zeros_like(attn))
        entropy = -(attn * log_alpha).sum(dim=-1)
        pos_idx = torch.arange(1, seq_len + 1, device=attn.device).float()
        log_n = pos_idx.log().clamp(min=1e-6).unsqueeze(0).unsqueeze(0)
        teff = entropy / log_n
        mask = attention_mask.unsqueeze(1).float()
        mask[:, :, 0] = 0
        teff_avg = (teff * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        layer_head_teff.append(teff_avg.mean(0).cpu().numpy())
    return np.stack(layer_head_teff, axis=0)


def run_residual_reheating(args):
    preset = MODEL_PRESETS[args.model]
    texts = None
    scale_results = []

    for residual_scale in args.residual_scales:
        model, tokenizer = load_model_with_config(
            preset["name"],
            device=args.device,
            trust_remote_code=preset.get("trust_remote_code", False),
            attn_impl=preset.get("attn_impl", "eager"),
        )
        patch_gpt2_residual_scale(model, residual_scale)
        if texts is None:
            texts = make_long_texts(tokenizer, args.max_seq_len, args.num_samples)

        layer_head_accum = []
        batches = 0
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start:start + args.batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
            ).to(args.device)
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            layer_head_teff = compute_teff_from_attentions(outputs, inputs["attention_mask"])
            layer_head_accum.append(layer_head_teff)
            batches += 1
            del outputs, inputs
            torch.cuda.empty_cache()

        layer_head_teff = np.mean(layer_head_accum, axis=0)
        layer_teff_mean = layer_head_teff.mean(axis=1)
        min_idx = int(np.argmin(layer_teff_mean))
        delta_reheat = float(layer_teff_mean[-1] - layer_teff_mean[min_idx])
        scale_results.append(
            {
                "residual_scale": float(residual_scale),
                "layer_teff_mean": [float(v) for v in layer_teff_mean],
                "min_layer": min_idx,
                "min_teff": float(layer_teff_mean[min_idx]),
                "final_teff": float(layer_teff_mean[-1]),
                "delta_reheat": delta_reheat,
            }
        )

        del model
        torch.cuda.empty_cache()

    scales = [item["residual_scale"] for item in scale_results]
    deltas = [item["delta_reheat"] for item in scale_results]
    corr = stats.spearmanr(scales, deltas)
    return {
        "experiment": "dhs_phase2_residual_reheating_coupling",
        "model_key": args.model,
        "model_name": preset["name"],
        "scale_results": scale_results,
        "spearman_residual_scale_vs_delta_reheat": {
            "rho": float(corr.statistic),
            "p_value": float(corr.pvalue),
        },
        "runtime_s": time.time() - args.start_time,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["rope_kappa", "residual_reheating"])
    parser.add_argument("--model", required=True, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rope-thetas", type=float, nargs="*", default=[1e4, 1e5, 1e6, 1e7])
    parser.add_argument("--residual-scales", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0, 1.25])
    return parser.parse_args()


def main():
    args = parse_args()
    args.start_time = time.time()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.experiment == "rope_kappa":
        result = run_rope_kappa(args)
    else:
        result = run_residual_reheating(args)

    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "model": result["model_name"],
                "output": str(out_path),
                "runtime_s": result["runtime_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
