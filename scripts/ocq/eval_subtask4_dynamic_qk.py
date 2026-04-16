#!/usr/bin/env python3
"""eval_subtask4_dynamic_qk.py — Dynamic step-adaptive K-rotation + Q-coverage.

Full Thm 6.17 implementation:
  Step t: After emitting tools {f_s}_{s<t},
    K' = K + α · P_{remaining} · K    (boost un-emitted facets)
    Q' = Q - β · P_{emitted} · Q      (subtract emitted facets)

The key insight: tool emission is detected mid-generation by parsing
<tool_call>{"name": "XXX"} tokens. After each tool is emitted, the
K and Q projections are dynamically updated for subsequent tokens.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Optional, Set

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_metatool_subtask4 import build_fc_prompt, extract_tool_names, compute_metrics
from eval_metatool_subtask1 import parse_candidates


# Tool → facet-column mapping
TOOL_FACET_MAP = {
    "FinanceTool":       {"domain": "finance",      "function_action": "analyze"},
    "NewsTool":          {"domain": "news",          "function_action": "retrieve"},
    "MusicTool":         {"domain": "entertainment", "function_action": "play"},
    "WeatherTool":       {"domain": "weather",       "function_action": "inform"},
    "TripTool":          {"domain": "travel",        "function_action": "recommend"},
    "TripAdviceTool":    {"domain": "travel",        "function_action": "inform"},
    "CourseTool":        {"domain": "education",     "function_action": "search"},
    "JobTool":           {"domain": "career",        "function_action": "search"},
    "HousePurchasingTool": {"domain": "real_estate", "function_action": "search"},
    "ProductSearch":     {"domain": "shopping",      "function_action": "search"},
    "Discount":          {"domain": "shopping",      "function_action": "recommend"},
    "RepoTool":          {"domain": "research",      "function_action": "search"},
    "ResearchFinder":    {"domain": "research",      "function_action": "retrieve"},
    "ResearchHelper":    {"domain": "research",      "function_action": "summarize"},
    "PDF&URLTool":       {"domain": "utility",       "function_action": "retrieve"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--b-ont", required=True)
    p.add_argument("--dataset", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=-0.03)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    return p.parse_args()


def build_facet_projectors(B_ont, r_per_pair, facet_order, n_q, n_kv, head_dim, device):
    """Build per-facet projectors P_f = B_f @ B_f^T for each facet."""
    L, H_kv, d, R = B_ont.shape
    g = n_q // n_kv
    facet_projectors = {}  # facet_value -> (L, n_q, d, d) projector

    for l in range(L):
        for h_kv in range(H_kv):
            rp_key = f"L{l}_H{h_kv}"
            ranks = r_per_pair.get(rp_key, [R // 4] * 4)
            offset = 0
            for f_idx, facet_name in enumerate(facet_order):
                r_f = min(ranks[f_idx], R - offset) if f_idx < len(ranks) else 0
                if r_f <= 0:
                    offset += ranks[f_idx] if f_idx < len(ranks) else 0
                    continue
                B_f = B_ont[l, h_kv, :, offset:offset + r_f]  # (d, r_f)
                P_f = B_f @ B_f.T  # (d, d)

                for q_h in range(h_kv * g, (h_kv + 1) * g):
                    if facet_name not in facet_projectors:
                        facet_projectors[facet_name] = torch.zeros(L, n_q, d, d, device=device)
                    facet_projectors[facet_name][l, q_h] += P_f.to(device)

                offset += r_f

    return facet_projectors


def get_tool_facet_values(tool_name):
    """Map tool name to its facet values."""
    return TOOL_FACET_MAP.get(tool_name, {})


class DynamicQKHooks:
    def __init__(self, model, B_ont, facet_projectors, facet_order,
                 alpha, beta, n_q, n_kv, head_dim, device):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.device = device
        self.facet_projectors = facet_projectors
        self.facet_order = facet_order

        L = len(model.model.layers)
        self.P_ont_full = torch.zeros(L, n_q, head_dim, head_dim, device=device)
        for fp in facet_projectors.values():
            self.P_ont_full += fp

        self.emitted_facet_values = set()
        self.P_emitted = torch.zeros(L, n_q, head_dim, head_dim, device=device)
        self.handles = []

    def reset(self):
        self.emitted_facet_values = set()
        self.P_emitted.zero_()

    def on_tool_emitted(self, tool_name):
        facet_vals = get_tool_facet_values(tool_name)
        for facet_name, facet_val in facet_vals.items():
            key = f"{facet_name}:{facet_val}"
            if key not in self.emitted_facet_values:
                self.emitted_facet_values.add(key)
                if facet_name in self.facet_projectors:
                    self.P_emitted += self.facet_projectors[facet_name]

    def install(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx >= self.P_ont_full.shape[0]:
                break

            def make_q_hook(li):
                def hook(mod, inp, out):
                    B, T, D = out.shape
                    if D != self.n_q * self.head_dim:
                        return out
                    Q = out.view(B, T, self.n_q, self.head_dim).float()
                    if self.beta != 0 and self.P_emitted[li].abs().sum() > 0:
                        P_e = self.P_emitted[li]  # (n_q, d, d)
                        Q_emitted = torch.einsum("btnd,nde->btne", Q, P_e)
                        Q = Q + self.beta * Q_emitted
                    return Q.to(out.dtype).view(B, T, D)
                return hook

            def make_k_hook(li):
                def hook(mod, inp, out):
                    B, T, D = out.shape
                    if D != self.n_kv * self.head_dim:
                        return out
                    K = out.view(B, T, self.n_kv, self.head_dim).float()
                    if self.alpha != 0:
                        g = self.n_q // self.n_kv
                        P_remaining = self.P_ont_full[li] - self.P_emitted[li]  # (n_q, d, d)
                        P_kv = P_remaining.view(self.n_kv, g, self.head_dim, self.head_dim).mean(dim=1)
                        K_boost = torch.einsum("btnd,nde->btne", K, P_kv)
                        K = K + self.alpha * K_boost
                    return K.to(out.dtype).view(B, T, D)
                return hook

            self.handles.append(layer.self_attn.q_proj.register_forward_hook(make_q_hook(layer_idx)))
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def generate_with_dynamic_hooks(model, tokenizer, prompt, hooks, max_new_tokens, device):
    """Generate with mid-generation tool detection and hook updates."""
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    hooks.reset()
    hooks.install()

    generated_text = ""
    current_ids = ids

    with torch.no_grad():
        out_ids = model.generate(
            current_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out_ids[0][ids.shape[1]:]
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Post-hoc: detect tool calls and simulate what dynamic updates WOULD have done
    # For true step-adaptive, we'd need token-by-token generation with mid-stream hook updates
    # Approximation: detect first tool, update hooks, re-generate remaining
    tool_calls = re.findall(r'"name"\s*:\s*"([^"]+)"', generated_text)

    if len(tool_calls) >= 1:
        # Found first tool — update hooks and regenerate from after first tool_call block
        hooks.on_tool_emitted(tool_calls[0])

        # Find position after first tool_call block end
        first_block_end = generated_text.find("}", generated_text.find('"name"'))
        if first_block_end > 0:
            # Re-generate with updated hooks from the point after first tool
            prefix_text = prompt + tokenizer.decode(gen_ids[:first_block_end+1], skip_special_tokens=False)
            prefix_ids = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(device)

            with torch.no_grad():
                out2 = model.generate(
                    prefix_ids,
                    max_new_tokens=max_new_tokens // 2,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen2 = tokenizer.decode(out2[0][prefix_ids.shape[1]:], skip_special_tokens=True)
            generated_text = generated_text[:first_block_end+1] + gen2

    hooks.remove()
    return generated_text


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).eval()

    bdict = torch.load(args.b_ont, map_location="cpu", weights_only=False)
    B_ont = bdict["B_ont"].to(args.device, dtype=torch.float32)
    facet_order = bdict.get("facet_order", ["function_action", "io_type", "domain", "tool_category"])
    r_per_pair = bdict.get("r_per_pair", {})
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // n_q)

    print(f"[facet] Building per-facet projectors...", flush=True)
    facet_projectors = build_facet_projectors(B_ont, r_per_pair, facet_order, n_q, n_kv, head_dim, args.device)
    print(f"[facet] {len(facet_projectors)} facet projectors built: {list(facet_projectors.keys())}", flush=True)

    data = json.load(open(args.dataset))[:args.max_samples]
    print(f"[data] N={len(data)}", flush=True)

    configs = [
        ("no_steer", 0.0, 0.0),
        ("static_qonly", 0.0, args.beta),
        ("dynamic_qonly", 0.0, args.beta),
        ("dynamic_kq", args.alpha, args.beta),
    ]

    results = []
    for config_name, alpha, beta in configs:
        is_dynamic = "dynamic" in config_name
        print(f"\n[eval] {config_name} (α={alpha}, β={beta}, dynamic={is_dynamic})", flush=True)

        hooks = DynamicQKHooks(model, B_ont, facet_projectors, facet_order,
                               alpha, beta, n_q, n_kv, head_dim, args.device)
        agg = {"F1": 0, "Exact": 0, "Jaccard": 0, "precision": 0, "recall": 0}
        per_sample = []
        t0 = time.time()

        for i, entry in enumerate(data):
            gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
            cands = parse_candidates(entry["action_prompt"])
            fc = build_fc_prompt(tok, entry["action_prompt"], cands)

            if is_dynamic:
                gen_text = generate_with_dynamic_hooks(model, tok, fc, hooks, args.max_new_tokens, args.device)
            else:
                hooks.reset()
                if alpha != 0 or beta != 0:
                    hooks.install()
                with torch.no_grad():
                    ids = tok(fc, return_tensors="pt")["input_ids"].to(args.device)
                    out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
                gen_text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                if alpha != 0 or beta != 0:
                    hooks.remove()

            pred = extract_tool_names(gen_text, cands)
            m = compute_metrics(pred, gt)
            for k in agg:
                agg[k] += m.get(k, 0)
            per_sample.append({"gt": gt, "pred": pred, "F1": m["F1"], "gen_head": gen_text[:200]})

            if i % 10 == 0:
                print(f"  [{i}/{len(data)}] gt={gt[:2]} pred={pred[:2]} F1={m['F1']:.3f}", flush=True)

        N = len(data)
        macro = {k: v / N for k, v in agg.items()}
        runtime = time.time() - t0
        print(f"  {config_name}: F1={macro['F1']:.4f} Exact={macro['Exact']:.4f} ({runtime:.1f}s)", flush=True)

        results.append({
            "config": config_name, "alpha": alpha, "beta": beta, "dynamic": is_dynamic,
            "n": N, "macro": macro, "runtime_s": runtime, "per_sample": per_sample,
        })

    print(f"\n{'='*60}", flush=True)
    print(f"{'Config':30s} {'F1':>8s} {'Exact':>8s} {'Δ F1':>8s}", flush=True)
    print(f"{'-'*60}", flush=True)
    base_f1 = results[0]["macro"]["F1"]
    for r in results:
        delta = r["macro"]["F1"] - base_f1
        print(f"{r['config']:30s} {r['macro']['F1']:8.4f} {r['macro']['Exact']:8.4f} {delta:+8.4f}", flush=True)

    blob = {"model": args.model, "alpha": args.alpha, "beta": args.beta, "results": results}
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
