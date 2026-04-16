#!/usr/bin/env python3
"""eval_subtask4_facet_adaptive_v3.py — Facet-Weighted Adaptive Q-Coverage + K-Rotation.

Algorithm:
  Step 0: Measure per-facet ε_f(q) = ||B_f^T q||^2 / ||q||^2 from prefill Q
  Step 1: Build query-adaptive weighted projection P = Σ w_f · P_f
          Generate first tool with this projection
  Step 2: Decay weights of satisfied facets → re-weight → generate second tool
  Repeat until ε_total < threshold or max_tools

This is the full facet-adaptive form: query-dependent facet weighting +
step-adaptive facet decay after each tool emission.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from eval_metatool_subtask4 import build_fc_prompt, extract_tool_names, compute_metrics
from eval_metatool_subtask1 import parse_candidates

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
    p.add_argument("--beta", type=float, default=-0.05)
    p.add_argument("--decay", type=float, default=0.1,
                   help="Facet weight decay factor after tool emission (0=full remove, 1=no decay)")
    p.add_argument("--eps-threshold", type=float, default=0.05)
    p.add_argument("--max-tools", type=int, default=4)
    p.add_argument("--max-new-tokens-per-step", type=int, default=100)
    p.add_argument("--measure-layer", type=int, default=13)
    p.add_argument("--out", required=True)
    return p.parse_args()


class FacetAdaptiveEngine:
    """Manages per-facet projectors and query-adaptive weighting."""

    def __init__(self, model, B_ont, r_per_pair, facet_order,
                 n_q, n_kv, head_dim, device, measure_layer=13):
        self.model = model
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.device = device
        self.g = n_q // n_kv
        self.facet_order = facet_order
        self.n_facets = len(facet_order)
        self.measure_layer = measure_layer

        L, H_kv, d, R = B_ont.shape
        self.L = L
        self.R = R

        # Build per-layer per-head facet column ranges
        self.facet_ranges = {}  # (l, h_kv) -> [(lo, hi), ...]
        for l in range(L):
            for h_kv in range(H_kv):
                rp_key = f"L{l}_H{h_kv}"
                ranks = r_per_pair.get(rp_key, [R // self.n_facets] * self.n_facets)
                ranges = []
                offset = 0
                for f_idx in range(self.n_facets):
                    r_f = min(ranks[f_idx] if f_idx < len(ranks) else 0, R - offset)
                    ranges.append((offset, offset + max(r_f, 0)))
                    offset += max(r_f, 0)
                self.facet_ranges[(l, h_kv)] = ranges

        # Store B_ont per layer in Q-head space
        self.B_ont = B_ont  # (L, H_kv, d, R)
        self.B_q = []  # per layer: (n_q, d, R)
        for l in range(L):
            B_l = B_ont[l].unsqueeze(1).expand(-1, self.g, -1, -1).reshape(n_q, d, R)
            self.B_q.append(B_l.to(device))

        self.handles = []

    def measure_facet_weights(self, prompt_ids):
        """Run prefill and measure per-facet ε_f from Q at measure_layer."""
        eps_per_facet = torch.zeros(self.n_facets, device=self.device)
        captured = {}

        def capture_hook(mod, inp, out):
            captured['q'] = out.detach()

        h = self.model.model.layers[self.measure_layer].self_attn.q_proj.register_forward_hook(capture_hook)
        with torch.no_grad():
            self.model(prompt_ids)
        h.remove()

        Q = captured['q']  # (1, T, n_q * d)
        B, T, D = Q.shape
        Q = Q.view(B, T, self.n_q, self.head_dim).float()
        q_last = Q[0, -1]  # (n_q, d) — last token Q vectors

        B_layer = self.B_q[self.measure_layer]  # (n_q, d, R)
        # Project onto each facet's columns
        for h_kv in range(self.n_kv):
            ranges = self.facet_ranges[(self.measure_layer, h_kv)]
            for f_idx, (lo, hi) in enumerate(ranges):
                if hi <= lo:
                    continue
                for q_h in range(h_kv * self.g, (h_kv + 1) * self.g):
                    B_f = B_layer[q_h, :, lo:hi]  # (d, r_f)
                    q_h_vec = q_last[q_h]  # (d,)
                    proj = B_f.T @ q_h_vec  # (r_f,)
                    energy = (proj ** 2).sum()
                    q_norm_sq = (q_h_vec ** 2).sum() + 1e-8
                    eps_per_facet[f_idx] += (energy / q_norm_sq).item()

        # Normalize across Q-heads
        eps_per_facet /= self.n_q
        # Normalize to weights
        total = eps_per_facet.sum() + 1e-8
        weights = eps_per_facet / total
        return weights

    def build_weighted_projector(self, weights, layer_idx):
        """Build P_weighted = Σ w_f · B_f @ B_f^T for given layer in Q-head space."""
        P = torch.zeros(self.n_q, self.head_dim, self.head_dim, device=self.device)
        for h_kv in range(self.n_kv):
            ranges = self.facet_ranges[(layer_idx, h_kv)]
            B_l = self.B_ont[layer_idx, h_kv].to(self.device)  # (d, R)
            for f_idx, (lo, hi) in enumerate(ranges):
                if hi <= lo or f_idx >= len(weights):
                    continue
                B_f = B_l[:, lo:hi]  # (d, r_f)
                P_f = B_f @ B_f.T  # (d, d)
                w = weights[f_idx].item()
                for q_h in range(h_kv * self.g, (h_kv + 1) * self.g):
                    P[q_h] += w * P_f
        return P

    def install_hooks(self, weights, alpha, beta):
        """Install Q-coverage and K-boost hooks with facet-weighted projectors."""
        self.remove_hooks()

        for layer_idx in range(min(self.L, len(self.model.model.layers))):
            P_w = self.build_weighted_projector(weights, layer_idx)
            P_kv = P_w.view(self.n_kv, self.g, self.head_dim, self.head_dim).mean(dim=1)
            layer = self.model.model.layers[layer_idx]

            def make_q_hook(P_layer, _beta=beta):
                def hook(mod, inp, out):
                    B, T, D = out.shape
                    if D != self.n_q * self.head_dim:
                        return out
                    Q = out.view(B, T, self.n_q, self.head_dim).float()
                    Q_proj = torch.einsum("btnd,nde->btne", Q, P_layer)
                    Q = Q + _beta * Q_proj
                    return Q.to(out.dtype).view(B, T, D)
                return hook

            def make_k_hook(P_kv_layer, _alpha=alpha):
                def hook(mod, inp, out):
                    B, T, D = out.shape
                    if D != self.n_kv * self.head_dim:
                        return out
                    K = out.view(B, T, self.n_kv, self.head_dim).float()
                    K_proj = torch.einsum("btnd,nde->btne", K, P_kv_layer)
                    K = K + _alpha * K_proj
                    return K.to(out.dtype).view(B, T, D)
                return hook

            self.handles.append(layer.self_attn.q_proj.register_forward_hook(make_q_hook(P_w)))
            if alpha != 0:
                self.handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(P_kv)))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def decay_weights(self, weights, emitted_tool, decay_factor):
        """Reduce weights of facets satisfied by the emitted tool."""
        facet_vals = TOOL_FACET_MAP.get(emitted_tool, {})
        new_weights = weights.clone()
        for facet_name, facet_val in facet_vals.items():
            if facet_name in self.facet_order:
                f_idx = self.facet_order.index(facet_name)
                new_weights[f_idx] *= decay_factor
        # Re-normalize
        total = new_weights.sum() + 1e-8
        new_weights = new_weights / total
        return new_weights


def extract_first_tool(text, candidates):
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        name = m.group(1)
        for c in candidates:
            if c.lower() == name.lower():
                return c
    return None


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

    print(f"[engine] Building facet-adaptive engine (facets={facet_order})", flush=True)
    engine = FacetAdaptiveEngine(model, B_ont, r_per_pair, facet_order,
                                 n_q, n_kv, head_dim, args.device, args.measure_layer)
    print(f"[engine] Ready", flush=True)

    data = json.load(open(args.dataset))[:args.max_samples]
    print(f"[data] N={len(data)}", flush=True)

    configs = [
        ("no_steer", False, 0.0, 0.0),
        ("static_uniform_q", False, 0.0, args.beta),
        ("adaptive_q_only", True, 0.0, args.beta),
        ("adaptive_kq", True, args.alpha, args.beta),
    ]

    all_results = []

    for config_name, adaptive, alpha, beta in configs:
        print(f"\n{'='*50}\n[eval] {config_name} (α={alpha}, β={beta}, adaptive={adaptive})", flush=True)

        agg = {"F1": 0, "Exact": 0, "Jaccard": 0}
        per_sample = []
        t0 = time.time()

        for i, entry in enumerate(data):
            gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
            cands = parse_candidates(entry["action_prompt"])
            fc = build_fc_prompt(tok, entry["action_prompt"], cands)

            if not adaptive:
                # Static: uniform weights, single generation
                if beta != 0 or alpha != 0:
                    uniform_w = torch.ones(engine.n_facets, device=args.device) / engine.n_facets
                    engine.install_hooks(uniform_w, alpha, beta)

                with torch.no_grad():
                    ids = tok(fc, return_tensors="pt")["input_ids"].to(args.device)
                    out = model.generate(ids, max_new_tokens=args.max_new_tokens_per_step * 3,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                engine.remove_hooks()
                pred = extract_tool_names(gen, cands)
            else:
                # Adaptive: measure facet weights → iterative generation
                ids = tok(fc, return_tensors="pt")["input_ids"].to(args.device)

                # Step 0: Measure per-facet weights from query
                weights = engine.measure_facet_weights(ids)

                emitted = []
                accumulated_text = ""

                for tool_step in range(args.max_tools):
                    # Install hooks with current weights
                    engine.install_hooks(weights, alpha, beta)

                    # Generate one tool_call
                    prompt_ids = tok(fc + accumulated_text, return_tensors="pt")["input_ids"].to(args.device)
                    with torch.no_grad():
                        out = model.generate(prompt_ids,
                                             max_new_tokens=args.max_new_tokens_per_step,
                                             do_sample=False,
                                             pad_token_id=tok.eos_token_id)
                    step_text = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
                    engine.remove_hooks()

                    accumulated_text += step_text
                    tool = extract_first_tool(step_text, cands)

                    if tool and tool not in emitted:
                        emitted.append(tool)
                        # Decay satisfied facets
                        weights = engine.decay_weights(weights, tool, args.decay)

                        # Check remaining energy
                        remaining_energy = weights.max().item()
                        if remaining_energy < args.eps_threshold:
                            break
                    else:
                        break

                pred = emitted

            m = compute_metrics(pred, gt)
            for k in agg:
                agg[k] += m.get(k, 0)
            per_sample.append({
                "gt": gt, "pred": pred, "F1": m["F1"],
                "facet_weights": weights.tolist() if adaptive else None,
                "gen_head": (accumulated_text if adaptive else gen)[:200],
            })

            if i % 10 == 0:
                w_str = f" w={[f'{w:.2f}' for w in weights.tolist()]}" if adaptive else ""
                print(f"  [{i}/{len(data)}] gt={gt} pred={pred} F1={m['F1']:.3f}{w_str}", flush=True)

        N = len(data)
        macro = {k: v / N for k, v in agg.items()}
        runtime = time.time() - t0
        print(f"  → {config_name}: F1={macro['F1']:.4f} Exact={macro['Exact']:.4f} ({runtime:.1f}s)", flush=True)

        all_results.append({
            "config": config_name, "alpha": alpha, "beta": beta,
            "adaptive": adaptive, "decay": args.decay,
            "n": N, "macro": macro, "runtime_s": runtime, "per_sample": per_sample,
        })

    print(f"\n{'='*60}", flush=True)
    print(f"{'Config':25s} {'F1':>8s} {'Exact':>8s} {'Δ F1':>8s}", flush=True)
    print(f"{'-'*60}", flush=True)
    base_f1 = all_results[0]["macro"]["F1"]
    for r in all_results:
        delta = r["macro"]["F1"] - base_f1
        print(f"{r['config']:25s} {r['macro']['F1']:8.4f} {r['macro']['Exact']:8.4f} {delta:+8.4f}", flush=True)

    blob = {
        "model": args.model, "alpha": args.alpha, "beta": args.beta,
        "decay": args.decay, "eps_threshold": args.eps_threshold,
        "facet_order": facet_order, "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
