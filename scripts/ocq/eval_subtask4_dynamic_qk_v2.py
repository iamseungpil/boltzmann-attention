#!/usr/bin/env python3
"""eval_subtask4_dynamic_qk_v2.py — Iterative multi-tool generation with dynamic K-rotation + Q-coverage.

Algorithm:
  emitted = {}
  while ε_q > threshold AND n_emitted < max_tools:
    1. Install hooks: K boost (P_ont - P_emitted), Q subtract P_emitted
    2. Generate ONE <tool_call> block
    3. Parse tool name → add to emitted
    4. Measure ε_q on last-token Q → check threshold
    5. Update P_emitted, loop

This is the iterative form of Thm 6.17 with ε_q stopping (Thm 6.20).
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
    p.add_argument("--beta", type=float, default=-0.03)
    p.add_argument("--eps-threshold", type=float, default=0.10,
                   help="ε_q threshold for stopping. Below this = no more tools.")
    p.add_argument("--max-tools", type=int, default=5)
    p.add_argument("--max-new-tokens-per-tool", type=int, default=80)
    p.add_argument("--layer-mode", default="uniform",
                   choices=["uniform", "layer_adaptive", "k_early_only", "q_late_only"],
                   help="Per-layer Q/K schedule. 'layer_adaptive': K early, Q mid+late.")
    p.add_argument("--stop-at-tool", action="store_true",
                   help="Stop generation at </tool_call> token so iterative loop "
                        "generates exactly one tool per step. Enables true P_emitted "
                        "subtraction between tools.")
    p.add_argument("--out", required=True)
    return p.parse_args()


def build_facet_projectors(B_ont, r_per_pair, facet_order, n_q, n_kv, head_dim, device):
    L, H_kv, d, R = B_ont.shape
    g = n_q // n_kv
    facet_projectors = {}

    for l in range(L):
        for h_kv in range(H_kv):
            rp_key = f"L{l}_H{h_kv}"
            ranks = r_per_pair.get(rp_key, [R // len(facet_order)] * len(facet_order))
            offset = 0
            for f_idx, facet_name in enumerate(facet_order):
                r_f = min(ranks[f_idx] if f_idx < len(ranks) else 0, R - offset)
                if r_f <= 0:
                    offset += ranks[f_idx] if f_idx < len(ranks) else 0
                    continue
                B_f = B_ont[l, h_kv, :, offset:offset + r_f]
                P_f = B_f @ B_f.T
                for q_h in range(h_kv * g, (h_kv + 1) * g):
                    if facet_name not in facet_projectors:
                        facet_projectors[facet_name] = torch.zeros(L, n_q, d, d, device=device)
                    facet_projectors[facet_name][l, q_h] += P_f.to(device)
                offset += r_f

    return facet_projectors


def make_layer_schedule(L, alpha, beta, mode="uniform"):
    """Per-layer alpha_K and beta_Q schedule.

    Modes:
      uniform: same alpha/beta at every layer (original behavior)
      layer_adaptive: K early, Q mid+late (U-shape motivated)
        - Layers 0..L//5:      alpha_K=alpha, beta_Q=0
        - Layers L//5..3L//4:  alpha_K=alpha*0.3, beta_Q=beta
        - Layers 3L//4..L:     alpha_K=0, beta_Q=beta
      k_early_only: K in first 1/4 layers only, Q everywhere
      q_late_only: Q in last 3/4 layers only, K everywhere
    """
    schedule = []
    for li in range(L):
        if mode == "uniform":
            schedule.append((alpha, beta))
        elif mode == "layer_adaptive":
            boundary1 = max(1, L // 5)      # ~layer 5 for L=28
            boundary2 = (3 * L) // 4        # ~layer 21 for L=28
            if li < boundary1:
                schedule.append((alpha, 0.0))           # K only
            elif li < boundary2:
                schedule.append((alpha * 0.3, beta))    # weak K + Q
            else:
                schedule.append((0.0, beta))            # Q only
        elif mode == "k_early_only":
            boundary = L // 4
            if li < boundary:
                schedule.append((alpha, beta))
            else:
                schedule.append((0.0, beta))
        elif mode == "q_late_only":
            boundary = L // 4
            if li < boundary:
                schedule.append((alpha, 0.0))
            else:
                schedule.append((alpha, beta))
        else:
            schedule.append((alpha, beta))
    return schedule


class IterativeDynamicHooks:
    def __init__(self, model, facet_projectors, facet_order, B_ont,
                 alpha, beta, n_q, n_kv, head_dim, device,
                 layer_mode="uniform"):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.layer_mode = layer_mode
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.device = device
        self.facet_projectors = facet_projectors
        self.B_ont = B_ont

        L = B_ont.shape[0]
        self.P_ont_full = torch.zeros(L, n_q, head_dim, head_dim, device=device)
        for fp in facet_projectors.values():
            self.P_ont_full += fp

        self.P_emitted = torch.zeros(L, n_q, head_dim, head_dim, device=device)
        self.emitted_keys = set()
        self.handles = []
        self.last_eps_q = 1.0

    def reset(self):
        self.emitted_keys = set()
        self.P_emitted.zero_()
        self.last_eps_q = 1.0

    def add_emitted_tool(self, tool_name):
        facet_vals = TOOL_FACET_MAP.get(tool_name, {})
        for facet_name, facet_val in facet_vals.items():
            # Bug fix: track at facet_name level (not facet_name:facet_val)
            # to avoid double-counting the same projector when two tools
            # share the same facet category (e.g., domain:travel + domain:finance).
            # See Lemma 6.17.C in THEOREM_SUPPLEMENTS_2026_04_16.md.
            if facet_name not in self.emitted_keys:
                self.emitted_keys.add(facet_name)
                if facet_name in self.facet_projectors:
                    self.P_emitted += self.facet_projectors[facet_name]

    def install(self):
        self.remove()
        L = self.B_ont.shape[0]
        eps_q_capture = []

        # Bug fix: register eps_q hook BEFORE Q-coverage hook on mid layer
        # so it measures unperturbed Q (Thm 6.20 requires original Q).
        mid_l = L // 2
        B_mid = self.B_ont[mid_l]  # (H_kv, d, r)
        g = self.n_q // self.n_kv
        B_q_mid = B_mid.unsqueeze(1).expand(-1, g, -1, -1).reshape(self.n_q, self.head_dim, B_mid.shape[-1])

        def eps_q_hook(mod, inp, out):
            B, T, D = out.shape
            if D != self.n_q * self.head_dim:
                return
            Q = out.view(B, T, self.n_q, self.head_dim).float()
            q_last = Q[:, -1]  # (B, n_q, d)
            proj = torch.einsum("bnd,ndr->bnr", q_last, B_q_mid.to(Q.device))
            energy = (proj ** 2).sum(-1)
            q_norm_sq = (q_last ** 2).sum(-1) + 1e-8
            eps = (energy / q_norm_sq).mean().item()
            eps_q_capture.append(eps)

        # Register eps_q hook first on mid layer (sees unperturbed Q)
        self.handles.append(
            self.model.model.layers[mid_l].self_attn.q_proj.register_forward_hook(eps_q_hook)
        )
        self._eps_q_capture = eps_q_capture

        # Build per-layer schedule
        schedule = make_layer_schedule(L, self.alpha, self.beta, self.layer_mode)

        for layer_idx in range(min(L, len(self.model.model.layers))):
            layer = self.model.model.layers[layer_idx]
            alpha_l, beta_l = schedule[layer_idx]

            def make_q_hook(li, _beta_l=beta_l):
                def hook(mod, inp, out):
                    B, T, D = out.shape
                    if D != self.n_q * self.head_dim:
                        return out
                    if _beta_l == 0.0:
                        return out
                    Q = out.view(B, T, self.n_q, self.head_dim).float()
                    if self.P_emitted[li].abs().sum() > 0:
                        Q_emitted = torch.einsum("btnd,nde->btne", Q, self.P_emitted[li])
                        Q = Q + _beta_l * Q_emitted
                    return Q.to(out.dtype).view(B, T, D)
                return hook

            def make_k_hook(li, _alpha_l=alpha_l):
                def hook(mod, inp, out):
                    B, T, D = out.shape
                    if D != self.n_kv * self.head_dim:
                        return out
                    if _alpha_l == 0.0:
                        return out
                    K = out.view(B, T, self.n_kv, self.head_dim).float()
                    g = self.n_q // self.n_kv
                    P_remaining = self.P_ont_full[li] - self.P_emitted[li]
                    P_kv = P_remaining.view(self.n_kv, g, self.head_dim, self.head_dim).mean(dim=1)
                    K_boost = torch.einsum("btnd,nde->btne", K, P_kv)
                    K = K + _alpha_l * K_boost
                    return K.to(out.dtype).view(B, T, D)
                return hook

            # Q-coverage hook registered AFTER eps_q hook on mid layer
            if beta_l != 0.0:
                self.handles.append(layer.self_attn.q_proj.register_forward_hook(make_q_hook(layer_idx)))
            if alpha_l != 0.0:
                self.handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))

    def get_last_eps_q(self):
        if self._eps_q_capture:
            self.last_eps_q = self._eps_q_capture[-1]
        return self.last_eps_q

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def extract_first_tool(text, candidates):
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        name = m.group(1)
        for c in candidates:
            if c.lower() == name.lower():
                return c
    return None


def extract_all_tools(text, candidates):
    """Extract ALL tool names from text (not just the first)."""
    found = []
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        name = m.group(1)
        for c in candidates:
            if c.lower() == name.lower() and c not in found:
                found.append(c)
                break
    return found


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

    print(f"[facet] Building projectors...", flush=True)
    facet_projectors = build_facet_projectors(B_ont, r_per_pair, facet_order, n_q, n_kv, head_dim, args.device)
    print(f"[facet] {len(facet_projectors)} facets", flush=True)

    hooks = IterativeDynamicHooks(model, facet_projectors, facet_order, B_ont,
                                  args.alpha, args.beta, n_q, n_kv, head_dim, args.device,
                                  layer_mode=args.layer_mode)

    # Stop token: </tool_call> forces 1 tool per generation step
    tool_call_end_id = tok.convert_tokens_to_ids("</tool_call>")
    if tool_call_end_id is None or tool_call_end_id == tok.unk_token_id:
        # Fallback: search by encoding
        tool_call_end_id = tok.encode("</tool_call>", add_special_tokens=False)
        tool_call_end_id = tool_call_end_id[0] if tool_call_end_id else None
    stop_token_ids = [tok.eos_token_id]
    if args.stop_at_tool and tool_call_end_id:
        stop_token_ids.append(tool_call_end_id)
        print(f"[stop] </tool_call> token id = {tool_call_end_id}, stop_at_tool=True", flush=True)
    elif args.stop_at_tool:
        print(f"[warn] </tool_call> token not found, stop_at_tool disabled", flush=True)

    data = json.load(open(args.dataset))[:args.max_samples]
    print(f"[data] N={len(data)}", flush=True)

    configs = [
        ("no_steer", False, 0.0, 0.0),
        ("static_q", False, 0.0, args.beta),
        ("iterative_q", True, 0.0, args.beta),
        ("iterative_kq", True, args.alpha, args.beta),
    ]

    all_results = []
    for config_name, iterative, alpha, beta in configs:
        hooks.alpha = alpha
        hooks.beta = beta
        print(f"\n[eval] {config_name} (α={alpha}, β={beta}, iterative={iterative})", flush=True)

        agg = {"F1": 0, "Exact": 0}
        per_sample = []
        t0 = time.time()

        for i, entry in enumerate(data):
            gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
            cands = parse_candidates(entry["action_prompt"])
            fc = build_fc_prompt(tok, entry["action_prompt"], cands)

            if not iterative:
                # Static: single generation
                hooks.reset()
                if alpha != 0 or beta != 0:
                    hooks.install()
                with torch.no_grad():
                    ids = tok(fc, return_tensors="pt")["input_ids"].to(args.device)
                    out = model.generate(ids, max_new_tokens=args.max_new_tokens_per_tool * 3,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                hooks.remove()
                pred = extract_tool_names(gen, cands)
            else:
                # Iterative: extract tools from each generation step.
                # Bug fix: use extract_all_tools instead of extract_first_tool,
                # because the model may emit multiple tool_call blocks in a
                # single generation step.
                hooks.reset()
                emitted_tools = []
                accumulated_gen = ""
                prompt_so_far = fc

                for tool_step in range(args.max_tools):
                    hooks._eps_q_capture = []
                    hooks.install()

                    with torch.no_grad():
                        ids = tok(prompt_so_far + accumulated_gen, return_tensors="pt")["input_ids"].to(args.device)
                        out = model.generate(ids, max_new_tokens=args.max_new_tokens_per_tool,
                                             do_sample=False, pad_token_id=tok.eos_token_id)

                    step_gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                    hooks.remove()

                    # Parse ALL new tools in this step (not just first)
                    new_tools = extract_all_tools(step_gen, cands)
                    new_tools = [t for t in new_tools if t not in emitted_tools]
                    accumulated_gen += step_gen

                    if new_tools:
                        for tool in new_tools:
                            emitted_tools.append(tool)
                            hooks.add_emitted_tool(tool)

                        eps_q = hooks.get_last_eps_q()
                        if eps_q < args.eps_threshold:
                            break
                    else:
                        break

                pred = emitted_tools

            m = compute_metrics(pred, gt)
            agg["F1"] += m["F1"]
            agg["Exact"] += m["Exact"]
            per_sample.append({"gt": gt, "pred": pred, "F1": m["F1"]})

            if i % 10 == 0:
                print(f"  [{i}/{len(data)}] gt={gt} pred={pred} F1={m['F1']:.3f}", flush=True)

        N = len(data)
        macro = {k: v / N for k, v in agg.items()}
        runtime = time.time() - t0
        print(f"  → {config_name}: F1={macro['F1']:.4f} Exact={macro['Exact']:.4f} ({runtime:.1f}s)", flush=True)

        all_results.append({
            "config": config_name, "alpha": alpha, "beta": beta,
            "iterative": iterative, "eps_threshold": args.eps_threshold,
            "n": N, "macro": macro, "runtime_s": runtime, "per_sample": per_sample,
        })

    print(f"\n{'='*60}", flush=True)
    print(f"{'Config':25s} {'F1':>8s} {'Exact':>8s} {'Δ F1':>8s}", flush=True)
    print(f"{'-'*60}", flush=True)
    base_f1 = all_results[0]["macro"]["F1"]
    for r in all_results:
        delta = r["macro"]["F1"] - base_f1
        print(f"{r['config']:25s} {r['macro']['F1']:8.4f} {r['macro']['Exact']:8.4f} {delta:+8.4f}", flush=True)

    blob = {"model": args.model, "configs": all_results}
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
