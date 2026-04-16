#!/usr/bin/env python3
"""eval_subtask4_facet_adaptive_v4.py — Multi-turn Facet-Narrowing Q-Coverage + K-Rotation.

3-level hierarchy:
  Level 1 (Conversation): cross-turn EMA — abstract → specific scope convergence
  Level 2 (Turn/within-turn): facet decay after each tool emission
  Level 3 (Token): Q-coverage + K-boost hooks per generation step

Algorithm per turn:
  1. Measure per-facet ε_f from current turn's query Q
  2. Blend with conversation-level facet memory (EMA)
  3. Build query-adaptive weighted projector P = Σ w_f · P_f
  4. Generate tools iteratively with facet decay after each emission
  5. Update conversation memory with decayed weights

Multi-turn simulation: MetaTool Subtask4 queries are paired into
synthetic 2-turn conversations where the second turn adds a
complementary tool need.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, random
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

ONTOLOGY_SUMMARY = {
    "function_action": ['translate', 'convert', 'analyze', 'summarize', 'recommend',
                        'book', 'track', 'create', 'search', 'retrieve', 'play', 'inform'],
    "io_type": ['video', 'image', 'audio', 'structured', 'numeric', 'text'],
    "domain": ['finance', 'news', 'health', 'education', 'travel', 'shopping',
               'entertainment', 'food', 'real_estate', 'weather', 'productivity',
               'research', 'legal', 'career', 'utility'],
    "tool_category": [],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--b-ont", required=True)
    p.add_argument("--dataset", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.03)
    p.add_argument("--beta", type=float, default=-0.03)
    p.add_argument("--decay", type=float, default=0.2)
    p.add_argument("--blend-gamma", type=float, default=0.6,
                   help="EMA blend: gamma*new + (1-gamma)*prev")
    p.add_argument("--eps-threshold", type=float, default=0.03)
    p.add_argument("--max-tools-per-turn", type=int, default=3)
    p.add_argument("--max-new-tokens-per-step", type=int, default=100)
    p.add_argument("--measure-layer", type=int, default=13)
    p.add_argument("--out", required=True)
    return p.parse_args()


class FacetAdaptiveEngine:
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
        self.B_ont = B_ont
        self.L = B_ont.shape[0]
        self.R = B_ont.shape[3]

        self.facet_ranges = {}
        for l in range(self.L):
            for h_kv in range(n_kv):
                rp_key = f"L{l}_H{h_kv}"
                ranks = r_per_pair.get(rp_key, [self.R // self.n_facets] * self.n_facets)
                ranges = []
                offset = 0
                for f_idx in range(self.n_facets):
                    r_f = min(ranks[f_idx] if f_idx < len(ranks) else 0, self.R - offset)
                    ranges.append((offset, offset + max(r_f, 0)))
                    offset += max(r_f, 0)
                self.facet_ranges[(l, h_kv)] = ranges

        self.B_q = []
        for l in range(self.L):
            B_l = B_ont[l].unsqueeze(1).expand(-1, self.g, -1, -1).reshape(n_q, head_dim, self.R)
            self.B_q.append(B_l.to(device))

        # Conversation-level state
        self.conv_weights = None
        self.handles = []

    def reset_conversation(self):
        self.conv_weights = None

    def measure_facet_weights(self, prompt_ids):
        eps_per_facet = torch.zeros(self.n_facets, device=self.device)
        captured = {}

        def capture_hook(mod, inp, out):
            captured['q'] = out.detach()

        h = self.model.model.layers[self.measure_layer].self_attn.q_proj.register_forward_hook(capture_hook)
        with torch.no_grad():
            self.model(prompt_ids)
        h.remove()

        Q = captured['q']
        B, T, D = Q.shape
        Q = Q.view(B, T, self.n_q, self.head_dim).float()
        q_last = Q[0, -1]

        B_layer = self.B_q[self.measure_layer]
        for h_kv in range(self.n_kv):
            ranges = self.facet_ranges[(self.measure_layer, h_kv)]
            for f_idx, (lo, hi) in enumerate(ranges):
                if hi <= lo:
                    continue
                for q_h in range(h_kv * self.g, (h_kv + 1) * self.g):
                    B_f = B_layer[q_h, :, lo:hi]
                    q_h_vec = q_last[q_h]
                    proj = B_f.T @ q_h_vec
                    energy = (proj ** 2).sum()
                    q_norm_sq = (q_h_vec ** 2).sum() + 1e-8
                    eps_per_facet[f_idx] += (energy / q_norm_sq).item()

        eps_per_facet /= self.n_q
        total = eps_per_facet.sum() + 1e-8
        return eps_per_facet / total

    def new_turn(self, prompt_ids, blend_gamma=0.6):
        measured = self.measure_facet_weights(prompt_ids)
        if self.conv_weights is None:
            self.conv_weights = measured.clone()
        else:
            self.conv_weights = blend_gamma * measured + (1 - blend_gamma) * self.conv_weights
            self.conv_weights /= self.conv_weights.sum() + 1e-8
        return self.conv_weights.clone()

    def end_turn(self, emitted_tools, decay):
        for tool in emitted_tools:
            self.conv_weights = self.decay_weights(self.conv_weights, tool, decay)

    def decay_weights(self, weights, tool_name, decay_factor):
        facet_vals = TOOL_FACET_MAP.get(tool_name, {})
        new_w = weights.clone()
        for facet_name in facet_vals:
            if facet_name in self.facet_order:
                f_idx = self.facet_order.index(facet_name)
                new_w[f_idx] *= decay_factor
        total = new_w.sum() + 1e-8
        return new_w / total

    def build_weighted_projector(self, weights, layer_idx):
        d = self.head_dim
        P = torch.zeros(self.n_q, d, d, device=self.device)
        for h_kv in range(self.n_kv):
            ranges = self.facet_ranges[(layer_idx, h_kv)]
            B_l = self.B_ont[layer_idx, h_kv].to(self.device)
            for f_idx, (lo, hi) in enumerate(ranges):
                if hi <= lo or f_idx >= len(weights):
                    continue
                B_f = B_l[:, lo:hi]
                P_f = B_f @ B_f.T
                w = weights[f_idx].item()
                for q_h in range(h_kv * self.g, (h_kv + 1) * self.g):
                    P[q_h] += w * P_f
        return P

    def install_hooks(self, weights, alpha, beta):
        self.remove_hooks()
        for layer_idx in range(min(self.L, len(self.model.model.layers))):
            P_w = self.build_weighted_projector(weights, layer_idx)
            P_kv = P_w.view(self.n_kv, self.g, self.head_dim, self.head_dim).mean(dim=1)
            layer = self.model.model.layers[layer_idx]

            def make_q_hook(P_l, _b=beta):
                def hook(mod, inp, out):
                    B, T, D = out.shape
                    if D != self.n_q * self.head_dim:
                        return out
                    Q = out.view(B, T, self.n_q, self.head_dim).float()
                    Q = Q + _b * torch.einsum("btnd,nde->btne", Q, P_l)
                    return Q.to(out.dtype).view(B, T, D)
                return hook

            def make_k_hook(P_kv_l, _a=alpha):
                def hook(mod, inp, out):
                    B, T, D = out.shape
                    if D != self.n_kv * self.head_dim:
                        return out
                    K = out.view(B, T, self.n_kv, self.head_dim).float()
                    K = K + _a * torch.einsum("btnd,nde->btne", K, P_kv_l)
                    return K.to(out.dtype).view(B, T, D)
                return hook

            self.handles.append(layer.self_attn.q_proj.register_forward_hook(make_q_hook(P_w)))
            if alpha != 0:
                self.handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(P_kv)))

    def remove_hooks(self):
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


def build_multi_turn_conversations(data, max_convs=50):
    """Pair Subtask4 entries into synthetic 2-turn conversations.
    Turn 1: abstract request mentioning first tool's domain
    Turn 2: specific request adding second tool's need
    GT: both tools from the original Subtask4 entry."""
    random.seed(42)
    convs = []
    for entry in data[:max_convs]:
        gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
        if len(gt) < 2:
            continue
        cands = parse_candidates(entry.get("action_prompt", ""))
        action_prompt = entry.get("action_prompt", "")

        tool1_facets = TOOL_FACET_MAP.get(gt[0], {})
        tool2_facets = TOOL_FACET_MAP.get(gt[1], {})
        domain1 = tool1_facets.get("domain", "general")
        domain2 = tool2_facets.get("domain", "general")

        turn1_query = f"I need help with something related to {domain1}. Can you help me find the right tools?"
        turn2_query = action_prompt

        convs.append({
            "turns": [
                {"query": turn1_query, "expected_tools": [gt[0]], "abstract": True},
                {"query": turn2_query, "expected_tools": gt, "abstract": False},
            ],
            "all_gt": gt,
            "candidates": cands,
            "original_prompt": action_prompt,
        })
    return convs


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

    engine = FacetAdaptiveEngine(model, B_ont, r_per_pair, facet_order,
                                 n_q, n_kv, head_dim, args.device, args.measure_layer)
    print(f"[engine] Ready ({len(facet_order)} facets)", flush=True)

    data = json.load(open(args.dataset))[:args.max_samples]
    print(f"[data] N={len(data)}", flush=True)

    # ============================================================
    # Experiment A: Single-turn (same as v3 but with v4 engine)
    # ============================================================
    print(f"\n{'='*60}\n[A] Single-turn evaluation\n{'='*60}", flush=True)

    single_configs = [
        ("no_steer", False, 0.0, 0.0),
        ("static_uniform_q", False, 0.0, args.beta),
        ("adaptive_q", True, 0.0, args.beta),
        ("adaptive_kq", True, args.alpha, args.beta),
    ]

    single_results = []
    for cname, adaptive, alpha, beta in single_configs:
        print(f"\n[A/{cname}] α={alpha}, β={beta}, adaptive={adaptive}", flush=True)
        agg = {"F1": 0, "Exact": 0}
        per_sample = []
        t0 = time.time()

        for i, entry in enumerate(data):
            gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
            cands = parse_candidates(entry["action_prompt"])
            fc = build_fc_prompt(tok, entry["action_prompt"], cands)
            ids = tok(fc, return_tensors="pt")["input_ids"].to(args.device)

            if not adaptive:
                if beta != 0 or alpha != 0:
                    uw = torch.ones(engine.n_facets, device=args.device) / engine.n_facets
                    engine.install_hooks(uw, alpha, beta)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=args.max_new_tokens_per_step * 3,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                engine.remove_hooks()
                pred = extract_tool_names(gen, cands)
            else:
                engine.reset_conversation()
                weights = engine.new_turn(ids, args.blend_gamma)
                emitted = []
                acc_text = ""

                for step in range(args.max_tools_per_turn):
                    engine.install_hooks(weights, alpha, beta)
                    p_ids = tok(fc + acc_text, return_tensors="pt")["input_ids"].to(args.device)
                    with torch.no_grad():
                        out = model.generate(p_ids, max_new_tokens=args.max_new_tokens_per_step,
                                             do_sample=False, pad_token_id=tok.eos_token_id)
                    step_text = tok.decode(out[0][p_ids.shape[1]:], skip_special_tokens=True)
                    engine.remove_hooks()
                    acc_text += step_text

                    tool = extract_first_tool(step_text, cands)
                    if tool and tool not in emitted:
                        emitted.append(tool)
                        weights = engine.decay_weights(weights, tool, args.decay)
                        if weights.max().item() < args.eps_threshold:
                            break
                    else:
                        break

                engine.end_turn(emitted, args.decay)
                pred = emitted

            m = compute_metrics(pred, gt)
            agg["F1"] += m["F1"]; agg["Exact"] += m["Exact"]
            per_sample.append({"gt": gt, "pred": pred, "F1": m["F1"]})
            if i % 10 == 0:
                print(f"  [{i}/{len(data)}] gt={gt} pred={pred} F1={m['F1']:.3f}", flush=True)

        N = len(data)
        macro = {k: v/N for k, v in agg.items()}
        print(f"  → {cname}: F1={macro['F1']:.4f} Exact={macro['Exact']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        single_results.append({"config": cname, "macro": macro, "per_sample": per_sample})

    # ============================================================
    # Experiment B: Multi-turn (synthetic 2-turn conversations)
    # ============================================================
    print(f"\n{'='*60}\n[B] Multi-turn evaluation (synthetic 2-turn)\n{'='*60}", flush=True)

    convs = build_multi_turn_conversations(data, args.max_samples)
    print(f"[B] {len(convs)} conversations built", flush=True)

    multi_configs = [
        ("no_steer_multi", False, 0.0, 0.0),
        ("adaptive_kq_multi", True, args.alpha, args.beta),
    ]

    multi_results = []
    for cname, adaptive, alpha, beta in multi_configs:
        print(f"\n[B/{cname}]", flush=True)
        agg = {"turn1_F1": 0, "turn2_F1": 0, "conv_F1": 0, "conv_Exact": 0}
        per_conv = []
        t0 = time.time()

        for ci, conv in enumerate(convs):
            engine.reset_conversation()
            all_emitted = []
            turn_details = []

            for turn_idx, turn in enumerate(conv["turns"]):
                cands = conv["candidates"]
                query = turn["query"]

                if turn["abstract"]:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant. Select tools."},
                        {"role": "user", "content": query},
                    ]
                    fc = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    fc = build_fc_prompt(tok, query, cands)

                ids = tok(fc, return_tensors="pt")["input_ids"].to(args.device)

                if not adaptive:
                    with torch.no_grad():
                        out = model.generate(ids, max_new_tokens=args.max_new_tokens_per_step * 2,
                                             do_sample=False, pad_token_id=tok.eos_token_id)
                    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                    turn_pred = extract_tool_names(gen, cands)
                else:
                    weights = engine.new_turn(ids, args.blend_gamma)
                    turn_emitted = []
                    acc = ""

                    for step in range(args.max_tools_per_turn):
                        engine.install_hooks(weights, alpha, beta)
                        p_ids = tok(fc + acc, return_tensors="pt")["input_ids"].to(args.device)
                        with torch.no_grad():
                            out = model.generate(p_ids, max_new_tokens=args.max_new_tokens_per_step,
                                                 do_sample=False, pad_token_id=tok.eos_token_id)
                        step_text = tok.decode(out[0][p_ids.shape[1]:], skip_special_tokens=True)
                        engine.remove_hooks()
                        acc += step_text

                        tool = extract_first_tool(step_text, cands)
                        if tool and tool not in turn_emitted and tool not in all_emitted:
                            turn_emitted.append(tool)
                            weights = engine.decay_weights(weights, tool, args.decay)
                            if weights.max().item() < args.eps_threshold:
                                break
                        else:
                            break

                    engine.end_turn(turn_emitted, args.decay)
                    turn_pred = turn_emitted

                all_emitted.extend([t for t in turn_pred if t not in all_emitted])
                turn_m = compute_metrics(turn_pred, turn["expected_tools"])
                turn_details.append({
                    "turn": turn_idx, "query_head": query[:100],
                    "pred": turn_pred, "gt": turn["expected_tools"],
                    "F1": turn_m["F1"], "weights": weights.tolist() if adaptive else None,
                })

                if turn_idx == 0:
                    agg["turn1_F1"] += turn_m["F1"]
                elif turn_idx == 1:
                    agg["turn2_F1"] += turn_m["F1"]

            conv_m = compute_metrics(all_emitted, conv["all_gt"])
            agg["conv_F1"] += conv_m["F1"]
            agg["conv_Exact"] += conv_m["Exact"]
            per_conv.append({"all_gt": conv["all_gt"], "all_pred": all_emitted,
                             "conv_F1": conv_m["F1"], "turns": turn_details})

            if ci % 10 == 0:
                print(f"  [{ci}/{len(convs)}] gt={conv['all_gt']} pred={all_emitted} "
                      f"conv_F1={conv_m['F1']:.3f}", flush=True)

        N = len(convs)
        macro = {k: v/max(N,1) for k, v in agg.items()}
        print(f"  → {cname}: conv_F1={macro['conv_F1']:.4f} turn1={macro['turn1_F1']:.4f} "
              f"turn2={macro['turn2_F1']:.4f} ({time.time()-t0:.1f}s)", flush=True)
        multi_results.append({"config": cname, "macro": macro, "per_conv": per_conv})

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print(f"[A] Single-turn summary:", flush=True)
    base = single_results[0]["macro"]["F1"]
    for r in single_results:
        d = r["macro"]["F1"] - base
        print(f"  {r['config']:25s} F1={r['macro']['F1']:.4f} Δ={d:+.4f}", flush=True)

    print(f"\n[B] Multi-turn summary:", flush=True)
    for r in multi_results:
        print(f"  {r['config']:25s} conv_F1={r['macro']['conv_F1']:.4f} "
              f"t1={r['macro']['turn1_F1']:.4f} t2={r['macro']['turn2_F1']:.4f}", flush=True)

    blob = {
        "model": args.model, "params": vars(args),
        "single_turn": single_results, "multi_turn": multi_results,
    }
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
