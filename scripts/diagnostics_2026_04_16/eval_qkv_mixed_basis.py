"""Mixed-basis QKV sweep for Thm 6.17 V-inert re-examination.

Uses SEPARATE B_ont for K/Q (K-space basis) vs V (V-space basis).
Evaluates Subtask4 full N=497 with methods:
- no_steer
- V-only (V-basis) at α_V ∈ {0.05, 0.1}
- Q-only at β = -0.1
- V+Q (V-basis for V, K-basis for Q)
- Q+K at α_K=0.025, β=-0.1  (reference from prior paper, K-basis for K)
- Trio: K-basis for K, V-basis for V, K-basis for Q (true Thm 6.17 with proper bases)
"""
import sys, json, time, argparse
from pathlib import Path
from contextlib import ExitStack
import torch
import numpy as np

REPO = Path("/home/woori/workspace_common/boltzmann-attention")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_metatool_subtask1 import (
    install_kbias_hooks, install_vbias_hooks, install_q_bias_hooks,
    parse_candidates,
)
from eval_metatool_subtask4 import (
    build_fc_prompt, extract_tool_names, compute_metrics,
)

T0 = time.time()
def log(m): print(f"[+{time.time()-T0:.1f}s] {m}", flush=True)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--b-ont-k", required=True, help="K-space B_ont path")
p.add_argument("--b-ont-v", required=True, help="V-space B_ont path")
p.add_argument("--device", default="cuda:0")
p.add_argument("--dataset", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask4.json")
p.add_argument("--n-queries", type=int, default=497)
p.add_argument("--max-new-tokens", type=int, default=256)
p.add_argument("--out", required=True)
args = p.parse_args()

log(f"loading {args.model}")
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device).eval()
log("loaded")

def load_bont(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    return o["B_ont"] if isinstance(o, dict) else o

B_k = load_bont(args.b_ont_k)
B_v = load_bont(args.b_ont_v)
log(f"B_k shape {tuple(B_k.shape)}, B_v shape {tuple(B_v.shape)}")

# Align ranks if different — pad smaller with zero columns
L_k, H_k, d_k, R_k = B_k.shape
L_v, H_v, d_v, R_v = B_v.shape
assert (L_k, H_k, d_k) == (L_v, H_v, d_v), "B_k/B_v dims must match except R"

n_kv = model.config.num_key_value_heads
n_q = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // n_q)

data = json.load(open(args.dataset))
if args.n_queries > 0:
    data = data[:args.n_queries]
log(f"N={len(data)} queries")

def run_eval(method_name, hooks_builder):
    """hooks_builder: callable that takes ExitStack and enters all hook contexts."""
    agg = {k: 0.0 for k in ["F1", "F_0.5", "EU", "Jaccard", "Exact", "precision", "recall"]}
    per_sample = []
    t0 = time.time()
    with ExitStack() as stack:
        hooks_builder(stack)
        for i, entry in enumerate(data):
            ap = entry["action_prompt"]
            gt = entry["tool"] if isinstance(entry["tool"], list) else [entry["tool"]]
            cands = parse_candidates(ap)
            fc = build_fc_prompt(tok, ap, cands)
            ids = tok(fc, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
            pred = extract_tool_names(gen, cands)
            m = compute_metrics(pred, gt)
            for k in agg: agg[k] += m[k]
            per_sample.append({"gt": gt, "pred": pred, "metrics": m,
                               "gen_head": gen[:200]})
            if i % 50 == 0:
                log(f"  [{method_name} {i}/{len(data)}] F1={m['F1']:.3f}")
    for k in agg: agg[k] /= max(len(data), 1)
    dt = time.time() - t0
    log(f"  {method_name}: F1={agg['F1']:.4f} Exact={agg['Exact']:.4f} runtime={dt:.0f}s")
    return {"method": method_name, "macro": agg, "per_sample": per_sample, "runtime_s": dt}

# Method list
method_specs = [
    ("no_steer", lambda s: None),
    ("vbias_Vbasis_a0.05",
        lambda s: s.enter_context(install_vbias_hooks(model, B_v, alpha=0.05, n_kv=n_kv, head_dim=head_dim))),
    ("vbias_Vbasis_a0.1",
        lambda s: s.enter_context(install_vbias_hooks(model, B_v, alpha=0.1, n_kv=n_kv, head_dim=head_dim))),
    ("qbias_Kbasis_b-0.1",
        lambda s: s.enter_context(install_q_bias_hooks(model, B_k, beta=-0.1, n_kv=n_kv, n_q=n_q, head_dim=head_dim))),
    ("VQ_Vbasis_V_Kbasis_Q_a0.05_b-0.1",
        lambda s: (
            s.enter_context(install_vbias_hooks(model, B_v, alpha=0.05, n_kv=n_kv, head_dim=head_dim)),
            s.enter_context(install_q_bias_hooks(model, B_k, beta=-0.1, n_kv=n_kv, n_q=n_q, head_dim=head_dim)),
        )),
    ("VQ_Vbasis_V_Kbasis_Q_a0.1_b-0.1",
        lambda s: (
            s.enter_context(install_vbias_hooks(model, B_v, alpha=0.1, n_kv=n_kv, head_dim=head_dim)),
            s.enter_context(install_q_bias_hooks(model, B_k, beta=-0.1, n_kv=n_kv, n_q=n_q, head_dim=head_dim)),
        )),
    ("trio_Kbasis_K_Vbasis_V_Kbasis_Q_aK0.025_aV0.05_bQ-0.1",
        lambda s: (
            s.enter_context(install_kbias_hooks(model, B_k, alpha=0.025, n_kv=n_kv, head_dim=head_dim)),
            s.enter_context(install_vbias_hooks(model, B_v, alpha=0.05, n_kv=n_kv, head_dim=head_dim)),
            s.enter_context(install_q_bias_hooks(model, B_k, beta=-0.1, n_kv=n_kv, n_q=n_q, head_dim=head_dim)),
        )),
]

results = []
for name, builder in method_specs:
    log(f"--- {name} ---")
    results.append(run_eval(name, builder))

blob = {"model": args.model, "b_ont_k": args.b_ont_k, "b_ont_v": args.b_ont_v,
        "n_queries": len(data), "results": results}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f: json.dump(blob, f, indent=2)
log(f"wrote {args.out}")
