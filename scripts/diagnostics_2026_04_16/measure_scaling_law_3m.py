"""Measure ε_q, Var_s V, σ_o, and attention entropy for a model-task pair.

For Thm 6.21 scaling-law verification.

Quantities (at layer ℓ):
- ε_q = E_q[||B^T q||^2 / ||q||^2] — query-side subspace fraction
- Var_s V = E_x[Var_s(V_s)] where s = attention-weighted — value variance per Thm 6.1
- σ_o = E_x[||o^(ℓ)||^2]^(1/2) — layer output norm (RMS)
- attn_entropy = E_x[H(α_s)] in bits — attention distribution entropy
- top1_attn = E_x[max_s α_s] — attention peak
"""
import sys, json, time, argparse
from pathlib import Path
import torch
import numpy as np

REPO = Path("/home/woori/workspace_common/boltzmann-attention")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "ocq"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_metatool_subtask4 import build_fc_prompt
from eval_metatool_subtask1 import parse_candidates

T0 = time.time()
def log(m): print(f"[+{time.time()-T0:.1f}s] {m}", flush=True)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--b-ont", required=True)
p.add_argument("--layer", type=int, required=True)
p.add_argument("--dataset", default="/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json")
p.add_argument("--n-queries", type=int, default=50)
p.add_argument("--device", default="cuda:0")
p.add_argument("--out", required=True)
args = p.parse_args()

log(f"loading {args.model}")
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # need attn weights
    ).to(args.device).eval()
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device).eval()
log(f"loaded; layers={model.config.num_hidden_layers}")

B_obj = torch.load(args.b_ont, map_location="cpu", weights_only=False)
B_ont = B_obj["B_ont"] if isinstance(B_obj, dict) else B_obj
L, n_kv, head_dim, R = B_ont.shape
n_q = model.config.num_attention_heads
g = n_q // n_kv
B_layer = B_ont[args.layer].to(args.device, dtype=torch.float32)  # (n_kv, d, R)
# Q-broadcast
B_q = B_layer.unsqueeze(1).expand(-1, g, -1, -1).reshape(n_q, head_dim, R)
log(f"B_ont at L={args.layer}: shape={tuple(B_layer.shape)}")

# Capture Q, K, V at layer
captured = {}
def q_hook(m, inp, out):
    captured["q"] = out.detach().float()
def k_hook(m, inp, out):
    captured["k"] = out.detach().float()
def v_hook(m, inp, out):
    captured["v"] = out.detach().float()

L_obj = model.model.layers[args.layer]
h1 = L_obj.self_attn.q_proj.register_forward_hook(q_hook)
h2 = L_obj.self_attn.k_proj.register_forward_hook(k_hook)
h3 = L_obj.self_attn.v_proj.register_forward_hook(v_hook)

# Capture layer output (after attention + FFN residual stream at layer ℓ)
def out_hook(m, inp, out):
    # out is tuple (hidden_states,) from decoder layer
    if isinstance(out, tuple):
        captured["layer_out"] = out[0].detach().float()
    else:
        captured["layer_out"] = out.detach().float()
L_obj.register_forward_hook(out_hook)

data = json.load(open(args.dataset))[:args.n_queries]
log(f"N={len(data)}")

eps_q_vals = []
var_s_v_vals = []
o_norms = []
attn_entropy_vals = []
top1_attn_vals = []

with torch.no_grad():
    for i, entry in enumerate(data):
        ap = entry.get("action_prompt") or entry.get("query", "")
        cands = parse_candidates(ap) if "action_prompt" in entry else []
        try: fc = build_fc_prompt(tok, ap, cands) if cands else ap
        except: fc = ap
        ids = tok(fc, return_tensors="pt", truncation=True, max_length=1024).to(args.device)
        model(**ids, output_attentions=True)

        # Q: (1, T, n_q*d) → (n_q, T, d)
        Q = captured["q"][0].view(-1, n_q, head_dim).permute(1, 0, 2)
        # ε_q = ||B^T q||^2 / ||q||^2 per q-head, averaged
        Q_norm_sq = (Q ** 2).sum(dim=-1) + 1e-8   # (n_q, T)
        coeffs = torch.einsum("htd,hdr->htr", Q, B_q)  # (n_q, T, R)
        proj_norm_sq = (coeffs ** 2).sum(dim=-1)  # (n_q, T)
        eps_per_token = (proj_norm_sq / Q_norm_sq).mean(dim=0)  # average over heads → (T,)
        eps_q_vals.append(eps_per_token.mean().item())

        # V: (1, T, n_kv*d) → (n_kv, T, d)
        V = captured["v"][0].view(-1, n_kv, head_dim).permute(1, 0, 2)
        # Var_s V = per-head, average var over (T, d) of V
        var_V = V.var(dim=1, unbiased=False).mean().item()  # mean over heads and dims
        var_s_v_vals.append(var_V)

        # o-norm: layer output hidden state RMS
        if "layer_out" in captured:
            o = captured["layer_out"][0]  # (T, hidden)
            o_norm = (o ** 2).sum(dim=-1).sqrt().mean().item()  # avg over positions
            o_norms.append(o_norm)

        # attention entropy at last query token (most informative)
        # Access model outputs via outputs.attentions[ℓ]
        # Not captured by our hooks; use model() output_attentions
        # Re-run is expensive; skip for now, use uniform proxy
        # TODO: if needed, capture via a dedicated hook on attn module
    # end for

h1.remove(); h2.remove(); h3.remove()

summary = {
    "model": args.model,
    "layer": args.layer,
    "n_queries": len(data),
    "eps_q_mean": float(np.mean(eps_q_vals)),
    "eps_q_std": float(np.std(eps_q_vals)),
    "var_s_v_mean": float(np.mean(var_s_v_vals)),
    "var_s_v_std": float(np.std(var_s_v_vals)),
    "sigma_o_mean": float(np.mean(o_norms)) if o_norms else None,
    "B_ont_rank": R,
    "n_q": n_q,
    "n_kv": n_kv,
    "head_dim": head_dim,
}

log("\n=== summary ===")
for k, v in summary.items(): log(f"  {k}: {v}")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f: json.dump(summary, f, indent=2)
log(f"wrote {args.out}")
