"""Quick A/B: V-bias with K-basis vs V-basis on single-token log-p.

For 30 Subtask1 queries, measure:
  Δlog_p(correct tool first token | prompt + JSON_PREFIX) under
  (a) no steer, (b) V-bias α=0.05 with K-basis, (c) V-bias α=0.05 with V-basis,
  (d) V-bias α=0.05 with RANDOM orthonormal basis of same shape as V-basis.

Interpretation:
  - If (c) causes much larger |Δlog_p| than (b) → V-basis captures real V-facet
    direction that K-basis misses → V-inert result is implementation artifact.
  - If (d) random gives same |Δlog_p| as (b) K-basis → K-basis-on-V is
    indistinguishable from random for V-purposes → V-inert result genuinely
    meaningless on K-basis.
  - If (c) V-basis and (d) random give similar |Δlog_p| → V-basis also lacks
    signal; Thm 6.17 V-inert is a fundamental V-channel property (not
    basis-selection artifact). This would rescue the current paper claim.
"""
import sys, json, time, argparse
from pathlib import Path
import torch
import numpy as np

REPO = Path("/home/woori/workspace_common/boltzmann-attention")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_metatool_subtask1 import install_vbias_hooks, parse_candidates
from eval_metatool_subtask4 import build_fc_prompt

T0 = time.time()
def log(m): print(f"[+{time.time()-T0:.1f}s] {m}", flush=True)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--b-ont-k", required=True)
p.add_argument("--b-ont-v", required=True)
p.add_argument("--alpha", type=float, default=0.05)
p.add_argument("--device", default="cuda:1")
p.add_argument("--n-queries", type=int, default=30)
p.add_argument("--out", required=True)
args = p.parse_args()

JSON_PREFIX = '<tool_call>{"name": "'

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

def load_b(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    return o["B_ont"] if isinstance(o, dict) else o

B_k = load_b(args.b_ont_k)
B_v = load_b(args.b_ont_v)
L, H, d, R_k = B_k.shape
_, _, _, R_v = B_v.shape
# Random orthonormal of same shape as V-basis (matched rank for fair compare)
torch.manual_seed(7)
B_r = torch.zeros_like(B_v)
for l in range(L):
    for h in range(H):
        M = torch.randn(d, R_v)
        Q, _ = torch.linalg.qr(M)
        B_r[l, h] = Q[:, :R_v]

n_kv = model.config.num_key_value_heads
head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // model.config.num_attention_heads)

# Prepare samples
data_all = json.load(open("/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json"))
samples = []
for entry in data_all:
    ap = entry.get("action_prompt") or entry.get("query", "")
    cands = parse_candidates(ap) if "action_prompt" in entry else []
    if not cands: continue
    try: fc = build_fc_prompt(tok, ap, cands)
    except: continue
    gt = entry.get("tool")
    gt_name = gt[0] if isinstance(gt, list) else gt
    if not gt_name: continue

    ctx = fc + JSON_PREFIX
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    full_ids = tok(ctx + gt_name, add_special_tokens=False)["input_ids"]
    k = 0
    while k < min(len(ctx_ids), len(full_ids)) and ctx_ids[k] == full_ids[k]:
        k += 1
    if k >= len(full_ids): continue
    samples.append((full_ids[:k], full_ids[k:], gt_name))
    if len(samples) >= args.n_queries: break
log(f"samples: {len(samples)}")

def measure(label, basis=None, alpha=0.0):
    """If basis=None, run without hook. Otherwise install V-bias with given basis."""
    per = []
    if basis is None:
        cm = torch.no_grad()
    else:
        cm = install_vbias_hooks(model, basis, alpha=alpha,
                                  n_kv=n_kv, head_dim=head_dim)
    with torch.no_grad():
        with cm:
            for prompt_ids, tool_ids, _ in samples:
                full = torch.tensor(prompt_ids + tool_ids,
                                    dtype=torch.long, device=args.device).unsqueeze(0)
                out = model(full)
                logits = out.logits[0].float()
                L_p = len(prompt_ids)
                logps = []
                for i, tid in enumerate(tool_ids):
                    pos = L_p + i - 1
                    logps.append(torch.log_softmax(logits[pos], dim=-1)[tid].item())
                per.append(sum(logps))
    arr = np.array(per)
    log(f"  {label}: mean_sum_logp = {arr.mean():.4f} ± {arr.std()/np.sqrt(len(arr)):.3f} SE")
    return arr

a0 = measure("no_steer", basis=None, alpha=0.0)
aK = measure(f"V-bias K-basis α={args.alpha}", basis=B_k, alpha=args.alpha)
aV = measure(f"V-bias V-basis α={args.alpha}", basis=B_v, alpha=args.alpha)
aR = measure(f"V-bias random-basis α={args.alpha}", basis=B_r, alpha=args.alpha)

print("\n=== summary ===")
print(f"  no_steer        : {a0.mean():.4f}")
print(f"  K-basis V-bias  : {aK.mean():.4f}  Δ={aK.mean()-a0.mean():+.4f}")
print(f"  V-basis V-bias  : {aV.mean():.4f}  Δ={aV.mean()-a0.mean():+.4f}")
print(f"  Random V-bias   : {aR.mean():.4f}  Δ={aR.mean()-a0.mean():+.4f}")

# Interpret
dK = abs(aK.mean() - a0.mean())
dV = abs(aV.mean() - a0.mean())
dR = abs(aR.mean() - a0.mean())
print("\n=== interpretation ===")
print(f"  |Δ K| = {dK:.4f}   |Δ V| = {dV:.4f}   |Δ R| = {dR:.4f}")
if dV > 1.5 * max(dK, dR):
    print("  → V-basis has strongest effect; K-basis is near-random on V-side")
    print("    ⇒ Thm 6.17 V-inert is LIKELY AN ARTIFACT of K-basis-on-V choice")
elif dK > 1.5 * dR:
    print("  → K-basis has real effect (not just random); V-basis doesn't help more")
    print("    ⇒ Thm 6.17 V-inert is LIKELY GENUINE for this benchmark")
else:
    print("  → all three effects similar magnitude")
    print("    ⇒ Signal is dominated by direction-free norm perturbation")

summary = {
    "model": args.model, "alpha": args.alpha,
    "no_steer_mean_sum_logp": float(a0.mean()),
    "K_basis_mean_sum_logp": float(aK.mean()),
    "V_basis_mean_sum_logp": float(aV.mean()),
    "random_basis_mean_sum_logp": float(aR.mean()),
    "n_samples": len(samples),
    "delta_K": float(aK.mean() - a0.mean()),
    "delta_V": float(aV.mean() - a0.mean()),
    "delta_R": float(aR.mean() - a0.mean()),
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f: json.dump(summary, f, indent=2)
print(f"\nwrote {args.out}")
