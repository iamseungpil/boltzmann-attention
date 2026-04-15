"""Thm 6.21 log-p measurement v3 (autoregressive full-target).

Fix over v2:
- Measures log p(full tool-name | prompt) = sum over tokens of log p(token_i |
  prompt + JSON_PREFIX + tool_tokens_{<i}). v2 measured only first-token log-p
  which ignores conditional mass on subsequent BPE sub-tokens.

Sanity: baseline argmax-first-token rate and mean per-token log-p at α=0.

Also report F1-style argmax-match rate (at α=0, whether first token argmax is
the correct tool's first token).
"""
import sys, time, json, argparse
from pathlib import Path
import torch
import numpy as np

REPO = Path("/home/woori/workspace_common/boltzmann-attention")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "ocq"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_metatool_subtask1 import parse_candidates, install_kbias_hooks
from eval_metatool_subtask4 import build_fc_prompt

T0 = time.time()
def log(m): print(f"[+{time.time()-T0:.1f}s] {m}", flush=True)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--b-ont", required=True)
p.add_argument("--alphas", type=float, nargs="+",
               default=[0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
p.add_argument("--device", default="cuda:1")
p.add_argument("--n-queries", type=int, default=100)
p.add_argument("--out", required=True)
args = p.parse_args()

log(f"loading {args.model}")
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
log("loaded")

B_obj = torch.load(args.b_ont, map_location="cpu", weights_only=False)
B_ont = B_obj["B_ont"] if isinstance(B_obj, dict) else B_obj
n_kv = model.config.num_key_value_heads
head_dim = getattr(model.config, "head_dim",
                   model.config.hidden_size // model.config.num_attention_heads)

JSON_PREFIX = '<tool_call>{"name": "'

data_all = json.load(open("/tmp/MetaTool/dataset/tmp_dataset/Task2-Subtask1.json"))

# Prepare: full prompt + full tool-name token sequence
samples = []  # list of (ids_prompt_including_prefix, tool_token_ids, tool_name)
skipped = 0
for entry in data_all:
    ap = entry.get("action_prompt") or entry.get("query", "")
    cands = parse_candidates(ap) if "action_prompt" in entry else []
    if not cands: skipped += 1; continue
    try: fc = build_fc_prompt(tok, ap, cands)
    except: skipped += 1; continue
    gt = entry.get("tool")
    gt_name = gt[0] if isinstance(gt, list) else gt
    if not gt_name: skipped += 1; continue

    # Full context ends right before tool-name: FC + '<tool_call>{"name": "'
    ctx = fc + JSON_PREFIX
    # Tool name tokens are whatever tokenizer produces for ctx+gt_name vs ctx alone
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    full_ids = tok(ctx + gt_name, add_special_tokens=False)["input_ids"]
    # Handle BPE prefix merging: find first diverging position
    k = 0
    while k < min(len(ctx_ids), len(full_ids)) and ctx_ids[k] == full_ids[k]:
        k += 1
    if k >= len(full_ids):
        skipped += 1; continue
    # ids_prompt is first k tokens (the common prefix); tool_tokens are the rest
    ids_prompt = full_ids[:k]
    tool_tokens = full_ids[k:]
    samples.append((ids_prompt, tool_tokens, gt_name))
    if len(samples) >= args.n_queries:
        break

log(f"samples kept: {len(samples)} (skipped {skipped})")
log(f"sample tool tokens[0]: len={len(samples[0][1])} tokens={samples[0][1]} (name={samples[0][2]!r})")
log(f"sample tool tokens[1]: len={len(samples[1][1])} tokens={samples[1][1]} (name={samples[1][2]!r})")

def measure(alpha):
    if alpha == 0.0:
        ctx_mgr = torch.no_grad()
    else:
        ctx_mgr = install_kbias_hooks(model, B_ont, alpha=alpha,
                                      n_kv=n_kv, head_dim=head_dim)
    per_sample = []
    with torch.no_grad():
        with ctx_mgr:
            for ids_prompt, tool_tokens, _name in samples:
                # Build full sequence: prompt + tool_tokens
                full = torch.tensor(ids_prompt + tool_tokens,
                                    dtype=torch.long, device=args.device).unsqueeze(0)
                out = model(full)
                logits = out.logits[0].float()  # (T, V)
                # For each tool_token at position p=len(ids_prompt) + i, prediction
                # is from logits[p-1] (log-prob of tool_tokens[i] given context up to p-1).
                L_p = len(ids_prompt)
                token_logps = []
                first_argmax_match = None
                for i, tid in enumerate(tool_tokens):
                    pos = L_p + i - 1  # predict token at pos+1 from logit[pos]
                    logp = torch.log_softmax(logits[pos], dim=-1)[tid].item()
                    token_logps.append(logp)
                    if i == 0:
                        first_argmax_match = int(logits[pos].argmax().item() == tid)
                per_sample.append({
                    "tool_tokens": tool_tokens,
                    "per_token_logp": token_logps,
                    "sum_logp": float(np.sum(token_logps)),
                    "first_token_logp": float(token_logps[0]) if token_logps else 0.0,
                    "first_argmax_match": first_argmax_match,
                })
    sum_logps = np.array([s["sum_logp"] for s in per_sample])
    first_lps = np.array([s["first_token_logp"] for s in per_sample])
    first_argmax = np.array([s["first_argmax_match"] for s in per_sample])
    return {
        "mean_sum_logp": float(sum_logps.mean()),
        "std_sum_logp": float(sum_logps.std()),
        "mean_first_logp": float(first_lps.mean()),
        "first_argmax_acc": float(first_argmax.mean()),
        "n": len(per_sample),
    }

log("--- sanity α=0 ---")
base = measure(0.0)
log(f"  base mean_sum_logp={base['mean_sum_logp']:.4f} (±{base['std_sum_logp']/np.sqrt(base['n']):.3f} SE)")
log(f"  base mean_first_logp={base['mean_first_logp']:.4f}")
log(f"  base first_argmax_acc={base['first_argmax_acc']:.4f}")
if base["first_argmax_acc"] < 0.1:
    log(f"*** WARNING: base argmax<0.1, measurement unreliable ***")

results = {0.0: base}
for a in args.alphas:
    if a == 0.0: continue
    log(f"--- α={a} ---")
    r = measure(a)
    d_sum = r["mean_sum_logp"] - base["mean_sum_logp"]
    d_first = r["mean_first_logp"] - base["mean_first_logp"]
    log(f"  mean_sum_logp={r['mean_sum_logp']:.4f} (Δ{d_sum:+.4f})  "
        f"first_logp={r['mean_first_logp']:.4f} (Δ{d_first:+.4f})  "
        f"first_argmax_acc={r['first_argmax_acc']:.4f}")
    results[a] = r

log("\n=== full-target α curve ===")
alphas_sorted = sorted(results.keys())
log(f"{'α':>6s} {'sum_logp':>10s} {'Δ_sum':>10s} {'first_lp':>10s} {'Δ_first':>10s} {'argmax':>8s}")
for a in alphas_sorted:
    r = results[a]
    log(f"{a:>6.3f} {r['mean_sum_logp']:>10.4f} {r['mean_sum_logp']-base['mean_sum_logp']:>+10.4f} "
        f"{r['mean_first_logp']:>10.4f} {r['mean_first_logp']-base['mean_first_logp']:>+10.4f} "
        f"{r['first_argmax_acc']:>8.4f}")

log("\n=== 2nd-diffs (sum_logp) ===")
for i in range(1, len(alphas_sorted)-1):
    a0,a1,a2 = alphas_sorted[i-1:i+2]
    l0,l1,l2 = (results[x]["mean_sum_logp"] for x in (a0,a1,a2))
    d2 = l2 - 2*l1 + l0
    log(f"  α=[{a0},{a1},{a2}]: 2nd_diff={d2:+.4f} ({'concave' if d2<0 else 'convex'})")

out = {"model": args.model, "b_ont": args.b_ont,
       "script_version": "v3_2026_04_16_autoregressive",
       "alphas": alphas_sorted, "per_alpha": results,
       "n_samples": len(samples), "n_skipped": skipped,
       "prefix_appended": JSON_PREFIX}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f: json.dump(out, f, indent=2)
log(f"wrote {args.out}")
