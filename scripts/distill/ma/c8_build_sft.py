#!/usr/bin/env python3
"""C8 data builder (C8_PROCEDURE_ROUTING_TRANSFER_DESIGN_2026_06_17).
Builds chat SFT JSONL teaching NL->op-IR routing for the 5 generators, GLOSS-FREE prompt by default
(the routing must be learned in weights, not spoon-fed). Eval sets: in-dist holdout (train seed,
fresh examples = convergence sanity) and held-out (DIFFERENT seed = different attr tokens/schema =
the transfer 'domain'). synth_depth re-isotropises every example, so the SFT forbids vocab
memorisation; transfer to a new seed isolates pure procedure-type routing. arm_B prompts carry NO
catalog (only NL + attr names) so sequences are short and training is fast.

Modes:
  --mode eval  : write c8_eval_indist.jsonl + c8_eval_heldout.jsonl (shared across all variants)
  --mode train : write ONE chat SFT file (--out) with options n / gloss_in_sft / cmp_heavy
"""
import json, random, argparse, os
from collections import Counter
from synth_depth import gen_example, render_nl, render_nl_diverse, OPS
from depth_eval import build_arm_B_user


def make_raw(seed, n, Ns, op_weights=None, iso=1, diverse=False, renderer=None):
    """op_weights: dict op->relative weight. diverse=True => expression-isotropic NL.
    renderer (fn(ex,rng)->nl) overrides => K-controlled expression pool (synth_expr.make_renderer)."""
    rng = random.Random(seed)
    ops = list(OPS)
    if op_weights:
        sched = []
        for op in ops:
            sched += [op] * int(op_weights.get(op, 1))
    else:
        sched = ops
    out = []
    i = 0
    while len(out) < n:
        op = sched[i % len(sched)]; N = Ns[i % len(Ns)]
        i += 1
        ex = gen_example(rng, iso, op, N)
        if ex is None:
            continue
        if renderer is not None:
            ex["nl"] = renderer(ex, rng)
        else:
            ex["nl"] = render_nl_diverse(ex, rng) if diverse else render_nl(ex, iso)
        out.append(ex)
    return out


def to_chat(ex, gloss=False):
    return {"messages": [
        {"role": "system", "content": "Output ONLY JSON."},
        {"role": "user", "content": build_arm_B_user(ex, gloss=gloss)},
        {"role": "assistant", "content": json.dumps(ex["op_ir"], ensure_ascii=False)}],
        "meta": {"op": ex["op"], "N": len(ex["items"])}}


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "train"], required=True)
    ap.add_argument("--outdir", default="/home/woori/scratch/depth/c8")
    ap.add_argument("--out", default="", help="train mode: output jsonl path")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n_eval", type=int, default=250)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--heldout_seed", type=int, default=777)
    ap.add_argument("--Ns", default="5,10,20,50")
    ap.add_argument("--gloss_in_sft", type=int, default=0, help="0=gloss-free (transfer); 1=gloss ablation")
    ap.add_argument("--cmp_heavy", type=int, default=0, help="1=weight comparative 3x (others 1x)")
    ap.add_argument("--diverse", type=int, default=0, help="1=expression-isotropic NL (verb/op decorrelated)")
    ap.add_argument("--K", type=int, default=0, help=">0 = K-controlled expression pool (synth_expr)")
    ap.add_argument("--method", default="random", choices=["random", "axis", "kcenter"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    Ns = [int(x) for x in args.Ns.split(",")]
    dv = bool(args.diverse)
    renderer = None
    if args.K > 0:
        from synth_expr import make_renderer
        renderer = make_renderer(args.K, args.method, args.train_seed)

    if args.mode == "eval":
        indist = make_raw(args.train_seed + 1, args.n_eval, Ns, diverse=dv)   # fresh, same seed-stream = convergence
        heldout = make_raw(args.heldout_seed, args.n_eval, Ns, diverse=dv)     # different seed = new vocab/schema = transfer
        sfx = "_div" if dv else ""
        dump(f"{args.outdir}/c8_eval_indist{sfx}.jsonl", indist)
        dump(f"{args.outdir}/c8_eval_heldout{sfx}.jsonl", heldout)
        print("indist ops:", dict(Counter(e["op"] for e in indist)))
        print("heldout ops:", dict(Counter(e["op"] for e in heldout)))
    else:
        ow = {"comparative": 3} if args.cmp_heavy else None
        train = make_raw(args.train_seed, args.n, Ns, op_weights=ow, diverse=dv, renderer=renderer)
        out = args.out or f"{args.outdir}/c8_train_sft.jsonl"
        dump(out, [to_chat(e, gloss=bool(args.gloss_in_sft)) for e in train])
        print("train ops:", dict(Counter(e["op"] for e in train)))
        # record train-expression diversity D (for the K-sweep curve)
        if args.K > 0:
            from expr_diversity import diversity
            dd = diversity([e["nl"] for e in train])
            json.dump({"K": args.K, "method": args.method, "D": dd}, open(out + ".D.json", "w"))
            print(f"D(K={args.K},{args.method}): eff_rank={dd['eff_rank']:.2f} mean_dist={dd['mean_dist']:.3f}")
