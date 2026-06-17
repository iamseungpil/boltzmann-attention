#!/usr/bin/env python3
"""C8 data builder (C8_PROCEDURE_ROUTING_TRANSFER_DESIGN_2026_06_17).
Builds: (1) chat SFT JSONL teaching NL->op-IR routing for all 5 generators, GLOSS-FREE prompt
(the routing must be learned in weights, not spoon-fed); (2) two eval sets — in-dist holdout
(train seed, fresh examples = convergence sanity) and held-out (DIFFERENT seed = different attr
tokens/schema = the transfer 'domain'). Because synth_depth re-isotropises every example (new
random vocab per item), the SFT itself forbids vocab memorisation — so transfer to a new seed
isolates pure procedure-type routing. SFT target = gold op_ir JSON; assistant-only loss via
lora_train_chat_toolcall.py.
"""
import json, random, argparse
from synth_depth import gen_example, render_nl, OPS
from depth_eval import build_arm_B_user


def make_raw(seed, n, Ns, iso=1):
    rng = random.Random(seed)
    ops = list(OPS)
    out = []
    i = 0
    while len(out) < n:
        op = ops[i % len(ops)]; N = Ns[i % len(Ns)]
        i += 1
        ex = gen_example(rng, iso, op, N)
        if ex is None:
            continue
        ex["nl"] = render_nl(ex, iso)
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
    ap.add_argument("--outdir", default="/home/woori/scratch/depth/c8")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=250)
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--heldout_seed", type=int, default=777)
    ap.add_argument("--Ns", default="5,10,20,50")
    ap.add_argument("--gloss_in_sft", type=int, default=0, help="0=gloss-free SFT (transfer test); 1=ablation")
    args = ap.parse_args()
    import os; os.makedirs(args.outdir, exist_ok=True)
    Ns = [int(x) for x in args.Ns.split(",")]

    # train (seed 0) — chat SFT, gloss-free
    train = make_raw(args.train_seed, args.n_train, Ns)
    dump(f"{args.outdir}/c8_train_sft.jsonl", [to_chat(e, gloss=bool(args.gloss_in_sft)) for e in train])
    # in-dist holdout = fresh examples from the SAME seed stream (convergence sanity)
    indist = make_raw(args.train_seed + 1, args.n_eval, Ns)
    dump(f"{args.outdir}/c8_eval_indist.jsonl", indist)
    # held-out = DIFFERENT seed = different vocab/schema = transfer domain
    heldout = make_raw(args.heldout_seed, args.n_eval, Ns)
    dump(f"{args.outdir}/c8_eval_heldout.jsonl", heldout)
    # per-op counts (balance check)
    from collections import Counter
    print("train ops:", dict(Counter(e["op"] for e in train)))
    print("heldout ops:", dict(Counter(e["op"] for e in heldout)))
