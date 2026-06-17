#!/usr/bin/env python3
"""K-controlled expression generator (EXPRESSION_DIVERSITY_TRANSFER_DESIGN §4).
A "combo" = (verb, op-phrase, among-template, order-bit) — a point in the axis product
L×S×P×R. For a given op and budget K, select K combos by:
  - random : uniform sample
  - axis   : stratified (rotate verbs × phrases × among independently = factorial coverage)
  - kcenter: greedy max-min on surface embeddings (orthogonal — widest D per K)
Then render NL by sampling from the chosen K combos. K=1 ~ single template (D≈0);
K=|pool| ~ full diversity. Lets us sweep D and measure transfer = f(D).
"""
import random
from synth_depth import VERBS, AMONG, OP_PHRASES, gen_example, render_nl_diverse  # noqa


def combo_pool(op):
    amongs = [None] if op == "filter" else AMONG
    return [(v, p, a, o) for v in VERBS for p in OP_PHRASES[op] for a in amongs for o in (0, 1)]


def render_combo(ex, combo):
    """Render NL for example ex using a fixed (verb,phrase,among,order) combo. Mirrors
    synth_depth.render_nl_diverse's surface so a model trained here sees consistent format."""
    v, p, a, o = combo
    f = ex["filter"]; oo = ex["ord_attr"]; op = ex["op"]
    fstr = " and ".join(f"{k} = {vv}" for k, vv in f.items())
    phrase = p.format(o=oo, f=fstr, a=ex["op_ir"].get("anchor_id", ""))
    if op == "filter":
        return f"{v} {phrase}."
    among = a.format(f=fstr)
    if o == 0:
        return f"{v} {phrase}, {among}."
    return f"{among.capitalize()}, {v.lower()} {phrase}."


def _dummy_render(op, combo):
    """Render a combo on a fixed dummy schema (for embedding-based kcenter selection)."""
    rng = random.Random(12345)
    ex = None
    for _ in range(50):
        ex = gen_example(rng, 1, op, 10)
        if ex:
            break
    return render_combo(ex, combo) if ex else str(combo)


def select_combos(op, K, method, seed):
    rng = random.Random((seed * 131 + hash(op)) % (2**31))
    pool = combo_pool(op)
    if K >= len(pool):
        return pool
    if method == "random":
        return rng.sample(pool, K)
    if method == "axis":
        # stratified: each index rotates verb, phrase, among, order independently
        amongs = [None] if op == "filter" else AMONG
        ph = OP_PHRASES[op]
        sel = []
        seen = set()
        i = 0
        while len(sel) < K and i < 10000:
            c = (VERBS[i % len(VERBS)], ph[i % len(ph)], amongs[i % len(amongs)], i % 2)
            if c not in seen:
                seen.add(c); sel.append(c)
            i += 1
        return sel
    if method == "kcenter":
        from expr_diversity import kcenter_indices
        texts = [_dummy_render(op, c) for c in pool]
        idx = kcenter_indices(texts, K, seed=seed)
        return [pool[i] for i in idx]
    raise ValueError(method)


def make_renderer(K, method, seed):
    """Returns render_fn(ex, rng) that samples from the K selected combos per op."""
    cache = {}

    def render_fn(ex, rng):
        op = ex["op"]
        if op not in cache:
            cache[op] = select_combos(op, K, method, seed)
        return render_combo(ex, rng.choice(cache[op]))
    return render_fn


if __name__ == "__main__":
    import argparse, json
    from expr_diversity import diversity
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="argmax")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--method", default="random", choices=["random", "axis", "kcenter"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    combos = select_combos(args.op, args.K, args.method, args.seed)
    texts = [_dummy_render(args.op, c) for c in combos]
    print(f"op={args.op} K={args.K} method={args.method} -> {len(combos)} combos")
    for t in texts[:8]:
        print("  ", t)
    print("D:", json.dumps(diversity(texts)))
