#!/usr/bin/env python3
"""Controlled-depth selection task + deterministic OPERATION engine (B_BUDGET_SCALE_DESIGN §0-bis).

THESIS test substrate: can a SMALL model overcome superlative/comparative ("most/best/better")
WITH the help of a deterministic structure? The structure = the LLM NAMES the operation (argmax over
attr, filter, rank-k, comparative-to-anchor) — a SHALLOW recognition — and a deterministic ENGINE
EXECUTES it (sort/scan/select) over the catalog (the deep part). This fixes the static-$select
failure (which forced an answer-criteria like zoom="max", impossible) by emitting the OPERATION.

Generates: abstract catalog (N items: categorical filter attrs + a numeric ordinal attr, isotropized)
+ a controlled-depth query (filter d1 / argmax|argmin d2 / rank-k d3 / comparative d2) + the gold id
+ the gold operation-IR. resolve_operation() is BOTH the engine (arm B) and the oracle (gold).
Gold is guaranteed UNIQUE (distinct ordinal values among candidates) so accuracy is well-defined.
"""
import json, random, string, argparse


def _tok(rng, n=5):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def gen_schema(rng, iso, n_cat=2):
    cat = {}
    for i in range(n_cat):
        key = f"attr_{_tok(rng)}" if iso else f"cat{i}"
        nv = rng.randint(2, 3)
        cat[key] = [f"v_{_tok(rng,4)}" if iso else f"c{i}v{j}" for j in range(nv)]
    ordk = f"score_{_tok(rng)}" if iso else "score"
    return cat, ordk


# ----------------------------- deterministic engine (= oracle) -----------------------------
def resolve_operation(op_ir, items):
    """Execute a named operation over the catalog. op_ir = {op, attr, among, k?, dir?, anchor_id?}.
    Returns item_id|None. Deterministic; arm-B engine AND the gold oracle."""
    if not isinstance(op_ir, dict):
        return None
    among = op_ir.get("among") or {}
    cands = [it for it in items if all(str(it.get(k)) == str(v) for k, v in among.items())]
    op = op_ir.get("op")
    attr = op_ir.get("attr")
    if op == "filter":
        return cands[0]["id"] if len(cands) == 1 else None
    if op == "substitute":
        # produce the variant SAME as the anchor on every option EXCEPT those in `set`.
        # target = anchor options (categorical only; ord attr is None for subst/create) <- set overrides.
        anchor = next((it for it in items if it["id"] == op_ir.get("anchor_id")), None)
        if anchor is None:
            return None
        setv = op_ir.get("set") or {}
        target = {k: anchor[k] for k in anchor if k != "id" and anchor[k] is not None}
        target.update({k: v for k, v in setv.items() if k in target})
        m = [it for it in items if all(str(it.get(k)) == str(v) for k, v in target.items())]
        return m[0]["id"] if len(m) == 1 else None
    if op == "create":
        # build a NEW item fully specified by `set` (no reference); match the unique item with those options.
        setv = op_ir.get("set") or {}
        if not setv:
            return None
        m = [it for it in items if all(str(it.get(k)) == str(v) for k, v in setv.items())]
        return m[0]["id"] if len(m) == 1 else None
    if attr is None:
        return None
    try:
        if op == "argmax":
            return max(cands, key=lambda it: it[attr])["id"] if cands else None
        if op == "argmin":
            return min(cands, key=lambda it: it[attr])["id"] if cands else None
        if op == "rank":
            k = int(op_ir.get("k", 1))
            s = sorted(cands, key=lambda it: -it[attr])
            return s[k - 1]["id"] if len(s) >= k else None
        if op == "comparative":
            anchor = next((it for it in items if it["id"] == op_ir.get("anchor_id")), None)
            if anchor is None:
                return None
            dir_ = op_ir.get("dir", "greater")
            pool = [it for it in cands if (it[attr] > anchor[attr] if dir_ == "greater" else it[attr] < anchor[attr])]
            # nearest to anchor (closest qualifying) = deterministic tie-rule
            if not pool:
                return None
            return min(pool, key=lambda it: abs(it[attr] - anchor[attr]))["id"]
    except Exception:
        return None
    return None


# ----------------------------- generator -----------------------------
OPS = {"filter": 1, "argmax": 2, "argmin": 2, "rank": 3, "comparative": 2,
       "substitute": 2, "create": 1}  # op -> depth d


def gen_subst_create(rng, iso, op, N, width=0):
    """substitute = same-as-anchor-except-`set` (keep-rest, the τ² exchange shape); create = fully
    specified `set` (the τ² book shape). PURELY CATEGORICAL catalog (options like color/size/material,
    no ordinal) — these content ops are about identity/override, not ranking. Gold is PLANTED and
    de-duplicated so the target combo is unique (accuracy well-defined).

    width>0 = WIDTH-CONTROLLED substitute: change EXACTLY `width` attrs (the rest kept). Probes the
    B(L,width) budget axis (multi-attr delta = parallel binding). n_cat = width+2 so >=2 attrs are
    always kept (keep-rest non-trivial). width=0 => default 1-2 changes (training mix)."""
    n_cat = (width + 2) if (op == "substitute" and width > 0) else 3
    cat, ordk = gen_schema(rng, iso, n_cat=n_cat)
    catk = list(cat)
    for _ in range(200):
        items = []
        for _ in range(N):
            it = {"id": f"I{_tok(rng,8)}"}
            for k, vs in cat.items():
                it[k] = rng.choice(vs)
            it[ordk] = None  # categorical-only catalog
            items.append(it)
        if op == "substitute":
            anchor = rng.choice(items)
            n_change = width if width > 0 else rng.randint(1, min(2, n_cat))
            n_change = min(n_change, n_cat)
            change_attrs = rng.sample(catk, n_change)
            setv = {}
            for a in change_attrs:
                alts = [v for v in cat[a] if v != anchor[a]]
                if not alts:
                    break
                setv[a] = rng.choice(alts)
            if len(setv) != n_change:
                continue
            target = {k: anchor[k] for k in catk}; target.update(setv)
            op_ir = {"op": "substitute", "anchor_id": anchor["id"], "set": setv}
            label = "substitute"; filt = setv
            gold_pool = [it for it in items if it is not anchor]
        else:  # create
            anchor = None
            target = {k: rng.choice(cat[k]) for k in catk}
            op_ir = {"op": "create", "set": dict(target)}
            label = "create"; filt = dict(target)
            gold_pool = list(items)
        if not gold_pool:
            continue
        # plant gold = one pool item set to target; perturb any OTHER collision so target is unique
        gold_it = rng.choice(gold_pool)
        for k in catk:
            gold_it[k] = target[k]
        ok = True
        for it in items:
            if it is gold_it:
                continue
            if all(it[k] == target[k] for k in catk):
                a = rng.choice(catk)
                alts = [v for v in cat[a] if v != target[a]]
                if not alts:
                    ok = False; break
                it[a] = rng.choice(alts)
        if not ok:
            continue
        gold = gold_it["id"]
        if resolve_operation(op_ir, items) != gold:
            continue
        return {"items": items, "ord_attr": ordk, "cat_attrs": catk, "filter": filt,
                "op": op, "depth": OPS[op], "gold_id": gold, "op_ir": op_ir, "label": label}
    return None


def gen_example(rng, iso, op, N, width=0):
    if op in ("substitute", "create"):
        return gen_subst_create(rng, iso, op, N, width=width)
    cat, ordk = gen_schema(rng, iso)
    catk = list(cat)
    n_filters = 1 if op in ("filter", "argmax", "argmin", "comparative") else 2
    n_filters = min(n_filters, len(catk))
    for _ in range(200):  # retry until non-degenerate
        items = []
        for _ in range(N):
            it = {"id": f"I{_tok(rng,8)}"}
            for k, vs in cat.items():
                it[k] = rng.choice(vs)
            it[ordk] = None  # fill below
            items.append(it)
        filt = {catk[i]: rng.choice(cat[catk[i]]) for i in range(n_filters)}
        cands = [it for it in items if all(it[k] == v for k, v in filt.items())]
        need = {"filter": 1, "argmax": 2, "argmin": 2, "rank": 3, "comparative": 3}[op]
        if len(cands) < need:
            continue
        # assign DISTINCT ordinal values to candidates (unique extremum/rank); others random
        vals = rng.sample(range(1, 100), len(items))
        for it, v in zip(items, vals):
            it[ordk] = v
        if op == "filter":
            # pure unique lookup (d=1 denotation): filter by the UNIQUE ordinal value of a target.
            tgt = rng.choice(items)
            filt = {ordk: tgt[ordk]}
            m = [it for it in items if it[ordk] == tgt[ordk]]
            if len(m) != 1:
                continue
            gold = m[0]["id"]; op_ir = {"op": "filter", "among": filt}
            label = "filter"
        elif op == "argmax":
            gold = max(cands, key=lambda it: it[ordk])["id"]; op_ir = {"op": "argmax", "attr": ordk, "among": filt}
            label = "argmax"
        elif op == "argmin":
            gold = min(cands, key=lambda it: it[ordk])["id"]; op_ir = {"op": "argmin", "attr": ordk, "among": filt}
            label = "argmin"
        elif op == "rank":
            s = sorted(cands, key=lambda it: -it[ordk]); gold = s[1]["id"]
            op_ir = {"op": "rank", "k": 2, "attr": ordk, "among": filt}; label = "rank2"
        elif op == "comparative":
            anchor = rng.choice([it for it in items if it not in cands] or items)
            pool = [it for it in cands if it[ordk] > anchor[ordk]]
            if not pool:
                continue
            gold = min(pool, key=lambda it: abs(it[ordk] - anchor[ordk]))["id"]
            op_ir = {"op": "comparative", "attr": ordk, "dir": "greater", "anchor_id": anchor["id"], "among": filt}
            label = "comparative"
        # round-trip: engine on gold op_ir must reproduce gold
        if resolve_operation(op_ir, items) != gold:
            continue
        return {"items": items, "ord_attr": ordk, "cat_attrs": catk, "filter": filt,
                "op": op, "depth": OPS[op], "gold_id": gold, "op_ir": op_ir, "label": label}
    return None


def render_nl(ex, iso):
    """controlled NL surface of the query (the LLM must recognize the operation)."""
    f = ex["filter"]; o = ex["ord_attr"]
    fstr = " and ".join(f"{k} = {v}" for k, v in f.items())
    op = ex["op"]
    if op == "filter":
        return f"Select the item where {fstr}."
    if op == "argmax":
        return f"Among items where {fstr}, select the one with the HIGHEST {o}."
    if op == "argmin":
        return f"Among items where {fstr}, select the one with the LOWEST {o}."
    if op == "rank":
        return f"Among items where {fstr}, select the one with the SECOND highest {o}."
    if op == "comparative":
        aid = ex["op_ir"]["anchor_id"]
        return (f"Among items where {fstr}, select the one whose {o} is just GREATER than "
                f"that of item {aid} (the closest one above it).")
    if op == "substitute":
        aid = ex["op_ir"]["anchor_id"]
        return f"Exchange item {aid}: keep its other options unchanged but set {fstr}."
    if op == "create":
        return f"Create a new item with {fstr}."
    return ""


# ----------------------------- DIVERSE renderer (expression-isotropy) -----------------------------
# Goal (feedback-expression-diversity-required-for-transfer): break the SURFACE<->op correlation that
# the single-template render_nl injected (verb "exchange" -> op=exchange on tau2). The VERB is sampled
# from a POOL SHARED across ALL ops (so the verb carries NO op information); the op is decided ONLY by
# the MEANING phrase, itself drawn from several paraphrases. Deeper ops get more paraphrases (K scales
# with depth). A learner can only fit op from meaning, not from a surface token.
VERBS = ["Select", "Pick", "Choose", "Find", "Get me", "I want", "I'd like", "Give me",
         "Exchange it for", "Swap it to", "Show me", "Return the one and take"]
AMONG = ["among items where {f}", "out of the ones with {f}", "from variants where {f}",
         "restricting to items with {f}", "considering only {f}", "within the {f} options"]
OP_PHRASES = {
    # filter: pin by the (ordinal-value) equality; among carries the constraint
    "filter": ["the item where {f}", "the one with {f}", "the variant that has {f}",
               "the item matching {f}", "the one for which {f}"],
    "argmax": ["the one with the HIGHEST {o}", "the one with the most {o}", "the top-{o} one",
               "the maximum-{o} variant", "the largest {o}", "the one that maxes out {o}",
               "the {o} leader", "whichever has the greatest {o}"],
    "argmin": ["the one with the LOWEST {o}", "the one with the least {o}", "the bottom-{o} one",
               "the minimum-{o} variant", "the smallest {o}", "whichever has the least {o}",
               "the one that minimizes {o}", "the {o} laggard"],
    "rank": ["the one with the SECOND highest {o}", "the runner-up in {o}", "the next-to-top {o} one",
             "the 2nd-best by {o}", "the one just below the {o} maximum", "the second-place {o} variant"],
    "comparative": ["the one whose {o} is just above item {a}'s", "the one with slightly more {o} than item {a}",
                    "the next step up in {o} from item {a}", "the closest {o} above item {a}",
                    "the one a notch higher than item {a} in {o}", "the smallest {o} that still beats item {a}",
                    "the one barely exceeding item {a} in {o}", "the immediately-higher-{o} option vs item {a}"],
    # substitute: SAME as reference item {a} on everything EXCEPT {f} (keep the rest). The op is decided
    # by the keep-rest + reference structure, NOT by the verb (verbs are op-agnostic, shared pool above).
    "substitute": ["the same as item {a} but with {f}", "item {a} except changed to {f}",
                   "a variant just like item {a}, only {f} instead", "item {a}'s configuration but {f}",
                   "what item {a} is, keeping the rest, but {f}", "item {a} with {f} swapped in and nothing else changed",
                   "the match that keeps item {a}'s other options and sets {f}", "item {a} reconfigured so that {f}"],
    # create: a NEW item fully given by {f}; no reference, nothing kept.
    "create": ["a new item with {f}", "a fresh one specified as {f}", "an item configured as {f}",
               "a brand-new variant with {f}", "one built from scratch with {f}", "a new record where {f}",
               "a newly made item having {f}", "an entirely new one set to {f}"],
}


def render_nl_diverse(ex, rng):
    """Expression-isotropic surface: verb (op-agnostic pool) + meaning phrase (op-deciding paraphrase)
    + among clause. Surface<->op correlation ~0 so the learner must use meaning, not a surface token."""
    f = ex["filter"]; o = ex["ord_attr"]; op = ex["op"]
    fstr = " and ".join(f"{k} = {v}" for k, v in f.items())
    verb = rng.choice(VERBS)
    phrase = rng.choice(OP_PHRASES[op]).format(o=o, f=fstr, a=ex["op_ir"].get("anchor_id", ""))
    if op in ("filter", "substitute", "create"):
        # constraint/identity IS the meaning phrase (no subset-restriction `among` clause): filter pins by
        # value, substitute keeps-rest-from-anchor, create fully specifies. The catalog is the whole pool.
        return f"{verb} {phrase}."
    among = rng.choice(AMONG).format(f=fstr)
    # randomize clause order too (among-first vs op-first) to avoid positional shortcut
    if rng.random() < 0.5:
        return f"{verb} {phrase}, {among}."
    return f"{among.capitalize()}, {verb.lower()} {phrase}."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--N", type=int, default=10, help="catalog size")
    ap.add_argument("--ops", default="filter,argmax,argmin,rank,comparative")
    ap.add_argument("--iso", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=0, help="substitute: change EXACTLY this many attrs (B-width sweep); 0=default mix")
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    w = None if args.validate_only else open(args.out, "w", encoding="utf-8")
    n_emit = 0; per = {o: 0 for o in ops}
    while n_emit < args.n:
        op = ops[n_emit % len(ops)]
        ex = gen_example(rng, args.iso, op, args.N, width=args.width)
        if ex is None:
            continue
        ex["nl"] = render_nl(ex, args.iso)
        if w:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
        per[op] += 1; n_emit += 1
    if w:
        w.close()
    print(f"emitted={n_emit} N={args.N} iso={args.iso} per-op={per} -> {'(validate)' if args.validate_only else args.out}")
