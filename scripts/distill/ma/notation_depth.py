#!/usr/bin/env python3
"""§7c notation-depth measurement — does d(e) predict LLM selection failure, or is it the IR/artifact?

For each tau2 exchange request: parse NL into a procedure skeleton, assign a coarse depth d(e)
(operator-nesting proxy: filter=1, +superlative/comparative/conditional/relational). Then join
with per-model dumps (m_sigma_transfer_eval_v4 --dump: base / m0 / aprov) and report
new_item_ids correctness by depth, PER MODEL.

DECISIVE READOUT (theory adjudication LIE_ABSTRACTION_THEORY §7c/§7d vs artifact):
  - If BASE (in-head literal) ALSO fails as d rises  -> depth/capability matters (theory survives).
  - If BASE succeeds across d but STRUCTURED ($select) fails (esp. at high d) -> the failure is the
    IR/forced-format STRIPPING the model's in-head capability, NOT a circuit-depth limit
    (=> §7d 'scale can't because depth' is MIS-APPLIED here; the ops are TC0-doable and base proves it).
  - The 'base-right-but-structured-wrong' set clustering at high d = IR can't express what the model CAN do.
"""
import json, re, argparse
from collections import defaultdict

# coarse operator detectors (heuristic; we test the CORRELATION direction, not perfect parsing)
SUPER = re.compile(r"\b(most expensive|cheapest|highest|lowest|largest|smallest|longest|shortest|"
                   r"maximum|minimum|max |min |biggest|best|greatest|fewest|most\b)", re.I)
RANKED = re.compile(r"\b(second|third|next cheapest|next most)\b", re.I)
COMPAR = re.compile(r"\b(bigger|smaller|brighter|less bright|lighter|heavier|larger|shorter|longer|"
                    r"more \w+|less \w+|\w+er than|better than)\b", re.I)
COND = re.compile(r"\b(if (?:the |it |that |several|multiple|not )|otherwise|prefer\b|"
                  r"not available|isn't available|unavailable|fall ?back|if you can'?t)\b", re.I)
REL = re.compile(r"\b(same .* as the other|as the other|same .* as my|matching the)\b", re.I)


def depth(nl):
    cats = []
    if SUPER.search(nl) or RANKED.search(nl): cats.append("superlative")
    if COMPAR.search(nl): cats.append("comparative")
    if COND.search(nl): cats.append("conditional")
    if REL.search(nl): cats.append("relational")
    d = 1 + len(cats)            # 1 = base filter level; each embedded operator +1
    return d, (cats or ["filter"])


def load_dump_correct(path):
    """task_id -> bool (new_item_ids resolved == gold)."""
    out = {}
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        g = (d.get("gold") or {}).get("new_item_ids")
        r = (d.get("resolved") or {}).get("new_item_ids")
        out[str(d.get("task_id"))] = (g is not None and r == g)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="/home/woori/scratch/ma_eval_cases.jsonl")
    ap.add_argument("--dumps", nargs="+", default=[
        "base:/home/woori/scratch/dump_base.jsonl",
        "m0:/home/woori/scratch/dump_m0.jsonl",
        "aprov:/home/woori/scratch/dump_aprov.jsonl"])
    args = ap.parse_args()
    cases = {str(json.loads(l)["task_id"]): json.loads(l) for l in open(args.cases, encoding="utf-8")}
    models = {}
    for spec in args.dumps:
        tag, path = spec.split(":", 1)
        try: models[tag] = load_dump_correct(path)
        except Exception as e: print(f"skip {tag}: {e}")

    rows = []
    for tid, c in cases.items():
        d, cats = depth(c["nl"])
        rows.append((tid, d, cats, {m: mc.get(tid) for m, mc in models.items()}))

    # per-model accuracy by depth
    print("=== new_item_ids accuracy by depth d(e) ===")
    by = {m: defaultdict(lambda: [0, 0]) for m in models}
    for tid, d, cats, mc in rows:
        for m in models:
            ok = mc.get(m)
            if ok is None: continue
            by[m][d][1] += 1; by[m][d][0] += int(ok)
    ds = sorted({d for _, d, _, _ in rows})
    hdr = "  d  | " + " | ".join(f"{m:>8}" for m in models) + " |  n"
    print(hdr); print("  " + "-" * (len(hdr)))
    for d in ds:
        cells = []
        n = 0
        for m in models:
            a = by[m][d]; n = max(n, a[1])
            cells.append(f"{a[0]}/{a[1]}({a[0]/max(a[1],1):.2f})" if a[1] else "   -    ")
        print(f"  {d}  | " + " | ".join(f"{c:>8}" for c in cells) + f" | {n}")

    # ★ base-right-but-structured-wrong (IR strips capability) — cluster at high d?
    if "base" in models:
        print("\n=== base-CORRECT but structured-WRONG (IR/format strips in-head capability) ===")
        for sm in [m for m in models if m != "base"]:
            hits = [(tid, d, cats) for tid, d, cats, mc in rows
                    if mc.get("base") is True and mc.get(sm) is False]
            dd = sorted(h[1] for h in hits)
            print(f"  {sm}: {len(hits)} cases · depths={dd}")
            for tid, d, cats in sorted(hits, key=lambda x: -x[1])[:8]:
                print(f"     task {tid:>4} d={d} {cats} :: {cases[tid]['nl'][:90]}")

    # correlation-ish summary
    print("\n=== shallow(d=1) vs deep(d>=2) accuracy per model ===")
    for m in models:
        sh = [mc[m] for _, d, _, mc in rows if d == 1 and mc.get(m) is not None]
        de = [mc[m] for _, d, _, mc in rows if d >= 2 and mc.get(m) is not None]
        f = lambda x: f"{sum(x)}/{len(x)}({sum(x)/max(len(x),1):.2f})"
        print(f"  {m:>8}: shallow={f(sh)}  deep={f(de)}")
