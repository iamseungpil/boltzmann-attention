#!/usr/bin/env python3
"""Held-out cross-domain transfer summary (FULL MATRIX).

Transfer = test on a domain the adapter NEVER trained on.
  - LODO-7:  lodo_<X> tested on X (the one domain held out of its training mix).
  - train-1: train1_<X> trained on X ONLY, tested on each of the other 6 (1->6 transfer).

Reports per (adapter, held-out domain): official success (evaluator.py:277, tool_full)
+ should_T/F split + should_T LOGINCALL-quirk count, vs released leaderboard max + Qwen2.5-7B base.
Missing evals render as NA so this runs incrementally as the eval batch fills in.
Output dirs (xdomain_eval_heldout.sh / finish_train1bank.sh conventions):
  LODO     stack       = xho_lodo_<X>_<X>_stack        (bank headline = eval_t1c_headline/bank)
  LODO     adapteronly = xho_lodo_<X>_<X>_adapteronly
  train-1  stack       = xho_train1<X>_<D>_stack
"""
import json, glob
OUT = "/home/woori/scratch/sft_alias_run"
ALL = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
JSON = "ast_tbox_v2-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"

def truthy(c):
    if c is True: return True
    if isinstance(c, (list, tuple)) and len(c): return c[0] is True
    if isinstance(c, str): return c.strip().startswith(("True", "(True", "[True"))
    return False

def stats(path):
    try: d = json.load(open(path))
    except Exception: return None
    if not isinstance(d, list) or not d: return None
    n = s = sT = nT = sF = nF = q = 0
    for e in d:
        ev = (e.get("evaluations") or [{}])[0]
        if "success" not in ev: continue   # sim ran but not yet eval_tasks-scored
        n += 1; ok = bool(ev.get("success")); s += ok
        if ev.get("action_should_succeed"):
            nT += 1; sT += ok
            if ok:
                lc = []
                for il in e.get("interactions") or []:
                    for m in il.get("interaction") or []:
                        if isinstance(m, dict) and m.get("tool_name") == "login_user":
                            lc.append(truthy(m.get("content")))
                if lc and not any(lc): q += 1
        else:
            nF += 1; sF += ok
    return None if n == 0 else (n, s, sT, nT, sF, nF, q)

def lb(domain):
    """released leaderboard max (any system) + Qwen2.5-7B react baseline, tool_full."""
    best = ("", 0.0); q7 = 0.0
    for fp in glob.glob(f"/home/woori/scratch/SOPBench/output/{domain}/ast_*tool_full*.json"):
        r = stats(fp)
        if r and r[0] > 0:
            pct = 100 * r[1] / r[0]
            if pct > best[1]:
                best = (fp.split("/")[-1].split("-mode")[0].replace("ast_", "")[:18], pct)
            if "qwen2.5-7b" in fp and "react" in fp: q7 = pct
    return best, q7

def fpath_lodo_stack(X):
    return (f"{OUT}/eval_t1c_headline/bank/{JSON}" if X == "bank"
            else f"{OUT}/xho_lodo_{X}_{X}_stack/{X}/{JSON}")
def fpath_lodo_adapter(X):
    return f"{OUT}/xho_lodo_{X}_{X}_adapteronly/{X}/{JSON}"
def fpath_train1(X, D):
    return f"{OUT}/xho_train1{X}_{D}_stack/{D}/{JSON}"

def pct(r):  return f"{100*r[1]/r[0]:.1f}%({r[1]}/{r[0]})" if r else "NA"
def pctf(r): return (100*r[1]/r[0]) if r else None

# ---------- LODO-7 ----------
print("="*112)
print("LODO-7  (lodo_<X> trained on the OTHER 6, tested held-out on X)")
print("="*112)
print(f"{'held-out X':<16}{'adapter-only':>14}{'STACK(official)':>18}{'Δscaf':>7}{'sT(ok/n)':>11}{'sF(ok/n)':>11}{'quirk':>6}{'LB-max':>22}{'Qwen7B':>8}")
print("-"*112)
lodo_pcts = {}
for X in ALL:
    a = stats(fpath_lodo_adapter(X)); s = stats(fpath_lodo_stack(X))
    (bm, bp), bq = lb(X)
    ap, st = pct(a), pct(s)
    dl = f"+{pctf(s)-pctf(a):.0f}" if (a and s) else "-"
    sT = f"{s[2]}/{s[3]}" if s else "-"
    sF = f"{s[4]}/{s[5]}" if s else "-"
    qk = str(s[6]) if s else "-"
    lodo_pcts[X] = pctf(s)
    print(f"{X:<16}{ap:>14}{st:>18}{dl:>7}{sT:>11}{sF:>11}{qk:>6}{f'{bm} {bp:.0f}%':>22}{f'{bq:.0f}%':>8}")

# ---------- train-1  7x6 matrix ----------
print()
print("="*112)
print("train-1  7x6  (rows=train domain X trained ONLY on X; cols=held-out test domain D; STACK official success%)")
print("="*112)
_corner = "train\\test"
hdr = f"{_corner:<14}" + "".join(f"{d[:9]:>11}" for d in ALL) + f"{'rowAvg':>9}"
print(hdr); print("-"*112)
col_vals = {d: [] for d in ALL}
for X in ALL:
    cells = []; rv = []
    for D in ALL:
        if D == X:
            cells.append(f"{'--':>11}"); continue
        r = stats(fpath_train1(X, D)); p = pctf(r)
        if p is not None:
            rv.append(p); col_vals[D].append(p)
            cells.append(f"{p:>10.1f} ")
        else:
            cells.append(f"{'NA':>11}")
    ravg = f"{sum(rv)/len(rv):>8.1f}" if rv else f"{'-':>8}"
    print(f"{('train1_'+X)[:14]:<14}" + "".join(cells) + f"{ravg:>9}")
print("-"*112)
# per-held-out-domain mean across training sources (transfer robustness) + leaderboard ref
favg = []
for d in ALL:
    favg.append(f"{(sum(col_vals[d])/len(col_vals[d])):>10.1f} " if col_vals[d] else f"{'-':>11}")
print(f"{'colAvg':<14}" + "".join(favg) + f"{'':>9}")
print(f"{'LODO-7':<14}" + "".join((f"{lodo_pcts[d]:>10.1f} " if lodo_pcts.get(d) is not None else f"{'NA':>11}") for d in ALL))
lbrow = []
for d in ALL:
    (_, bp), _ = lb(d); lbrow.append(f"{bp:>10.1f} ")
print(f"{'LB-max':<14}" + "".join(lbrow))
print("="*112)
