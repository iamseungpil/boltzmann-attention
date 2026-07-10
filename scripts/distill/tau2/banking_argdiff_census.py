#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""banking 전모델 전수 인자-수준 diff census.
모든 실패 sim(gold 비어있지 않음)에 대해 gold콜↔실행콜을 키(도구/발견도구명)로 짝짓고,
불일치를 {MISSING_CALL, ARG_NUMERIC, ARG_ID, ARG_ENUM, ARG_BOOL, ARG_DATE, ARG_TEXT, EXTRA_WRITE}로 분류.
sim-수준 1차 귀속 + pooled 필드-클래스 카운트 + per-model 표.
usage: py -3 banking_argdiff_census.py C:\\tmp\\traj [out.json]"""
import sys, glob, os, json, io, re
from collections import Counter, defaultdict
import fine_function_decomp as F

ID_RE = re.compile(r"(^|_)(id|ids)$")
DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$|^\d{4}-\d{2}-\d{2}")
NUMSTR = re.compile(r"^-?\d+(\.\d+)?$")

def getargs(x):
    if isinstance(x, str):
        try: x = json.loads(x)
        except Exception: return None, {}
    if not isinstance(x, dict): return None, {}
    nm = x.get("agent_tool_name")
    a = x.get("arguments", x if nm is None else {})
    if isinstance(a, str):
        try: a = json.loads(a)
        except Exception: a = {}
    return nm, (a if isinstance(a, dict) else {})

def keyof(n, raw):
    nm, a = getargs(raw)
    if n in ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool",
             "give_discoverable_user_tool", "call_discoverable_user_tool"):
        return f"{n}:{nm}", a
    return n, (raw if isinstance(raw, dict) else getargs(raw)[1])

def canon(v):
    if isinstance(v, str):
        s = v.strip()
        if NUMSTR.match(s):
            try: return float(s)
            except Exception: pass
        return s.lower()
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, list): return tuple(canon(x) for x in v)
    if isinstance(v, dict): return tuple(sorted((k, canon(x)) for k, x in v.items()))
    return v

def field_class(k, gv, ev):
    if isinstance(gv, bool) or isinstance(ev, bool): return "ARG_BOOL"
    def isnum(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
    if isnum(canon(gv)) and isnum(canon(ev)): return "ARG_NUMERIC"
    if ID_RE.search(k.lower()): return "ARG_ID"
    sv = str(gv)
    if DATE_RE.match(sv) or "date" in k.lower(): return "ARG_DATE"
    if isinstance(gv, str) and len(gv) <= 40 and " " not in gv.strip(): return "ARG_ENUM"
    return "ARG_TEXT"

def main(d, out):
    files = sorted(glob.glob(os.path.join(d, "*_banking.json")))
    pooled_field = Counter()
    pooled_primary = Counter()
    permodel = {}
    for f in files:
        lbl = os.path.basename(f).replace("_banking.json", "")
        sims = json.load(io.open(f, encoding="utf-8"))["simulations"]
        fld = Counter(); prim = Counter(); nfail_g = 0
        for s in sims:
            if not F.is_fail(s): continue
            try: G0 = F.gold_acts(s)
            except Exception: G0 = []
            if not G0: continue
            nfail_g += 1
            G = [keyof(n, a) for n, a in G0]
            E = []
            for m in s.get("messages", []):
                for tc in (m.get("tool_calls") or []):
                    n = tc.get("name") or (tc.get("function") or {}).get("name")
                    ar = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or {}
                    E.append(keyof(n, ar))
            # greedy exact-match consume
            left = list(E)
            miss_call = 0
            sim_fields = Counter()
            for k, ga in G:
                cg = {kk: canon(v) for kk, v in (ga or {}).items()}
                hit = None
                for i, (ek, ea) in enumerate(left):
                    if ek == k and {kk: canon(v) for kk, v in (ea or {}).items()} == cg:
                        hit = i; break
                if hit is not None:
                    left.pop(hit); continue
                cands = [(ek, ea) for ek, ea in E if ek == k]
                if not cands:
                    miss_call += 1; continue
                # nearest candidate by fewest differing fields
                best = None; bestdiff = None
                for ek, ea in cands:
                    ce = {kk: canon(v) for kk, v in (ea or {}).items()}
                    diff = {kk for kk in set(cg) | set(ce) if cg.get(kk) != ce.get(kk)}
                    if best is None or len(diff) < len(bestdiff):
                        best, bestdiff = (ek, ea), diff
                ce = {kk: canon(v) for kk, v in (best[1] or {}).items()}
                for kk in bestdiff:
                    if kk not in cg: sim_fields["ARG_EXTRA_FIELD"] += 1
                    elif kk not in ce: sim_fields["ARG_MISSING_FIELD"] += 1
                    else: sim_fields[field_class(kk, (ga or {}).get(kk), dict(best[1] or {}).get(kk))] += 1
            gk = set(k for k, _ in G)
            extra_w = sum(1 for ek, _ in E
                          if (ek.startswith("call_discoverable_agent_tool") or F.aclass(ek.split(":")[0]) == "MUTATE")
                          and ek not in gk)
            fld.update(sim_fields)
            fld["MISSING_CALL"] += miss_call
            fld["EXTRA_WRITE"] += (1 if extra_w else 0)
            # primary attribution
            if miss_call >= max(1, len(G) // 3): p = "MISSING_CALL(미실행지배)"
            elif sim_fields:
                top = sim_fields.most_common(1)[0][0]
                p = top
            elif extra_w: p = "EXTRA_WRITE_ONLY"
            else: p = "NO_CALL_DIFF(순서/기타)"
            prim[p] += 1
        permodel[lbl] = {"nfail_gold": nfail_g, "primary": dict(prim)}
        pooled_field.update(fld); pooled_primary.update(prim)
        print(f"{lbl:15} nfail(gold)={nfail_g:4d} primary_top={prim.most_common(3)}")
    print("\n#### pooled sim-level PRIMARY attribution ####")
    tot = sum(pooled_primary.values())
    for k, v in pooled_primary.most_common():
        print(f"  {k:28} {v:5d} ({100*v/tot:.1f}%)")
    print("\n#### pooled FIELD-class counts (all mismatched fields) ####")
    for k, v in pooled_field.most_common():
        print(f"  {k:20} {v}")
    json.dump({"permodel": permodel, "pooled_primary": dict(pooled_primary),
               "pooled_field": dict(pooled_field)},
              open(out, "w"), indent=1)
    print("WROTE", out)

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else r"C:\tmp\traj"
    out = sys.argv[2] if len(sys.argv) > 2 else r"C:\workspace\_cdp_private_local\banking_argdiff.json"
    main(d, out)
