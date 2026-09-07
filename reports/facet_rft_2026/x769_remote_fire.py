#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x769 - 맡은 레버군의 **발화 실측** (리모트 · GPU 0 · CPU 만).

T2_SPEC_ARG_FACTS : 엔진 술어 두 갈래(type-deny / enum-deny)를 46 핀 sim 에 그대로 적용.
T2_SPEC_AT_WRITE  : `_env_spec_for` + dist>=MIN + 도구당 1회 를 **반사실**로 적용.
                    (라이브에서는 T2_DECIDE_BEFORE_WRITE 안에 갇혀 안 돈다)
⛔ 판정하지 않는다. 세기만 한다.
"""
import io, os, sys, json
from pathlib import Path
from loguru import logger

logger.remove()
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = "/home/woori/workspace_common/boltzmann-attention-pi"
sys.path.insert(0, REPO + "/scripts/distill/tau2")
import t2_gate_patch as G
import gate_interpreter as GI
from tau2.data_model.simulation import Results

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
A2 = GI.load_domain_a2("banking_knowledge")
WRSET = (G._confirm_write_tools(A2)
         | set(((A2 or {}).get("eplan") or {}).get("write_tools") or []))
MIN = 8


def specfacts_hits(msgs):
    seen, th, eh = set(), [], []
    for i, m in enumerate(msgs):
        tcs = getattr(m, "tool_calls", None) or []
        if str(getattr(m, "role", "")) != "assistant" or not tcs:
            continue
        dpt = G._declared_params_by_tool(msgs[:i])
        for c in tcs:
            tn = str(G._exact_tool_name(c) or "")
            d2 = dpt.get(tn) or {}
            if not d2:
                continue
            av = dict(G._prov_scan_args(c, selectors=None))
            bad = [k for k, v in av.items()
                   if (d2.get(k) or ("", []))[0] == "boolean" and not isinstance(v, bool)]
            if bad and (tn, "\0bool") not in seen:
                seen.add((tn, "\0bool"))
                th.append((i, tn, sorted(bad), [repr(av.get(k)) for k in sorted(bad)]))
                break
            fired = False
            for ek, ev in sorted(av.items()):
                en3 = (d2.get(ek) or ("", []))[1]
                es = str(ev).strip()
                if not en3 or not es or es in en3:
                    continue
                if (tn, ek, es) in seen:
                    continue
                seen.add((tn, ek, es))
                eh.append((i, tn, ek, es))
                fired = True
                break
            if fired:
                break
    return th, eh


def specatwrite_hits(msgs):
    seen, out = set(), []
    for i, m in enumerate(msgs):
        tcs = getattr(m, "tool_calls", None) or []
        if str(getattr(m, "role", "")) != "assistant" or not tcs:
            continue
        wc = next((c for c in tcs
                   if G._eff_tool_name(c) in WRSET
                   or str(getattr(c, "name", "")) in WRSET), None)
        if wc is None:
            continue
        spec, si, sd = G._env_spec_for(wc, msgs[:i])
        k = str(G._exact_tool_name(wc) or "")
        if not spec or sd < MIN:
            out.append((i, G._eff_tool_name(wc), sd, 0, "MISS"))
        elif k in seen:
            out.append((i, G._eff_tool_name(wc), sd, len(spec), "CAPPED"))
        else:
            seen.add(k)
            out.append((i, G._eff_tool_name(wc), sd, len(spec), "FIRE"))
    return out


def main(pairs_path):
    pairs = [ln.split() for ln in open(pairs_path).read().strip().splitlines() if ln.strip()]
    cache = {}
    TT = TE = TF = TR = 0
    for tag, task, simid in pairs:
        if tag not in cache:
            p = Path(SIMROOT) / (tag + ".json")
            if not p.exists():
                p = Path(SIMROOT) / tag / "results.json"
            try:
                cache[tag] = Results.load(p)
            except Exception as e:
                print("LOADFAIL %s %r" % (tag, e))
                cache[tag] = None
        res = cache[tag]
        if res is None:
            continue
        sim = next((s for s in res.simulations if str(s.id) == simid), None)
        if sim is None:
            print("NOSIM %s %s" % (tag, task))
            continue
        msgs = list(sim.messages or [])
        th, eh = specfacts_hits(msgs)
        sw = specatwrite_hits(msgs)
        fire = [x for x in sw if x[4] == "FIRE"]
        TT += len(th); TE += len(eh); TF += len(fire); TR += len(sw)
        print("SAF %s nmsg=%d type=%d enum=%d | SAW reach=%d fire=%d dists=%s"
              % (task, len(msgs), len(th), len(eh), len(sw), len(fire),
                 [(x[0], x[2], x[3]) for x in sw][:8]))
        for h in th:
            print("   TYPE %s msg=%d tool=%s args=%s vals=%s" % (task, h[0], h[1], h[2], h[3]))
        for h in eh:
            print("   ENUM %s msg=%d tool=%s arg=%s val=%r" % (task, h[0], h[1], h[2], h[3]))
    print("TOTAL type-deny=%d enum-deny=%d | SAW reach=%d fire=%d" % (TT, TE, TR, TF))


if __name__ == "__main__":
    main(sys.argv[1])
