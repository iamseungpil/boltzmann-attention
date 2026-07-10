#!/usr/bin/env python3
"""Phase 0 — G→BC split per-case 확정 (airline/telecom/banking).
정본: DOMAIN_TRANSFER_STATUS_AND_PLAN_2026_07_09.md §3.4.
결과(2026-07-09): airline G6=BC3 compute(baggage 수치) · telecom G4=BC6 조기포기(fix 미시도) ·
                  banking G2=BC2 부하(KB검색은 함·조립 누락). ★같은 G가 도메인마다 다른 BC."""
import json, io, re, sys
from collections import Counter
import fine_function_decomp as F

NUM = re.compile(r"(amount|price|cost|total|number|count|fee|income|baggage)", re.I)
FIX = re.compile(r"(toggle|reboot|reseat|grant_|set_network|enable_|disable_|refuel|reset_|apply_)", re.I)


def _ctx(s): return F.toolctx(s) + " " + F.userctx(s)


def _seq(s):
    by = {m.get("id") or m.get("tool_call_id"): m for m in s.get("messages", []) if m.get("role") == "tool"}
    o = []
    for m in s.get("messages", []):
        if m.get("role") != "assistant": continue
        for tc in m.get("tool_calls") or []:
            tm = by.get(tc.get("id"))
            o.append((tc.get("name"), not (tm is not None and tm.get("error"))))
    return o


def airline_operand(path):  # G6 → BC3(compute) vs BC4(select) vs PROV
    d = json.load(io.open(path, encoding="utf-8")); sims = d["simulations"]
    bc = Counter()
    for s in [x for x in sims if F.is_fail(x)]:
        G = [(n, a) for n, a in F.gold_acts(s) if F.aclass(n) == "MUTATE"]
        O = [(n, a) for n, a in F.exec_acts(s)[0] if F.aclass(n) == "MUTATE"]
        if not G or not O or len(G) != len(O) or len(G) > 6: continue
        pm = F.pair(G, O); c = _ctx(s)
        for i in range(len(G)):
            gn, ga = G[i]; on, oa = O[pm[i]]
            if gn != on: continue
            for k in F.dkeys(ga, oa):
                v = oa.get(k); vals = v if isinstance(v, list) else [v]
                if NUM.search(k): bc[f"{k}:BC3_compute"] += 1
                elif any(str(x) and str(x) not in c for x in vals if x is not None): bc[f"{k}:PROV_fab"] += 1
                else: bc[f"{k}:BC4_select"] += 1
    print("AIRLINE G6:", dict(bc.most_common(10)))


def telecom_persist(path):  # G4 → BC6 (escalate while fixable unfixed)
    d = json.load(io.open(path, encoding="utf-8"))
    fails = [s for s in d["simulations"] if F.is_fail(s)]
    a = b = c = 0
    for s in fails:
        unfixed = sum(1 for x in (F.ri(s).get("env_assertions") or []) if x.get("met") is False)
        if not unfixed: continue
        sq = _seq(s)
        esc = any("transfer" in (n or "").lower() or "human" in (n or "").lower() for n, _ in sq)
        fixes = sum(1 for n, ok in sq if FIX.search(n or "") and ok)
        if esc and fixes == 0: a += 1
        elif esc: b += 1
        else: c += 1
    print(f"TELECOM G4: escalate&no-fix(BC6 순수)={a} · escalate&partial={b} · no-escalate={c}")


def banking_reach(path):  # G2 → BC2(load: searched but not assembled) vs no-search
    d = json.load(io.open(path, encoding="utf-8"))
    fails = [s for s in d["simulations"] if F.is_fail(s)]
    load = no = 0
    for s in fails:
        g = set(a.get("action", {}).get("arguments", {}).get("agent_tool_name")
                for a in (F.ri(s).get("action_checks") or [])
                if a.get("action", {}).get("name") == "unlock_discoverable_agent_tool")
        e = set()
        for m in s.get("messages", []):
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") == "unlock_discoverable_agent_tool":
                    e.add((tc.get("arguments") or {}).get("agent_tool_name"))
        if not (g - e): continue
        sq = _seq(s)
        if any("search" in (n or "").lower() for n, _ in sq): load += 1
        else: no += 1
    print(f"BANKING G2: KB검색함·조립누락(BC2 부하)={load} · 검색안함={no}")


if __name__ == "__main__":
    airline_operand(r"C:/tmp/traj/opus45_airline.json")
    telecom_persist(r"C:/tmp/traj/opus45_telecom.json")
    banking_reach(r"C:/tmp/traj/opus45_banking.json")
