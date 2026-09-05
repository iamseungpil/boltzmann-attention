# -*- coding: utf-8 -*-
"""x766-f: 3도메인 `gates` 선언의 대상-실재 점검(관문2 analog) + 사이드카 발화 대조. 읽기 전용."""
import json
import os

ENG = r"C:\workspace\ba-frft\scripts\distill\tau2"
A2 = os.path.join(ENG, "a2")

env = json.load(open(os.path.join(A2, "env_surface.json"), encoding="utf-8"))
env2 = json.load(open(os.path.join(A2, "env_surface_airline_retail.json"), encoding="utf-8"))


def inv(d):
    s = set()
    t = d.get("tools")
    s |= set(t.keys()) if isinstance(t, dict) else set(t or [])
    for e in ("exposed", "discoverable_user_tools", "user_tools", "discoverable_agent_tools"):
        v = d.get(e)
        if isinstance(v, list):
            s |= set(v)
        elif isinstance(v, dict):
            s |= set(v.keys())
    return s


INV = {"banking": inv(env["banking_knowledge"]),
       "retail": inv(env2["retail"]),
       "airline": inv(env2["airline"])}
FILES = {"banking": "banking_knowledge.gate.json", "retail": "retail.gate.json",
         "airline": "airline.gate.json"}

for dom, fn in FILES.items():
    d = json.load(open(os.path.join(A2, fn), encoding="utf-8"))
    g = d.get("gates") or []
    print("### %s  gates=%d  inventory=%d tools" % (dom, len(g), len(INV[dom])))
    for e in g:
        if not isinstance(e, dict):
            print("   (non-dict)", e)
            continue
        gid = e.get("id")
        kind = e.get("kind")
        ap = e.get("applies_to") or []
        if isinstance(ap, str):
            ap = [ap]
        present = [t for t in ap if t in INV[dom]]
        missing = [t for t in ap if t not in INV[dom]]
        sat = e.get("satisfiers") or []
        if isinstance(sat, str):
            sat = [sat]
        sat_present = [t for t in sat if t in INV[dom]]
        sat_missing = [t for t in sat if t not in INV[dom]]
        print("   %-34s kind=%-14s applies_to %d (present %d / MISSING %d) satisfiers %d (present %d / MISSING %d)"
              % (gid, kind, len(ap), len(present), len(missing), len(sat), len(sat_present), len(sat_missing)))
        if missing:
            print("        applies_to MISSING:", missing[:8])
        if sat_missing:
            print("        satisfiers MISSING:", sat_missing[:8])
        print("        fields:", sorted(e.keys()))
    print()
