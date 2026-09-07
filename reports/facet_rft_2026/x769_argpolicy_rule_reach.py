#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x769 - T2_RULE_AT_WRITE / T2_ARG_POLICY_AT_WRITE 의 **트리거 술어 실측** (오프라인).
세기만 한다. 판정 0. 코드 수정 0."""
import io, json, os, sys
HERE = r"C:\workspace\ba-frft\scripts\distill\tau2"
sys.path.insert(0, HERE)
import t2_gate_patch as G
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DOMAIN = "banking_knowledge"
a2 = G._domain_a2(DOMAIN)
surf = json.load(io.open(os.path.join(HERE, "a2", "env_surface.json"), encoding="utf-8"))[DOMAIN]["tools"]

wrset = G._confirm_write_tools(a2) | set(((a2 or {}).get("eplan") or {}).get("write_tools") or [])
print("[_wrset] confirm-gate=%r  eplan.write_tools=%r" % (
    sorted(G._confirm_write_tools(a2)), sorted(set(((a2 or {}).get("eplan") or {}).get("write_tools") or []))))
print("[_wrset] = %r  (|_wrset|=%d)" % (sorted(wrset), len(wrset)))
print()

class TC(object):
    def __init__(self, n): self.name = n; self.arguments = {}

# env_surface 의 실제 도구 이름(접미 숫자 포함) 중 _wrset 접두와 맞는 것
print("=== env_surface 의 mutating 도구 중 _wrset 과 매칭되는 실이름 ===")
live = []
for t in sorted(surf):
    for w in wrset:
        if t == w or t.startswith(w):
            live.append(t); print("   %-52s mutates=%s args=%d" % (t, surf[t].get("mutates"), len(surf[t].get("args") or [])))
print()

print("=== 도구별 RULE / ARG_POLICY 페이로드 (닫힌 술어 실행) ===")
for t in live:
    r = G._declared_rules_for(TC(t), a2)
    params = list(surf[t].get("args") or ())
    p = G._policy_rows_for(a2, params)
    axes = sorted({str(x.get("axis") or "") for x in G._policy_facts(a2)} & set(params))
    print("--- %s" % t)
    print("    RULE       : %s (%d자)" % ("HIT" if r else "MISS", len(r or "")))
    if r:
        for ln in r.splitlines(): print("        %s" % ln[:170])
    print("    ARG_POLICY : %s (%d자, cap=%s) 조인축 %d/%d = %r"
          % ("HIT" if p else "MISS(=0행 또는 cap초과)", len(p or ""),
             os.environ.get("T2_ARG_POLICY_CAP", "4000"), len(axes), len(params), axes))
print()

# ARG_POLICY 가 cap 때문에 죽는가 - 원문 길이(cap 무시)
print("=== ARG_POLICY cap 진단: cap 을 무한대로 두면 몇 자인가 ===")
os.environ["T2_ARG_POLICY_CAP"] = "10000000"
for t in live:
    params = list(surf[t].get("args") or ())
    p = G._policy_rows_for(a2, params)
    print("   %-52s %d자  (cap4000 통과=%s)" % (t, len(p or ""), (len(p or "") <= 4000 and bool(p))))
