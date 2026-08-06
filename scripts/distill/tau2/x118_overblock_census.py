# -*- coding: utf-8 -*-
"""치환이 gold를 막을 수 있는가 — 사전 등록 계량 (오프라인·전 태스크).

지배 규칙은 push의 표적이 **미충족 게이트의 `applies_to`** 에 있을 때 그 명령을 게이트의 요건
문장으로 바꾼다. 차단이 아니라 치환이므로 손해가 날 수 있는 경로는 하나뿐이다 — gold이 그 게이트의
satisfier보다 **먼저** 그 행동을 요구하는 경우. gold `action_checks`는 순서가 있는 목록이라 셀 수 있다.

통과 조건(사전 등록): **satisfier가 앞에 없는 건수 = 0**.
"""
import glob
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TAU2 = os.environ.get("GO_TAU2", "/home/woori/scratch/tau2-bench")
DOM = os.path.join(TAU2, "data", "tau2", "domains", "banking_knowledge")
HERE = os.path.dirname(os.path.abspath(__file__))
A2 = os.environ.get("A2_GATE") or os.path.join(HERE, "a2", "banking_knowledge.gate.json")

a2 = json.load(io.open(A2, encoding="utf-8"))
gates = a2.get("gates") or []

tasks = []
for p in (glob.glob(os.path.join(DOM, "tasks*.json"))
          + glob.glob(os.path.join(DOM, "tasks", "*.json"))):
    d = json.load(io.open(p, encoding="utf-8"))
    for t in (d.get("tasks") if isinstance(d, dict) else d) or []:
        if isinstance(t, dict) and t.get("id"):
            tasks.append(t)

print("게이트 %d개 · 태스크 %d개" % (len(gates), len(tasks)))

tot_user, covered, viol = 0, 0, []
per_gate = {}
for t in tasks:
    acts = []
    for c in ((t.get("evaluation_criteria") or {}).get("actions") or []):
        acts.append((c.get("name"), c.get("requestor") or "assistant"))
    names = [n for n, _ in acts]
    for i, (nm, req) in enumerate(acts):
        if req != "user":
            continue
        tot_user += 1
        for g in gates:
            if nm not in set(g.get("applies_to") or ()):
                continue
            if nm in set((g.get("applies_when") or {}).get("not_in") or ()):
                continue
            sat = set((g.get("satisfiers") or {}).keys())
            if not sat:
                continue
            covered += 1
            per_gate[g.get("id")] = per_gate.get(g.get("id"), 0) + 1
            if not (sat & set(names[:i])):
                viol.append((t.get("id"), nm, g.get("id"), names[:i]))
            break

print("gold의 user-실행 액션           : %d건" % tot_user)
print("  게이트 applies_to에 덮인 것   : %d건  %s" % (covered, per_gate))
print("  ★satisfier가 앞에 없는 건수  : %d  (통과 조건 = 0)" % len(viol))
for v in viol[:15]:
    print("     ", v[0], v[1], "gate=", v[2], "선행 gold=", v[3])
print("판정:", "PASS — 치환이 gold를 막을 경로 없음" if not viol else "FAIL — 오차단 위험 실재")
