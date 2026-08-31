# -*- coding: utf-8 -*-
"""x710 — 착지한 모든 sim 에서 **주거나 잠금해제해 놓고 안 쓴 도구**를 센다 (읽기 전용).

정책 축자: "Do not unlock tools that you do not plan on giving to the user and actually using:
           this causes issues in database logging."
판정은 우리 정본 함수(`t2_gate_patch.unused_grants`)로 한다 — 사본 금지([[67]]).
"""
import glob, io, json, os, re, sys

GP = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
sys.path.insert(0, GP)

A2 = json.load(io.open(os.path.join(GP, "a2/banking_knowledge.gate.json"), encoding="utf-8"))
DRS = A2.get("dispatcher_role_check") or {}
EPL = A2.get("eplan") or {}
KEYS = ("agent_tool_name", "user_tool_name", "discoverable_tool_name",
        EPL.get("dispatch_name_key") or "agent_tool_name")


def inner(ar):
    for k in KEYS:
        v = (ar or {}).get(k)
        if v:
            return re.sub(r"_\d+$", "", str(v))
    return ""


def grants(msgs):
    given = unlocked = None
    given, unlocked, ran_u, ran_a = set(), set(), set(), set()
    for m in msgs or []:
        role = m.get("role")
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name")
            ar = tc.get("arguments")
            if isinstance(ar, str):
                try:
                    ar = json.loads(ar)
                except Exception:
                    ar = {}
            iv = inner(ar)
            if nm == DRS.get("give_tool") and iv:
                given.add(iv)
            elif nm == DRS.get("unlock_tool") and iv:
                unlocked.add(iv)
            elif nm == DRS.get("user_call") and iv:
                ran_u.add(iv)
            elif nm == (DRS.get("agent_call") or EPL.get("dispatch_tool")) and iv:
                ran_a.add(iv)
            if role == "user" and nm:
                ran_u.add(re.sub(r"_\d+$", "", str(nm)))
    return sorted(given - ran_u), sorted(unlocked - ran_a), len(given), len(unlocked)


rows = []
for p in sorted(glob.glob("/home/woori/iso_tau3/tau2-bench/data/simulations/*/results.json")
                + glob.glob("/home/woori/scratch/tau2-bench/data/simulations/*/results.json")):
    try:
        if os.path.getsize(p) < 50000:
            continue
        d = json.load(io.open(p, encoding="utf-8"))
    except Exception:
        continue
    tag = p.split("/")[-2]
    for s in (d.get("simulations") or []):
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is None:
            continue
        ug, uu, ng, nu = grants(s.get("messages"))
        rows.append((tag, s.get("task_id"), r, (ri.get("db_check") or {}).get("db_match"),
                     ug, uu, ng, nu))

print("착지 sim %d 건" % len(rows))
viol = [x for x in rows if x[4] or x[5]]
print("위반(준/잠금해제 뒤 안 쓴 도구 보유) %d 건 = %.0f%%" % (len(viol), 100.0 * len(viol) / max(len(rows), 1)))
print()
print("%-14s %-12s %-8s %s" % ("", "reward=0", "reward>0", "합"))
for lab, sel in (("위반 있음", lambda x: x[4] or x[5]), ("위반 없음", lambda x: not (x[4] or x[5]))):
    sub = [x for x in rows if sel(x)]
    z = sum(1 for x in sub if not x[2])
    print("%-14s %-12d %-8d %d" % (lab, z, len(sub) - z, len(sub)))
print()
print("-- 위반 예시 12건")
for t, tid, r, dbm, ug, uu, ng, nu in viol[:12]:
    print("  %-34s %-9s reward=%-4s db=%-5s 준뒤안씀=%s 잠금뒤안씀=%s"
          % (t[:34], tid, r, dbm, ug or "-", uu or "-"))
