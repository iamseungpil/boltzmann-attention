# -*- coding: utf-8 -*-
"""실패 sim 정직 분해: 미충족 gold 액션을 (A)tool-never-called=coverage vs (B)called-wrong-args=args.
+ sim-pass 상한: 'coverage만 강제(모든 gold 액션 수행)' / 'coverage+args(전 액션 정답)' 두 arm.
name-only 행 = 'tool 호출됨?' 체크, args 행 = 값 판정. 도구 family로 페어링."""
import json, glob, re, sys, io
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def Nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}
fam = lambda n: re.sub(r"_\d+$", "", str(n))

def agent_called_families(s):
    """agent가 실제 호출한 도구 family 집합 (discoverable dispatch inner + 직접 호출)."""
    fams = Counter()
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or ""
            if nm == "call_discoverable_agent_tool":
                inner = Nd(tc.get("arguments")).get("agent_tool_name", "")
                if inner:
                    fams[fam(inner)] += 1
            elif nm:
                fams[fam(nm)] += 1
    return fams

unmet_kind = Counter()
sim_arm = Counter()
n_fail = 0
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        ri = s.get("reward_info") or {}
        if ri.get("reward") in (None, 1.0):
            continue
        acs = ri.get("action_checks") or []
        if not acs:
            continue
        n_fail += 1
        called = agent_called_families(s)
        # gold 액션의 args 행만 값판정에 사용(arguments 키 有 + agent_tool_name 有)
        unmet_cov = 0   # 미충족 & tool 미호출 = coverage
        unmet_arg = 0   # 미충족 & tool 호출됨   = args
        for ac in acs:
            a = ac.get("action") or {}
            outer = Nd(a.get("arguments"))
            atn = outer.get("agent_tool_name", "")
            if not atn or "arguments" not in outer:
                continue  # name-only 행 또는 비-도구 assertion → 값판정에서 제외
            met = ac.get("action_reward")
            if met is None:
                met = 1.0 if ac.get("action_match") else 0.0
            if float(met) >= 1.0:
                continue
            tf = fam(atn)
            if called.get(tf, 0) > 0:
                unmet_arg += 1
            else:
                unmet_cov += 1
        # sim-pass 상한 arm:
        #  coverage-arm: 미충족이 전부 coverage(tool미호출)면, 강제열거+수행으로 닫힘 가정 → pass
        #  args-arm    : coverage+args 둘다 닫아야 pass
        if unmet_cov == 0 and unmet_arg == 0:
            sim_arm["non-arg 판정 실패(assertion/기타)"] += 1
        elif unmet_arg == 0:
            sim_arm["coverage-only (강제열거로 닫힘 상한)"] += 1
        elif unmet_cov == 0:
            sim_arm["args-only (값 정답필요)"] += 1
        else:
            sim_arm["coverage+args 혼합"] += 1
        unmet_kind["coverage(미호출)"] += unmet_cov
        unmet_kind["args(호출·오답)"] += unmet_arg

print("=== 실패 sim %d개: 미충족 gold-액션(args행) 분해 ===" % n_fail)
tot = sum(unmet_kind.values())
for k, v in unmet_kind.most_common():
    print("  %-22s %6d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
print("\n=== 실패 sim 분류 (coverage vs args) ===")
tt = sum(sim_arm.values())
for k, v in sim_arm.most_common():
    print("  %-40s %5d (%.1f%%)" % (k, v, 100 * v / max(tt, 1)))
print("\n[[08]] coverage 지배 → E-PLAN loop(강제열거+coverage-track)이 진짜 레버(C80 정합).")
print("       args = compute/⋈/gather (inner router). dispute는 args의 부분집합일 뿐.")
