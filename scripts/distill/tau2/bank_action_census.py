# -*- coding: utf-8 -*-
"""실패 sim의 미충족(met=0) action_check를 dispute vs non-dispute로 분해.
dispute-only 컨트롤러가 닫을 수 있는 상한 = dispute만 미충족인 sim."""
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

sim_class = Counter()          # 실패 sim 분류
unmet_tool = Counter()         # 미충족 액션의 도구 family
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
        unmet_disp = 0
        unmet_nondisp = 0
        for ac in acs:
            # met: action_reward==1.0 이면 충족. args 체크(arguments 키 有)만 실질 판정.
            a = ac.get("action") or {}
            outer = Nd(a.get("arguments"))
            atn = outer.get("agent_tool_name", "")
            # args 체크 행만(값 판정). name-only 행(arguments 키 없음)은 skip.
            if "arguments" not in outer and atn:   # name-only 행
                pass
            met = ac.get("action_reward")
            if met is None:
                met = 1.0 if ac.get("action_match") else 0.0
            if met and float(met) >= 1.0:
                continue
            # 미충족 액션
            tf = fam(atn) if atn else "(no_tool)"
            unmet_tool[tf] += 1
            if "transaction_dispute" in tf:
                unmet_disp += 1
            else:
                unmet_nondisp += 1
        if unmet_disp and not unmet_nondisp:
            sim_class["DISPUTE-only 미충족 (컨트롤러 사정권)"] += 1
        elif unmet_disp and unmet_nondisp:
            sim_class["dispute + 비-dispute 둘다 미충족"] += 1
        elif unmet_nondisp and not unmet_disp:
            sim_class["비-dispute만 미충족 (dispute 컨트롤러 무관)"] += 1
        else:
            sim_class["미충족 0 (args 판정 외 실패)"] += 1

print("=== 실패 sim %d개 분류 (미충족 action_check 기준) ===" % n_fail)
tot = sum(sim_class.values())
for k, v in sim_class.most_common():
    print("  %-42s %5d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
print("\n=== 미충족 액션 도구 family Top15 ===")
for k, v in unmet_tool.most_common(15):
    print("  %-46s %5d" % (k, v))
