# -*- coding: utf-8 -*-
"""compute-like 필드의 결정론 규칙 역설계 (gold 입력→출력·[[05]] 발명금지·[[08]] gold-fit).
각 필드가 어느 도구에·어떤 동반 필드와 함께 나오는지, gold 값 분포·상관을 본다."""
import json, glob, re, sys, io
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
def Nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}
fam = lambda n: re.sub(r"_\d+$", "", str(n))

TARGETS = ["provisional_credit_eligible", "eligible_for_provisional_credit",
           "amount_difference", "expected_apy", "partial_refund_amount", "new_rewards_earned"]

# 각 타깃 필드가 등장하는 도구 + 동반 필드 + gold값 예시
by_field = defaultdict(lambda: {"tools": Counter(), "coargs": Counter(), "examples": []})
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        ri = s.get("reward_info") or {}
        for ac in (ri.get("action_checks") or []):
            a = ac.get("action") or {}
            outer = Nd(a.get("arguments"))
            atn = outer.get("agent_tool_name", "")
            if not atn or "arguments" not in outer:
                continue
            ga = Nd(outer.get("arguments"))
            for t in TARGETS:
                if t in ga:
                    e = by_field[t]
                    e["tools"][fam(atn)] += 1
                    for k in ga:
                        if k != t:
                            e["coargs"][k] += 1
                    if len(e["examples"]) < 6:
                        # 동반 필드 값 + 타깃 값
                        e["examples"].append({k: ga.get(k) for k in ga})

for t in TARGETS:
    e = by_field[t]
    print("\n===== %s =====" % t)
    print("  도구:", dict(e["tools"].most_common(3)))
    print("  동반필드 Top:", [k for k, _ in e["coargs"].most_common(12)])
    print("  gold 예시(동반+타깃):")
    for ex in e["examples"][:4]:
        # 타깃과 관련 동반필드만 축약
        keys = [k for k in ex if any(w in k for w in ("date", "amount", "apy", "rate", "tier",
                "class", "balance", "expected", "actual", "days", "eligible", "provisional",
                "credit", "difference", "reward", "disputed"))]
        print("   ", {k: ex.get(k) for k in keys}, "→", t, "=", ex.get(t))
