# -*- coding: utf-8 -*-
"""[[08]] 종료사유·infra 혼입 + '비-args 실패' sim 정체 규명."""
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
    fams = Counter()
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or ""
            if nm == "call_discoverable_agent_tool":
                inner = Nd(tc.get("arguments")).get("agent_tool_name", "")
                if inner: fams[fam(inner)] += 1
            elif nm: fams[fam(nm)] += 1
    return fams

term = Counter()
reward_keys = Counter()
nonarg_examples = []
n_fail = 0
for f in glob.glob("C:/tmp/traj/*_banking.json"):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        ri = s.get("reward_info") or {}
        if ri.get("reward") in (None, 1.0):
            continue
        n_fail += 1
        # 종료사유
        tr = s.get("termination_reason") or s.get("term_reason") or ri.get("info", {})
        term[str(s.get("termination_reason"))] += 1
        for k in ri.keys():
            reward_keys[k] += 1
        # 비-args 실패 sim 판정
        acs = ri.get("action_checks") or []
        called = agent_called_families(s)
        cov = arg = 0
        for ac in acs:
            a = ac.get("action") or {}
            outer = Nd(a.get("arguments"))
            atn = outer.get("agent_tool_name", "")
            if not atn or "arguments" not in outer: continue
            met = ac.get("action_reward")
            if met is None: met = 1.0 if ac.get("action_match") else 0.0
            if float(met) >= 1.0: continue
            if called.get(fam(atn), 0) > 0: arg += 1
            else: cov += 1
        if cov == 0 and arg == 0 and len(nonarg_examples) < 3:
            # 이 sim은 args행 판정으론 전부 met인데 reward=0 → 왜?
            nonarg_examples.append((s.get("task_id"), f.split("/")[-1], ri))

print("=== 실패 sim %d ===" % n_fail)
print("termination_reason 분포:", dict(term))
print("reward_info 키:", dict(reward_keys))
print("\n=== '비-args 실패' 예시 3건: reward_info 구조 ===")
for tid, fn, ri in nonarg_examples:
    print("\n-- %s (%s) reward=%s" % (tid, fn, ri.get("reward")))
    for k, v in ri.items():
        if k == "action_checks":
            acs = v or []
            unmet = [Nd((ac.get('action') or {}).get('arguments')).get('agent_tool_name', '') or '(assertion)'
                     for ac in acs if float(ac.get('action_reward') or (1.0 if ac.get('action_match') else 0.0)) < 1.0]
            print("   action_checks: %d개, 미충족 액션명: %s" % (len(acs), unmet[:10]))
        elif k in ("reward", "info", "task_id"):
            print("   %s = %s" % (k, json.dumps(v, ensure_ascii=False)[:400]))
        else:
            print("   %s = %s" % (k, json.dumps(v, ensure_ascii=False)[:200]))
