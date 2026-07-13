# -*- coding: utf-8 -*-
"""banking 재채점 + gold↔우리샘플 경로 분석 (2026-07-13·requestor-aware·[[08]]).
버그였던 assistant-only 스캔 폐기 → 하네스 action_checks[].action_match(정확 채점) 사용.
각 실패 gold action을: (A)시도했으나 인자틀림=OPERAND/⋈  (B)미도달(발견체인 실패)=REACH
                       (C)coverage(일부 action만)  로 궤적 대조 분류.
frontier=aggregate만(궤적 소실·[[47]]): opus4.5 24.7·gpt5.5 37.4 vs 목표격차.
용법: python bank_rescore_pathdiff.py [floor.gz]"""
import gzip, json, sys, os
from collections import Counter, defaultdict

P = sys.argv[1] if len(sys.argv) > 1 else \
    "C:/workspace/ba-frft/reports/facet_rft_2026/sim_results/bankxfer_floor_bank_t4.results.json.gz"
d = json.load(gzip.open(P))
sims = d["simulations"]
tasks = {str(t.get("id")): t for t in (d.get("tasks") or [])}


def all_calls(msgs):
    """(role, tool_name, args) 전 호출 — user·assistant 양쪽 (banking action=user-실행 포함)."""
    out = []
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            out.append((m.get("role"), tc.get("name"), tc.get("arguments") or {}))
    return out


DISCOVERY = {"KB_search", "give_discoverable_user_tool", "unlock_discoverable_agent_tool",
             "call_discoverable_agent_tool"}

npass = ninfra = 0
fail_action_class = Counter()      # 실패 gold action별 분류
task_fail_class = Counter()        # sim별 지배 실패
reward_bases = Counter()
per_task_pass = defaultdict(list)

for x in sims:
    tid = str(x["task_id"])
    ri = x.get("reward_info") or {}
    rew = ri.get("reward")
    if rew is None:
        ninfra += 1
        continue
    dbm = ((ri.get("db_check") or {}).get("db_match"))
    passed = (rew == 1.0)
    per_task_pass[tid].append(1 if passed else 0)
    reward_bases[tuple(ri.get("reward_basis") or [])] += 1
    if passed:
        npass += 1
        continue
    # ── 실패: gold action별 경로 대조 ──
    msgs = x.get("messages") or []
    calls = all_calls(msgs)
    called_by = defaultdict(set)       # tool -> {roles that called it}
    for role, nm, args in calls:
        called_by[nm].add(role)
    has_discovery = any(nm in DISCOVERY for _, nm, _ in calls)
    acs = ri.get("action_checks") or []
    sim_classes = []
    for ac in acs:
        if ac.get("action_match"):
            continue
        a = ac.get("action") or {}
        nm = a.get("name"); req = a.get("requestor")
        attempted = req in called_by.get(nm, set()) or bool(called_by.get(nm))
        if attempted:
            cls = "OPERAND/⋈(시도·인자틀림)"
        elif has_discovery:
            cls = "REACH(발견체인 미완)"
        else:
            cls = "NO-START(발견조차 안함)"
        fail_action_class[cls] += 1
        sim_classes.append(cls)
    # env/nl 기준 실패(action 다 맞았는데 db_fail)
    if not sim_classes:
        if (ri.get("env_assertions") is not None) or (ri.get("nl_assertions") is not None):
            sim_classes.append("ENV/NL(action OK·상태/발화 기준)")
        else:
            sim_classes.append("OTHER(action_check 없음)")
    # sim 지배 = 첫 실패 클래스
    task_fail_class[sim_classes[0]] += 1

n = npass + sum(task_fail_class.values())
print("=== banking floor 재채점 (n=%d valid, infra=%d) ===" % (n, ninfra))
print("PASS(reward=1.0): %d (%.1f%%)" % (npass, 100 * npass / max(n, 1)))
print("reward_basis 분포:", {"+".join(k): v for k, v in reward_bases.items()})
print("\n=== 실패 gold-action 분류 (action_match=False·전 action) ===")
for c, k in fail_action_class.most_common():
    print("  %-28s %d" % (c, k))
print("\n=== sim 지배 실패 (첫 실패 클래스) ===")
for c, k in task_fail_class.most_common():
    print("  %-28s %d" % (c, k))
# robust: 태스크별 4-trial 전패 vs 부분
allfail = sum(1 for t, r in per_task_pass.items() if not any(r))
anypass = sum(1 for t, r in per_task_pass.items() if any(r))
print("\n=== 태스크 단위(robust) ===")
print("  태스크 수:", len(per_task_pass), "| 전-trial 전패:", allfail, "| 1+ pass:", anypass)
