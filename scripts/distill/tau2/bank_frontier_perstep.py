# -*- coding: utf-8 -*-
"""banking frontier 17모델 전수 per-step 실패 분류 (2026-07-13·[[08]]·[[48]] 서술형).
C:/tmp/traj/*_banking.json (user-sim gpt-5.2). requestor-aware(action_match) + per-step 서명.
출력: (1) 모델별 pass+retrieval (2) pooled 실패원인 분포 (3) hard core(전모델 공통실패)+원인."""
import json, glob, os
from collections import Counter, defaultdict

A2 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
# ★A2 `action_tool_executor` 삭제(2026-07-31·[[23]] 감사): 출처가 gold `action_checks[].requestor`
#   였고 env 도구 소속으로 7/7 재현되어 엔진이 도출하도록 바꿨다. 이 분석 스크립트도 같은 술어를 쓴다.
EXEC = {}   # 도구→실행주체: 에이전트 도구 목록에 있으면 assistant, 아니면 user(env 구조)

USER_EXEC = {t for t, e in EXEC.items() if e == "user"}
DISCOVER = {"unlock_discoverable_agent_tool", "call_discoverable_agent_tool"}
XFER = {"transfer_to_human_agents", "request_human_agent_transfer"}
GATHER = "get_user_information_by"
VERIFY_TOOL = "log_verification"


def calls_by_role(msgs):
    out = []
    for m in msgs:
        r = m.get("role")
        for tc in (m.get("tool_calls") or []):
            out.append((r, tc.get("name"), tc.get("arguments") or {}))
    return out


def keyval(args):
    return str(args.get("agent_tool_name") or args.get("card_type")
              or args.get("transaction_type") or "")


def classify_sim(s):
    """실패 sim → (failed_action_classes[list], sim_signatures[set])."""
    ri = s.get("reward_info") or {}
    msgs = s.get("messages") or []
    cl = calls_by_role(msgs)
    called_names = {n for _, n, _ in cl}
    # 도구별 사용된 key값 (right-requestor 무관·attempted 판정용)
    used_keys = defaultdict(set)
    for r, n, a in cl:
        used_keys[n].add(keyval(a))
    has_kb = "KB_search" in called_names
    classes = []
    for ac in (ri.get("action_checks") or []):
        if ac.get("action_match"):
            continue
        a = ac.get("action") or {}
        nm = a.get("name"); req = a.get("requestor")
        gkey = keyval(a.get("arguments") or {})
        attempted = nm in called_names
        if not attempted:
            if nm in DISCOVER or (has_kb and nm in DISCOVER):
                classes.append("reach-discovery(도구 미발견/미호출)")
            elif nm == VERIFY_TOOL:
                classes.append("verify-미완(log_verification 누락)")
            elif nm in USER_EXEC:
                classes.append("user-실행 미도달(제안/유도 실패)")
            else:
                classes.append("coverage-미완(도구 미호출)")
        else:
            # 시도했으나 오답
            if gkey and gkey not in used_keys.get(nm, set()):
                if nm in DISCOVER:
                    classes.append("operator-⋈(틀린 도구명)")
                elif nm in USER_EXEC:
                    classes.append("operand-⋈(틀린 값·카드 등)")
                else:
                    classes.append("operand-⋈(agent 틀린 인자)")
            else:
                classes.append("타인자/기준 오류(핵심키 정답)")
    # sim 서명
    sig = set()
    la = [m for m in msgs if m.get("role") == "assistant"][-1:] or [{}]
    lc = {tc.get("name") for tc in (la[0].get("tool_calls") or [])}
    if (not lc) or (lc <= XFER):
        sig.add("포기종결(조언/transfer)")
    # over-action: gold에 없는 write action 실행
    gold_names = {(ac.get("action") or {}).get("name") for ac in (ri.get("action_checks") or [])}
    write_done = {n for _, n, _ in cl if n in EXEC}
    if write_done - gold_names:
        sig.add("over-action(gold밖 write)")
    return classes, sig


def main():
    files = sorted(glob.glob("C:/tmp/traj/*_banking.json"))
    pooled = Counter(); pooled_sig = Counter()
    per_task_pass = defaultdict(lambda: [0, 0])   # task -> [pass, total] pooled
    per_task_fail_cause = defaultdict(Counter)
    print("=== 모델별 pass + retrieval ===")
    for f in files:
        nm = os.path.basename(f).replace("_banking.json", "")
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print("  %-14s LOAD FAIL %r" % (nm, e)); continue
        retr = d.get("info", {}).get("retrieval_config")
        np = nt = 0
        for s in d["simulations"]:
            r = (s.get("reward_info") or {}).get("reward")
            if r is None:
                continue
            tid = str(s["task_id"]); nt += 1
            per_task_pass[tid][1] += 1
            if r == 1.0:
                np += 1; per_task_pass[tid][0] += 1
                continue
            cls, sig = classify_sim(s)
            for c in cls:
                pooled[c] += 1; per_task_fail_cause[tid][c] += 1
            for g in sig:
                pooled_sig[g] += 1
        print("  %-14s pass=%.1f%% (n=%d) retr=%s" % (nm, 100 * np / max(nt, 1), nt, retr))
    print("\n=== POOLED 실패 gold-action 원인 (전 모델·전 실패) ===")
    tot = sum(pooled.values())
    for c, k in pooled.most_common():
        print("  %-34s %6d (%.1f%%)" % (c, k, 100 * k / max(tot, 1)))
    print("\n=== POOLED sim 서명 ===")
    for c, k in pooled_sig.most_common():
        print("  %-30s %d" % (c, k))
    # hard core: 전 모델 pooled pass율 낮은 태스크
    print("\n=== HARD CORE (pooled pass ≤ 10%·전 frontier 공통실패) + 지배원인 ===")
    hard = [(t, p[0], p[1]) for t, p in per_task_pass.items() if p[1] >= 10 and p[0] / p[1] <= 0.10]
    hard.sort(key=lambda z: z[1] / z[2])
    print("  hard-core 태스크 수:", len(hard), "/", len(per_task_pass))
    for t, p, n in hard[:25]:
        cause = per_task_fail_cause[t].most_common(1)
        print("  %-10s pass=%d/%d  지배원인=%s" % (t, p, n, cause[0] if cause else "-"))
    # hard core 원인 집계
    hc_cause = Counter()
    for t, p, n in hard:
        for c, k in per_task_fail_cause[t].items():
            hc_cause[c] += k
    print("\n=== HARD CORE 지배원인 집계 ===")
    for c, k in hc_cause.most_common():
        print("  %-34s %d" % (c, k))


if __name__ == "__main__":
    main()
