#!/usr/bin/env python
"""ⓟ1 비결정 원천 전수 궤적 census (2026-06-14 사용자 재조사):
det 런(temp0+enforce-eager+seqs1+seed로도 4-trial 0% 동일)의 *첫 발산 지점*을 태스크별로 찾아
**user 턴(gpt-4.1 user-sim API 비결정) vs assistant 턴(7B vLLM 커널 비결정)**으로 귀속.

핵심 판별: batch-invariant는 *agent vLLM*만 결정론화 → 발산 root가
  - user 턴이면 = user-sim API가 원천 → **batch-invariant로 못 고침**(user-sim도 결정론화 필요).
  - assistant 턴(앞 context 동일한데 다름)이면 = agent vLLM → batch-invariant가 고침.

Usage: t2_divergence_census.py --simdir <results.json 폴더>
"""
import argparse, json
from collections import defaultdict, Counter


def sig(m):
    """메시지 시그니처 — role별 비교 단위."""
    r = m.get("role")
    c = (m.get("content") or "")
    if r == "assistant":
        tc = m.get("tool_calls") or []
        acts = []
        for t in tc:
            f = t.get("function", {}) if isinstance(t, dict) else {}
            acts.append(str(f.get("name")) + "(" + str(f.get("arguments", "")) + ")")
        return ("assistant", c.strip(), tuple(acts))
    return (r, c.strip(), ())


def is_action(m):
    return m.get("role") == "assistant" and bool(m.get("tool_calls"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simdir", required=True)
    a = ap.parse_args()
    d = json.load(open(f"{a.simdir}/results.json"))
    by = defaultdict(list)
    for s in d["simulations"]:
        by[s["task_id"]].append(s)

    role_first_div = Counter()    # 첫 발산 role
    action_first_div = Counter()  # 첫 *행동* 발산 role (직전까지 행동 동일)
    n_tasks = n_divergent = 0
    examples = []
    for tid, sims in by.items():
        if len(sims) < 2:
            continue
        n_tasks += 1
        seqs = [[sig(m) for m in s.get("messages", [])] for s in sims]
        # 첫 발산 (전 시그니처)
        L = min(len(x) for x in seqs)
        first = None
        for i in range(L):
            if len({seqs[t][i] for t in range(len(seqs))}) > 1:
                first = i
                break
        if first is None and len({len(x) for x in seqs}) > 1:
            first = L  # 길이만 다름(앞은 동일) = 발산은 L 위치(누군가 더 말함)
        if first is None:
            continue  # 완전 동일
        n_divergent += 1
        role_at = seqs[0][first][0] if first < L else "(length)"
        role_first_div[role_at] += 1
        # 첫 *행동* 발산: 행동 시그니처만 비교
        amaps = []
        for s in sims:
            amaps.append([sig(m) for m in s.get("messages", []) if is_action(m)])
        aL = min(len(x) for x in amaps)
        afirst_role = None
        for i in range(aL):
            if len({amaps[t][i] for t in range(len(amaps))}) > 1:
                # 이 행동 직전에 발산한 turn의 role 찾기 (root)
                afirst_role = role_at  # 전체 첫발산 role이 root
                break
        if afirst_role:
            action_first_div[afirst_role] += 1
        if len(examples) < 6 and first < L:
            ctx = seqs[0][first - 1][:2] if first > 0 else ("(start)",)
            examples.append((tid, first, role_at, [seqs[t][first][1][:45] for t in range(len(seqs))]))

    print(f"[divergence census] tasks={n_tasks} divergent={n_divergent}")
    print("=== 첫 발산 turn의 role (비결정 root) ===")
    for r, c in role_first_div.most_common():
        print(f"  {r:>12}: {c:>3} ({c/max(n_divergent,1)*100:.0f}%)")
    print("  ★user = gpt-4.1 user-sim API 비결정(batch-invariant 무관) / assistant = 7B vLLM(batch-invariant 고침)")
    print("=== 첫 *행동* 발산의 root role ===")
    for r, c in action_first_div.most_common():
        print(f"  {r:>12}: {c:>3}")
    print("=== 예시 (첫 발산) ===")
    for tid, pos, role, texts in examples:
        print(f"  task={tid} pos={pos} role={role}")
        for j, t in enumerate(texts):
            print(f"     trial{j}: {t!r}")


if __name__ == "__main__":
    main()
