# -*- coding: utf-8 -*-
"""banking frontier 실패의 정밀 arg-level 분류 (2026-07-13·[[08]]).
'attempted but action_match=False' = 실제 호출 인자 vs gold 인자 diff → 어느 인자가 틀리나.
gold action에 매칭되는 실제 호출(best-match) 찾아 mismatch 인자 집계. + 미도달/⋈ 유지."""
import json, glob, os
from collections import Counter, defaultdict

A2 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
# ★A2 `action_tool_executor` 삭제(2026-07-31·[[23]] 감사): 출처가 gold `action_checks[].requestor`
#   였고 env 도구 소속으로 7/7 재현되어 엔진이 도출하도록 바꿨다. 이 분석 스크립트도 같은 술어를 쓴다.
EXEC = {}   # 도구→실행주체: 에이전트 도구 목록에 있으면 assistant, 아니면 user(env 구조)



def calls(msgs):
    return [(m.get("role"), tc.get("name"), tc.get("arguments") or {})
            for m in msgs for tc in (m.get("tool_calls") or [])]


def best_match(name, gold_args, cl):
    """같은 tool 호출 중 gold와 인자 가장 많이 겹치는 것 (best attempt)."""
    cand = [a for r, n, a in cl if n == name]
    if not cand:
        return None
    def score(a):
        return sum(1 for k, v in gold_args.items() if str(a.get(k)) == str(v))
    return max(cand, key=score)


def main():
    files = sorted(glob.glob("C:/tmp/traj/*_banking.json"))
    arg_mismatch = Counter()          # (tool, arg) -> 틀린 횟수
    tool_fail = Counter()             # tool -> attempted-fail 횟수
    never_called = Counter()          # tool -> 미호출 실패
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for s in d["simulations"]:
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            cl = calls(s.get("messages") or [])
            called = {n for _, n, _ in cl}
            for ac in (ri.get("action_checks") or []):
                if ac.get("action_match"):
                    continue
                a = ac.get("action") or {}
                nm = a.get("name"); gold = a.get("arguments") or {}
                if nm not in called:
                    never_called[nm] += 1
                    continue
                tool_fail[nm] += 1
                bm = best_match(nm, gold, cl)
                if bm is None:
                    continue
                for k, v in gold.items():
                    if str(bm.get(k)) != str(v):
                        arg_mismatch[(nm, k)] += 1
    print("=== 미호출(never called) 실패 gold action (tool별) ===")
    for t, k in never_called.most_common(12):
        print("  %-34s %d" % (t, k))
    print("\n=== attempted-fail 실패 (tool별) ===")
    for t, k in tool_fail.most_common(12):
        print("  %-34s %d" % (t, k))
    print("\n=== ★어느 인자가 틀리나 (tool.arg별·attempted-fail 중 mismatch) ===")
    for (t, arg), k in arg_mismatch.most_common(30):
        print("  %-30s . %-24s %d" % (t, arg, k))


if __name__ == "__main__":
    main()
