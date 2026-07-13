# -*- coding: utf-8 -*-
"""STEP 0 (설계 R4): banking hard-core param 실패의 compute-사정권 정량 (무료·라이브 전제).
사정권 = 우측 도구 호출 ∧ 계산-대상 param 오답 ∧ 필요 입력 문맥 수집(silent-repair 가능).
사정권 밖 = 틀린 도구(operator-⋈)·param 미제공·계산불가형·입력 미수집(reach)."""
import json, glob, os
from collections import Counter, defaultdict

# 결정론 op 대상 param → 필요 입력(그 param을 계산하려면 있어야 할 다른 nested 필드)
COMPUTED = {
    "customer_max_liability_amount": ["transaction_date", "discovery_date", "disputed_amount"],
    "amount_difference": [],                      # diff — 입력은 도구별 상이(있으면 계산가능 추정)
    "expected_apy": [],                           # lookup by account — record 필요
    "provisional_credit_eligible": ["disputed_amount"],
    "eligible_for_provisional_credit": ["disputed_amount"],
}
FORMALIZE = {"account_class"}                     # NL-판단(결정론 사정권 밖·learn)


def nd(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            r = json.loads(v)
            return r if isinstance(r, dict) else {}
        except Exception:
            return {}
    return {}


def main():
    # hard-core 태스크(전 frontier pooled ≤10%) 먼저 산출
    per_task = defaultdict(lambda: [0, 0])
    files = sorted(glob.glob("C:/tmp/traj/*_banking.json"))
    data = {}
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        data[f] = d
        for s in d["simulations"]:
            r = (s.get("reward_info") or {}).get("reward")
            if r is None:
                continue
            t = str(s["task_id"]); per_task[t][1] += 1
            if r == 1.0:
                per_task[t][0] += 1
    hard = {t for t, p in per_task.items() if p[1] >= 10 and p[0] / p[1] <= 0.10}

    scope = Counter()
    param_scope = Counter()
    for f, d in data.items():
        for s in d["simulations"]:
            if str(s["task_id"]) not in hard:
                continue
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            cl = [(tc.get("name"), nd(tc.get("arguments")))
                  for m in (s.get("messages") or []) for tc in (m.get("tool_calls") or [])]
            for ac in (ri.get("action_checks") or []):
                if ac.get("action_match"):
                    continue
                a = ac.get("action") or {}
                if a.get("name") != "call_discoverable_agent_tool":
                    scope["기타도구(non-discoverable)"] += 1
                    continue
                g = nd(a.get("arguments")); gtool = g.get("agent_tool_name"); gn = nd(g.get("arguments"))
                # 우측 도구 호출 있나
                same = [nd(ar.get("arguments")) for n, ar in cl
                        if n == "call_discoverable_agent_tool" and str(ar.get("agent_tool_name")) == str(gtool)]
                if not same:
                    scope["사정권밖: 틀린도구/미호출(operator-⋈·reach)"] += 1
                    continue
                an = same[0]
                # 어느 nested param이 틀리나
                mism = [k for k, v in gn.items() if str(an.get(k)) != str(v)]
                classified = False
                for k in mism:
                    if k in FORMALIZE:
                        param_scope["formalize형(account_class·learn)"] += 1; classified = True
                    elif k in COMPUTED:
                        inputs = COMPUTED[k]
                        got = an.get(k) is not None            # 에이전트가 값 시도함
                        ins_ok = all(an.get(i) not in (None, "") for i in inputs) if inputs else True
                        if got and ins_ok:
                            param_scope["★compute-사정권(입력수집·값오답)"] += 1; classified = True
                        elif not ins_ok:
                            param_scope["사정권밖: 입력미수집(reach)"] += 1; classified = True
                        else:
                            param_scope["사정권밖: param 미제공"] += 1; classified = True
                    else:
                        param_scope["비계산 param(범주/참조/기타)"] += 1
                if classified:
                    scope["우측도구·계산param 관련"] += 1
                elif mism:
                    scope["우측도구·비계산param만"] += 1
                else:
                    scope["우측도구·param맞음(타기준)"] += 1

    print("=== hard-core 태스크:", len(hard), "/", len(per_task), "===")
    print("\n=== 실패 action 사정권 (sim×action) ===")
    tot = sum(scope.values())
    for k, v in scope.most_common():
        print("  %-42s %6d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
    print("\n=== 틀린 param 사정권 (param 단위) ===")
    ptot = sum(param_scope.values())
    for k, v in param_scope.most_common():
        print("  %-42s %6d (%.1f%%)" % (k, v, 100 * v / max(ptot, 1)))
    incore = param_scope.get("★compute-사정권(입력수집·값오답)", 0)
    print("\n★compute-사정권 param 비율(전 틀린 param 중): %.1f%%" % (100 * incore / max(ptot, 1)))


if __name__ == "__main__":
    main()
