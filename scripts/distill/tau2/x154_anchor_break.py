# -*- coding: utf-8 -*-
"""x154 — 자기-정박을 깨는 것은 무엇인가. 유료 0·로컬 32B([[18]]·[[57]]).

## 왜 이 프로브인가

099 는 라이브 **12 sim 전수 실패**이고 제출물 **10/12 가 손님이 이미 보유한 계좌**다
(`Navy Blue`·`Cobalt Blue`·`Hunter Green`). 통과-집합을 눈앞에 두고도 그렇다.
이미 아는 것 둘:
  · x149 : 깨끗한 표+사실+질문 = **5/5** · 같은 정보를 실제 궤적에 담으면 **0/5** ⇒ 부하
  · x150 : 도구 출력을 빼도 무효(Q1·Q2 0/5) · **에이전트 자기 발화**를 빼야 움직인다(Q3 2/5)
  · C310 : 정박원은 우리 문구가 아니라 **원 도구 출력**이라 은폐로는 못 없앤다

⇒ 남은 질문은 *"어떻게 더 보여줄까"* 가 아니다(그 길은 닫혔다). **깨끗한 문맥에서 나온
   판단을 오염된 문맥 안으로 되돌려 넣으면 채택되는가**다.

## 설계 — 2단

1단(깨끗) : 표 + 손님 사실 + 질문만으로 **모델 자신이** 고른다. gold 를 넣지 않는다 —
             무엇이 나오든 그것을 2단에 싣는다(그래서 이 프로브는 gold 무참조다).
2단(오염) : 실제 궤적 + 그 답을 되돌려 넣고 다시 묻는다.

  R0 ctx                  실제 궤적만                          ← 부정통제(기존 0/5)
  R1 ctx + 1단답          *"깨끗한 재분석은 X 를 고른다"*
  R2 ctx + 표 + 1단답     통과 집합까지 함께
  R3 ctx + 보유-사실      *"당신이 방금 말한 것은 손님이 이미 보유한 계좌다"* (정박 반증)
  R4 ctx + 표             표만(=현행 라이브) — R1/R2 의 증분을 가르는 기준선

실행: py -3 x154_anchor_break.py [TAG] [N]
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

TASK = "task_099"
DAYS, DEPOSIT = 730, 30000
HELD = "Navy Blue, Cobalt Blue and Hunter Green"     # 궤적에 실재(손님 발화·도구 출력)


def table(a2):
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    return LG.eligible_text(DAYS, {}, maps, spec,
                            {"qualifying_deposit_usd": DEPOSIT}).strip()


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_elig_20260809i"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    a2 = load_domain_a2("banking_knowledge")
    tbl = table(a2)
    facts = X.FACTS[TASK]
    Q = X.QUESTION
    ctx = "Here is a customer-service conversation so far.\n\n" + Y.render(Y.msgs_of(tag, TASK))

    # ── 1단: 깨끗한 문맥에서 모델 자신이 고른다 (gold 무참조) ────────────────
    clean_q = tbl + "\n\n" + facts + "\n\n" + Q
    picks = [X.ask(clean_q, 0.0 if i == 0 else 0.7) for i in range(n)]
    top = collections.Counter(" ".join(p.split())[:40] for p in picks).most_common()
    pick = top[0][0]
    print("1단(깨끗) 답 분포: %s" % top)
    print("→ 2단에 실을 답: %r  (gold 를 보고 고른 것이 아니다)\n" % pick)

    handoff = ("A separate analysis, working only from the policy constants on record and this "
               "customer's stated situation, selects: %s" % pick)
    held = ("Note on the accounts this customer already holds (%s): holding an account type does "
            "not make it the right one to refer someone to, and it does not restrict referring "
            "someone to a different type." % HELD)

    arms = collections.OrderedDict()
    arms["R0 ctx"] = ctx + "\n\n" + facts + "\n\n" + Q
    arms["R1 ctx+handoff"] = ctx + "\n\n" + handoff + "\n\n" + facts + "\n\n" + Q
    arms["R2 ctx+table+handoff"] = ctx + "\n\n" + tbl + "\n\n" + handoff + "\n\n" + facts + "\n\n" + Q
    arms["R3 ctx+held-note"] = ctx + "\n\n" + tbl + "\n\n" + held + "\n\n" + facts + "\n\n" + Q
    arms["R4 ctx+table"] = ctx + "\n\n" + tbl + "\n\n" + facts + "\n\n" + Q

    gold = X.GOLD[TASK]
    out = collections.OrderedDict()
    for label, prompt in arms.items():
        ans = [X.ask(prompt, 0.0 if i == 0 else 0.7) for i in range(n)]
        hit = sum(1 for a in ans if gold.lower() in str(a).lower())
        out[label] = (hit, collections.Counter(re.sub(r"\s+", " ", a)[:24]
                                               for a in ans).most_common(2))
        print("%-22s gold=%-12s %d/%d   %s" % (label, gold, hit, n, out[label][1]))

    print("\n=== 요약 ===")
    print("  1단(깨끗)             %d/%d" % (sum(1 for p in picks if gold.lower() in p.lower()), n))
    for k, (h, _) in out.items():
        print("  %-22s %d/%d" % (k, h, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
