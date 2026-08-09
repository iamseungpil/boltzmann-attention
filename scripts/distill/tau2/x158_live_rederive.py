# -*- coding: utf-8 -*-
"""x158 — **라이브가 실제로 조립하는 재도출 프롬프트**로 재본다. 유료 0.

왜 또 재나: 이 세션에서 반복된 실패가 *"측정된 조건과 다른 구성을 만들어 놓고 같은 것으로
취급"* 이었다. x154/x155 의 5/5 는 x149 의 **손으로 쓴 FACTS**(보유 계좌·사업체 신설·예치액을
문장으로 서술)를 썼는데, 라이브가 조립하는 사실 문장은 **엔진이 쥔 수치뿐**이라 훨씬 얇다:

    days since the earliest account was opened = 730
    qualifying_deposit_usd = 30000

얇아진 사실로도 1단이 서는지 **켜기 전에** 확인한다. 안 서면 5/5 는 우리 것이 아니다.

  L1 live-compose   표 + (엔진 사실) + (형식화된 목적)      ← 라이브 그대로
  L2 x149-facts     표 + 손으로 쓴 FACTS + 고정 질문        ← 5/5 를 냈던 조건(기준선)
  L3 live-노목적    표 + 엔진 사실만                         ← 목적 구절의 기여 분리

실행: py -3 x158_live_rederive.py [N]
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
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

CASE = {"task_099": (730, 30000), "task_100": (65, 31000)}
# 목적은 궤적에 실재하는 손님 말에서 온다 — 여기서는 형식화 결과를 손으로 넣지 않고
# 실제 프롬프트를 그대로 태워 본다(아래 main 참조).


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}

    for task, (days, dep) in CASE.items():
        gold = X.GOLD[task]
        table = LG.eligible_text(days, {}, maps, spec,
                                 {"qualifying_deposit_usd": dep}).strip()
        facts = ("days since the earliest account was opened = %d\n"
                 "qualifying_deposit_usd = %d" % (days, dep))
        # 목적: 형식화가 낼 법한 형태를 **손님 말에서** 그대로 옮긴 것(프롬프트 실측은 라이브에서).
        obj = ("the customer wants the biggest possible referral bonus for themselves"
               if task != "task_098" else
               "the customer wants the biggest combined bonus for both of them")
        tpl = spec["rederive_prompt"]
        arms = collections.OrderedDict()
        arms["L1 live-compose"] = tpl.format(table=table, facts=facts, asked=obj)
        arms["L2 x149-facts"] = table + "\n\n" + X.FACTS[task] + "\n\n" + X.QUESTION
        arms["L3 live-노목적"] = tpl.format(table=table, facts=facts, asked="")
        print("\n######## %s (gold=%s)" % (task, gold))
        for label, p in arms.items():
            ans = [X.ask(p, 0.0 if i == 0 else 0.7) for i in range(n)]
            hit = sum(1 for a in ans if gold.lower() in str(a).lower())
            print("  %-18s %d/%d   %s" % (label, hit, n,
                  collections.Counter(re.sub(r"\s+", " ", a)[:26] for a in ans).most_common(2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
