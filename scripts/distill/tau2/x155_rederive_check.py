# -*- coding: utf-8 -*-
"""x155 — 2단 재도출을 **켜기 전에** 두 가지를 잰다. 유료 0·로컬 32B.

x154 가 099 에서 0/5 → 5/5 를 냈다(궤적 + 자기 답 되돌림). 켜기 전 남은 질문 둘:

  ① **100 에서도 같은가** — 099 는 부하 진단, 100 은 능력 진단이었다(x149). 기전이 다르면
     같은 처방이 들을 이유가 없다.
  ② **1단이 틀리면 어떻게 되나** — 이 레버의 진짜 위험이다. 우리가 실어 나른 답이 틀렸을 때
     모델이 그대로 따라가면, 우리는 **오답에 권위를 붙여 배달**하는 장치를 만든 것이다([[25]]).
     그래서 *일부러 열화시킨 1단*(예치 기준을 못 건 표)을 실어 보고 추종률을 잰다.

  S0 ctx                  실제 궤적만                    ← 부정통제
  S1 ctx + 좋은 1단답      제대로 거른 표에서 나온 답
  S2 ctx + 열화 1단답      예치 기준을 **못 건** 표에서 나온 답  ← 위험 계량
  S3 ctx + 표             현행 라이브 기준선

gold 는 어디에도 주입하지 않는다 — 1단이 무엇을 내든 그것을 2단에 싣는다.
실행: py -3 x155_rederive_check.py [TAG] [N]
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

CASE = {"task_099": {"days": 730, "deposit": 30000},
        "task_100": {"days": 65, "deposit": 31000}}


def tables(a2, task):
    """(제대로 거른 표, 예치 기준을 못 건 표) — 후자가 '1단 열화' 조건이다."""
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    c = CASE[task]
    good = LG.eligible_text(c["days"], {}, maps, spec,
                            {"qualifying_deposit_usd": c["deposit"]}).strip()
    weak = LG.eligible_text(c["days"], {}, maps, spec, {}).strip()
    subs = sorted({s for ax in maps for s in maps[ax]})
    return good, weak, subs


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_elig_20260809i"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    a2 = load_domain_a2("banking_knowledge")
    Q = X.QUESTION

    for task in ("task_099", "task_100"):
        good, weak, subs = tables(a2, task)
        facts = X.FACTS[task]
        gold = X.GOLD[task]
        ctx = ("Here is a customer-service conversation so far.\n\n"
               + Y.render(Y.msgs_of(tag, task)))

        def stage1(tbl):
            ans = [X.ask(tbl + "\n\n" + facts + "\n\n" + Q, 0.0 if i == 0 else 0.7)
                   for i in range(n)]
            top = collections.Counter(" ".join(a.split())[:40] for a in ans).most_common()
            return top[0][0], top

        pg, tg = stage1(good)
        pw, tw = stage1(weak)
        print("\n######## %s (gold=%s)" % (task, gold))
        print("  1단 제대로 : %-24r %s" % (pg, tg))
        print("  1단 열화   : %-24r %s" % (pw, tw))

        def hand(p):
            return ("A separate analysis, working only from the policy constants on record and "
                    "this customer's stated situation, selects: %s" % p)

        arms = collections.OrderedDict()
        arms["S0 ctx"] = ctx + "\n\n" + facts + "\n\n" + Q
        arms["S1 ctx+good"] = ctx + "\n\n" + hand(pg) + "\n\n" + facts + "\n\n" + Q
        arms["S2 ctx+degraded"] = ctx + "\n\n" + hand(pw) + "\n\n" + facts + "\n\n" + Q
        arms["S3 ctx+table"] = ctx + "\n\n" + good + "\n\n" + facts + "\n\n" + Q

        for label, prompt in arms.items():
            ans = [X.ask(prompt, 0.0 if i == 0 else 0.7) for i in range(n)]
            hit = sum(1 for a in ans if gold.lower() in str(a).lower())
            foll = sum(1 for a in ans
                       if (pw.split()[0].lower() if pw else "\0") in str(a).lower())
            extra = "  추종(열화답)=%d/%d" % (foll, n) if label == "S2 ctx+degraded" else ""
            print("  %-18s gold %d/%d   %s%s"
                  % (label, hit, n,
                     collections.Counter(re.sub(r"\s+", " ", a)[:24] for a in ans).most_common(2),
                     extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
