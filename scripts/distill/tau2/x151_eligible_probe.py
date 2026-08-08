# -*- coding: utf-8 -*-
"""x151 — **통과 집합**(엔진이 거른 것)이 선택을 닫는가. 유료 0·로컬 32B([[18]]·[[57]]).

x150 이 낸 진단: 100 은 **능력**(같은 표 0/5 · 자격 미달 행을 뺀 표 5/5), 099 는 **부하**
(깨끗한 표 5/5 · 실제 궤적 문맥 0/5). 처방이 하나로 모인다 — **엔진이 걸러 통과 집합만 준다**.
그 처방은 이제 구현돼 있다(`t2_ledger.eligible_text` + A2 `eligible` 선언). 이 프로브는
**우리 층이 실제로 내보내는 문장 그대로**를 넣고 재는 것이다 — 프로브용으로 다시 만든 표가
아니다. 그래야 여기서 산 것이 라이브에서도 산다.

  100 계열 (능력 축 · 손님 관계기간 65일)
    A0 raw-table       A3 전체 표 + 사실 + 질문                    ← x150 P0 대응(0/5였다)
    A1 eligible        **우리 문장**(통과 집합)만 + 사실 + 질문     ← 신규 레버
    A2 eligible+table  통과 집합 + 전체 표 둘 다                   ← handoff §4 ⚠의 미결 질문

  099 계열 (부하 축 · 손님 관계기간 약 2년)
    B0 raw-table       A3 전체 표 + 사실 + 질문
    B1 eligible        우리 문장만 + 사실 + 질문
    B2 ctx+table       **실제 궤적 문맥** + 전체 표                 ← x150 Q0 대응(0/5였다)
    B3 ctx+eligible    실제 궤적 문맥 + 우리 문장                   ← 부하 아래서도 사는가

부정통제([[57]]): A0/B0·B2 가 그 통제다 — 같은 모델·같은 질문·같은 사실에서 **거른 것만**
다르다. 통과 집합 arm 이 올라가고 raw arm 이 그대로면 오른 것은 필터이지 재시도가 아니다.

⚠이 표는 **이름 통일 후**의 A3 다(x152). 통일 전에는 `World Blue Balance`(보너스 300·문턱 없음)가
  필터를 그냥 통과했다 — 그 상태로 이 레버를 켰으면 65일 손님에게 오답을 최고액으로 얹어 줬다.

실행: py -3 x151_eligible_probe.py [TAG] [N]
"""
import collections
import json
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

DOMAIN = "banking_knowledge"
Q = X.QUESTION
# 각 sim 의 손님 사실에서 **경과일**만 뽑아 온다(궤적에 실재하는 수·gold 무참조).
DAYS = {"task_100": 65, "task_099": 730}


def our_sentence(a2, task):
    """라이브가 실제로 내보내는 통과-집합 문장. 프로브가 문구를 다시 짓지 않는다."""
    specs = a2.get("ledger_metrics") or []
    spec = next((s for s in specs if s.get("eligible_text")), None)
    if spec is None:
        raise SystemExit("A2 에 eligible_text 선언이 없다 — 배선 확인")
    rows = (a2.get("policy_ontology") or {}).get("rows") or ()
    axes = list((spec.get("eligible") or {}).get("show_axes") or ())
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in axes}
    return LG.eligible_text(DAYS[task], {}, maps, spec).strip()


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_remeas_20260808f"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    a2 = load_domain_a2(DOMAIN)
    table = X.a3_table()

    arms = collections.OrderedDict()
    f100, f099 = X.FACTS["task_100"], X.FACTS["task_099"]
    e100, e099 = our_sentence(a2, "task_100"), our_sentence(a2, "task_099")

    # ★1차(2026-08-08) 결과가 문구 축을 열었다: 필터는 옳게 돌았는데(100 에서 `World Blue`가
    #   실제로 빠졌다) 답이 안 맞았고, **099 는 5/5 → 0/5 로 떨어졌다**. 내용은 x150 P2 와
    #   같은데 우리 문장에는 P2 에 없던 **머리말·꼬리말**이 있다 — *"제외했다 · 남은 것들이다"*.
    #   가설: 완결을 주장하는 순간 모델이 **자기 검사를 끈다**(예치 하한을 더 안 본다).
    #   그래서 행은 그대로 두고 **감싸는 말만** 뺀 arm 을 둔다(정보량 동일·[[18]]).
    #   2차가 그 가설을 확증했다: 행이 같은데 **감싸는 말만** 빼니 100 이 0/5 → **5/5**.
    #   그리고 099 는 감싸는 말과 무관하게 전부 0/5 였는데, B0(raw) 와 B4(bare) 를 축자 대조하니
    #   **줄 수·글자 수까지 같고 차이는 열 순서 하나**였다 — `qualifying_deposit_usd` 가
    #   `annual_referral_limit` 뒤로 밀려 있었다(내가 x149 표의 순서를 임의로 바꿔 둔 것).
    #   099 를 가르는 축이 예치 하한이므로 그 열이 뒤에 있으면 안 보인다. A2 순서를 복원했다.
    # ⇒ 3차는 **감싸는 말을 부분 격리**한다: 머리말만 · 꼬리말만 · 짧고 참인 머리말.
    NEUTRAL = table.splitlines()[0]
    SHORT = ("Policy constants on record, for the products not already ruled out by this "
             "customer's tenure or by this year's counts (each value from a retrieved document):")

    def parts(text):
        rows = [l for l in text.splitlines() if l.startswith("  ")]
        other = [l for l in text.splitlines() if l and not l.startswith("  ")]
        return "\n".join(rows), other[0], other[-1]

    r100, h100, t100 = parts(e100)
    r099, h099, t099 = parts(e099)

    arms[("task_100", "A0 raw-table")] = table + "\n\n" + f100 + "\n\n" + Q
    arms[("task_100", "A1 full")] = e100 + "\n\n" + f100 + "\n\n" + Q
    arms[("task_100", "A3 bare")] = NEUTRAL + "\n" + r100 + "\n\n" + f100 + "\n\n" + Q
    arms[("task_100", "A4 header-only")] = h100 + "\n" + r100 + "\n\n" + f100 + "\n\n" + Q
    arms[("task_100", "A5 footer-only")] = (NEUTRAL + "\n" + r100 + "\n" + t100 + "\n\n"
                                            + f100 + "\n\n" + Q)
    arms[("task_100", "A6 short")] = SHORT + "\n" + r100 + "\n\n" + f100 + "\n\n" + Q

    ctx = Y.render(Y.msgs_of(tag, "task_099"))
    head = "Here is a customer-service conversation so far.\n\n"
    b099 = NEUTRAL + "\n" + r099
    s099 = SHORT + "\n" + r099
    arms[("task_099", "B0 raw-table")] = table + "\n\n" + f099 + "\n\n" + Q
    arms[("task_099", "B1 full")] = e099 + "\n\n" + f099 + "\n\n" + Q
    arms[("task_099", "B2 ctx+table")] = head + ctx + "\n\n" + table + "\n\n" + f099 + "\n\n" + Q
    arms[("task_099", "B3 ctx+full")] = head + ctx + "\n\n" + e099 + "\n\n" + f099 + "\n\n" + Q
    arms[("task_099", "B4 bare")] = b099 + "\n\n" + f099 + "\n\n" + Q
    arms[("task_099", "B6 short")] = s099 + "\n\n" + f099 + "\n\n" + Q
    arms[("task_099", "B7 ctx+short")] = head + ctx + "\n\n" + s099 + "\n\n" + f099 + "\n\n" + Q

    print("=== 통과 집합 (task_100 · %d일) ===" % DAYS["task_100"])
    print(e100)
    print()
    print("=== 통과 집합 (task_099 · %d일) ===" % DAYS["task_099"])
    print(e099)
    print()

    res = collections.OrderedDict()
    for key, prompt in arms.items():
        answers = []
        for i in range(n):
            try:
                answers.append(X.ask(prompt, 0.0 if i == 0 else 0.7))
            except Exception as e:
                answers.append("ERR %r" % (e,))
        res[key] = answers
        gold = X.GOLD[key[0]]
        hit = sum(1 for a in answers if gold.lower() in str(a).lower())
        print("%-9s %-18s gold=%-12s %d/%d   %s"
              % (key[0], key[1], gold, hit, len(answers),
                 collections.Counter(re.sub(r"\s+", " ", a)[:26] for a in answers).most_common(2)))

    print()
    print("=== 요약 ===")
    for key, answers in res.items():
        gold = X.GOLD[key[0]]
        print("  %-9s %-18s %d/%d" % (key[0], key[1],
                                      sum(1 for a in answers if gold.lower() in str(a).lower()),
                                      len(answers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
