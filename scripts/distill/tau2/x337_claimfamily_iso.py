# -*- coding: utf-8 -*-
r"""x337 — **허위-주장/요구-조건 가족**(0% 태스크 9개)의 대표 063 격리. `t2_probe` 정본 호출.

## 왜 이 가족인가 (census 재분석·2026-08-16)

`bank_task_taxonomy_20260810.json`(4229 sim) 재분석: 0% 태스크 **59**개가 둘로 갈린다 —
**썼는데 틀림**(`fail_wrote` 45~100%) ↔ **아예 안 씀**(0). 앞쪽에 `traps=주장` 이 몰려 있고
(055·057·059·063·064·066·067·068·069·081), **write 는 이미 하므로 문턱이 가장 얕다**.
069 는 벤치 결함이라 뺀다([[23]]) ⇒ 실효 **9개**.

## 대표 063 의 실제 모양 (궤적 직독·gold 무접촉)

손님: *"$8,000 있고 1년 이자를 최대로 · **종이 명세서 필수** · $10~15 차이도 중요"*
우리: `Diamond Elite`(savings) + `Platinum Rewards`(card) → gold 는 `Silver Plus` + `Silver Rewards`.
`Diamond Elite` 는 **최소잔고 $250,000**(055 감사에서 문서 축자 확인) ⇒ $8,000 손님은 **자격 미달**.
⇒ 055 와 **같은 축**(선언 수치 대 손님 금액)이고, 거기서는 재료 전달만으로 **0/24 → 24/24** 였다(C494).

## 셀 4 (컷 = 063#0 msg 8 = 두 요구를 다 듣고 재확인한 자리)

    A_REF     라이브 축자(재료 없음)              ← 실패 재현
    B_DOCS    + savings 문서 재료(정본 경로)       ← 전달만으로 닫히나
    C_CARD    + savings·card 재료 둘 다            ← 두 결정을 같이 물으면 섞이나(x248 경고)
    D_FLIP    B 와 같되 손님 금액을 **$500,000**   ← ★부정통제: Diamond Elite 가 **옳은** 자리

⚠통제 유효성은 **기계가 본다**(`t2_probe`): 뒤집기 지표(DIAMOND)가 어느 팔에서도 안 열리면
  판정 보류. 여기서는 `A_REF` 가 그것을 낸다고 예상되므로 도달 가능성이 확보될 것이다 —
  그러나 **예상은 근거가 아니다**. 코드가 판정한다.
⚠[[62]] ③: 엔진은 읽어 나르기만. 고르지 않고 지목하지 않는다.
⚠금액 치환은 문자열 치환뿐(의미 판단 0·[[59]]).

실행(리모트·8141·유료 런이 없을 때만):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x337_claimfamily_iso.py [k] [nb]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_probe as P                                              # noqa: E402
import t2_search as S                                             # noqa: E402

TAG, TASK, CUT = "bank_t7295_b_20260815n", "task_063", 8
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
A2P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "a2", "banking_knowledge.specific.json")
AMOUNT = "$8,000"
RICH = "$500,000"
ASK = ("\n[instruction] Do NOT call any tool. Reply with ONE line only: the full official name of "
       "the ONE savings account class you would open for this customer, nothing else.")
MARKS = {"SILVERPLUS": "SILVER PLUS", "DIAMOND": "DIAMOND ELITE", "GOLD": "GOLD ACCOUNT"}


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    site = P.site(TAG, TASK, CUT)
    if AMOUNT not in site["base"]:
        print("컷 %d 안에 손님 금액(%s)이 없다 — 정보-맞춤 위반 ⇒ 중단" % (CUT, AMOUNT))
        return 1
    a2 = json.load(io.open(A2P, encoding="utf-8"))
    sav, st1 = S.material_for(a2, "savings_accounts", doc_dir=DOCS, windowed="general", per_doc=400)
    card, st2 = S.material_for(a2, "credit_cards", doc_dir=DOCS, windowed="general", per_doc=400)

    flip_site = dict(site)
    flip_site["base"] = site["base"].replace(AMOUNT, RICH)
    if flip_site["base"] == site["base"]:
        print("부정통제 치환 실패 — 중단")
        return 1

    print("x337 · savings 재료 %d자(유지 %d) · card 재료 %d자(유지 %d)\n"
          % (len(sav), st1["kept"], len(card), st2["kept"]))
    P.run("x337", site,
          [("A_REF", ""), ("B_DOCS", sav), ("C_CARD", sav + "\n\n" + card)],
          MARKS,
          "B SILVERPLUS ≥18 → 전달로 닫힌다(가족 9개의 표적 확정) · B≈A → 전달 아님(다른 축) · "
          "C < B −5 → 두 결정을 섞으면 나빠진다(x248 재현)",
          ASK, None, k, nb)
    print("\n── 부정통제(금액 %s → %s · Diamond Elite 가 옳은 자리) ──" % (AMOUNT, RICH))
    P.run("x337-neg", flip_site, [("A_REF", ""), ("D_FLIP", sav)], MARKS,
          "D DIAMOND 가 B 의 DIAMOND 와 갈리면 모델이 금액을 쓴다(통제 유효)",
          ASK, ("D_FLIP", "A_REF", "DIAMOND"), k, nb)


if __name__ == "__main__":
    sys.exit(main() or 0)
