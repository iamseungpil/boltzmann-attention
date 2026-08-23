# -*- coding: utf-8 -*-
"""일회성 — R6(2026-08-24 · refute_3 ⒠): `get_reward_discrepancies`(ratefix) `return_template`
에서 **전제-딸린 갱신 명령문**을 제거한다. 3사본(specific/gate/split-core) 동기.

── 결함 ────────────────────────────────────────────────────────────────────────
구 반환문 축자: *"The CORRECT total reward per policy is shown for each — **after its dispute is
resolved, update that transaction's rewards to EXACTLY the correct value shown**: {details}"*

이 site 는 **원장 술어가 없다**. `scaffold_get_tools[*].return_template` 은 탐지 시점에 렌더되고
분쟁 레코드를 한 줄도 읽지 않는다 ⇒ 전제("its dispute is resolved")의 성립을 **관측할 수 없다**.
관측 없이 전제를 말하면 전제는 무시되고 **명령만 남는다**. 실측(20런·refute_3 ⒠ 재관측):
  · `bank_t7308_ctl_20260818c|task_027#s373753` msg 33(우리 출력) → **msg 34** 에 갱신 4건,
    인자값 `6300/1020/3800/1499` 가 우리 출력과 바이트 동일 · 분쟁은 0건 resolve.
  · `bank_t7308_treat_20260818c|task_028` msg 27(우리 출력) → **msg 30** 갱신 1건 — 손님이 목록을
    보기도 전에. 같은 write 가 msg 54 에서 정상 경로(분쟁 성사 후)로 **다시** 나갔다 = 중복.

── 일반화 시험(사용자 지시 2026-08-24) ──────────────────────────────────────────
"관측 가능한 상태로 조건화하라, 태스크 id·제품명·문구로는 안 된다. 관측이 없으면 명령문을 버리고
값만 보고하고, 그것이 028 에 무엇을 파는지 밝혀라."
  · **이 site 에는 관측이 없다**(원장 술어 부재 = 구조적). ⇒ 명령문을 버린다.
  · 명령문의 정당한 자리는 **원장 술어를 가진 site** 뿐 — `follow_up_chains[3]`(`after=
    submit_cash_back_dispute_0589` · `requires=update_transaction_rewards_3847`)이 이미 그것이고
    문구도 이미 양방향이다([[64]]). **신설 0**([[62]]) — 기구는 있고 문면만 옮긴다.
  · **028 비용 = 실측 0.** 20런 028 의 갱신 25건 중 **23건이 (해당 txn 분쟁 성사 ∧ 손님이 그 txn
    지목)** 이고 24건이 손님 지목이다. 우리 반환문이 **유일 근거**인 것은 **1건**뿐이며 그 1건이
    위의 msg 30 중복이다(같은 write 를 msg 54 가 재발행·reward 1.0 불변).
    같은 술어로 019/020/027/029 에서는 **21건**이 사라진다(총 22건 / 7 sim).
    ⇒ 태스크별 부호가 갈리지 않는다 = [[70]] 절충 대상이 아니라 **순 제거**다.

── 근거 문구의 출처(정책 축자·[[23]]) ──────────────────────────────────────────
KB `doc_credit_cards_credit_cards_(general)_004` "Applying Resolved Cash Back Dispute Corrections
(Internal)" 축자:
  · *"After a cash back dispute is resolved and approved, you must update the affected
    transaction(s) with the correct rewards value."*
  · *"Look up the user's resolved disputes in the cash_back_disputes database to find the
    transaction_id values that need rewards adjustments."*
즉 **정책이 이미 탐지와 적용을 분리**하고 갱신의 피연산자 출처를 *해소된 분쟁* 으로 못박는다.
우리 반환문이 그 둘을 한 문장에 붙여 놓은 것이 결함이었다. gold 열람 0.

── 수리(문면·선언만·op/isolate/params 불변) ────────────────────────────────────
 ① ratefix `return_template`: 갱신 명령문 삭제 → 값 보고(= {details} 유지, 026 회귀 방지) +
    검사 범위 명시(분쟁 레코드 미열람) + [[64]] fix-naming(다음 단계 = 분쟁 그 자체 · 갱신의
    피연산자는 해소된 분쟁에서 읽는다).
 ② `follow_up_chains[3].feedback`: 구 문면이 *"The reward-check tool's own output states …"* 로
    **우리 출력을 인용**하는데 ①이후 그 인용은 거짓이 된다 ⇒ [[25]](우리 도구 100% 정답 의무)
    위반이므로 **필수 동반 수리**. 인용처를 정책 문서로 옮긴다. 발화 조건(`after`/`requires`)·
    `resign_th`·양방향 문구는 **불변**(test_c204_nextrun 이 검사하는 축자 그대로 보존).
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]
TOOL = "get_reward_discrepancies"

RET = (
    "Transactions whose recorded reward does NOT match the expected reward under the reward-rate "
    "policy (each needs a cash back dispute). The CORRECT total reward per policy is shown next "
    "to each id, so you can state it to the customer and carry it into that dispute: {details}. "
    "SCOPE OF THIS RESULT - detection only: this compared the transaction rows you passed in "
    "against the reward-rate policy. It read no dispute record, so it does not know whether any "
    "dispute exists for these transactions, nor how one was decided. Do NOT change any "
    "transaction's rewards on the strength of this result. Under the bank's internal procedure "
    "the rewards correction is a separate, later step: a transaction's rewards are corrected "
    "only for a dispute that has been resolved and approved, and the transaction_ids for that "
    "correction are read from the resolved disputes themselves, never from a discrepancy list. "
    "The step this result calls for is the dispute, one per id above.")

FB = (
    "Error: [FOLLOW-UP] cash back disputes were submitted, but the ledger shows {missing} was "
    "never called. The bank's internal procedure for applying resolved cash back dispute "
    "corrections says that after a cash back dispute is resolved and approved, the affected "
    "transaction must be updated with the correct rewards value, and that the transaction_ids "
    "for that update come from the disputes that are resolved. If the dispute record or the "
    "customer's own confirmation shows these disputes resolved and approved, unlock and call the "
    "update tool now, using for each of those transactions the correct value the reward-rate "
    "check reported - a verbal summary does not update any record. If resolution has NOT been "
    "confirmed yet, do not update; tell the customer plainly what remains and that no update has "
    "happened.")

NOTE_RET = (
    "R6(2026-08-24 · refute_3 ⒠ · 20런 재관측): 구 문면의 '— after its dispute is resolved, "
    "update that transaction's rewards to EXACTLY the correct value shown' 를 **삭제**했다. "
    "이 site 는 원장 술어가 없어(탐지 시점 렌더·분쟁 레코드 0회 열람) 전제를 관측할 수 없고, "
    "관측 없는 전제는 무시되고 명령만 남는다 — t7308_ctl 027#s373753 msg33(우리 출력)→msg34 "
    "갱신 4건(값 6300/1020/3800/1499 바이트 동일·분쟁 0 resolve) · t7308_treat 028 msg27→msg30 "
    "갱신 1건(손님이 목록 보기 전). 실측 귀속: 우리 반환문이 **유일 근거**인 갱신 22건/7 sim "
    "(019·020·027·029 = 21 · 028 = 1이며 그 1건은 msg54 정상 경로가 재발행한 중복). "
    "028 의 25건 중 23건은 (분쟁 성사 ∧ 손님 지목) 이라 명령문과 무관 ⇒ **태스크별 부호가 "
    "갈리지 않는 순 제거**([[70]] 절충 불필요). 대체 site 신설 0([[62]]) — 갱신 지시는 원장 "
    "술어를 가진 follow_up_chains(after=submit_cash_back_dispute)에 이미 있다. {details}(기대값 "
    "노출)는 **유지**(2026-07-19 026 회귀 방지). 문구 출처=KB doc_credit_cards_credit_cards_"
    "(general)_004 축자('After a cash back dispute is resolved and approved …' / 'Look up the "
    "user's resolved disputes … to find the transaction_id values that need rewards "
    "adjustments') — 정책이 이미 탐지와 적용을 분리한다·gold 열람 0([[23]]). op/isolate/params/"
    "grounded_params/return_template_empty 불변.")

NOTE_FB = (
    " | R6(2026-08-24): 구 문면이 'The reward-check tool's own output states that AFTER a dispute "
    "is resolved …' 로 **우리 반환문을 인용**했는데 같은 날 그 명령문을 제거했으므로 그 인용은 "
    "거짓이 된다 ⇒ [[25]](우리 도구 100% 정답 의무) 위반이라 필수 동반 수리. 인용처를 정책 "
    "문서(doc_credit_cards_credit_cards_(general)_004 'Applying Resolved Cash Back Dispute "
    "Corrections')로 옮기고, 값의 소재는 사실 지시('the correct value the reward-rate check "
    "reported')로 남겼다. **발화 조건 불변** — after/requires/resign_th 그대로이고 양방향 축자 "
    "'If resolution has NOT been confirmed yet, do not update' 도 바이트 보존(test_c204_nextrun "
    "D8 검사축). 이 site 는 원장 술어를 가지므로 전제-딸린 지시가 **허용되는 유일한 자리**다.")


def main():
    tools, chains = [], []
    for rel in PATHS:
        p = os.path.join(HERE, rel)
        j = json.load(io.open(p, encoding="utf-8"))

        hit = next((t for t in j.get("scaffold_get_tools") or [] if t.get("name") == TOOL), None)
        if hit is None:
            print("MISSING tool in %s" % rel)
            sys.exit(1)
        rf = (hit.get("variants") or {}).get("ratefix")
        if rf is None:
            print("MISSING ratefix variant in %s" % rel)
            sys.exit(1)
        rf["return_template"] = RET
        rf["_note_return_imperative"] = NOTE_RET

        ch = next((c for c in j.get("follow_up_chains") or []
                   if any("update_transaction_rewards" in str(r) for r in (c.get("requires") or []))),
                  None)
        if ch is None:
            print("MISSING update chain in %s" % rel)
            sys.exit(1)
        before = dict(ch)
        ch["feedback"] = FB
        if NOTE_FB not in (ch.get("_note") or ""):
            ch["_note"] = (ch.get("_note") or "") + NOTE_FB
        # 발화 조건 불변 검산 — 이 스크립트가 문면 외에는 아무것도 못 바꾸게 한다.
        for k in ("after", "requires", "resign_th", "decision_tools", "reserve"):
            assert before.get(k) == ch.get(k), "발화 조건이 바뀌었다: %s (%s)" % (k, rel)

        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(j, f, ensure_ascii=False, indent=1)
            f.write("\n")
        tools.append(rf)
        chains.append(ch)
        print("updated %s" % rel)

    print("ratefix return_template 3사본 동일:",
          tools[0]["return_template"] == tools[1]["return_template"] == tools[2]["return_template"])
    print("chain feedback 3사본 동일:",
          chains[0]["feedback"] == chains[1]["feedback"] == chains[2]["feedback"])


if __name__ == "__main__":
    main()
