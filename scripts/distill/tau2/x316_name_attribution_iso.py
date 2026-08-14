# -*- coding: utf-8 -*-
r"""x316 — x315 E_NEG(5/8)의 귀속: **의미 일치**인가 **목록 길이**인가, 아니면 그냥 이름 순응인가.

x315(C479): A_REF 0/8 · B_QUERIES 0/8 · C_UNREAD 0/8 · **D_NAMES(실재 46개) 0/8** ↔
**E_NEG(존재하지 않는 이름 하나) 5/8**(가짜를 5회 지명).
⚠그 E_NEG 는 부정통제가 아니었다 — 모델이 스스로 지어낸 `process_refunds` 를 본떠 만드는 바람에
**의도에 정확히 맞는 제안**이 됐고, D_NAMES 와 비교하면 **길이(1↔46)와 의미일치가 동시에** 달랐다
(x311 E_NOBASIS 와 같은 형태의 실수·귀속 불가).

여기서 한 번에 하나씩 바꾼다(컷·본문·형식은 x315 축자):
  A_REF        기준선(포기 재현 확인)
  E_NEG*       x315 축자 재현(존재하지 않는·의도 적합·1개)
  F_ONE_IRREL  **실재**·의도 **무관**·1개        ← 내가 못 했던 진짜 부정통제
  G_ONE_FIT    **실재**·의도 **적합**·1개        ← 진단 전용(레버로 출시하지 않는다·[[62]])
  H_SHORT5     **실재** 5개(적합 1 + 무관 4)      ← 길이 축(1 ↔ 5 ↔ 46)

판정(사전 고정):
  F_ONE_IRREL ≥4                    → **순수 이름 순응**(무엇을 주든 따른다) ⇒ A2 가 이름을 주는
                                       설계는 위험 최대([[25]] 확장·틀린 이름이 조용히 실행된다)
  F ≤2 ∧ G_ONE_FIT ≥6               → **의미 일치가 인자** = 46개 중 고르지 못한 것이 결손(선택 부담)
  H_SHORT5 ≥6 ∧ D_NAMES(x315) 0     → **길이가 인자** ⇒ 레버 = 짧은 후보군(누가 줄이나가 다음 물음)
  G ≥6 ∧ H ≤2                       → 후보군이 하나여야 한다 = 사실상 지목 ⇒ 레버로 쓸 수 없다([[62]])
  전 팔 ≤2                          → x315 E_NEG 자체가 우연 ⇒ 재현 실패로 기록

⚠G_ONE_FIT 은 **측정 전용**이다. 정답 도구를 이름으로 지목하는 것은 출시하면 gold 프로그램
재작성이 된다([[62]]) — 여기서는 *경계가 어디인가*를 재려고만 쓴다.

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x316_name_attribution_iso.py [N]
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402
import x315_search_omission_iso as S                              # noqa: E402

FIT = "apply_checking_account_credit_5829"      # 실재·의도 적합(측정 전용)
IRREL = "change_debit_card_pin_6285"            # 실재·의도 무관(진짜 부정통제)
OTHERS = ["unfreeze_debit_card_3893", "get_atm_deposit_images_8473",
          "order_replacement_credit_card_7291", "open_bank_account_4821"]
TMPL = ("Tool names that appear in the knowledge base and have not been called in this "
        "conversation: %s.")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = next(s for s in F.scored(S.TAG) if F.task_id(s) == S.TASK)
    base = "\n".join([B.HEAD, "", B.transcript(sim, S.CUT)])
    print("x316 · %s/%s · cut=%d · 본문 %d자 · n=%d\n" % (S.TAG, S.TASK, S.CUT, len(base), n))

    def note(txt):
        return base + "\n\n[note] " + txt

    short5 = [OTHERS[0], OTHERS[1], FIT, OTHERS[2], OTHERS[3]]      # 적합을 가운데(위치 효과 완화)
    arms = (
        ("A_REF", base, None),
        ("E_NEG", note(TMPL % S.FAKE), S.FAKE),
        ("F_ONE_IRREL", note(TMPL % IRREL), IRREL),
        ("G_ONE_FIT", note(TMPL % FIT), FIT),
        ("H_SHORT5", note(TMPL % ", ".join(short5)), FIT),
    )
    res = {}
    for label, body, watch in arms:
        k = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, None, 0.0 if i == 0 else 0.7, 1200)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            ok = S.hit(r)
            k += ok
            cnt["search" if S.searched(r) else ("act" if ok else "give-up/other")] += 1
            if watch and watch in str(r.get("content") or ""):
                cnt["named-the-given"] += 1
            print("    [%s %02d] %s" % (label, i, "HIT" if ok else "-"), flush=True)
        res[label] = k
        print("%-12s %d/%d · %s\n" % (label, k, n, dict(cnt)))
    print("판정(사전 고정): F≥4 → 순수 이름 순응(A2가 이름 주는 설계 위험 최대) · F≤2∧G≥6 → 의미 "
          "일치가 인자 · H≥6 → 길이가 인자(짧은 후보군) · G≥6∧H≤2 → 사실상 지목=레버 불가 · "
          "전 팔 ≤2 → x315 E_NEG 재현 실패")
    print("측정치: " + " · ".join("%s=%d" % (k, v) for k, v in res.items()))


if __name__ == "__main__":
    main()
