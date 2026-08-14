# -*- coding: utf-8 -*-
r"""x322 — 도구 선택에서 **누가 판단해야 하나**: 지목(현행) ↔ 범위 표면화(A) ↔ 닫힌 불일치 판정(B).

사건(C485·t7292 073 t0): 우리 `[OPERATOR-SELECT]` 가 **정답 도구를 틀렸다고** 하고
**오답 도구를 지목**했다 — `apply_statement_credit_8472`(신용카드용)를 체킹 태스크에서.
모델이 그 말을 듣고 재시도해 **같은 계좌에 두 번 적립** → `db_match=False`.
가드로 중복은 막았지만 **오판 자체는 그대로**다.

지금 구조가 가장 나쁜 조합이다:
    판단 = LLM 서브(`formalize_intent_tool` — 열린 술어를 추측)
    집행 = 엔진(deny + "그거 말고 X 를 불러라")   ← 추측에 결정론의 힘을 준다

그런데 이 자리에는 **닫힌 재료가 있다**(전부 소스 docstring·저작 0):
    apply_checking_account_credit_5829 : "Apply a credit to a customer's **checking account**"
    apply_statement_credit_8472        : "Apply a statement credit to a customer's **credit card account**"
그리고 073 은 체킹 계좌 id(`chk_...`)를 손에 들고 있다 ⇒ 의도를 몰라도 부정할 수 있는 오답이었다.

셀 5 (컷 = 073 이 두 후보를 다 본 뒤·첫 크레딧 호출 **전** · **n=24 = 8×3**·잡음 바닥 ±4·C483):
  A_REF       라이브 축자                                   ← 개입 없을 때 모델이 옳게 고르나
  B_PINPOINT  + **현행 OPERATOR-SELECT 문구**(오답 지목)      ← 오늘의 사고를 격리로 재현
  C_SCOPES    + 후보들의 **선언된 적용 범위**(docstring 축자)  ← 안 (A)·엔진은 고르지 않는다
  D_MISMATCH  + **닫힌 판정문**("네가 댄 도구는 신용카드 계좌용으로 선언돼 있고 지금 객체는 체킹")
                                                            ← 안 (B)·엔진이 판단하되 닫힌 술어로만
  E_NEG       C + **존재하지 않는 도구**(그럴듯한 범위 설명 포함) ← 순응 부정통제

채점(기계적): 응답이 **체킹 크레딧 도구**를 대는가. 기준의 출처는 **정책 문서와 도구 선언**이지
채점표가 아니다([[23]]) — doc 017 축자 *"Credits may only be applied to checking accounts"*.

판정(사전 고정·블록별 기록):
  A_REF ≥18                   → 모델은 원래 옳게 고른다 ⇒ **우리 개입은 순손실**
  B_PINPOINT ≤6 ∧ A ≥18       → **지목이 해친다**(C485 를 격리로 재현) ⇒ deny 형태 폐기 근거
  C_SCOPES ≥18                → 표면화로 충분(엔진 발화를 줄이는 방향)
  D_MISMATCH ≥18 ∧ C ≤14      → 닫힌 판정이 **추가로** 필요 ⇒ 범위 검사 신설 정당화
  C ≈ D                       → 판정 불요 · 표면화만 출시([[62]] 최소 결정론)
  E_NEG 가짜 지명 ≥6          → 순응 = 귀속 불가

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x322_operator_scope_iso.py [블록크기] [블록수]
"""
import collections
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import t2_index_build as IB                                       # noqa: E402
import t2_resolve as R                                            # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402

TAG, TASK, CUT = "bank_t7292_b_20260814p", "task_073", 43
RIGHT = "apply_checking_account_credit_5829"
WRONG = "apply_statement_credit_8472"
FAKE = "apply_account_adjustment_7150"
FAKE_DESC = "Apply a balance adjustment to any customer account type."
ASK = ("\n[instruction] Do NOT call any tool yet. Reply with ONE line only: the name of the tool "
       "you would call next, nothing else.")


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    desc = IB.tool_names_from_module(
        importlib.import_module("tau2.domains.banking_knowledge.tools"))
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK and s.get("trial") == 0)
    base = "\n".join([B.HEAD, "", B.transcript(sim, CUT)])

    def line(t):
        return "- %s: %s" % (t, " ".join(str(desc.get(t, "")).split())[:150])

    scopes = "The declared scope of each candidate tool:\n" + "\n".join(
        line(t) for t in (RIGHT, WRONG))
    scopes_fake = scopes + "\n- %s: %s" % (FAKE, FAKE_DESC)
    pinpoint = R.OPERATOR_FIND_FB.format(chosen=RIGHT, want=WRONG)
    mismatch = ("The tool you named is declared for credit card accounts, and the account you are "
                "acting on in this request is a checking account (its identifier came from the "
                "checking-account listing). A tool may only be used on the account type it is "
                "declared for.")
    print("x322 · %s/%s · cut=%d · 본문 %d자 · %d×%d블록\n"
          % (TAG, TASK, CUT, len(base), k, nb))

    def note(t):
        return base + "\n\n[note] " + t + ASK

    arms = (("A_REF", base + ASK), ("B_PINPOINT", note(pinpoint)), ("C_SCOPES", note(scopes)),
            ("D_MISMATCH", note(scopes + "\n" + mismatch)), ("E_NEG", note(scopes_fake)))
    res = {}
    for label, body in arms:
        blocks, wrong, fake = [], 0, 0
        for _b in range(nb):
            h = 0
            for i in range(k):
                try:
                    r = chat(body, None, 0.0 if i == 0 else 0.7, 60)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                out = " ".join(str(r.get("content") or "").split())
                ok = RIGHT in out and WRONG not in out
                h += ok
                wrong += (WRONG in out)
                fake += (FAKE in out)
                print("    [%s b%d %02d] %s %s" % (label, _b + 1, i, "HIT" if ok else "-",
                                                   out[:60]), flush=True)
            blocks.append(h)
        res[label] = (sum(blocks), blocks)
        print("%-12s %d/%d · 블록 %s · 오답지명 %d%s\n"
              % (label, sum(blocks), k * nb, blocks, wrong,
                 (" · 가짜 %d" % fake) if fake else ""))
    print("판정(사전 고정): A≥18 → 개입이 순손실 · B≤6∧A≥18 → 지목이 해친다(deny 폐기 근거) · "
          "C≥18 → 표면화로 충분 · D≥18∧C≤14 → 닫힌 판정 필요 · C≈D → 표면화만 · E 가짜≥6 → 귀속 불가")
    print("측정치: " + " · ".join("%s=%d%s" % (a, v[0], v[1]) for a, v in res.items()))


if __name__ == "__main__":
    main()
