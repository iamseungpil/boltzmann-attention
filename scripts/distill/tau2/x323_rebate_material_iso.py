# -*- coding: utf-8 -*-
r"""x323 — 072 의 $2.00: **재료만 주면 모델이 계산하나**(전달) vs **엔진이 빼야 하나**(결정론).

규명(C487·원장 직독·gold 무접촉): Bluest 계좌에 `ATM FEE REBATE` **5건**($2.00×5)이 있고,
미환급 잔액 = 0.50(11/20 부분환급) + **2.00(11/14 환급 없음)** + 3.50 + 8.00(foreign) = **$14.00**
= gold 정확 일치. 우리 `select_discrepant` 는 **부과 vs 기대**만 비교하고 환급을 안 본다 —
11/14 를 *"기대 2.00 = 부과 2.00"* 이라 정상 판정해 놓쳤고, 격리 서브 지시문도 `atm_fee` 라인만
뽑는다 ⇒ **재료에 rebate 가 없다**. 모델은 도구가 준 $12.00 을 정직하게 썼다(무죄).

⛔[[62]] 순서: 결정론을 늘리기 **전에** 재는 것 — *"재료를 주면 모델이 하나?"* 가 먼저다.
   된다면 레버는 **전달뿐**(서브 지시문에 rebate 라인 추가)이고, 엔진 뺄셈은 불필요하다.

셀 5 (**n=24 = 8×3**·잡음 바닥 ±4·C483 · 근거는 전부 **원장 축자**):
  A_REF        현행 재료(fee 라인 + 우리 도구 결과 $12.00)        ← 재현: 12.00 을 그대로 쓰나
  B_REBATE     + **환급 기록 5줄 축자**(원장에서 그대로)           ← 전달만으로 되나 ★핵심
  C_POLICY     B + 정책 축자(*"third-party 수수료는 월 $50 까지 환급"*)
  D_ENGINE     엔진이 이미 뺀 값($14.00)을 준다                    ← 천장(출시 아님·[[62]] 진단용)
  E_NEG        B + **원장에 없는 가짜 환급 1줄**($5.00)            ← 맹목 뺄셈 부정통제

채점(기계적): 응답이 **14.00**(또는 14) 을 내는가. 12.00 을 유지하면 miss.
  ⚠E_NEG 는 **9.00**(가짜 5.00 을 더 뺀 값)을 내면 맹목 순응으로 별도 계수.

판정(사전 고정):
  A_REF ≤6                    → 재현(이하 전제)
  B_REBATE ≥18                → **전달만으로 닫힌다** ⇒ 서브 지시문에 rebate 추가·엔진 뺄셈 불요
  B ≤6 ∧ C_POLICY ≥18         → 정책 문면이 있어야 한다(A2 문구 추가·여전히 전달 축)
  B,C ≤6 ∧ D_ENGINE ≥18       → 계산 자체가 결손 ⇒ **엔진 뺄셈 정당화**(최소 결정론으로 한정)
  E_NEG 맹목 ≥6               → 근거 없는 값도 빼 준다 = 위험 기록([[25]])
  전 팔 ≤6                    → 이 자리는 전달 축이 아니다

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x323_rebate_material_iso.py [블록크기] [블록수]
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

from x216_read_and_offset import chat                             # noqa: E402

HEAD = ("You are a banking agent finishing an ATM fee audit for one checking account "
        "(a Bluest Account). You must decide the single fee_refund credit amount to apply.")
TOOL_OUT = ("[tool] ATM fee lines whose charged amount does NOT match the documented fee schedule "
            "for this account level: btxn_fcd7ef3a24ed (charged $2.50, documented fee $2.00, "
            "difference $0.50); btxn_6a3453e0afd9 (charged $3.50, documented fee $0.00, "
            "difference $3.50); btxn_344585b826eb (charged $8.00, documented fee $0.00, "
            "difference $8.00). The credit policy requires ONE fee_refund credit for the net "
            "correction across all identified fee discrepancies of THIS account.")
FEES = ("[tool] Third-party ATM fee lines on this account: 11/20 $2.50; 11/18 $2.00; "
        "11/14 $2.00; 11/10 $2.00; 11/05 $2.00; 11/02 $2.00. "
        "Foreign ATM fee lines: 11/12 $3.50; 11/08 $8.00.")
REBATES = ("[tool] ATM FEE REBATE records on this account: 11/20 $2.00; 11/18 $2.00; "
           "11/10 $2.00; 11/05 $2.00; 11/02 $2.00.")
FAKE_REBATE = " 11/14 $5.00."
POLICY = ("[policy] Bluest Account: third-party ATM fees are rebated up to $50 per monthly "
          "statement cycle. Foreign ATM withdrawal fee: $0.00.")
ENGINE = ("[tool] Unrebated remainder computed from the ledger: 11/20 $0.50 + 11/14 $2.00 + "
          "11/12 $3.50 + 11/08 $8.00 = $14.00.")
ASK = ("\n[instruction] Reply with ONE line only: the dollar amount of the single fee_refund "
       "credit you would apply, as a number.")
NUM = re.compile(r"(\d+(?:\.\d+)?)")


def amount(txt):
    m = NUM.findall(str(txt or "").replace(",", ""))
    return [float(x) for x in m] if m else []


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    arms = (
        ("A_REF", "\n\n".join([HEAD, TOOL_OUT]) + ASK),
        ("B_REBATE", "\n\n".join([HEAD, TOOL_OUT, FEES, REBATES]) + ASK),
        ("C_POLICY", "\n\n".join([HEAD, TOOL_OUT, FEES, REBATES, POLICY]) + ASK),
        ("D_ENGINE", "\n\n".join([HEAD, TOOL_OUT, ENGINE]) + ASK),
        ("E_NEG", "\n\n".join([HEAD, TOOL_OUT, FEES, REBATES[:-1] + FAKE_REBATE]) + ASK),
    )
    print("x323 · 072 Bluest · 목표 14.00 (원장 검산·gold 무접촉) · %d×%d블록\n" % (k, nb))
    res = {}
    for label, body in arms:
        blocks, keep12, blind9 = [], 0, 0
        for _b in range(nb):
            h = 0
            for i in range(k):
                try:
                    r = chat(body, None, 0.0 if i == 0 else 0.7, 60)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                out = " ".join(str(r.get("content") or "").split())
                v = amount(out)
                ok = any(abs(x - 14.0) < 1e-6 for x in v)
                h += ok
                keep12 += any(abs(x - 12.0) < 1e-6 for x in v)
                blind9 += any(abs(x - 9.0) < 1e-6 for x in v)
                print("    [%s b%d %02d] %s %s" % (label, _b + 1, i, "HIT" if ok else "-",
                                                   out[:40]), flush=True)
            blocks.append(h)
        res[label] = (sum(blocks), blocks)
        print("%-11s %d/%d · 블록 %s · 12.00유지 %d%s\n"
              % (label, sum(blocks), k * nb, blocks, keep12,
                 (" · 9.00(맹목) %d" % blind9) if blind9 else ""))
    print("판정(사전 고정): A≤6 전제 · B≥18 → **전달만으로 닫힘**(서브에 rebate 추가·엔진 뺄셈 불요) · "
          "B≤6∧C≥18 → 정책 문면 필요 · B,C≤6∧D≥18 → 엔진 뺄셈 정당화 · E 맹목≥6 → 위험 · 전 팔 ≤6 → 전달 축 아님")
    print("측정치: " + " · ".join("%s=%d%s" % (a, v[0], v[1]) for a, v in res.items()))


if __name__ == "__main__":
    main()
