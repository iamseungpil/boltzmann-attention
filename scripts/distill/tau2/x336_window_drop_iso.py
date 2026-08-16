# -*- coding: utf-8 -*-
r"""x336 — 유효창(만료 고지): **우리 엔진이 빼 줘야 하나, 모델이 스스로 날짜로 고르나**.

## 왜 (070·071 · [[62]] ②③)

`t2_search.drop_expired` 는 우리 층의 **결정론 제거**다 — 유효창이 지난 문서를 빼고 나른다.
그런데 그것이 **필요한지 잰 적이 없다**. 재료를 나르는 것(전달)과 만료를 빼 주는 것(제거)은
다른 레버이고, 후자는 [[62]] 가 경고하는 *"모델이 이미 하는 일을 대신하는 결정론"* 일 수 있다.
055 에서 이미 같은 일이 있었다: 비교기를 지을 뻔했는데 **전달만으로 24/24** 였다(C494).

사실관계(전부 문서 축자·now = `2025-11-14`, 궤적의 `get_current_time` 반환):

    doc_bank_accounts_bank_accounts_(general)_013  ACTIVE 11/01–11/30  → (1) Sky Blue (2) Lime Green
    doc_bank_accounts_bank_accounts_(general)_014  ACTIVE 10/12–11/12  → (1) Lime Green (2) Hunter Green

070 gold = `Sky Blue`. 라이브에서 우리 에이전트는 `True Blue` 를 골랐다(고지 자체를 못 봤다).
071 도 같은 축이고 `Lime Green` 을 열었다 — **만료된 10월 고지의 1순위**다.

## 셀 4 (컷 = 070#0 msg 22 = 손님이 *"ONE 을 골라라"* 라고 한 직후 · n=24=8×3)

    A_REF       라이브 축자(재료 없음)                     ← 실패 재현
    B_DROPPED   + 재료(**엔진이 만료 제거**·라이브 정본)    ← 현행 스택
    C_BOTH      + 재료(**두 고지 다** · 날짜 그대로 · now 명시) ← 모델이 스스로 고르나
    D_FLIP      C 와 같되 **두 고지의 날짜만 맞바꿈**       ← ★부정통제: 정답이 Lime Green 으로 뒤집힘

⚠**D_FLIP 이 생명이다**([[57]]·x335 의 교훈). C 에서 Sky Blue 가 나오는 것이 *날짜를 쓴 것*인지
*11월 고지의 1순위라는 위치* 나 이름 고착인지 가른다. 날짜만 맞바꿨으므로 **문면상 정답은
Lime Green** 이 된다. C 높고 D 도 Sky Blue 면 날짜를 안 쓴 것이다.
⚠치환은 **문자열 치환**이고 두 고지의 날짜 구간을 서로 바꾸는 것뿐이다 — 의미 판단 0([[59]]).
⚠엔진은 읽어 나르기만 한다. 어느 상품이 옳은지 말하지 않는다([[62]] ③).

## 판정 (사전 고정 · 잡음 바닥 ±4·C483 ⇒ **차 ≥5 만 인용**)

    C SKY ≥18 ∧ D LIME ≥18   → **모델이 날짜로 고른다** ⇒ 엔진의 만료 제거는 **뺄 후보**([[63]]).
                                레버는 전달 하나로 줄어든다
    C SKY ≥18 ∧ D SKY ≥18    → 날짜 미사용(위치·이름 고착) ⇒ C 의 적중은 비교가 아니다.
                                엔진 제거를 **유지**한다
    C SKY ≤6                 → 두 고지를 같이 주면 모델이 무너진다 ⇒ 엔진 제거가 **정당**하다
    B SKY ≥18 ∧ A SKY ≤6     → 전달(+제거)이 이 축을 산다 = 라이브 예측
    B ≈ A                    → 전달로도 안 된다 ⇒ 결손은 다른 축(끝맺음·발견)

실행(리모트·8141·[[30]] 포트 분리):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x336_window_drop_iso.py [블록크기] [블록수]
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

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import t2_search as S                                             # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402

TAG, TASK, CUT = "bank_t7295_a_20260815n", "task_070", 22
GROUP = "business_checking_accounts"
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
A2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "a2", "banking_knowledge.specific.json")
NOW = "2025-11-14"                      # 궤적의 `get_current_time` 반환(축자)
NOV = "ACTIVE FROM 11/01/2025 TO 11/30/2025"
OCT = "ACTIVE FROM 10/12/2025 TO 11/12/2025"
SKY, LIME = "SKY BLUE", "LIME GREEN"
MAXTOK = 60
ASK = ("\n[instruction] Do NOT call any tool. Reply with ONE line only: the full official name of "
       "the ONE business checking account class you would open for this customer, nothing else.")


def both_material(a2, per_doc=400):
    """두 고지를 **함께** 실은 재료 — 만료 제거만 끄고 나머지는 정본 경로 그대로."""
    ids = list(S.docs_for(a2, GROUP))
    idx = (a2.get("policy_ontology") or {}).get("doc_index") or {}
    gen = {d for subs in idx.values() for d in (subs.get("_general_") or ())}
    for d in sorted(set(S.declared_windows(a2)) & gen):
        if d not in ids:
            ids.append(d)
    read, missing = S.read_docs(ids, DOCS, None)
    return S.as_material(read, (), per_doc=per_doc), {"read": len(read), "missing": missing}


def classify(text):
    t = (text or "").upper()
    return (SKY in t, LIME in t)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK)
    base = "\n".join([B.HEAD, "", B.transcript(sim, CUT)])
    a2 = json.load(io.open(A2, encoding="utf-8"))

    dropped, st = S.material_for(a2, GROUP, doc_dir=DOCS, windowed="general", per_doc=400, now=NOW)
    both, st2 = both_material(a2)
    nownote = "\n\n[note] The current date is %s." % NOW

    if NOV not in both or OCT not in both:
        print("두 고지가 재료에 없다 — 중단(계기 결함)"); return 1
    if OCT in dropped:
        print("만료 고지가 제거되지 않았다 — 중단(엔진 경로 확인)"); return 1
    flip = both.replace(NOV, "@@TMP@@").replace(OCT, NOV).replace("@@TMP@@", OCT)
    if flip == both:
        print("날짜 맞바꾸기 실패 — 중단"); return 1

    arms = (("A_REF", base + ASK),
            ("B_DROPPED", base + "\n\n" + dropped + nownote + ASK),
            ("C_BOTH", base + "\n\n" + both + nownote + ASK),
            ("D_FLIP", base + "\n\n" + flip + nownote + ASK))
    print("x336 · %s/%s · cut=%d · now=%s · 제거재료 %d자(유지 %d·뺀 것 %d) · 양쪽재료 %d자(읽음 %d) · %d×%d블록\n"
          % (TAG, TASK, CUT, NOW, len(dropped), st["kept"], len(st["dropped"]), len(both),
             st2["read"], k, nb))

    res = {}
    for label, body in arms:
        sb, lb = [], []
        for b in range(nb):
            sv = lv = 0
            for i in range(k):
                try:
                    r = chat(body, None, 0.0 if i == 0 else 0.7, MAXTOK)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                out = " ".join(str(r.get("content") or "").split())
                s_, l_ = classify(out)
                sv += s_
                lv += l_
                print("    [%s b%d %02d] %s%s %s" % (label, b + 1, i, "SKY" if s_ else "-",
                                                     "/LIME" if l_ else "", out[:60]), flush=True)
            sb.append(sv)
            lb.append(lv)
        res[label] = (sum(sb), sb, sum(lb), lb)
        print("%-11s SKY %d/%d %s · LIME %d/%d %s\n"
              % (label, sum(sb), k * nb, sb, sum(lb), k * nb, lb))

    print("판정(사전 고정·차 ≥5 만 인용): C SKY≥18 ∧ D LIME≥18 → **모델이 날짜로 고른다**(엔진 만료제거는 뺄 후보) · "
          "C SKY≥18 ∧ D SKY≥18 → 날짜 미사용(고착) ⇒ 엔진 제거 유지 · C SKY≤6 → 같이 주면 무너짐 ⇒ 제거 정당 · "
          "B SKY≥18 ∧ A SKY≤6 → 전달이 이 축을 산다 · B≈A → 다른 축(끝맺음·발견)")
    print("측정치: " + " · ".join("%s SKY=%d%s LIME=%d%s" % (a, v[0], v[1], v[2], v[3])
                                 for a, v in res.items()))


if __name__ == "__main__":
    sys.exit(main() or 0)
