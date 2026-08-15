# -*- coding: utf-8 -*-
r"""x335 — 자격 **제거**(닫힌 수치 비교): 모델이 못 지우는가, 재료가 안 갔는가.

## 사건 (t7295 · task_055 · 3 sim 전부 실패)

손님(msg 3 축자):
> *"I can keep maybe **three to four thousand** in this travel account, **five at most**.
>   Nothing crazy like fifty grand though"*

우리 에이전트는 msg 14 에서 **Bluest Account** 를 고른다(gold = `Purple Account`).
문서 축자(`doc_checking_accounts_bluest_account_001`)는 월수수료 면제 최소잔고 **$112,500** 이고
Purple 은 **$3,750** 이다 ⇒ **문서만 읽으면 Bluest 는 탈락**한다(§22 감사·[[23]] 만족:
근거가 gold 가 아니라 **선언 문서**다).

⚠**069 는 표적이 아니다**(§21): 거기서는 근거가 gold 뿐이라 레버를 만들면 [[23]] 위반이다.
제거 축의 표본 = **055 · 070 · 071**.

## 왜 격리부터인가 ([[62]] ①②③)

"닫힌 수치 비교니까 비교기를 짓자"가 **틀린 순서**다. 문서를 주면 모델이 스스로 $112,500 을 보고
지우는가? 지우면 레버는 **전달(부하 축소)뿐**이고 비교기는 짓지 않는다. 안 지울 때만 그 다음을
논한다. 이 프로브가 그 분기를 만든다.

## 셀 4 (컷 = 055#0 **msg 14 직전** = 검색 결과까지 보고 아직 고르지 않은 자리)

    A_REF     라이브 축자(전달 없음)                ← 지금의 실패를 격리로 재현하나
    B_DOCS    + **제품 문서 재료**(t2_search 정본)   ← 전달만으로 닫히나
    C_QUOTE   B + 손님 금액 **축자 재인용**          ← 문맥에 묻힌 것인가(부하)
    D_NEG     C 와 같되 손님 금액을 **$200,000** 로  ← ★부정통제: Bluest 가 **옳은** 자리

⚠**D_NEG 가 생명이다**([[57]]). C 에서 Bluest 가 사라진 것이 *금액을 쓴 것*인지 *비싼 이름을
그냥 피한 것*인지 가른다. C 낮고 D 높아야 비교가 실제로 일어난 것이다.
⚠[[62]] ③④ · [[59]]: 엔진은 **읽어 나르고 축자로 인용**만 한다 — 최소잔고 필드를 뽑지 않고,
비교하지 않으며, 후보를 지목하지도 제거하지도 않는다. 판단은 끝까지 모델 몫이다.
⚠D_NEG 의 금액 치환은 **문자열 치환**(궤적의 그 문장 + 재인용 둘 다)이라 의미 판단 0.
⚠재료는 라이브와 **같은 경로**(`material_for(..., windowed="general")` · `per_doc=400`)로 만든다 —
  프로브 전용 재료를 새로 짜면 재는 것이 달라진다([[67]] 정본).

## 판정 (사전 고정 · n=24=8×3 · 잡음 바닥 ±4·C483 ⇒ **차 ≥5 만 인용** · 블록 병기)

    지표 = 응답이 대는 checking 클래스 (BLUEST=제거 실패 · PURPLE=gold 일치 · 그 외=기타)

    A_REF BLUEST ≥18                     → 라이브 실패가 격리에서 재현(부하 아님)
    B_DOCS BLUEST ≤ A_REF − 5            → **전달만으로 닫힌다** ⇒ 레버 = 전달. 비교기 금지
    B ≈ A ∧ C_QUOTE BLUEST ≤ B − 5       → 재료는 갔으나 **묻힌다**(부하) ⇒ 레버 = 표면화(축자)
    C ≈ B ∧ D_NEG BLUEST ≥18             → 금액을 안 쓴다 ⇒ 표면화 무효, 그때 비교기를 논한다
    C 낮은데 D 도 낮다                     → **부정통제 실패** = 이 프로브 무효(과잉 제거)
    A_REF BLUEST ≤6                      → 격리선 지킨다 ⇒ 결손은 궤적 부하 or 우리 문구

⚠PURPLE 상승은 **부수 지표**다. 이 프로브가 재는 것은 *제거*(BLUEST 소멸)이지 정답 적중이 아니다.

실행(리모트·8141·[[30]] 포트 분리 — **유료 런이 없을 때만**):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x335_eligibility_drop_iso.py [블록크기] [블록수]
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

TAG, TASK, CUT = "bank_t7295_b_20260815n", "task_055", 14
GROUP = "checking_accounts"
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
A2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "a2", "banking_knowledge.specific.json")

# 손님 발화 축자 — 궤적에서 **부분문자열로 찾는다**(손으로 옮겨 적지 않는다·검산 포함)
AMOUNT = "three to four thousand in this travel account, five at most"
RICH = "two hundred thousand in this travel account, two hundred fifty at most"

MISS = "BLUEST"      # 제거 실패
GOLD = "PURPLE"      # gold 일치
ASK = ("\n[instruction] Do NOT call any tool yet. Reply with ONE line only: the full official "
       "name of the ONE checking account class you would open for this customer, nothing else.")
MAXTOK = 60          # ★고정 — 런마다 바꾸면 팔 간 비교가 깨진다(어젯밤 x330~332 사고)


def quote_from(sim, needle):
    """궤적에서 그 문장을 담은 **손님 발화 축자**를 되찾는다. 못 찾으면 중단(계기 결함)."""
    for m in (sim.get("messages") or []):
        if m.get("role") != "user":
            continue
        c = " ".join(str(m.get("content") or "").split())
        if needle in c:
            return c
    return ""


def classify(text):
    t = (text or "").upper()
    return (MISS in t, GOLD in t)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK)
    said = quote_from(sim, AMOUNT)
    if not said:
        print("손님 금액 축자를 궤적에서 못 찾음 — 중단(계기 결함·컷/태그 확인)")
        return 1
    base = "\n".join([B.HEAD, "", B.transcript(sim, CUT)])
    if AMOUNT not in base:
        print("컷 %d 안에 금액 발화가 없다 — 정보-맞춤 위반(§1.4) ⇒ 중단" % CUT)
        return 1

    a2 = json.load(io.open(A2, encoding="utf-8"))
    material, stat = S.material_for(a2, GROUP, doc_dir=DOCS, windowed="general", per_doc=400)
    note = "\n\n[note] The customer said (verbatim): \"%s\"" % said
    rich_note = note.replace(AMOUNT, RICH)
    rich_base = base.replace(AMOUNT, RICH)
    if rich_base == base or rich_note == note:
        print("부정통제 치환 실패 — 중단(치환 대상 문자열 불일치)")
        return 1

    arms = (("A_REF", base + ASK),
            ("B_DOCS", base + "\n\n" + material + ASK),
            ("C_QUOTE", base + "\n\n" + material + note + ASK),
            ("D_NEG", rich_base + "\n\n" + material + rich_note + ASK))
    print("x335 · %s/%s · cut=%d · 재료 %d자(링크 %d·읽음 %d·유지 %d·제외 %d) · 본문 %d자 · %d×%d블록"
          % (TAG, TASK, CUT, len(material), stat["linked"], stat["read"], stat["kept"],
             len(stat["dropped"]), len(base), k, nb))
    print("손님 축자: %s\n" % said[:200])

    res = {}
    for label, body in arms:
        miss_b, gold_b = [], []
        for b in range(nb):
            mv = gv = 0
            for i in range(k):
                try:
                    r = chat(body, None, 0.0 if i == 0 else 0.7, MAXTOK)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                out = " ".join(str(r.get("content") or "").split())
                m, g = classify(out)
                mv += m
                gv += g
                print("    [%s b%d %02d] %s%s %s" % (label, b + 1, i, "BLUEST" if m else "-",
                                                     "/PURPLE" if g else "", out[:60]), flush=True)
            miss_b.append(mv)
            gold_b.append(gv)
        res[label] = (sum(miss_b), miss_b, sum(gold_b), gold_b)
        print("%-9s BLUEST %d/%d %s · PURPLE %d/%d %s\n"
              % (label, sum(miss_b), k * nb, miss_b, sum(gold_b), k * nb, gold_b))

    print("판정(사전 고정·차 ≥5 만 인용): A≥18 → 격리 재현 · B ≤ A−5 → **전달로 닫힘**(비교기 금지) · "
          "B≈A ∧ C ≤ B−5 → 부하(축자 표면화) · C≈B ∧ D≥18 → 금액 미사용(그때 비교기 논의) · "
          "C 낮고 D 도 낮음 → 부정통제 실패 = 무효 · A≤6 → 격리선 지킴(궤적 부하/우리 문구)")
    print("측정치: " + " · ".join("%s BLUEST=%d%s PURPLE=%d%s" % (a, v[0], v[1], v[2], v[3])
                                 for a, v in res.items()))


if __name__ == "__main__":
    sys.exit(main() or 0)
