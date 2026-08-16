# -*- coding: utf-8 -*-
r"""x335b — 문서를 주면 모델이 **방향을 뒤집는가**(x335 부정통제 재설계·C493 ⒡).

## 왜 다시 하나

x335 는 하나를 열었다: 문서 전달만으로 055 checking 이 **0/24 → 24/24**(gold `Purple`).
그러나 부정통제 `D_NEG` 가 **무너졌다** — 손님 잔고를 $200k 로 바꿔도 24/24 `Purple` 이었다.
당연하다. **잔고를 올려도 Purple 은 여전히 자격을 충족한다**(면제 최소잔고는 낮을수록 유리).
즉 나는 *Bluest 가 옳은 자리* 를 만들지 못했고, 그래서 24/24 가 **비교의 결과인지 여행이라는
낱말에 대한 고착인지** 가르지 못했다.

## 이번엔 무엇을 뒤집나 (전부 **문서 축자**·gold 무접촉·[[23]])

문서상 Bluest 가 **지배하는** 축이 실재한다:

    항목                         Bluest                     Purple
    월 무료 국내 전신            **10건**(_005)             선언 없음
    ATM 리베이트 월 상한         **$50**(_007)              $30(_004)
    모바일 수표 입금 한도(일)    **$10,000**(_001)          $5,000(_001)
    체킹 APY                     **2.25%**(_003)            선언 없음
    면제 최소잔고                $112,500(_002)             $3,750(_002)
    해외거래 수수료              선언 없음                  **0%**(_003)

⇒ **국내만 쓰고·잔고가 크고·국내 전신과 큰 모바일 입금이 잦은 손님**에게는 Bluest 가 문서상
   우세하고, 여행 손님에게는 Purple 이 우세하다. **같은 재료로 답이 갈리는 두 자리**가 생긴다.

## 셀 4 (n=24=8×3 · 재료는 라이브와 같은 경로 `t2_search.material_for`)

    A_LIVE      055 라이브 컷(msg 14 직전) + 재료        ← x335 B_DOCS 재현(같은 런 내 기준점)
    B_TRAVEL    **구성 손님(여행)** + 재료               ← 구성 틀이 라이브와 같은 답을 주나
    C_DOMESTIC  **구성 손님(국내·고잔고)** + 재료        ← ★진짜 부정통제: Bluest 가 **옳은** 자리
    D_NODOCS    C 와 같되 **재료 없음**                  ← 전달 효과의 크기(C − D)

⚠B 와 C 는 **같은 틀에 축만 교체**한다(문장 수·어투 동일). 상품명은 어디에도 넣지 않는다.
⚠엔진은 읽어 나르기만 한다 — 비교·제거·지목 0([[59]]·[[62]] ③).
⚠손님 발화는 내가 썼다(저작). 출처는 **선언 문서의 축**뿐이고 gold 를 보지 않았다([[23]]).
  그래서 이 프로브는 *레버* 가 아니라 **계기**다 — 여기서 얻은 문구를 A2/엔진에 심지 않는다.

## 판정 (사전 고정 · 잡음 바닥 ±4·C483 ⇒ **차 ≥5 만 인용**)

    C_DOMESTIC BLUEST ≥18                  → **선언 수치로 방향을 뒤집는다** ⇒ 비교는 실재하고
                                             전달이 그것을 산다. **비교기를 짓지 않는다**([[62]] ②)
    C_DOMESTIC BLUEST ≤6 ∧ B_TRAVEL PURPLE ≥18
                                           → 문서를 읽고도 **한쪽으로만 간다** = 이름/낱말 고착
                                             ⇒ 다음 레버는 **축자 표면화**(비교기는 아직 아니다)
    B ≈ C (같은 답)                        → 손님 축을 안 본다 ⇒ x335 의 24/24 는 **비교 아님**
    C − D ≥5                               → 전달이 그 뒤집기를 샀다(전달 효과 확증)
    A_LIVE PURPLE ≤18                      → 재현 실패 = 런 간 조건 불일치 ⇒ 이 런 무효

실행(리모트·8141·[[30]] — **유료 런이 없을 때만**):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x335b_eligibility_flip_iso.py [블록크기] [블록수]
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
MISS, GOLD = "BLUEST", "PURPLE"
MAXTOK = 60                       # ★고정(팔 간 비교 보존)

HEAD = ("You are the banking agent. A customer is asking you to pick a personal checking account "
        "for them. Below is what they told you.")
ASK = ("\n[instruction] Do NOT call any tool. Reply with ONE line only: the full official name of "
       "the ONE checking account class you would open for this customer, nothing else.")

# 같은 틀·축만 교체(상품명 없음·전부 문서에 선언된 축의 언어)
TRAVEL = ("I travel internationally three or four times a year and use ATMs abroad constantly, so "
          "I want those operator fees reimbursed. I also hate paying a fee on every purchase I make "
          "in a foreign currency, and I sometimes get paid in yen or euros, so holding those "
          "currencies would help. I can keep about four thousand dollars in this account, five at "
          "most.")
DOMESTIC = ("I never travel abroad and I never spend in a foreign currency, so none of that matters "
            "to me. What I do is send about eight domestic wires every month, I deposit large "
            "checks by phone (often nine or ten thousand dollars at a time), and I run up around "
            "forty dollars a month in ATM operator fees that I want reimbursed. I keep a hundred "
            "and fifty thousand dollars sitting in my checking account at all times and I would "
            "like it to earn interest where it sits.")


def classify(text):
    t = (text or "").upper()
    return (MISS in t, GOLD in t)


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    sim = next(s for s in F.scored(TAG) if F.task_id(s) == TASK)
    live = "\n".join([B.HEAD, "", B.transcript(sim, CUT)])
    a2 = json.load(io.open(A2, encoding="utf-8"))
    material, stat = S.material_for(a2, GROUP, doc_dir=DOCS, windowed="general", per_doc=400)

    def made(body):
        return "\n".join([HEAD, "", "[user] " + body])

    arms = (("A_LIVE", live + "\n\n" + material + ASK),
            ("B_TRAVEL", made(TRAVEL) + "\n\n" + material + ASK),
            ("C_DOMESTIC", made(DOMESTIC) + "\n\n" + material + ASK),
            ("D_NODOCS", made(DOMESTIC) + ASK))
    print("x335b · 재료 %d자(링크 %d·읽음 %d·유지 %d) · 라이브 %d자 · 구성 %d/%d자 · %d×%d블록\n"
          % (len(material), stat["linked"], stat["read"], stat["kept"], len(live),
             len(TRAVEL), len(DOMESTIC), k, nb))

    res = {}
    for label, body in arms:
        mb, gb = [], []
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
            mb.append(mv)
            gb.append(gv)
        res[label] = (sum(mb), mb, sum(gb), gb)
        print("%-11s BLUEST %d/%d %s · PURPLE %d/%d %s\n"
              % (label, sum(mb), k * nb, mb, sum(gb), k * nb, gb))

    print("판정(사전 고정·차 ≥5 만 인용): C BLUEST≥18 → **수치로 뒤집는다**(비교기 금지·전달이 산다) · "
          "C≤6 ∧ B PURPLE≥18 → 한쪽 고착(다음은 축자 표면화) · B≈C → 손님 축 미사용(x335 24/24 는 비교 아님) · "
          "C−D≥5 → 전달이 뒤집기를 샀다 · A_LIVE PURPLE≤18 → 재현 실패 = 이 런 무효")
    print("측정치: " + " · ".join("%s BLUEST=%d%s PURPLE=%d%s" % (a, v[0], v[1], v[2], v[3])
                                 for a, v in res.items()))


if __name__ == "__main__":
    sys.exit(main() or 0)
