# -*- coding: utf-8 -*-
"""X24 — 후보 축소 → 선택 분리 조성 시험 (2026-07-31·GPU만·user-sim 0).

**닫으려는 것(C265의 유일한 미확정)**: C265는 wrong-pick이 *의미 경계가 아님*을 두 통제로 보였다 —
순서만 뒤집으면 답이 8/8 바뀌고(위치 함수), 다건 허용 시 gold가 9/16 살아남는다(후보의 24~43%).
그런데 **어느 arm도 정답 0**이라 설계가 정의한 '부하'(A정답·B오답)로는 확정하지 못했다.
이 시험이 그 자리를 메운다: **좁히기와 고르기를 분리하면 정답이 나오는가.**

조성:
  Stage-A (LLM·생성)   같은 정보로 "해당하는 것 **모두**" → shortlist S (후보와 교집합·중복 제거)
  Stage-B (LLM·선택)   **S의 레코드만** 제시하고 하나 고르기
  ★결정론 몫 = 파이프라인(교집합·중복 제거·순서 정규화)뿐. 도메인 의미는 하나도 넣지 않는다.

측정(전부 사전 선언):
  ① Stage-B 정답률 vs 단발 0/16 (C265 기준선)
  ② **조건부 정답률** = gold ∈ S 인 사례만 — '하나로 줄이는' 단계 자체의 성능
  ③ **순서 안정성**: S를 정순/역순 두 번 물어 답이 같은가 (C265에서 전체 후보로는 0/8 안정)
  ④ S 크기 (좁히기의 실효)

★[[05]] 3질문 ([[17]] 상설 의무) — 이 조성이 **엔진 처방으로 승격될 때** 무엇을 요구하는가:
  1. 도메인-특화 순증? **아니다.** 두 단계 모두 같은 도구 스키마·같은 레코드 원문만 쓴다.
     결정론 몫은 집합 연산과 순서 정규화뿐이고 A2 키를 요구하지 않는다.
  2. 유동 판단을 결정론에 동결? **아니다.** 기준 판단(어느 거래가 해당하는가)은 **전부 LLM**에 남고,
     결정론은 *무엇을 물을지*와 *어떤 순서로 보여줄지*만 정한다.
  3. 모델 대신 도메인 행동 수행? **아니다.** 도구를 부르지 않는다. 질문을 두 번 나눠 던질 뿐이다.
  ⇒ 셋 다 no. 단 **비용은 있다**: 결정점마다 LLM 호출 1회 추가(토큰). 승격 판단은 ②의 크기로.

용법: py -3 x24_narrow_then_pick.py --cases txn_cases_v4.jsonl --base http://…/v1 --out r.jsonl
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x22_txnid_isoprobe as X22   # noqa: E402  (프롬프트·후보 규약 재사용 = 조성 비교 가능)

TXN = X22.TXN


def ask(model, base, prompt):
    import litellm
    try:
        r = litellm.completion(model="openai/" + model, api_base=base, api_key="x",
                               temperature=0.0, messages=[{"role": "user", "content": prompt}])
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return "ERROR: %r" % (e,)


def pick_prompt(case, ids, reverse=False):
    """S의 레코드만 담은 단일-선택 프롬프트. 목적(모델 자신이 부르려던 도구)은 그대로 둔다."""
    idx = {t: r for t, r in zip(case["candidates"], case.get("cand_records") or case["candidates"])}
    seq = list(reversed(ids)) if reverse else list(ids)
    ctx = "\n\n".join(idx.get(t, t) for t in seq)
    purpose = ("You are about to call `%s`. " % case["decision_tool"]) if case.get("decision_tool") else ""
    return ("Customer request:\n%s\n\nTransaction records available to you:\n%s\n\n%s"
            % (case["user_text"], ctx,
               purpose + "Which transaction_id should be used? Answer with exactly one id "
                         "and nothing else."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8")]
    rows = []
    for c in cases:
        cand = c["candidates"]
        # ── Stage-A: 좁히기 (LLM)
        pa = X22.prompt_for(c, "A_all")
        # 오염 가드 = x22와 동일 계약: 프롬프트가 gold 필드를 읽지 않았음을 그림자 case로 확인
        assert X22.prompt_for({k: v for k, v in c.items() if k != "gold"}, "A_all") == pa, \
            "★프롬프트가 gold 필드를 참조했다 — 중단"
        out_a = ask(a.model, a.base, pa)
        S = [t for t in dict.fromkeys(TXN.findall(out_a)) if t in cand]   # 결정론: 교집합·중복 제거
        if not S:
            S = list(cand)      # 좁히기 실패 = 전체 유지(불리하게·정직하게)
        # ── Stage-B: 고르기 (LLM) — 정순/역순 두 번 = 순서 안정성 계측
        b1 = ask(a.model, a.base, pick_prompt(c, S, reverse=False))
        b2 = ask(a.model, a.base, pick_prompt(c, S, reverse=True))
        g1 = next(iter(TXN.findall(b1)), None)
        g2 = next(iter(TXN.findall(b2)), None)
        row = {"task": c["task"], "trial": c["trial"], "gold": c["gold"],
               "n_cand": len(cand), "n_short": len(S), "gold_in_S": c["gold"] in S,
               "pick_fwd": g1, "pick_rev": g2,
               "ok_fwd": g1 == c["gold"], "ok_rev": g2 == c["gold"],
               "stable": (g1 is not None and g1 == g2)}
        rows.append(row)
        print("  %-10s |S|=%-3d gold∈S=%-5s fwd=%-5s rev=%-5s stable=%s"
              % (row["task"], row["n_short"], row["gold_in_S"], row["ok_fwd"], row["ok_rev"],
                 row["stable"]))

    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    ins = [r for r in rows if r["gold_in_S"]]
    print("\n=== 판정 (사전 선언 4지표)")
    print(" ① Stage-B 정답 (정순): %d/%d   ← C265 단발 기준선 0/16" % (sum(r["ok_fwd"] for r in rows), n))
    print("    Stage-B 정답 (역순): %d/%d" % (sum(r["ok_rev"] for r in rows), n))
    print(" ② 조건부(gold∈S 인 %d건)만: 정순 %d · 역순 %d"
          % (len(ins), sum(r["ok_fwd"] for r in ins), sum(r["ok_rev"] for r in ins)))
    print(" ③ 순서 안정: %d/%d   ← C265 전체후보 기준선 0/8" % (sum(r["stable"] for r in rows), n))
    print(" ④ 축소율: |S| 중앙값 %d / 후보 중앙값 %d"
          % (sorted(r["n_short"] for r in rows)[n // 2], sorted(r["n_cand"] for r in rows)[n // 2]))
    print("\n⚠읽는 법: ②가 오르면 '하나로 줄이기'는 **정보 부족이 아니라 절차 문제**였다는 뜻이고,")
    print("  ③이 낮으면 축소해도 위치가 이긴다는 뜻이다(그러면 선택 자체를 결정론으로 내려야 한다).")


if __name__ == "__main__":
    main()
