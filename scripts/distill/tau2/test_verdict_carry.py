# -*- coding: utf-8 -*-
"""`t2_search.verdict_lines` + 결정점 배선 검정 (L-V 판정 이월).

리뷰 지적을 핀으로 박는다:
  ⑤ **후보를 제거하지 않는다** — 전 후보가 줄로 실린다(제거형은 별개 레버).
  ⑥ 엔진이 읽는 토큰은 **VIOLATES/OK** 이고 `CONFLICTS` 는 아니다(오코딩 시 판정이 통째로 죽는다).
  ⑦ 후보 수 상한 초과 시 **미발화**(조용한 절단 금지).
  + fail-safe: 근거 미검산이어도 후보를 남긴다 · 줄 0개면 호출부가 종전 재료로 떨어진다.
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as TS

A2DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
DOC_A = "Silver Plus Account. Up to 15 free withdrawals per month. Daily compounding."
DOC_B = "Green Account (savings). Monthly fee applies after 3 withdrawals."
CORPUS = {"doc_a": DOC_A, "doc_b": DOC_B}
SPEC = {"doc_index": {"g": {"silver_plus_account": ["doc_a"], "green_account": ["doc_b"]}},
        "verdict_prompt": "{req}\n{doc}", "verdict_line_template": "- {name}: {verdict}{why}",
        "verdict_max_candidates": 12}


class _FakeSC(object):
    """서브콜 대역 — 후보마다 정해진 답을 준다(모델 호출 없음)."""
    def __init__(self, answers):
        self.answers = answers
        self.i = 0

    def sub_generate(self, agent, la, UM, body, tag):
        a = self.answers[min(self.i, len(self.answers) - 1)]
        self.i += 1
        return a


def run(answers, spec=None):
    real = TS.SC
    TS.SC = _FakeSC(answers)
    try:
        return TS.verdict_lines("agent", "la", "UM", spec or SPEC, "req", "g", corpus=CORPUS)
    finally:
        TS.SC = real


def main():
    bad = 0

    # ⚠후보는 **슬러그 정렬**로 돈다: green_account → silver_plus_account. 답도 그 순서로 준다
    #   (1차 작성에서 내가 뒤집어 적어 cited=0 이 났다 — 코드가 아니라 테스트가 틀렸다).
    lines, st = run(["VIOLATES\nMonthly fee applies after 3 withdrawals.", "OK\nDaily compounding."])
    print("① 두 후보 다 실린다: %s" % lines)
    if len(lines) != 2 or st["OK"] != 1 or st["VIOLATES"] != 1:
        print("   FAIL — 제거가 일어났거나 판정이 안 읽혔다"); bad += 1
    if st["cited"] != 2:
        print("   FAIL — 근거 검산 수가 %d(기대 2)" % st["cited"]); bad += 1

    lines, st = run(["CONFLICTS\nMonthly fee applies after 3 withdrawals.", "OK\nDaily compounding."])
    print("② `CONFLICTS` 는 엔진 토큰이 아니다 → UNCLEAR: %s" % [l[:40] for l in lines])
    if st["UNCLEAR"] != 1:
        print("   FAIL — VIOLATES/OK 핀이 풀렸다"); bad += 1

    lines, st = run(["VIOLATES\n(문서에 없는 문장)", "OK\n(역시 없는 문장)"])
    print("③ 근거 미검산이어도 후보는 남는다: %d줄 · cited=%d" % (len(lines), st["cited"]))
    if len(lines) != 2 or st["cited"] != 0:
        print("   FAIL — 근거 없다고 후보를 지웠다([[25]] 위반)"); bad += 1

    cap = dict(SPEC); cap["verdict_max_candidates"] = 1
    lines, st = run(["OK\nx", "OK\ny"], cap)
    print("④ 상한 초과 → 미발화: %d줄 · skip=%r" % (len(lines), st["skip"]))
    if lines or st["skip"] != "over-cap":
        print("   FAIL — 조용히 잘랐다"); bad += 1

    lines, st = run(["OK\nx"], {"doc_index": SPEC["doc_index"]})
    if lines or st["skip"] != "no-template-or-req":
        print("⑤ FAIL — 템플릿 없이 발화했다"); bad += 1
    else:
        print("⑤ 템플릿 없으면 미발화(종전 경로) OK")

    a = json.load(io.open(os.path.join(A2DIR, "banking_knowledge.specific.json"), encoding="utf-8"))
    g = json.load(io.open(os.path.join(A2DIR, "banking_knowledge.gate.json"), encoding="utf-8"))
    pa, pg = a.get("policy_ontology") or {}, g.get("policy_ontology") or {}
    same = pa.get("verdict_prompt") == pg.get("verdict_prompt") and bool(pa.get("verdict_prompt"))
    print("⑥ A2 두 층 동기화: %s" % same)
    if not same:
        print("   FAIL — [[24]] 양방향 규칙 위반"); bad += 1
    if "VIOLATES" not in str(pa.get("verdict_prompt")) or "OK" not in str(pa.get("verdict_prompt")):
        print("   FAIL — 프롬프트가 엔진 토큰을 안 요구한다"); bad += 1

    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_gate_patch.py"),
                  encoding="utf-8").read()
    wired = ("_ts.verdict_lines(" in src and "T2_VERDICT_CARRY" in src
             and "_vmat = _mat" in src)
    print("⑦ 결정점 배선 + 폴백: %s" % wired)
    if not wired:
        print("   FAIL — 배선/폴백 없음"); bad += 1

    print("\n%s" % ("test_verdict_carry PASS" if not bad else "test_verdict_carry FAIL %d건" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
