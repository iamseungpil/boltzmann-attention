# -*- coding: utf-8 -*-
r"""x364c — 자격 문면을 고치면 라벨이 붙는가 (구현 前 마지막 무료 측정·[[55]] 2단계 "우리 문구").

## 왜

`x364` ①: 라벨 적중 **22/29(76%)** 인데 오류 7건이 **전부 같은 방향**이다 —
gold 가 사업자인 태스크(024·025·047·056·062·070·071)에서 모델이 `INDIVIDUAL` 이라 답했다.
모델 탓이 아니라 **질문이 틀렸다**: 024 의 Marcus Chen 은 *"개인(사람)"* 이면서 *"사업 목적"* 으로
카드를 원한다. 물어야 할 것은 *"이 사람이 개인인가"* 가 아니라 **"어느 자격으로 묻는가"** 다.

`x364b` 후속 계수: 라벨이 **틀린 7건에서도 해악 0건**(B 가 A 보다 나빠진 적 없음)이고, 이득은
**라벨이 맞은 20건에 전부**(EXTRA 6→2·BIZ 3→0) 몰려 있다 ⇒ 문면 수리는 **순수 상승**이다.

## 셀 (태스크마다·det·같은 재료 = 손님 발화 = x364 가 고른 재료원 F_SAY)

    W0  원문(x364 축자)   "Is this customer an individual person or a business entity?"
    W1  자격 프레임        "Is this customer asking as a personal customer, or on behalf of a business?"
    W2  상품 프레임        "Is the customer asking about products for personal use, or for a business?"

라벨 채점은 **분석 전용**(gold 축의 군이 `business_` 로 시작하면 BUSINESS·[[23]] 레버 무관).
엔진 몫은 첫 낱말이 **닫힌 두 값 중 하나인가** + 인용 실재(`quote_in`)뿐이다([[66]]).

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    최고 문면의 라벨 적중 ≥26/29(90%) ∧ 인용 검산 ≥26/29 → **그 문면으로 구현**
    어느 문면도 <26                                       → 원문(W0)으로 구현(무해가 실측됐다)
    W1·W2 가 W0 보다 **낮으면**                            → 원문 유지 · 이 프로브는 음성 기록

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
      /home/woori/venvs/seka_env/bin/python x364c_elig_wording.py [part] [nparts]
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

import t2_search as TS                                            # noqa: E402
import x341_docbody_verdict as X341                               # noqa: E402
import x357_verdict_carry_multitask as M                          # noqa: E402
import x364_eligibility_axis_iso as E                             # noqa: E402

HEAD = "Below is what is known about a customer.\n\n{record}\n\n"
TAIL = ("Answer on the first line with {a} or {b}. On the second line, quote VERBATIM one line "
        "from the text above that decides your answer. Nothing else.")
WORDINGS = {
    "W0": (E.ELIG, ("INDIVIDUAL", "BUSINESS")),
    "W1": (HEAD + "Is this customer asking as a personal customer, or on behalf of a business? "
           + TAIL.format(a="PERSONAL", b="BUSINESS"), ("PERSONAL", "BUSINESS")),
    "W2": (HEAD + "Is the customer asking about products for their own personal use, or for a "
           "business? " + TAIL.format(a="PERSONAL", b="BUSINESS"), ("PERSONAL", "BUSINESS")),
}


def read_head(ans, values):
    """첫 낱말이 **닫힌 두 값 중 하나**인가만 본다(엔진 판단 0·[[22]] 닫힌 술어)."""
    ls = [x.strip().strip('"').strip() for x in str(ans or "").split("\n") if x.strip()]
    if not ls:
        return None, ""
    head = ls[0].upper().replace("*", "").strip()
    v = next((x for x in values if head.startswith(x)), None)
    q = next((l for l in ls[1:] if TS.quote_in(l, ans)), "")
    return v, q


def main():
    part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    form = {}
    for fn in sorted(os.listdir(E.REPORTS)):
        if fn.startswith("x364_part") and fn.endswith(".json"):
            form.update((json.load(io.open(os.path.join(E.REPORTS, fn), encoding="utf-8"))
                         or {}).get("form") or {})
    keys = [t for i, t in enumerate(sorted(form)) if i % nparts == part]
    print("x364c · 조각 %d/%d · 태스크 %d개 · 문면 %d종" % (part, nparts, len(keys), len(WORDINGS)))
    print("판정(사전 고정): 최고 문면 라벨 ≥26/29 ∧ 인용 ≥26/29 → 그 문면으로 구현 · "
          "전부 미달 → 원문 W0 유지(무해 실측) · W1/W2 가 W0 보다 낮으면 원문 유지\n")

    res = []
    for tid in keys:
        req = M.instructions(tid)
        gold = form[tid]["label"]                  # 분석 라벨(BUSINESS/INDIVIDUAL)
        row = {"task": tid, "gold": gold, "out": {}}
        for wk, (tpl, vals) in sorted(WORDINGS.items()):
            ans, det = E.det_ask(tpl.format(record=req[:6000]), 200)
            v, q = read_head(ans, vals)
            # 값 이름이 문면마다 다르므로 **BUSINESS 인가**로만 비교한다(분석 전용 정규화).
            is_biz = None if v is None else (v == "BUSINESS")
            ok = int(is_biz is not None and is_biz == (gold == "BUSINESS"))
            row["out"][wk] = {"v": v, "q": q, "ok": ok, "cited": int(bool(q)), "det": det}
        res.append(row)
        print("   %-9s gold=%-11s %s" % (tid, gold, " · ".join(
            "%s %s%s" % (k, row["out"][k]["v"], "" if row["out"][k]["ok"] else "✗")
            for k in sorted(row["out"]))))

    print("\n" + "=" * 96)
    for wk in sorted(WORDINGS):
        rs = [r["out"][wk] for r in res]
        print("%-3s n=%-3d 라벨 적중 %-3d 인용 검산 %-3d 비결정 %d"
              % (wk, len(rs), sum(x["ok"] for x in rs), sum(x["cited"] for x in rs),
                 sum(0 if x["det"] else 1 for x in rs)))
    out = os.path.join(E.REPORTS, "x364c_part%d.json" % part)
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
