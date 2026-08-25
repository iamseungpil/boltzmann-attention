# -*- coding: utf-8 -*-
"""T2_ARG_POLICY_AT_WRITE 래칫 (2026-08-25) — 조인이 **동일성**인가, 그리고 고르지 않는가.

왜: 이것은 `write_rules` 의 일반형이다. 손으로 고른 문장 대신 *이 write 가 선언한 인자 이름과
A3 행의 `axis` 가 같은* 행을 놓는다. 어제 한 번 **토큰 검색기**를 짰다가 검산에서 엉뚱한 문장을
집어 폐기했다([[71]]③ bm25·embedding 금지). 그 표류가 되돌아오지 않도록 여기서 고정한다.

이 검정이 지키는 것:
  ① **동일성 조인** — 축이 인자 이름과 다르면 절대 안 실린다(부분일치·유사도 금지)
  ② **축자 인용만** — 산출의 각 줄이 A3 파일의 `quote` 에 그대로 있다(요약·재서술 0)
  ③ **순위 0** — 함수 본문에 점수·유사도·상위N 어휘가 없다
  ④ **전부 아니면 전무** — 상한을 넘으면 잘라내지 않고 None(자르면 우리가 고른 것이 된다)
  ⑤ **결정론 순서** — 같은 입력에 같은 산출
  ⑥ 기본 OFF · 인자가 없으면 무발화(fail-open)
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    import t2_gate_patch as G
    from gate_interpreter import load_domain_a2

    a2 = load_domain_a2("banking_knowledge")
    rows = G._policy_facts(a2)
    assert len(rows) > 100, "A3 정본을 못 읽었다(행 %d) — 포인터가 끊겼다" % len(rows)

    quotes = set()
    axes = set()
    for r in rows:
        axes.add(str(r.get("axis") or ""))
        for s in (r.get("sources") or []):
            q = str(s.get("quote") or "").strip()
            if q:
                quotes.add((str(r.get("axis") or ""), q))

    args = ["contacted_merchant", "eligible_for_provisional_credit", "card_action"]
    txt = G._policy_rows_for(a2, args)
    assert txt, "조인이 아무것도 못 냈다 — 축 일치가 죽었다"

    # ① 동일성 · ② 축자
    for line in txt.splitlines():
        assert line.startswith("- "), line
        ax, _, q = line[2:].partition(": ")
        assert ax in args, "선언하지 않은 축이 실렸다(부분일치·유사도 의심): %r" % ax
        assert (ax, q) in quotes, "A3 에 없는 문장이 실렸다(요약·재서술 금지): %r" % q[:60]

    # 부분일치 금지: 실재하지만 **다른** 축 이름은 안 실린다
    other = sorted(a for a in axes if a and a not in args)[:1]
    if other:
        t2 = G._policy_rows_for(a2, args)
        assert other[0] not in (t2 or ""), "다른 축이 새어 들어왔다: %r" % other[0]
    # 접두만 같은 이름으로는 아무것도 안 나온다
    assert G._policy_rows_for(a2, ["contacted"]) is None or \
        "contacted_merchant" not in G._policy_rows_for(a2, ["contacted"]), \
        "접두 부분일치가 통과했다 — 동일성 조인이 아니다"

    # ⑤ 결정론
    assert G._policy_rows_for(a2, args) == txt
    assert G._policy_rows_for(a2, list(reversed(args))) == txt, "입력 순서가 산출을 바꾼다"

    # ⑥ fail-open
    assert G._policy_rows_for(a2, []) is None
    assert G._policy_rows_for(a2, ["존재하지않는축"]) is None

    # ④ 전부 아니면 전무
    os.environ["T2_ARG_POLICY_CAP"] = "10"
    try:
        assert G._policy_rows_for(a2, args) is None, "상한을 넘겼는데 **잘라서** 줬다"
    finally:
        os.environ.pop("T2_ARG_POLICY_CAP", None)

    # ③ 순위 0
    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    i = src.index("def _policy_rows_for(")
    body = src[i:src.index("\ndef ", i + 1)]
    q = body.index('"""')
    code = body[body.index('"""', q + 3) + 3:]
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))
    for bad in ("score", "rank", "similar", "bm25", "embed", "max(", "[:1]", "top"):
        assert bad not in code, "조인이 고르기 시작했다: %r" % bad

    # ⑥ 기본 OFF
    gs = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    assert "T2_ARG_POLICY_AT_WRITE=0" in gs, "기본 OFF 가 아니다"

    print("OK T2_ARG_POLICY_AT_WRITE: A3 행 %d · 동일성 조인 %d줄 · 축자 인용 전량 · 순위 0 · "
          "전부-아니면-전무 · 결정론 · 기본 OFF" % (len(rows), len(txt.splitlines())))


if __name__ == "__main__":
    main()
