#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`_over_rows` 소속 술어 래칫 (C+A · 2026-08-29 · 사용자 지시 *"C A 진행하라"*).

## 왜 (밤샘 4런 실측)

세기 기반 술어(`sub > kind_rows`)가 **39회** 총액 단언을 막았고 **구제한 sim 은 0** 이다.
막힌 값은 `27.00 · 4.75 · 3.70` = 그 태스크 gold 넷 중 셋이고, 틀린 총액(chk_2 의 5·7)은
**한 번도** 안 막았다. 그리고 '초과 1' 은 그 태스크의 **상수**다 — 같은 `(sub, kind)` 조합이
t7378 에도 있었고 거기서는 이 경로가 없어 총액이 그대로 나갔고 **2/4 통과**했다([[57]] 부정통제).

분모가 틀렸기 때문이다: `type:` 텍스트 개수는 감사 대상 집합의 **상계가 아니다**(같은 파일
`_short_rows` 독스트링의 *"수수료 줄이 없는 인출"*). ⇒ 세기 대신 **소속**을 묻는다.

## 이 검정이 지키는 것

  C  A2 에 `return_template_over` 가 **없다** — 있으면 호출부가 총액을 들어낸 문면으로 갈아탄다.
  A  술어가 alien/conflict 로만 참이 된다(세기 초과만으로는 거짓).

실행: PYTHONIOENCODING=utf-8 py -3 test_over_rows_membership.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import t2_scaffold_get as S                                      # noqa: E402

FAIL = []
A2_COPIES = ["a2/banking_knowledge.gate.json",
             "a2/banking_knowledge.specific.json",
             "a2/split/banking_knowledge.core.json"]
TOOL = "get_atm_fee_discrepancies"


def chk(c, m, extra=""):
    if not c:
        FAIL.append(m)
    print("  %s %s%s" % ("ok  " if c else "FAIL", m, ("  " + str(extra)) if extra else ""))


def tool(path):
    d = json.load(io.open(os.path.join(HERE, path), encoding="utf-8"))
    for t in d.get("scaffold_get_tools") or []:
        if t.get("name") == TOOL:
            return t
    return {}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("① C — 총액을 들어내는 대체 문면이 선언에서 빠졌다")
    for p in A2_COPIES:
        t = tool(p)
        chk("return_template_over" not in t, "%s 에 over 템플릿 없음" % p.split("/")[-1])
    chk(all("return_template" in tool(p) for p in A2_COPIES), "기본 반환문은 그대로 있다")
    chk(all(tool(p).get("_note_row_count_over") for p in A2_COPIES), "되돌림 근거가 선언에 적혔다")
    src = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
    chk('if _over and d.get("return_template_over"):' in src,
        "호출부가 선언 유무로 갈린다(미선언 = 종전 문면)")
    chk("대체 템플릿 미선언이라" in src, "미선언 경로가 로그만 남긴다")

    print("② A — 술어가 세기가 아니라 소속으로 참이 된다")
    base = {"kind": "atm_withdrawal", "kind_rows": 17, "sub": 18}
    chk(S._over_rows(dict(base, alien=0, conflict=0)) is None,
        "세기 초과 1 · 이물 0 · 충돌 0 → 판정 안 함(074 의 상수)")
    chk(S._over_rows(dict(base, alien=1, conflict=0)) == (1, "atm_withdrawal", 17),
        "이물 1행 → 판정")
    chk(S._over_rows(dict(base, alien=0, conflict=3)) == (3, "atm_withdrawal", 17),
        "내용충돌 3건 → 판정(원 계기 t7378 s361454 의 중복 3행)")
    chk(S._over_rows({"kind": None, "kind_rows": 0, "sub": 9, "alien": 2}) is None,
        "종류 미선언이면 판정 안 함")
    chk(S._over_rows(None) is None, "dict 아니면 판정 안 함")
    chk(S._over_rows({"kind": "x", "kind_rows": 3, "sub": 99}) is None,
        "새 키가 없는 옛 호출자에게도 안전(기본 0)")

    print("③ 재료 — 두 수가 실제로 만들어져 저장된다")
    chk("_alien_rows, _conf_rows = 0, 0" in src, "두 계수기가 초기화된다")
    chk('"alien": _alien_rows, "conflict": _conf_rows' in src, "`_sr` 에 저장된다")
    chk("if _rid and not any(_rid in _t for _t in _ok_outs):" in src,
        "이물 판정 = 원천 텍스트에 id 가 없나(문자열 포함 검사 하나)")
    chk("세기-초과" in src and "소속-초과" in src,
        "두 판정을 나란히 로그로 남긴다(다음 런이 어긋남을 잰다)")

    print("④ 대칭 — 부족(`_short_rows`) 축은 손대지 않았다")
    chk(all("return_template_short" in tool(p) for p in A2_COPIES), "short 문면은 그대로")
    chk("def _short_rows(" in src, "`_short_rows` 그대로")

    print()
    if FAIL:
        print("FAILED %d" % len(FAIL))
        for f in FAIL:
            print("  - %s" % f)
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
