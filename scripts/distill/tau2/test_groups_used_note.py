#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""집계 완결 주장(`_groups_used_note`) 래칫 — 스모크 없이 **초 단위로**. (094 · OL-4)

## 왜 (2026-08-29 · t7387·t7388 실측)

`get_correct_savings_apy` 의 반환문이 *"base + highest checking boost + highest card bonus +
all relationship/tier bonuses"* 라고 **완결을 단언**했는데, 같은 계좌에 대해 한 런 안에서
세 값이 나왔다:

    t7387_hB1  클래스 3 · sub=8 rows   -> 6.1
    t7387_hB1  클래스 6 · sub=11 rows  -> 6.85   (= gold expected_apy)
    t7387·t7388 클래스 2~3 · sub=3 rows -> 6.275  (모델이 신고한 값)

갈린 것은 그때그때 들어온 성분뿐인데 문장은 늘 네 갈래를 다 반영했다고 말했다. 우리 도구가
그 대화의 유일한 권위원이므로 그 거짓이 그대로 신고됐다([[25]]).

## 이 검정이 보는 것

계산은 **안 바꾼다**(같은 입력 → 같은 값). 바뀌는 것은 문장뿐이고, 선언 그룹이 전부 왔을 때는
**한 글자도 안 붙는다**(거동 보존 = 093 형 통과 보호).

실행: PYTHONIOENCODING=utf-8 py -3 test_groups_used_note.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import t2_compute as C                                           # noqa: E402
import t2_scaffold_get as S                                      # noqa: E402

FAIL = []
A2_COPIES = ["a2/banking_knowledge.gate.json",
             "a2/banking_knowledge.specific.json",
             "a2/split/banking_knowledge.core.json"]
TOOL = "get_correct_savings_apy"


def chk(c, m, extra=""):
    if not c:
        FAIL.append(m)
    print("  %s %s%s" % ("ok  " if c else "FAIL", m, ("  " + str(extra)) if extra else ""))


def _tool(path=None):
    d = json.load(io.open(os.path.join(HERE, path or A2_COPIES[1]), encoding="utf-8"))
    for t in d.get("scaffold_get_tools") or []:
        if t.get("name") == TOOL:
            return t
    return None


def _run(tool, comps):
    ctx = {"components": comps}
    res = C.apply_op(tool["op"], ctx)
    return res, ctx, S._render_scalar(tool, ctx, res) + S._groups_used_note(tool, ctx, res)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    tool = _tool()
    groups = list((tool.get("op") or {}).get("reducers") or {})

    print("① 배선 — 엔진은 자기 집계의 전사만 하고, 선언이 없으면 아무 말도 안 한다")
    src = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
    chk("def _groups_used_note(" in src, "`_groups_used_note` 가 엔진에 있다")
    chk("_txt += _groups_used_note(d, _ctx, _res)" in src, "스칼라 렌더 뒤에 배선됐다")
    chk(S._groups_used_note({}, {"_gr_used": {"absent": ["x"]}}, 1.0) == "",
        "A2 `groups_used_note` 미선언 = 빈 문자열(거동 보존)")
    chk(S._groups_used_note(tool, {}, 1.0) == "", "`_gr_used` 없으면 빈 문자열")
    chk(S._groups_used_note(tool, {"_gr_used": {"used": groups, "absent": []}}, 1.0) == "",
        "빠진 그룹이 없으면 한 글자도 안 붙는다")

    print("② 계산은 안 바뀐다 — 문장만 바뀐다")
    full = [{"kind": g, "value": v} for g, v in zip(groups, (4.0, 1.0, 1.5, 0.25, 0.1))]
    res_full, ctx_full, txt_full = _run(tool, full)
    chk(abs(res_full - 6.85) < 1e-9, "다섯 그룹 전부 → 6.85(종전과 같은 산수)", res_full)
    chk("[components]" not in txt_full, "완결 입력에는 주석이 안 붙는다")
    res_base, ctx_base, txt_base = _run(tool, [{"kind": groups[0], "value": 4.0}])
    chk(abs(res_base - 4.0) < 1e-9, "반쪽 입력도 값은 종전과 같다(abstain 아님)", res_base)

    print("③ 반쪽 입력이면 **무엇이 빠졌는지** 말한다([[64]])")
    chk("[components]" in txt_base, "주석이 붙는다")
    for g in groups[1:]:
        chk(g in txt_base.split("[components]")[1], "빠진 그룹 '%s' 를 이름으로 댄다" % g)
    chk(groups[0] in txt_base.split("[components]")[1], "들어온 그룹도 이름으로 댄다")

    print("④ 사이드채널 — `_gr_used` 는 사실만 담는다(판단 0)")
    gu = ctx_base.get("_gr_used") or {}
    chk(sorted(gu.get("declared") or []) == sorted(groups), "declared = A2 reducers 키")
    chk(gu.get("used") == [groups[0]], "used = 실제로 온 그룹", gu.get("used"))
    chk(sorted(gu.get("absent") or []) == sorted(groups[1:]), "absent = declared − used")

    print("⑤ 반환문이 더는 완결을 단언하지 않는다")
    tmpl = tool.get("return_template") or ""
    chk("computed from the components supplied in this call" in tmpl,
        "값의 출처를 '이 호출에 들어온 성분' 으로 좁혔다")
    chk("stacking policy is" in tmpl, "정책 문장 자체는 남는다(사실이므로)")

    print("⑥ A2 3사본 동일([[24]])")
    sigs = {json.dumps({k: (_tool(p) or {}).get(k) for k in
                        ("return_template", "groups_used_note", "op")},
                       ensure_ascii=False, sort_keys=True) for p in A2_COPIES}
    chk(len(sigs) == 1, "세 사본이 바이트 동일", "%d 종" % len(sigs))

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
