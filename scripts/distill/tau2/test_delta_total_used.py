# -*- coding: utf-8 -*-
"""delta_total 래칫 (2026-08-26) — 엔진이 계산한 부호 합이 **실제로 인쇄되는가**.

왜: 엔진은 2026-08-13 부터 `{delta_total}`(= 남긴 delta 들의 **부호 합**)을 템플릿 인자로 내놓고
있었는데, ATM 수수료 도구의 `return_template` 이 그것을 **안 썼다** — 死설정([[24]]). 그 대가가
t7356 에서 그대로 나왔다: 모델이 부호를 버리고 절댓값을 더한다.

    lb  부호합 14.50 ↔ 제출 19.50 · dg 4.75 ↔ 10.25 · ev 3.70 ↔ 9.30
    purple 은 **음수가 하나도 없어** 27.00 으로 유일하게 정확했다(자연 실험)
    072 도 동형: 3.50 ↔ 6.50

이 검정이 지키는 것:
  ① 엔진이 `{delta_total}` 을 **부호 합**으로 계산한다(절댓값 합이 아니다)
  ② 그 인자를 쓰는 A2 템플릿이 있고, 두 정본 층이 같다([[24]])
  ③ 그 문면이 **기호와 부호를 섞지 않는다** — `${delta_total}` 처럼 붙이지 않는다
     (`$-2.50` 이 그 계열의 사고였다·사용자 지적 2026-08-26)
  ④ 엔진이 값을 **고르지 않는다** — 합만 낸다(정렬·최댓값·선택 어휘 금지)
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    src = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()

    # ① 부호 합 — abs() 가 끼면 안 된다
    i = src.index("_dtot = ")
    line = src[i:src.index("\n", i)]
    assert "sum(" in line and "delta" in line, line
    assert "abs(" not in line, "부호 합이 아니라 **절댓값 합**을 계산한다: %s" % line
    for bad in ("max(", "sorted(", "argmax"):
        assert bad not in line, "엔진이 고르기 시작했다(%r): %s" % (bad, line)
    assert "delta_total" in src, "템플릿 인자로 넘기지 않는다"

    # ②③ 선언
    from gate_interpreter import load_domain_a2
    a2 = load_domain_a2("banking_knowledge")
    users = [t for t in (a2.get("scaffold_get_tools") or [])
             if "{delta_total}" in str(t.get("return_template") or "")]
    assert users, "`{delta_total}` 을 쓰는 도구 선언이 없다 — 엔진이 계산해도 아무도 안 읽는다"
    for t in users:
        rt = str(t["return_template"])
        assert not re.search(r"[$€₩]\s*\{delta_total\}", rt), (
            "기호와 부호를 붙였다(%s) — `$-2.50` 계열 사고다" % t.get("name"))

    gate = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                             encoding="utf-8"))

    def find(o):
        if isinstance(o, dict):
            if "scaffold_get_tools" in o:
                return o["scaffold_get_tools"]
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v)
                if r is not None:
                    return r
        return None

    g = {t.get("name"): str(t.get("return_template") or "") for t in (find(gate) or [])}
    for t in users:
        assert g.get(t["name"]) == str(t["return_template"]), (
            "정본 층과 gate.json 이 갈렸다([[24]]): %s" % t.get("name"))

    print("OK delta_total: 엔진=부호 합(절댓값 아님·선택 0) · 쓰는 선언 %d · 기호/부호 분리 · 두 층 일치"
          % len(users))


if __name__ == "__main__":
    main()
