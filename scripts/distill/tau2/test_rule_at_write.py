# -*- coding: utf-8 -*-
"""write_rules 래칫 (2026-08-25) — 선언·출처·엔진 분기가 함께 살아 있는지.

왜: T2_RULE_AT_WRITE 는 **선언된 절차 문장을 결정점에 그대로 싣는** 것 하나만 한다. 검색기도
순위도 없다(초판은 궤적을 토큰으로 긁다가 검산에서 다른 도구의 unlock 문면을 집어 폐기됐다).
그래서 이 레버의 품질은 **A2 선언의 품질**이 전부다 — 출처를 못 대는 문장이 들어오는 순간
[[23]] 위반이고 실험이 무효가 된다.

이 검정이 지키는 것:
  ① 병합 A2 에 write_rules 선언이 있다 (사라지면 레버가 조용히 죽는다)
  ② 문장마다 _note_ 가 있고, 그 안에 **정책 축자 인용**이 있다 (따옴표 안 문면)
  ③ 엔진이 여전히 **선언에서만** 읽는다 — _declared_rules_for 가 write_rules 를 본다
  ④ 엔진이 순위를 매기지 않는다 — 정렬/점수 어휘가 그 함수 본문에 없다
  ⑤ 기본 OFF (플래그 없이 문면이 바뀌지 않는다)
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _fn_body(src, name):
    """그 함수의 본문만 잘라낸다 — 파일 전체를 보면 이웃 코드가 검정을 통과시킨다."""
    i = src.index("def %s(" % name)
    j = src.find("\ndef ", i + 1)
    return src[i:j if j > 0 else len(src)]


def main():
    from gate_interpreter import load_domain_a2
    a2 = load_domain_a2("banking_knowledge")
    rules = a2.get("write_rules") or []
    assert rules, "write_rules 선언이 없다 — T2_RULE_AT_WRITE 가 발화할 수 없다"
    for r in rules:
        t = str(r.get("text") or "").strip()
        assert t, "text 가 빈 선언이 있다: %r" % (r.get("applies_to"),)
        assert str(r.get("applies_to") or "").strip(), "applies_to 가 없다: %r" % t[:40]
        note = " ".join(str(v) for k, v in r.items() if str(k).startswith("_note"))
        assert note, "출처 주석이 없다([[23]]): %r" % t[:60]
        quoted = re.findall(r'"([^"]{20,})"', note)
        assert quoted, ("_note_ 에 **축자 인용**(따옴표 안 20자 이상)이 없다 — "
                        "출처를 못 대면 넣지 마라([[23]]): %r" % t[:60])

    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    body = _fn_body(src, "_declared_rules_for")
    assert '"write_rules"' in body, "엔진이 write_rules 를 더 이상 읽지 않는다"
    for bad in ("sorted(", "score", "rank", "similar", "bm25", "embed"):
        assert bad not in body, ("_declared_rules_for 가 순위를 매기기 시작했다(%r) — "
                                 "엔진은 고르지 않는다([[62]]④)" % bad)
    assert 'os.environ.get("T2_RULE_AT_WRITE") == "1"' in src, "플래그 술어가 사라졌다"
    assert "T2_RULE_AT_WRITE=0" in io.open(os.path.join(HERE, "go_stack.sh"),
                                           encoding="utf-8").read(), "기본 OFF 가 아니다"
    print("OK write_rules: 선언 %d · 출처 축자 인용 완비 · 엔진은 선언에서만 읽고 순위 0 · 기본 OFF"
          % len(rules))


if __name__ == "__main__":
    main()
