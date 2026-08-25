# -*- coding: utf-8 -*-
"""값-목록 갈래 래칫 (2026-08-25) — 선언과 엔진 분기가 함께 살아 있는지.

왜: t7348 에서 040 의 **gold 호출 8건**이 env 에 거절됐고 사유가 `Invalid <arg>. Must be one of:`
였다(정본 `t2_forensic.action_diff` 귀속 `deny=env`). 그 열거값은 **스키마에 없다** — agent 가
보는 도구 중 enum 을 선언한 인자는 하나뿐이고 표적 도구는 discoverable 이다. 그래서 값을
**도구 사용법 문서 축자**에서 A2 로 선언했고, 엔진에 `values` 갈래를 열었다.

이 검정이 지키는 것:
  ① 병합 A2 에 `values` 를 가진 선언이 있다 (선언이 사라지면 레버가 조용히 죽는다)
  ② 값마다 `_note_` 에 **출처 인용**이 있다([[23]] — 못 대면 넣지 않는다)
  ③ 엔진에 `values` 갈래가 남아 있다 (A3 색인 경로로 되돌아가면 이 레버는 발화 0)
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    from gate_interpreter import load_domain_a2
    a2 = load_domain_a2("banking_knowledge")
    specs = [s for s in (a2.get("write_arg_enum") or []) if s.get("values")]
    assert specs, "write_arg_enum 에 values 갈래 선언이 없다 — 레버가 죽는다"
    for s in specs:
        vals = [str(v) for v in s["values"]]
        assert vals, "values 가 비었다: %r" % s.get("arg")
        note = " ".join(str(v) for k, v in s.items() if str(k).startswith("_note"))
        missing = [v for v in vals if v not in note]
        assert not missing, ("출처 인용에 없는 값이 있다(%s): %r — [[23]] 위반"
                             % (s.get("arg"), missing[:3]))
    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    assert '_sp.get("values")' in src, "엔진의 values 갈래가 사라졌다"
    print("OK write_arg_enum values: 선언 %d · 출처 인용 완비 · 엔진 갈래 생존" % len(specs))


if __name__ == "__main__":
    main()
