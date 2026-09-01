# -*- coding: utf-8 -*-
"""★레버는 **정의만으로 존재하지 않는다** — 호출부가 있어야 존재한다([[81]]).

2026-09-01 실물: `distinct_args_violation` 을 구현하고 A2 선언까지 하고 검정 7/7 을 통과시킨 뒤
**호출부를 빠뜨렸다**. T2′ 한 런이 통째로 그 술어 없이 돌았고, 나는 결과를 그 레버 덕으로
보고할 뻔했다. 같은 사고가 이 세션에서만 세 번째다(`T2_GUIDED` 표면형 · viewscale 팔 무력화).

이 검정은 **새 술어마다 한 줄**만 추가하면 된다: 정의 파일에서 그 이름이 **정의 밖에서도**
쓰이는지 본다(호출부 존재). 발화 여부는 런이 증명하지만, **호출조차 없는 것**은 여기서 막는다.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# (술어 이름, 정의 파일, 호출부가 있어야 하는 파일)
LEVERS = [
    ("free_text_drop", "t2_gate_patch.py", "t2_gate_patch.py"),
    ("sibling_paren_arg", "t2_gate_patch.py", "t2_gate_patch.py"),
    ("distinct_args_violation", "t2_gate_patch.py", "t2_gate_patch.py"),
    ("iso_split_injection", "t2_scaffold_get.py", "t2_scaffold_get.py"),
    ("iso_keys_satisfied", "t2_scaffold_get.py", "t2_scaffold_get.py"),
    ("view_thresholds", "t2_gate_patch.py", "t2_gate_patch.py"),
]


def _src(name):
    return io.open(os.path.join(HERE, name), encoding="utf-8").read()


def test_every_lever_has_a_call_site():
    for fn, deffile, callfile in LEVERS:
        src = _src(callfile)
        uses = [m.start() for m in re.finditer(re.escape(fn) + r"\s*\(", src)]
        defs = [m.start() for m in re.finditer(r"def\s+" + re.escape(fn) + r"\s*\(", src)]
        callsites = [u for u in uses if u not in defs]
        assert callsites, "%s: 정의만 있고 호출부가 없다 — 그 레버는 없는 것이다([[81]])" % fn


def test_each_call_site_is_switched_by_an_env_flag():
    """호출부는 env 뒤에 있어야 같은 sha 로 대조군을 돌릴 수 있다([[54]])."""
    src = _src("t2_gate_patch.py")
    for fn, flag in (("sibling_paren_arg", "T2_SIBLING_PAREN"),
                     ("distinct_args_violation", "T2_DISTINCT_ARGS")):
        i = src.index(flag)
        seg = src[i:i + 900]
        assert fn + "(" in seg, "%s 호출부가 %s 스위치 안에 없다" % (fn, flag)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
