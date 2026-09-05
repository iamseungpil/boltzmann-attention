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
    ("sibling_paren_strip", "t2_gate_patch.py", "t2_gate_patch.py"),
    ("distinct_args_violation", "t2_gate_patch.py", "t2_gate_patch.py"),
    ("iso_split_injection", "t2_scaffold_get.py", "t2_scaffold_get.py"),
    ("iso_keys_satisfied", "t2_scaffold_get.py", "t2_scaffold_get.py"),
    ("view_thresholds", "t2_gate_patch.py", "t2_gate_patch.py"),
    ("group_dup_value", "t2_gate_patch.py", "t2_gate_patch.py"),
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
    """호출부는 env 뒤에 있어야 같은 sha 로 대조군을 돌릴 수 있다([[54]]).

    ⚠2026-09-05: 앵커를 **맨이름**(`src.index(flag)`)에서 `os.environ.get("<flag>")` 로 옮겼다.
      옛 앵커는 그 이름을 **주석·독스트링에서 먼저** 만나면 엉뚱한 구간을 봤다(§T-8 무장 때
      실제로 그렇게 깨졌다). 스위치의 실물은 env 읽기이므로 그것을 앵커로 삼는 편이 더 좁다.
    """
    src = _src("t2_gate_patch.py")
    for fn, flag in (("sibling_paren_arg", "T2_SIBLING_PAREN"),
                     ("sibling_paren_strip", "T2_SIBLING_PAREN"),
                     ("distinct_args_violation", "T2_DISTINCT_ARGS"),
                     ("group_dup_value", "T2_GROUP_DUP")):
        anchor = 'os.environ.get("%s")' % flag
        assert anchor in src, "%s 를 읽는 자리가 없다 — 스위치가 아예 없다" % flag
        i = src.index(anchor)
        seg = src[i:i + 1400]
        assert fn + "(" in seg, "%s 호출부가 %s 스위치 안에 없다" % (fn, flag)


def test_launcher_value_is_admitted_by_the_switch():
    """★[[84]] — **레버가 강제하는 형식은 상대와 짝이다**. 호출부가 있어도 정본 런처가 내보내는
    값을 스위치가 **안 받으면** 그 레버는 라이브에서 죽는다(=[[81]] 그대로 재발).

    2026-09-05 실물: §T-8 무장 후 부정통제 NC-C 로 스위치 목록을 `("log","deny")` 로만 되돌렸더니
    `go_stack.sh` 는 여전히 `=strip` 을 내보내는데 위 두 검정이 **둘 다 통과**했다. 이 검정이
    그 구멍이다 — 런처의 값 축자를 읽어 스위치 조건식 구간에 그 리터럴이 있는지 본다.
    """
    src = _src("t2_gate_patch.py")
    go = _src("go_stack.sh")
    for flag in ("T2_SIBLING_PAREN",):
        exports = re.findall(r"^\s*export\s+" + flag + r"=(\S+)", go, re.M)
        assert exports, "%s 가 정본 런처(go_stack.sh)에 없다 — [[81]] 미등재" % flag
        val = exports[-1].strip("\"'")          # 마지막 export 가 이긴다(shell 의미론)
        anchor = 'os.environ.get("%s")' % flag
        i = src.index(anchor)
        # ⚠축자 grep 은 여기서 **못 잡는다**: 안쪽 `== "strip"` 가 같은 구간에 남아 있으면
        #   바깥 관문이 그 값을 버려도 통과한다(NC-C 실물). 조건식을 **판정**해야 한다.
        tail = src[i + len(anchor):src.index("\n", i)]
        m_in = re.match(r"\s*(not\s+)?in\s*\(([^)]*)\)", tail)
        m_eq = re.match(r"\s*(==|!=)\s*(\"[^\"]*\"|'[^']*')", tail)
        if m_in:
            allowed = set(re.findall(r"\"([^\"]*)\"|'([^']*)'", m_in.group(2)))
            allowed = {a or b for a, b in allowed}
            ok = (val in allowed) != bool(m_in.group(1))
        elif m_eq:
            ok = (val == m_eq.group(2).strip("\"'")) == (m_eq.group(1) == "==")
        else:
            raise AssertionError("%s 스위치 조건식을 못 읽었다: %r" % (flag, tail[:80]))
        assert ok, "%s: 런처는 %r 를 내보내는데 관문이 그 값을 **안 받는다** — 라이브에서 죽는다 (%r)" \
                   % (flag, val, tail[:80])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
