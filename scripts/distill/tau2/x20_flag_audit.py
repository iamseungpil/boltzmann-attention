# -*- coding: utf-8 -*-
"""X20 — 레버 플래그 회계 · **침묵 OFF 감사** (2026-07-31·무료).

리뷰 지적(E): 설계서가 레버 수를 손으로 적어 실측과 어긋났고("96/65/20여 개" vs 실측 103/67/36),
무엇보다 **OFF는 명시가 아니라 부재**다 — `go_stack.sh`에 `=0` 대입은 **0건**이고, 꺼진 레버는
그냥 안 적혀 있을 뿐이다. [[19]](합성-우선)이면 "왜 꺼져 있나"에 각각 답할 수 있어야 하므로,
사유가 적히지 않은 침묵 OFF는 **발사 전 감사 대상**이다.

★재현 가능하게 기계로 센다(손 목록 금지 — 그게 이 지적의 원인이다):
  · 레버 = 엔진 **import 폐포**(x6h와 동일 스코프)가 `== "1"` 계열로 읽는 `T2_*` 이름
  · ON   = go_stack.sh 가 `=1` 로 대입
  · 파라미터 = go_stack.sh 가 1이 아닌 값으로 대입(CAP·K·TH 등)
  · 침묵 OFF = go_stack.sh 에 이름 자체가 없음 → 그중 **주석에 이름이 언급되면 '사유 기재'**

용법: py -3 x20_flag_audit.py [--json out.json]
"""
import argparse
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_GO = os.path.join(_HERE, "go_stack.sh")

# `== "1"` · `in ("1",…)` · `!= "1"` 모두 on/off 레버로 본다(파라미터는 값 비교가 아니다)
_BOOL_PAT = [
    re.compile(r'environ\.get\(\s*[\'"](T2_[A-Z0-9_]+)[\'"][^)]*\)\s*[=!]=\s*[\'"]1[\'"]'),
    re.compile(r'getenv\(\s*[\'"](T2_[A-Z0-9_]+)[\'"][^)]*\)\s*[=!]=\s*[\'"]1[\'"]'),
    re.compile(r'environ\.get\(\s*[\'"](T2_[A-Z0-9_]+)[\'"][^)]*\)\s+in\s+\([^)]*[\'"]1[\'"]'),
]
# ★기본값이 "1"인 플래그 = **미설정이면 켜짐**. 초판은 "go_stack에 없다=OFF"로 세어
#   `T2_NOTICE_REPEAT`(기본 "1")를 침묵 OFF로 **오분류**했다. 부재는 OFF가 아니라 **기본값**이다.
_DEFAULT_ON = re.compile(
    r'environ\.get\(\s*[\'"](T2_[A-Z0-9_]+)[\'"]\s*,\s*[\'"]1[\'"]\s*\)')


def engine_files():
    try:
        from x6h_engine_literal_audit import discover_engine_files
        return discover_engine_files()
    except Exception:
        return ["t2_gate_patch.py", "gate_interpreter.py"]


def default_on():
    """기본값이 "1"인 = 미설정이어도 켜지는 플래그."""
    out = set()
    for f in engine_files():
        try:
            src = io.open(os.path.join(_HERE, f), encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        out |= set(_DEFAULT_ON.findall(src))
    return out


def collect():
    levers = {}
    for f in engine_files():
        try:
            src = io.open(os.path.join(_HERE, f), encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for pat in _BOOL_PAT:
            for m in pat.finditer(src):
                levers.setdefault(m.group(1), set()).add(f)
    return levers


def go_stack_state():
    src = io.open(_GO, encoding="utf-8", errors="replace").read()
    code, comment = [], []
    for line in src.splitlines():
        (comment if line.lstrip().startswith("#") else code).append(line)
    code_s, comm_s = "\n".join(code), "\n".join(comment)
    on = set(re.findall(r"\b(T2_[A-Z0-9_]+)=1\b", code_s))
    assigned = dict(re.findall(r"\b(T2_[A-Z0-9_]+)=([^\s;]+)", code_s))
    zeroed = {k for k, v in assigned.items() if v == "0"}
    return on, assigned, zeroed, comm_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    levers = collect()
    defon = default_on()
    on, assigned, zeroed, comments = go_stack_state()
    on |= defon                       # ★기본값 ON = 미설정이어도 켜져 있다
    param = {k: v for k, v in assigned.items() if v != "1" and k not in levers}
    silent, documented = [], []
    for k in sorted(levers):
        if k in on:
            continue
        if k in assigned:                      # 1이 아닌 값으로 대입 = 의도적 설정
            documented.append((k, "값=" + assigned[k]))
            continue
        (documented if k in comments else silent).append(
            (k, "주석에 사유 언급") if k in comments else k)

    print("=" * 78)
    print("레버 회계 (기계 산출 · 스코프 = 엔진 import 폐포 %d파일)" % len(engine_files()))
    print("=" * 78)
    print("  on/off 레버        %3d" % len(levers))
    print("  ON                %3d  (go_stack =1 %d + **기본값 ON** %d)"
          % (len([k for k in levers if k in on]),
             len([k for k in levers if k in on and k not in defon]),
             len([k for k in levers if k in defon])))
    print("  사유 있는 OFF      %3d  (주석 언급 또는 비-1 값 대입)" % len(documented))
    print("  ★무사유 침묵 OFF  %3d" % len(silent))
    print("  `=0` 명시적 OFF    %3d  ← **OFF는 명시가 아니라 부재다**" % len(zeroed))
    print("\n  ★무사유 침묵 OFF 목록 (발사 전 감사 대상·[[19]]):")
    for k in silent:
        print("     %-28s  읽는 곳: %s" % (k, ", ".join(sorted(levers[k]))[:60]))
    print("\n  사유 있는 OFF:")
    for k, why in documented:
        print("     %-28s  %s" % (k, why))

    # ★런처가 반드시 직접 export 해야 하는 것 — go_stack에 없으므로 빠뜨리면 조용히 안 켜진다
    must = [k for k in ("T2_DECLFIRST", "T2_DECLFIRST_GUIDE", "T2_DECLFIRST_ENFORCE",
                        "T2_FB_SIDECAR") if k not in on]
    print("\n  ⚠**런처가 직접 export 해야 하는 것**(go_stack에 없음 = 빠뜨리면 무음 실패):")
    for k in must:
        print("     %s" % k)
    print("     → 발사 직후 `env | grep ^T2_ > flags.txt` 스냅샷으로 **실제로 찍혔는지** 확인할 것.")

    if args.json:
        with io.open(args.json, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"levers": {k: sorted(v) for k, v in levers.items()},
                                "on": sorted(k for k in levers if k in on),
                                "silent_off": silent,
                                "documented_off": documented}, ensure_ascii=False, indent=1))
        print("\n  → %s" % args.json)


if __name__ == "__main__":
    main()
