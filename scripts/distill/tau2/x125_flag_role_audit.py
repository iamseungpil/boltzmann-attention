# -*- coding: utf-8 -*-
"""플래그가 **판정을 하는가(레버)** 아니면 **경로를 여는가(채널)** — 코드에서 기계 판정.

왜 필요한가 (2026-08-07·같은 함정에 두 번 빠진 뒤):
  · `T2_GATE_REGEN`/`T2_PROV_REGEN` — 레버로 알고 껐더니 **계약 코드가 설치조차 안 됐다**.
  · `T2_READ_DEDUP`  — "계약 밖·레버 아님"으로 분류해 껐더니 **훅 본문 절반이 통째로 꺼졌고**,
    그 안에 배선한 원장 산수와 계측 probe가 여섯 번 무음이었다.
두 번 다 **arm이 조용히 능력을 잃은 채** "계약만 켠 arm"이라 불렸다. 분류를 주석이 아니라
**코드**에서 해야 하는 이유다([[55]] 우리 배관 먼저).

판정 규칙(도메인 무관·순수 구문):
  각 `os.environ.get("T2_X") == "1"` 이 **if 조건**에 쓰였을 때, 그 if가 감싸는 블록의 줄 수를 센다.
    큰 블록(≥ THRESH)         → **채널 후보**: 그 안에 다른 기능이 산다
    작은 블록                 → **레버 후보**: 국소 판정
    if 밖(대입·기본값)        → **설정 후보**
  ⚠줄 수는 *징후*지 증명이 아니다. 큰 블록 안에 무엇이 사는지는 사람이 봐야 한다 —
    그래서 블록 안의 **다른 플래그·print 태그**를 함께 찍는다(무엇을 데리고 죽는지).

usage: x125_flag_role_audit.py [FLAG ...]   (인자 없으면 '계약 밖 17개')
"""

import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
import glob as _glob
# ★엔진 파일 전수. 목록을 손으로 적으면 빠진 파일의 플래그가 "사용처 없음"으로 보이고,
#   그건 "레버 아님"과 구별되지 않는다 — 오늘 세 개가 그렇게 보였다.
SRCS = sorted(os.path.basename(x) for x in _glob.glob(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "t2_*.py")))
THRESH = 25

OUTSIDE = ["T2_OVERFLOW_GUARD", "T2_TRUNC_GUARD", "T2_ENVELOPE_GUARD", "T2_DYN_MT",
           "T2_VIEW_COMPACT", "T2_VIEW_ANNOTATE", "T2_STALE_STRIP", "T2_READ_DEDUP",
           "T2_PAIRCHECK", "T2_PAIRFIX", "T2_DUP_REPRESENT", "T2_FAILED_PERSIST",
           "T2_MAXPROMPT", "T2_GUIDED", "T2_ACTION_PROGRESS_REFUND",
           "T2_FOLLOWUP_PROGRESS_REFUND", "T2_A2_VARIANT"]


def block_span(lines, i):
    """i행이 `if ...:` 이면 그 블록의 (끝행, 줄 수)."""
    ind = len(lines[i]) - len(lines[i].lstrip())
    j = i + 1
    while j < len(lines):
        l = lines[j]
        if l.strip() and not l.strip().startswith("#"):
            if (len(l) - len(l.lstrip())) <= ind:
                break
        j += 1
    return j, j - i - 1


def aliases(lines, fl):
    """`x = ... os.environ.get("FL") ...` 형태의 **별칭 변수** 이름들.

    ★1차판이 놓친 것이 정확히 이것이다: `T2_READ_DEDUP`은 `if` 조건에 직접 안 쓰이고
    `dedup_on = (os.environ.get("T2_READ_DEDUP") == "1" ...)` 로 받은 뒤 `if dedup_on:` 이
    **훅 본문 절반**을 감싼다. 별칭을 안 따라가면 그 플래그가 "설정"으로 보이고, 끄면
    조용히 기능이 사라진다 — 오늘 여섯 번의 무음이 그 대가였다.
    """
    out = set()
    for l in lines:
        if '"%s"' % fl not in l:
            continue
        m = re.match(r"\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=", l)
        if m:
            out.add(m.group(1))
    return out


def main():
    flags = [f if f.startswith("T2_") else "T2_" + f for f in sys.argv[1:]] or OUTSIDE
    for fl in flags:
        rows = []
        # 별칭 경유 블록 (파일별)
        for fn in SRCS:
            p = os.path.join(HERE, fn)
            if not os.path.exists(p):
                continue
            lines = io.open(p, encoding="utf-8", errors="replace").read().split("\n")
            for al in aliases(lines, fl):
                for i, l in enumerate(lines):
                    s2 = l.strip()
                    if not (s2.startswith("if %s" % al) or s2 == "if %s:" % al):
                        continue
                    end, n = block_span(lines, i)
                    inner = sorted({m.group(1) for j in range(i, end)
                                    for m in re.finditer(r'"(T2_[A-Z_0-9]+)"', lines[j])} - {fl})
                    tags = sorted({m.group(1) for j in range(i, end)
                                   for m in re.finditer(r'print\("(\[[A-Z0-9_ -]+\])', lines[j])})
                    rows.append((fn, i + 1, "별칭 %s" % al, n, s2[:60], inner + tags))
        for fn in SRCS:
            p = os.path.join(HERE, fn)
            if not os.path.exists(p):
                continue
            lines = io.open(p, encoding="utf-8", errors="replace").read().split("\n")
            for i, l in enumerate(lines):
                if '"%s"' % fl not in l:
                    continue
                stripped = l.strip()
                is_if = stripped.startswith("if ") or stripped.startswith("elif ") \
                    or (stripped.startswith("and ") or stripped.startswith("or "))
                if not is_if:
                    rows.append((fn, i + 1, "설정/대입", 0, stripped[:70], []))
                    continue
                # 조건이 여러 줄이면 `if`가 있는 줄까지 거슬러 올라간다
                k = i
                while k > 0 and not (lines[k].strip().startswith("if ")
                                     or lines[k].strip().startswith("elif ")):
                    k -= 1
                end, n = block_span(lines, k)
                inner = sorted({m.group(1) for j in range(k, end)
                                for m in re.finditer(r'"(T2_[A-Z_0-9]+)"', lines[j])} - {fl})
                tags = sorted({m.group(1) for j in range(k, end)
                               for m in re.finditer(r'print\("(\[[A-Z0-9_ -]+\])', lines[j])})
                rows.append((fn, k + 1, "블록", n, lines[k].strip()[:70], inner + tags))
        if not rows:
            print("%-30s (사용처 없음)" % fl)
            continue
        big = max((r[3] for r in rows), default=0)
        verdict = ("★채널" if big >= THRESH else ("레버" if big > 0 else "설정"))
        print("=" * 100)
        print("%-30s → %-6s (최대 블록 %d줄)" % (fl, verdict, big))
        for fn, ln, kind, n, txt, inner in rows[:6]:
            print("   %-22s:%-5d %-8s %3d줄  %s" % (fn, ln, kind, n, txt))
            if inner:
                print("      └ 이 블록이 데리고 죽는 것: %s" % ", ".join(inner[:10]))


if __name__ == "__main__":
    main()
