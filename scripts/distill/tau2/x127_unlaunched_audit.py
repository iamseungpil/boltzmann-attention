# -*- coding: utf-8 -*-
"""x127 — **코드가 읽는데 런처가 안 주는 플래그**를 전수로 센다.

사용자 질문 2026-08-07: *"모든 레버는 다 켜져 있는 거 아닌가?"*
[[60]]은 *"레버는 전부 항상 켠다"* 이지만 그것은 **지시**이고, 실제로 켜지는 것은
`go_stack.sh`에 있는 것뿐이다. 오늘 하루에만 그 간극이 네 번 드러났다
(`T2_LEDGER`·`T2_SOURCE` 死배선 · `T2_NOTICE_REPEAT` 기본-ON 은닉 · `_reqs` UnboundLocal).

이 감사는 그 간극을 **수로** 만든다:
  · 코드가 `os.environ.get("T2_*")`로 **읽는** 플래그 전부
  · 그중 `go_stack.sh`가 **주지 않는** 것
  · 그리고 **기본값이 있는지**(기본 ON이면 런처에 없어도 살아 있다 = 감사의 사각)

⚠이 감사는 *플래그가 읽히는가*만 본다. **그 플래그가 켜졌을 때 실제로 발화하는가는 별개**다
(오늘 실측: `T2_SOURCE=1`을 켰는데 바깥 `T2_ARBITRATE` 블록에 막혀 도달조차 못 했다).
"""
import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r'os\.environ\.get\(\s*"(T2_[A-Z_0-9]+)"\s*(?:,\s*("[^"]*"))?\s*\)')


def main():
    read = {}                       # flag -> set(defaults seen)
    for fn in sorted(os.listdir(HERE)):
        if not fn.endswith(".py"):
            continue
        try:
            src = io.open(os.path.join(HERE, fn), encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for m in PAT.finditer(src):
            flag, dflt = m.group(1), m.group(2)
            read.setdefault(flag, set()).add(dflt)

    gs = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8",
                 errors="replace").read()
    launched = set(re.findall(r"\bT2_[A-Z_0-9]+(?==)", gs))

    missing = sorted(set(read) - launched)
    default_on = [f for f in missing if any(d and d.strip('"') == "1" for d in read[f])]
    truly_off = [f for f in missing if f not in default_on]

    print("코드가 읽는 T2_* 플래그 %d종 · go_stack이 주는 것 %d종\n"
          % (len(read), len(launched)))
    print("★런처에 **없는** 플래그 %d종" % len(missing))
    print("\n  [기본값이 1 = 런처에 없어도 살아 있다 — 감사의 사각] %d종" % len(default_on))
    for f in default_on:
        print("    %s" % f)
    print("\n  [기본값 없음 = 꺼져 있다] %d종" % len(truly_off))
    for f in truly_off:
        print("    %s" % f)

    print("\n⇒ [[60]] *\"레버는 전부 항상 켠다\"* 와의 간극이 이 %d종이다." % len(truly_off))
    print("   단 전부가 레버는 아니다 — 실험-전용·측정-격리 스위치가 섞여 있으므로")
    print("   `t2_levers.py`의 셀 배치와 대조해 **레버만** 골라야 한다.")

    try:
        import t2_levers as L
        cells = set()
        for _n, (_c, _m, _p, flags) in L.CELLS.items():
            cells |= set(flags)
        lever_off = sorted(set(truly_off) & cells)
        print("\n★그중 **셀에 배치된 레버**(= 꺼져 있으면 안 되는 것) %d종:" % len(lever_off))
        for f in lever_off:
            print("    %-28s ← %s" % (f, L.cell_of(f)))
        if not lever_off:
            print("    (없음)")
    except Exception as e:
        print("\n  t2_levers 대조 실패: %r" % (e,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
