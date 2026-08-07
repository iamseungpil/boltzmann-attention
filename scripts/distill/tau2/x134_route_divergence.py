# -*- coding: utf-8 -*-
"""x134 — 단계 1 게이트⒝: **현행이 고른 것 ≠ `route()`가 골랐을 것** 사례 목록 (유료 0).

정본 = `FACT_DAG_DESIGN_2026_08_08.md` §7e. 이 목록이 **단계 2b(`speak()` 승격)의 착수 조건**이다:
**N=0이면 2b를 착수하지 않는다** — 뒤집을 이유가 없다는 뜻이기 때문이다.

입력 = 런 stderr 로그(`$LOG/<TAG>.log`). 읽는 줄:

    [T2_STACK] audit route=[(층, 표적, 플래그), …] chose=[(채널, 표적), …] differs=True/False suppressed=[…]

⚠**갈림 판정은 표적 축으로만** 한다(`differs` = `target_differs`). 채널 이름(`proc`)과 플래그
(`T2_*`)는 이름 공간이 달라 맞대면 언제나 "다름"이 나온다 — 그 거짓 계기를 `test_audit_divergence`가
막는다.
⚠`route=[]`(등록분이 층 미분류라 통째로 버려진 경우)도 `differs=True`로 보인다. **사라진 것**과
**다르게 고른 것**은 다르므로 여기서 갈라 센다.

usage: x134_route_divergence.py <log> [<log> …] [--full]
"""

import ast
import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LINE = re.compile(r"\[T2_STACK\] audit route=(\[.*?\]) chose=(\[.*?\]) differs=(True|False) "
                  r"suppressed=(\[.*?\])\s*$")


def _lit(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def scan(paths):
    rows, seen_any = [], 0
    for p in paths:
        for ln in io.open(p, encoding="utf-8", errors="replace"):
            m = LINE.search(ln.rstrip("\n"))
            if not m:
                continue
            seen_any += 1
            pick, chose, differs, supp = (_lit(m.group(1)), _lit(m.group(2)),
                                          m.group(3) == "True", _lit(m.group(4)))
            rows.append({"log": p, "pick": pick, "chose": chose,
                         "differs": differs, "suppressed": supp,
                         "vanished": differs and not pick})
    return rows, seen_any


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    if not args:
        print(__doc__)
        return 2
    rows, seen = scan(args)
    if not seen:
        print("⚠`[T2_STACK] audit` 줄이 **0건**이다. 셋 중 하나다: ①런이 이 배선 이전 것 "
              "②stderr를 안 남겼다(오늘 6런이 그랬다) ③등록이 0이라 감사가 None을 돌려줬다.\n"
              "   ⇒ 게이트⒝는 **미측정**이지 통과가 아니다.")
        return 1

    diverge = [r for r in rows if r["differs"] and not r["vanished"]]
    vanished = [r for r in rows if r["vanished"]]
    same = len(rows) - len(diverge) - len(vanished)

    print("감사 지점 %d · **갈림 N=%d** · 사라짐 %d · 같음 %d"
          % (len(rows), len(diverge), len(vanished), same))
    if vanished:
        print("  ⚠사라짐 = `route()`가 후보를 통째로 버렸다(층 미분류 후보). 갈림으로 세지 않는다 — "
              "`test_audit_divergence.test_every_registering_lever_has_a_layer` 를 돌려라.")

    for r in (diverge if full else diverge[:20]):
        print("  · 현행 %-46s ↔ route %s"
              % (", ".join("%s→%s" % (c, t) for c, t in r["chose"]) or "(없음)",
                 ", ".join("%s/%s" % (l, t) for l, t, _f in r["pick"]) or "(없음)"))
    if not full and len(diverge) > 20:
        print("  … %d건 더(`--full`)" % (len(diverge) - 20))

    print("\n%s" % ("[게이트⒝ 충족 — 2b 착수 가능] 갈림이 실재한다"
                    if diverge else
                    "[게이트⒝ 미충족 — 2b 착수하지 않는다] 두 판정이 갈리는 자리가 없다"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
