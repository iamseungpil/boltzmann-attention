# -*- coding: utf-8 -*-
r"""**플래그 레지스트리 래칫** — 엔진이 읽는 `T2_*` 중 정본에 **선언되지 않은 것**이 늘지 않게 한다.

## 왜 (2026-08-18·C521⒤ 확장)

`go_stack.sh:2` 는 스스로를 *"정본 GO-STACK 런처 (single source of truth)"* 라 선언하지만 마지막
커밋은 **2026-08-09** 이고, 그 뒤 만들어진 레버들이 **개별 `run_t73xx.sh` 의 export 줄에만** 산다.
워크플로는 그 수를 22종으로 봤는데 **전수로 세면 130종**이다(엔진이 읽는 268 중).

무엇이 문제인가 — 세 가지가 동시에 깨진다:
  ⑴ **"스택"이 정의되지 않는다.** 어떤 런의 거동은 *어느 런처를 썼는가*에 달린다(비교 불가).
  ⑵ **[[24]] 역방향 감사의 사각지대**: `t2_levers.py` 레지스트리에도 없으면 *선언됐는데 死코드인가*를
     물을 대상 자체가 없다. 실제로 `T2_PROV_OURS` 는 어떤 셸도 export 하지 않아 死배선이었고
     (C522), `T2_PENDING_DISCOVERED` 도 같은 상태다.
  ⑶ **동시 침묵**: 서로 AND 로 물린 레버들(`SEARCH_ON_PROCEED`·`ACTION_INDEX`·`DECISION_CARRY`)은
     드라이버가 **한 줄만 빠뜨려도 함께 죽는다**.

## 이 검정이 하는 일 (거동 변경 0)

지금의 미선언 집합을 **기준선으로 박고**, 그보다 **늘면 실패**한다(래칫). 줄면 통과하고 기준선을
갱신하라고 알린다. 플래그를 켜거나 끄지 않는다 — 오직 *선언되지 않은 채 새로 생기는 것*을 막는다.

⚠이 검정이 통과한다고 스택이 정의된 것은 아니다. 130 을 줄이는 것은 별도 작업이고, 그 작업은
  **켜고 끄는 결정을 동반**하므로 측정 없이 하면 안 된다([[60]]: 분류는 로그용·레버는 항상 켠다).
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "flag_registry_baseline.json")
TUNING = ("_CAP", "_K", "_MODE", "_TH", "_MIN", "_POS", "_DIR", "_J", "_LINECAP")


def engine_flags():
    """엔진 코드가 **실제로 읽는** 환경변수(정규식은 소스 스캔용·엔진 경로 아님)."""
    out = set()
    for fn in sorted(os.listdir(HERE)):
        if not (fn.startswith("t2_") and fn.endswith(".py")):
            continue
        s = io.open(os.path.join(HERE, fn), encoding="utf-8").read()
        out |= set(re.findall(r'environ\.get\(\s*["\'](T2_[A-Z_0-9]+)["\']', s))
        out |= set(re.findall(r'environ\[\s*["\'](T2_[A-Z_0-9]+)["\']', s))
    return out


def declared():
    s = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    return set(re.findall(r"\b(T2_[A-Z_0-9]+)\b", s))


def main():
    read, dec = engine_flags(), declared()
    gap = sorted(read - dec)
    main_gap = sorted(g for g in gap if not any(g.endswith(t) for t in TUNING))
    print("엔진이 읽는 T2_*  : %d" % len(read))
    print("go_stack 선언     : %d" % len(read & dec))
    print("미선언            : %d (그중 튜닝 노브 제외한 **본 플래그 %d**)"
          % (len(gap), len(main_gap)))

    if not os.path.exists(BASE):
        io.open(BASE, "w", encoding="utf-8").write(
            json.dumps({"undeclared": gap, "_note": "래칫 기준선 — 늘면 실패, 줄면 갱신하라"},
                       ensure_ascii=False, indent=1))
        print("기준선 신설: %s (%d개)" % (os.path.basename(BASE), len(gap)))
        return 0

    base = set((json.load(io.open(BASE, encoding="utf-8")) or {}).get("undeclared") or ())
    new = sorted(set(gap) - base)
    gone = sorted(base - set(gap))
    if gone:
        print("✅ 줄었다(%d): %s" % (len(gone), ", ".join(gone[:12])))
        print("   → 기준선을 갱신하라(이 파일을 다시 만들면 된다).")
    if new:
        print("\n⛔**새로 생긴 미선언 플래그 %d개** — 정본(`go_stack.sh`)에 선언하거나 "
              "`t2_levers.py` 레지스트리에 등재하라:" % len(new))
        for n in new:
            print("     %s" % n)
        print("\ntest_flag_registry FAIL")
        return 1
    print("\ntest_flag_registry PASS (미선언 %d개·기준선 이하)" % len(gap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
