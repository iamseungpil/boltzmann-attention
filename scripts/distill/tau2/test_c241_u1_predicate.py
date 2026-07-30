#!/usr/bin/env python3
"""C241 U1' 회귀 — 실효-write 술어의 도메인별 등가성 + 누출 부재.

리뷰 §8-D 요구: "5도메인 122도구 before/after diff 0"은 **각 도메인을 자기 A2로 로드**해
돌려야 의미가 있다(구 표현이면 banking A2로 전 도메인 도는 것도 통과).

검사 3종:
  T1 등가성 — 각 도메인 A2 하에서 구 술어(하드코딩 정규식)와 신 술어(A2 파생)의 판정이 동일
  T2 누출 부재(리뷰 B2) — banking A2를 로드한 뒤 airline A2로 판정해도 banking 어휘가 새지 않음
  T3 회귀조건(`t2_gate_patch.py:4543`) — `_is_effective_write("give_…", banking_a2) is False`
"""
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import t2_gate_patch as G  # noqa: E402
from gate_interpreter import load_domain_a2  # noqa: E402

# 구 술어(U1' 이전) — 도메인 어휘가 엔진에 박혀 있던 판
_OLD_PROC = re.compile(
    r"(^log_|^verify_|_verification$|^kb_|^shell$|discoverable|transfer_to_human|^give_|^unlock_)",
    re.I)


def old_is_write(name):
    return bool(name) and not G._READ_PREFIX_RE.match(name) and not _OLD_PROC.search(name)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ns = json.load(open(os.path.join(_HERE, "tau2_domain_toolnames.json"), encoding="utf-8"))

    def names_of(dom):
        """★그 도메인이 **실제로 갖는** 도구명만. 초판은 122개 전체를 모든 도메인에 돌려
        airline/retail에서 banking 도구명 diff 7건이 나왔는데, 그 이름은 해당 도메인에
        존재하지 않으므로 **테스트 아티팩트**였다(리뷰가 요구한 '자기 A2' 원칙의 대칭)."""
        out = [x for x in ns.get(dom, []) if not x.startswith("_")]
        return sorted(set(out + ["close_credit_card_account_7834"] if dom == "banking_knowledge"
                          else out))

    fails = 0
    print("=== T1 등가성: 각 도메인을 **자기 A2**로 로드 ===")
    for dom in sorted(ns):
        a2 = load_domain_a2(dom)
        proc = G._a2_procedural(a2)
        diff = [n for n in names_of(dom)
                if old_is_write(n) != G._is_effective_write(n, a2)]
        tag = "A2 없음(=구 술어와 다를 수 있음·기대)" if a2 is None else ""
        print(f"  {dom:20s} A2={'있음' if a2 else '없음':4s} 도구 {len(names_of(dom)):3d}개 "
              f"· 절차집합 {len(proc)}개 · diff {len(diff)}건 {tag}")
        if a2 is not None and diff:
            print(f"      ⚠diff: {diff[:8]}")
            fails += 1

    print()
    print("=== T2 누출 부재(리뷰 B2): banking 로드 후 airline 판정 ===")
    b = load_domain_a2("banking_knowledge")
    assert b is not None, "banking A2 없음 — 검사 불가"
    _ = G._a2_procedural(b)                       # banking 먼저 '로드'
    air = load_domain_a2("airline")
    leak = [n for n in ("give_discoverable_user_tool", "unlock_discoverable_agent_tool",
                        "call_discoverable_agent_tool")
            if not G._is_effective_write(n, air)]
    if leak:
        print(f"  ⚠누출: airline 판정인데 banking 어휘가 절차로 잡힘 {leak}")
        fails += 1
    else:
        print("  ✓ airline 판정에 banking 어휘 누출 없음 "
              "(전역 상태 부재 = 순서 의존·last-wins 둘 다 해소)")

    print()
    print("=== T3 C211 회귀조건(`:4543`) ===")
    for n in ("give_discoverable_user_tool", "unlock_discoverable_agent_tool",
              "call_discoverable_agent_tool", "list_discoverable_agent_tools"):
        v = G._is_effective_write(n, b)
        ok = (v is False)
        print(f"  _is_effective_write({n[:34]:34s}, banking) = {v}  {'✓' if ok else '⚠'}")
        if not ok:
            fails += 1

    print()
    print("=== T4 a2=None 안전측 ===")
    print(f"  _is_effective_write('log_verification', None) = "
          f"{G._is_effective_write('log_verification', None)} (범용 가지 → False 기대)")
    print(f"  _is_effective_write('transfer_to_human_agents', None) = "
          f"{G._is_effective_write('transfer_to_human_agents', None)} (5/5 공통 → False 기대)")

    print()
    if fails:
        print(f"❌ FAIL {fails}건")
        sys.exit(1)
    print("✅ C241 U1' 회귀 전부 통과")


if __name__ == "__main__":
    main()
