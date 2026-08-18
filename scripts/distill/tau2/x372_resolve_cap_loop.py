#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""x372 — 창 순환의 기전 확정: `_resolve_cap_ok` 의 *진행* 판정이 무엇으로 리셋되는가.

## 왜 (C527⒢ · C529⒣)

t7307 은 두 sim 이 turn 99 에서 7,800초 표류해 5/24 에서 중단됐고, t7308 은 `max_steps=120`
상한으로 완주했지만 **순환 자체는 그대로**였다(resign 창 143/177 · 동일지문 접힘 172/171).
상한(`_resolve_cap_ok`·cap=3)은 **살아 있고 실제로 물린다**(`stop=resolve_cap` 194/236).
그런데 resolve deny 가 sim 당 **11~23회** 나온다 ⇒ 카운터가 **여러 번 리셋**됐다는 뜻이다.

리셋 경로는 둘이다:
  ⓐ `done - prev` — **새로 실행된 도구 이름**이 하나라도 늘면 리셋. **로그를 찍지 않는다**(조용함).
  ⓑ 새로 **회수된** unlockable 이름 — `[T2_RESOLVE_CAP] 리셋` 마커를 찍는다.

t7308 전수에서 **ⓑ 는 0회**다 ⇒ 남는 것은 ⓐ뿐이다. 이 프로브는 그것을 **궤적으로 확증**한다.

★구조적 근거(소스 직독): `_exact_tool_name` 이 `call_*` 래퍼를 **내부 레지스트리 이름으로 푼다**
(`t2_gate_patch.py:2305-2310`). 그리고 `_executed_tool_names` 는 그 이름들의 **집합**을 돌려준다.
이 도메인의 discoverable 도구는 **에이전트-측 44 · 손님-측 4** 이므로, 모델이 *새 도구를 하나씩
부르기만 해도* 집합이 계속 커지고 상한이 계속 되돌아간다 — **요건은 한 번도 충족되지 않은 채로**.

## 계기 규율

  ⚠엔진과 **같은 술어**를 쓴다(사본 금지·[[67]]) — 성공 판정은 `error` 플래그 + `Error:` 접두 +
    A2 `failure_markers`, 이름은 `call_*` 언랩. 엔진 코드에서 그대로 옮기지 않고 **읽어서 맞춘다**.
  ⚠**상관은 인과가 아니다** — 이 프로브가 보이는 것은 *"리셋 가능 횟수 ≥ 실제 deny 수"* 라는
    **상한 정합**이지 개별 리셋의 인과가 아니다. 인과 주장은 하지 않는다([[08]]).
  ⚠**손해를 따로 센다**: 순환이 실제로 무엇을 태웠는가(턴·초).

사용: PYTHONPATH=<tau2>/src python x372_resolve_cap_loop.py <ctl_tag> <treat_tag>
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F  # noqa: E402

SUFFIX = ".results.json.gz"
TAGS = sys.argv[1:3] or ["bank_t7308_ctl_20260818c", "bank_t7308_treat_20260818c"]

# A2 의 실패 표지(엔진과 같은 출처) — 없으면 종전 거동(플래그 + "Error:")
try:
    import json
    _a2p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "a2", "banking_knowledge.settings.json")
    FAIL_MARKS = tuple(json.load(open(_a2p, encoding="utf-8")).get("failure_markers") or ())
except Exception:
    FAIL_MARKS = ()


def exact_name(tc):
    """엔진 `_exact_tool_name` 과 같은 규칙 — `call_*` 래퍼는 **안쪽 이름**으로 푼다."""
    nm = F.nameof(tc)
    if nm.startswith("call_"):
        a = F.argsof(tc)
        inner = (a.get("agent_tool_name") or a.get("user_tool_name")
                 or a.get("discoverable_tool_name") or "")
        if inner:
            return str(inner)
    return nm


def executed_names_growth(sim):
    """성공 실행된 도구 **이름 집합**이 커진 시점의 수 = 경로ⓐ 리셋 **가능** 횟수 상한."""
    msgs = sim.get("messages") or []
    pending, seen, growth = {}, set(), 0
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            pending[tc.get("id")] = exact_name(tc)
        if str(m.get("role")) != "tool":
            continue
        nm = pending.get(m.get("id") or m.get("tool_call_id"))
        txt = str(m.get("content") or "").lstrip()
        failed = (m.get("error") or txt.startswith("Error:")
                  or any(txt.startswith(k) for k in FAIL_MARKS))
        if nm and not failed and nm not in seen:
            seen.add(nm)
            growth += 1
    return growth, len(seen)


def main():
    print("=" * 78)
    print("x372 — 창 순환 기전: 상한 리셋 경로 확정")
    print("=" * 78)
    print("\n[A2] failure_markers %d종" % len(FAIL_MARKS))

    for tag in TAGS:
        arm = "ctl" if "_ctl_" in tag else "treat"
        print("\n" + "-" * 78)
        print("[%s] %s" % (arm, tag))
        print("-" * 78)

        denies = {k: len(v) for k, v in
                  F.by_sim(tag, r"\[T2_RESOLVE\] action-required").items()}
        resets_b = {k: len(v) for k, v in
                    F.by_sim(tag, r"\[T2_RESOLVE_CAP\] 리셋").items()}
        capped = {k: len(v) for k, v in F.by_sim(tag, r"stop=resolve_cap").items()}

        print("  %-22s %6s %7s %7s %7s %8s %7s" %
              ("sim", "deny", "리셋ⓑ", "capped", "이름↑", "고유이름", "steps"))
        tot = collections.Counter()
        viol = []
        for s in F.scored(tag, SUFFIX):
            k = F.simtag(s)
            g, u = executed_names_growth(s)
            d = denies.get(k, 0)
            print("  %-22s %6d %7d %7d %7d %8d %7d"
                  % (k, d, resets_b.get(k, 0), capped.get(k, 0), g, u,
                     len(s.get("messages") or [])))
            tot["deny"] += d; tot["resetb"] += resets_b.get(k, 0)
            tot["capped"] += capped.get(k, 0); tot["growth"] += g; tot["uniq"] += u
            # 상한 정합: deny 는 (리셋 횟수 + 1) × cap 을 넘을 수 없다
            if d > (g + resets_b.get(k, 0) + 1) * 3:
                viol.append((k, d, g))
        print("  %-22s %6d %7d %7d %7d %8d" %
              ("합계", tot["deny"], tot["resetb"], tot["capped"], tot["growth"], tot["uniq"]))

        print("\n  판독:")
        print("   · 리셋ⓑ(마커 있는 경로) 합 = %d" % tot["resetb"])
        print("   · deny 합 %d 이 cap(3)×sim(%d)=%d 을 넘는 초과분 = %d"
              % (tot["deny"], len(F.scored(tag, SUFFIX)), 3 * len(F.scored(tag, SUFFIX)),
                 tot["deny"] - 3 * len(F.scored(tag, SUFFIX))))
        print("   · 새 이름 등장 횟수 합 = %d  (경로ⓐ 리셋 **가능** 상한)" % tot["growth"])
        if viol:
            print("   ⚠상한 정합 위반(설명 안 되는 deny) %d 건: %s" % (len(viol), viol[:5]))
        else:
            print("   ✅상한 정합 — 모든 sim 에서 deny ≤ (이름↑ + 리셋ⓑ + 1) × 3")

    print("\n" + "=" * 78)
    print("⚠이 프로브는 **상한 정합**을 보일 뿐 개별 리셋의 인과를 증명하지 않는다.")
    print("=" * 78)


if __name__ == "__main__":
    main()
