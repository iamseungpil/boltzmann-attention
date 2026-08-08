# -*- coding: utf-8 -*-
"""회귀 검정 (C324): 상한·문턱 대조는 **피연산자 가용성**에만 달린다.

무엇을 막는 검정인가 — 이 산수는 원래 `if _reqs or _bad:`(아직 밀어낼 요건이 남아 있는가)
안에 살고 있었다. 요건이 다 풀린 뒤의 표적 턴, 즉 **손님이 실행하기 직전의 바로 그 자리**에서는
그 조건이 거짓이라 문장이 나가지 못했다. 한 sim이 정확히 그래서 산수 **0회**로 끝났고,
손님은 이미 소진된 그룹을 골라 실행했다. 오프라인 재현으로 문장 자체는 만들어짐이 확증됐다.

⇒ 여기서는 `_reqs`를 **아예 주지 않고** 호출한다. 그래도 문장이 나와야 한다.
   (엔진 인자가 아니라 함수 경계로 못박는다 — 다시 분기 안으로 들어가면 이 검정이 죽는다.)

오프라인 전용: tau2·서버·LLM 불요. A2/A3는 repo의 정본 파일을 그대로 읽는다.
실행: py -3 test_limit_reduce_operand_gated.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as G                                    # noqa: E402
from gate_interpreter import load_domain_a2                   # noqa: E402

DOMAIN = "banking_knowledge"
FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


class _Agent(object):
    """`_t2_ledger_ops`만 지닌 최소 대역 — 함수가 그 밖의 상태를 보지 않음도 함께 검정한다."""

    def __init__(self, ops):
        self._t2_ledger_ops = ops


def main():
    a2 = load_domain_a2(DOMAIN)
    if not a2:
        print("A2 없음 — skip")
        return 0
    specs = a2.get("ledger_metrics") or []
    tally_spec = next((s for s in specs if s.get("exhausted_text")), None)
    chk(tally_spec is not None, "A2에 상한 대조 선언(exhausted_text)이 있다")
    if tally_spec is None:
        return 1

    # 상한 축의 주어 두 개를 **A3에서** 가져온다(테스트가 도메인 어휘를 짓지 않는다).
    import t2_factdag as FD
    rows = (a2.get("policy_ontology") or {}).get("rows") or ()
    lims = FD._a3_map(rows, {"axis": "annual_referral_limit"})
    chk(bool(lims), "A3에서 상한 축이 조회된다 (n=%d)" % len(lims))
    if not lims:
        return 1
    subj, (cap, _q) = sorted(lims.items())[0]

    gf = tally_spec.get("group_field")
    ops = {tally_spec.get("trigger_tool"): {
        "spec": tally_spec,
        "tally": {subj: int(cap)},          # 정확히 소진된 그룹 하나
        "days": None}}

    # ── 본 검정: `_reqs` 없이도 문장이 나온다 ────────────────────────────────
    txt = G._limit_reduce_text(_Agent(ops), a2, [])
    chk(bool(txt), "요건(_reqs) 없이도 대조 문장이 생성된다  ← C324가 막는 결손")
    chk(subj in txt, "소진된 그룹의 이름이 문장에 실린다")
    print("     문장 앞부분: %s" % " ".join(txt.split())[:120])

    # ── 음성 통제: 피연산자가 없으면 침묵한다([[57]]) ────────────────────────
    chk(G._limit_reduce_text(_Agent({}), a2, []) == "",
        "피연산자가 없으면 침묵한다 (음성 통제)")
    chk(G._limit_reduce_text(_Agent(None), a2, []) == "",
        "원장 미형성 상태에서도 예외 없이 침묵한다")

    # ── 여유가 있는 그룹만 있으면 말하지 않는다(과잉 발화 방지·Δspurious) ──
    ops2 = {tally_spec.get("trigger_tool"): {
        "spec": tally_spec, "tally": {subj: 0}, "days": None}}
    chk(G._limit_reduce_text(_Agent(ops2), a2, []) == "",
        "소진된 그룹이 없으면 침묵한다 (Δspurious 억제)")

    chk(gf is not None, "선언이 group_field를 지닌다(엔진이 필드명을 짓지 않는다)")

    # ── C327: 상한 행이 없는 원장 그룹은 **이름을 말한다**(조용히 빠지지 않는다) ──
    #   실물 사고: 원장 표기가 A3에 다른 표기로만 있어 7건짜리 그룹이 판정 자체를 못 받았고,
    #   모델 쪽에서 침묵은 *검사 통과*와 구별되지 않았다. 이름이 같은 것인지는 모델 몫이라
    #   엔진은 집합 뺄셈만 한다([[22]]).
    ghost = subj + " __not_in_a3__"          # A3에 없는 표기(테스트가 도메인 어휘를 짓지 않는다)
    ops3 = {tally_spec.get("trigger_tool"): {
        "spec": tally_spec, "tally": {subj: int(cap), ghost: 7}, "days": None}}
    txt3 = G._limit_reduce_text(_Agent(ops3), a2, [])
    chk(ghost in txt3, "상한 행이 없는 그룹이 이름과 함께 표면화된다  ← C327")
    chk("NOT checked" in txt3, "그 그룹은 '판정 못 했다'로 말한다(아는 척하지 않는다)")
    chk(G._limit_reduce_text(_Agent(ops), a2, []).find("NOT checked") < 0,
        "전부 A3에 있으면 미매칭 문장은 안 나온다 (Δspurious 억제)")
    import t2_ledger as _LG
    chk(_LG.unmatched_text({ghost: 7}, lims, {}) == "",
        "A2가 문구를 선언하지 않으면 침묵한다(구판 거동 보존)")

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
