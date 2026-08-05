# -*- coding: utf-8 -*-
"""Does the speak-time prohibition silence the message it was measured to silence, and nothing else?

Four things have to hold before this is worth running live, and three of them are the ways a
silencer fails quietly:

  ① 표적          the running procedure forbids the target → silent
  ② 비활성        the same target, with the procedure not running → speaks (over-block 0)
  ③ 플래그 OFF    flag unset → speaks, byte for byte the old behaviour
  ④ 부정통제      a target the declaration does NOT forbid → speaks
                  (a gate that silences everything also passes ①, so ② and ④ are the test)

Values come from the real A2 — a fixture would prove the fixture, not the declaration.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gate_interpreter as GI          # noqa: E402
import t2_speak as SPK                 # noqa: E402

A2 = GI.load_domain_a2("banking_knowledge") or {}
TRIGGER = "get_reward_discrepancies"          # cash_back_dispute.enter_when.tool_any
TARGET = "get_card_last_4_digits"             # cash_back_dispute.prohibits
OTHER = "get_credit_card_transactions_by_user"

fails = []


def check(name, got, want):
    ok = got == want
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL",
                            "" if ok else " — got %r want %r" % (got, want)))
    if not ok:
        fails.append(name)


print("== A2 선언 확인(값이 실재해야 이 검정이 의미가 있다) ==")
procs = [p for p in (A2.get("procedures") or []) if p.get("id") == "cash_back_dispute"]
check("cash_back_dispute 절차 실재", bool(procs), True)
spec = ((procs[0].get("prohibits") or {}).get(TARGET) or {}) if procs else {}
check("prohibits[%s]._quote 실재" % TARGET, bool(spec.get("_quote")), True)

print("\n== 순수 판정(silence_reason·플래그 무관) ==")
pid, quote = SPK.silence_reason(A2, {TRIGGER}, TARGET)
check("① 절차 활성 + 금지된 표적 → 사유 반환", pid, "cash_back_dispute")
check("① 사유에 정책 축자가 실린다", bool(quote), True)
check("② 절차 비활성(트리거 미실행) → 사유 없음", SPK.silence_reason(A2, set(), TARGET)[0], None)
check("④ 절차 활성 + 금지되지 않은 표적 → 사유 없음",
      SPK.silence_reason(A2, {TRIGGER}, OTHER)[0], None)
check("표적 None → 사유 없음", SPK.silence_reason(A2, {TRIGGER}, None)[0], None)

print("\n== 플래그 게이트(prohibits_target) ==")
os.environ.pop("T2_SPEAK_PROHIBIT", None)
check("③ 플래그 OFF → 침묵하지 않는다(거동 변화 0)",
      SPK.prohibits_target(A2, {TRIGGER}, TARGET, lever="VALUE-ACQUIRE"), False)
os.environ["T2_SPEAK_PROHIBIT"] = "1"
check("① 플래그 ON + 표적 → 침묵", SPK.prohibits_target(A2, {TRIGGER}, TARGET,
                                                lever="VALUE-ACQUIRE"), True)
check("② 플래그 ON + 절차 비활성 → 발화", SPK.prohibits_target(A2, set(), TARGET,
                                                     lever="VALUE-ACQUIRE"), False)
check("④ 플래그 ON + 비금지 표적 → 발화", SPK.prohibits_target(A2, {TRIGGER}, OTHER,
                                                     lever="VALUE-ACQUIRE"), False)

print("\n== 배선(레버가 실제로 이 판정을 통과한다) ==")
import t2_gate_patch as G                                             # noqa: E402
import inspect                                                        # noqa: E402
src = inspect.getsource(G._value_acquire_fb)
check("VALUE-ACQUIRE가 t2_speak를 부른다", "t2_speak" in src, True)
check("표적을 문자열 파싱이 아니라 변수로 넘긴다", "prohibits_target(a2, executed, acq" in src, True)
sig = inspect.signature(G._value_acquire_fb)
check("호출부가 넘길 수 있게 a2/executed 인자가 있다",
      ("a2" in sig.parameters and "executed" in sig.parameters), True)
call_src = inspect.getsource(G)
check("호출부가 실제로 넘긴다(死인자 방지)",
      "_value_acquire_fb(am, state.messages, va_specs, a2=a2" in call_src, True)
check("플래그가 정본 런처에 등재돼 있다",
      "T2_SPEAK_PROHIBIT" in open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read(), True)

print("\n결과: %s" % ("ALL PASS" if not fails else "FAIL %d — %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
