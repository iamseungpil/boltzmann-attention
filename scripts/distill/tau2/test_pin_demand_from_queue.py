# -*- coding: utf-8 -*-
"""회귀 검정 (C330): 핀의 수요 신호는 **우리 요건 큐**에서 온다.

무엇을 막는 검정인가 — 라이브 2 sim에서 P1 핀은 표적을 정확히 해소할 수 있는 상태였는데
(피의존 4·read 접두·미실행·레지스트리 유일 해소) **한 번도 시도되지 않았다**. 수요 신호 셋이
이 계열에서 원리적으로 못 뜨기 때문이다:
  ⒜ 의존 도구가 **손님 실행**이라 assistant 호출 집합에 영영 없다
  ⒝ 찾는 태그가 현 어휘에 없고, 게다가 우리 통지는 **비커밋**이라 `messages`에 안 나타난다
  ⒞ tau2 실제 문구 `No records found in '...'` 가 `"not found"` 에 안 걸린다
그동안 요건 큐는 매 턴 그 read를 **이름으로** 요구하고 있었다 ⇒ 원천을 직접 쓴다.

여기서는 ⒜⒝⒞를 **일부러 전부 죽은 상태로** 두고 ⒟만으로 핀이 서는지 본다.
오프라인 전용(서버·LLM 불요). 실행: py -3 test_pin_demand_from_queue.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_pin_read as PR                                      # noqa: E402
import t2_dominance as DOM                                    # noqa: E402

FAILED = []


def chk(cond, label):
    print(("  OK   " if cond else "  FAIL ") + label)
    if not cond:
        FAILED.append(label)


READ = "get_all_user_accounts_by_user_id"
FULL = READ + "_3847"
DEP = "submit_referral"
# 피의존 2 이상이어야 후보가 된다(§1.7) — 실제 A2도 이 read를 4개 선언이 요구한다.
A2 = {
    "eplan": {"unlock_tool": "unlock_discoverable_agent_tool"},
    "require_tool_before": {DEP: [READ], "open_bank_account": [READ]},
}


class _Orch(object):
    """`tools`도 레지스트리도 없는 최소 대역. 해소는 monkeypatch 로 고정한다."""

    def __init__(self, demanded=None):
        if demanded is not None:
            self._t2_demanded_reads = set(demanded)


def _msg(role, content=None, calls=()):
    tcs = [types.SimpleNamespace(name=n, arguments={}) for n in calls]
    return types.SimpleNamespace(role=role, content=content, tool_calls=tcs or None)


def main():
    os.environ["T2_PIN_READ"] = "1"
    PR._resolve = lambda orch, base: FULL if base == READ else None   # 레지스트리 해소 고정

    # 라이브에서 관측된 그대로의 궤적: 손님이 의존 도구를 실행했고, env 는 'No records found',
    # 우리 통지는 (비커밋이라) 아예 없다.
    msgs = [
        _msg("assistant", calls=["KB_search_dense"]),
        _msg("tool", "No records found in 'referrals'."),
        _msg("user", calls=[DEP]),                       # ⒜ 는 assistant 만 세므로 신호 아님
        _msg("assistant", "Next, I need to check your existing accounts."),   # 호출 0
    ]
    am = _msg("assistant", "…")

    decls = PR._declarations(A2)
    chk(PR._refcount(decls).get(READ, 0) >= 2, "이 read 는 피의존 2 이상 (후보 자격)")

    # ── ① 구판 신호 셋은 이 궤적에서 전부 죽어 있다 ─────────────────────────
    chk(PR._demand(msgs, decls) == set(),
        "⒜⒝⒞ 만으로는 수요 0  ← 라이브에서 핀이 침묵한 이유")
    chk(PR.pin_for(_Orch(), am, A2, msgs) is None,
        "따라서 구판 조건에서는 고정하지 않는다(회귀 기준선)")

    # ── ② 요건 큐가 그 read 를 요구하면 ⒟ 로 선다 ──────────────────────────
    got = PR.pin_for(_Orch([READ]), am, A2, msgs)
    chk(got == ("unlock_discoverable_agent_tool", "agent_tool_name", FULL),
        "요건 큐가 요구하면 고정된다 (본 값: %s)  ← C330" % (got,))

    # ── ③ 이미 읽었으면 요구가 남아 있어도 고정하지 않는다 ─────────────────
    msgs2 = msgs + [_msg("assistant", calls=[FULL])]
    chk(PR.pin_for(_Orch([READ]), am, A2, msgs2) is None,
        "이미 실행된 read 는 요구가 남아 있어도 고정 안 함")

    # ── ④ 요구 집합에 없는 read 는 여전히 무발화(⒟가 만능 통과권이 아니다) ─
    chk(PR.pin_for(_Orch(["some_other_read"]), am, A2, msgs) is None,
        "요구되지 않은 read 는 고정하지 않는다")

    # ── ⑤ 요건 id 접두는 정본 상수에서 온다(소비자가 리터럴을 짓지 않는다) ─
    chk(DOM.READS_PREFIX == "reads:", "READS_PREFIX 가 정본에 있다")
    rs = DOM.requirements_for(A2, [], DEP)
    ids = [r.get("id") for r in (rs or [])]
    sat = [s for r in (rs or []) if str(r.get("id") or "").startswith(DOM.READS_PREFIX)
           for s in (r.get("satisfiers") or [])]
    print("     요건 id: %s / reads-satisfiers: %s" % (ids, sat))
    chk((not rs) or READ in sat,
        "reads 요건의 satisfiers 가 read 이름을 그대로 진다(gate_patch 가 읽는 자리)")

    # ── ⑥ **머리일 때만** 무장한다 — 게이트가 앞서 있으면 read 는 머리가 아니다 ──
    #    이 조건이 없으면 신원확인이 아직 머리인 첫 발화에서 계좌 read 가 고정돼
    #    **우리 게이트(검증 우선)를 우리가 위반**시키고 1회뿐인 핀도 거기서 탄다
    #    (오프라인 재현으로 확인된 실패 형태). 라이브 순서는 GB1 → GB3 → reads 다.
    from gate_interpreter import load_domain_a2
    real = load_domain_a2("banking_knowledge")
    if real:
        rs2 = DOM.requirements_for(real, [], DEP)
        ids2 = [str(r.get("id")) for r in (rs2 or [])]
        print("     실제 A2 요건 순서: %s" % ids2)
        first_reads = next((i for i, x in enumerate(ids2)
                            if x.startswith(DOM.READS_PREFIX)), None)
        chk(bool(ids2) and first_reads not in (0, None) if len(ids2) > 1 else True,
            "게이트가 남아 있는 동안 reads 요건은 머리가 아니다 (조기 무장 방지)")

    print("\n%s  (%d 실패)" % ("PASS" if not FAILED else "FAIL", len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
