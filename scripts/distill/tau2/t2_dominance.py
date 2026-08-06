# -*- coding: utf-8 -*-
"""Which of our own messages may command, when two of them point opposite ways.

The failure this exists for is measured, not imagined. In task_102 the declaration already
covered the action: `gates[GB1].applies_to` lists `submit_referral`, and `require_tool_before`
names a read that must precede it. Neither ever spoke. Both levers attach to the moment the
*agent* calls the tool, and `submit_referral` is run by the customer, so there was no place for
them to attach — while the action-push lever, which attaches to the agent's *utterance*, fired
every time. The customer then submitted five referrals where the task wanted one.

So the defect is not a wording clash and not a channel confusion. It is that **declaration scope
and enforcement point are different objects that were allowed to diverge**: the gate's scope is
an action, its enforcement site is a call, and for a customer-run action no call of ours exists.

The rule implemented here (user instruction, 2026-08-06): **the directive whose predicate rests
on stronger evidence commands; the other loses its imperative.** A gate's predicate is the
execution ledger — did `log_verification` actually run — which is the strongest evidence we have.
An action-push's target comes from a formalize sub-call over prose, which is the weakest. So the
gate commands. Nothing here needs a new priority list: the ranking follows from where each
predicate gets its facts.

Two boundaries keep this from becoming a silencer:

  · **Scope, not blanket.** A gate dominates only pushes whose target its own `applies_to`
    declares. Outside that set a higher-evidence lever takes nothing away. Without this cut the
    order is a pure lexicographic one, and the lower rank is consulted only on exact ties —
    which is to say never (Tercan & Prabhu, ECAI 2024). Thresholding is what buys the lower
    rank a place to speak (Gabor, Kalmar & Szepesvari, ICML 1998).

  · **Replace, never delete.** The dominated push does not vanish; it is rewritten into the
    gate's requirement with the same target named. A commitment that cannot be discharged
    converts into an obligation to announce that fact rather than into silence (Cohen &
    Levesque, *Teamwork*, 1991). Deleting it reproduces the 012 failure, where our own deny
    removed the trigger of the message that was doing the work.

The predicate is A2 declarations plus call history. No domain vocabulary, no gold, no tool-name
literals — `applies_to`, `satisfiers` and `applies_when` are already declared for every gate, so
this costs the domain nothing new.
"""

__all__ = ["dominating_gate", "requirement_text", "DEFAULT_FEEDBACK"]

DEFAULT_FEEDBACK = (
    "Error: [ORDER] '{target}' cannot be carried out yet - not by you, and not by the customer "
    "acting on your instruction. This has to hold first: {requirement}. Do that now with the "
    "real tool calls. Once it holds, then tell the customer to run '{target}'. Telling them to "
    "run it before that is the same as doing it early yourself."
)


def _satisfier_names(gate):
    return set((gate.get("satisfiers") or {}).keys())


def _exempt(gate, target):
    """`applies_when.not_in`이 이 표적을 면제하는가.

    면제는 게이트 자신이 선언한다(banking GB1: 이관·사고 도구는 검증 불요). 면제 목록은
    동시에 **사이클 차단기**다 — 검증에 필요한 읽기가 다시 검증을 요구하면 아무것도 못 한다.
    """
    aw = gate.get("applies_when") or {}
    return target in set(aw.get("not_in") or [])


def dominating_gate(a2, messages, target, executed=None, unwrap=None):
    """`target`을 덮는 **미충족** 게이트 하나(없으면 None).

    `executed`는 호출자가 가진 실행-성공 집합(없으면 호출 이력으로 대신한다). `unwrap`은 도구
    호출을 환경 이름으로 바꾸는 함수 — 디스패처를 벗기는 지식은 호출자 쪽 한 곳에 둔다.
    """
    if not target:
        return None
    done = set(executed or ())
    if not done and messages:
        for m in messages:
            for tc in (getattr(m, "tool_calls", None) or []):
                n = unwrap(tc) if unwrap else getattr(tc, "name", None)
                if n:
                    done.add(n)
    for g in ((a2 or {}).get("gates") or []):
        if target not in set(g.get("applies_to") or ()):
            continue
        if _exempt(g, target):
            continue
        sat = _satisfier_names(g)
        if not sat:
            continue                      # 충족을 판정할 수 없는 게이트는 지배하지 않는다
        if sat & done:
            continue                      # 이미 충족 = 지배 없음
        return g
    return None


def requirement_text(a2, gate, target):
    """지배당한 push를 대체할 문구. 문구는 A2, 엔진은 이름만 채운다([[05]] Q2)."""
    tpl = str((((a2 or {}).get("arbitration") or {}).get("dominated_push_feedback"))
              or DEFAULT_FEEDBACK)
    req = str(gate.get("predicate") or gate.get("id") or "the declared precondition")
    return tpl.replace("{target}", str(target)).replace("{requirement}", req)


if __name__ == "__main__":                                     # 자기검정 (오프라인)
    class _C:
        def __init__(self, name):
            self.name = name
            self.arguments = {}

    class _M:
        def __init__(self, names):
            self.tool_calls = [_C(n) for n in names]

    A2 = {"gates": [{"id": "G", "predicate": "identity verified",
                     "satisfiers": {"log_verification": []},
                     "applies_to": ["submit_referral", "call_x"],
                     "applies_when": {"arg": "agent_tool_name", "not_in": ["call_x"]}}]}

    assert dominating_gate(A2, [_M([])], "submit_referral") is not None      # 미충족 → 지배
    assert dominating_gate(A2, [_M(["log_verification"])], "submit_referral") is None
    assert dominating_gate(A2, [_M([])], "call_x") is None                   # 면제
    assert dominating_gate(A2, [_M([])], "other_tool") is None               # 범위 밖
    assert dominating_gate({}, [_M([])], "submit_referral") is None          # 미선언 도메인 = no-op
    assert dominating_gate({"gates": [{"id": "G2", "applies_to": ["t"]}]},
                           [_M([])], "t") is None                            # satisfier 없음
    g = dominating_gate(A2, [_M([])], "submit_referral")
    txt = requirement_text(A2, g, "submit_referral")
    assert "submit_referral" in txt and "identity verified" in txt
    assert dominating_gate(A2, [], "submit_referral", executed=set()) is not None
    assert dominating_gate(A2, [], "submit_referral",
                           executed={"log_verification"}) is None
    print("t2_dominance self-check OK")
