# -*- coding: utf-8 -*-
"""Which phase is this conversation in, so that only the message that owns it speaks.

Every contradiction found today was two of our own messages, each locally right, arriving
in the same window: `GB1` said verify first while the discovery message said run the
discovery chain now (050 burned twenty-four turns searching the knowledge base for a tool
it already had); the discovery message said do not transfer while the notice gate named the
standard transfer (032); the chain reminder was silenced exactly where it was holding the
exit closed (050/051 lost six gold actions). Patching the sentences did not hold — the pair
that clashes changes with the run.

What separates the cases is observable, and that is the whole point: a phase is derived from
declarations and call history only, never from prose. The phases, in the order they claim
ownership:

  verify     the declared auth gate is unsatisfied and its verification has been attempted
  discover   a step's tool is named by a declaration but has never been unlocked
  procedure  an entered procedure still has steps nobody has done
  decide     the procedure's prerequisites are met and its terminal decision is pending
  open       none of the above — no phase owns the turn

`verify` is measured: conditioning the action-push levers on it silences nine firings across
the failing runs and **zero** across the passing ones (the same levers fire eight times in
passing simulations, all of them before verification was ever attempted). The rest are
declared here so each lever can say which phase it belongs to instead of firing whenever its
own predicate happens to hold.
"""

__all__ = ["phase_of", "owns", "PHASES"]

PHASES = ("verify", "discover", "procedure", "decide", "open")


def _auth_gates(a2):
    return [g for g in ((a2 or {}).get("gates") or []) if g.get("kind") == "auth"]


def _called(messages, unwrap):
    out = set()
    for m in messages or []:
        for tc in (getattr(m, "tool_calls", None) or []):
            n = unwrap(tc)
            if n:
                out.add(n)
    return out


def phase_of(a2, messages, unwrap, executed=None, procedures_state=None):
    """(phase, why) — the phase this turn belongs to, and the observation that decided it.

    `unwrap` maps a tool call to the environment's name for what it executes (the dispatcher
    argument, not the dispatcher), so the caller keeps that knowledge in one place. `executed`
    is the set of tools whose call actually returned without error, when the caller has it;
    call history is used otherwise. `procedures_state` is (has_unmet, has_ready_decision) as
    the procedure module computes it — passed in rather than recomputed so this stays pure.
    """
    called = _called(messages, unwrap)
    done = set(executed or called)

    for g in _auth_gates(a2):
        sat = set((g.get("satisfiers") or {}).keys())
        if not sat or (sat & done):
            continue
        # 검증을 **시도했는가**가 경계다. 시작도 안 한 상태까지 검증 단계로 부르면, 통과 런에서
        # 정당하게 나가던 행동-유도까지 지운다(실측: 통과 8/8이 그 자리였다).
        gather = set(g.get("verify_gather_prefix") or []) | {"verify_identity"}
        if called & gather:
            return "verify", "auth gate %s unsatisfied, verification attempted" % (g.get("id") or "?")

    unmet, decision_ready = (procedures_state or (False, False))
    if decision_ready:
        return "decide", "procedure prerequisites met, terminal decision pending"
    if unmet:
        return "procedure", "entered procedure has steps nobody has done"
    return "open", "no declared phase owns this turn"


def owns(phase, lever_phases):
    """May a lever declaring these phases speak now? An empty declaration means always."""
    if not lever_phases:
        return True
    return phase in set(lever_phases)
