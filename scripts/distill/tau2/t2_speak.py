# -*- coding: utf-8 -*-
"""May we recommend this tool right now, or has the running procedure forbidden it?

`task_022` received both of these in the same turn, three times over:

    [VALUE-ACQUIRE] … give get_card_last_4_digits to the customer NOW …
    [PROCEDURE]     The policy forbids 'get_card_last_4_digits' in this procedure - verbatim:
                    "Do not collect sensitive card details; the tool uses the identifiers provided by the user."

Both facts were already in the engine's hands: the dispute procedure was running (its trigger had
executed) and its declaration forbids that tool, quoting the policy sentence that forbids it. The
prohibition was only ever consulted when the model *called* something (`t2_procedure.decide`), never
when we *named* something. So one of our messages spent three cycles pushing the tool the other one
was blocking.

This is the check for the naming side, and it is deliberately one rule, not a framework:

    a lever that chose its own target does not name a tool the running procedure prohibits.

**speak-time 계약** (호출 시점과 다르다·설계서 §5.1). `t2_procedure.prohibited` treats a procedure as
running when the *name under test* is one of its triggers — harmless when the caller is a call the
model just made, wrong here, because the target is a tool **we** are about to recommend and mentioning
it must not switch a procedure on. So activity is decided from `executed` alone (`active_procedures`)
and the target is used for the prohibition lookup only.

**push 레버에만 건다.** The lever that reports a prohibition names the prohibited tool too; gating it
would delete the sentence enforcing the rule and keep the one breaking it. The instrument made exactly
that mistake before it was corrected (`x104` §4.2), which is why the caller passes its own target
rather than having text parsed out of it.

Off unless `T2_SPEAK_PROHIBIT=1`, and returns False on any failure — a silencer that fails open.
"""

import os
import sys

__all__ = ["prohibits_target", "silence_reason"]


def _procs(a2):
    return (a2 or {}).get("procedures") or []


def silence_reason(a2, executed, target):
    """(procedure_id, quote) when a running procedure forbids `target`, else (None, None).

    Pure — no environment lookup, no flag check — so tests can call it directly.
    """
    if not target:
        return None, None
    try:
        import t2_procedure as PR
        for p in PR.active_procedures(_procs(a2), set(executed or ())):
            spec = (p.get("prohibits") or {}).get(target)
            # 인용 없는 금지는 무시한다 — 순서 선언과 같은 규약(t2_procedure.prohibited)
            if spec and spec.get("_quote"):
                return p.get("id"), spec.get("_quote")
    except Exception:
        return None, None
    return None, None


def prohibits_target(a2, executed, target, lever=None, messages=None):
    """True면 이 레버는 이번 턴에 말하지 않는다. 플래그 OFF면 항상 False(거동 변화 0)."""
    if os.environ.get("T2_SPEAK_PROHIBIT") != "1":
        return False
    pid, quote = silence_reason(a2, executed, target)
    if pid is None:
        return False
    print("[T2_SPEAK_PROHIBIT] silent lever=%s target=%s procedure=%s"
          % (lever, target, pid), file=sys.stderr, flush=True)
    try:                                   # 침묵도 계측 대상이다(x104 부정통제·[[24]] 탐지자 왕복)
        import t2_fbsidecar as _fbsc
        _fbsc.record("speak-prohibit", "[SILENCED] %s → %s (%s)" % (lever, target, pid),
                     messages, channel="speak_gate", kind_note="silence")
    except Exception:
        pass
    return True
