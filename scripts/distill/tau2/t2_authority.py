# -*- coding: utf-8 -*-
"""May a lever silence another lever? Only if it can name why.

The expensive lesson is C13. A repeat-suppression lever was added because repeating the same
message looked useless, and it silenced the repetition that was in fact producing the gold
actions — 050 took four rounds and 051 took six. Nothing in the codebase asked that lever what
entitled it to remove another one's effect, because there was nowhere to ask.

Ranking would not have caught it. C13 and its victim were never in conflict about a fact; C13
removed the *conditions under which the other could fire*. So the arbitration rule that settles
opposing directives — the one whose predicate rests on stronger evidence commands — says nothing
here. What decides it is authority, not evidence: a lever that suppresses must hold a warrant.

A warrant is one of two things, and both are checkable:

  · **policy** — a verbatim sentence from a policy document that licenses the silence. This is
    the pattern `procedures[].enforce` already follows: it blocks only when `_quote_order` carries
    the policy's own "MUST be followed in the exact order listed", and surfaces otherwise.
  · **measurement** — a pre-registered count over the full sweep showing the silence costs no
    passing simulation. This is what `T2_PHASE_OWNER` has and what C13 never had.

Numeric priority is deliberately not an option. The literature has rejected it repeatedly and its
own implementers say so: Davis and Buchanan call it "opaque and likely to cause bugs" because it
reduces incommensurate factors to one number with no record of how it was reached; the CLIPS
manual warns against leaning on salience; KAoS admits in a footnote that it relies on numeric
priorities for want of anything better. Soar has the shape we want instead — `require` and
`prohibit` dominate `best` and `better` by **type**, evaluated earlier, not by score.

Declaring a warrant costs the domain nothing: warrants live in A2 next to the thing they justify,
and a lever with no entry is not ranked last — it simply may not suppress. It can still surface.
"""

import os

__all__ = ["may_suppress", "warrant_of"]


def warrant_of(a2, key):
    """`key` 레버의 억제 근거 선언(없으면 None). A2 `suppression_authority[key]`."""
    spec = ((a2 or {}).get("suppression_authority") or {}).get(key)
    if not isinstance(spec, dict):
        return None
    kind = spec.get("kind")
    if kind == "policy" and str(spec.get("quote") or "").strip():
        return spec
    if kind == "measurement" and str(spec.get("measure") or "").strip():
        return spec
    return None                       # 종류가 없거나 근거가 비면 선언이 아니다


def may_suppress(a2, key):
    """이 레버가 다른 레버를 침묵시켜도 되는가.

    `T2_SUPPRESS_AUTH` 미설정이면 **종전 거동**(전부 허용)을 유지한다 — 켜는 것은 선언을 채운 뒤다.
    켜져 있으면 근거를 댄 레버만 억제할 수 있고, 못 댄 레버는 조용히 통과시킨다(표면화는 그대로).
    """
    if os.environ.get("T2_SUPPRESS_AUTH") != "1":
        return True
    ok = warrant_of(a2, key) is not None
    if not ok:
        try:
            import sys as _s
            print("[T2_SUPPRESS_AUTH] refused lever=%s — 억제 근거 미선언" % key,
                  file=_s.stderr, flush=True)
        except Exception:
            pass
    return ok


if __name__ == "__main__":                                     # 자기검정 (오프라인)
    A2 = {"suppression_authority": {
        "policy_lever": {"kind": "policy", "quote": "MUST be followed in the exact order listed"},
        "measured_lever": {"kind": "measurement", "measure": "x104 §C: over-block 0 / 194 sim"},
        "empty_quote": {"kind": "policy", "quote": "   "},
        "no_kind": {"note": "그냥 메모"},
    }}
    os.environ.pop("T2_SUPPRESS_AUTH", None)
    assert may_suppress(A2, "anything") is True          # 미설정 = 종전 거동
    os.environ["T2_SUPPRESS_AUTH"] = "1"
    assert may_suppress(A2, "policy_lever") is True
    assert may_suppress(A2, "measured_lever") is True
    assert may_suppress(A2, "empty_quote") is False      # 축자가 비면 근거 아님
    assert may_suppress(A2, "no_kind") is False
    assert may_suppress(A2, "unlisted") is False         # 미선언 = 억제 불가
    assert may_suppress({}, "x") is False                # 미선언 도메인
    assert warrant_of(A2, "policy_lever")["kind"] == "policy"
    os.environ.pop("T2_SUPPRESS_AUTH", None)
    print("t2_authority self-check OK")
