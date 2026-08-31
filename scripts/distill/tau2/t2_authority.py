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

# ★선언 없음 → 집행 없음 (2026-08-29 · x607 실측 · 특허 B §13 한정).
#
# 이 파일의 규칙은 **선언을 채운 도메인에서만** 뜻이 있다. 한 칸도 안 채운 도메인에서 켜면
# 규칙이 하나 더해지는 게 아니라 **중재가 통째로 사라진다** — 아무 레버도 남을 못 죽이므로
# 전 레버가 동시에 말한다. 그건 이 모듈이 막으려던 것(C13)의 정반대 고장이고, 위 독스트링이
# 이미 *"켜는 것은 선언을 채운 뒤다"* 라고 적어 둔 조건이 지켜지지 않은 상태다.
#
# 실측(x607·retail 12태스크·같은 sha·같은 모델):
#   A_control(T2_SUPPRESS_AUTH=1·retail 선언 0)  pass 1/12 · `[T2_SUPPRESS_AUTH] refused` 216회
#   C_undeclared(그 플래그만 0)                   pass 4/12 · 같은 마커 0회
# retail·airline 의 `suppression_authority` 는 둘 다 `None`, banking 만 2칸(+_note).
#
# 반증 조건: 선언 0인 도메인에서 이 게이트를 되돌려도 pass 가 안 떨어지면 이 귀속은 거짓이다.
#
# 도메인 리터럴 0 — 보는 것은 *그 도메인이 이 표를 채웠는가* 하나뿐이다. banking 은 채웠으므로
# 거동 불변이고, 새 도메인은 표를 채우는 순간 집행이 켜진다.
_DECL_WARNED = set()


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


def _declares_any(a2):
    """이 도메인이 억제 자격 표를 **한 칸이라도** 유효하게 채웠는가(`_` 주석 키는 셈에서 뺀다)."""
    sa = (a2 or {}).get("suppression_authority")
    if not isinstance(sa, dict):
        return False
    return any(warrant_of(a2, k) is not None for k in sa if not str(k).startswith("_"))


def may_suppress(a2, key):
    """이 레버가 다른 레버를 침묵시켜도 되는가.

    `T2_SUPPRESS_AUTH` 미설정이면 **종전 거동**(전부 허용)을 유지한다 — 켜는 것은 선언을 채운 뒤다.
    플래그가 켜져 있어도 **그 도메인이 표를 한 칸도 안 채웠으면 마찬가지로 종전 거동**이다
    (위 `_DECL_WARNED` 주석 — 집행이 아니라 중재 제거가 되기 때문).
    켜져 있으면 근거를 댄 레버만 억제할 수 있고, 못 댄 레버는 조용히 통과시킨다(표면화는 그대로).
    """
    if os.environ.get("T2_SUPPRESS_AUTH") != "1":
        return True
    if not _declares_any(a2):
        _k = id(a2) if a2 is not None else 0
        if _k not in _DECL_WARNED:
            _DECL_WARNED.add(_k)
            try:
                import sys as _s
                print("[T2_SUPPRESS_AUTH] inert — 이 도메인은 suppression_authority 를 "
                      "한 칸도 선언하지 않았다(선언 없음 → 집행 없음)",
                      file=_s.stderr, flush=True)
            except Exception:
                pass
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
    assert may_suppress(A2, "unlisted") is False         # 표를 채운 도메인 안에서 미선언 = 억제 불가
    # ★선언 0인 도메인은 **종전 거동**이다(2026-08-29). 옛 판정은 `False` 였고 그 탓에
    #   retail 에서 중재가 통째로 사라졌다(x607: 1/12 ↔ 플래그만 끈 팔 4/12).
    assert may_suppress({}, "x") is True                 # 표 자체가 없는 도메인
    assert may_suppress({"suppression_authority": {"_note": "메모뿐"}}, "x") is True
    assert may_suppress({"suppression_authority": {"a": {"note": "근거 아님"}}}, "x") is True
    assert _declares_any(A2) is True
    assert warrant_of(A2, "policy_lever")["kind"] == "policy"
    os.environ.pop("T2_SUPPRESS_AUTH", None)
    print("t2_authority self-check OK")
