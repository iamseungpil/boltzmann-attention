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

__all__ = ["dominating_gate", "requirement_text", "DEFAULT_FEEDBACK",
           "requirements_for", "merged_text", "DEFAULT_MERGED"]

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


DEFAULT_MERGED = (
    "Error: [ORDER] '{target}' cannot be carried out yet - not by you, and not by the customer "
    "acting on your instruction.\n"
    "Do this now, with a real tool call: {first}\n"
    "Still outstanding after that (do not do them in this reply): {rest}\n"
    "When all of them hold, then tell the customer to run '{target}'. Telling them to run it "
    "earlier is the same as doing it early yourself."
)


def _executed(messages, executed=None, unwrap=None):
    done = set(executed or ())
    if not done and messages:
        for m in messages:
            for tc in (getattr(m, "tool_calls", None) or []):
                n = unwrap(tc) if unwrap else getattr(tc, "name", None)
                if n:
                    done.add(n)
    return done


def _fam(n):
    """접미사(`_1234`)를 떼어 base 이름으로 — 선언은 base로 적히고 호출은 접미사가 붙는다."""
    s = str(n or "")
    i = s.rfind("_")
    return s[:i] if i > 0 and s[i + 1:].isdigit() else s


def requirements_for(a2, messages, target, executed=None, unwrap=None):
    """`target`을 덮는 **미충족 요건 전부**. 첫 하나가 아니라 전부다.

    라이브가 가르쳐 준 것: 치환 24회가 **전부 GB1**이고 뒤에 선 게이트는 **0회**였다. push가 발화하는
    창은 좁은데(검증 전·캡 소진 전) 그 창에서 늘 같은 게이트가 이기니, 뒤에 선 요건은 선언돼 있어도
    존재하지 않는 것과 같다. 순수 우선순위는 하위를 굶긴다.

    그래서 **하나를 고르지 않는다.** 같은 표적을 덮는 요건을 모두 모아 한 번에 말한다 —
    명령은 하나, 사실은 합집합. 이렇게 하면 뒤에 선 요건이 굶지 않고, 동시에 다른 절차를 요구하는
    선언(예: 같은 행동 앞의 다른 선행 read)이 **밀려나지 않는다**.

    출처는 세 선언뿐이고 전부 이미 있다: `gates[]` · `require_tool_before` ·
    `scaffold_get_tools[].requires_reads`. 새 A2 키 0.
    """
    if not target:
        return []
    done = _executed(messages, executed, unwrap)
    done_fam = {_fam(n) for n in done}
    out, seen = [], set()

    for g in ((a2 or {}).get("gates") or []):
        if target not in set(g.get("applies_to") or ()):
            continue
        if _exempt(g, target):
            continue
        sat = _satisfier_names(g)
        if not sat or (sat & done):
            continue
        gid = g.get("id") or "gate"
        if gid in seen:
            continue
        seen.add(gid)
        out.append({"id": gid,
                    "predicate": str(g.get("predicate") or gid),
                    "satisfiers": sorted(sat)})

    def _reads(dep, reads):
        if _fam(dep) != _fam(target):
            return
        miss = sorted({r for r in (reads or []) if _fam(r) not in done_fam})
        if not miss:
            return
        key = "reads:" + ",".join(miss)
        if key in seen:
            return
        seen.add(key)
        out.append({"id": key,
                    "predicate": "the prior read(s) this action requires have been done",
                    "satisfiers": miss})

    for dep, reads in ((a2 or {}).get("require_tool_before") or {}).items():
        _reads(dep, reads)
    for e in ((a2 or {}).get("scaffold_get_tools") or []):
        if isinstance(e, dict) and e.get("requires_reads"):
            _reads(e.get("tool") or e.get("name") or "", e["requires_reads"])
    return out


def merged_text(a2, reqs, target):
    """요건 여럿을 **한 문장**으로. 문구는 A2, 엔진은 이름만 채운다([[05]] Q2)."""
    if not reqs:
        return ""
    if len(reqs) == 1:
        tpl = str((((a2 or {}).get("arbitration") or {}).get("dominated_push_feedback"))
                  or DEFAULT_FEEDBACK)
        r = reqs[0]
        req = r["predicate"]
        # 요건을 **무엇으로 충족하는지**가 빠지면 read 요건은 이행 불가한 지시가 된다.
        # 술어가 이미 그 도구를 이름으로 말하고 있으면 덧붙이지 않는다(중복 방지).
        sats = [s for s in (r.get("satisfiers") or []) if s not in req]
        if sats:
            req = "%s (do it with: %s)" % (req, ", ".join(sats))
        return tpl.replace("{target}", str(target)).replace("{requirement}", req)
    # ★합병 ≠ 나열 (2026-08-07·궤적 실측). 초판은 미충족 요건을 **전부 명령형 목록**으로 냈다.
    #   task_101 turn 4: 네 요건을 한 번에 받은 모델이 **첫 항목만** 집어 신원 확인으로 가고
    #   나머지(원장 조회 포함)를 흘렸다 — 그 뒤 열 턴이 신원 확인 왕복이었다. 경합은 없앴는데
    #   **우선순위 없는 나열**이 되어 omission을 만들었다(IFScale이 보고한 지배적 오류와 같은 형태).
    #   [[56]]에 "명령은 하나, 사실은 합집합"이라 써 두고 구현은 전부 명령이었다.
    #   ⇒ **지금 할 하나만 명령**하고 나머지는 평서로 남긴다. 다음 턴에 다시 발화해 그 다음으로 간다.
    tpl = str((((a2 or {}).get("arbitration") or {}).get("merged_requirement_feedback"))
              or DEFAULT_MERGED)
    head = reqs[0]
    first = "%s (do it with: %s)" % (head["predicate"], ", ".join(head["satisfiers"]))
    rest = "; ".join(r["predicate"] for r in reqs[1:]) or "nothing else"
    return (tpl.replace("{target}", str(target))
               .replace("{first}", first).replace("{rest}", rest))


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

    # ── C3 합병 ────────────────────────────────────────────────────────────
    A2M = {"gates": [
        {"id": "G1", "predicate": "identity verified", "satisfiers": {"log_verification": []},
         "applies_to": ["submit_referral"]},
        {"id": "G3", "predicate": "referral record checked",
         "satisfiers": {"get_referrals_by_user": []}, "applies_to": ["submit_referral"]}],
        "require_tool_before": {"submit_referral": ["get_all_user_accounts_by_user_id"]}}

    # 굶주림 재현 방지: 뒤에 선 요건도 **반드시** 나온다.
    rs = requirements_for(A2M, [_M([])], "submit_referral")
    ids = [r["id"] for r in rs]
    assert ids[:2] == ["G1", "G3"], ids
    assert any(r["id"].startswith("reads:") for r in rs), ids
    txt = merged_text(A2M, rs, "submit_referral")
    # ★명령은 하나뿐이고, 나머지는 사실로만 남는다(101 turn 4가 나열의 대가를 보였다).
    assert txt.count("Do this now") == 1, txt
    assert "identity verified (do it with: log_verification)" in txt, txt
    head, tail = txt.split("Still outstanding", 1)
    assert "referral record checked" not in head, head        # 2·3번은 명령부에 없다
    assert "referral record checked" in tail and "prior read" in tail, tail

    # 충족된 것은 빠지고, 남은 것만 말한다.
    rs2 = requirements_for(A2M, [_M(["log_verification", "get_referrals_by_user"])],
                           "submit_referral")
    assert [r["id"] for r in rs2] == ["reads:get_all_user_accounts_by_user_id"], rs2
    assert "get_all_user_accounts_by_user_id" in merged_text(A2M, rs2, "submit_referral")

    # 접미사 붙은 실제 호출도 base 선언과 맞는다.
    rs3 = requirements_for(A2M, [_M(["log_verification", "get_referrals_by_user",
                                     "get_all_user_accounts_by_user_id_3847"])],
                           "submit_referral")
    assert rs3 == [], rs3
    assert merged_text(A2M, rs3, "submit_referral") == ""

    # 하나뿐이면 단수 문구로 떨어진다(기존 거동 유지).
    rs4 = requirements_for(A2, [_M([])], "submit_referral")
    assert len(rs4) == 1 and "identity verified" in merged_text(A2, rs4, "submit_referral")
    assert requirements_for({}, [_M([])], "submit_referral") == []
    print("t2_dominance self-check OK")
