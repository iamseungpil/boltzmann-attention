# -*- coding: utf-8 -*-
"""C2 선행 — **표적 행동의 미충족 조상이 있으면, 그 요건이 먼저 말한다.**

계약 정의 = `GENERAL_CONTRACTS_DESIGN_2026_08_06.md` §2-C2.

이 모듈이 계약의 **집이다**. 자료구조(의존 그래프)와 그 위의 두 질문만 갖는다:

    prereq_map(a2)          흩어진 3선언 → 하나의 그래프
    first_step(...)         뿌리까지 내려간 **지금 가능한 한 걸음**
    frontier(...)           선행이 **전부** 충족된 노드 전부 (사용자 지시 2026-08-07)

`prereq_map`·`first_step`은 `t2_dominance`에 있던 것을 **그대로 재수출**한다. 복사하지 않는다 —
같은 술어가 두 벌 있으면 갈라지고, 갈라지면 T1(사실 모순)이 이 층에서 다시 생긴다. 옮기는 것은
다음 단계이고, 지금 필요한 것은 **계약이 자기 이름의 진입점을 갖는 것**이다: arm-5가 실패한 이유가
정확히 이것이었다(계약이 `T2_RESOLVE` 안의 가지라 껍데기와 함께 죽었다).

집행 원시연산 3종(설계서 §2-C2 표)은 아직 호출부에 흩어져 있다:
  1 거부(deny)   — 에이전트가 미충족 표적을 호출        (truncation)
  2 치환(replace) — 손님 실행이라 붙을 자리가 없음        (suppression + replacement)
  3 고정(pin)    — 잔여 단계가 env가 read로 선언한 유일 도구일 때 채널을 그 호출로 고정 (insert)
여기서는 **어느 원시연산이 적용 가능한가**만 판정하고(`primitive_for`), 실제 집행은 호출부가 한다.
"""

import re

__all__ = ["prereq_map", "first_step", "frontier", "ancestors", "primitive_for",
           "graph_for", "DENY", "REPLACE", "PIN"]


def _fam(n):
    """접미사(`_1234`)를 떼어 base 이름으로 — 선언은 base로 적히고 호출은 접미사가 붙는다."""
    s = str(n or "")
    i = s.rfind("_")
    return s[:i] if i > 0 and s[i + 1:].isdigit() else s


def prereq_map(a2):
    """`{도구: [선행 도구...]}` — 흩어진 선언 셋을 **하나의 그래프**로 모은다.

    출처는 전부 이미 있는 선언이다: `gates[].satisfier_requires` · `require_tool_before` ·
    `scaffold_get_tools[].requires_reads`. 새 A2 키 0이고 도메인 어휘도 0이다 — 여기서 하는 일은
    같은 관계(X 앞에 Y)를 세 어휘로 적어 둔 것을 한 자료구조로 합치는 것뿐이다.
    """
    edges = {}

    def add(dep, reqs):
        d = _fam(dep)
        if not d:
            return
        cur = edges.setdefault(d, [])
        for r in (reqs or []):
            r = _fam(r)
            if r and r != d and r not in cur:
                cur.append(r)

    for g in ((a2 or {}).get("gates") or []):
        for s_name, reqs in (g.get("satisfier_requires") or {}).items():
            add(s_name, reqs)
    for dep, reqs in ((a2 or {}).get("require_tool_before") or {}).items():
        add(dep, reqs)
    for e in ((a2 or {}).get("scaffold_get_tools") or []):
        if isinstance(e, dict) and e.get("requires_reads"):
            add(e.get("tool") or e.get("name") or "", e["requires_reads"])
    return edges


def first_step(name, done_fam, edges, _seen=None):
    """`name`을 하려면 **지금 바로 할 수 있는 첫 걸음**. 뿌리까지 전이적으로 내려간다.

    한 단계만 보면 실행 불가한 지시가 나간다(102 실측: `log_verification`을 명령했지만 그것은
    혼자 부를 수 없어, 모델이 `verify_identity`를 다섯 번 부르다 이관으로 끝났다).

    ⚠**순환 차단**: 선언이 서로를 요구하면 그 자리에서 멈추고 그 도구 자신을 돌려준다.
    """
    n = _fam(name)
    if not n or n in done_fam:
        return None
    seen = set(_seen or ())
    if n in seen:
        return n                                  # 순환 — 여기서 끊고 자신을 명령한다
    seen.add(n)
    for p in (edges.get(n) or []):
        if p in done_fam:
            continue
        step = first_step(p, done_fam, edges, seen)
        if step:
            return step
    return n

DENY, REPLACE, PIN = "deny", "replace", "pin"


def graph_for(a2, target):
    """표적 하나에 대한 **단일** 선행 그래프.

    ★설계서는 C2의 자료구조를 *"가드된·타입된 의존 그래프"* 하나로 적었는데, 코드에는 **둘**이 있다:
      · `prereq_map` 간선 — `satisfier_requires`·`require_tool_before`·`requires_reads`
      · `gates[]` 요건   — `applies_to ∋ target` 인 게이트의 satisfier
    `requirements_for`가 호출될 때마다 둘을 즉석에서 합쳐 왔다. 그래서 그래프만 보면 조상이
    한 개(`get_all_user_accounts_by_user_id`)로 보이고, 실제 사슬(verify_identity → get_current_time
    → log_verification → get_referrals_by_user)이 **자료구조에 존재하지 않는다**.
    두 벌이 있는 한 "선행이 전부 충족됐는가"를 한 번에 물을 수 없으므로 여기서 합쳐 하나로 만든다.

    가드(`applies_when.not_in`)는 그대로 존중한다 — 그것이 사이클 차단기다(검증에 필요한 읽기가
    다시 검증을 요구하는 순환).
    """
    edges = dict((k, list(v)) for k, v in (prereq_map(a2) or {}).items())
    t = _fam(target)
    for g in ((a2 or {}).get("gates") or []):
        if target not in set(g.get("applies_to") or ()):
            continue
        if target in set((g.get("applies_when") or {}).get("not_in") or ()):
            continue
        for s in sorted((g.get("satisfiers") or {}).keys()):
            s = _fam(s)
            if s and s != t and s not in edges.setdefault(t, []):
                edges[t].append(s)
    return edges


def ancestors(target, edges, _seen=None):
    """`target`에 도달하려면 거쳐야 하는 노드 전부(자기 자신 제외·순환 안전)."""
    t = _fam(target)
    seen = set(_seen or ())
    out = []
    for p in (edges.get(t) or []):
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        for q in ancestors(p, edges, seen):
            if q not in out:
                out.append(q)
    return out


def frontier(target, done, edges):
    """지금 **바로 실행 가능한** 미충족 조상 전부 — 선행이 모두 done인 노드.

    사용자 지시(2026-08-07): *"DAG의 tree 구조에서 선순위 조건들 모두 만족하면 그다음 실행하는
    엔진을 만들라."* `first_step`이 한 걸음만 돌려주는 것과 달리, 이것은 그 층 전체를 돌려준다 —
    호출부가 하나를 명령하고 나머지를 사실로 합류시킬 수 있게(C3 합병).

    빈 리스트 = 조상이 전부 충족 = **표적을 열어도 된다**.
    """
    done_fam = {_fam(d) for d in (done or ())}
    out = []
    for n in ancestors(target, edges):
        if n in done_fam:
            continue
        if all(_fam(p) in done_fam for p in (edges.get(n) or [])):
            out.append(n)
    return out


def primitive_for(step, executor, read_tools=()):
    """이 걸음에 어느 집행 원시연산이 맞는가 — **행동이 아니라 채널이 정한다**.

    설계서가 못박은 것: *"실행자는 간선에 넣지 않는다. 간선은 행동에만 걸고, 실행자에 따라
    **집행 원시연산만** 바뀐다."* 102의 구조적 결함이 정확히 그 반대였다 — 선언은 행동을 덮는데
    집행은 채널에 붙어 있었다.
    """
    if step and step in set(read_tools or ()):
        return PIN
    if executor == "user":
        return REPLACE
    return DENY
