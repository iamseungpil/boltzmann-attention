#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`T2_DIAG_UNAMBIGUOUS` 래칫 — 스모크 없이 **초 단위로** 기능 전체를 검정한다.

## 무엇을 잠그나 (측정 정본 = `x554_diag_mispick_iso.py` · 큐 `findings_2026_08_26_night2.P1`)

진단 서브가 답하는 단위는 **이름**(`group_field`)인데 원장의 단위는 **행**이다. 한 이름이 여러
상태를 동시에 이면 *"어느 record 가 미지급인가"* 는 그 문맥에서 하나로 정해지지 않는다 — 그때
*"A separate check was run … It answers: X"* 로 단언하면 그것이 날조다([[25]]).
016 이 정확히 그 자리이고(네 이름 전부 상태 2~3종·발화 22/22 reward 0), 010·098·099 는 아니다
(이름당 상태 1종). 레버는 **그 경계를 술어로 박는 것**이고, 이 검정이 그 경계를 잠근다.

## 재료는 **실제 궤적**이다 (합성 픽스처 아님)

영속 궤적의 도구 출력에서 선언된 `row_keys` 로 행을 되살려 먹인다 — 016·010·098·099 넷 다.
한 자리만 실제 행의 **부분집합**을 쓴다(§5): 실물 넷 중 *"어떤 이름은 단일, 답한 이름은 다중"*
인 것이 없어서, 016 행에서 Platinum 전부 + Bronze 한 줄만 남겨 그 형상을 만든다.

## ⛔이 검정이 판정하지 않는 것

*"침묵시키면 016 이 통과하는가"* 는 여기서 안 잰다 — 그건 런이 잰다([[62]]·[[69]]).
여기서 잠그는 것은 ⑴술어가 넷을 옳게 가르는가 ⑵플래그 OFF 면 **바이트 불변**인가
⑶다중이면 **묻지도 않는가**(서브 호출 절약) ⑷답한 이름이 다중이면 **배달을 접는가**
⑸메모가 침묵을 기억하는가.

실행: PYTHONIOENCODING=utf-8 py -3 test_diag_unambiguous.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gate_interpreter as GI                                      # noqa: E402
import t2_ledger as LG                                             # noqa: E402
import x554_diag_mispick_iso as X                                  # noqa: E402

FAIL = []
CASES = (("task_016", "bank_t7356_grpB3_20260826", "task_016#s626729", True),
         ("task_010", "bank_t7295_a_20260815n", "task_010#s626729", False),
         ("task_098", "bank_t7361_smoke_20260826", "task_098#s626729", False),
         ("task_099", "bank_t7295_a_20260815n", "task_099#s626729", False))


def chk(c, m, extra=""):
    if not c:
        FAIL.append(m)
    print("  %s %s%s" % ("ok  " if c else "FAIL", m, ("  " + str(extra)) if extra else ""))


class _Agent(object):
    """진단 서브가 읽고 쓰는 속성 하나만 갖는 껍데기(`_t2_diag` 메모)."""


class _Stub(object):
    """`t2_subcall.sub_generate` 대역 — **호출 여부까지** 센다(묻지 않는 팔을 가리려고)."""

    def __init__(self, answer):
        self.answer, self.calls, self.prompts = answer, 0, []

    def __call__(self, agent, la, UserMessage, prompt, call_name, temperature=None):
        self.calls += 1
        self.prompts.append(prompt)
        return self.answer


def run(spec, a3rows, rows, answer, flag):
    """`diagnose_choice` 1회 — 모델 없이. 반환 (결과, 스텁, 에이전트)."""
    stub = _Stub(answer)
    real, ag = LG.SC.sub_generate, _Agent()
    prev = os.environ.get("T2_DIAG_UNAMBIGUOUS")
    LG.SC.sub_generate = stub
    if flag:
        os.environ["T2_DIAG_UNAMBIGUOUS"] = "1"
    else:
        os.environ.pop("T2_DIAG_UNAMBIGUOUS", None)
    try:
        block = LG.onto_context(rows, spec, a3rows)
        return LG.diagnose_choice(ag, object(), object(), spec, block, rows), stub, ag
    finally:
        LG.SC.sub_generate = real
        os.environ.pop("T2_DIAG_UNAMBIGUOUS", None)
        if prev is not None:
            os.environ["T2_DIAG_UNAMBIGUOUS"] = prev


def main():
    a2 = GI.load_domain_a2("banking_knowledge") or {}
    spec = next((s for s in (a2.get("ledger_metrics") or []) if s.get("diagnose_prompt")), None)
    if not spec:
        print("선언에 `diagnose_prompt` 가 없다 — 잠글 것이 없다")
        return 2
    a3rows = ((a2.get("policy_ontology") or {}).get("rows") or ())
    keys = list(spec.get("row_keys") or ())

    print("§1 술어가 네 태스크를 옳게 가르는가 (실제 궤적)")
    rows = {}
    for task, tag, simtag, want_amb in CASES:
        r = X.rows_from_traj(tag, simtag, keys)
        rows[task] = r
        if not r:
            chk(False, "%s 행 되살리기" % task, "궤적 없음 — 이 칸은 잴 수 없다")
            continue
        m = LG.status_multiplicity(r, spec)
        multi = [k for k, v in m.items() if len(v) > 1]
        blocked = bool(m) and not any(len(v) == 1 for v in m.values())
        chk(blocked == want_amb,
            "%s 이름 %d · 다중 %d ⇒ %s" % (task, len(m), len(multi),
                                           "침묵" if blocked else "배달"),
            "다중=%s" % (sorted(multi) or "없음"))

    r16, r10 = rows.get("task_016") or [], rows.get("task_010") or []
    if not (r16 and r10):
        print("\n실제 행이 없어 §2~§5 를 건너뛴다 — 없는 것을 통과로 세지 않는다([[25]])")
        return 1
    pick16 = "Platinum Rewards Card — an error has occurred throughout the process."
    pick10 = "Platinum Rewards Card — the user has too many referral processes going on."

    print("\n§2 플래그 OFF = **바이트 불변**(구판 거동)")
    for task, r, ans in (("task_016", r16, pick16), ("task_010", r10, pick10)):
        out, stub, _ = run(spec, a3rows, r, ans, flag=False)
        chk(out is not None and out[1] == " ".join(ans.split()) and stub.calls == 1,
            "%s OFF → 배달(서브 1회 호출)" % task, out and out[0])

    print("\n§3 플래그 ON · 모든 이름이 다중 ⇒ **묻지도 않는다**")
    out, stub, ag = run(spec, a3rows, r16, pick16, flag=True)
    chk(out is None, "016 ON → 배달 없음", out)
    chk(stub.calls == 0, "016 ON → 서브 호출 **0회**(프롬프트 비용도 안 쓴다)", stub.calls)
    chk(getattr(ag, "_t2_diag", None) == "", "016 ON → 메모가 침묵을 기억(재질의 없음)",
        repr(getattr(ag, "_t2_diag", None)))

    print("\n§4 플래그 ON · 이름당 상태 1종 ⇒ **종전대로 배달**")
    for task in ("task_010", "task_098", "task_099"):
        r = rows.get(task) or []
        if not r:
            chk(False, "%s ON → 잴 수 없음" % task, "궤적 없음")
            continue
        nm = sorted(LG.status_multiplicity(r, spec))[0]
        out, stub, _ = run(spec, a3rows, r, "%s — …" % nm, flag=True)
        chk(out is not None and out[0] == nm and stub.calls == 1,
            "%s ON → 배달(불변)" % task, out and out[0])

    print("\n§5 플래그 ON · 어떤 이름은 단일인데 **답한 이름이 다중** ⇒ 배달을 접는다")
    gf = spec.get("group_field")
    keep, seen = [], 0
    for r in r16:                       # 실제 016 행의 부분집합 — Platinum 전부 + Bronze 한 줄
        g = str(r.get(gf) or "")
        if "Platinum" in g:
            keep.append(r)
        elif "Bronze" in g and seen == 0:
            keep.append(r)
            seen = 1
    m = LG.status_multiplicity(keep, spec)
    chk(any(len(v) == 1 for v in m.values()) and any(len(v) > 1 for v in m.values()),
        "부분집합이 혼합 형상인가", {k: sorted(v) for k, v in sorted(m.items())})
    amb_name = next(k for k, v in m.items() if len(v) > 1)
    out, stub, _ = run(spec, a3rows, keep, "%s — …" % amb_name, flag=True)
    chk(out is None, "다중 이름을 답하면 → 배달 없음", out)
    chk(stub.calls == 1, "이 팔은 **묻고 나서** 접는다(사전 차단 아님)", stub.calls)
    uniq_name = next(k for k, v in m.items() if len(v) == 1)
    out2, stub2, _ = run(spec, a3rows, keep, "%s — …" % uniq_name, flag=True)
    chk(out2 is not None and out2[0] == uniq_name,
        "같은 행에서 **단일 이름**을 답하면 → 배달", out2 and out2[0])

    print("\n%s  (%d 실패)" % ("FAIL" if FAIL else "ALL OK", len(FAIL)))
    for m2 in FAIL:
        print("  - %s" % m2)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
