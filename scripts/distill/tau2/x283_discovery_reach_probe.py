# -*- coding: utf-8 -*-
r"""x283 — 검색-회피 구간에 발견체인 푸시가 닿으면 열리는가 (p런 070/071 write 미도달).

사전등록: `PROBE_X283_X285_DESIGN_2026_08_13.md` §x283 (2026-08-13 · 실행 전 기입).

## 근거 (P_RUN_FORENSIC_2026_08_13.md)

070/071 실패 8 sim 에서 [ORDER] 기계(ARBITRATE/FRONTIER/CALL_FORM) 발화 **0** — 기전은
`resolve_action_operator` 의 `_agent_ending` 게이트다: KB_search 표류 턴은 "진행 중"으로 세어
탈출구가 안 열린다(a#4: 46msg·검색 6회·푸시 1회). 문구 자체는 이미 검증돼 있다(x273 E_STEP2).
이 프로브는 **그 문구가 그 자리(표류 중턴)에 닿기만 하면 체인이 열리는지**를 가른다 —
열리면 라이브 수리는 발화 조건(불진행 카운터·닫힌 술어)뿐이고 문면 변경 0 이다.

## 팔 (n=8 · 계기 = 다음 도구 호출 · 문턱 ≥6/8 사전 고정)

  A_LIVE   궤적 + 사이드카 축자 그대로(푸시 없음 재현)   ← 결손 재현(낮아야 유효)
  B_DISC   + DISCOVERY_REQUIRED_FB 축자(이름 단정 없음)   ← 트리거-절벽 가설 팔
  C_STEP2  + DISCOVERY_STEP2_FB 축자(레지스트리 실명)     ← 이름-공급 가설 팔
  D_NEG    + 무내용 재촉 문구(길이 통제)                  ← 부정통제([[57]])

통과(HIT) = 다음 호출이 unlock/call 디스패처이고 인자에 open_bank_account 가족명이 있는 것.

실행(리모트·GPU 유휴 확인 후): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  python x283_discovery_reach_probe.py [N]
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import t2_resolve as R                                            # noqa: E402
from x266_decide_ask_axis import a2 as _a2                        # noqa: E402
import t2_fbsidecar as FB                                         # noqa: E402

TAG = "bank_judge6p_a_20260813p"
TASKS = ("task_070", "task_071")
FAMILY = "open_bank_account"
LOCKED = "open_bank_account_4821"
NEG = ("Note: please continue helping the customer with their request in this "
       "conversation - review what they asked for and respond accordingly.")


def our_lines(sim):
    """사이드카 축자(턴별) — A2 재현용. 궤적은 한 글자도 안 고친다."""

    class _M(object):
        def __init__(s, r, c):
            s.role, s.content = r, c

    keyed = FB._sim_key([_M(m.get("role"), m.get("content")) for m in sim["messages"]])
    ours = collections.defaultdict(list)
    p = "/home/woori/scratch/logs/fb_%s.jsonl" % TAG
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8", errors="replace"):
            o = json.loads(ln)
            if o.get("sim") == keyed and (o.get("text") or "").strip():
                ours[o.get("turn")].append(" ".join(o["text"].split()))
    return ours


def pick_cut(sim):
    """마지막 순수-텍스트(도구 0) 어시스턴트 턴 중 open-체인 이전 지점 — 표류의 한복판.

    a#4 는 설계서의 [24] 가 이 규칙으로 재현된다(수기 인덱스 박제 대신 규칙으로 —
    같은 규칙이 다른 실패 sim 에도 그대로 적용되게).
    """
    opened = None
    for i, m in enumerate(sim["messages"]):
        for tc in (m.get("tool_calls") or []):
            nm = (tc.get("function") or {}).get("name") or tc.get("name") or ""
            args = json.dumps(tc.get("arguments") or {}, ensure_ascii=False)
            if FAMILY in args or FAMILY in nm:
                opened = i
                break
        if opened is not None:
            break
    last = None
    for i, m in enumerate(sim["messages"][: (opened or len(sim["messages"]))]):
        if m.get("role") == "assistant" and not (m.get("tool_calls") or []) \
                and str(m.get("content") or "").strip():
            last = i
    return (last + 1) if last is not None else None


def build(sim, cut, inject):
    ours = our_lines(sim)
    out = []
    for i, m in enumerate(sim["messages"][:cut]):
        r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
        tcs = [(tc.get("function") or {}).get("name") or tc.get("name")
               for tc in (m.get("tool_calls") or [])]
        if tcs:
            out.append("[%s calls] %s" % (r, ", ".join(tcs)))
        if c:
            out.append("[%s] %s" % (r, c[:700]))
        for t in ours.get(i, ()):
            out.append("[system] %s" % t[:1100])
    if inject:
        out.append("[system] %s" % inject)
    return "\n".join(out)


def hit(resp):
    for tc in (resp.get("tool_calls") or []):
        nm = (tc.get("function") or {}).get("name") or tc.get("name") or ""
        raw = (tc.get("function") or {}).get("arguments") or tc.get("arguments") or ""
        s = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        if nm in ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool") \
                and FAMILY in s:
            return "HIT"
        if FAMILY in nm:
            return "HIT"
    return "MISS"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    a2 = _a2()
    ep = (a2.get("eplan") or {})
    unlock, call, lst = ep.get("unlock_tool"), ep.get("dispatch_tool"), ep.get("list_tool")
    disc = R._discoverable_dispatchers(a2)
    getter = disc.get(FAMILY, "the search tool")
    b_disc = R.DISCOVERY_REQUIRED_FB.format(target=FAMILY, getter=getter,
                                            unlock=unlock, call=call, list=lst)
    c_step2 = R.DISCOVERY_STEP2_FB.format(name=LOCKED, unlock=unlock)
    sims = [s for s in X.load(TAG)
            if s["task_id"] in TASKS and (s.get("reward_info") or {}).get("reward") != 1]
    print("실패 궤적 %d개 · n=%d · URL=%s\n"
          % (len(sims), n, os.environ.get("T2_PROBE_URL", "localhost:8140")))
    grand = collections.Counter()
    cells = 0
    for sim in sims:
        cut = pick_cut(sim)
        if cut is None:
            print("  (순수-텍스트 어시스턴트 턴 없음 — 건너뜀 %s trial=%s)"
                  % (sim["task_id"], sim.get("trial")))
            continue
        tools = U.tools_of(sim)
        print("== %s trial %s · cut=%d · 도구 %d개"
              % (sim["task_id"], sim.get("trial"), cut, len(tools)))
        for label, inject in (("A_LIVE", None), ("B_DISC", b_disc),
                              ("C_STEP2", c_step2), ("D_NEG", NEG)):
            body = build(sim, cut, inject)
            c = collections.Counter()
            for i in range(n):
                try:
                    r = chat(body, tools, 0.0 if i == 0 else 0.7, 200)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                c[hit(r)] += 1
            grand[label] += c["HIT"]
            cells += 1
            print("  %-8s 문맥 %6d자 · HIT %d/%d   %s"
                  % (label, len(body), c["HIT"], n, c.most_common(3)))
    print("\n합계(%d셀): " % cells
          + " · ".join("%s %d" % (k, v) for k, v in sorted(grand.items())))
    print("※ 판정(사전 고정): B≥6/8 ∧ A≤2/8 ∧ D≤2/8 → 트리거 절벽(발화 조건만 수리·문면 0)."
          "\n  C만 ≥6/8 → 이름 공급이 관건. A≥6/8 → 재현 실패 = 시점 재선정 1회 후 폐기.")


if __name__ == "__main__":
    main()
