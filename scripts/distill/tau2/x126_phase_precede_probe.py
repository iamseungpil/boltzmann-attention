# -*- coding: utf-8 -*-
"""x126 — `(no DAG req)`가 **표적 미확정**인가 **DAG 침묵**인가 (무료 격리·[[09]]).

20260807c 실측: `[T2_PHASE_OWNER] action-push silent — phase=verify (no DAG req)` **4건**.
치환 경로는 진입했는데 `requirements_for`가 빈 손이었다. 두 갈래가 가능하다:

  ⓐ **표적 미확정** — `_tgt_pre`(formalize 산출)가 `action_tools` 안에 없고 후보가 2개 이상이라
     `_t17 = None`이 되어 **그래프에 묻지도 않았다**. 그렇다면 고칠 곳은 표적 선택이다.
  ⓑ **DAG 침묵** — 물어봤는데 그래프가 그 표적에 미충족 조상을 0개로 준다. 그렇다면 고칠 곳은
     선언(그래프)이지 코드가 아니다.

라이브 로그로는 못 가른다(진단을 나중에야 갈랐다). 그래서 **궤적을 그대로 재생해 오프라인으로 묻는다** —
유료 런 없이 판정된다. 이 프로브는 아무것도 실행하지 않고 선언과 히스토리만 읽는다.

⚠`_tgt_pre`는 LLM 서브콜 산출이라 재생 불가다. 그래서 **더 강한 질문**을 대신 던진다:
  *"그 시점에 `action_tools` 중 **어느 하나라도** 미충족 조상을 갖는가?"*
  · 하나라도 있으면 ⓐ다(그래프는 할 말이 있었는데 표적을 못 골라 못 물었다).
  · 전부 0이면 ⓑ다(표적을 뭘로 골랐든 그래프는 조용했다).

사용법: python x126_phase_precede_probe.py <results.json> [domain]
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class _Obj(object):
    """results.json의 dict 메시지를 엔진 헬퍼가 기대하는 속성 접근으로 감싼다."""

    def __init__(self, d):
        self._d = d if isinstance(d, dict) else {}
        tcs = self._d.get("tool_calls") or []
        self.tool_calls = [_Call(t) for t in tcs] if tcs else None

    def __getattr__(self, k):
        return self._d.get(k)


class _Call(object):
    def __init__(self, d):
        self._d = d if isinstance(d, dict) else {}

    def __getattr__(self, k):
        return self._d.get(k)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    domain = sys.argv[2] if len(sys.argv) > 2 else "banking_knowledge"
    if not path or not os.path.exists(path):
        print("usage: x126_phase_precede_probe.py <results.json> [domain]")
        return 2

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import t2_gate_patch as GP
    import t2_phase as PH
    import t2_dominance as DOM

    a2 = GP._domain_a2(domain)
    if a2 is None:
        print("A2 없음: %s" % domain)
        return 2
    acts = sorted({t for t in ((a2 or {}).get("action_tools") or [])})
    print("A2 action_tools %d개: %s" % (len(acts), acts[:8]))
    print("A2 auth gates: %s" % [g.get("id") for g in (a2.get("gates") or [])
                                 if g.get("kind") == "auth"])

    d = json.load(open(path, encoding="utf-8"))
    verify_turns = 0
    with_req = 0
    detail = []

    for s in (d.get("simulations") or []):
        tid = s.get("task_id")
        msgs = [_Obj(m) for m in (s.get("messages") or [])]
        for i, m in enumerate(msgs):
            if getattr(m, "role", None) != "assistant":
                continue
            hist = msgs[:i + 1]
            try:
                ph = PH.phase_of(a2, hist, GP._exact_tool_name,
                                 executed=GP._executed_tool_names(hist, a2))[0]
            except Exception as e:
                print("  phase_of 실패 %s@%d: %r" % (tid, i, e))
                continue
            if ph != "verify":
                continue
            verify_turns += 1
            hits = []
            for act in acts:
                try:
                    rq = DOM.requirements_for(a2, hist, act,
                                              executed=GP._executed_tool_names(hist, a2),
                                              unwrap=GP._exact_tool_name)
                except Exception as e:
                    rq = []
                    hits.append((act, "ERR:%r" % (e,)))
                    continue
                if rq:
                    hits.append((act, [r.get("id") for r in rq]))
            if hits:
                with_req += 1
            detail.append((tid, i, len(hits), hits[:3]))

    print("\n=== phase=verify 턴 %d개 ===" % verify_turns)
    for tid, i, n, hits in detail:
        print("  %-9s turn=%-3d 미충족 조상을 가진 action_tool %d개  %s" % (tid, i, n, hits))

    print("\n=== 판정 ===")
    if verify_turns == 0:
        print("  이 궤적엔 phase=verify 턴이 없다 — 다른 런으로 물어야 한다.")
    elif with_req > 0:
        print("  ⓐ **표적 미확정**: verify 턴 %d개 중 %d개에서 그래프가 **할 말이 있었다**."
              % (verify_turns, with_req))
        print("     ⇒ 고칠 곳은 코드(표적 선택). `_t17`이 None이라 묻지도 않은 것이다.")
        print("     ⇒ 처방: 표적 하나를 고르지 말고 **action_tools 전체의 요건 합집합**을 낸다")
        print("        ([[56]] C3 merge와 같은 형태 — 명령 하나·사실 합집합).")
    else:
        print("  ⓑ **DAG 침묵**: verify 턴 %d개 전부에서 어느 action_tool도 미충족 조상이 없다."
              % verify_turns)
        print("     ⇒ 고칠 곳은 선언(그래프)이지 코드가 아니다. 표적을 뭘로 골랐어도 빈 손이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
