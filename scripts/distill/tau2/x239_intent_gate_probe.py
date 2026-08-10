# -*- coding: utf-8 -*-
r"""x239 — **결정 재료를 막은 것은 무엇인가**: 의도-도구 형식화를 궤적 위에서 재생한다
(유료 0 · 로컬 LLM · 새 엔진 0).

## 왜 (사용자 지시 2026-08-10: *"포렌식 원인 규명해서 12 pass 만들라"*)

x238 이 실패 3건의 자리를 특정했다 — 099 t2 는 **결정 블록이 0회**, 010 t2 는 상태 분해가
**0회**, 010 t0 는 **10턴 늦게**. 사이드카(우리 층이 실제로 말한 것)가 그것을 축자로 보여 준다.

코드에서 그 재료(`_limit_reduce_text`)는 한 조건 **안**에서만 만들어진다:

    if _utgt in _upending:            # _utgt = formalize_intent_tool(...) 의 답
        _add = _limit_reduce_text(...)

`formalize_intent_tool` 은 **마지막 손님 발화 6개**만 보고 *"에이전트가 지금 불러야 할 도구
하나"* 를 고른다. 손님이 하위 질문으로 새면(*"보너스 금액부터 확인해 줘"*) 이 답은 `none` 이
되고, **미실행 액션은 그대로인데 재료만 죽는다.**

⇒ 가설 H: *실패 sim 에서는 이 형식화가 결정 턴 부근에서 손님-소유 액션을 지목하지 못했다.*

## 어떻게 재는가 (짐작 금지 · [[55]]·handoff §10)

같은 함수를 **같은 프롬프트로** 궤적 위에서 재생한다(재구현하지 않는다 — 두 벌이 되면 갈린다).
각 assistant 턴마다 `messages[:i]` 를 주고 답을 받아, **통과 sim ↔ 실패 sim** 을 나란히 인쇄한다.

읽는 법 — 실패 sim 에서 결정 턴 부근이 `none` 이고 통과 sim 은 표적 도구를 지목하면 H 는 산다.
양쪽이 같으면 H 는 죽고 원인은 다른 곳이다(그때는 이 인쇄물이 그 사실의 기록이다).

실행(리모트): python x239_intent_gate_probe.py [태그]
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_resolve as RZ                                           # noqa: E402
from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402


class _Msg(object):
    def __init__(self, role, content):
        self.role = role
        self.content = content


class _UM(object):
    """`UserMessage` 자리 — 형식화 함수가 만드는 그대로 받는다."""

    def __init__(self, role=None, content=None):
        self.role = role or "user"
        self.content = content


class _LA(object):
    """`llm_agent` 자리 — 같은 프롬프트를 로컬 서버로 보낸다(모델·온도는 라이브와 같은 결정론)."""

    @staticmethod
    def generate(model=None, tools=None, messages=(), call_name=None, **kw):
        p = "".join(str(getattr(m, "content", "") or "") for m in messages)
        try:
            out = chat(p, None, 0.0, 40).get("content", "")
        except Exception as e:
            out = "ERR %s" % type(e).__name__
        return _Msg("assistant", out)


class _Agent(object):
    llm = None
    llm_args = {}


def user_owned(tid):
    _, uo = X.gold_actions(tid)
    return uo


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_asubON_20260810"
    sims = X.load(tag)
    acts = X.a2_action_tools()
    want = ("task_099", "task_010")
    rows = collections.defaultdict(list)
    for s in sorted(sims, key=lambda x: (x["task_id"], str(x.get("trial")))):
        tid = s["task_id"]
        if tid not in want:
            continue
        rew = (s.get("reward_info") or {}).get("reward")
        uo = user_owned(tid)
        msgs = [_Msg(m.get("role"), m.get("content")) for m in s["messages"]]
        # 액션 집합은 라이브와 같은 것을 준다 — A2 선언 액션 도구 ∪ gold 가 손님 몫이라 한 것
        pool = set(acts) | set(uo)
        seen = set()
        out = []
        for i, m in enumerate(s["messages"]):
            if m.get("role") != "assistant":
                continue
            # 손님 발화가 늘지 않았으면 답도 같다(같은 6개를 본다) — 호출을 아낀다
            k = tuple(x.content[:40] for x in msgs[:i] if x.role == "user")[-6:]
            if k in seen:
                continue
            seen.add(k)
            got = RZ.formalize_intent_tool(_Agent(), _LA, _UM, msgs[:i], pool)
            out.append((i, got))
        rows["%s t%s rew=%s" % (tid, s.get("trial"), rew)] = out
        print("== %-22s 손님소유=%s" % ("%s t%s rew=%s" % (tid, s.get("trial"), rew),
                                        ",".join(sorted(uo))))
        for i, got in out:
            hit = "★" if got in uo else " "
            print("   턴%-3d %s %s" % (i, hit, got or "none"))
    print("\n※ 읽는 법 — ★ 가 결정 턴 부근에 있으면 재료가 나갈 수 있었고, 없으면 조건이 닫혔다."
          "\n  실패 sim 에만 ★ 가 없으면 가설 H(의도 형식화가 관문)가 산다.")
    json.dump({k: v for k, v in rows.items()},
              open(os.environ.get("T2_X239_OUT", "x239_out.json"), "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
