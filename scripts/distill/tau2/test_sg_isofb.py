#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""서브-내 ground 피드백(`T2_SG_ISOFB=1`) 오프라인 배선 테스트 (2026-07-21 §2bb·r095g).
검정: ①ungrounded 답 → 피드백 UserMessage 되먹임·라운드 계속 ②재답(grounded) 채택
③OFF=현행(첫 답 그대로 반환·거동보존) ④마지막-직전 라운드 답엔 피드백 후 마감 라운드로
⑤trace에 ground_fb 기록.
⚠️단위통과≠라이브발화([[30]]) — 배선만 본다.
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_msg = types.ModuleType("tau2.data_model.message")


class UserMessage:
    def __init__(self, role="user", content=""):
        self.role, self.content = role, content


class ToolMessage:
    def __init__(self, id=None, role="tool", requestor="assistant", content="", error=False):
        self.id, self.role, self.requestor, self.content, self.error = id, role, requestor, content, error


_msg.UserMessage, _msg.ToolMessage = UserMessage, ToolMessage
_msg.MultiToolMessage = type("MultiToolMessage", (), {})
sys.modules.setdefault("tau2", types.ModuleType("tau2"))
sys.modules.setdefault("tau2.data_model", types.ModuleType("tau2.data_model"))
sys.modules["tau2.data_model.message"] = _msg
_la = types.ModuleType("tau2.agent.llm_agent")
sys.modules.setdefault("tau2.agent", types.ModuleType("tau2.agent"))
sys.modules["tau2.agent.llm_agent"] = _la

import t2_scaffold_get as SG  # noqa: E402

A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
APY = next(x for x in A2["scaffold_get_tools"] if x["name"] == "get_correct_savings_apy")
ISO = APY["isolate"]

# KB 스텁: 값 1.35가 실재하는 문서 (도메인명 캐시에 직접 주입 — 파일시스템 불요)
_DOC = ("Certain personal checking and savings account pairings provide bonus APY. "
        "Purple Account (checking) + Gold Account (savings) APY boost: 1.35%. "
        "Your account earns an APY of 5.5% as base.")
SG._DOC_CACHE["bank_test"] = [{"title": "pairing", "content": _DOC}]

CALLS = []
SCRIPT = []


class _TC:
    def __init__(self, name, arguments):
        self.id, self.name, self.arguments, self.requestor = "c1", name, arguments, "assistant"


class _Resp:
    def __init__(self, content=None, tool_calls=None):
        self.role, self.content, self.tool_calls = "assistant", content, tool_calls


def fake_generate(model=None, tools=None, messages=None, call_name=None, **kw):
    CALLS.append({"n_msgs": len(messages or []), "tools": tools})
    return SCRIPT.pop(0)


_la.generate = fake_generate

BAD = ('{"components": [{"kind": "base", "value": 5.5, "source": "Your account earns an APY '
       'of 5.5% as base."}, {"kind": "checking", "value": 0.75, "source": "Certain personal '
       'checking and savings account pairings provide bonus APY."}]}')
GOOD = ('{"components": [{"kind": "base", "value": 5.5, "source": "Your account earns an APY '
        'of 5.5% as base."}, {"kind": "checking", "value": 1.35, "source": "Purple Account '
        '(checking) + Gold Account (savings) APY boost: 1.35%."}]}')


def mk_orch():
    tool = types.SimpleNamespace(name="KB_search_bm25")
    return types.SimpleNamespace(
        agent=types.SimpleNamespace(tools=[tool], llm="fake", llm_args={"temperature": 0.0}),
        environment=types.SimpleNamespace(domain_name="bank_test"))


def run_env(tcs):
    return [ToolMessage(id=t.id, content=_DOC) for t in tcs]


CTX = {"savings_account_type": "Gold Account", "customer_products": "Purple checking; EcoCard"}


def main():
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  ✓ " if cond else "  ✗ ") + msg)

    trace = []
    SG._isolate_trace, _orig_tr = (lambda iso, d, rec: trace.append(rec)), SG._isolate_trace

    print("① ON: ungrounded 답 → 피드백 → 재답 채택:")
    os.environ["T2_SG_ISOFB"] = "1"
    CALLS.clear(); SCRIPT.clear(); trace.clear()
    SCRIPT.extend([_Resp(tool_calls=[_TC("KB_search_bm25", {"query": "pairing"})]),
                   _Resp(content=BAD), _Resp(content=GOOD)])
    sub = SG._sub_fetch_formalize(mk_orch(), APY, ISO, dict(CTX), run_env)
    chk(len(CALLS) == 3, "3회 생성(getter→오답→재답)·라운드 계속")
    chk(isinstance(sub, dict) and any(
        c.get("value") == 1.35 for c in sub.get("components") or []),
        "재답(1.35·값-실재 인용) 채택")
    chk(CALLS[2]["n_msgs"] == CALLS[1]["n_msgs"] + 2,
        "피드백=assistant답+UserMessage 2건 되먹임")
    chk(trace and trace[-1].get("ground_fb") == 1, "trace에 ground_fb=1 기록")

    print("② OFF: 현행 거동보존(첫 답 그대로):")
    os.environ.pop("T2_SG_ISOFB", None)
    CALLS.clear(); SCRIPT.clear(); trace.clear()
    SCRIPT.extend([_Resp(tool_calls=[_TC("KB_search_bm25", {"query": "pairing"})]),
                   _Resp(content=BAD)])
    sub = SG._sub_fetch_formalize(mk_orch(), APY, ISO, dict(CTX), run_env)
    chk(len(CALLS) == 2 and isinstance(sub, dict), "OFF=피드백 없음·첫 답 반환")
    chk(any(c.get("value") == 0.75 for c in sub.get("components") or []),
        "OFF서 ungrounded 성분도 그대로(메인 관문1이 드롭=기존 경로)")

    print("③ ON: grounded 첫 답 = 피드백 없이 즉시 채택:")
    os.environ["T2_SG_ISOFB"] = "1"
    CALLS.clear(); SCRIPT.clear(); trace.clear()
    SCRIPT.extend([_Resp(tool_calls=[_TC("KB_search_bm25", {"query": "pairing"})]),
                   _Resp(content=GOOD)])
    sub = SG._sub_fetch_formalize(mk_orch(), APY, ISO, dict(CTX), run_env)
    chk(len(CALLS) == 2 and trace and trace[-1].get("ground_fb") == 0,
        "grounded 답=무피드백 채택(ground_fb=0)")

    print("④ ON: 끝까지 ungrounded면 마지막 답 반환(거동보존·메인이 재검증):")
    CALLS.clear(); SCRIPT.clear(); trace.clear()
    SCRIPT.extend([_Resp(tool_calls=[_TC("KB_search_bm25", {"query": "pairing"})])]
                  + [_Resp(content=BAD)] * (int(ISO.get("max_rounds", 5))))
    sub = SG._sub_fetch_formalize(mk_orch(), APY, ISO, dict(CTX), run_env)
    chk(isinstance(sub, dict) and any(
        c.get("value") == 0.75 for c in sub.get("components") or []),
        "소진 시 마지막 답 그대로 반환(None 아님·폴백 아님)")
    os.environ.pop("T2_SG_ISOFB", None)
    SG._isolate_trace = _orig_tr

    print("\n%s" % ("PASS — T2_SG_ISOFB 배선 정상 (라이브 발화는 별도 검증·[[30]])" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
