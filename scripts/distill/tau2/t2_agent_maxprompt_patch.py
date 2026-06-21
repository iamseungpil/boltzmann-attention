#!/usr/bin/env python
"""최대-강도·위치반복 프롬프트 주입 (FETCHFIRST_PROMPT_FAILURE_MECHANISM·사용자 2026-06-22).

질문: 7B에서 *최대로 강하게* + *앞/중간/끝 반복*해도 schema-example-copy prior가 안 죽나
(= 프롬프트 한계 확정). t2_agent_rules_patch는 system(앞)만 → 이건 messages 조립 시점에
규칙을 begin/middle/end 메시지 content에 *append*(포맷-안전·새 메시지 추가 아님).

활성: T2_MAXPROMPT=<file>  [T2_MAXPROMPT_POS=begin,middle,end]  → t2_run_gated가 apply().
도메인-일반 텍스트(도구명 0)·harness 패치(scaffold 엔진 아님).
"""
import os

_RULE = None
_POS = set()


def apply(path=None):
    global _RULE, _POS
    path = path or os.environ.get("T2_MAXPROMPT")
    if not path:
        raise SystemExit("[maxprompt] T2_MAXPROMPT 미지정")
    with open(path, encoding="utf-8") as f:
        _RULE = f.read().strip()
    _POS = set((os.environ.get("T2_MAXPROMPT_POS") or "begin,middle,end").split(","))

    import tau2.agent.llm_agent as la
    _orig = la.generate

    B = "\n\n=== MANDATORY RULE — READ FIRST ===\n" + _RULE
    M = "\n\n=== MID-CONTEXT REMINDER (do not skip) ===\n" + _RULE
    E = "\n\n=== CRITICAL — APPLY NOW, BEFORE THIS TOOL CALL ===\n" + _RULE

    def _app(msg, txt):
        try:
            if isinstance(getattr(msg, "content", None), str):
                return msg.model_copy(update={"content": msg.content + txt})
        except Exception:
            pass
        return msg

    def _wrapped(*a, messages=None, **k):
        msgs = list(messages or [])
        if msgs and "begin" in _POS:
            msgs[0] = _app(msgs[0], B)
        if len(msgs) >= 3 and "middle" in _POS:
            mid = len(msgs) // 2
            msgs[mid] = _app(msgs[mid], M)
        if msgs and "end" in _POS:
            msgs[-1] = _app(msgs[-1], E)
        return _orig(*a, messages=msgs, **k)

    la.generate = _wrapped
    print(f"[maxprompt] ON · {path} ({len(_RULE)} chars) · pos={sorted(_POS)}")


if __name__ == "__main__":
    import sys
    apply(sys.argv[1] if len(sys.argv) > 1 else None)
