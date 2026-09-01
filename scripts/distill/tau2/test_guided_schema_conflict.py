# -*- coding: utf-8 -*-
"""guided 문법 ↔ response_format 스키마 **동시 지정 금지** 단위검정 (2026-09-01).

왜: 라이브에서 서버가 거부했다 — "You can only use one kind of structured outputs constraint
but multiple are specified". `agent_claimprov` 가 **6회 전부 no-op** 이 되어 날조-완료 차단
가드가 꺼져 있었다. 스키마가 걸린 콜에는 문법을 붙이지 않는다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_guided_patch as GP

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


def capture():
    seen = {}

    def inner(model, messages, tools=None, tool_choice=None, call_name=None, **kw):
        seen.update(kw)
        return "ok"
    return GP._make_wrapper(inner), seen


def main():
    os.environ["T2_GUIDED"] = "1"
    # ★[[84]] 표면형 선언이 없으면 문법을 아예 안 붙인다 — 이 검정의 관심사가 아니므로 선언한다.
    os.environ.setdefault("T2_TOOL_SURFACE", "qwen3_xml")
    r = []

    w, seen = capture()
    w("m", [], tools=TOOLS, tool_choice=None, call_name="agent_response")
    got = "structured_outputs" in (seen.get("extra_body") or {})
    r.append(got)
    print(("ok  " if got else "FAIL") + " ① 스키마 없음 → 문법 부착")

    w, seen = capture()
    w("m", [], tools=TOOLS, tool_choice=None, call_name="agent_claimprov",
      response_format={"type": "json_schema", "json_schema": {"name": "t", "schema": {}}})
    got = "structured_outputs" not in (seen.get("extra_body") or {})
    r.append(got)
    print(("ok  " if got else "FAIL") + " ② 스키마 있음 → 문법 **미부착**(서버 거부 방지)")

    print("ALL PASS" if all(r) else "SOME FAILED")
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
