# -*- coding: utf-8 -*-
"""발사 전 **표면형 검산** — 선언한 문법이 이 서버에서 실제로 파싱되는가.

사용: x704_surface_preflight.py <base_url> <model_id>     (환경변수 T2_TOOL_SURFACE 를 읽는다)
반환: 0 = 통과 · 1 = 불일치(발사 중단) · 2 = 검사 불가(서버 무응답 등·호출부가 판단)

★왜 (2026-08-31·[[84]]): `T2_GUIDED` 의 문법 표면형이 hermes 인 채로 서버만 `qwen3_coder` 로
  바뀌어, 두 달치 런이 **네이티브 도구 파싱 0%** 로 돌았다. 선언은 조용히 어긋난다 —
  그래서 선언을 믿지 않고 **한 번 쏴 본다**. 요청 2개·max_tokens 128 이라 비용은 무시할 수준이다.

판정(둘 다 만족해야 통과):
  ① 문법 없이 도구를 주면 서버가 **네이티브로 파싱**한다 → 서버·모델이 정상이다(음성 대조).
  ② 선언한 표면형 문법을 걸어도 **네이티브로 파싱**된다 → 문법이 파서와 짝이다.
  ②가 깨지면 그 문법은 모델을 자기 서버가 못 읽는 형식에 가두는 것이다.
⚠판정은 `tool_calls` **개수 하나**로 한다(도메인 판단 0). 어떤 도구를 고르는지는 안 본다.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOOLS = [{"type": "function", "function": {
    "name": "preflight_probe_tool", "description": "Look up a record by id.",
    "parameters": {"type": "object",
                   "properties": {"record_id": {"type": "string", "description": "the id"}},
                   "required": ["record_id"]}}}]
MSGS = [{"role": "system", "content": "You are an assistant with tools. Use them."},
        {"role": "user", "content": "Look up record r_001 with the tool."}]


def _fire(base, model, grammar=None, timeout=120):
    body = {"model": model, "messages": MSGS, "tools": TOOLS,
            "max_tokens": 128, "temperature": 0.0}
    if grammar:
        body["structured_outputs"] = {"grammar": grammar}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    m = ((d.get("choices") or [{}])[0] or {}).get("message") or {}
    return len(m.get("tool_calls") or []), str(m.get("content") or "")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8143/v1"
    model = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("T2_EXPECT_MODEL", "")
    surface = (os.environ.get("T2_TOOL_SURFACE") or "").strip()
    if not surface:
        print("[x704] ⛔T2_TOOL_SURFACE 미선언 — 프로필이 없거나 안 실렸다")
        return 1
    if os.environ.get("T2_GUIDED") != "1":
        print("[x704] T2_GUIDED=0 이라 문법을 안 건다 — 표면형 검산 생략")
        return 0

    import t2_guided_patch as G
    grammar = G.grammar_for_tools(TOOLS)
    if not grammar:
        print("[x704] ⛔문법 생성 실패(surface=%s)" % surface)
        return 1

    try:
        n_plain, _ = _fire(base, model)                    # ① 음성 대조
        n_guided, body = _fire(base, model, grammar)       # ② 선언한 표면형
    except Exception as e:
        print("[x704] 검사 불가(서버 응답 없음): %r" % (e,))
        return 2

    print("[x704] surface=%s · 문법없음 tool_calls=%d · 문법적용 tool_calls=%d"
          % (surface, n_plain, n_guided))
    if n_plain == 0:
        print("[x704] ⛔문법 없이도 파싱이 0이다 — 표면형 문제가 아니라 서버/모델/도구 스키마를 봐라")
        return 1
    if n_guided == 0:
        print("[x704] ⛔불일치: '%s' 문법을 걸면 서버가 도구를 못 뽑는다. 본문 앞머리: %r"
              % (surface, body[:120]))
        print("[x704]   서버의 --tool-call-parser 를 확인하고 프로필의 T2_TOOL_SURFACE 를 맞춰라")
        print("[x704]   (hermes ↔ qwen3_xml · 근거: model_profiles/README.md)")
        return 1
    print("[x704] 통과 — 선언한 표면형이 이 서버의 파서와 짝이다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
