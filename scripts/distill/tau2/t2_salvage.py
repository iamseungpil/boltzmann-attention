# -*- coding: utf-8 -*-
"""P3 살리기 — hermes 파서가 텍스트로 강등한 도구 호출 회수 (C248 처방·2026-07-31).

설계 정본 = `RUNAWAY_DEMOTION_REMEDIATION_DESIGN_2026_07_31.md`(rev2·W1~W3 통과).

★왜 필요한가(C248 진단): 모델이 **첫 호출을 정상으로 완결한 뒤 멈추지 못해** 같은 블록을 최대
385개까지 복제하고, 상한에서 마지막 블록이 잘린다. 그러면 vLLM hermes 파서가 예외 경로
(`except: tools_called=False, content=model_output`)로 떨어져 **앞의 완결 호출까지 전부 텍스트로
강등**한다(all-or-nothing). 시스템은 "에이전트가 행동하지 않았다"로 기록하고, 그 오분류가
실패 census의 미실행 축에 섞인다.

★검증(오프라인·`x15_salvage_verify.py`): 궤적 42건 중 **복제형 38/38 = 100% 회수** ·
정상 호출 2,909 / 산문 1,756 / give-flow 서술 77건에서 **오탐 0** · 최대 복제 385개에서도 **회수 1개**.

★[[05]]/[[10]] 정합: 어느 도구를 부를지는 **모델이 이미 정했다**. 이 모듈은 파서가 하려던 일을
대신할 뿐, 새 호출을 발명하지 않는다. 복제분은 **버린다**(첫 블록만) — 385개를 실행하면
over-action 재앙이고, 복제는 의도가 아니라 정지 실패의 산물이다.

기본 **OFF**(`T2_SALVAGE=1`일 때만 동작) — 비교성 보존·롤백은 플래그 복원으로 값싸다.
"""
import json
import os
import re
import sys

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"

# ★두 표면형을 **한 곳에서** 읽는다 (2026-08-31·[[67]] 사본 금지·[[84]]).
#   서버의 도구 파서가 무엇이냐에 따라 모델이 내는 형식이 다르다:
#     hermes      : <tool_call>{"name": …, "arguments": {…}}</tool_call>   (Qwen2.5 계열 런)
#     qwen3_xml   : <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>
#                   (vLLM `parser/qwen3.py:7-11` 축자 · Qwen3.8 계열 런)
#   ⛔한쪽만 읽으면 모델을 바꾸는 순간 **구제망이 눈이 먼다** — 그 사고를 2026-08-31 에 겪었다.
#   ⚠엔진은 **형식 복구만** 한다: 이름·인자는 모델이 쓴 문자열 그대로다(선택·해석 0·[[10]]).
_HERMES_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_XML_RE = re.compile(r"<tool_call>\s*<function=(.*?)>(.*?)</function>\s*</tool_call>", re.S)
_XML_PARAM_RE = re.compile(r"<parameter=(.*?)>(.*?)</parameter>", re.S)


def extract_calls(content):
    """본문의 완결 호출을 **선언 순서대로** [(name, arguments dict), …] 로. 표면형 무관."""
    if not isinstance(content, str) or _OPEN not in content:
        return []
    out = []
    for _m in _XML_RE.finditer(content):                 # 서버 표면형(XML)
        nm = (_m.group(1) or "").strip()
        if not nm:
            continue
        args = {}
        for k, v in _XML_PARAM_RE.findall(_m.group(2) or ""):
            k, v = (k or "").strip(), (v or "").strip()
            try:                    # 숫자·불리언·객체는 파서와 같은 해석으로 복원
                args[k] = json.loads(v)
            except Exception:
                args[k] = v
        out.append((nm, args))
    for _b in _HERMES_RE.findall(content):               # hermes 표면형(JSON)
        try:
            o = json.loads(_b)
        except Exception:
            continue                                     # 미종결/깨진 블록은 파서와 같은 판정
        if isinstance(o, dict) and o.get("name"):
            a = o.get("arguments", {})
            out.append((str(o["name"]), a if isinstance(a, dict) else (a or {})))
    return out


def strip_calls(content):
    """본문에서 호출 블록만 지운 나머지(산문). 표면형 무관."""
    if not isinstance(content, str):
        return ""
    return _XML_RE.sub("", _HERMES_RE.sub("", content))


def find_first_call(content):
    """본문에서 **첫 완결 블록**의 호출 하나만 회수. 없으면 None (순수 함수·selftest 대상)."""
    calls = extract_calls(content)
    if not calls:
        return None
    return {"name": calls[0][0], "arguments": calls[0][1]}


def salvage_message(msg):
    """강등된 assistant 메시지를 제자리 복구. 반환 = 회수했으면 True.

    술어(닫힘): `tool_calls`가 비어 있고 ∧ 본문에 `<tool_call>`이 있다.
    회수 시 `content`는 **첫 블록 앞부분만** 남긴다(복제 텍스트가 히스토리를 오염시키지 않게).
    """
    if os.environ.get("T2_SALVAGE") != "1":
        return False
    try:
        if getattr(msg, "tool_calls", None):
            return False
        content = getattr(msg, "content", None)
        call = find_first_call(content)
        if call is None:
            return False
        from tau2.data_model.message import ToolCall
        n_blocks = content.count(_OPEN)
        msg.tool_calls = [ToolCall(id="salvage_0", name=call["name"], arguments=call["arguments"])]
        msg.content = content[:content.find(_OPEN)].strip() or None
        setattr(msg, "_t2_salvaged", True)
        print("[T2_SALVAGE] 강등 회수: %s (본문 블록 %d개 중 첫 1개·나머지 폐기)"
              % (call["name"], n_blocks), file=sys.stderr, flush=True)
        try:
            import t2_fbsidecar as _sc
            _sc.record("salvage", call["name"], None, blocks=n_blocks, tool=call["name"])
        except Exception:
            pass
        return True
    except Exception as e:                      # 살리기 실패가 런을 깨면 안 된다
        print("[T2_SALVAGE] 실패(무시): %r" % (e,), file=sys.stderr, flush=True)
        return False


if __name__ == "__main__":
    # ── selftest ─────────────────────────────────────────────────────────────
    good = _OPEN + ' {"name": "KB_search", "arguments": {"query": "dispute"}}' + _CLOSE
    cases = [
        ("정상 1블록", good, "KB_search"),
        ("복제 3블록", "prose " + good * 3, "KB_search"),
        ("복제+마지막 절단", good * 2 + _OPEN + ' {"name": "KB_', "KB_search"),
        ("산문만", "I will help you with that.", None),
        ("give-flow 서술", 'Use this: {"name": "x", "arguments": {}}', None),
        ("첫 블록 깨짐", _OPEN + ' {"name": "a", "arguments": "{\\"x\\": ' + _CLOSE, None),
        # ★qwen3_xml 표면형(2026-08-31) — 서버 파서가 qwen3_coder 인 런의 강등은 이 형태다
        ("XML 1블록", _OPEN + "<function=KB_search><parameter=query>dispute</parameter>"
                       "</function>" + _CLOSE, "KB_search"),
        ("XML 복제 2블록", ("x" + _OPEN + "<function=KB_search><parameter=query>a</parameter>"
                            "</function>" + _CLOSE) * 2, "KB_search"),
        ("XML 미종결", _OPEN + "<function=KB_search><parameter=query>a</parameter>", None),
    ]
    ok = 0
    for name, txt, want in cases:
        got = find_first_call(txt)
        gotname = got["name"] if got else None
        flag = "OK" if gotname == want else "FAIL"
        ok += (gotname == want)
        print("  %-18s want=%-10s got=%-10s %s" % (name, want, gotname, flag))
    print("find_first_call selftest %d/%d" % (ok, len(cases)))

    class _M:
        def __init__(self, tc, c):
            self.tool_calls, self.content = tc, c

    os.environ.pop("T2_SALVAGE", None)
    m = _M([], "prose " + good * 5)
    assert salvage_message(m) is False and not m.tool_calls, "기본 OFF여야 한다"
    print("  기본 OFF no-op: OK")
    sys.exit(0 if ok == len(cases) else 1)
