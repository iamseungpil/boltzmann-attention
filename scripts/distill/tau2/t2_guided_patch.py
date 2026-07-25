#!/usr/bin/env python
"""tau2 guided-decoding hook: 에이전트 tool_call **이름**을 라이브 스키마로 decode-레벨 제약.

★표적(C154·C159·C162): 모델이 KB 문구("Use get_x_3847")를 따라 **스키마 밖 이름**을 top-level
  tool_call로 방출 → tau2가 실행은 하지만 dispatcher를 안 거쳐 `agent_discoverable_tools` CALLED
  레코드가 안 남음 → full-DB 해시(db_match) 실패. 043 nt4 잔여 blocker가 정확히 이것
  (sim0 ndiff=2·sim2 ndiff=1·상태차 0·C159).

★왜 decode-레벨인가([[45]]/[[07]]/[[42]]): deny 게이트·프롬프트는 **모델 준수에 의존=soft**
  (C152 재발행 포기·C153 프롬프트 무력). 문법 제약은 **토큰 마스킹**이라 준수 불요 = hard.
  단 `tool_choice="required"`는 순수 대화 턴서도 도구를 강제 발화(C162 실측)→대화 불가.
  그래서 **auto 유지 + 이름만 제약**하는 문법을 쓴다(C162: 3조건 검증 통과).

★[[05]] 준수: 문법의 이름 집합은 **호출 시점 `tools` 인자(라이브 스키마)서 생성**한다.
  엔진에 도메인 도구명·도메인 분기 리터럴 **0**. A2 항목 추가도 **불필요**(A2 순증 0).
  새 도메인 = 스키마가 바뀌면 문법이 자동으로 따라감 = ABox-swap만으로 동작.
★[[10]] 준수: 문법은 **형식(이름 집합)만** 제약한다. 어떤 도구를 언제 부를지·인자값은
  전부 모델 결정(유동성 동결 0). 스캐폴드가 tool_call을 주입·재작성하지 않는다.

활성화: `import t2_guided_patch; t2_guided_patch.apply()` + env `T2_GUIDED=1`.
전달 경로(C162 실측): litellm `extra_body`가 OpenAI-호환 body 최상위로 평탄화되어
  vLLM `structured_outputs={"grammar": ...}`로 도달. (raw HTTP body에 "extra_body" 키를
  그대로 넣으면 **무시**됨 — C162 음성대조.)
"""
import os
import re
import sys

_APPLIED = False
_MARKS = []


def _mark(msg):
    _MARKS.append(msg)
    if os.environ.get("T2_GUIDED_VERBOSE"):
        print("[T2_GUIDED] %s" % msg, file=sys.stderr, flush=True)


def marks():
    return list(_MARKS)


# ── 문법 생성 (도메인-일반·라이브 스키마 구동) ───────────────────────────────
# hermes 표면형: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
# text는 "<t"로 시작하는 시퀀스만 배제(=<tool_call> 진입을 문법 분기로 강제).
_GRAMMAR_TMPL = r'''root ::= text | calls | text calls
text ::= textchar+
textchar ::= [^<] | "<" [^t]
calls ::= call (ws call)*
call ::= "<tool_call>" ws obj ws "</tool_call>"
obj ::= "{" ws (namefirst | argsfirst) ws "}"
namefirst ::= "\"name\"" ws ":" ws name ws "," ws "\"arguments\"" ws ":" ws value
argsfirst ::= "\"arguments\"" ws ":" ws value ws "," ws "\"name\"" ws ":" ws name
name ::= %(alts)s
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair ::= string ws ":" ws value
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" schar* "\""
schar ::= [^"\\] | "\\" esc
esc ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
number ::= "-"? int frac? exp?
int ::= "0" | [1-9] [0-9]*
frac ::= "." [0-9]+
exp ::= [eE] [+-]? [0-9]+
ws ::= [ \t\n\r]*
'''


def _esc_name(n):
    return n.replace("\\", "\\\\").replace('"', '\\"')


def build_grammar(names):
    """허용 tool 이름 집합 → EBNF. names=라이브 스키마서 온 문자열 리스트(도메인 무관)."""
    uniq = [n for n in dict.fromkeys(str(x) for x in names if x)]
    if not uniq:
        return None
    alts = " | ".join('"\\"%s\\""' % _esc_name(n) for n in uniq)
    return _GRAMMAR_TMPL % {"alts": alts}


def _names_from_tools(tools):
    """도구 컨테이너 → 이름 리스트. 도메인 무관·형태 무관:
      · dict(name→tool)  : env.tools.get_tools() 반환형(키가 이름)
      · list[Tool]       : generate()가 받는 형태(openai_schema.function.name)
      · list[dict]       : 이미 OpenAI 스키마인 경우
    """
    if not tools:
        return []
    if isinstance(tools, dict):
        return [str(k) for k in tools.keys()]
    out = []
    for t in tools:
        nm = None
        if isinstance(t, str):
            nm = t
        elif isinstance(t, dict):
            nm = (t.get("function") or {}).get("name") or t.get("name")
        else:
            sch = getattr(t, "openai_schema", None)
            if isinstance(sch, dict):
                nm = (sch.get("function") or {}).get("name")
            if not nm:
                nm = getattr(t, "name", None)
        if nm:
            out.append(str(nm))
    return out


_CACHE = {}


def grammar_for_tools(tools):
    names = _names_from_tools(tools)
    if not names:
        return None
    key = tuple(sorted(names))
    if key not in _CACHE:
        _CACHE[key] = build_grammar(names)
        _mark("grammar built for %d tools: %s" % (len(names), ",".join(sorted(names)[:4]) + "..."))
    return _CACHE[key]


# ── 패치 배선 ────────────────────────────────────────────────────────────────
# ★C166 교훈(032 관통): identity-check 체이닝은 다른 패치(gate/maxprompt)가 la.generate를
#   먼저 감쌌으면 스킵됐고, gate의 regen은 자기 apply 시점의 la.generate를 캡처(_og_gen)해
#   직접 호출한다 → guided가 에이전트 경로에 사실상 비활성이었다. 수정:
#   ① la.generate를 **무조건 체인-랩**(현재 값이 무엇이든 그 위에)
#   ② 드라이버에서 guided를 **gate보다 먼저** 적용(gate의 _og_gen 캡처에 guided가 포함되도록)
#   ③ call_name 필터 폐기 — 격리는 *모듈 바인딩*으로 달성:
#      la.generate = 에이전트 모듈 경유 호출만 / user-sim은 자기 모듈의 from-import 바인딩
#      (원본)을 쓰므로 llm_utils.generate 패치의 영향을 받지 않는다(import 시점 캡처).
#   주입 조건 = tools 有 ∧ tool_choice∈{None,"auto"} (tools=None 서브콜은 자동 제외).


def _make_wrapper(inner):
    def _generate(model, messages, tools=None, tool_choice=None, call_name=None, **kwargs):
        if (os.environ.get("T2_GUIDED") == "1"
                and tools
                and tool_choice in (None, "auto")):
            g = grammar_for_tools(tools)
            if g:
                eb = dict(kwargs.get("extra_body") or {})
                if "structured_outputs" not in eb and "guided_grammar" not in eb:
                    eb["structured_outputs"] = {"grammar": g}
                    kwargs["extra_body"] = eb
                    _mark("guided applied (call=%s tools=%d)" % (call_name, len(tools)))
        return inner(model, messages, tools=tools, tool_choice=tool_choice,
                     call_name=call_name, **kwargs)
    return _generate


def apply():
    """generate 몽키패치(체인-랩). T2_GUIDED=1일 때만 문법 주입.
    드라이버는 이 apply()를 gate/eplan/scaffold_get **이전**에 불러야 한다(위 ② 참조)."""
    global _APPLIED
    if _APPLIED:
        return
    from tau2.utils import llm_utils
    llm_utils.generate = _make_wrapper(llm_utils.generate)
    try:
        from tau2.agent import llm_agent as _la
        _la.generate = _make_wrapper(_la.generate)   # 현재 체인 위에 무조건 랩
    except Exception:
        pass
    _APPLIED = True
    _mark("patch applied (chain-wrap)")


if __name__ == "__main__":
    # 오프라인 자기검사: 문법 생성만(서버 불요)
    demo = ["unlock_discoverable_agent_tool", "call_discoverable_agent_tool", "get_current_time"]
    g = build_grammar(demo)
    assert g and 'name ::= ' in g
    for n in demo:
        assert ('\\"%s\\"' % n) in g, n
    assert "get_all_user_accounts_by_user_id_3847" not in g
    print("selftest OK · grammar %d chars · %d names" % (len(g), len(demo)))
