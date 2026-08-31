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


# ★서버 파서의 표면형 (2026-08-31·x702 격리로 확정). vLLM `parser/qwen3.py:7-11` 축자:
#     <tool_call>
#     <function=func_name>
#     <parameter=key>value</parameter>
#     </function>
#     </tool_call>
#   ⛔이 파일이 위(hermes)에 적어 둔 표면형은 **다른 파서의 것**이다. 표면형이 파서와 어긋나면
#     문법은 문법대로 걸리고(음성대조: 존재하지 않는 이름을 강제하면 그대로 출력된다) 서버는
#     **도구를 하나도 못 뽑는다** — x702 실측 n=3: base native 3/3 ↔ guided native **0/3**·
#     content 에 태그 3/3. 라이브 x693 의 SALVAGED 83/85(98%)가 그 자국이다.
_QWEN3XML_TMPL = r'''root ::= text | calls | text calls
text ::= textchar+
textchar ::= [^<] | "<" [^t]
calls ::= call (ws call)*
call ::= "<tool_call>" ws "<function=" name ">" ws params "</function>" ws "</tool_call>"
params ::= (param ws)*
param ::= "<parameter=" pname ">" pvalue "</parameter>"
pname ::= pnchar+
pnchar ::= [^>]
pvalue ::= pvchar*
pvchar ::= [^<]
ws ::= [ \t\n\r]*
name ::= %(alts)s
'''

SURFACE_HERMES = "hermes"
SURFACE_QWEN3XML = "qwen3_xml"


def surface():
    """서버 도구 파서의 표면형 — **선언되지 않으면 None**(그러면 문법을 안 건다).

    ★왜 기본값을 두지 않나 (2026-08-31): 종전 기본은 hermes 였고, 서버가 Q3.8 과 함께
      `--tool-call-parser qwen3_coder` 로 바뀐 뒤에도 **아무도 안 바꿔서** 두 달치 런이
      전량 강등된 채 돌았다(serve8142_32b_x624.log = hermes ↔ serve8143_pfx.log = qwen3_coder).
      기본값이 있으면 같은 사고가 조용히 재발한다 — 런처가 **선언하게** 만든다([[07]] hard).
    """
    v = (os.environ.get("T2_TOOL_SURFACE") or "").strip().lower()
    return v if v in (SURFACE_HERMES, SURFACE_QWEN3XML) else None


def _esc_name(n):
    return n.replace("\\", "\\\\").replace('"', '\\"')


def _esc_literal(n):
    """XML 표면형에서는 이름이 따옴표 없이 리터럴로 들어간다."""
    return str(n).replace("\\", "\\\\").replace('"', '\\"')


def build_grammar_xml(names):
    """허용 tool 이름 집합 → **qwen3_coder/qwen3_xml 표면형** EBNF (도메인 무관)."""
    uniq = [n for n in dict.fromkeys(str(x) for x in names if x)]
    if not uniq:
        return None
    alts = " | ".join('"%s"' % _esc_literal(n) for n in uniq)
    return _QWEN3XML_TMPL % {"alts": alts}


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


_WARNED = []


def grammar_for_tools(tools):
    """표면형 선언(`T2_TOOL_SURFACE`)에 맞는 문법. **미선언이면 None**(문법 미부착·fail-safe)."""
    names = _names_from_tools(tools)
    if not names:
        return None
    sf = surface()
    if sf is None:
        if not _WARNED:
            _WARNED.append(1)
            print("[T2_GUIDED] ⛔T2_TOOL_SURFACE 미선언 — 문법을 걸지 않는다. "
                  "서버의 --tool-call-parser 에 맞춰 hermes | qwen3_xml 중 하나를 선언하라 "
                  "(어긋나면 네이티브 도구 파싱이 전량 죽는다·x702)", file=sys.stderr, flush=True)
        return None
    key = (sf,) + tuple(sorted(names))
    if key not in _CACHE:
        _CACHE[key] = (build_grammar_xml(names) if sf == SURFACE_QWEN3XML
                       else build_grammar(names))
        _mark("grammar built [%s] for %d tools: %s"
              % (sf, len(names), ",".join(sorted(names)[:4]) + "..."))
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
                    from t2_lever_beat import beat as _beat
                    _beat("T2_GUIDED")   # 효과 증거(VERBOSE 없이도·3회 상한)
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
