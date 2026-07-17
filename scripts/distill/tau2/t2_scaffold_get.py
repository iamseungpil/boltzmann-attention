# -*- coding: utf-8 -*-
"""t2_scaffold_get.py — scaffold-제공 GET 도구 (A2-선언·2026-07-16·BANK_IMPL_REDESIGN).
LLM은 구체 계산을 직접 안 하고 이 GET 도구를 *호출*→scaffold가 결정론 계산(t2_compute.apply_op)→결과 반환.
tau2 네이티브 아님·우리가 A2로 제공하는 일반 GET 함수. [[05]] 엔진=도메인일반·계산공식=A2 op-spec.
활성=T2_SCAFFOLD_GET=1. gate/unified 뒤에 apply(체이닝)."""
import os, json, sys as _sys

# ★삭제됨(2026-07-16·[[03b]]): `_parse_records`/`_gather` = 엔진이 tool 출력 텍스트를 정규식 파싱해
#   operand를 추출('$' strip 포함)하던 코드 = **엔진-formalize = 구현 속임**(이번 세션 위반 #2의 잔해).
#   호출부 0(전 repo grep 확인)이었으나, 남겨두면 재사용 유혹 = 표류 원천이므로 제거.
#   정본 경로: operand는 **LLM이 formalize**해 도구 인자로 넘기고(아래 exec2), 엔진은 op 실행만 한다([[10]]).


def _build_tool(Tool, d):
    """A2 선언 → tau2 Tool. ★tau2 `Tool.__init__`은 **함수 객체에서** 스키마를 유도한다
    (`tool.py:61-73`: `name = func.__name__` · `parse_data(sig, doc, ...)`) — 우리가 넘기는
    `name=`/`long_desc=`/`params=`는 **무시된다**(predefined는 params 제외용).
    ⇒ 진짜 이름·시그니처·docstring을 가진 함수를 동적 생성해서 넘긴다.
    docstring은 `docstring_parser`가 읽는 형식(요약 + `:param x: 설명`)이어야 인자 설명이 스키마에 실린다."""
    name = d["name"]
    params = d.get("params") or {}
    doc = [str(d.get("description") or name).strip(), ""]
    for p, desc in params.items():
        doc.append(":param %s: %s" % (p, str(desc).replace("\n", " ").strip()))
    src = "def %s(%s):\n    pass\n" % (name, ", ".join("%s: str" % p for p in params))
    ns = {}
    exec(compile(src, "<a2_tool:%s>" % name, "exec"), ns)          # noqa: S102 — A2 선언서 생성
    fn = ns[name]
    fn.__doc__ = "\n".join(doc)
    return Tool(fn, examples=list(d.get("examples") or []))


def _a2_named_in_args(tc, decls):
    """(a1) 호출 `tc`의 인자 값 중 **우리 A2 도구 이름과 정확히 일치**하는 게 있으면 그 이름을 반환.
    구조적 사실만 본다 — env 응답 텍스트를 읽지 않고([[03b]] 엔진-formalize 금지), 도메인 도구명을
    리터럴로 갖지 않는다([[05]]). 부분일치는 **안 본다**(자유 텍스트 오탐 방지: 산문 안에 도구명이
    스쳐도 가로채면 안 됨 — `FAB_PROBES §12` 오탐 사고와 동종)."""
    _args = getattr(tc, "arguments", None) or {}
    if not isinstance(_args, dict):
        return None
    for _v in _args.values():
        if isinstance(_v, str) and _v.strip() in decls:
            return _v.strip()
    return None


def apply():
    if os.environ.get("T2_SCAFFOLD_GET") != "1":
        return None
    from tau2.orchestrator.orchestrator import BaseOrchestrator
    from tau2.data_model.message import ToolMessage
    from tau2.environment.tool import Tool
    from pydantic import create_model
    import t2_compute as _c
    import t2_gate_patch as _g

    # (1) 도구 스키마 주입 (per-sim·orchestrator init 후 agent.tools에 append)
    orig_init = BaseOrchestrator.__init__

    def init2(self, *a, **kw):
        orig_init(self, *a, **kw)
        env = getattr(self, "environment", None)
        ag = getattr(self, "agent", None)
        a2 = _g._domain_a2(getattr(env, "domain_name", None)) if env is not None else None
        if not a2 or ag is None:
            return
        decls = a2.get("scaffold_get_tools") or []
        # ★키스톤 toggle(C103·2026-07-17): T2_SG_EXCLUDE=이름들(콤마) → 해당 A2 도구를 주입서 제외.
        #   단일 변수 대조(대안 도구 유/무)용 실험 스위치 — 엔진은 이름 필터만(도메인 리터럴 0).
        #   제외된 도구는 known-set에도 안 들어가므로 TOOLGATE가 진짜 부재처럼 취급(일관).
        _excl = {x.strip() for x in (os.environ.get("T2_SG_EXCLUDE") or "").split(",") if x.strip()}
        if _excl:
            decls = [d for d in decls if d.get("name") not in _excl]
            print("[T2_SCAFFOLD_GET] EXCLUDED by env: %s" % sorted(_excl), file=_sys.stderr, flush=True)
        tools = getattr(ag, "tools", None)
        if not decls or tools is None:
            return
        existing = {getattr(t, "name", None) for t in tools}
        for d in decls:
            if d["name"] in existing:
                continue
            try:
                tools.append(_build_tool(Tool, d))
            except Exception as e:
                print("[T2_SCAFFOLD_GET] inject fail %s: %r" % (d["name"], e), file=_sys.stderr, flush=True)
                continue
        # ★★라이브 검증 의무([[30]]: 단위통과≠라이브발화). 주입 결과 스키마를 실제로 찍는다 —
        #   2026-07-16 사고: Tool이 name/desc/params를 func에서 유도하는데 더미 `def _f(**k)`를 넘겨
        #   **이름 `_f`·설명 `_f`·인자 `k`** 로 들어갔고, 모델은 우리 도구를 *본 적이 없었다*.
        for t in tools:
            if getattr(t, "name", None) in {x["name"] for x in decls}:
                _s = t.openai_schema["function"]
                print("[T2_SCAFFOLD_GET] injected name=%s desc=%dch params=%s"
                      % (_s["name"], len(_s.get("description") or ""),
                         list((_s.get("parameters") or {}).get("properties") or {})),
                      file=_sys.stderr, flush=True)
        self._t2_sg_a2 = a2
        try:
            self._t2_known_tools = {getattr(t, "name", None) for t in (getattr(ag, "tools", None) or [])}
        except Exception:
            self._t2_known_tools = set()

    BaseOrchestrator.__init__ = init2

    # (2) 호출 intercept: 우리 도구면 결정론 계산·반환·env 우회 (gate/unified 뒤 체이닝)
    orig_exec = BaseOrchestrator._execute_tool_calls

    def exec2(self, tool_calls):
        a2 = getattr(self, "_t2_sg_a2", None)
        decls = {d["name"]: d for d in ((a2 or {}).get("scaffold_get_tools") or [])}
        for _x in {x.strip() for x in (os.environ.get("T2_SG_EXCLUDE") or "").split(",") if x.strip()}:
            decls.pop(_x, None)          # 제외 도구는 실행 경로서도 부재(주입 필터와 일관)
        if not decls:
            return orig_exec(self, tool_calls)
        ours = {}
        rest = []
        for tc in tool_calls:
            # ★★requestor 격리 (2026-07-16 버그픽스·`ASSERTION_PROVENANCE_ARMS_DESIGN` §7):
            #   orchestrator._execute_tool_calls는 **AGENT와 USER 양쪽**의 호출을 처리한다
            #   (orchestrator.py:882 `from_role in [AGENT, USER] and to_role == ENV`).
            #   우리 도구·TOOLGATE는 *에이전트* 도구집합(agent.tools) 기준이므로 user 호출에 적용하면:
            #     (1) 사용자의 **gold 액션**을 차단 (task_019 gold 4/6 = requestor:user
            #         `call_discoverable_user_tool`) → 그 task는 **통과 불가**가 된다.
            #     (2) requestor 불일치 ToolMessage가 user-sim 히스토리에 들어가
            #         `user_simulator_base.py:102` ValueError → Retry → infrastructure_error.
            #   ⇒ 에이전트 호출만 다룬다. 나머지는 원본 실행 경로로.
            if getattr(tc, "requestor", "assistant") != "assistant":
                rest.append(tc)
                continue
            if getattr(tc, "name", None) in decls:
                d = decls[getattr(tc, "name")]
                # ★LLM이 formalize한 clean operand(각 인자)를 ctx로([[10]]). 엔진은 op 실행만·원시파싱 안함.
                _args = getattr(tc, "arguments", None) or {}
                _ctx = {}
                for _k, _v in (_args.items() if isinstance(_args, dict) else []):
                    if isinstance(_v, str):
                        try:
                            _v = json.loads(_v)
                        except Exception:
                            pass
                    _ctx[_k] = _v
                _res = _c.apply_op(d.get("op"), _ctx)
                if isinstance(_res, list):                    # 목록형(discrepancy ids)
                    _res = [str(i) for i in _res if i]
                    _txt = d.get("return_template", "{ids}").format(ids=", ".join(_res) if _res else "(none)")
                    _n = len(_res)
                else:                                         # 스칼라형(verdict 등)
                    _txt = d.get("return_template", "{result}").format(result=_res if _res is not None
                                                                       else d.get("missing_hint", "(could not compute — check your arguments)"))
                    _n = _res
                # requestor는 tau2 원본과 동형으로 **미러링**(environment.get_response: requestor=message.requestor).
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                           requestor=getattr(tc, "requestor", "assistant"), content=_txt)
                print("[T2_SCAFFOLD_GET] %s -> %s" % (getattr(tc, "name"), _n), file=_sys.stderr, flush=True)
            elif (os.environ.get("T2_SG_TRUTH") == "1"
                  and _a2_named_in_args(tc, decls)):
                # ★(a1) 인터페이스-사실 정정 (2026-07-18·FAB_PROBES §5.2). 우리가 A2 도구를 **도구 목록에만**
                #   주입하고 env의 도달 레지스트리엔 등록하지 않아서, env가 우리 도구에 대해 **거짓을 말한다**
                #   ("This tool is not available") → 모델이 자기 도구 목록보다 env를 믿고 → 눈대중 → 완료 날조.
                #   ⇒ env가 **우리 도구 이름을 인자로** 받은 호출은 우리가 가로채 **사실만** 답한다.
                #   [[05]]/[[16]] 경계: 판정 기준 = 우리 A2 도구명과의 **정확 일치**뿐(env 응답 텍스트 파싱 0·
                #   도메인 리터럴 0·엔진은 자기 도구명만 안다). 선택 유도 아님 — 모델은 **이미 그 도구를 골랐고**,
                #   우리는 호출 인터페이스 사실만 정정한다.
                #   ⚠️over-action 위험(모트 제1원리: 레버는 하나 사면 하나 판다) = 우리 도구명을 인자로 쓰는
                #   **정당한** 호출(예: KB 검색 query)까지 가로챌 수 있다 → Δspurious 계측 전엔 기본 OFF.
                _tn = _a2_named_in_args(tc, decls)
                _msg = ("`%s` is not managed by `%s`. `%s` is already one of the tools provided to you — "
                        "call `%s` directly with its arguments." % (_tn, getattr(tc, "name", "") or "", _tn, _tn))
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                           requestor=getattr(tc, "requestor", "assistant"), content=_msg)
                print("[T2_SG_TRUTH] '%s(%s)' -> interface fact (env would have denied our tool)"
                      % (getattr(tc, "name", "") or "", _tn), file=_sys.stderr, flush=True)
            elif (os.environ.get("T2_TOOLGATE") == "1"
                  and getattr(self, "_t2_known_tools", None)
                  and getattr(tc, "name", None) not in self._t2_known_tools):
                # ★invalid 선택 → ASK (GET/FIND/INFER/ASK의 ASK 분기·단순 fail/forcing/추천 아님).
                #   LLM은 유한 도구집합서 선택만 함(생성 아님). 매칭 실패 = 필요값을 사용자에게 물어라.
                _msg = ("'%s' is not one of your available tools, so nothing was called. Do not invent tools — "
                        "you may only call tools that are provided to you. If you are missing information needed "
                        "to use one of your available tools, ASK the customer to provide that information, then "
                        "call an available tool." % (getattr(tc, "name", "") or ""))
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool",
                                           requestor=getattr(tc, "requestor", "assistant"),
                                           error=True, content=_msg)
                print("[T2_TOOLGATE] invalid selection '%s' -> ASK prompt"
                      % (getattr(tc, "name", "") or ""), file=_sys.stderr, flush=True)
            else:
                rest.append(tc)
        rest_res = orig_exec(self, rest) if rest else []
        ri = iter(rest_res)
        out = []
        for tc in tool_calls:
            if id(tc) in ours:
                out.append(ours[id(tc)])
            else:
                try:
                    out.append(next(ri))
                except StopIteration:
                    pass
        return out

    BaseOrchestrator._execute_tool_calls = exec2
    print("[T2_SCAFFOLD_GET] ON", file=_sys.stderr, flush=True)
    return True
