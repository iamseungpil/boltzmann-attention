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
        tools = getattr(ag, "tools", None)
        if not decls or tools is None:
            return
        existing = {getattr(t, "name", None) for t in tools}
        for d in decls:
            if d["name"] in existing:
                continue
            fields = {p: (str, ...) for p in (d.get("params") or {})}
            Params = create_model(d["name"] + "Params", **fields)
            Ret = create_model(d["name"] + "Ret", result=(str, ""))
            def _f(**k):
                return ""
            try:
                tools.append(Tool(_f, name=d["name"], short_desc=d.get("description", ""),
                                  long_desc=d.get("description", ""), params=Params, returns=Ret,
                                  examples=list(d.get("examples") or [])))
            except Exception as e:
                print("[T2_SCAFFOLD_GET] inject fail %s: %r" % (d["name"], e), file=_sys.stderr, flush=True)
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
