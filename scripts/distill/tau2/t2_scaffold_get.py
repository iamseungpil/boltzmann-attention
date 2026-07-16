# -*- coding: utf-8 -*-
"""t2_scaffold_get.py — scaffold-제공 GET 도구 (A2-선언·2026-07-16·BANK_IMPL_REDESIGN).
LLM은 구체 계산을 직접 안 하고 이 GET 도구를 *호출*→scaffold가 결정론 계산(t2_compute.apply_op)→결과 반환.
tau2 네이티브 아님·우리가 A2로 제공하는 일반 GET 함수. [[05]] 엔진=도메인일반·계산공식=A2 op-spec.
활성=T2_SCAFFOLD_GET=1. gate/unified 뒤에 apply(체이닝)."""
import os, re, json, sys as _sys


def _parse_records(text):
    """일반 파서: 'N. Record ID:' 분할 → 'key: value' 추출 → amount $ strip."""
    recs = []
    for chunk in re.split(r"\d+\.\s+Record ID:", text or "")[1:]:
        d = {m.group(1): m.group(2).strip()
             for m in re.finditer(r"(\w+):\s*([^\n]+?)(?:\s{2,}|$)", chunk)}
        for k in list(d):
            if "amount" in k:
                d[k] = d[k].replace("$", "")
        if d:
            recs.append(d)
    return recs


def _gather(messages):
    """레코드를 담은 최신 tool 출력 파싱(도메인일반·records_from 힌트는 옵션)."""
    recs = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role == "tool" and isinstance(content, str):
            r = _parse_records(content)
            if r:
                recs = r
    return recs


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
                                  long_desc=d.get("description", ""), params=Params, returns=Ret))
            except Exception as e:
                print("[T2_SCAFFOLD_GET] inject fail %s: %r" % (d["name"], e), file=_sys.stderr, flush=True)
        self._t2_sg_a2 = a2

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
                ours[id(tc)] = ToolMessage(id=tc.id, role="tool", requestor="assistant", content=_txt)
                print("[T2_SCAFFOLD_GET] %s -> %s" % (getattr(tc, "name"), _n), file=_sys.stderr, flush=True)
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
