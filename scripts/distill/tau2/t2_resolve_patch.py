#!/usr/bin/env python
"""resolve_selection live wiring (설계서 INTEGRATED_SCAFFOLD §3·B + C0).

오프라인 op-eval(tau2_op_eval·태스크/카탈로그 직접 주입) 폐기. 이 패치는 **실 τ² user-sim e2e
안에서** facet3 스페셜리스트(또는 base)가 emit하는 native tool_call `resolve_selection(op, set, ...)`을
(1) 실제 도구로 모델에 노출(environment.get_tools)
(2) 실행을 가로채 결정론 엔진(ma/tau2_op_resolver.resolve_op_tau2)으로 실행
    — catalog/anchor를 *대화 맥락(직전 fetch한 tool 출력)*서 grounding(intensional→extensional).
anchor_id는 모델-가시 인자서 제외(§3a)·엔진이 context anchor로 grounding(order_id 날조 차단).

활성화: import t2_resolve_patch; t2_resolve_patch.apply()
v1 = retail substitute/exchange 우선(product variants + order item anchor). airline cabin·일반화는 후속.
"""
import json
import os
import sys

_MA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ma")
sys.path.insert(0, _MA)
from tau2_op_resolver import resolve_op_tau2  # noqa: E402


def resolve_selection(op: str, attr: str = None, among: dict = None,
                      dir: str = None, k: int = None, set: dict = None) -> str:
    """Identify the catalog item the user wants by NAMING the selection operation.

    Do NOT invent any item id. Name the operation and its operands only; the engine
    resolves the concrete item over the full catalog, grounded from prior tool results
    (e.g. the product/order you already fetched). For an exchange, name only the changed
    attribute(s) in `set` — the engine keeps every other attribute of the current item.

    Args:
        op: one of filter, argmax, argmin, rank, comparative, substitute, create.
        attr: the ordinal/numeric attribute (for argmax/argmin/rank/comparative).
        among: categorical filter constraints as {attribute: value}.
        dir: comparative direction, "greater" or "less".
        k: rank position (1 = top).
        set: for substitute/create, the changed attribute values as {attribute: value}.

    Returns:
        The resolved item_id.
    """
    return ""  # 실제 실행은 _execute_tool_calls 인터셉트(아래)·이 본문은 호출되지 않음.


# ─────────────────────────────────────────────────────────────────────────────
def _args_dict(tc):
    a = getattr(tc, "arguments", None)
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return {}


def _tool_outputs(orch):
    """role==tool·비-error 메시지 content를 JSON 파싱(최근→과거 순)."""
    outs = []
    try:
        msgs = orch.get_messages()
    except Exception:
        return outs
    for m in reversed(msgs):
        if getattr(m, "role", None) != "tool" or getattr(m, "error", False):
            continue
        c = getattr(m, "content", None)
        if not isinstance(c, str):
            continue
        try:
            outs.append(json.loads(c))
        except Exception:
            continue
    return outs


def _ground_retail(outs):
    """직전 fetch(tool 출력)서 (catalog, anchor_id) grounding.
    catalog = 최근 get_product_details 출력의 variants. anchor = 그 product의 order내 현재 item."""
    catalog, anchor, pid = None, None, None
    for o in outs:
        if isinstance(o, dict) and isinstance(o.get("variants"), dict):
            catalog = [{"item_id": iid,
                        "options": (vv or {}).get("options", {}),
                        "available": (vv or {}).get("available", True)}
                       for iid, vv in o["variants"].items()]
            pid = o.get("product_id")
            break
    if pid is not None:
        for o in outs:
            items = o.get("items") if isinstance(o, dict) else None
            if isinstance(items, list):
                cand = [it for it in items
                        if isinstance(it, dict) and it.get("product_id") == pid]
                if cand:
                    anchor = cand[0].get("item_id")
                    break
    return catalog, anchor


def _resolve(orch, tc):
    from tau2.data_model.message import ToolMessage
    args = _args_dict(tc)
    op_ir = {k: v for k, v in args.items() if v is not None and k != "anchor_id"}
    outs = _tool_outputs(orch)
    catalog, anchor = _ground_retail(outs)
    rid = None
    if catalog:
        try:
            rid = resolve_op_tau2(op_ir, catalog, anchor_id=anchor)
        except Exception as e:
            rid = None
            _err = f"resolver error: {type(e).__name__}: {e}"
    if rid:
        content = json.dumps({"status": "ok", "tool": "resolve_selection", "item_id": rid})
        return ToolMessage(id=tc.id, role="tool", requestor="assistant", error=False, content=content)
    content = ("Error: resolve_selection could not ground the item. First fetch the relevant "
               "product (get_product_details) and the order (get_order_details), then name the "
               "operation again. Do NOT guess an item id.")
    return ToolMessage(id=tc.id, role="tool", requestor="assistant", error=True, content=content)


def apply():
    from tau2.environment.environment import Environment
    from tau2.environment.tool import as_tool
    from tau2.orchestrator.orchestrator import BaseOrchestrator

    rtool = as_tool(resolve_selection)

    _orig_get_tools = Environment.get_tools

    def _get_tools(self):
        tools = list(_orig_get_tools(self))
        if not any(getattr(t, "name", None) == "resolve_selection" for t in tools):
            tools.append(rtool)
        return tools

    Environment.get_tools = _get_tools

    _orig_exec = BaseOrchestrator._execute_tool_calls

    def _exec(self, tool_calls):
        out = []
        for tc in tool_calls:
            if getattr(tc, "name", None) == "resolve_selection" \
                    and getattr(tc, "requestor", "assistant") == "assistant":
                out.append(_resolve(self, tc))
            else:
                out.extend(_orig_exec(self, [tc]))
        return out

    BaseOrchestrator._execute_tool_calls = _exec
    print("[t2_resolve_patch] resolve_selection wired (tool exposed + execution intercepted)")
    return _orig_exec


if __name__ == "__main__":
    apply()
