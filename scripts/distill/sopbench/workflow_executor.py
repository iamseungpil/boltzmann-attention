#!/usr/bin/env python3
"""workflow_executor.py — deterministic call-graph executor + SOP-Bench WorkflowAgent.

Implements the 8-relation call-graph ontology (WORKFLOW_ONTOLOGY_DESIGN §2.2/§3):
  realizes(step->function) · arg(fn,param<-slot) · produces(fn->{ret_key:slot}) ·
  precondition(fn->pred) · next(step-> step | [[pred,step]]) · terminate([[pred,outcome_slot=value]]) ·
  output(required, format)
Everything is a function call: a SOP-Bench tool (via ToolManager.execute_tool) OR a
wrapped pure function (domain compute/decide), registered in WRAPPED.

Predicates are SIMPLE: "slot OP value" with OP in {==,!=,<,<=,>,>=,in,truthy,falsy}.
No eval(); parsed and applied over the slot dict.

This is the procedure-given executor (the call graph is supplied = compiled from sop.txt).
The goal-only planner (L0/L1/L2) reuses the same function-call layer; only `next`/order is
produced by the planner instead of given.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Wrapped-function registry: domain compute/decide steps with no provided tool.
# A wrapped fn takes the current slot dict and returns a dict of new slots.
# Domains register theirs (e.g. cs_functions.register(WRAPPED)).
# ---------------------------------------------------------------------------
WRAPPED: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def register(name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    WRAPPED[name] = fn


# ---------------------------------------------------------------------------
# Simple predicate evaluation over slots
# ---------------------------------------------------------------------------
_PRED_RE = re.compile(r"^\s*([A-Za-z_][\w\.]*)\s*(==|!=|<=|>=|<|>|in|truthy|falsy)\s*(.*)$")


def _coerce(v: str) -> Any:
    s = v.strip().strip("'\"")
    if s in ("True", "true"):
        return True
    if s in ("False", "false"):
        return False
    if s in ("None", "null", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def eval_pred(pred: Optional[str], slots: Dict[str, Any]) -> bool:
    """Evaluate 'slot OP value' against slots. None/'' -> True (unconditional)."""
    if not pred:
        return True
    m = _PRED_RE.match(pred)
    if not m:
        return False
    key, op, rhs = m.group(1), m.group(2), m.group(3)
    left = slots.get(key)
    if op == "truthy":
        return bool(left)
    if op == "falsy":
        return not bool(left)
    right = _coerce(rhs)
    try:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left is not None and left < right
        if op == "<=":
            return left is not None and left <= right
        if op == ">":
            return left is not None and left > right
        if op == ">=":
            return left is not None and left >= right
        if op == "in":
            return left in right if hasattr(right, "__contains__") else False
    except TypeError:
        return False
    return False


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class CallGraphExecutor:
    def __init__(self, ontology: Dict[str, Any]):
        self.ont = ontology
        self.realizes = ontology.get("realizes", {})
        self.arg = ontology.get("arg", {})
        self.produces = ontology.get("produces", {})
        self.precondition = ontology.get("precondition", {})
        self.next = ontology.get("next", {})
        self.terminate = ontology.get("terminate", [])
        self.output = ontology.get("output", {})
        self.start = ontology.get("start") or (ontology.get("steps") or [None])[0]

    def _call(self, fn: str, slots: Dict[str, Any], tools: Any, trace: List[Dict]) -> None:
        args = {p: slots.get(s) for p, s in self.arg.get(fn, {}).items()}
        if fn in WRAPPED:
            out = WRAPPED[fn](slots)
            trace.append({"tool": fn, "parameters": args, "result": out, "wrapped": True})
        else:
            try:
                result = tools.execute_tool(fn, args)
            except Exception as e:  # tool validation failure — record and skip outputs
                trace.append({"tool": fn, "parameters": args, "error": str(e)})
                return
            # SOP-Bench ToolManager returns a ToolCall wrapper; unwrap to the dict result.
            if hasattr(result, "result"):
                if getattr(result, "success", True) is False:
                    trace.append({"tool": fn, "parameters": args,
                                  "error": getattr(result, "error", "tool failed")})
                    return
                result = result.result
            out = {}
            if isinstance(result, dict):
                # (a) merge ALL returned keys under a normalized slot name
                #     ("outage detected" -> outage_detected) — robust to unknown exact keys
                for k, v in result.items():
                    out[re.sub(r"\s+", "_", str(k).strip().lower())] = v
                # (b) explicit produces mapping overrides/augments
                for ret_key, slot in self.produces.get(fn, {}).items():
                    if ret_key in result:
                        out[slot] = result[ret_key]
            trace.append({"tool": fn, "parameters": args, "result": result})
        for k, v in out.items():
            slots[k] = v

    def _check_terminate(self, slots: Dict[str, Any]) -> bool:
        for rule in self.terminate:
            pred, slot, value = rule["pred"], rule["set"], rule["to"]
            if eval_pred(pred, slots):
                slots[slot] = value
                return True
        return False

    def _next_step(self, step: str, slots: Dict[str, Any]) -> Optional[str]:
        nx = self.next.get(step)
        if nx is None:
            return None
        if isinstance(nx, str):
            return nx
        for pred, target in nx:  # [[pred, step], ...] first match wins
            if eval_pred(pred, slots):
                return target
        return None

    def run(self, inputs: Dict[str, Any], tools: Any, max_steps: int = 50):
        slots: Dict[str, Any] = dict(inputs)
        trace: List[Dict] = []
        step = self.start
        seen = 0
        while step is not None and seen < max_steps:
            seen += 1
            if self._check_terminate(slots):
                break
            fn = self.realizes.get(step)
            if fn and eval_pred(self.precondition.get(fn), slots) and fn not in {t["tool"] for t in trace}:
                self._call(fn, slots, tools, trace)
            self._check_terminate(slots)
            step = self._next_step(step, slots)
        return slots, trace

    def render(self, slots: Dict[str, Any]) -> str:
        req = self.output.get("required", [])
        fmt = self.output.get("format", "xml")
        if fmt == "xml":
            return "".join(f"<{k}>{slots.get(k, '')}</{k}>" for k in req)
        return json.dumps({k: slots.get(k) for k in req}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# SOP-Bench agent wrapper
# ---------------------------------------------------------------------------
try:
    from amazon_sop_bench.agents import BaseAgent, AgentResult
except Exception:  # allow import without the package (e.g. local lint)
    BaseAgent = object  # type: ignore
    AgentResult = None  # type: ignore


class WorkflowAgent(BaseAgent):
    """Runs the deterministic call-graph executor against SOP-Bench tasks."""

    def __init__(self, ontology: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.name = "WorkflowAgent"
        self.executor = CallGraphExecutor(ontology)

    def execute(self, sop: str, task: Dict[str, Any], tools: Any):
        slots, trace = self.executor.run(task, tools)
        output = self.executor.render(slots)
        reasoning = "\n".join(
            f"{t['tool']}({t.get('parameters')}) -> {t.get('result', t.get('error'))}" for t in trace
        )
        tool_calls = [{"tool": t["tool"], "parameters": t.get("parameters", {}),
                       "result": t.get("result", t.get("error"))} for t in trace]
        if AgentResult is None:
            return {"output": output, "tool_calls": tool_calls, "reasoning": reasoning}
        return AgentResult(output=output, tool_calls=tool_calls, reasoning_trace=reasoning, success=True)
