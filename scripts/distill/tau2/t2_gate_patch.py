#!/usr/bin/env python
"""tau2 게이트 hook: BaseOrchestrator._execute_tool_calls 몽키패치 (BENCH_PORTFOLIO §3.6 ③).

SOPBench two_stage_client 패턴 동형 — 에이전트 툴콜을 실행 *전* RetailGate로 검사,
deny면 실행 없이 게이트 메시지를 ToolMessage(error)로 반환(모델이 보고 교정),
allow면 원본 실행 후 결과로 게이트 상태 갱신(find 성공 -> 인증 확립).

활성화: 시뮬 실행 파이썬에서 `import t2_gate_patch; t2_gate_patch.apply()`
또는 환경변수 T2_GATE=1 로 sitecustomize 류에서 호출. 게이트는 orchestrator
인스턴스당 1개(_t2_gate) — 대화별 인증 상태 분리.

G2(쓰기-전-확인)는 직전 user 발화를 메시지 이력에서 추출. user 턴이 아직 없으면
(이론상 WRITE가 첫 액션일 수 없음) deny 쪽으로 떨어짐 — 정책 부합.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_gate import AUTH_TOOLS, TRANSFER_MSG, RetailGate  # noqa: E402

GATE_DOMAINS = {"retail"}  # airline 등은 게이트 컴파일 후 추가

# ─── L2 provenance 게이트 (R1B, env T2_PROVENANCE=1; D1 직교·G1-G4와 합성) ───
PROV_ARG_HINT = ("email", "name", "zip", "user_id", "order_id", "username", "id",
                 "payment", "address", "phone", "item")


def _flatten(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _flatten(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _flatten(x)
    else:
        yield v


def _context_text(orch):
    """provenance 출처 = 모든 user 발화 + 도구 출력(assistant 제외)."""
    parts = []
    try:
        for m in orch.get_messages():
            r = getattr(m, "role", None)
            c = getattr(m, "content", None)
            if r in ("user", "tool") and c is not None:
                parts.append(c if isinstance(c, str) else str(c))
    except Exception:
        pass
    return " ".join(parts).lower()


def _provenance_deny(tc, ctx):
    """identifying 인자값이 컨텍스트에 없으면 fabricated → (gate, reason) 반환, 아니면 None."""
    args = tc.arguments or {}
    if not isinstance(args, dict):
        return None
    for k, v in args.items():
        if not any(h in k.lower() for h in PROV_ARG_HINT):
            continue
        for val in _flatten(v):
            s = str(val).strip()
            if len(s) < 4:
                continue
            if s.lower() not in ctx:
                return ("PROVENANCE_R1B",
                        f"argument '{k}'='{s}' was not provided by the user nor returned by any tool — it looks invented "
                        "(possibly copied from a schema example value). Do NOT call any tool with a guessed/placeholder value. "
                        "Instead OBTAIN the real value first: if a lookup/getter tool can produce it "
                        "(e.g. call get_user_details to retrieve the user's orders, payment methods, or addresses), call that and read the value from its output; "
                        "otherwise ASK the user for it.")
    return None


def apply():
    from tau2.orchestrator.orchestrator import BaseOrchestrator

    orig = BaseOrchestrator._execute_tool_calls

    def gated(self, tool_calls):
        env = self.environment
        if getattr(env, "domain_name", None) not in GATE_DOMAINS:
            return orig(self, tool_calls)
        gate = getattr(self, "_t2_gate", None)
        if gate is None:
            gate = self._t2_gate = RetailGate(db=env.tools.db)
        last_user = _last_user_text(self)
        tms = _transfer_msg_sent(self)

        prov_on = os.environ.get("T2_PROVENANCE") == "1"
        # T2_PROV_SOFT=1: provenance 거부(가로채기+redirect)를 hard-error로 안 셈 — budget 보존
        #   (사용자 Q1: 거부가 10-error budget 까먹으면 효과 안 보임). 무한루프 방지=별도 cap.
        prov_soft = os.environ.get("T2_PROV_SOFT") == "1"
        prov_cap = int(os.environ.get("T2_PROV_CAP", "12"))
        ctx = _context_text(self) if prov_on else None

        results = []
        for tc in tool_calls:
            if getattr(tc, "requestor", "assistant") != "assistant":
                results.extend(orig(self, [tc]))  # user-side 툴콜(타 도메인)은 비대상
                continue
            if prov_on:  # L2 provenance: 날조 인자값 차단 (R1B)
                pd = _provenance_deny(tc, ctx)
                if pd:
                    n_prov = getattr(self, "_t2_prov_denies", 0) + 1
                    self._t2_prov_denies = n_prov
                    if prov_soft:
                        if n_prov > prov_cap:  # 무한 redirect 방지
                            self.num_errors += 1
                    else:
                        self.num_errors += 1
                    results.append(_deny_msg(tc, pd[0], pd[1]))
                    continue
            ok, g, why = gate.check(tc.name, tc.arguments or {}, last_user_msg=last_user,
                                    transfer_msg_sent=tms)
            if not ok:
                self.num_errors += 1
                results.append(_deny_msg(tc, g, why))
                continue
            out = orig(self, [tc])
            results.extend(out)
            if tc.name in AUTH_TOOLS and out and not out[0].error:
                gate.observe(tc.name, tc.arguments, _content_str(out[0]))
        return results

    BaseOrchestrator._execute_tool_calls = gated
    return orig


def _last_user_text(orch):
    try:
        for m in reversed(orch.get_messages()):
            if getattr(m, "role", None) == "user" and getattr(m, "content", None):
                return m.content if isinstance(m.content, str) else str(m.content)
    except Exception:
        pass
    return None


def _transfer_msg_sent(orch):
    """G4: 고정 transfer 문구가 어시스턴트 발화로 이미 송신됐는가 (불가 판단 시 None)."""
    try:
        for m in orch.get_messages():
            if getattr(m, "role", None) == "assistant":
                c = getattr(m, "content", None)
                if isinstance(c, str) and TRANSFER_MSG in c:
                    return True
        return False
    except Exception:
        return None


def _content_str(tool_msg):
    c = tool_msg.content
    if isinstance(c, str):
        try:
            v = json.loads(c)
            return v if isinstance(v, str) else c
        except (ValueError, TypeError):
            return c
    return str(c)


def _deny_msg(tc, gate_name, reason):
    from tau2.data_model.message import ToolMessage
    return ToolMessage(
        id=tc.id, role="tool", requestor="assistant", error=True,
        content=f"Error: [POLICY GATE {gate_name}] {reason}",
    )


if __name__ == "__main__":
    apply()
    print("[t2_gate_patch] applied")
