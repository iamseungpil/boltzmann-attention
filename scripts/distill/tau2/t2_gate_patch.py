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
from t2_gate import AUTH_TOOLS, RetailGate  # noqa: E402

GATE_DOMAINS = {"retail"}  # airline 등은 게이트 컴파일 후 추가


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

        results = []
        for tc in tool_calls:
            if getattr(tc, "requestor", "assistant") != "assistant":
                results.extend(orig(self, [tc]))  # user-side 툴콜(타 도메인)은 비대상
                continue
            ok, g, why = gate.check(tc.name, tc.arguments or {}, last_user_msg=last_user)
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
