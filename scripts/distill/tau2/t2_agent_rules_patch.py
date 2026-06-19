#!/usr/bin/env python
"""agent 시스템프롬프트에 도메인-일반 rules-prompt 주입 (prompt-vs-SFT 실험·arm B).

가설(2026-06-19 사용자): #6에서 선택규칙이 프롬프트로 풀렸듯, P1-P9/R 규칙도 *명시*하면 SFT 없이
base가 따를 수 있나. 이 패치는 닫힌 기저 규칙(a2/RULES_PROMPT.txt)을 LLMAgent.system_prompt에 append.
도메인-일반(전 도메인 동일·도메인 사실 0)이라 fixed-scaffold/instruction에 해당·per-domain A2 아님.

활성화: t2_agent_rules_patch.apply(rules_path)  또는  T2_RULES_PROMPT=<file>.
판정: base+rules(이 arm) vs base-floor vs SFT-adapter를 같은 e2e로 비교 → 규칙별 프롬프트-충분성.
"""
import os

_RULES = None


def apply(rules_path=None):
    global _RULES
    rules_path = rules_path or os.environ.get("T2_RULES_PROMPT")
    if not rules_path:
        raise SystemExit("[t2_agent_rules_patch] rules 파일 미지정 (apply(path) 또는 T2_RULES_PROMPT)")
    with open(rules_path, encoding="utf-8") as f:
        _RULES = f.read().strip()

    from tau2.agent.llm_agent import LLMAgent
    _orig = LLMAgent.system_prompt.fget  # property getter

    def _sp(self):
        base = _orig(self)
        return base + "\n\n" + _RULES

    LLMAgent.system_prompt = property(_sp)
    print(f"[t2_agent_rules_patch] rules-prompt 주입 ON · {rules_path} ({len(_RULES)} chars)")


if __name__ == "__main__":
    import sys
    apply(sys.argv[1] if len(sys.argv) > 1 else None)
