# -*- coding: utf-8 -*-
"""회귀 — 사고 예산은 **매 호출에 따로** 주어지고, 본 응답에도 붙는다.

★결손 (2026-08-31): 예산이 **프로브 가지에만** 걸려 있었다. 본 응답에는 없어서, 상한이 사고
  도중에 걸리면 생성 전량이 reasoning 으로 분류되고 `content=None` 이 된다 → tau2 가
  "AssistantMessage must have either content or tool_calls" 로 **태스크를 통째로 버린다**
  (x693 에서 1,590초 폐기).

★격리 x705 (같은 서버·같은 프롬프트·n=2·전부 결정론):
    예산 없음 mt=512   전손 2/2   reason 2,250B · content 0B
    예산 없음 mt=2048  전손 2/2   reason 8,340B · content 0B
    예산 256  mt=512   전손 0/2   reason 1,131B · content 1,046B
    예산 1024 mt=2048  전손 0/2   finish=stop (절단 자체가 사라진다)
    예산 1024 + 도구   tool_calls 1 정상
  ⇒ 상한을 키우는 것이 아니라 **사고에 예산을 걸어 답 자리를 남긴다**.

⚠[[70]] 무엇을 파나: 사고가 예산에서 끊긴다. 너무 조이면 답이 바뀐다(선행 실측 486토큰).
  그래서 값은 모델 프로필에 **선언**한다 — 코드에서 파생하는 것은 선언이 없을 때의 폴백이다.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "t2_run_gated.py"), encoding="utf-8").read()


def _derive(env, cap):
    """러너의 파생 규칙을 그대로 옮긴 것이 아니라, **소스에 그 규칙이 있는지**를 본다."""
    try:
        return int(env) if env else max(256, int(cap) // 2)
    except Exception:
        return max(256, int(cap or 8192) // 2)


def test_main_call_gets_a_budget():
    assert 'if not _is_probe:' in SRC
    assert '_kw["thinking_token_budget"] = _tbm' in SRC, "본 응답에 예산이 안 붙는다"


def test_budget_is_declared_first_derived_second():
    assert 'os.environ.get("T2_THINK_BUDGET")' in SRC
    assert 'max(256, int(_capm) // 2)' in SRC, "선언이 없을 때의 폴백이 없다"


def test_probe_budget_is_separate():
    assert 'os.environ.get("T2_PROBE_THINK_BUDGET")' in SRC, "프로브 예산이 본 응답과 같은 변수를 쓴다"


def test_derivation_leaves_room_for_the_answer():
    for cap in (2048, 3072, 8192):
        assert _derive(None, cap) == cap // 2, cap
    assert _derive("4096", 8192) == 4096          # 선언이 이긴다
    assert _derive(None, 100) == 256              # 하한


def test_qwen38_profile_declares_both_budgets():
    p = os.path.join(HERE, "model_profiles", "Qwen__Qwen3.8-27B-FP8.env")
    txt = open(p, encoding="utf-8").read()
    assert re.search(r"^export T2_THINK_BUDGET=\d+", txt, re.M), "본 응답 예산 미선언"
    assert re.search(r"^export T2_PROBE_THINK_BUDGET=\d+", txt, re.M), "프로브 예산 미선언"
    assert "x705" in txt, "값의 출처(격리 번호)가 없다"


def test_qwen25_profiles_do_not_declare_a_budget():
    """이 모델은 reasoning_parser 없이 뜬다 — 근거 없는 값을 남기지 않는다."""
    for n in ("Qwen__Qwen2.5-32B-Instruct-GPTQ-Int8.env", "Qwen__Qwen2.5-7B-Instruct.env"):
        txt = open(os.path.join(HERE, "model_profiles", n), encoding="utf-8").read()
        assert not re.search(r"^export T2_THINK_BUDGET=", txt, re.M), n
        assert "reasoning_parser" in txt, "왜 미선언인지가 안 적혀 있다"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASS")
