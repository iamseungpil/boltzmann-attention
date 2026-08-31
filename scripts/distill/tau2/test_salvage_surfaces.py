# -*- coding: utf-8 -*-
"""회귀 — 강등 구제망은 **두 표면형을 모두** 읽는다(hermes ↔ qwen3_xml).

★왜 (2026-08-31·[[84]]): 서버의 도구 파서가 모델과 함께 바뀐다.
    Qwen2.5 계열 런  --tool-call-parser hermes       → <tool_call>{"name":…}</tool_call>
    Qwen3.8 계열 런  --tool-call-parser qwen3_coder  → <tool_call><function=…><parameter=…>
  한쪽만 읽으면 모델을 바꾸는 순간 구제망이 눈이 먼다 — 실제로 두 달을 그렇게 돌았다.
⚠파싱은 `t2_salvage` **한 곳**에만 있다([[67]] 사본 금지). 러너는 그것을 import 해서 쓴다.
⚠엔진은 형식 복구만 한다 — 이름·인자는 모델이 쓴 문자열 그대로다(선택·해석 0·[[10]]).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_salvage as S

XML = ('<tool_call>\n<function=get_account_details>\n'
       '<parameter=account_id>acc_991</parameter>\n'
       '<parameter=limit>5</parameter>\n</function>\n</tool_call>')
HERMES = '<tool_call>\n{"name": "get_time", "arguments": {"tz": "EST"}}\n</tool_call>'


def test_xml_surface():
    calls = S.extract_calls(XML)
    assert len(calls) == 1
    name, args = calls[0]
    assert name == "get_account_details"
    assert args["account_id"] == "acc_991"
    assert args["limit"] == 5                    # 숫자는 파서와 같은 해석으로 복원


def test_hermes_surface():
    calls = S.extract_calls(HERMES)
    assert calls == [("get_time", {"tz": "EST"})], calls


def test_both_in_one_body():
    calls = S.extract_calls("prose " + XML + " middle " + HERMES)
    assert sorted(n for n, _ in calls) == ["get_account_details", "get_time"]


def test_truncated_block_is_dropped():
    assert S.extract_calls('<tool_call>\n{"name": "a", "argu') == []
    assert S.extract_calls('<tool_call><function=a><parameter=k>v</parameter>') == []


def test_strip_leaves_prose_only():
    left = S.strip_calls("I'll check that.\n" + XML + "\n" + HERMES + "\nDone.").strip()
    assert "<tool_call>" not in left
    assert left.startswith("I'll check") and left.endswith("Done.")


def test_runner_uses_the_canonical_module():
    """러너가 자기 정규식을 다시 갖지 않는다 — 갈리면 한쪽만 고쳐지고 다른 쪽이 눈이 먼다."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_run_gated.py"),
               encoding="utf-8").read()
    assert "import t2_salvage as _SALV" in src
    assert "_re_tr.compile(r\"<tool_call>" not in src, "러너에 표면형 정규식 사본이 되살아났다"


def test_first_call_only():
    assert S.find_first_call(XML * 3)["name"] == "get_account_details"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASS")
