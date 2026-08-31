# -*- coding: utf-8 -*-
"""회귀 — 자유서술 기본값 인자는 **근거 없으면 안 넘긴다** (`T2_FREE_TEXT_ARG`·R-A1).

★선언 출처는 **env 시그니처**다(gold 근거 0):
    tools.py:2508  close_bank_account_7392(..., reason: str = "Customer requested closure", ...)
    독스트링       "reason (string, optional)"
★정책 축자(`prompts/components/policy_header.md:8`):
    "Do not make up policies, information or actions that you can take on behalf of the user."
★결손(base 전수): gold 는 이 인자를 **안 넘겨** 행이 기본값으로 남는데 모델은 매번 자기 문장을
  채운다 — 060 065 066 067 068 069, 전부 `gold=None ↔ act='Customer …'`.
★부호표(base 98 sim · **자기-그라운딩 제거**): ⊕실패 sim 발화 6 · ⊖**통과 sim 발화 0** · 무발화 92.
⚠호출은 그대로 실행하고 **그 인자만** 뺀다 — 엔진 기본값이 정본이고, 우리가 값을 고르지 않는다.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
A2 = json.load(open(os.path.join(HERE, "a2/banking_knowledge.gate.json"), encoding="utf-8"))


def test_declaration_exists_in_all_three_layers():
    for f in ("a2/banking_knowledge.gate.json", "a2/banking_knowledge.specific.json",
              "a2/split/banking_knowledge.core.json"):
        d = json.load(open(os.path.join(HERE, f), encoding="utf-8"))
        assert d.get("free_text_defaults") == {"close_bank_account_7392": ["reason"]}, f


def test_declaration_matches_the_env_signature():
    """선언은 env 시그니처에서 왔다 — 디스패처 페이로드(`arguments`)는 **대상이 아니다**."""
    d = A2["free_text_defaults"]
    assert list(d) == ["close_bank_account_7392"]
    for wrapper in ("call_discoverable_agent_tool", "give_discoverable_user_tool",
                    "call_discoverable_user_tool"):
        assert wrapper not in d, "디스패처 페이로드를 지우면 호출이 깨진다"


def test_note_carries_the_sign_table():
    n = A2.get("_note_free_text_defaults") or ""
    assert "부호표" in n and "통과 sim 발화" in n, "부호표 없이 켠 레버는 안 된다([[70]])"
    assert "policy_header" in n and "tools.py" in n, "출처(정책·환경)가 안 적혀 있다"


def test_engine_drops_only_the_argument_not_the_call():
    seg = SRC.split("[T2_FREE_TEXT_ARG]")[1][:4000]
    assert "_bag9.pop(_k9, None)" in seg, "인자를 빼야 한다"
    for forbidden in ("tool_calls = []", "return None", "am.tool_calls.remove"):
        assert forbidden not in seg, "호출 자체를 없애면 안 된다: " + forbidden


def test_corpus_is_prior_context_only():
    """자기-그라운딩 금지 — 우리가 방금 보낸 값이 메아리쳐 오면 다음 호출부터 늘 '실재'가 된다."""
    seg = SRC.split("[T2_FREE_TEXT_ARG]")[1][:4000]
    assert "state.messages" in seg
    assert "am.tool_calls" not in seg.split("_corp = \" \".join")[0], "현재 턴 출력이 코퍼스에 들어갔다"


def test_flag_off_is_byte_identical():
    seg = SRC.split("[T2_FREE_TEXT_ARG] 자유서술")[1][:1500]
    assert 'os.environ.get("T2_FREE_TEXT_ARG") == "1"' in SRC.split("[T2_FREE_TEXT_ARG] 자유서술")[1][:2000]


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASS")
