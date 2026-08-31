# -*- coding: utf-8 -*-
"""회귀 — 문법 표면형은 **서버 파서와 같아야 한다** (`T2_TOOL_SURFACE`).

★사고 (2026-08-31 · x702/x703 격리):
  `T2_GUIDED` 의 문법은 hermes(`<tool_call>{"name":…}</tool_call>`)인데 서버 파서는
  `qwen3_coder`(XML)로 바뀌어 있었다. 문법은 걸리고(음성대조: 존재하지 않는 이름을 강제하면
  그대로 나온다) 서버는 도구를 **하나도** 못 뽑는다.
  실측 n=3: 문법없음 native 3/3 · hermes **0/3** · qwen3_xml **3/3**(병렬 2호출 유지).
  런 전수 대조: base x644/x617/x599 본문 hermes 0% ↔ ours x670/x659 91.3%/74.2%.

⚠기본값을 두지 않는다 — 서버가 또 바뀌면 조용히 재발한다. 미선언이면 문법을 안 건다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_guided_patch as G

TOOLS = [{"function": {"name": "get_account_details"}}, {"function": {"name": "unlock_x"}}]


def _fresh():
    G._CACHE.clear(); G._WARNED[:] = []


def test_surface_unset_attaches_nothing():
    _fresh(); os.environ.pop("T2_TOOL_SURFACE", None)
    assert G.surface() is None
    assert G.grammar_for_tools(TOOLS) is None          # fail-safe: 어긋난 문법을 걸지 않는다


def test_qwen3xml_surface_matches_server_form():
    """vLLM `parser/qwen3.py:7-11` 축자 형식과 같은 토큰을 강제해야 한다."""
    _fresh(); os.environ["T2_TOOL_SURFACE"] = "qwen3_xml"
    g = G.grammar_for_tools(TOOLS)
    assert g and "<function=" in g and "<parameter=" in g and "</tool_call>" in g
    assert '"{"' not in g and "namefirst" not in g      # hermes JSON 문법이 섞이면 안 된다
    for n in ("get_account_details", "unlock_x"):
        assert '"%s"' % n in g, n
    _ws = g.split(chr(10)+"ws ::= [")[1].split("]")[0]
    assert chr(92)+"t" in _ws and chr(10) not in _ws   # 문자클래스에 실개행이 들어가면 xgrammar 가 500 을 낸다


def test_hermes_surface_still_available():
    _fresh(); os.environ["T2_TOOL_SURFACE"] = "hermes"
    g = G.grammar_for_tools(TOOLS)
    assert g and "namefirst" in g and "<function=" not in g


def test_cache_is_keyed_by_surface():
    _fresh(); os.environ["T2_TOOL_SURFACE"] = "hermes"
    h = G.grammar_for_tools(TOOLS)
    os.environ["T2_TOOL_SURFACE"] = "qwen3_xml"
    x = G.grammar_for_tools(TOOLS)
    assert h and x and h != x, "표면형이 캐시 키에 없으면 첫 팔이 두 번째를 오염시킨다"


def test_names_only_from_live_schema():
    _fresh(); os.environ["T2_TOOL_SURFACE"] = "qwen3_xml"
    g = G.grammar_for_tools(TOOLS)
    assert "get_all_user_accounts_by_user_id" not in g     # 도메인 리터럴 0([[05]])


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASS")
