# -*- coding: utf-8 -*-
"""회귀 — 강등 구제망은 **서버 표면형(XML)** 도 줍는다.

★왜 (2026-08-31): 구판 `_TC_RE` 는 hermes JSON 만 봤다. 그 형은 우리 문법이 강제할 때만
  나오므로, 문법을 서버에 맞추는 순간 구제망은 **구조적으로 눈이 먼다** — 강등이 나면
  본문은 vLLM `parser/qwen3.py:7-11` 축자 형식이다:
      <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>
⚠엔진은 형식 복구만 한다 — 이름·인자는 모델이 쓴 문자열 그대로다(선택·해석 0·[[10]]).
"""
import json, re, sys, os

# 러너의 정규식을 **소스에서 그대로** 꺼내 검정한다(사본을 짜면 갈린다·[[67]]).
SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_run_gated.py"),
           encoding="utf-8").read()


def _re_from_source(varname):
    m = re.search(varname + r"\s*=\s*_re_tr\.compile\((r\"[^\n]*?\"),\s*\n?\s*_re_tr\.S\)", SRC, re.S)
    assert m, varname + " 를 소스에서 못 찾았다"
    return re.compile(eval(m.group(1)), re.S)


XTC = _re_from_source("_XTC_RE")
XPARAM = _re_from_source("_XPARAM_RE")
TC = _re_from_source("_TC_RE")

XML = ('<tool_call>\n<function=get_account_details>\n'
       '<parameter=account_id>acc_991</parameter>\n'
       '<parameter=limit>5</parameter>\n</function>\n</tool_call>')
HERMES = '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'


def test_xml_is_extracted():
    ms = list(XTC.finditer(XML))
    assert len(ms) == 1
    assert ms[0].group(1).strip() == "get_account_details"
    kv = dict(XPARAM.findall(ms[0].group(2)))
    assert kv["account_id"].strip() == "acc_991"
    assert json.loads(kv["limit"].strip()) == 5          # 숫자는 파서와 같은 해석으로 복원


def test_hermes_still_extracted():
    assert TC.findall(HERMES) and json.loads(TC.findall(HERMES)[0])["name"] == "get_time"


def test_surfaces_do_not_cross_match():
    assert not TC.findall(XML), "XML 을 hermes 정규식이 먹으면 인자가 사라진다"
    assert not XTC.findall(HERMES)


def test_stripping_leaves_prose():
    body = "I'll check that.\n" + XML + "\nDone."
    left = XTC.sub("", TC.sub("", body)).strip()
    assert "<tool_call>" not in left and left.startswith("I'll check")


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASS")
