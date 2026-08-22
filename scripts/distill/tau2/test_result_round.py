# -*- coding: utf-8 -*-
"""회귀 — 우리 도구가 낸 **금액 표현**이 우리 증거 게이트와 어긋나지 않는다 (`result_round`).

★왜 (2026-08-22 · t7341 093 실시간 포렌식):
  `get_interest_correction` 이 **32.999999999999986** 을 냈다(op 의 `0.08333333333333333`
  = 1/12 근사 탓에 잔차가 마지막 자리에 남는다). 모델은 그것을 통화로 **옳게** 33.0 으로
  반올림해 write 했는데, `T2_WRITE_EVIDENCE` 가
      "the amount_difference (33.0) does not appear in any get_interest_correction tool output"
  로 **10회 반려**했다. 모델은 아무것도 틀리지 않았다 — 도구를 부르고, 값을 읽고, 달러로 접었다.
  **우리 표현 오차가 우리 게이트를 스스로 막은 것**이다([[25]] 우리 도구는 100% 정답 의무:
  출력 결함이 유일한 근거원을 오염시킨다).

⇒ 산수를 고치는 대신 **표현을 접는다**. 반올림은 통화의 정의이고, 접은 값이 곧 우리가 증거로
  내미는 값이어야 한다. 자릿수는 A2 `result_round` 선언뿐이고 엔진 리터럴은 0 이다([[05]]).
  gold 는 참조하지 않았다([[23]]) — 근거는 크레딧 도구의 인자 계약이 **달러 금액**을 요구한다는
  정책 축자다.

⚠[[70]] 무엇을 파는가: 접는 만큼 정밀도를 잃는다. 그래서 **금액을 내는 스칼라 도구에만**
  선언한다 — `get_correct_savings_apy` 는 APY(2.775)라 2자리로 접으면 **값 자체가 바뀌므로**
  건드리지 않는다(근거 없는 확대 금지·[[62]]).

오프라인 전용(모델 0·env 0). 실행: py -3 test_result_round.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LAYERS = ["a2/banking_knowledge.gate.json",
          "a2/banking_knowledge.specific.json",
          "a2/split/banking_knowledge.core.json"]
SG = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


def tools_of(rel):
    d = json.load(io.open(os.path.join(HERE, rel), encoding="utf-8"))
    return {t.get("name"): t for t in (d.get("scaffold_get_tools") or [])}


print("\n[① 실물 재현 — 접기 전 값이 정말 어긋난다]")
# ⓟ 양성대조: 라이브가 낸 바로 그 수. 모델이 통화로 접으면 문자열이 달라진다.
RAW = 32.999999999999986
chk("ⓟ 도구가 낸 값과 모델이 쓴 값의 문자열이 다르다(WEV substring 불일치)",
    str(RAW) != str(33.0) and str(33.0) not in str(RAW), "%r vs %r" % (RAW, 33.0))
# ⓝ 수리 후: 접으면 같아진다
chk("ⓝ 2자리로 접으면 모델이 쓴 값과 일치한다", round(RAW, 2) == 33.0, round(RAW, 2))
# ⓒ 부정통제: 접기가 값을 바꾸지 않는 자리(이미 통화 정밀도)
chk("ⓒ 이미 센트 정밀도인 값은 접어도 그대로", round(30.0, 2) == 30.0 and round(12.34, 2) == 12.34)

print("\n[② A2 선언 — 금액 도구에만, 3층 동일]")
declared = {}
for rel in LAYERS:
    ts = tools_of(rel)
    ic = ts.get("get_interest_correction") or {}
    declared[rel] = ic.get("result_round")
    chk("%-34s get_interest_correction: result_round=2" % rel.split("/")[-1],
        ic.get("result_round") == 2, ic.get("result_round"))
    chk("%-34s 출처 주석이 붙어 있다([[23]] gold 미참조 명기)" % rel.split("/")[-1],
        "gold 미참조" in str(ic.get("_note_result_round") or ""))
    # ⓒ 근거 없는 확대 금지 — APY 도구는 접지 않는다(2.775 가 2.78 로 바뀐다)
    apy = ts.get("get_correct_savings_apy") or {}
    chk("%-34s get_correct_savings_apy 는 접지 않는다(값이 바뀐다)" % rel.split("/")[-1],
        apy.get("result_round") is None)
chk("3층이 같은 자릿수를 선언한다([[24]] 양방향)", len(set(declared.values())) == 1, declared)

print("\n[③ 엔진 배선]")
_blk = SG[SG.find('_res = _c.apply_op(d.get("op"), _ctx)'):][:2200]
chk("배선: 선언이 있을 때만 접는다(미선언 도구 거동 변화 0)",
    '_rr = d.get("result_round")' in _blk and "_rr is not None" in _blk)
chk("배선: 숫자에만 적용(bool 제외·dict/list 반환 도구 무영향)",
    "isinstance(_res, (int, float))" in _blk and "not isinstance(_res, bool)" in _blk)
chk("배선: 자릿수는 A2 에서 읽는다(엔진 리터럴 0·[[05]])",
    "round(float(_res), int(_rr))" in _blk)
chk("배선: 접기가 **범위 게이트보다 앞**이다(두 검사와 반환문이 같은 수를 본다)",
    SG.find('_rr = d.get("result_round")') < SG.find('_rrg = d.get("result_range")'))
chk("계기: 접은 사실을 인쇄한다(포렌식이 셀 수 있게)", "[T2_SG_ROUND]" in _blk)
chk("⚠[[70]] 무엇을 파는가 명기", "무엇을 파는가" in SG[SG.find("★A2 `result_round`"):][:2000]
    or "정밀도" in SG[SG.find("★A2 `result_round`"):][:2000])

print("\n[④ 라이브 술어로 왕복 — 접은 값이 WEV 를 통과하는 형태인가]")
# WEV 는 write 인자의 값이 도구 출력 문자열에 **부분문자열로** 있는지 본다.
_tool_out_before = "Correction amount = principal x (expected-actual)/100 / 12 = %r." % RAW
_tool_out_after = "Correction amount = principal x (expected-actual)/100 / 12 = %r." % round(RAW, 2)
chk("ⓟ 접기 전 출력에는 모델이 쓴 '33.0' 이 없다", "33.0" not in _tool_out_before,
    _tool_out_before[-40:])
chk("ⓝ 접은 출력에는 '33.0' 이 축자로 있다", "33.0" in _tool_out_after, _tool_out_after[-40:])

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
