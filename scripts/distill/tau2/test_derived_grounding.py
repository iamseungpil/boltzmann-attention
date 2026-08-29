#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""파생 필드 접지(`_derived_grounded`) 래칫 — 스모크 없이 **초 단위로**. (094 · P1′)

## 왜 (2026-08-29 · `x592_derived_grounding_iso.py`)

`get_interest_correction.params.actual_apy` 는 축자로 파생을 지시하는데(*"Derive it from the
latest MONTHLY INTEREST CREDIT ... monthly credit amount x 12 / principal x 100"*) 접지 술어는
**존재검사**였다. 파생값이 원장에 문자로 없는 것은 정상이므로 **옳은 파생일수록 반드시 드롭**된다.
실물: `bank_t7388_hB2_20260829 task_094#s626729` 에서 옳은 `actual_apy=5.1` 이 msg[28]·[34]·[42]
세 번 드롭됐고(`get_interest_correction -> None` 3회) 모델이 그 값을 **보고서로 써 넣은 뒤**
msg[58] 에서야 통과했다 — 우리 게이트가 정답을 막고 write-세탁만 통과시킨 것이다([[25]]).

## 재료는 **실제 런의 궤적**이다

합성 픽스처가 아니다. 그 sim 들의 메시지를 그대로 얹고 **엔진 함수 자체**(`_derived_grounded`
· `_corpus_texts` · `_val_grounded`)를 호출해 판정을 다시 낸다. 기대값은 gold 가 아니라
**x592 가 이미 낸 표**다(gold 는 어느 술어에도 안 들어간다·[[23]]).

⚠이 검정은 *"이 술어가 점수를 사는가"* 를 판정하지 않는다 — 그건 런이 잰다([[62]]).

실행: PYTHONIOENCODING=utf-8 py -3 test_derived_grounding.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import t2_forensic as F                                          # noqa: E402
import t2_scaffold_get as S                                      # noqa: E402

FAIL = []
A2_COPIES = ["a2/banking_knowledge.gate.json",
             "a2/banking_knowledge.specific.json",
             "a2/split/banking_knowledge.core.json"]
TOOL = "get_interest_correction"


def chk(c, m, extra=""):
    if not c:
        FAIL.append(m)
    print("  %s %s%s" % ("ok  " if c else "FAIL", m, ("  " + str(extra)) if extra else ""))


class _M(object):
    """영속 메시지 dict → 엔진이 기대하는 속성 인터페이스(`_evidence_ctx` 가 보는 것만)."""

    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.id = d.get("id")
        self.requestor = d.get("requestor", "assistant")
        self.tool_calls = [_TC(t) for t in (d.get("tool_calls") or [])]


class _TC(object):
    def __init__(self, d):
        d = F._as_dict(d)
        self.id = d.get("id")
        self.name = F.nameof(d)


class _Orch(object):
    def __init__(self, msgs):
        self._m = [_M(x) for x in msgs if isinstance(x, dict)]
        self.environment = None

    def get_messages(self):
        return self._m


def _scf(a2_path=None):
    """검정 대상 선언을 **A2 에서 읽는다**(코드에 값 복사 금지·[[05]])."""
    d = json.load(io.open(os.path.join(HERE, a2_path or A2_COPIES[1]), encoding="utf-8"))
    for t in d.get("scaffold_get_tools") or []:
        if t.get("name") == TOOL:
            for sf in ((t.get("ground") or {}).get("scalar_fields") or []):
                if sf.get("param") == "actual_apy":
                    return sf, t
    return None, None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("① 배선 — 엔진 안에 있고, 선언이 없으면 거동이 안 바뀐다")
    src = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
    chk("def _derived_grounded(" in src, "`_derived_grounded` 가 엔진에 있다")
    chk("if _derived_grounded(orch, scf, ctx):" in src, "존재검사 실패 **뒤에** 걸린다(우회 아님)")
    chk(S._derived_grounded(_Orch([]), {"param": "x"}, {"x": 1}) is False,
        "`derived_from` 미선언 = False = 종전대로 드롭")
    chk(S._derived_grounded(_Orch([]), {"param": "x", "derived_from": {"op": {"op": "ref",
        "path": "nope"}}}, {"x": 1}) is False, "역산이 None 이면 검산 불가 = 드롭(fail-closed)")

    print("② 선언 — 세 사본이 같고, 역산 op 가 도구 자신의 op 에서 나왔다")
    sigs = []
    for p in A2_COPIES:
        sf, _t = _scf(p)
        sigs.append(json.dumps(sf, ensure_ascii=False, sort_keys=True))
    chk(len(set(sigs)) == 1, "A2 3사본 동일([[24]])", "%d 종" % len(set(sigs)))
    sf, tool = _scf()
    df = (sf or {}).get("derived_from") or {}
    chk(bool(df.get("op")), "`derived_from.op` 선언 실재")
    chk(df.get("corpus") == ["ledger_tools"],
        "코퍼스 = 도구 출력 전용(손님 주장 배제·[[21]])", df.get("corpus"))
    consts = json.dumps(df.get("op"), ensure_ascii=False)
    chk("0.08333333333333333" in consts and "0.01" in consts,
        "상수가 도구 자신의 `op` 에서 왔다(새 도메인 지식 0)")
    chk(bool(sf.get("fail_feedback")), "거부 문면이 무엇을 하면 풀리는지 담는다([[64]])")

    print("③ 술어 — **재료가 들어온 sim**(t7388 s373753)에서 정답을 살리고 오답을 거절한다")
    tag, want = "bank_t7388_hB2_20260829", "task_094#s373753"
    sims = [x for x in F.sims(tag) if F.simtag(x) == want]
    chk(len(sims) == 1, "그 sim 을 읽었다", tag)
    ok_true = ok_false = 0
    if sims:
        msgs = sims[0].get("messages") or []
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            for tc in (m.get("tool_calls") or []):
                if F.nameof(tc) != TOOL:
                    continue
                a = F.argsof(tc)
                try:
                    prin = float(a.get("principal"))
                    proposed = float(a.get("actual_apy"))
                except Exception:
                    continue
                orch = _Orch(msgs[:i])
                # 모델이 실제로 낸 값(전부 오답 — gold 는 이 판정에 안 들어간다)
                if S._derived_grounded(orch, sf, {"principal": prin, "actual_apy": proposed}):
                    ok_false += 1
                # 같은 자리에서 **거래 레코드가 함의하는 값**(408 -> 5.1)
                if S._derived_grounded(orch, sf, {"principal": prin, "actual_apy": 5.1}):
                    ok_true += 1
    chk(ok_true >= 9, "거래 read 이후 옳은 파생이 살아난다", "통과 %d 회" % ok_true)
    chk(ok_false == 0, "모델이 낸 오답은 한 건도 안 통과한다([[57]] 부정통제)",
        "오통과 %d 건" % ok_false)

    print("④ 부정통제 — **재료가 안 들어온 sim** 에서는 이 축이 아무것도 안 산다")
    tag2, want2 = "bank_t7387_hB1_20260829", "task_094#s626729"
    s2 = [x for x in F.sims(tag2) if F.simtag(x) == want2]
    bought = 0
    if s2:
        msgs = s2[0].get("messages") or []
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            for tc in (m.get("tool_calls") or []):
                if F.nameof(tc) != TOOL:
                    continue
                if S._derived_grounded(_Orch(msgs[:i]), sf,
                                       {"principal": 96000, "actual_apy": 5.1}):
                    bought += 1
    chk(bought == 0, "거래 read 가 없으면 통과 0 = `T2_SG_REQREADS_CANON` 과 짝이다",
        "통과 %d 회" % bought)

    print("⑤ 회귀 — 다른 스칼라 필드는 손대지 않았다")
    _sf2, tool2 = _scf()
    others = [x.get("param") for x in ((tool2.get("ground") or {}).get("scalar_fields") or [])
              if x.get("param") != "actual_apy"]
    chk(others == ["principal", "period_start", "period_end"], "나머지 필드 목록 불변", others)
    chk(all("derived_from" not in x
            for x in ((tool2.get("ground") or {}).get("scalar_fields") or [])
            if x.get("param") != "actual_apy"), "파생 선언은 그 한 필드에만 붙었다")

    print()
    if FAIL:
        print("FAILED %d" % len(FAIL))
        for f in FAIL:
            print("  - %s" % f)
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
