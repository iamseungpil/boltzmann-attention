# -*- coding: utf-8 -*-
"""★§T-12 — TERSE 프로브의 **출력 스키마**. 없으면 게이트가 조용히 꺼진다.

라이브 실측(x722 양 팔): `agent_claimprov` 가 스키마 없이 **산문 1,825B** 를 뱉고 상한 512 에
정확히 닿아 잘렸다(`gen=512 **TRUNC**`). 잘린 JSON 은 소비부에서 파스 실패 → `except` →
`if not _cl and not _pd: break` ⇒ **날조-완료 차단이 그 턴에 무력화**된다.
TRUNC 1 ↔ `declaration failed (no-op)` 1 로 **1:1 대응**했다.
"""
import io
import json
import os
import re
import sys

SRC = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_run_gated.py"),
              encoding="utf-8").read()


def t_schema_is_registered_for_claimprov():
    m = re.search(r"_t2_terse_schemas\s*=\s*\{(.{0,1400})", SRC, re.S)
    assert m and "agent_claimprov" in m.group(1)


_HERE = os.path.dirname(os.path.abspath(__file__))


def _schema_item_keys():
    """스키마가 **항목 수준에서 허용하는 키**를 claims/pending 별로 뽑는다."""
    m = re.search(r"_t2_terse_schemas\s*=\s*\{(.{0,2600}?)\n        \}\n", SRC, re.S)
    assert m, "스키마 블록을 못 찾았다"
    body = m.group(1)
    out = {}
    for lst in ("claims", "pending"):
        seg = re.search(r'"%s":\s*\{.*?"properties":\s*\{(.*?)\}\}' % lst, body, re.S)
        assert seg, lst
        out[lst] = set(re.findall(r'"(\w+)":\s*\{"type"', seg.group(1)))
    return out


def _a2_question_keys():
    """A2 가 모델에게 **실제로 요구하는** 키 — gate.json 이 계약의 출발점이다."""
    gj = io.open(os.path.join(_HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8").read()
    q = json.loads(gj)["claim_prov"]["question"]
    out = {}
    for lst in ("claims", "pending"):
        seg = re.search(r'\\?"%s\\?":\s*\[\{(.*?)\}\]' % lst, q, re.S)
        assert seg, lst
        out[lst] = set(re.findall(r'\\?"(\w+)\\?":', seg.group(1)))
    return out


def _consumer_item_keys():
    """소비부가 claim/pending **항목에서 읽는 키**를 코드에서 뽑는다(리터럴 단언 금지)."""
    gp = io.open(os.path.join(_HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    i = gp.index("def _desc3(cc):")
    seg = gp[i:i + 2400]                      # _desc3 + 그 아래 feedback 조립부
    keys = set(re.findall(r'(?:\(?[cp]\b(?: or \{\})?\)?)\.get\("(\w+)"\)', seg))
    assert keys, "소비부에서 키를 못 뽑았다 — 계약 위치가 바뀌었다"
    return keys, gp


def t_schema_matches_what_the_consumer_reads():
    """★D8 회귀 검정 (2026-09-05).

    구판은 리터럴 `'"claim"'` 을 단언했다. 그래서 `f6224e26` 이 `what` 을 `claim` 으로 개명해
    전송 문면 **73/73 을 `None: None`** 으로 만들었을 때, 검정이 그 오답을 **통과시킨 게 아니라
    못박았다**. 이제는 어느 낱말도 단언하지 않고 **A2 질문 → 스키마 → 소비부** 세 계약을 서로
    대조한다 — 셋 중 하나만 개명해도 실패한다.
    """
    sch = _schema_item_keys()
    ask = _a2_question_keys()
    con, gp = _consumer_item_keys()

    # ⑴ A2 가 요구하는 키는 스키마가 전부 허용해야 한다 — guided decoding 은 스키마 밖 키의
    #    **생성 자체를 막으므로**, 빠진 키는 「모델이 안 냈다」가 아니라 「우리가 못 내게 했다」다.
    for lst in ("claims", "pending"):
        missing = ask[lst] - sch[lst]
        assert not missing, "A2 질문이 요구하는데 %s 스키마가 막는 키: %s" % (lst, sorted(missing))

    # ⑵ 소비부가 읽는 키는 스키마 안에 있어야 한다 — 아니면 영원히 None 이다.
    for lst in ("claims", "pending"):
        orphan = con - sch[lst]
        assert not orphan, "소비부가 읽는데 %s 스키마에 없는 키: %s" % (lst, sorted(orphan))

    # ⑶ 리스트 계약 자체.
    assert '_j2["claims"]' in gp and '_j2.get("pending")' in gp, "소비부 계약이 바뀌었다"


def t_applied_in_the_terse_branch_not_judge():
    """JUDGE 로 옮기면 상한 8192·사고 4096 이라 ~100콜에 비용이 폭증한다 — 형식만 묶는다."""
    i = SRC.index('_tsch = _t2_terse_schemas.get')
    seg = SRC[i:i + 400]
    assert '"t2terse"' in seg and 'response_format' in seg
    assert '_kw["_t2_terse"] = "TERSE"' in seg


def t_untouched_calls_get_no_schema():
    i = SRC.index('_tsch = _t2_terse_schemas.get')
    assert "if _tsch:" in SRC[i:i + 200], "미등록 프로브는 바이트 동일이어야 한다"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for f in fns:
        f()
        print("ok %s" % f.__name__)
    print("PASS %d/%d" % (len(fns), len(fns)))
