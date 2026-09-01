# -*- coding: utf-8 -*-
"""★§T-12 — TERSE 프로브의 **출력 스키마**. 없으면 게이트가 조용히 꺼진다.

라이브 실측(x722 양 팔): `agent_claimprov` 가 스키마 없이 **산문 1,825B** 를 뱉고 상한 512 에
정확히 닿아 잘렸다(`gen=512 **TRUNC**`). 잘린 JSON 은 소비부에서 파스 실패 → `except` →
`if not _cl and not _pd: break` ⇒ **날조-완료 차단이 그 턴에 무력화**된다.
TRUNC 1 ↔ `declaration failed (no-op)` 1 로 **1:1 대응**했다.
"""
import io
import os
import re
import sys

SRC = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_run_gated.py"),
              encoding="utf-8").read()


def t_schema_is_registered_for_claimprov():
    m = re.search(r"_t2_terse_schemas\s*=\s*\{(.{0,1400})", SRC, re.S)
    assert m and "agent_claimprov" in m.group(1)


def t_schema_matches_what_the_consumer_reads():
    """소비부는 `_j2[\"claims\"]`(리스트)·항목의 `tool`·`_j2[\"pending\"]`(리스트)를 읽는다."""
    m = re.search(r"_t2_terse_schemas\s*=\s*\{(.{0,1400})", SRC, re.S)
    body = m.group(1)
    for k in ('"claims"', '"pending"', '"tool"', '"claim"'):
        assert k in body, k
    gp = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_gate_patch.py"),
                 encoding="utf-8").read()
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
