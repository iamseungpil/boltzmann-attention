# -*- coding: utf-8 -*-
"""회귀+신규 검정: **축-소진 재무장** (`T2_SEARCH_REARM`·2026-08-21·T7336 016 포렌식 처방 1).

무엇을 막는 검정인가 —
 ⒜ **군 단위 영구 잠금(016형)**: 결정문 배달 뒤 대화가 **다른 계열**을 축자로 확정했는데
    재요청이 전부 침묵 — 요건 문서 전달 경로가 구조적으로 소멸(정본
    `T7336_FORENSIC_016_2026_08_21.md` §레버 대조).
 ⒝ **정당한 침묵 훼손**: **같은 계열** 재수요는 여전히 침묵해야 한다(재요청 루프 방지 보존).
 ⒞ **플래그 OFF 거동 변화**: OFF 면 종전과 동일(침묵) — 진행 중 본런 보존.
 ⒟ **포함-등장 오인**: 짧은 계열명이 긴 계열명 **안에서만** 나오면 재수요가 아니다
    (`groups_in` 정본의 억제를 전 색인 우주로 쓰는지).
 ⒠ **도구 덤프 오인**: tool 출력에만 나온 계열명은 수요가 아니다(레코드 덤프는 전 계열명을
    담을 수 있다).
 ⒡ **델타 초과 배달**: 재무장 배달에 기배달 계열·타 계열 문서가 실리면 안 된다(만료 제외는
    이유와 함께 남는다·C327).

오프라인 전용(LLM 0 — formalize 두 자리는 고정 스텁·서브 호출 없음).
실행: py -3 test_search_rearm.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_search as S                                            # noqa: E402
import t2_ledger as L                                            # noqa: E402
import t2_gate_patch as GP                                       # noqa: E402


# ── tau2 하네스 스텁 (오프라인 전용) ────────────────────────────────────────────────
# `_search_material` 은 함수 안에서 `tau2.agent.llm_agent`/`UserMessage` 를 import 하지만
# 이 검정은 formalize 두 자리를 스텁으로 대체하므로 그 심볼이 **호출되지 않는다**.
# 로컬(하네스 미설치)에서도 돌도록 모듈 뼈대만 세운다 — 판단 대역은 없다.
def _stub_mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


if "tau2" not in sys.modules:
    _t = _stub_mod("tau2")
    _t.agent = _stub_mod("tau2.agent")
    _t.agent.llm_agent = _stub_mod("tau2.agent.llm_agent")
    _t.data_model = _stub_mod("tau2.data_model")
    _t.data_model.message = _stub_mod("tau2.data_model.message",
                                      UserMessage=type("UserMessage", (), {}))

FAILED = []


def chk(c, label):
    print(("  OK   " if c else "  FAIL ") + label)
    if not c:
        FAILED.append(label)


# ── 픽스처 (중립 어휘·색 이름 관용구 = test_search_agent 와 동일 계열) ──────────────────
A2 = {"policy_ontology": {
    "doc_index": {
        "g": {"sky_blue": ["doc_g_sky_001"],
              "lime_green": ["doc_g_lime_010", "doc_g_lime_011"],
              "beige": ["doc_g_beige_001"],
              "_general_": []},
        "biz_g": {"big_sky_blue": ["doc_biz_big_sky_001"], "_general_": []}},
    "doc_windows": [{"doc": "doc_g_lime_010", "from": "2025-09-01", "to": "2025-10-31"}],
    "group_prompt": "{groups}{text}",
    "doc_decide_prompt": "{ask}{material}",
    "decided_by_docs_text": "decided: {choice}"},
    "ledger_metrics": [{"now_prompt": "x", "now_tool": "clock",
                        "date_formats": ["%m/%d/%Y"]}]}

CORPUS = {"doc_g_sky_001": "Sky Blue: APY 1.25%.",
          "doc_g_lime_010": "Lime Green (old): requires 500 within 30 days.",
          "doc_g_lime_011": "Lime Green: requires 750 within 60 days.",
          "doc_g_beige_001": "Beige: no requirement.",
          "doc_biz_big_sky_001": "Big Sky Blue: enterprise only."}


class _Pipe(object):
    state = {"doc_content_map": dict(CORPUS)}


class _Tools(object):
    _kb_pipeline = _Pipe()


class _Env(object):
    tools = _Tools()


def _msg(role, content):
    return types.SimpleNamespace(role=role, content=content, tool_calls=None, id=None)


def _agent(done=("g",), served=None, served_at=None):
    a = types.SimpleNamespace(llm=None, llm_args={},
                              _t2_orch=types.SimpleNamespace(environment=_Env()))
    a._t2_search_done = set(done)
    a._t2_search_served = {k: set(v) for k, v in (served or {}).items()}
    a._t2_search_served_at = dict(served_at or {})
    return a


# formalize 두 자리는 고정 스텁 — LLM 0. 군 형식화·시계 형식화의 **판단**은 이 검정의
# 대상이 아니다(각자 검정이 따로 있다). 여기서는 소진-재무장 술어와 배달 경로만 잰다.
S.formalize_groups = lambda *a, **k: ["g"]
L.formalize_now = lambda *a, **k: "11/14/2025"

BASE = [_msg("user", "I want the newest plan."),
        _msg("assistant", "Let me check."),
        _msg("user", "Thanks.")]

print("\n§1 배달 이력 부기 — 결정문/본문이 덮은 계열(닫힌 집합 대조)")
po = A2["policy_ontology"]
chk(GP._served_subjects(po, "g", decided="decided: Sky Blue") == {"sky_blue"},
    "결정문 배달 → 표시명이 든 계열만 덮인다")
chk(GP._served_subjects(po, "g", delivered="[doc_g_lime_011]\nLime Green ...")
    == {"lime_green"}, "본문 배달 → 실린 문서 id 헤더의 계열만 덮인다")
chk(GP._served_subjects(po, "없는군") == set(), "모르는 군은 빈 집합(예외 아님)")

print("\n§2 재무장 술어 — 배달 이후 user/assistant 발화의 신규 계열 축자 등장만 연다")
srv = {"g": {"sky_blue"}}
msgs = BASE + [_msg("assistant", "The record shows a Lime Green plan.")]
chk(GP._rearm_subjects(_agent(served=srv, served_at={"g": 2}), po, ["g"], {"g"}, msgs)
    == ("g", ["lime_green"]), "⒜ 신규 계열 재수요 → (군, 계열) 반환")
msgs2 = BASE + [_msg("assistant", "Sky Blue again, as before.")]
chk(GP._rearm_subjects(_agent(served=srv, served_at={"g": 2}), po, ["g"], {"g"}, msgs2)
    == (None, None), "⒝ 같은 계열 재수요 → 침묵 유지")
msgs3 = BASE + [_msg("assistant", "We should check Big Sky Blue.")]
chk(GP._rearm_subjects(_agent(served=srv, served_at={"g": 2}), po, ["g"], {"g"}, msgs3)
    == (None, None), "⒟ 긴 이름 안의 포함 등장(Sky Blue⊂Big Sky Blue)은 재수요가 아니다")
msgs4 = BASE + [_msg("tool", "row 1: Lime Green; row 2: Beige")]
chk(GP._rearm_subjects(_agent(served=srv, served_at={"g": 2}), po, ["g"], {"g"}, msgs4)
    == (None, None), "⒠ 도구 출력만의 계열명은 수요가 아니다")
chk(GP._rearm_subjects(_agent(served=srv, served_at={"g": 4}), po, ["g"], {"g"}, msgs)
    == (None, None), "배달 **이전** 등장은 세지 않는다(시점 창)")
chk(GP._rearm_subjects(_agent(served=srv, served_at={}), po, ["g"], {"g"}, msgs)
    == (None, None), "배달 이력이 없는 군은 열지 않는다([[25]])")

print("\n§3 진입점 통합 — 016형 재현 (첫 배달=결정문, 이후 신규 계열 확정)")
for k in ("T2_SEARCH_REARM", "T2_PROCEED_DOCBODY", "T2_DOCS_AT_WRITE",
          "T2_SUB_REQUIREMENT", "T2_VERDICT_CARRY", "T2_ELIG_LINE"):
    os.environ.pop(k, None)
ag = _agent(served={"g": {"sky_blue"}}, served_at={"g": 2})
live = BASE + [_msg("assistant", "The record shows a Lime Green plan.")]
out_off = GP._search_material(ag, A2, live)
chk(out_off == "", "⒞ 플래그 OFF → 종전 거동(침묵) 그대로")
os.environ["T2_SEARCH_REARM"] = "1"
try:
    out_b = GP._search_material(ag, A2, live)
    chk("requires 750 within 60 days" in out_b,
        "⒜ 재무장 → 신규 계열의 요건 문서가 배달된다")
    chk("[doc_g_lime_011]" in out_b, "배달은 선언 id 의 본문(doc-only)이다")
    chk("doc_g_sky_001" not in out_b and "Beige: no requirement" not in out_b,
        "⒡ 기배달 계열·미수요 계열 문서는 실리지 않는다(델타만)")
    chk("Excluded as out of date" in out_b and "doc_g_lime_010" in out_b,
        "만료 문서는 이유와 함께 빠진다(C327·drop_expired 경유)")
    chk("lime_green" in ag._t2_search_served.get("g", set()),
        "배달된 계열이 소진 키에 적힌다 — (군, 계열) 1회")
    out_b2 = GP._search_material(ag, A2, live)
    chk(out_b2 == "", "⒝ 같은 계열 재수요 → 다시 침묵(루프 방지 보존)")
    live2 = live + [_msg("assistant", "Or maybe the Beige plan fits.")]
    out_b3 = GP._search_material(ag, A2, live2)
    chk("[doc_g_beige_001]" in out_b3 and "doc_g_lime_011" not in out_b3,
        "다음 신규 계열은 다음 재수요가 연다(각 1회·델타만)")
    out_b4 = GP._search_material(ag, A2, live2)
    chk(out_b4 == "", "전 계열 소진 후 → 침묵")
finally:
    os.environ.pop("T2_SEARCH_REARM", None)

print("\n§4 정본 라이브러리 — `material_for(subjects=…)` 델타 경로 ([[67]] 정본에 추가)")
mat, info = S.material_for(A2, "g", corpus=CORPUS, now="2025-11-14",
                           subjects=["lime_green"], general=False, windowed="none")
chk(info["kept"] == 1 and info["dropped"] == ["doc_g_lime_010"],
    "계열로 좁혀 읽고 만료를 뺀다 (%s)" % info)
chk("doc_g_sky_001" not in mat, "타 계열 문서가 새지 않는다")
mat0, info0 = S.material_for(A2, "g", corpus=CORPUS, now="2025-11-14")
chk(info0["kept"] == 3, "기본 인자는 종전 거동 그대로(전체 군·%s)" % info0["kept"])

print("\n%s  (%d/%d)" % ("FAIL" if FAILED else "ALL PASS", 21 - len(FAILED), 21))
sys.exit(1 if FAILED else 0)
