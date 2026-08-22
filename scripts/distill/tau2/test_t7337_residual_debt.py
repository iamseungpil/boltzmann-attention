# -*- coding: utf-8 -*-
"""잔여 부채 3건 검정 (2026-08-22 · t7336 핸드오프 §2 "남은 부채") — 오프라인·모델 0·env 0.

  ① A9 호출부   F8(`T2_ARG_PRODUCERS`) 억제 술어를 정본 `user_tool_value_ready` 로 재배선
  ② OL-55 형제  `T2_STALE_STRIP` 노트도 빈 본문이면 손님 발화 전체가 된다
  ③ 누수        `T2_WRITE_ARG_ENUM` 후보 명단의 `' General '`(= `_general_` 슬러그)

항목마다 **세 칸**을 고정한다([[73]] 루프·G1 검정 관례 동일):
  ⓟ 양성대조 = 수리 **전** 결함이 실재했음을 이 자리에서 재현(구판 술어를 그대로 평가)
  ⓝ 수리 후  = 같은 입력에서 결함이 사라진다
  ⓒ 부정통제 = 레버가 **죽지 않았다**(원 표적은 그대로 잡힌다) · 무엇을 파는지 고정([[70]])

⛔라이브 코드를 부른다 — 술어를 다시 구현하지 않는다([[03b]]).
실행: py -3 test_t7337_residual_debt.py
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP                                          # noqa: E402

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
PRE = io.open(os.path.join(HERE, "t2_prekb_patch.py"), encoding="utf-8").read()
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


def seg(src, anchor, n=4000):
    i = src.find(anchor)
    return src[i:i + n] if i >= 0 else ""


class TC(object):
    def __init__(self, tid, name, arguments=None, requestor="assistant"):
        self.id, self.name, self.arguments = tid, name, (arguments or {})
        self.requestor = requestor

    def model_dump(self):
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


class MSG(object):
    def __init__(self, role, mid=None, content="", error=False, tool_calls=None):
        self.role, self.id, self.content = role, mid, content
        self.error, self.tool_calls = error, tool_calls

    def model_dump(self):
        return {"role": self.role, "id": self.id, "content": self.content,
                "tool_calls": [tc.model_dump() for tc in (self.tool_calls or [])]}


GIVE, UCALL = "give_discoverable_user_tool", "call_discoverable_user_tool"
PROD = "get_direct_deposit_details"


def _old_seen_tools(msgs):
    """구판 `_seen_tools` — 양성대조 전용(수리 전 소스 그대로)."""
    out = set()
    for _m4 in (msgs or []):
        _md4 = _m4.model_dump() if hasattr(_m4, "model_dump") else {}
        for _tc4 in (_md4.get("tool_calls") or []):
            out.add(str(_tc4.get("name") or ""))
            _a4 = _tc4.get("arguments")
            if isinstance(_a4, str):
                out |= {w for w in re.findall(r"[a-z0-9_]+", _a4)}
            elif isinstance(_a4, dict):
                out |= {str(v) for v in _a4.values() if isinstance(v, str)}
    return out


def _boom(_ask):
    raise RuntimeError("regen died")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[① A9 호출부] F8 억제 술어 = '이름이 등장' → '값을 얻음'")
# ══════════════════════════════════════════════════════════════════════════════
# 건네기만 했다: give 인자에 생산자 이름이 실린다 · 손님은 아직 실행 안 함
gave_only = [MSG("assistant", tool_calls=[TC("g1", GIVE, {"discoverable_tool_name": PROD})]),
             MSG("tool", "g1", content="Tool handed to the customer.", error=False)]
# 손님이 실제로 실행했다(디스패처 경유)
ran_disp = gave_only + [MSG("assistant",
                            tool_calls=[TC("u1", UCALL, {"discoverable_tool_name": PROD})])]
# 손님이 직접 호출했다(role=user) — 실측 궤적의 두 번째 형태
ran_direct = gave_only + [MSG("user", tool_calls=[TC("u2", PROD, {})])]

# ⓟ 양성대조: 구판은 **건네기만 해도** 생산자를 seen 으로 봤다 → 넛지 영구 침묵
chk("ⓟ 구판이 give-만-한 궤적에서 생산자를 억제 집합에 넣는다(t7336 발화 0 의 자리)",
    PROD in _old_seen_tools(gave_only))
# ⓝ 수리 후: 건네기만 했으면 값은 아직 없다 → 억제하지 않는다
chk("ⓝ 정본 술어는 give-만-한 궤적에서 생산자를 억제하지 않는다",
    PROD not in GP.user_tool_value_ready(gave_only, GIVE, UCALL))
# ⓒ 부정통제 ①: 실제로 값을 얻었으면 **여전히** 억제(넛지가 시끄러워지지 않는다)
chk("ⓒ 디스패처 경유 실행 → 억제 유지",
    PROD in GP.user_tool_value_ready(ran_disp, GIVE, UCALL))
chk("ⓒ 손님 직접 호출(role=user) → 억제 유지",
    PROD in GP.user_tool_value_ready(ran_direct, GIVE, UCALL))
# ⓟ/ⓝ 구판의 **과폭**: 인자 문자열 토막이 통째로 억제 집합에 들어갔다([[59]] 위반)
noise = [MSG("assistant",
             tool_calls=[TC("x1", "search_kb", {"query": "get_direct_deposit_details"})])]
chk("ⓟ 구판은 인자 JSON 의 토막까지 억제 집합에 넣었다(패턴매칭·[[59]])",
    PROD in _old_seen_tools(noise))
chk("ⓝ 정본 술어는 인자 문자열을 뜯지 않는다",
    PROD not in GP.user_tool_value_ready(noise, GIVE, UCALL))
# 배선
_f8 = seg(PRE, 'os.environ.get("T2_ARG_PRODUCERS") == "1"', 3000)
chk("배선: 호출부가 정본을 부른다(사본 0·[[67]])",
    "import t2_gate_patch as _g9" in _f8 and "_g9.user_tool_value_ready(" in _f8)
chk("배선: 인자 토막 파싱이 억제 경로에서 제거됐다([[59]])",
    're.findall(r"[a-z0-9_]+", _a4)' not in _f8)
chk("배선: 디스패처 미선언 도메인 fallback = 이름 집합만(인자 파싱 0)",
    '_drc.get("give_tool")' in _f8
    and '_seen_tools.add(str(_tc4.get("name") or ""))' in _f8)
chk("⚠[[70]] 무엇을 파는가 명기(F8 이 더 자주 운다)", "무엇을 파는가" in _f8)

# ══════════════════════════════════════════════════════════════════════════════
print("\n[② OL-55 형제] 기계 노트가 손님 발화 전체가 되는 자리 — STALE_STRIP")
# ══════════════════════════════════════════════════════════════════════════════
NOTE, ASK = GP._STALE_NOTE % 3, GP._STALE_NOTE_ASK

# ⓟ 양성대조: 구판은 빈 본문에 노트를 그대로 붙여 **노트가 본문 전체**가 됐다
am_p = MSG("assistant", content="")
am_p.content = (am_p.content or "") + NOTE
chk("ⓟ 구판 자리: 빈 본문 + 노트 = 기계 노트가 손님 발화 전체",
    am_p.content.strip().startswith("[Note:"))

# ⓝ 수리 후: 재생성으로 모델 본문을 받고 노트는 뒤에 붙는다
am_n = MSG("assistant", content="")
r = GP._commit_machine_note(am_n, NOTE, ASK, regen=lambda a: "Here is where things stand.",
                            tag="T2_STALE_NOTE")
chk("ⓝ 빈 본문 → 재생성 본문 + 노트(노트가 전체가 아니다)",
    r == "regen" and am_n.content.startswith("Here is where things stand.")
    and NOTE in am_n.content)
# ⓝ 재생성 실패/빈 응답이면 노트를 **커밋하지 않는다**
am_e = MSG("assistant", content="")
r_e = GP._commit_machine_note(am_e, NOTE, ASK, regen=lambda a: "", tag="T2_STALE_NOTE")
chk("ⓝ 재생성이 비면 노트도 붙이지 않는다(빈 본문 유지)",
    r_e == "empty" and am_e.content == "")
am_x = MSG("assistant", content="")
r_x = GP._commit_machine_note(am_x, NOTE, ASK, regen=_boom, tag="T2_STALE_NOTE")
chk("ⓝ 재생성 예외도 흡수 — 노트가 본문 전체가 되지 않는다",
    r_x == "empty" and am_x.content == "")
am_z = MSG("assistant", content="")
r_z = GP._commit_machine_note(am_z, NOTE, ASK, regen=None, tag="T2_STALE_NOTE")
chk("ⓝ regen=None(구 호출부·단위검정) 도 빈 본문 유지",
    r_z == "empty" and am_z.content == "")
# ⓒ 부정통제 ①: 본문이 있으면 종전대로 뒤에 붙는다(거동 보존)
am_c = MSG("assistant", content="I checked your account.")
r_c = GP._commit_machine_note(am_c, NOTE, ASK, regen=lambda a: "SHOULD NOT BE CALLED",
                              tag="T2_STALE_NOTE")
chk("ⓒ 본문이 있으면 재생성 없이 뒤에 붙인다(거동 보존)",
    r_c == "appended" and am_c.content == "I checked your account." + NOTE)
# ⓒ 부정통제 ②: A15 원본(_commit_block_note)이 정본 위에서 그대로 작동한다
am_b = MSG("assistant", content="")
r_b = GP._commit_block_note(am_b, "[g1] needs authentication", regen=lambda a: "Body.")
chk("ⓒ A15 원본 거동 보존(_BLOCK_NOTE 조립 + 재생성)",
    r_b == "regen" and am_b.content.startswith("Body.")
    and GP._BLOCK_NOTE in am_b.content and "[g1] needs authentication" in am_b.content)
am_b2 = MSG("assistant", content="Prose.")
chk("ⓒ A15 원본: 본문 있으면 append",
    GP._commit_block_note(am_b2, "r", regen=None) == "appended"
    and am_b2.content == "Prose." + GP._BLOCK_NOTE + " (r)")
# 배선
_ss = seg(SRC, 'os.environ.get("T2_STALE_STRIP") == "1"', 3000)
chk("배선: 정본 하나만 존재(사본 0·[[67]])",
    SRC.count("def _commit_machine_note(") == 1 and SRC.count("def _commit_block_note(") == 1)
chk("배선: _commit_block_note 는 정본의 얇은 래퍼",
    "_commit_machine_note(am, _BLOCK_NOTE" in seg(SRC, "def _commit_block_note(", 900))
chk("배선: STALE_STRIP 이 정본을 쓴다", "_commit_machine_note(" in _ss)
chk("배선: 남은 호출이 있으면 도구 호출 턴이라 종전대로 붙인다(재생성 없음)",
    "if _kept:" in _ss and 'am.content = (am.content or "") + _snote' in _ss)
chk("배선: 재생성은 **도구 없이**(새 호출이 게이트를 우회하지 못한다)",
    'call_name="agent_stalenote_body"' in _ss and "tools=None" in _ss
    and '_kw.pop("tools", None)' in _ss)
chk("배선: 노트 문면이 결과를 단언하지 않는다(A1/OL-18 보존)",
    "says nothing about whether the earlier attempt succeeded" in GP._STALE_NOTE
    and "completed" not in GP._STALE_NOTE)
chk("배선: ask 가 [[64]] 를 따른다(무엇이 틀렸나 + 무엇을 하면 풀리나)",
    "were not sent again" in ASK and "Write that message yourself" in ASK
    and "do not emit tool calls" in ASK)
chk("계기: 자리별 태그가 갈린다(포렌식 집계 보존)",
    'tag="T2_STALE_NOTE"' in _ss and 'tag="T2_BLOCK_NOTE"' in SRC)

# ══════════════════════════════════════════════════════════════════════════════
print("\n[③ 누수] WRITE_ARG_ENUM 후보 명단의 ' General '")
# ══════════════════════════════════════════════════════════════════════════════
# ⓟ 양성대조: 구판 명단 생성은 `_general_` 을 그대로 전개해 실었다
subs = {"_general_": 1, "sky_blue_checking_account": 1, "green_fee-free_account": 1}
old_names = sorted(GP._slug_disp(k) for k in subs)
chk("ⓟ 구판 명단에 ' General ' 이 실린다(존재하지 않는 이름·[[25]] 근거원 오염)",
    " General " in old_names, old_names)
# ⓝ 수리 후
new_names = GP._display_slugs(subs)
chk("ⓝ 정본 명단에서 빠진다", " General " not in new_names, new_names)
# ⓒ 부정통제 ①: 실제 제품명은 그대로 — 하이픈 대문자화(FIX-6)도 보존
chk("ⓒ 실제 제품명은 그대로 남는다(FIX-6 하이픈 대문자화 보존)",
    new_names == ["Green Fee-Free Account", "Sky Blue Checking Account"], new_names)
# ⓒ 부정통제 ②: `_general_` 하나뿐인 그룹은 **빈 명단** → fail-open(deny 안 함)
chk("ⓒ 표시명이 하나도 없는 그룹은 빈 명단(fail-open 으로 넘어간다)",
    GP._display_slugs({"_general_": 1}) == [])
# ⓒ 부정통제 ③: 술어는 이름 리터럴 0 — 같은 형상의 다른 키도 걸린다
chk("ⓒ 술어가 닫힌 형상 판정(이름 리터럴 0)",
    GP._display_slugs({"_faq_": 1, "_index_": 1, "a_b": 1}) == ["A B"]
    and "_general_" not in seg(SRC, "def _display_slugs", 1600).split('"""')[-1])
# 실물 A2 로 검산
_hit = []
for _f in ("a2/banking_knowledge.gate.json", "a2/banking_knowledge.specific.json"):
    _p = os.path.join(HERE, _f)
    if not os.path.exists(_p):
        continue
    _di = ((json.load(io.open(_p, encoding="utf-8")).get("policy_ontology") or {})
           .get("doc_index") or {})
    for _g, _s in _di.items():
        if " General " in GP._display_slugs(_s):
            _hit.append((_f, _g))
chk("실물 A2 전수: 어느 그룹의 명단에도 ' General ' 이 없다", not _hit, _hit)
# ⓒ 부정통제 ④ — **거동 동치**: 정본 형상 술어가 구판 리터럴 술어(`s != "_general_"`)와
#   실물 A2 전수에서 같은 답을 낸다. 통합([[67]])이 거동을 바꾸지 않았음을 못박는다.
_tot, _dif = 0, []
for _f in ("a2/banking_knowledge.gate.json", "a2/banking_knowledge.specific.json",
           "a2/split/banking_knowledge.core.json"):
    _p = os.path.join(HERE, _f)
    if not os.path.exists(_p):
        continue
    _di = ((json.load(io.open(_p, encoding="utf-8")).get("policy_ontology") or {})
           .get("doc_index") or {})
    for _g, _s in _di.items():
        _tot += 1
        _old = {k for k in (_s or {}) if k != "_general_"}     # 구판 리터럴 술어
        _new = GP._subject_keys(_s)                            # 정본 형상 술어
        if _old != _new or (not _old) != (not _new):
            _dif.append((_f, _g))
chk("ⓒ 실물 A2 전수에서 구판 리터럴 술어와 **동치**(통합이 거동을 안 바꿨다)",
    _tot > 0 and not _dif, "군 %d개 · 불일치 %d" % (_tot, len(_dif)))
chk("ⓒ 퇴화 군 판정도 정본을 쓴다(A14 거동 보존)",
    GP._degenerate_axes({"doc_index": {"g1": {"_general_": 1},
                                       "g2": {"a_b": 1, "_general_": 1}}}) == {"g1"})

# 배선
_en = seg(SRC, 'os.environ.get("T2_WRITE_ARG_ENUM") == "1"', 6000)
chk("배선: 후보 생성이 정본 술어를 쓴다", "_names = _display_slugs(_subs)" in _en)
chk("배선: fail-open 술어가 **명단** 기준이다(빈 후보로 deny 하지 않는다·[[64]])",
    "if not (_val and _grp and _names):" in _en
    and "if not (_val and _grp and _subs):" not in SRC)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
