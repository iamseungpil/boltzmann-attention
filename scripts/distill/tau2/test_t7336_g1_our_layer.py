# -*- coding: utf-8 -*-
"""G1 검정 (2026-08-22·t7336 마스터 §6.1 A1·A2·A3·A5·A13·A15) — 우리 층 거짓 발화·자기차단.

항목마다 **세 칸**을 고정한다:
  ⓟ 양성대조 = 수리 **전** 결함이 실재했음을 이 자리에서 재현(구판 술어·구판 문면·구판 슬라이스)
  ⓝ 수리 후  = 같은 입력에서 결함이 사라진다
  ⓒ 부정통제 = 레버가 **죽지 않았다**(원 표적은 그대로 잡힌다) · 파는 것이 무엇인지 고정

⛔이 검정은 라이브 코드를 부른다 — 술어를 다시 구현하지 않는다([[03b]]). 인라인 폐포 안의
  조건(A5)은 **소스에서 조건 문자열을 뽑아 그대로 평가**한다(사본이 아니라 그 코드 자체).

오프라인 전용(LLM·env·서버 불요). 실행: py -3 test_t7336_g1_our_layer.py
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
import t2_resolve as RZ                                             # noqa: E402

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


def seg(anchor, n=4000):
    """소스에서 anchor 이후 n자 — 배선 생존 검사용([[67]] 0단계)."""
    i = SRC.find(anchor)
    return SRC[i:i + n] if i >= 0 else ""


class TC(object):
    def __init__(self, tid, name, arguments=None):
        self.id, self.name, self.arguments = tid, name, (arguments or {})


class MSG(object):
    def __init__(self, role, mid=None, content="", error=False, tool_calls=None):
        self.role, self.id, self.content = role, mid, content
        self.error, self.tool_calls = error, tool_calls


class FakeTk(object):
    def __init__(self, disc, allt=None):
        self._d = set(disc)
        self.tools = {k: object() for k in (allt if allt is not None else disc)}

    def get_discoverable_tools(self):
        return set(self._d)


class FakeEnv(object):
    def __init__(self, agent_disc=(), user_disc=(), user_all=None):
        self.tools = FakeTk(agent_disc)
        self.user_tools = FakeTk(user_disc, user_all)


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A1 / OL-17·OL-18] _stale_call_ids 에러-형상 게이트 + 노트 사실화")
# ══════════════════════════════════════════════════════════════════════════════
WT = {"apply_checking_account_credit"}
_args = {"account_id": "acc_evergreen_01", "amount": 27.0}
_fail_tm = MSG("tool", "c1", content="Error: amount must be greater than 0", error=False)
hist_fail = [MSG("assistant", tool_calls=[TC("c1", "apply_checking_account_credit", _args)]),
             _fail_tm]
am_retry = MSG("assistant", tool_calls=[TC("c2", "apply_checking_account_credit", _args)])

# ⓟ 양성대조: env 는 에러를 **플래그 없이 content 로** 준다 — 구판 술어(`not m.error`)는 성공으로 봤다
chk("ⓟ 구판 술어(`not m.error`)가 실패 결과를 '성공'으로 판정",
    (not getattr(_fail_tm, "error", False)) is True
    and str(_fail_tm.content).lstrip().startswith("Error"))
# ⓝ 수리 후: 정본 술어가 실패로 보고, 정당한 재시도가 살아난다
chk("ⓝ 정본 술어 `_result_ok` = 실패", RZ._result_ok(_fail_tm) is False)
chk("ⓝ 실패한 write 의 동일-인자 재시도가 strip 되지 않는다(085#1 ×8 의 자리)",
    GP._stale_call_ids(am_retry, hist_fail, WT) == set())
# ⓒ 부정통제 ①: 진짜 성공한 write 의 재호출은 **여전히** strip (규칙② 생존)
hist_ok = [MSG("assistant", tool_calls=[TC("c1", "apply_checking_account_credit", _args)]),
           MSG("tool", "c1", content="Credit of $27.00 applied to acc_evergreen_01.", error=False)]
chk("ⓒ 성공한 write 재호출은 그대로 strip(규칙② 생존)",
    len(GP._stale_call_ids(am_retry, hist_ok, WT)) == 1)
# ⓒ 부정통제 ②: 같은 턴 완전중복(규칙①)은 이력과 무관하게 strip
_d1, _d2 = TC("d1", "get_bank_account", {"x": 1}), TC("d2", "get_bank_account", {"x": 1})
chk("ⓒ 같은 턴 완전중복은 그대로 strip(규칙① 생존)",
    GP._stale_call_ids(MSG("assistant", tool_calls=[_d1, _d2]), [], WT) == {id(_d2)})
# ⓒ 부정통제 ③: 우리 deny 채널(error=True)도 성공 아님 — 구판·신판 모두 동의
chk("ⓒ error=True 도 성공 아님(구판과 동일 방향)",
    RZ._result_ok(MSG("tool", "c1", content="ok", error=True)) is False)
chk("배선: `_stale_call_ids` 가 정본 술어를 재사용(사본 0·[[67]])",
    "import t2_resolve as _rz_ok" in seg("def _stale_call_ids", 1600)
    and "_rz_ok._result_ok" in seg("def _stale_call_ids", 1600))
chk("배선: import 실패는 fail-open(규칙② 비활성 — 모름은 안 지운다)",
    "if _ok is not None else set()" in seg("def _stale_call_ids", 1600))

# ── OL-18 노트 문면 ──────────────────────────────────────────────────────────
_note_blk = seg('os.environ.get("T2_STALE_STRIP") == "1"', 2600)
# ★2026-08-22 (OL-55 형제): 노트가 호출부 인라인에서 **정본 상수 `_STALE_NOTE`** 로 옮겨졌다
#   (빈 본문이면 `_commit_machine_note` 가 재생성을 태우는 자리와 문면을 공유해야 하므로).
#   같은 주장을 **옮긴 자리에서** 그대로 검사한다 — 문면 자체는 한 글자도 바뀌지 않았다.
_note_lit = GP._STALE_NOTE
chk("ⓟ 구판 한국어·거짓 노트가 커밋 경로에서 사라졌다(축자 인용은 주석에만)",
    all(ln.lstrip().startswith("#") for ln in SRC.splitlines() if "중복 호출 제거" in ln))
chk("ⓝ 노트는 영어다(C125 규칙·같은 함수 안에 축자로 존재)",
    bool(_note_lit) and not re.search(r"[가-힣]", _note_lit), _note_lit[:80])
chk("ⓝ 노트가 '이미 완료'를 단언하지 않는다([[25]])",
    "already" not in _note_lit.lower() and "completed" not in _note_lit.lower(), _note_lit[:120])
chk("ⓝ 노트는 **한 일**만 말한다(안 보냄)", "were not sent" in _note_lit)
chk("ⓝ 노트가 다음 행동을 지목한다([[64]])",
    "re-read the tool results" in _note_lit and "says nothing about whether" in _note_lit)
chk("[[70]] 계측 의무가 수정 자리에 적혀 있다(dropped ↔ 동일-인자 재호출 짝)",
    "무엇을 파는가" in _note_blk and "DUP" in seg("def _stale_call_ids", 1600))


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A2 / OL-21] _evs 를 **성공한 호출**로 좁힘")
# ══════════════════════════════════════════════════════════════════════════════
msgs_led = [
    MSG("assistant", tool_calls=[TC("e1", "apply_checking_account_credit", _args)]),
    MSG("tool", "e1", content="Error: Account acc_evergreen_01 not found", error=False),
    MSG("assistant", tool_calls=[TC("e2", "get_bank_account", {"id": "acc_1"})]),
    MSG("tool", "e2", content='{"balance": 100}', error=False),
    MSG("assistant", tool_calls=[TC("e3", "record_update", {"k": 1})]),          # 짝 없음
    MSG("assistant", tool_calls=[TC("e4", "file_dispute", {"k": 2})]),
    MSG("tool", "e4", content="denied by policy gate", error=True),
]
names, dropped = GP._ledger_event_names(msgs_led)
chk("ⓟ 그 호출은 궤적에 실재한다 — 구판은 **이름만** 보고 원장에 넣었다",
    any(getattr(t, "name", "") == "apply_checking_account_credit"
        for m in msgs_led for t in (m.tool_calls or [])))
chk("ⓝ env 가 거부한 호출은 원장에서 빠진다", "apply_checking_account_credit" not in names)
chk("ⓝ 우리 deny(error=True)도 원장 아님", "file_dispute" not in names)
chk("ⓒ 성공한 호출은 그대로 원장", "get_bank_account" in names)
chk("ⓒ 짝을 못 찾은 호출은 남긴다(fail-open — 실패를 *증명한* 것만 뺀다·[[25]])",
    "record_update" in names)
chk("계기: 제외 목록이 사유와 함께 반환된다([[70]])",
    sorted(d[0] for d in dropped) == ["apply_checking_account_credit", "file_dispute"], dropped)
chk("배선: 호출부가 `_ledger_event_names` 를 쓴다(구판 인라인 루프 제거)",
    "_evs, _evs_drop = _ledger_event_names(state.messages)" in SRC
    and "_evs.add(_eff_tool_name(_tc3))" not in SRC)
chk("배선: 정본 술어 재사용(사본 0·[[67]])",
    "_rz_ev._result_ok" in seg("def _ledger_event_names", 2200))
chk("[[70]] 계측 의무가 적혀 있다(unbacked ↔ kind-index rescued 짝)",
    "unbacked" in seg("def _ledger_event_names", 2200)
    and "무엇을 파는가" in seg("def _ledger_event_names", 2200))


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A3 / OL-19] T2_UNAVAIL_PROMISE — 원장-실재 전제 + 구절 분할")
# ══════════════════════════════════════════════════════════════════════════════
KNOWN = {"apply_checking_account_credit", "transfer_to_human_agents", "get_bank_account"}
DISC = {"approve_credit_limit_increase_5847"}
traj = [MSG("user", content="please credit the ATM fees"),
        MSG("assistant", content="I will apply_checking_account_credit_9001 with the widget total, "
                                 "then approve_credit_limit_increase_5847, and send_otp_sms to you.",
            tool_calls=[TC("t1", "get_bank_account", {"account_id": "acc_evergreen_01"})]),
        MSG("tool", "t1", content='{"balance": 100}')]
LED = GP._ledger_text(traj)

chk("_ledger_text: content·tool_call 이름·인자값을 모두 담는다",
    "send_otp_sms" in LED and "get_bank_account" in LED and "acc_evergreen_01" in LED)

# ⓟ/ⓝ 유령 이름(궤적 0회·우리 서브 산출)
ghost = [{"tool": "apply_credits_to_account_1234"}]
chk("ⓟ 구판 거동(ledger_text 미전달) = 유령 이름에 '없다'를 통보",
    len(GP._unavailable_promises(ghost, KNOWN, discoverable=DISC)[0]) == 1)
chk("ⓟ 그 이름은 궤적에 **0회** 등장", "apply_credits_to_account_1234" not in LED)
chk("ⓝ 원장-실재 전제 → 침묵",
    GP._unavailable_promises(ghost, KNOWN, discoverable=DISC, ledger_text=LED) == ([], []))

# ⓟ/ⓝ 구절 분할
ph_with = [{"tool": "apply_checking_account_credit_9001 with the widget total"}]
ph_paren = [{"tool": "transfer_to_human_agents(summary)"}]
chk("ⓟ 구판 분할자(`[,;/]| or `)로는 구절이 통째 한 이름이 된다",
    all(re.sub(r"_\d+$", "", x.strip()) not in KNOWN
        for x in re.split(r"[,;/]| or ", ph_with[0]["tool"]) if x.strip()))
chk("ⓝ `A with B` 분할 → 보유 도구 인식 → 침묵",
    GP._unavailable_promises(ph_with, KNOWN, discoverable=DISC, ledger_text=LED) == ([], []))
chk("ⓝ `A(B)` 분할 → 침묵",
    GP._unavailable_promises(ph_paren, KNOWN, discoverable=DISC, ledger_text=LED) == ([], []))

# ⓟ/ⓝ 순서 회귀 — 원장 게이트가 `known` 검사보다 **앞**에 오면 새 거짓 발화가 생긴다
LED_partial = "the widget total is 27.00"          # A 는 없고 B 만 있는 원장
chk("ⓟ 순서 결함의 전제: 원장에 A 는 없고 B 만 있다",
    "apply_checking_account_credit_9001" not in LED_partial and "widget" in LED_partial)
chk("ⓝ 보유 판정이 원장 게이트보다 **먼저** — 그래도 침묵(단조 억제)",
    GP._unavailable_promises(ph_with, KNOWN, discoverable=DISC,
                             ledger_text=LED_partial) == ([], []))

# ⓒ 부정통제 — 레버는 죽지 않았다
real_ghost = [{"tool": "send_otp_sms"}]
chk("ⓒ 모델이 궤적에서 실제로 약속한 미보유 기능은 그대로 발화(C207/C2-a 원 표적 생존)",
    len(GP._unavailable_promises(real_ghost, KNOWN, discoverable=DISC, ledger_text=LED)[0]) == 1)
locked_p = [{"tool": "approve_credit_limit_increase_5847"}]
chk("ⓒ 잠금(discoverable) 분기 보존 — '없다'가 아니라 locked",
    GP._unavailable_promises(locked_p, KNOWN, discoverable=DISC, ledger_text=LED)
    == ([], locked_p))
chk("ⓒ 센티널(`omit`)은 여전히 판정 제외",
    GP._unavailable_promises([{"tool": "omit"}], KNOWN, discoverable=DISC,
                             ledger_text=LED) == ([], []))
chk("ⓒ `tool` 미선언 항목은 판정하지 않는다(구판 하위호환)",
    GP._unavailable_promises([{"what": "something"}], KNOWN, ledger_text=LED) == ([], []))
chk("배선: 호출부가 원장을 **서브가 본 그대로**(`work + [am]`) 만든다",
    "_led3 = _ledger_text(list(work) + [am])" in SRC and "ledger_text=_led3" in SRC)
chk("계기: 원장-실재 전제가 몇 건을 침묵시켰나를 인쇄한다([[70]])",
    "ledger-absent silenced" in SRC)


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A5 / OL-01] T2_UNLOCK_PROV 출처 집합에 env 레지스트리")
# ══════════════════════════════════════════════════════════════════════════════
chk("_agent_discoverable: env 레지스트리를 읽는다(리터럴 0)",
    GP._agent_discoverable(FakeEnv(agent_disc={"approve_credit_limit_increase_5847"}))
    == {"approve_credit_limit_increase_5847"})
chk("_agent_discoverable: 조회 실패는 fail-open(빈 집합)",
    GP._agent_discoverable(None) == set() and GP._agent_discoverable(object()) == set())

# ── deny 조건을 **소스에서 뽑아 그대로** 평가한다(사본 아님·[[03b]]) ──
_m5 = re.search(r'if \(_uv in getattr\(self, "_t2_unknown_bl", set\(\)\)\s*\n(.*?)\):\s*\n',
                SRC, re.S)
chk("소스에서 UNLOCK_PROV deny 조건을 추출", bool(_m5))
if _m5:
    _cond = ("(_uv in _BL\n" + _m5.group(1) + ")").replace("\n", " ")
    _ns = dict(_BL=set(), _ctx2="", _ours2=set(), _reg2=set(),
               _uv="approve_credit_limit_increase_5847")
    chk("ⓟ 구판(레지스트리 미조회) = 실재 gold 이름을 'unprovenanced' 로 deny (오차단 3/4)",
        eval(_cond, {}, dict(_ns)) is True)                                  # noqa: S307
    _ns_fix = dict(_ns, _reg2={"approve_credit_limit_increase_5847"})
    chk("ⓝ 레지스트리 실재 = 출처 있음 → deny 안 함",
        eval(_cond, {}, _ns_fix) is False)                                   # noqa: S307
    _ns_hall = dict(_ns_fix, _uv="approve_credit_limit_increase_9999")
    chk("ⓒ 레지스트리에 없는 접미사-환각은 **그대로** deny(원 표적 생존)",
        eval(_cond, {}, _ns_hall) is True)                                   # noqa: S307
    _ns_bl = dict(_ns_fix, _BL={"approve_credit_limit_increase_5847"})
    chk("ⓒ env-거부 이력(`_t2_unknown_bl`)이 레지스트리보다 **먼저** — 여전히 deny",
        eval(_cond, {}, _ns_bl) is True)                                     # noqa: S307
    _ns_ctx = dict(_ns, _ctx2="…approve_credit_limit_increase_5847…")
    chk("ⓒ 대화-근거(ctx) 경로 불변", eval(_cond, {}, _ns_ctx) is False)      # noqa: S307
_a5 = seg("★A5/OL-01", 2600)
chk("배선: `_reg2` 를 `_agent_discoverable` 로 조회(사본 0)", "_reg2 = _agent_discoverable(" in _a5)
chk("배선: 조회 실패 fail-open(구판 거동)", "_reg2 = set()" in _a5)
chk("계기: `registry-provenanced` 를 인쇄한다", "registry-provenanced" in _a5)
chk("⚠[[70]] **판다**가 주석에 명기 — 엉뚱한 이름의 unlock 통과 + 4칸 계측",
    "무엇을 파는가" in _a5 and "over-action" in _a5 and "T2_PROV_OURS=1↔0" in _a5)


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A13 / OL-05] feedback_user_tool_is_agents — 부정 존재 단언 삭제 + 선조회 + 병기")
# ══════════════════════════════════════════════════════════════════════════════
_env13 = FakeEnv(agent_disc={"open_bank_account_4821", "record_deposit_4102"},
                 user_disc={"deposit_check_3847", "get_card_last_4_digits"},
                 user_all={"deposit_check_3847", "get_card_last_4_digits", "pay_bill_1122"})
chk("_user_discoverable: 손님-측 discoverable",
    GP._user_discoverable(_env13) == {"deposit_check_3847", "get_card_last_4_digits"})
chk("_user_all_tools ⊋ discoverable(둘의 구분이 실재)",
    "pay_bill_1122" in GP._user_all_tools(_env13))
chk("_user_*: 조회 실패는 fail-open",
    GP._user_discoverable(None) == set() and GP._user_all_tools(None) == set())
_ureg = GP._user_discoverable(_env13) | GP._user_all_tools(_env13)
chk("ⓟ 구판 전제: 손님 도구 `deposit_check_3847` 은 **실재**한다", "deposit_check_3847" in _ureg)
chk("ⓟ 구판 발화 근거: `_tok_overlap` 은 토큰 1개만 겹쳐도 에이전트 항목을 돌려준다",
    GP._tok_overlap("deposit_check", {"open_bank_account_4821", "record_deposit_4102"})
    == ["record_deposit_4102"])
chk("ⓝ 손님-측 선조회가 후보를 찾는다 → 소유권 주장을 접는다",
    GP._tok_overlap("deposit_check", _ureg) == ["deposit_check_3847"])
chk("ⓒ 손님-측에 후보가 없으면 종전대로 소유권 문면(x298 B_OWN 6/8 자리 보존)",
    GP._tok_overlap("open_account", _ureg) == []
    and GP._tok_overlap("open_account", {"open_bank_account_4821"}) == ["open_bank_account_4821"])

_dnc = {}
for _lay in ("specific", "gate"):
    _dnc[_lay] = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.%s.json" % _lay),
                                   encoding="utf-8"))["discoverable_name_check"]
chk("[[24]] A3 두 층 json-등가(`discoverable_name_check` 전체)", _dnc["specific"] == _dnc["gate"])
_own_fb, _lst_fb = (_dnc["gate"].get("feedback_user_tool_is_agents") or "",
                    _dnc["gate"].get("feedback_user_registry_listing") or "")
chk("ⓟ 구판 부정 존재 단언이 두 층 **발화 문면**에서 사라졌다(축자 인용은 `_note_` 에만)",
    all("there is no customer-side tool" not in str(v)
        for lay in _dnc.values() for k, v in lay.items() if not k.startswith("_note")))
chk("ⓝ `feedback_user_registry_listing` 신설(두 층·치환 자리 완비)",
    bool(_lst_fb) and "{name}" in _lst_fb and "{names}" in _lst_fb)
chk("ⓝ 새 문면도 부정 존재를 단언하지 않는다([[25]])",
    "is not a tool" not in _lst_fb and "no customer-side" not in _lst_fb, _lst_fb[:90])
chk("ⓝ [[64]] 두 칸: 무엇이 틀렸나 + 무엇을 하면 풀리나(양쪽 문면)",
    ("was not handed to the customer" in _own_fb and "unlock it and call it yourself" in _own_fb)
    and ("hand the customer that exact name" in _lst_fb
         and "unlock it and call it yourself" in _lst_fb))
chk("출처: env 레지스트리 뿐 — 정책·gold 경유 0을 `_note_` 가 축자로 댄다([[23]])",
    "env 레지스트리" in (_dnc["gate"].get("_note_feedback_user_registry_listing") or ""))
_a13 = seg("★A13/OL-05", 4200)
chk("배선: 손님-측 레지스트리 **선조회**", "_user_discoverable(_uenv8)" in _a13
    and "_user_all_tools(_uenv8)" in _a13)
chk("배선: 선조회 실패는 fail-open", "_ureg8 = set()" in _a13)
chk("배선: 후보가 겹치면 소유권 주장을 접는다", "_own8 = None" in _a13
    and "suppressed(user-side)" in _a13)
chk("배선: 목록 **병기**(선점 금지)", "feedback_user_registry_listing" in _a13
    and "feedback_registry_listing" in _a13)
chk("배선: 기존 레지스트리-목록 분기가 손님-측 억제를 존중",
    "elif _fb8 and not _same8 and not _uown8:" in _a13)
chk("⚠[[70]] 계측 의무 명기(fired ↔ suppressed 짝 · give 성사율·오-give)",
    "무엇을 파는가" in _a13 and "suppressed(user-side)" in _a13 and "give 성사율" in _a13)


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A15 / OL-55] _BLOCK_NOTE 를 본문 전체로 커밋 금지 + 사유 절단 수정")
# ══════════════════════════════════════════════════════════════════════════════
_why = "this account has been closed by the customer and the operation cannot proceed now"
_old_cut, _new_cut = _why[:70], GP._trunc_reason(_why)
_core = _new_cut[:-3] if _new_cut.endswith("...") else _new_cut
chk("ⓟ 구판 `[:70]` 은 단어 중간에서 잘린다(`has been c` 부류)",
    not (len(_old_cut) == len(_why) or _why[len(_old_cut)] == " "), repr(_old_cut[-14:]))
chk("ⓝ 사유는 단어 경계에서 잘린다",
    _why.startswith(_core) and (len(_core) == len(_why) or _why[len(_core)] == " "),
    repr(_new_cut))
chk("ⓒ 짧은 사유는 손대지 않는다", GP._trunc_reason("account is closed") == "account is closed")
chk("ⓒ 공백 정규화만·None 안전", GP._trunc_reason(None) == "" and GP._trunc_reason("  a  b ") == "a b")

_am_full = MSG("assistant", content="Here is what I found so far.")
chk("ⓒ 본문이 있으면 종전대로 뒤에 붙인다(거동 보존)",
    GP._commit_block_note(_am_full, "[GB1] not verified") == "appended"
    and _am_full.content.startswith("Here is what I found so far.")
    and "[GB1] not verified" in _am_full.content)
_am_e1 = MSG("assistant", content="")
chk("ⓟ 구판 결함의 전제: 이 턴의 모델 생성분은 **빈 문자열**이다", _am_e1.content == "")
_r1 = GP._commit_block_note(_am_e1, "[GB1] not verified",
                            regen=lambda ask: "I could not complete that yet - I still need to "
                                              "verify your identity first.")
chk("ⓝ 빈 본문 → 모델에게 본문을 다시 받는다",
    _r1 == "regen" and _am_e1.content.startswith("I could not complete that yet"))
chk("ⓝ 재생성 본문 뒤에 노트가 붙는다(노트가 본문 전체가 아니다)",
    GP._BLOCK_NOTE in _am_e1.content and not _am_e1.content.startswith(GP._BLOCK_NOTE))
_am_e2 = MSG("assistant", content="")
chk("ⓝ 재생성이 또 비면 노트를 커밋하지 않는다",
    GP._commit_block_note(_am_e2, "[GB1] x", regen=lambda ask: "  ") == "empty"
    and _am_e2.content == "")


def _boom(_ask):
    raise RuntimeError("llm down")


_am_e3 = MSG("assistant", content="")
chk("ⓝ 재생성 예외도 흡수 — 노트는 여전히 본문이 되지 않는다",
    GP._commit_block_note(_am_e3, "[GB1] x", regen=_boom) == "empty" and _am_e3.content == "")
_am_e4 = MSG("assistant", content="")
chk("ⓒ regen 미전달(구 호출부·단위검정)도 노트가 본문이 되지 않는다",
    GP._commit_block_note(_am_e4, "[GB1] x") == "empty" and _am_e4.content == "")
chk("ⓟ 구판 슬라이스·직접 커밋이 파일에서 사라졌다",
    "(why or '')[:70]" not in SRC
    and 'am.content = (am.content or "") + _BLOCK_NOTE' not in SRC)
chk("배선: 두 R8 종단 호출부가 모두 `_commit_block_note` 를 쓴다(정의 1 + 호출 2)",
    SRC.count("_commit_block_note(am, note, regen=_bn_regen") == 2
    and SRC.count("def _commit_block_note(") == 1)
chk("배선: 두 호출부가 `_trunc_reason` 을 쓴다", SRC.count("_trunc_reason(why)") == 2)
chk("배선: 재생성은 **도구 없이**(새 호출이 게이트를 우회하지 못한다)",
    SRC.count('call_name="agent_blocknote_body"') == 2
    and seg("_BLOCK_NOTE_ASK", 40) != ""
    and all("tools=None" in seg(a, 700) for a in ("def _bn_regen(", "def _bn_regen_u(")))
chk("⚠[[70]] 계측 의무 명기(empty-body 턴 수 ↔ regen ok 수)",
    "empty-body" in SRC and "regen ok" in SRC and "무엇을 파는가" in seg("★A15/OL-55", 1400))

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
