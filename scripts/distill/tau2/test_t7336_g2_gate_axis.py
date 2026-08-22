# -*- coding: utf-8 -*-
"""G2 검정 (2026-08-22 · t7336 마스터 §6.1 A7 · A9 · A12 · A14) — 게이트 축·타이밍.

항목마다 **세 칸**을 고정한다(G1 규약 그대로):
  ⓟ 양성대조 = 수리 **전** 결함이 실재했음을 이 자리에서 재현(구판 술어를 축자로 평가)
  ⓝ 수리 후  = 같은 입력에서 결함이 사라진다
  ⓒ 부정통제 = 레버가 **죽지 않았다**(원 표적은 그대로 잡힌다) · 무엇을 파는지 고정

⛔사본 금지([[03b]]·[[67]]): 인라인 폐포 안의 조건·보정은 **소스에서 그 코드 자체를 뽑아
  실행**한다(재구현이 아니다). 구판 술어도 마찬가지로 **옛 파일의 코드**를 뽑아 돌린다.

오프라인 전용(LLM·env·서버 불요). 실행: py -3 test_t7336_g2_gate_axis.py
"""
import io
import json
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP                                          # noqa: E402
import t2_eplan_patch as EP                                         # noqa: E402
import t2_search as TS                                              # noqa: E402
import t2_prekb_patch as PK                                         # noqa: E402

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
PKSRC = io.open(os.path.join(HERE, "t2_prekb_patch.py"), encoding="utf-8").read()
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


def note(msg):
    print("  NOTE %s" % msg)


def block(src, start, end):
    """소스에서 `start` 줄머리 ~ `end`(제외) 를 뽑아 dedent — **그 코드 자체**를 돌리기 위한 것."""
    i = src.find(start)
    assert i >= 0, "anchor not found: %r" % start[:60]
    i = src.rfind("\n", 0, i) + 1                  # 줄머리까지 되감아야 dedent 가 성립한다
    j = src.find(end, i)
    assert j > i, "end anchor not found: %r" % end[:60]
    return textwrap.dedent(src[i:j])


def expr_in_if(src, anchor):
    """`if (…):` 한 덩어리에서 괄호 안 조건식만 뽑는다(그 코드 자체·평가용)."""
    i = src.find(anchor)
    assert i >= 0, anchor
    i = src.rfind("if (", 0, i + len(anchor))
    depth, k = 0, i + 3
    while k < len(src):
        if src[k] == "(":
            depth += 1
        elif src[k] == ")":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return " ".join(src[i + 4:k].split())


class TC(object):
    def __init__(self, tid, name, arguments=None, requestor="assistant"):
        self.id, self.name, self.arguments = tid, name, (arguments or {})
        self.requestor = requestor

    def model_dump(self):
        return {"id": self.id, "name": self.name, "arguments": self.arguments,
                "requestor": self.requestor}


class MSG(object):
    def __init__(self, role, mid=None, content="", error=False, tool_calls=None):
        self.role, self.id, self.content = role, mid, content
        self.error, self.tool_calls = error, tool_calls

    def model_dump(self):
        return {"role": self.role, "id": self.id, "content": self.content,
                "error": self.error,
                "tool_calls": [t.model_dump() for t in (self.tool_calls or [])]}


class SelfStub(object):
    pass


A2G = json.load(io.open(os.path.join(HERE, "a2/banking_knowledge.gate.json"), encoding="utf-8"))
A2S = json.load(io.open(os.path.join(HERE, "a2/banking_knowledge.specific.json"), encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A7] T2_WRITE_ARG_GROUND / ref_verify 를 do_gate·do_prov 축과 분리 (OL-34)")
# ──────────────────────────────────────────────────────────────────────────────
# 진입 술어를 **소스에서 그대로** 뽑아 평가한다.
_setup = block(SRC, "_fab_only = bool(do_gate or do_prov)",
               "if (_wev_live and ep_fb is None")
NEW_COND = expr_in_if(SRC, "_wev_live and ep_fb is None")
# 구판(2026-08-22 이전) 조건 — 이 자리에 축자로 있던 문자열. 양성대조 전용.
OLD_COND = ("(wev_specs or wag_specs or rv_specs or ae_on) "
            "and not do_gate and not do_prov and ep_fb is None "
            "and cons_fb is None and ra_fb is None and te_fb is None and proc_fb is None "
            "and wev_rounds < int(os.environ.get(\"T2_WEV_ROUNDS\", \"1\")) "
            "and getattr(self, \"_t2_wev_deny\", 0) < _wev_cap")


def gate_ns(do_gate=False, do_prov=False, wev=(), wag=(), rv=(), ae=False):
    return {"do_gate": do_gate, "do_prov": do_prov,
            "wev_specs": list(wev), "wag_specs": list(wag), "rv_specs": list(rv),
            "ae_on": ae, "ep_fb": None, "cons_fb": None, "ra_fb": None, "te_fb": None,
            "proc_fb": None, "wev_rounds": 0, "_wev_cap": 8, "os": os, "self": SelfStub()}


def live(ns, cond):
    ns = dict(ns)
    exec(_setup, ns)
    return bool(eval(cond, ns))


def old(ns):
    return bool(eval(OLD_COND, dict(ns)))


_gate_wag = gate_ns(do_gate=True, wag=[{"applies_to": "log_verification"}])
chk("ⓟ 구판: 게이트가 한 호출에 붙으면 WAG 가 그 턴 통째로 죽는다(074#1 기전)",
    old(_gate_wag) is False)
chk("ⓝ 수리 후: 같은 입력에서 날조-차단 계열이 산다",
    live(_gate_wag, NEW_COND) is True)
_prov_rv = gate_ns(do_prov=True, rv=[{"applies_to": "call_discoverable_agent_tool"}])
chk("ⓟ 구판: prov 라운드도 ref_verify 를 껐다", old(_prov_rv) is False)
chk("ⓝ 수리 후: prov 라운드에서도 ref_verify 가 산다", live(_prov_rv, NEW_COND) is True)

_gate_advice = gate_ns(do_gate=True, wev=[{"applies_to": "x"}], ae=True)
chk("ⓒ 조언 계열(WEV·ARG_EMPTY)만 있으면 게이트 턴엔 **여전히 침묵**(문구 모순 방지)",
    live(_gate_advice, NEW_COND) is False)
_plain = gate_ns(wev=[{"a": 1}], wag=[{"b": 2}], rv=[{"c": 3}], ae=True)
chk("ⓒ 게이트·prov 없는 평범한 턴은 구판과 **동치**(통과 런 불변)",
    live(_plain, NEW_COND) == old(_plain) is True)
_plain_none = gate_ns()
chk("ⓒ 선언이 하나도 없으면 종전대로 침묵(플래그 OFF 경로)",
    live(_plain_none, NEW_COND) is False and old(_plain_none) is False)

# 계열 분리가 **분기 단위로도** 지켜지는가 (소스 축자)
chk("ⓝ 조언 3분기가 `not _fab_only` 로 닫혔다(WEV·ARG_EMPTY·UNKNOWN_BOOL·HANDOFF)",
    "wd = None if _fab_only else _wev_deny_msgs(_wev_msgs, c, wev_specs)" in SRC
    and "if not wd and ae_on and not _fab_only:" in SRC
    and "if not wd and not _fab_only:\n                            # ★N2b" in SRC
    and "if not wd and wag_specs and not _fab_only:" in SRC)
chk("ⓝ 날조-차단 2분기는 **조건 없이** 돈다(WAG·REF_VERIFY)",
    "if not wd and wag_specs:\n                            # ★값-grounding" in SRC
    and "if not wd and rv_specs:" in SRC
    and "_fab_only else _write_arg_ground_deny" not in SRC
    and "_fab_only else _ref_verify_deny" not in SRC)
chk("ⓝ prov 표적 호출은 건너뛴다(main_prov 가 이기는 자리 = cap 낭비 방지)",
    "if _fab_only and _pcall is not None and c is _pcall[0]:" in SRC)
chk("⚠[[70]] 계측 의무 명기 — 이 수리가 산 차단을 셀 수 있다",
    "[T2_WAG_DECOUPLED] fired tag=%s phase=%s tool=%s" in SRC
    and "무엇을 파는가" in block(SRC, "★A7 / OL-34", "_fab_only = bool("))

# ⓒ 레버 자체가 살아 있는가 — **라이브 선언**으로 실제 판정을 돌린다(074#1 의 그 인자).
WAG = A2G.get("write_arg_grounding") or []
_lv = TC("c1", "log_verification", {"customer_id": "u1",
                                    "time_verified": "2023-11-14 15:30:00 EST"})
_no_clock = [MSG("tool", "t0", "account_id: acc_1\ndate_opened: 2023-11-14")]
_fb_wag = GP._write_arg_ground_deny(_no_clock, _lv, WAG)
chk("ⓒ WAG 정본이 074#1 의 날조 timestamp 를 여전히 잡는다",
    bool(_fb_wag) and "time_verified" in str(_fb_wag), str(_fb_wag)[:60])
_clock = [MSG("tool", "t0", "The current time is 2023-11-14 15:30:00 EST")]
chk("ⓒ 시계 출력이 실재하면 통과한다(false-block 0)",
    GP._write_arg_ground_deny(_clock, _lv, WAG) is None)


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A9] F8 억제 술어 = 이름-등장 → 값-가용 정본 (OL-46)")
# ──────────────────────────────────────────────────────────────────────────────
GIVE = "give_discoverable_user_tool"
UCALL = "call_discoverable_user_tool"
PROD = "get_card_last_4_digits"


def handed_only():
    """생산자를 **건네기만** 하고 손님은 실행하지 않은 궤적(040#1 형)."""
    return [MSG("assistant", tool_calls=[TC("g1", GIVE, {"discoverable_tool_name": PROD})]),
            MSG("tool", "g1", "Tool handed to the customer."),
            MSG("assistant", tool_calls=[TC("w1", "file_credit_card_transaction_dispute", {})]),
            MSG("tool", "w1", "Error: card_last_4_digits is a required property", error=False)]


def handed_and_ran():
    m = handed_only()
    m.insert(2, MSG("user", tool_calls=[TC("u1", PROD, {}, requestor="user")]))
    m.insert(3, MSG("tool", "u1", "card_last_4_digits: 5320"))
    return m


# 구판 술어 = **수리 전 `t2_prekb_patch` 의 그 코드 축자 스냅샷**.
# ★2026-08-22 (A9 호출부 재배선): 종전에는 살아 있는 소스에서 블록을 뽑아 돌렸다
#   (`block(PKSRC, "_seen_tools = set()", "_id2nm2 = {")`). 그런데 **호출부를 정본
#   `user_tool_value_ready` 로 교체하면서 구판 코드가 소스에서 사라졌다** — 추출은
#   원리상 더 불가능하다. 양성대조는 "그때 그 코드가 이랬다" 는 사실을 남기는 것이
#   목적이므로 축자 스냅샷으로 전환한다(아래 배선 검사가 **구판이 돌아오지 않았음**을
#   따로 지킨다). 스냅샷 = 커밋 `e7dcb97d` 시점 `t2_prekb_patch.py` 의 그 블록.
_OLD_SEEN_SNAPSHOT = """
for _m4 in (msgs or []):
    _md4 = _m4.model_dump() if hasattr(_m4, "model_dump") else {}
    for _tc4 in (_md4.get("tool_calls") or []):
        _seen_tools.add(str(_tc4.get("name") or ""))
        _a4 = _tc4.get("arguments")
        if isinstance(_a4, str):
            _seen_tools |= {w for w in re.findall(r"[a-z0-9_]+", _a4)}
        elif isinstance(_a4, dict):
            _seen_tools |= {str(v) for v in _a4.values()
                            if isinstance(v, str)}
"""


def seen_tools(msgs):
    ns = {"msgs": msgs, "re": re, "_seen_tools": set()}
    exec(_OLD_SEEN_SNAPSHOT, ns)
    return ns["_seen_tools"]


chk("ⓟ 구판: **건네기만** 해도 이름이 등장해 F8 이 영구 침묵한다(040#1 [84]/[86])",
    PROD in seen_tools(handed_only()))
chk("ⓝ 정본 값-가용 술어는 건네기를 값으로 세지 않는다 — F8 이 살아난다",
    PROD not in GP.user_tool_value_ready(handed_only(), GIVE, UCALL))
chk("ⓝ 배선: 호출부가 정본을 부른다(A9 미완 부채 해소·2026-08-22)",
    "import t2_gate_patch as _g9" in PKSRC and "_g9.user_tool_value_ready(" in PKSRC)
chk("ⓝ 배선: 구판 인자-토막 파싱이 소스에서 사라졌다([[59]]·되돌아오지 않았다)",
    're.findall(r"[a-z0-9_]+", _a4)' not in PKSRC)
chk("ⓒ 손님이 **실제로 실행**했으면 억제된다(값 가용 = 넛지 불필요)",
    PROD in GP.user_tool_value_ready(handed_and_ran(), GIVE, UCALL))
chk("ⓒ 디스패처 경유 실행도 값-가용으로 센다",
    PROD in GP.user_tool_value_ready(
        [MSG("assistant", tool_calls=[TC("d1", UCALL, {"discoverable_tool_name": PROD})])],
        GIVE, UCALL))
chk("ⓒ 정본 분해가 `give_exec_idle` 을 바꾸지 않는다(차집합 동치·사본 0)",
    GP.give_exec_idle(handed_only(), GIVE, UCALL) == [PROD]
    and GP.give_exec_idle(handed_and_ran(), GIVE, UCALL) == [])
_given, _ran = GP.give_exec_state(handed_and_ran(), GIVE, UCALL)
chk("ⓒ `give_exec_state` 두 집합이 idle 의 정의를 그대로 낸다",
    sorted(_given - _ran) == GP.give_exec_idle(handed_and_ran(), GIVE, UCALL))
chk("ⓒ 실패한 give 는 인계 성사가 아니다(종전 규약 보존)",
    GP.give_exec_state(
        [MSG("assistant", tool_calls=[TC("g9", GIVE, {"discoverable_tool_name": PROD})]),
         MSG("tool", "g9", "Error: no such tool", error=True)], GIVE, UCALL)[0] == set())

# ⓒ **오발화 미재발 축**(t7335 085): KB 본문은 에러 형상이 아니므로 F8 트리거가 아니다.
_kb_body = ("credit_cards_(general)_014: Filing a dispute. The following fields are required: "
            "card_last_4_digits, transaction_id, dispute_reason.")
chk("ⓒ KB 본문(에러 아님)은 F8 트리거가 아니다 — 085 오발화 미재발",
    PK._argprod_hits(A2G, _kb_body, is_error=False) == [])
chk("ⓒ 진짜 도구 에러는 여전히 트리거다(원 표적 보존)",
    [t for _, t in PK._argprod_hits(
        A2G, "Error: card_last_4_digits is a required property", is_error=False)] == [PROD])
chk("⚠[[70]] 계측 의무 명기 — 무엇을 파는가",
    "건넸다 ≠ 값을 얻었다" in SRC and "t7328 **7**·t7335 **5**" in SRC)
if "t not in _seen_tools" in PKSRC:
    note("⚠A9 **배선 미완**: `t2_prekb_patch.py` 는 이 그룹 소유 밖이라 호출부(`_hits = [(a, t) "
         "for a, t in _hits if t not in _seen_tools]`)를 바꾸지 않았다. 정본 술어만 설치했다 "
         "— 보고서 §미수리의 축자 패치 참조.")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A12] E-PLAN L2 — `list_from_reads` 도메인에서 '목록엔 있는데 안 읽었다' 는 항상 거짓 (OL-20)")
# ──────────────────────────────────────────────────────────────────────────────
BSPEC = dict(A2G.get("eplan") or {})
chk("전제: 라이브 선언이 `list_from_reads:true` 이고 형제 enumerator 가 둘이다",
    BSPEC.get("list_from_reads") is True
    and isinstance(BSPEC.get("list_enumerator"), list)
    and len(BSPEC["list_enumerator"]) == 2
    and BSPEC.get("detail_reader") not in (BSPEC["list_enumerator"][1],),
    "detail_reader=%s" % BSPEC.get("detail_reader"))

_ROWS = ("transaction_id: btxn_1001  amount: 12.50  merchant_name: Corner Cafe\n"
         "transaction_id: btxn_1002  amount: 31.00  merchant_name: Metro Transit\n"
         "transaction_id: btxn_1003  amount:  9.75  merchant_name: Book Barn")


def bank_msgs():
    """085#1 형: 손님이 3건을 말하고, **체킹** enumerator 한 번으로 전 필드가 실려 왔다."""
    return [MSG("user", content="please dispute these 3 transactions on my checking account"),
            MSG("assistant", tool_calls=[TC("r1", "get_bank_account_transactions",
                                            {"account_id": "acc_1"})]),
            MSG("tool", "r1", _ROWS)]


def build(spec, msgs, writes):
    return EP.build_ledger_from_messages(msgs, spec, set(writes))


_led_raw = build(BSPEC, bank_msgs(), BSPEC.get("write_tools") or [])
_ids_before = EP.discovery_L2(_led_raw, "file_debit_card_transaction_dispute")
chk("ⓟ 구판: 바로 앞 출력에 **전량 실린** 레코드를 L2 가 미검토로 판정한다",
    _ids_before == ["btxn_1001", "btxn_1002", "btxn_1003"], _ids_before)
_fb_before = EP.l2_feedback(list(_ids_before), BSPEC)
chk("ⓟ 구판: 그 문면이 **체킹 레코드에 credit 도구**를 지목한다([[25]] 거짓 지목)",
    "get_credit_card_transactions_by_user" in _fb_before
    and "get_bank_account_transactions" not in _fb_before)

# 수리 = **소스에서 그 보정 코드 자체**를 뽑아 돌린다.
_fix = block(SRC, 'if ep_led is not None and ep_spec.get("list_from_reads"):',
             "            except Exception as _e:")
_fix = textwrap.dedent(_fix)


def apply_fix(led, spec):
    ns = {"ep_led": led, "ep_spec": spec, "_sys": sys, "print": lambda *a, **k: None}
    exec(_fix, ns)
    return led


_led_fixed = apply_fix(build(BSPEC, bank_msgs(), BSPEC.get("write_tools") or []), BSPEC)
chk("ⓝ 수리 후: 같은 궤적에서 L2 가 침묵한다(085#1 4턴 회복)",
    EP.discovery_L2(_led_fixed, "file_debit_card_transaction_dispute") == [])
chk("ⓝ 같은 보정이 `T2_READALL` 의 unread 도 비운다(소비자 둘·자리 하나)",
    GP.readall_unread(_led_fixed.listed, _led_fixed.examined) == [])

# ⓒ 부정통제 ①: 선언 없는 도메인(retail 형)은 **바이트 불변** — L2 가 그대로 산다.
RSPEC = {"list_enumerator": "list_tool", "detail_reader": "detail_tool",
         "entity_key": "order_id", "items_key": "item_ids", "write_tools": ["exchange"]}
_r_msgs = [MSG("user", content="I need to exchange items in 2 orders"),
           MSG("assistant", tool_calls=[TC("l1", "list_tool", {})]),
           MSG("tool", "l1", json.dumps({"orders": [{"order_id": "o_1"}, {"order_id": "o_2"}]}))]
_r_raw = build(RSPEC, _r_msgs, ["exchange"])
_r_before = EP.discovery_L2(_r_raw, "exchange")
_r_after = EP.discovery_L2(apply_fix(build(RSPEC, _r_msgs, ["exchange"]), RSPEC), "exchange")
chk("ⓒ `list_from_reads` 미선언 도메인은 보정 no-op — L2 가 그대로 산다",
    _r_before == _r_after == ["o_1", "o_2"], (_r_before, _r_after))

# ⓒ 부정통제 ②: L1(목록 도구 미호출)은 그대로 발화한다 — read-강제 전부를 판 것이 아니다.
_l1_msgs = [MSG("user", content="please dispute these 3 transactions")]
_l1_led = apply_fix(build(BSPEC, _l1_msgs, BSPEC.get("write_tools") or []), BSPEC)
chk("ⓒ L1(목록 미조회)은 여전히 발화한다 — 판 것은 L2 뿐",
    EP.discovery_L1(_l1_led) is True)
chk("ⓒ 보정은 `listed` 를 **늘리지 않는다**(examined 로만 채운다·관측 불변)",
    _led_fixed.listed == _led_raw.listed)
chk("ⓝ 범위 표면화: READALL 렌더가 정본 `_tool_phrase` 를 쓴다(목록 선언 가능)",
    "reader=_epmod._tool_phrase(" in SRC
    and EP._tool_phrase(["a", "b"]) == "a or b" and EP._tool_phrase("a") == "a")
chk("⚠[[70]] 계측 의무 명기 — 이 도메인에서 판 것 = L2·READALL 의 read-강제",
    "[T2_EPLAN_LISTED_IS_READ]" in SRC
    and "무엇을 파는가" in block(SRC, "★A12 / OL-20", "except Exception as _e:"))
if 'if tool_name == self.spec.get("detail_reader"):' in io.open(
        os.path.join(HERE, "t2_eplan_patch.py"), encoding="utf-8").read():
    note("⚠A12 **선언 미변경**: `detail_reader` 를 목록으로 바꾸려면 "
         "`t2_eplan_patch.note_read` 의 `==` 비교를 집합 소속으로 같이 고쳐야 한다 "
         "(그 파일은 이 그룹 소유 밖) — 선언만 바꾸면 `examined` 가 구조적 공집합이 된다.")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[A14] 퇴화 축에서 `decide_from_docs` 결정문 미배달 (OL-15)")
# ──────────────────────────────────────────────────────────────────────────────
PO = A2S.get("policy_ontology") or {}
IDX = PO.get("doc_index") or {}
DEGEN = GP._degenerate_axes(PO)
chk("전제: 라이브 선언에 퇴화 축이 정확히 하나 있다(085#1 의 그 군)",
    DEGEN == {"bank_accounts_bank_accounts"}, sorted(DEGEN))
chk("전제: 그 군의 계열이 `_general_` 뿐이다",
    set(IDX["bank_accounts_bank_accounts"]) == {"_general_"})
chk("ⓟ 구판: 그 군이 결정 서브로 가면 표시명이 곧 답이 된다(*It answers: General.*)",
    TS._disp_name("_general_") == "General"
    and "It answers" in str(PO.get("decided_by_docs_text") or ""),
    str(PO.get("decided_by_docs_text") or "")[:60])

_sel = block(SRC, "    # ★A14 / OL-15 (2026-08-22",
             "\n    # ★관측 전용 계기 (2026-08-18·C517⒟)")
_sel = textwrap.dedent(_sel)


def pick(gs, done=(), decide=True):
    ns = {"_gs": list(gs), "_done": set(done), "decide": decide, "_po": PO,
          "_degenerate_axes": GP._degenerate_axes, "sys": sys,
          "print": lambda *a, **k: None}
    exec(_sel, ns)
    return ns["_g"], ns["_skip_degen"]


def pick_old(gs, done=()):
    return next((g for g in gs if g not in set(done)), None)


chk("ⓟ 구판 선택기는 퇴화 축을 그대로 집는다",
    pick_old(["bank_accounts_bank_accounts", "credit_cards"]) == "bank_accounts_bank_accounts")
chk("ⓝ 수리 후: 퇴화 축을 건너뛰고 **실재하는 축**이 결정점을 받는다",
    pick(["bank_accounts_bank_accounts", "credit_cards"])[0] == "credit_cards")
chk("ⓝ 퇴화 축뿐이면 결정문을 만들지 않는다(침묵)",
    pick(["bank_accounts_bank_accounts"])[0] is None)
chk("ⓝ 건너뛴 사실이 로그로 남는다([[70]] 계수 재료)",
    pick(["bank_accounts_bank_accounts"])[1] == ["bank_accounts_bank_accounts"])
chk("ⓒ 비-퇴화 축은 구판과 **동일 선택**(31 축 거동 불변)",
    all(pick([g])[0] == pick_old([g]) for g in IDX if g not in DEGEN))
chk("ⓒ 소진(`_done`) 규약은 그대로다",
    pick(["credit_cards", "savings_accounts"], done={"credit_cards"})[0] == "savings_accounts")
chk("ⓒ 문서-본문 배달(`decide=False`)은 퇴화 축을 **건너뛰지 않는다**(KB 채널 보존)",
    pick(["bank_accounts_bank_accounts"], decide=False)[0] == "bank_accounts_bank_accounts")
chk("ⓒ 술어는 **선언만** 본다 — 색인 없으면 빈 집합(fail-open)",
    GP._degenerate_axes({}) == set() and GP._degenerate_axes(None) == set()
    and GP._degenerate_axes({"doc_index": {"g": {"a": [], "_general_": []}}}) == set())
chk("⚠[[70]] 계측 의무 명기 — 판 것 = 그 축의 DOCDECIDE 배달 수",
    "[T2_DEGENERATE_AXIS]" in SRC
    and "무엇을 파는가" in block(SRC, "★A14 / OL-15 (2026-08-22 · t7336", "_degen = _degenerate_axes"))


# ══════════════════════════════════════════════════════════════════════════════
print("\n[규율] 엔진 도메인 리터럴 0 · 선언 출처")
_added = [b for b in (block(SRC, "def _degenerate_axes(po):", "def _served_subjects("),
                      block(SRC, "def give_exec_state(messages", "def user_tool_value_ready("),
                      block(SRC, "def user_tool_value_ready(messages", "\n\ndef _regen_last_user"))]
_code = "\n".join(l for b in _added for l in b.splitlines() if not l.strip().startswith("#"))
_code = re.sub(r'"""[\s\S]*?"""', "", _code)
chk("신설 3함수의 **코드**에 도메인 어휘 0(계열·군·도구명 리터럴 없음)",
    not any(w in _code for w in ("bank_account", "credit_card", "savings", "checking",
                                 "dispute", "General", "btxn_", "card_last_4")), _code[:80])
# ★2026-08-22 강화: `"_general_"` 리터럴이 **네 자리**로 갈릴 참이었다(퇴화 군·배달 계열·
#   표시명 색인 + WRITE_ARG_ENUM 후보 명단). 정본 술어 `_subject_keys` 하나로 모았고,
#   술어가 **형상 판정**이라 이름 리터럴이 아예 0 이 됐다([[67]]·[[59]]).
chk("퇴화 판정에 이름 리터럴이 없다(정본 술어 `_subject_keys` 공유)",
    _code.count('"_general_"') == 0 and "_subject_keys(subs)" in _code)
chk("`_general_` 을 아는 자리가 정본 하나뿐이다(사본 0·[[67]])",
    SRC.count("def _subject_keys(") == 1
    and all("_subject_keys(" in block(SRC, a, b) for a, b in (
        ("def _degenerate_axes(po):", "def _served_subjects("),
        ("def _served_subjects(po, group", "def _record_served("),
        ("def _display_slugs(subs):", "def _flatten("))))

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
