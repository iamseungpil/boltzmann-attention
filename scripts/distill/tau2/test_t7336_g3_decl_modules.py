# -*- coding: utf-8 -*-
"""t7336 G3 회귀 — 선언·타 모듈 수리 (A4·A6·A8·A10·A16 · A11 부정통제).

정본 = `reports/facet_rft_2026/T7336_FAILURE_MASTER_2026_08_22.md` §6.1 / §5.
오프라인 전용(모델 0·네트워크 0). 실행: `py -3 test_t7336_g3_decl_modules.py`

각 항목마다 셋을 다 둔다 — **⑴수리 전 결함 재현(양성대조) ⑵수리 후 ⑶부정통제**.

  A4  `T2_DISCOVERY_STEP2` 가 지목한 이름을 `agent._t2_our_names` 에 등재      (OL-02)
  A6  `requires_reads` 선언 3종 + `relations` 동기                            (OL-37·38·39·23)
  A8  `get_interest_correction` 부호 게이트(결과<0 → abstain + 이름 있는 지목)  (OL-11)
  A10 `_byref_require_fields` 를 `_iso_owns` 우회 `try` **안**으로             (OL-48)
  A16 `t2_forensic.action_diff` (=`action_checks` 기반 MATCH/MISSING)          (OL-49)
  A11 부정통제만 — 수리 자체는 `t2_gate_patch.py`(G1·G2 소유)라 여기서 안 한다  (OL-44)
"""
import collections
import glob
import gzip
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

FAILED = []
NOTES = []


def chk(cond, label, extra=""):
    print(("  OK   " if cond else "  FAIL ") + label + ((" — " + str(extra)) if extra else ""))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def note(msg):
    NOTES.append(msg)
    print("  note " + msg)


def load_json(rel):
    with io.open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return json.load(f)


SGT_PATHS = ["a2/banking_knowledge.specific.json",
             "a2/banking_knowledge.gate.json",
             "a2/split/banking_knowledge.core.json"]
REL_PATHS = ["a2/banking_knowledge.specific.json",
             "a2/banking_knowledge.gate.json"]
SG_SRC = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
RZ_SRC = io.open(os.path.join(HERE, "t2_resolve.py"), encoding="utf-8").read()


def sgt(doc, name):
    return next((t for t in (doc.get("scaffold_get_tools") or [])
                 if t.get("name") == name), None)


# ═════════════════════════ 공통 스텁 (라이브 객체 흉내) ═════════════════════════
class _TC:
    def __init__(self, name, args=None, id=None):
        self.name, self.arguments, self.id = name, (args or {}), id


class _M:
    def __init__(self, role, content="", calls=None, error=False, id=None):
        self.role, self.content, self.error, self.id = role, content, error, id
        self.tool_calls = calls or []


class _Tool:
    def __init__(self, name, desc=""):
        self.name, self.description = name, desc


# ═══════════════════════════════ A4 (OL-02) ═══════════════════════════════════
def test_a4():
    print("\n[A4/OL-02] STEP2 가 지목한 이름을 `_t2_our_names` 에 등재한다")
    import t2_resolve as R

    NAME = "get_all_user_accounts_by_user_id_3847"
    OTHER = "some_tool_that_is_not_in_any_registry_9999"

    class _Toolkit:
        def __init__(self, names):
            self.tools = {n: object() for n in names}

        def get_discoverable_tools(self):
            return dict(self.tools)

    class _Env:
        def __init__(self, names):
            self.tools = _Toolkit(names)

    class _Orch:
        def __init__(self, names):
            self.environment = _Env(names)

    class _Agent:
        def __init__(self, names):
            self.tools = [_Tool("call_discoverable_agent_tool")]
            self._t2_orch = _Orch(names)
            self.llm = "stub-model"           # `t2_subcall.sub_generate` 가 요구
            self.llm_args = {}

    class _Sub:
        content = json.dumps({"tool": NAME})

    class _LA:
        @staticmethod
        def generate(*a, **k):
            return _Sub()

    def _UM(**k):
        return _M("user", k.get("content", ""))

    A2 = {"eplan": {"unlock_tool": "unlock_discoverable_agent_tool",
                    "dispatch_tool": "call_discoverable_agent_tool",
                    "list_tool": "list_discoverable_agent_tools"},
          "operands": {"call_discoverable_agent_tool":
                       {"agent_tool_name": {"operator_resolution": "discoverable",
                                            "name_pattern": r"[a-z_]+_\d{4}",
                                            "getter": "KB_search_bm25"}}},
          "action_tools": ["call_discoverable_agent_tool"]}
    # 회수 텍스트에 이름이 실재하는 대화(=STEP2 가 (2)단계를 말하는 조건)
    msgs = [_M("user", "please look up my accounts"),
            _M("tool", "the tool for this is %s" % NAME, id="t1")]
    am = _M("assistant", "I will explain how to do it")          # 회피 턴(순수 텍스트)

    os.environ["T2_DISCOVERY_STEP2"] = "1"
    os.environ["T2_PROV_OURS"] = "1"
    agent = _Agent([NAME, OTHER])

    res = R.resolve_action_operator(
        {}, am, msgs, A2, target_tool="call_discoverable_agent_tool",
        transfer_tools=set(), known_names={NAME}, agent=agent, la=_LA, UserMessage=_UM)
    chk(res.get("reason") == "discovery-step2", "STEP2 가 지목했다", res.get("reason"))
    chk(NAME in (getattr(agent, "_t2_our_names", None) or set()),
        "수리 후: 지목한 이름이 `_t2_our_names` 에 등재됐다",
        sorted(getattr(agent, "_t2_our_names", None) or ()))

    # ⑵ 소비자(resolve_operator)가 그 등재를 실제로 소비한다 — 후보 집합에 없어도 통과
    opspec = {"operator_resolution": "discoverable", "arg": "agent_tool_name",
              "name_pattern": r"[a-z_]+_\d{4}", "getter": "KB_search_bm25"}
    cand_msgs = [_M("tool", "found tool other_read_1111", id="c1")]   # 후보엔 NAME 이 없다
    ok = R.resolve_operator(opspec, {"agent_tool_name": NAME}, cand_msgs, agent=agent)
    chk(ok.get("status") == "ok",
        "수리 후: `operator-fab` 이 우리 지목을 더는 막지 않는다", ok.get("reason"))

    # ⑴ 양성대조(수리 전 재현) — 등재를 지우면 같은 자리가 `operator-fab` deny 다
    agent._t2_our_names = set()
    bad = R.resolve_operator(opspec, {"agent_tool_name": NAME}, cand_msgs, agent=agent)
    chk(bad.get("status") == "deny" and bad.get("reason") == "operator-fab",
        "양성대조: 등재가 없으면 085#0 의 `operator-fab` 이 재현된다", bad.get("reason"))

    # ⑶ 부정통제 A — 레지스트리 **밖** 이름은 등재되지 않는다(날조 통과 구조적 불가)
    class _Sub2:
        content = json.dumps({"tool": "totally_made_up_4242"})

    class _LA2:
        @staticmethod
        def generate(*a, **k):
            return _Sub2()

    agent2 = _Agent([NAME])
    msgs2 = [_M("user", "hi"), _M("tool", "totally_made_up_4242 is the tool", id="t2")]
    R.resolve_action_operator({}, am, msgs2, A2, target_tool="call_discoverable_agent_tool",
                              transfer_tools=set(), known_names={"totally_made_up_4242"},
                              agent=agent2, la=_LA2, UserMessage=_UM)
    chk("totally_made_up_4242" not in (getattr(agent2, "_t2_our_names", None) or set()),
        "부정통제: 레지스트리 밖 이름은 등재 0",
        sorted(getattr(agent2, "_t2_our_names", None) or ()))

    # ⑶ 부정통제 B — 등재되지 않은 날조 이름은 여전히 deny 된다
    fab = R.resolve_operator(opspec, {"agent_tool_name": "totally_made_up_4242"},
                             cand_msgs, agent=agent2)
    chk(fab.get("status") == "deny" and fab.get("reason") == "operator-fab",
        "부정통제: 날조 이름은 그대로 막힌다", fab.get("reason"))

    chk("[T2_OUR_NAMES] 등재" in RZ_SRC, "등재는 로그 마크를 남긴다([[55]])")
    for k in ("T2_DISCOVERY_STEP2", "T2_PROV_OURS"):
        os.environ.pop(k, None)


# ═══════════════════════════════ A6 (OL-37·38·39·23) ═══════════════════════════
def test_a6():
    print("\n[A6/OL-37·38·39·23] `requires_reads` 선언 3종 + relations 동기")
    env = load_json("a2/env_surface.json")["banking_knowledge"]["tools"]
    reads_ok = {k.rsplit("_", 1)[0] if re.search(r"_\d+$", k) else k
                for k, v in env.items() if not v.get("mutates")}

    docs = [load_json(p) for p in SGT_PATHS]

    # ⑴ get_atm_fee_discrepancies
    atm = [sgt(d, "get_atm_fee_discrepancies") for d in docs]
    want = ["get_all_user_accounts_by_user_id", "get_bank_account_transactions"]
    chk(all((t or {}).get("requires_reads") == want for t in atm),
        "⑴ get_atm_fee_discrepancies.requires_reads = 목록+거래 (3사본 동일)",
        [(t or {}).get("requires_reads") for t in atm])
    fb = (atm[0] or {}).get("requires_reads_feedback") or ""
    chk(all(r in fb for r in want),
        "⑴ 문면이 두 read 를 **이름으로** 댄다([[64]] fix-naming)")
    chk("the accounts listing" not in fb,
        "⑴ 양성대조: 이름 없는 산문 *'the accounts listing'* 이 사라졌다(072#1 재발 방지)")

    # ⑶ get_correct_savings_apy
    apy = [sgt(d, "get_correct_savings_apy") for d in docs]
    chk(all((t or {}).get("requires_reads") == ["get_all_user_accounts_by_user_id"] for t in apy),
        "⑶ get_correct_savings_apy.requires_reads = 계좌목록 (3사본 동일)",
        [(t or {}).get("requires_reads") for t in apy])
    chk(all("_043" in ((t or {}).get("_note_requires_reads") or "") for t in apy),
        "⑶ 정책 축자 출처(doc_…_043)가 note 에 있다([[23]])")

    # 불변식 — 이름은 env 레지스트리 실재 read · 접미사 없음 (선언 전체)
    bad = []
    for d in docs:
        for t in (d.get("scaffold_get_tools") or []):
            for r in (t.get("requires_reads") or []):
                if re.search(r"_\d+$", str(r)):
                    bad.append("%s: %s 에 접미사" % (t.get("name"), r))
                if r not in reads_ok:
                    bad.append("%s: %s 가 env 비변이 read 가 아니다" % (t.get("name"), r))
    chk(not bad, "선언된 read 는 전부 env 레지스트리의 **비변이** 도구다(기계도출·[[23]])",
        " · ".join(bad[:4]))

    # ⑵ relations — file_credit_card_transaction_dispute 신설 + 3행 동기
    for p in REL_PATHS:
        R = load_json(p)["relations"]
        decls = {(x["source"], x["dep"]): x for x in R["declarations"]}
        ccd = decls.get(("requires_reads", "file_credit_card_transaction_dispute"))
        chk(ccd is not None, "⑵ %s: file_credit_card_transaction_dispute 항목 실재" % p)
        if ccd:
            chk(ccd["reads"] == ["get_credit_card_transactions_by_user",
                                 "get_user_dispute_history"],
                "⑵ %s: 신용카드 거래 read + 분쟁 이력 read" % p, ccd["reads"])
            chk("_014" in (ccd.get("note") or "") and "_015" in (ccd.get("note") or ""),
                "⑵ %s: 정책 축자 출처(doc_…_014/_015)가 note 에 있다([[23]])" % p)
        acc = decls.get(("requires_reads", "apply_checking_account_credit"))
        chk(acc and "get_all_user_accounts_by_user_id" in acc["reads"],
            "⑴ %s: apply_checking_account_credit 에 계좌목록 read" % p,
            acc and acc["reads"])
        chk(acc and "_017" in (acc.get("note") or ""),
            "⑴ %s: 정책 축자 출처(doc_…_017)가 note 에 있다" % p)
        # edges / by_tool 동기([[24]] 양방향)
        drift = []
        for x in R["declarations"]:
            e = R["edges"].get(x["dep"]) or []
            if any(r not in e for r in x["reads"]):
                drift.append("edges:%s" % x["dep"])
            bt = (R["by_tool"].get(x["dep"]) or {}).get("requires") or []
            if any(r not in bt for r in x["reads"]):
                drift.append("by_tool:%s" % x["dep"])
        chk(not drift, "%s: declarations ↔ edges ↔ by_tool 동기" % p, " · ".join(drift[:4]))

    # 두 층(specific·gate)이 바이트-등가([[24]])
    a, b = (load_json(p)["relations"] for p in REL_PATHS)
    chk(json.dumps(a, sort_keys=True, ensure_ascii=False)
        == json.dumps(b, sort_keys=True, ensure_ascii=False),
        "relations 2층 json-등가([[24]])")

    # ⑶ 부정통제 — gold 경유 0: 이번에 손댄 선언/문면에 gold 참조가 없다
    touched = []
    for p in SGT_PATHS:
        d = load_json(p)
        for n in ("get_atm_fee_discrepancies", "get_correct_savings_apy"):
            touched.append(sgt(d, n))
    for p in REL_PATHS:
        touched.append(load_json(p)["relations"]["declarations"])
    blob = json.dumps(touched, ensure_ascii=False)
    chk(not any(w in blob for w in ("reward_info", "action_checks", "reward_basis",
                                    "tasks/", "gold 액션")),
        "부정통제: 이번에 손댄 선언·문면에 gold 참조 0([[23]])")

    # ⑴ 양성대조 — 옛 값(거래 read 하나뿐)은 검정이 잡는다
    old = {"name": "x", "requires_reads": ["get_bank_account_transactions"],
           "requires_reads_feedback": "… copied from the accounts listing …"}
    chk(not all(r in old["requires_reads_feedback"] for r in want),
        "양성대조: 옛 선언은 ⑴ 검사를 통과하지 못한다")


# ═══════════════════════════════ A8 (OL-11) ═══════════════════════════════════
def test_a8():
    print("\n[A8/OL-11] get_interest_correction 부호 게이트 — 결과<0 이면 abstain")
    import t2_compute as TC
    docs = [load_json(p) for p in SGT_PATHS]
    ic = [sgt(d, "get_interest_correction") for d in docs]
    rng = (ic[0] or {}).get("result_range") or {}
    chk(rng == {"min_exclusive": 0}, "선언: result_range = {min_exclusive: 0}", rng)
    chk(all((t or {}).get("result_range") == rng for t in ic), "3사본 동일")
    fb = (ic[0] or {}).get("result_range_feedback") or ""
    chk(all((t or {}).get("result_range_feedback") == fb for t in ic), "3사본 문면 동일")
    chk("must be greater than 0" in fb,
        "문면이 env/정책 인자 계약을 **축자로** 인용한다([[25]])")
    chk("get_correct_savings_apy" in fb and "get_all_user_accounts_by_user_id" in fb,
        "문면이 해소 경로를 **이름으로** 댄다([[64]] 무엇을 하면 풀리나)")
    chk("{result}" in fb and "{expected_apy}" in fb and "{actual_apy}" in fb,
        "문면은 치환 슬롯만 쓴다(엔진 리터럴 0)")

    # ⑵ 수리 후 — 선언된 op 로 실제 부호를 잰다(엔진 정본 `t2_compute.apply_op` 사용)
    op = (ic[0] or {}).get("op")
    neg = TC.apply_op(op, {"principal": 10000.0, "expected_apy": 4.0, "actual_apy": 5.0})
    pos = TC.apply_op(op, {"principal": 10000.0, "expected_apy": 5.5, "actual_apy": 5.0})
    chk(isinstance(neg, (int, float)) and float(neg) <= float(rng["min_exclusive"]),
        "093#1 형(expected<actual): 결과가 음수 → 술어 성립(abstain)", neg)
    chk(isinstance(pos, (int, float)) and float(pos) > float(rng["min_exclusive"]),
        "부정통제: 정상형(expected>actual)은 술어 불성립 → 종전대로 값이 나간다", pos)
    # 문면의 치환 슬롯이 **비어 나갈 수 없다**([[25]]): 피연산자가 하나라도 없으면 op 이
    # abstain(None) 이라 게이트 자체가 안 걸린다 ⇒ 걸릴 때는 세 값이 전부 ctx 에 있다.
    part = [TC.apply_op(op, c) for c in
            ({"expected_apy": 4.0, "actual_apy": 5.0},
             {"principal": None, "expected_apy": 4.0, "actual_apy": 5.0},
             {"principal": 10000.0, "actual_apy": 5.0},
             {"principal": 10000.0, "expected_apy": 4.0, "actual_apy": None})]
    chk(all(p is None for p in part),
        "피연산자 결손이면 op 이 abstain(None) → 게이트 미발화(빈 슬롯 문면 불가)", part)

    # 엔진 배선 — 선언을 읽고 abstain 으로 빠지는 경로가 실재한다
    chk('d.get("result_range")' in SG_SRC, "엔진이 `result_range` 선언을 읽는다")
    chk('T2_SG_RESULT_RANGE' in SG_SRC, "끄기 스위치가 있다(A/B·기본 ON·[[60]])")
    chk('"1") != "0"' in SG_SRC or 'T2_SG_RESULT_RANGE", "1"' in SG_SRC,
        "기본값 = ON")
    chk("[T2_SG_RESULT_RANGE]" in SG_SRC, "발화가 로그 마크를 남긴다([[55]])")
    m = re.search(r'_rrg = d\.get\("result_range"\).*?continue', SG_SRC, re.S)
    body = m.group(0) if m else ""
    chk(bool(body), "게이트 블록을 찾았다")
    for pat, why in ((r"\bmax\s*\(", "max("), (r"\bargmax\b", "argmax"),
                     (r"re\.(search|findall|match)\s*\(", "정규식")):
        chk(not re.search(pat, body), "판단·선택기 없음: %s" % why)

    # ⑶ 부정통제 — 미선언 도구는 거동 변화 0
    others = [t.get("name") for t in (docs[0].get("scaffold_get_tools") or [])
              if t.get("result_range") is not None]
    chk(others == ["get_interest_correction"],
        "부정통제: `result_range` 를 선언한 도구는 이 하나뿐(나머지 거동 변화 0)", others)


# ═══════════════════════════════ A10 (OL-48) ══════════════════════════════════
DUMP_NO_FIELDS = ("Found 2 record(s) in a table:\n"
                  "Record ID: btxn_aaa\n  amount: 2.00\n"
                  "Record ID: btxn_bbb\n  amount: 3.00\n")


class _OrchB:
    def __init__(self, text):
        self._msgs = [_M("assistant", calls=[_TC("get_bank_account_transactions", id="c1")]),
                      _M("tool", text, id="c1")]

    def get_messages(self):
        return self._msgs


def _spec(with_isolate):
    d = {"name": "probe_tool",
         "op": {"op": "sum", "over": "transactions", "value_field": "fee_amount"}}
    if with_isolate:
        d["isolate"] = {"mode": "fetch_formalize", "operand_keys": ["transactions"]}
    return d


def test_a10():
    print("\n[A10/OL-48] `_byref_require_fields` 를 `_iso_owns` 우회 안으로")
    import t2_scaffold_get as SG

    # ⑵ 수리 후 — isolate(fetch_formalize) 가 그 키를 산출하면 컬럼 부재로 죽이지 않는다
    ctx = {"transactions": "@last:get_bank_account_transactions"}
    orch = _OrchB(DUMP_NO_FIELDS)
    raised = None
    try:
        SG._byref_resolve(orch, _spec(True), ctx)
    except Exception as e:
        raised = e
    chk(raised is None, "수리 후: 컬럼 부재여도 deny 하지 않는다(손-전사 요구 0)", repr(raised))
    chk(ctx["transactions"] == "@last:get_bank_account_transactions",
        "폴백 유지: `@last:` 문자열이 남아 서브가 덮어쓴다(over-str 검사 살아 있음)",
        ctx["transactions"])

    # ⑴ 양성대조(수리 전 재현) — isolate 미선언이면 종전대로 지목이 나간다
    ctx2 = {"transactions": "@last:get_bank_account_transactions"}
    raised2 = None
    try:
        SG._byref_resolve(_OrchB(DUMP_NO_FIELDS), _spec(False), ctx2)
    except SG._ByrefError as e:
        raised2 = e
    chk(raised2 is not None, "양성대조: isolate 미선언 도구에서는 컬럼 부재 지목이 그대로다")
    chk("fee_amount" in str(raised2 or ""),
        "양성대조: 그 지목이 결핍 컬럼을 이름으로 댄다", str(raised2 or "")[:70])

    # ⑶ 부정통제 — 컬럼이 **있으면** isolate 선언 여부와 무관하게 정상 해소된다
    good = ("Found 1 record(s) in a table:\n"
            "Record ID: btxn_ccc\n  fee_amount: 3.50\n")
    ctx3 = {"transactions": "@last:get_bank_account_transactions"}
    SG._byref_resolve(_OrchB(good), _spec(True), ctx3)
    chk(isinstance(ctx3["transactions"], list) and len(ctx3["transactions"]) == 1,
        "부정통제: 정상 덤프는 종전대로 rows 로 치환된다", ctx3["transactions"])

    # ⑶ 부정통제 B — `@last` 자체가 못 풀리는 경우의 우회는 종전 그대로
    ctx4 = {"transactions": "@last:never_called_tool"}
    r4 = None
    try:
        SG._byref_resolve(_OrchB(good), _spec(True), ctx4)
    except Exception as e:
        r4 = e
    chk(r4 is None and ctx4["transactions"] == "@last:never_called_tool",
        "부정통제: 미호출 참조도 종전 우회대로 통과(거동 보존)", repr(r4))

    # 구조 — 두 호출이 `try` 안, `except _ByrefError` 안에 `_iso_owns` 우회
    m = re.search(r"_iso_owns = .*?\n(.*?)\n        ctx\[k\] = rows", SG_SRC, re.S)
    blk = m.group(1) if m else ""
    call = "_byref_require_fields(d, k, rows)"
    chk("try:" in blk and call in blk, "구조: 요구 검사 호출을 찾았다")
    ti, te, tc = blk.find("try:"), blk.find("except _ByrefError"), blk.find(call)
    chk(0 <= ti < tc < te,
        "구조: 요구 검사 호출이 `try` 와 `except` **사이**(=try 블록 내부)에 있다",
        (ti, tc, te))
    chk(blk.count(call) == 1 and blk.count("_byref_map_fields(d, rows)") == 1,
        "구조: 호출은 각각 한 번뿐(사본 0)")


# ═══════════════════════════════ A16 (OL-49) ══════════════════════════════════
def _fixture_action_sim():
    """ACTION-basis + discoverable 래퍼 gold — OL-49 가 말한 그 모양(합성·gold 무열람)."""
    unlock = {"name": "unlock_discoverable_agent_tool",
              "arguments": {"agent_tool_name": "get_user_dispute_history_7291"},
              "action_id": "a1"}
    call = {"name": "call_discoverable_agent_tool",
            "arguments": {"agent_tool_name": "get_user_dispute_history_7291",
                          "arguments": {"user_id": "u_1"}},
            "action_id": "a2"}
    return {
        "task_id": "fixture_action", "trial": 0,
        "reward_info": {"reward": 0.0, "reward_basis": ["ACTION"],
                        "action_checks": [{"action": unlock, "action_match": True,
                                           "tool_type": "read"},
                                          {"action": call, "action_match": False,
                                           "tool_type": "read"}]},
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"name": "unlock_discoverable_agent_tool", "id": "x1",
                 "arguments": {"agent_tool_name": "get_user_dispute_history_7291"}}]},
            {"role": "tool", "id": "x1", "content": "Tool unlocked."},
            {"role": "assistant", "tool_calls": [
                {"name": "call_discoverable_agent_tool", "id": "x2",
                 "arguments": {"agent_tool_name": "get_user_dispute_history_7291",
                               "arguments": {"user_id": "u_1"}}}]},
            {"role": "tool", "id": "x2",
             "content": "Error: [READ-FIRST] you must read the records first."},
        ]}


def _fixture_db_sim():
    """DB-basis 변이 gold — `mutation_diff` 가 여전히 제 일을 하는지(부정통제)."""
    act = {"name": "call_discoverable_agent_tool",
           "arguments": {"agent_tool_name": "apply_checking_account_credit_5829",
                         "arguments": {"account_id": "ca_1", "amount": "9.50",
                                       "credit_type": "fee_refund"}},
           "action_id": "b1"}
    return {
        "task_id": "fixture_db", "trial": 0,
        "reward_info": {"reward": 1.0, "reward_basis": ["DB"],
                        "action_checks": [{"action": act, "action_match": True,
                                           "tool_type": "write"}]},
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"name": "call_discoverable_agent_tool", "id": "y1",
                 "arguments": {"agent_tool_name": "apply_checking_account_credit_5829",
                               "arguments": {"account_id": "ca_1", "amount": 9.5,
                                             "credit_type": "fee_refund"}}}]},
            {"role": "tool", "id": "y1", "content": "Credit applied."},
        ]}


def test_a16():
    print("\n[A16/OL-49] `t2_forensic.action_diff` — ACTION-basis 대조표")
    import t2_forensic as F

    s = _fixture_action_sim()
    md = F.mutation_diff(s)
    chk(not md["gold"],
        "⑴ 양성대조(결손 재현): `mutation_diff` 는 이 모양에서 gold **전 항목 빈칸**", md["gold"])

    ad = F.action_diff(s)
    chk(ad["basis"] == ["ACTION"] and ad["reward"] == 0.0,
        "채점 축과 reward 를 함께 싣는다([[69]])", (ad["basis"], ad["reward"]))
    chk(ad["n_gold"] == 2, "⑵ 수리 후: gold 액션 2행이 보인다(필터 0)", ad["n_gold"])
    chk(ad["n_matched"] == 1 and len(ad["missing"]) == 1,
        "⑵ MATCH 1 · MISSING 1 (권위 = 벤치 `action_match`)",
        (ad["n_matched"], len(ad["missing"])))
    miss = ad["missing"][0]
    chk(miss["inner"] == "get_user_dispute_history_7291",
        "⑵ 래퍼 안쪽 대상 도구를 이름으로 싣는다", miss["inner"])
    chk(miss["called_name"] is False and miss["called_exact"] is False,
        "⑵ 원인 칸: 성공한 같은 실행이 없다")
    chk(miss["blocked"] and miss["blocked"]["deny"] == "ours",
        "⑵ 원인 칸: 시도했으나 **우리 층**이 막았다(deny 주체 표기·[[55]])",
        miss.get("blocked"))
    chk(len(ad["blocked"]) == 1, "blocked 집계", len(ad["blocked"]))

    # ⑶ 부정통제 — DB-basis 에서 `mutation_diff` 는 종전대로 작동하고 두 표가 어긋나지 않는다
    s2 = _fixture_db_sim()
    md2 = F.mutation_diff(s2)
    ad2 = F.action_diff(s2)
    chk(len(md2["gold"]) == 1 and len(md2["matched"]) == 1 and md2["clean"],
        "⑶ 부정통제: DB-basis 에서 `mutation_diff` 는 그대로 작동한다",
        (len(md2["gold"]), len(md2["matched"])))
    chk(ad2["n_gold"] == 1 and ad2["clean"],
        "⑶ 부정통제: 같은 sim 에서 `action_diff` 도 어긋나지 않는다")
    chk(F.reward_basis(s2) == ["DB"] and F.reward_basis({}) == [],
        "reward_basis: 채점표 부재는 빈 리스트")

    # ⑶ 부정통제 — 채점표가 없는 sim 은 조용히 빈 표(예외 0)
    ad3 = F.action_diff({"messages": []})
    chk(ad3["n_gold"] == 0 and ad3["clean"] and ad3["reward"] is None,
        "⑶ 부정통제: `reward_info` 부재 sim 은 빈 표(074#0·079#1 형)")

    # 라이브 코퍼스 — 있으면 실물에서 한 번 돌린다([[67]] 死배선 방지)
    tag = "bank_t7336_halfB_20260821b"
    p = os.path.join(os.path.abspath(F.BASE), tag + ".results.json.gz")
    if os.path.exists(p):
        sims = F.sims(tag, suffix=".results.json.gz")
        act = [x for x in sims if "ACTION" in F.reward_basis(x)]
        rows = [(F.sim_key(x), F.action_diff(x), F.mutation_diff(x)) for x in act]
        chk(bool(rows), "실물 ACTION-basis sim 을 찾았다 (%d)" % len(rows))
        chk(all(not m["gold"] for _k, _a, m in rows),
            "실물: 그 sim 들에서 `mutation_diff` gold = 0 (OL-49 재현)")
        chk(all(a["n_gold"] > 0 for _k, a, _m in rows),
            "실물: `action_diff` 는 gold 행을 낸다",
            [(k, a["n_gold"], a["n_matched"], a["reward"]) for k, a, _m in rows])
        chk(all((a["reward"] == 1.0) == a["clean"] for _k, a, _m in rows),
            "실물: MATCH 전부 ↔ reward 1.0 이 일치한다(ACTION 축)",
            [(k, a["n_matched"], a["n_gold"], a["reward"]) for k, a, _m in rows])
    else:
        note("실물 코퍼스 없음 — 라이브 대조는 건너뜀 (%s)" % tag)


# ═══════════════════════ A11 부정통제만 (OL-44 · 수리는 G1·G2) ═══════════════════
def test_a11_negative_control():
    print("\n[A11/OL-44] 부정통제 — operator 인자 치환이 **정답이던 사례**가 있었나")
    base = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                        "reports", "facet_rft_2026", "sim_results"))
    logs = sorted(glob.glob(os.path.join(base, "*.log.gz")))
    if not logs:
        note("로그 코퍼스 없음 — 부정통제 건너뜀")
        return
    surf = load_json("a2/env_surface.json")
    reg = set()
    for dom in surf:
        reg |= set(surf[dom]["tools"])
    rx = re.compile(r"\[T2_GROUND\] substituted arg=([A-Za-z0-9_]+) val=(.*?) -> (.*)$")
    opargs = ("agent_tool_name", "discoverable_tool_name", "user_tool_name", "tool_name")
    cnt = collections.Counter()
    tags = set()
    for p in logs:
        try:
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                txt = f.read()
        except Exception:
            continue
        for line in txt.split("\n"):
            m = rx.search(line)
            if not m:
                continue
            k, new = m.group(1), m.group(3).strip().split("2026-")[0].strip()
            if k not in opargs:
                cnt["other_arg"] += 1
                continue
            cnt["oparg"] += 1
            tags.add(os.path.basename(p)[:-7])
            if new in reg:
                cnt["landed_on_real_tool"] += 1
            if "_" in new:
                cnt["looks_like_tool_name"] += 1
    print("    로그 %d편 · operator-인자 치환 %d회(런 %d개) · 그 외 인자 치환 %d회"
          % (len(logs), cnt["oparg"], len(tags), cnt["other_arg"]))
    chk(cnt["oparg"] > 0, "치환 사건이 코퍼스에 실재한다(계측 생존)", cnt["oparg"])
    chk(cnt["landed_on_real_tool"] == 0,
        "부정통제: 치환값이 **실재 도구 이름**이던 사례 = 0 (치환이 정답이던 적이 없다)",
        cnt["landed_on_real_tool"])
    chk(cnt["looks_like_tool_name"] == 0,
        "부정통제: 치환값에 `_` 조차 없다 — tau2 도구명은 전부 `_` 를 포함한다",
        cnt["looks_like_tool_name"])
    note("A11 수리 자체는 `t2_gate_patch._grounded_candidates`(정의 1 · 호출부 2)에 있고 "
         "그 파일은 G1·G2 소유라 여기서 고치지 않았다 — 보고서 §A11 참조")


def main():
    print("test_t7336_g3_decl_modules — A4·A6·A8·A10·A16 (+A11 부정통제)")
    for fn in (test_a4, test_a6, test_a8, test_a10, test_a16,
               test_a11_negative_control):
        fn()
    print("\nRESULT: %s" % ("ALL PASS" if not FAILED else "FAIL %d" % len(FAILED)))
    for f in FAILED:
        print("  - %s" % f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
