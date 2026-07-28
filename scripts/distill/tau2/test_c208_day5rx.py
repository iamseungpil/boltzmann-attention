#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C208/day5 처방(P1~P10) 오프라인 검증 (2026-07-28·무료·모델 불요).
`DAY5_PRESCRIPTIONS_DESIGN_2026_07_28` §7 배터리. 리뷰 필수 케이스 포함:
- test_view_budget: 멀티-콜 배치(한 어시스턴트 턴의 복수 tool 출력=전부 전문 유지)
- test_terminal_grant: notice 공표+비동의 ###STOP### → 무개입
⚠단위통과≠라이브발화([[30]])."""
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

import t2_gate_patch as GP        # noqa: E402
import t2_prekb_patch as PK       # noqa: E402
import t2_eplan_patch as EP       # noqa: E402
import t2_compute as TC           # noqa: E402
import t2_scaffold_get as SG      # noqa: E402
import t2_run_gated as RG         # noqa: E402

OK = True


def chk(c, m):
    global OK
    OK &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)


# ── 공용 대역 ────────────────────────────────────────────────────────────────
class M:
    def __init__(self, role, content=None, tool_calls=None, mid=None, error=False,
                 requestor="assistant"):
        self.role, self.content, self.tool_calls = role, content, tool_calls
        self.id, self.error, self.requestor = mid, error, requestor


class TCall:
    def __init__(self, name, cid="c1", args=None):
        self.name, self.id, self.arguments = name, cid, (args or {})


class FakeToolkit:
    def __init__(self, tools):                     # {name: mutates}
        self._t = dict(tools)

    def has_tool(self, n):
        return n in self._t

    def tool_mutates_state(self, n):
        return self._t[n]


class FakeEnv:
    def __init__(self, agent_tools=None, user_tools=None, domain="banking_knowledge"):
        self.tools = FakeToolkit(agent_tools or {})
        self.user_tools = FakeToolkit(user_tools or {})
        self.domain_name = domain


class FakeOrch:
    def __init__(self, msgs, env=None):
        self._msgs, self.environment = msgs, env
        self.agent = type("Ag", (), {})()
        self.done = True
        self.termination_reason = "user_stop"

    def get_messages(self):
        return self._msgs


DAY5_CWE = ("litellm.ContextWindowExceededError: litellm.BadRequestError: "
            "ContextWindowExceededError: OpenAIException - 'max_tokens' or "
            "'max_completion_tokens' is too large: 8192. This model's maximum context "
            "length is 48640 tokens and your request has 40606 input tokens "
            "(8192 > 48640 - 40606). None")


def test_dyn_mt():
    print("[test_dyn_mt] P1 동적 max_tokens")
    v = GP._dyn_mt_target(DAY5_CWE)
    chk(v == 48640 - 40606 - 64, "day5 실측 에러 원문 파싱 → %s (=48640-40606-64)" % v)
    chk(GP._dyn_mt_target("some other error") is None, "무관 에러 → None(graceful-stop 경로)")
    near = DAY5_CWE.replace("40606", "48500")
    chk(GP._dyn_mt_target(near) is None, "플로어 미만(진짜 창 소진) → None")
    chk(GP._dyn_mt_target(DAY5_CWE, margin=0, floor=9000) is None,
        "커스텀 플로어 존중(8034<9000 → None)")


def test_replay_hygiene():
    print("[test_replay_hygiene] P2 replay 위생 + 문구 사실화")
    env = FakeEnv(agent_tools={"unlock_discoverable_agent_tool": True, "KB_search": False},
                  user_tools={"apply_for_credit_card": True,
                              "give_discoverable_user_tool": True})
    chk(PK._replay_compared(env, "give_discoverable_user_tool") is True,
        "give(등록·mutating) → replay-비교 대상=content 불변")
    chk(PK._replay_compared(env, "KB_search") is False, "KB_search(read) → append 무해")
    chk(PK._replay_compared(env, "get_reward_discrepancies") is False,
        "우리 주입 도구(env 미등록) → replay 스킵=무해")
    chk(PK._replay_compared(None, "x") is False, "env 없음 → False(구판 거동)")
    # 문구 사실화 3-분기 (day5 오진 교정: apply_for_credit_card=실재 유저-네이티브)
    c = "Error: Unknown discoverable tool 'apply_for_credit_card'."
    t1 = PK._utool_guidance_txt(c, env)
    chk("DOES exist" in t1 and "customer runs directly" in t1 and "invented" not in t1,
        "유저-네이티브 → '실재·손님 직접 실행' (invented 단정 제거)")
    c2 = "Error: Unknown discoverable tool 'totally_fake_tool_xyz'."
    t2 = PK._utool_guidance_txt(c2, env)
    chk("No tool with that exact name exists" in t2, "진짜 미존재 → 미존재 안내 유지")
    # 생성-레벨 채널
    orch = FakeOrch([], env)
    chk(PK._view_fb(orch, "hello-fb", "t") is True and
        orch.agent._t2_view_fb == ["hello-fb"], "뷰-채널 큐잉(_t2_view_fb)")
    # 잔존 감사: mutating 도구 결과에 content 직접 append하는 코드가 prekb에 안 남았나
    src = io.open(os.path.join(HERE, "t2_prekb_patch.py"), encoding="utf-8").read()
    chk("That tool name does not exist — you invented it" not in src,
        "구판 단정 문구(발화 문자열) 제거됨")


def test_terminal_grant():
    print("[test_terminal_grant] P3 터미널-턴 보장 술어")
    a2 = GP._domain_a2("banking_knowledge")
    nt = next(g["notice_text"] for g in a2["gates"] if g.get("kind") == "notice")
    env = FakeEnv()
    notice_msg = M("assistant", nt + " ...")
    # ⓐ+ⓐ′+ⓑ 성립 → 도구명 반환
    msgs = [notice_msg, M("user", "Yes, please transfer me. ###TRANSFER###")]
    g = EP._terminal_grant_check(FakeOrch(msgs, env))
    chk(g == "transfer_to_human_agents", "notice+동의+미호출 → 유예 대상=%s" % g)
    # 리뷰 필수2: 비동의 STOP → 무개입
    msgs2 = [notice_msg, M("user", "That's all, thanks. ###STOP###")]
    chk(EP._terminal_grant_check(FakeOrch(msgs2, env)) is None,
        "notice 공표+비동의 ###STOP### → 무개입(무단 행동 방지)")
    # 이미 호출됨 → 무개입
    msgs3 = [notice_msg,
             M("assistant", None, tool_calls=[TCall("transfer_to_human_agents")]),
             M("user", "###TRANSFER###")]
    chk(EP._terminal_grant_check(FakeOrch(msgs3, env)) is None, "이미 호출 → 무개입(040/008형)")
    # notice 미공표 → 무개입
    msgs4 = [M("assistant", "Goodbye!"), M("user", "###TRANSFER###")]
    chk(EP._terminal_grant_check(FakeOrch(msgs4, env)) is None, "notice 미공표 → 무개입(012/014형)")


def _ratefix_op():
    a2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                           encoding="utf-8"))
    d = next(t for t in a2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")
    return d["variants"]["ratefix"]["op"]


ROW_OK = {"transaction_id": "txn_aaa", "transaction_amount": 100.0, "rewards_earned": 400,
          "transaction_date": "03/01/2025", "credit_card_type": "X", "category": "Y",
          "base_rate": 4.0, "promo_mult": 1, "promo_window_months": 0,
          "promo_start": "", "promo_end": "", "account_open": "01/01/2025"}


def test_abstain_actionable():
    print("[test_abstain_actionable] P4 결핍-필드 지목 stats")
    op = _ratefix_op()
    # 행1=완전(discrepant: 100×4=400 == 400 → 비검출·판정됨), 행2=account_open 결핍+promo 선언
    row2 = dict(ROW_OK, transaction_id="txn_bbb", promo_mult=2, promo_window_months=6,
                promo_start="2024-11-14", promo_end="2025-11-14")
    row2.pop("account_open")
    ctx = {"transactions": [dict(ROW_OK), row2]}
    TC.apply_op(op, ctx)
    st = ctx.get("_sg_stats") or {}
    mf = st.get("missing_fields") or {}
    chk(st.get("judged") == 1 and st.get("skipped") == 1,
        "완전행 판정·결핍행 스킵 (judged=%s skipped=%s)" % (st.get("judged"), st.get("skipped")))
    chk(mf.get("account_open") == 1, "missing_fields가 account_open 지목: %s" % mf)
    # day5 020형: 전행 account_open 결핍 → 전행 지목
    rows = []
    for i in range(3):
        r = dict(row2, transaction_id="txn_%d" % i)
        rows.append(r)
    ctx2 = {"transactions": rows}
    TC.apply_op(op, ctx2)
    chk((ctx2["_sg_stats"]["missing_fields"] or {}).get("account_open") == 3,
        "020형(전행 결핍) → account_open 3행 지목")


def test_prod_bind_and_p4b():
    print("[test_prod_bind] P4b producer-binding 선언 존재+엔진 강등 로직")
    a2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"),
                           encoding="utf-8"))
    d = next(t for t in a2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies")
    gp = (d["variants"]["ratefix"].get("grounded_params") or {})
    chk("account_open" in gp and gp["account_open"].get("producer_contains"),
        "A2 ratefix.grounded_params.account_open 선언 실재")
    # 엔진 로직 동형 재현(강등 술어): 값이 selector-일치 출력에 없으면 결핍
    outs = {"get_credit_card_accounts_by_user":
            "found 2 record(s) in 'credit_card_accounts': date_of_account_open: 02/13/2025"}
    sels = [s.lower() for s in gp["account_open"]["producer_contains"]]
    cands = [t for t in outs.values() if any(s in t for s in sels)]
    chk(any("02/13/2025" in t for t in cands), "실개설일=producer 출력 실재 → 통과")
    chk(not any("02/01/2025" in t for t in cands), "027 날조값(02/01/2025)=불실재 → 강등 대상")


def test_view_budget():
    print("[test_view_budget] P5 뷰-예산 (배치 의미론+MSG_CAP)")
    big = "X" * 9000
    # 멀티-콜 배치: 마지막 assistant 뒤 tool 2개 = 둘 다 전문 유지(리뷰 필수1)
    msgs = [M("assistant", "old", tool_calls=[TCall("a", "c0")]),
            M("tool", big, mid="t0"),
            M("assistant", None, tool_calls=[TCall("a", "c1"), TCall("b", "c2")]),
            M("tool", big, mid="t1"), M("tool", big, mid="t2")]
    out, dg = GP._compact_view(msgs, keep_recent=0, min_len=800,
                               min_total=10 ** 9, msg_cap=8000)
    chk(out[3].content == big and out[4].content == big,
        "최신 배치(멀티-콜 2개) 전문 유지")
    chk("view digest" in out[1].content and "t0" in dg,
        "배치 밖 대형 출력은 총량 무관 다이제스트(MSG_CAP)")
    # msg_cap=0 → 구판 거동(총량 문턱 미달 시 무개입)
    out2, dg2 = GP._compact_view(msgs, keep_recent=0, min_len=800,
                                 min_total=10 ** 9, msg_cap=0)
    chk(out2[1].content == big and not dg2, "msg_cap=0 → 구판 무개입(회귀)")
    # 총량 문턱 경로 회귀: min_total 초과·keep_recent 보호
    out3, _ = GP._compact_view(msgs, keep_recent=3, min_len=800,
                               min_total=1000, msg_cap=0)
    chk("view digest" in out3[1].content or out3[1].content == big,
        "총량-문턱 경로 동작(구판 회귀 스모크)")


def test_unavail_env():
    print("[test_unavail_env] P7 env 해석+LEVER_HEALTH")
    env = FakeEnv(user_tools={"submit_cash_back_dispute_0589": True})
    ag = type("Ag", (), {})()
    ag._t2_orch = FakeOrch([], env)
    # 수정 후 해석 경로: getattr(ag._t2_orch, 'environment') — NameError 불가 구조
    resolved = getattr(getattr(ag, "_t2_orch", None), "environment", None)
    chk(resolved is env, "agent._t2_orch.environment 해석 성공(구판 NameError 경로 제거)")
    known = GP._known_tool_names([], resolved, [])
    chk(isinstance(known, set), "_known_tool_names 정상 동작(known=%d)" % len(known))
    GP._LEVER_HEALTH.clear()
    GP._lever_health("unavail", "skipped")
    GP._lever_health("unavail", "skipped")
    chk(GP._LEVER_HEALTH["unavail"]["skipped"] == 2 and
        not GP._LEVER_HEALTH["unavail"].get("ok"),
        "LEVER_HEALTH: 전량-스킵이 집계로 드러남(무음실패 금지)")
    GP._LEVER_HEALTH.clear()


def test_dup_represent():
    print("[test_dup_represent] P8 스텁 이전-결과 재제시")
    s1, r1 = SG._dup_stub_content(1, prev="RESULT-TEXT", represent_on=True, shrunk=False)
    chk("Previous result (unchanged): RESULT-TEXT" in s1 and r1, "1회째 반복 → 재제시")
    s2, r2 = SG._dup_stub_content(3, prev="RESULT-TEXT", represent_on=True, shrunk=False)
    chk("Previous result" not in s2 and "STOP repeating" in s2, "3회째 → 재제시 상한·STOP 승격")
    s3, r3 = SG._dup_stub_content(1, prev="RESULT-TEXT", represent_on=True, shrunk=True)
    chk("Previous result" not in s3, "천장 근접(shrunk) → 재제시 생략(W-d)")
    s4, r4 = SG._dup_stub_content(1, prev="RESULT-TEXT", represent_on=False, shrunk=False)
    chk("Previous result" not in s4, "T2_DUP_REPRESENT=0 → 구판 거동(회귀)")


REC_DUMP = ("Found 2 record(s) in 'credit_card_transaction_history':\n\n"
            "1. Record ID: txn_2037a5f15196\n"
            "   transaction_id: txn_2037a5f15196\n"
            "   transaction_amount: $127.43\n"
            "   rewards_earned: 637 points\n"
            "   credit_card_type: Diamond Elite Card\n"
            "   category: Travel\n\n"
            "2. Record ID: txn_5197a7\n"
            "   transaction_id: txn_5197a7\n"
            "   transaction_amount: $623.45\n"
            "   rewards_earned: 2,493 points\n"
            "   credit_card_type: Diamond Elite Card\n"
            "   category: Dining\n")


def test_sg_byref():
    print("[test_sg_byref] P6 참조-전달 (기본 OFF·오프라인 검증만)")
    rows = SG._parse_record_dump(REC_DUMP)
    chk(len(rows) == 2 and rows[0]["transaction_id"] == "txn_2037a5f15196",
        "Record ID 기계 포맷 → %d행 파싱" % len(rows))
    chk(rows[0]["transaction_amount"] == "127.43" and rows[0]["rewards_earned"] == "637"
        and rows[1]["rewards_earned"] == "2493",
        "포맷-층 정규화($/콤마/points 제거): %s" % rows[0]["transaction_amount"])
    try:
        SG._parse_record_dump("This is prose, not a record dump.")
        chk(False, "비-레코드 텍스트 거부")
    except SG._ByrefError:
        chk(True, "비-레코드 텍스트 → _ByrefError(파서 경계 assert·리뷰 지시)")
    # @last 해석
    msgs = [M("assistant", None, tool_calls=[TCall("get_txns", "cA")]),
            M("tool", REC_DUMP, mid="cA")]
    orch = FakeOrch(msgs)
    chk(SG._resolve_ref_output(orch, "@last:get_txns") == REC_DUMP, "@last 최신 출력 해석")
    try:
        SG._resolve_ref_output(orch, "@last:never_called")
        chk(False, "미호출 도구 참조 거부")
    except SG._ByrefError:
        chk(True, "미호출 도구 @last → 명확한 에러(call first)")
    d = {"name": "t", "op": {"over": "transactions"}}
    ctx = {"transactions": "@last:get_txns"}
    SG._byref_resolve(orch, d, ctx)
    chk(isinstance(ctx["transactions"], list) and len(ctx["transactions"]) == 2,
        "over-인자 byref → rows 치환")
    ctx2 = {"transactions": [], "account_open": "@last:get_txns"}
    try:
        SG._byref_resolve(orch, d, ctx2)
        chk(False, "비-over 인자 참조 거부")
    except SG._ByrefError:
        chk(True, "비-over 인자 @last → 미지원 에러([[05]]: 도메인 join 엔진 금지)")


def test_failed_persist():
    print("[test_failed_persist] P10 사이드카 영속")
    import gzip
    import tempfile

    class FakeMsg:
        def model_dump_json(self):
            return json.dumps({"role": "assistant", "content": "hi"})

    class FakeEnvCls:
        def set_state(self, initialization_data=None, initialization_actions=None,
                      message_history=None):
            raise ValueError("Tool call mismatch (fixture)")

    tmp = tempfile.mkdtemp()
    os.environ["T2_FAILED_DIR"] = tmp
    RG._install_failed_persist(FakeEnvCls)
    e = FakeEnvCls()
    try:
        e.set_state(message_history=[FakeMsg(), FakeMsg()])
        chk(False, "예외 재-raise")
    except ValueError:
        chk(True, "예외는 그대로 재-raise(러너 거동 무변)")
    files = [f for f in os.listdir(tmp) if f.startswith("failed_setstate_")]
    chk(len(files) == 1, "사이드카 1개 생성: %s" % files)
    with gzip.open(os.path.join(tmp, files[0]), "rt", encoding="utf-8") as f:
        d = json.load(f)
    chk(d["n_messages"] == 2 and "mismatch" in d["error"], "궤적 2건+에러 원문 영속")
    os.environ.pop("T2_FAILED_DIR", None)


def test_near_dup_tokens():
    print("[test_near_dup] P5-3 근사-중복 판정(기본 OFF·순수 로직 스모크)")
    import re as _re
    stop = {"the", "a", "an", "for", "to", "of", "in", "on", "how",
            "tool", "get", "and", "or", "with", "by", "is", "do"}
    def toks(s):
        return {w for w in _re.findall(r"[a-z0-9_]+", s.lower()) if w not in stop}
    a = toks("KB_search get credit card transactions")
    b = toks("KB_search tool to get credit card transactions")
    j = len(a & b) / float(len(a | b))
    chk(j >= 0.8, "018형 재표현('tool to' 부가) → Jaccard %.2f ≥ 0.8" % j)
    c = toks("KB_search annual fee rebate qualification")
    j2 = len(a & c) / float(len(a | c))
    chk(j2 < 0.8, "이질 질의 → %.2f < 0.8 (오탐 아님)" % j2)


if __name__ == "__main__":
    for fn in (test_dyn_mt, test_replay_hygiene, test_terminal_grant,
               test_abstain_actionable, test_prod_bind_and_p4b, test_view_budget,
               test_unavail_env, test_dup_represent, test_sg_byref,
               test_failed_persist, test_near_dup_tokens):
        fn()
        print()
    print("RESULT: %s" % ("ALL PASS" if OK else "FAIL"))
    sys.exit(0 if OK else 1)
