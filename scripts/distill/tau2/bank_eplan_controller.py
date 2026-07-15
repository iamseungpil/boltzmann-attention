# -*- coding: utf-8 -*-
"""bank_eplan_controller.py — banking E-PLAN 아우터 loop 컨트롤러 (엔진 + 오프라인 harness).

정본 설계 = `reports/facet_rft_2026/BANK_EPLAN_CONTROLLER_DESIGN_2026_07_15.md` (handoff §0-a).

  엔진(도메인일반·[[05]]·리터럴0)  = {FIND-enumerate · coverage-track · per-item COMPUTE · ASK · H_min-stop}
                                     도메인지식(liability 규칙·도구명·필드명)은 전부 ABox
                                     (a2/banking_knowledge.gate.json: eplan/compute_ops/reference_filter)서 로드.
  harness(오프라인·[[09]]·무료)     = 17 궤적(C:/tmp/traj)서 per-sim state 복원 → 엔진 walk → ceiling 재현.

세 arm (설계 §3·확립수치 대조):
  COMPUTE-only : COMPUTE만(제출된 dispute 한정)                → bank_operator_replay 19.2% *정확 재현*(harness 검증)
  LOAD         : COMPUTE + FIND-enumerate + coverage(reach 완화) → ~60% 상한
  GAP 잔여      : open-world ASK + ⋈ 비결정 + gather              → ~40%

사용:
  py bank_eplan_controller.py            # 오프라인 replay (17 궤적·무료)
  py bank_eplan_controller.py --selftest # 엔진 순수 단위테스트 (궤적·db 없이)
"""
import json, glob, re, sys, io, os, argparse
from datetime import datetime
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── ABox 로드 (도메인지식은 여기서만·엔진 하드코딩 0·[[05]]) ─────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ABOX_PATH = os.path.join(_HERE, "a2", "banking_knowledge.gate.json")


def load_abox(path=_ABOX_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 도메인일반 파서 유틸 (금액·날짜·정규화) ─────────────────────────────────────
def _pd(s):
    for f in ("%m/%d/%Y", "%m/%d/%y", "%m/%d"):
        try:
            return datetime.strptime(str(s).strip()[:10], f)
        except Exception:
            pass
    return None


def _amt(v):
    try:
        return round(abs(float(re.sub(r"[$,]", "", str(v)))), 2)
    except Exception:
        return None


def _norm(v):
    """gold 대조용 정규화: 금액→float, bool어휘→bool, 그외 소문자 문자열."""
    s = str(v).strip().lower()
    m = re.sub(r"[$,]", "", s)
    try:
        return round(float(m), 2)
    except Exception:
        pass
    return {"true": True, "yes": True, "false": False, "no": False}.get(s, s)


# ══════════════════════════════════════════════════════════════════════════════
#  엔진 = 도메인일반 (dict-in/dict-out·tau2 임포트 0·오프라인 단위테스트 가능)
# ══════════════════════════════════════════════════════════════════════════════

class AboxOpError(Exception):
    pass


def eval_op(spec, params):
    """ABox 연산 스펙(compute_ops)의 결정론 해석기. 도메인지식 0 — 스펙이 규칙을 담는다.
    지원 op: lookup_table · days_between · ref · (스칼라 리터럴).
    params = 제출 인자 dict (transaction_date/discovery_date/disputed_amount 등)."""
    if isinstance(spec, str) and spec.startswith("params."):
        return params.get(spec[len("params."):])      # "params.X" ref 축약
    if not isinstance(spec, dict):
        return spec                                   # 스칼라 리터럴 (예: 50, 500)
    op = spec.get("op")
    if op is None:
        raise AboxOpError("no op in %r" % spec)
    if op == "ref":
        path = spec["path"]
        assert path.startswith("params."), path
        return params.get(path[len("params."):])
    if op == "days_between":
        a = _pd(params.get(spec["a"].split(".", 1)[1]))
        b = _pd(params.get(spec["b"].split(".", 1)[1]))
        if a is None or b is None:
            return None
        return (b - a).days
    if op == "lookup_table":
        key = eval_op(spec["key"], params)
        if key is None:
            return None
        for row in spec["table"]:
            cmp = row.get("cmp")
            if cmp is None:                            # default(마지막 행)
                return eval_op(row["result"], params)
            thr = row["thr"]
            if cmp == "<=" and key <= thr:
                return eval_op(row["result"], params)
            if cmp == "<" and key < thr:
                return eval_op(row["result"], params)
            if cmp == ">=" and key >= thr:
                return eval_op(row["result"], params)
            if cmp == ">" and key > thr:
                return eval_op(row["result"], params)
        return None
    # ── 산술 op (compute 규칙 확장·2026-07-16·gold-fit 확증분) ──
    if op in ("subtract", "add", "multiply", "divide"):
        a = eval_op(spec["a"], params); b = eval_op(spec["b"], params)
        if a is None or b is None:
            return None
        try:
            a, b = float(a), float(b)
        except Exception:
            return None
        if op == "subtract": r = a - b
        elif op == "add": r = a + b
        elif op == "multiply": r = a * b
        else: r = a / b if b else None
        return r
    if op == "round":
        v = eval_op(spec["value"], params)
        if v is None:
            return None
        return round(float(v), spec.get("ndigits", 2))
    if op == "const":
        return spec.get("value")
    raise AboxOpError("unknown op %r" % op)


def compute_fields(tool_family, params, abox):
    """per-item COMPUTE (엔진 §3): 이 도구의 compute_ops 각 필드를 params서 결정론 재계산.
    반환 {field: computed_value}. 도구에 compute_ops 없으면 {} (COMPUTE no-op)."""
    ops = abox.get("compute_ops") or {}
    spec = None
    for key, fieldmap in ops.items():
        if key in tool_family or tool_family in key:   # family 매칭(_NN suffix 무시)
            spec = fieldmap
            break
    if not spec:
        return {}
    out = {}
    for field, opspec in spec.items():
        out[field] = eval_op(opspec, params)
    return out


def compute_field_names(tool_family, abox):
    ops = abox.get("compute_ops") or {}
    for key, fieldmap in ops.items():
        if key in tool_family or tool_family in key:
            return set(fieldmap.keys())
    return set()


def classify_dispute(gold, record, tool_family, abox, on_record_fields):
    """단일 dispute를 컨트롤러 후 판정 (엔진 로직·per-case 정직).

    record          = 컨트롤러가 제출에 쓸 인자(agent 제출분 또는 enumerate가 surface한 gold-record).
    on_record_fields= 거래record서 오는 필드(FIND-enumerate가 정정 가능). off-record(user-gather)는 제외.

    반환 dict: {ok, wrong_after, cause}
      cause ∈ {"pass","compute","gather","xmatch"} (실패 시 잔여 원인·[[08]] per-case)."""
    comp_names = compute_field_names(tool_family, abox)
    # 1) FIND-enumerate 효과: on-record 필드는 surfaced record(=gold) 값으로 정정
    eff = dict(record)
    for k in on_record_fields:
        if k in gold:
            eff[k] = gold[k]
    # 2) per-item COMPUTE: compute 필드를 결정론 재계산 (eff의 입력으로)
    computed = compute_fields(tool_family, eff, abox)
    for k, v in computed.items():
        if v is not None:
            eff[k] = v
    # 3) gold 대조
    wrong = {k for k, gv in gold.items()
             if k != "transaction_id" and _norm(eff.get(k)) != _norm(gv)}
    if not wrong:
        return {"ok": True, "wrong_after": set(), "cause": "pass"}
    # 잔여 원인 분류
    if wrong & comp_names:
        cause = "compute"          # compute 필드가 여전히 틀림 = off-record 입력(discovery_date) gather 오류
    elif wrong - on_record_fields - comp_names:
        cause = "xmatch"           # on-record도 compute도 아닌 필드 = ⋈/비결정 → ASK
    else:
        cause = "gather"
    return {"ok": False, "wrong_after": wrong, "cause": cause}


def controller(state, abox, arm="LOAD", on_record_fields=("transaction_date", "disputed_amount",
                                                          "merchant", "description", "type")):
    """도메인일반 아우터 loop. state(궤적 복원) → sim-level 판정.

    state = {gold: {tid:args}, subs: {tid:args}, seen: set(tid), universe: set(tid), tool_family: str}
    arm   = "COMPUTE"(제출된 것만·reach 미완화) | "LOAD"(FIND-enumerate+coverage로 reach 완화)

    반환 {sim_pass, per_dispute:[...], stop_reason}."""
    gold, subs, seen, universe = state["gold"], state["subs"], state["seen"], state["universe"]
    tf = state["tool_family"]
    on_rec = set(on_record_fields)
    results = []
    all_ok = True
    for tid, ga in gold.items():
        # ── FIND-enumerate + coverage-track: reachability 판정 (H_min: 미커버=미완) ──
        if tid in subs:
            reach, record = "agent_submitted", dict(subs[tid])
        elif arm == "LOAD" and tid in seen:
            reach, record = "closed_world_A_seen", dict(ga)     # 열거가 surface(id-correct free)
        elif arm == "LOAD" and tid in universe:
            reach, record = "closed_world_B_queryable", dict(ga)
        else:
            reach, record = "unreachable", None
        if record is None:
            results.append({"tid": tid, "reach": reach, "ok": False, "cause": "open_world_ASK"})
            all_ok = False
            continue
        # enumerate가 surface한 dispute(agent 미제출)는 on-record 정정을 항상 적용;
        # agent 제출분은 LOAD arm서만 on-record 정정(열거가 record 필드 정정) 적용.
        apply_onrec = on_rec if (arm == "LOAD" or reach != "agent_submitted") else set()
        verdict = classify_dispute(ga, record, tf, abox, apply_onrec)
        verdict.update({"tid": tid, "reach": reach})
        results.append(verdict)
        if not verdict["ok"]:
            all_ok = False
    stop = "H_min_floor(all_covered)" if all_ok else "residual(uncovered_or_wrong)"
    return {"sim_pass": all_ok, "per_dispute": results, "stop_reason": stop}


# ══════════════════════════════════════════════════════════════════════════════
#  harness = 궤적 state 복원 (오프라인·tau2 임포트 0)
# ══════════════════════════════════════════════════════════════════════════════

def _nd(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return {}
    return x if isinstance(x, dict) else {}


_fam = lambda n: re.sub(r"_\d+$", "", str(n))
_TXN = re.compile(r"txn_[0-9a-f]+")


def reconstruct_states(files, abox):
    """실패 sim마다 state 복원. universe = task별 어느 run이든 tool result에 등장한 txn."""
    rf = abox.get("reference_filter") or {}
    dispatch = rf.get("dispatch_tool", "call_discoverable_agent_tool")
    disp_fam = rf.get("tool_prefix", "transaction_dispute")
    # 1st pass: per-task universe (queryable txn 상한)
    universe = {}
    data = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        data[f] = d
        for s in d.get("simulations", []):
            t = str(s.get("task_id"))
            u = universe.setdefault(t, set())
            for m in (s.get("messages") or []):
                if m.get("role") == "tool":
                    u |= set(_TXN.findall(str(m.get("content"))))
    # 2nd pass: 실패 sim state
    states = []
    for f, d in data.items():
        model = f.replace("\\", "/").split("/")[-1].replace("_banking.json", "")
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue                                 # 실패 sim만
            t = str(s.get("task_id"))
            # gold disputes
            gold, tool_family = {}, ""
            for ac in (ri.get("action_checks") or []):
                a = ac.get("action") or {}
                atn = _nd(a.get("arguments")).get("agent_tool_name", "")
                if disp_fam in _fam(atn):
                    ga = _nd(_nd(a.get("arguments")).get("arguments"))
                    tid = str(ga.get("transaction_id") or "")
                    if tid:
                        gold[tid] = ga
                        tool_family = _fam(atn)
            if not gold:
                continue
            # agent 제출
            subs = {}
            seen = set()
            for m in (s.get("messages") or []):
                if m.get("role") in ("tool", "user"):
                    seen |= set(_TXN.findall(str(m.get("content"))))
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name") == dispatch and disp_fam in _fam(_nd(tc.get("arguments")).get("agent_tool_name", "")):
                        a = _nd(_nd(tc.get("arguments")).get("arguments"))
                        tid = str(a.get("transaction_id") or "")
                        if tid:
                            subs.setdefault(tid, a)
            states.append({
                "model": model, "task_id": t,
                "gold": gold, "subs": subs, "seen": seen,
                "universe": universe.get(t, set()), "tool_family": tool_family,
            })
    return states


def run_offline(abox):
    files = glob.glob("C:/tmp/traj/*_banking.json")
    states = reconstruct_states(files, abox)
    summary = {}
    gap_cause = Counter()          # LOAD arm 실패 sim의 잔여 원인 (per-case)
    for arm in ("COMPUTE", "LOAD"):
        npass = 0
        for st in states:
            r = controller(st, abox, arm=arm)
            if r["sim_pass"]:
                npass += 1
            elif arm == "LOAD":
                # 잔여원인: 이 sim이 실패한 지배 원인 (첫 실패 dispute)
                for pd_ in r["per_dispute"]:
                    if not pd_.get("ok"):
                        gap_cause[pd_.get("cause", "?")] += 1
                        break
        summary[arm] = (npass, len(states))
    return summary, gap_cause, len(files), len(states)


def selftest():
    """엔진 순수 단위테스트 (궤적·db·ABox파일 없이 인라인 ABox로)."""
    abox = {"compute_ops": {"file_debit_card_transaction_dispute": {
        "customer_max_liability_amount": {"op": "lookup_table",
            "key": {"op": "days_between", "a": "params.transaction_date", "b": "params.discovery_date"},
            "table": [{"cmp": "<=", "thr": 30, "result": 50},
                      {"cmp": "<=", "thr": 60, "result": 500},
                      {"result": {"op": "ref", "path": "params.disputed_amount"}}]}}}}
    tf = "file_debit_card_transaction_dispute"
    # (1) <=30일 → 50
    v = compute_fields(tf, {"transaction_date": "01/01/2024", "discovery_date": "01/15/2024",
                            "disputed_amount": "200"}, abox)
    assert v["customer_max_liability_amount"] == 50, v
    # (2) 31~60일 → 500
    v = compute_fields(tf, {"transaction_date": "01/01/2024", "discovery_date": "02/20/2024",
                            "disputed_amount": "200"}, abox)
    assert v["customer_max_liability_amount"] == 500, v
    # (3) >60일 → disputed_amount
    v = compute_fields(tf, {"transaction_date": "01/01/2024", "discovery_date": "05/01/2024",
                            "disputed_amount": "200"}, abox)
    assert _norm(v["customer_max_liability_amount"]) == 200.0, v   # ref=원값·_norm이 대조 시 정규화
    # (4) dispute 판정: agent가 liability만 틀림(입력 gold) → COMPUTE가 닫음
    gold = {"transaction_id": "txn_a", "transaction_date": "01/01/2024",
            "discovery_date": "01/15/2024", "disputed_amount": "200", "customer_max_liability_amount": "50"}
    rec = dict(gold); rec["customer_max_liability_amount"] = "200"   # agent 오답
    d = classify_dispute(gold, rec, tf, abox, on_record_fields=set())
    assert d["ok"], d
    # (5) reach: agent 미제출·seen에 있음 → LOAD가 닫음, COMPUTE는 못 닫음
    st = {"gold": {"txn_a": gold}, "subs": {}, "seen": {"txn_a"}, "universe": {"txn_a"},
          "tool_family": tf}
    assert controller(st, abox, arm="COMPUTE")["sim_pass"] is False
    assert controller(st, abox, arm="LOAD")["sim_pass"] is True
    # (6) open-world: 어디에도 없음 → LOAD도 못 닫음(ASK 잔여)
    st2 = {"gold": {"txn_z": gold}, "subs": {}, "seen": set(), "universe": set(), "tool_family": tf}
    r2 = controller(st2, abox, arm="LOAD")
    assert r2["sim_pass"] is False and r2["per_dispute"][0]["cause"] == "open_world_ASK", r2
    # (7) 산술 op 확장: amount_difference = round((exp-act)/100*balance) — gold 실측 검증
    abox2 = {"compute_ops": {"submit_interest_discrepancy_report": {
        "amount_difference": {"op": "round", "ndigits": 2, "value": {
            "op": "multiply",
            "a": {"op": "divide", "a": {"op": "subtract", "a": "params.expected_apy", "b": "params.actual_apy"},
                  "b": {"op": "const", "value": 100}},
            "b": "params.balance"}}}}}
    v = compute_fields("submit_interest_discrepancy_report",
                       {"expected_apy": 4.275, "actual_apy": 4.0, "balance": 12000}, abox2)
    assert v["amount_difference"] == 33.0, v            # gold: 0.275%×12000=33.0
    v = compute_fields("submit_interest_discrepancy_report",
                       {"expected_apy": 6.85, "actual_apy": 5.1, "balance": 8000}, abox2)
    assert v["amount_difference"] == 140.0, v           # gold: 1.75%×8000=140.0
    # (8) DAG 컨트롤러: 합성 sim — 미호출 write(coverage)=closable·enum 오답=inner router
    syn = {"reward_info": {"action_checks": [
        {"action": {"arguments": {"agent_tool_name": "close_debit_card_1",
                                  "arguments": {"card_id": "dbc_x", "reason": "fraud"}}}, "action_reward": 0.0},
        {"action": {"arguments": {"agent_tool_name": "file_debit_card_transaction_dispute_2",
                                  "arguments": {"transaction_id": "t", "dispute_category": "card_present_fraud"}}}, "action_reward": 0.0},
    ]},
        # dispute 도구는 *호출*됨(오답)·close_debit_card는 미호출 → coverage + F3-enum 혼재
        "messages": [{"role": "assistant", "tool_calls": [
            {"name": "call_discoverable_agent_tool",
             "arguments": {"agent_tool_name": "file_debit_card_transaction_dispute_2",
                           "arguments": {"transaction_id": "t", "dispute_category": "card_not_present_fraud"}}}]}]}
    # ABox field_ops(엔진 리터럴0·리뷰 ❶) — dispute_category=enum·reason=enum
    abox_fo = {"compute_ops": {}, "field_ops": {
        "enum": ["dispute_category", "reason"], "id_ref": ["card_id", "transaction_id"],
        "compute": [], "judgment": []}}
    pl = dag_plan(syn, abox_fo)
    ops = {op for (_, op, _) in pl["steps"]}
    assert "F3-enum" in ops, pl           # dispute_category(호출·오답) 또는 close reason=enum
    assert not pl["hard_only"] and not pl["hard_soft"], pl   # F3-enum → soft로도 안 닫힘(❷)
    assert pl["with_B"], pl               # F3-enum만이면 B 가정 시 닫힘
    assert "F3-enum" in pl["residual_ops"], pl
    # ❷ 회귀: uncalled write의 enum arg(reason)도 잡히나 — close_debit_card 미호출·reason=enum
    syn2 = {"reward_info": {"action_checks": [
        {"action": {"arguments": {"agent_tool_name": "close_debit_card_1",
                                  "arguments": {"card_id": "dbc_x", "reason": "fraud"}}}, "action_reward": 0.0}]},
        "messages": []}
    pl2 = dag_plan(syn2, abox_fo)
    assert not pl2["hard_soft"], pl2      # 미호출 write지만 reason=enum(F3) → coverage(soft) 아니라 inner
    assert "F3-enum" in {op for (_, op, _) in pl2["steps"]}, pl2
    print("selftest: ALL PASS (엔진 순수·ABox field_ops·리터럴0·산술op·DAG tier ❶❷❺❼ 확인)")


# ══════════════════════════════════════════════════════════════════════════════
#  ★일반화 전-액션 harness (재앵커·2026-07-15) — dispute-only 부분모델 폐기.
#  reward=DB-basis(81%)·dispute=실패 20%. 아우터 loop coverage를 *전 gold 액션타입*에.
#  action-level 판정(met=1 / coverage-miss=미호출 / args-miss=호출·오답). proxy ~90% tight.
# ══════════════════════════════════════════════════════════════════════════════

def _agent_called_families(s):
    """agent 실호출 도구 family Counter (discoverable inner + 직접)."""
    c = Counter()
    for m in (s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or ""
            if nm == "call_discoverable_agent_tool":
                inner = _nd(tc.get("arguments")).get("agent_tool_name", "")
                if inner:
                    c[_fam(inner)] += 1
            elif nm:
                c[_fam(nm)] += 1
    return c


def classify_sim_allaction(s, abox):
    """전-액션 컨트롤러 오프라인 판정 (DB-basis 실패 sim). action-level.
    반환 {arm, n_cov, n_arg, n_arg_compute, db_match, has_argrows}."""
    ri = s.get("reward_info") or {}
    db = (ri.get("db_check") or {}).get("db_match")
    called = _agent_called_families(s)
    comp_ops = abox.get("compute_ops") or {}
    n_cov = n_arg = n_arg_compute = 0
    argrows = 0
    for ac in (ri.get("action_checks") or []):
        a = ac.get("action") or {}
        outer = _nd(a.get("arguments"))
        atn = outer.get("agent_tool_name", "")
        if not atn or "arguments" not in outer:
            continue                                   # name-only/assertion 행 제외
        argrows += 1
        met = ac.get("action_reward")
        if met is None:
            met = 1.0 if ac.get("action_match") else 0.0
        if float(met) >= 1.0:
            continue
        tf = _fam(atn)
        if called.get(tf, 0) == 0:
            n_cov += 1                                 # coverage-miss (미호출)
        else:
            n_arg += 1                                 # args-miss (호출·오답)
            # ABox compute 규칙 보유 도구면 compute-closable 후보(dispute liability만 현재)
            if any(k in tf or tf in k for k in comp_ops):
                n_arg_compute += 1
    # sim arm 분류 (아우터 loop coverage 레버 사정권)
    if argrows == 0:
        arm = "pure_DB(action-check 없음·사각지대)"
    elif n_cov == 0 and n_arg == 0:
        arm = "all_actions_met(DB만 실패=over-action/order)"
    elif n_arg == 0:
        arm = "coverage-only(아우터loop 강제열거 사정권)"
    elif n_cov == 0:
        arm = "args-only(inner router: compute/⋈/gather)"
    else:
        arm = "coverage+args 혼합"
    return {"arm": arm, "n_cov": n_cov, "n_arg": n_arg,
            "n_arg_compute": n_arg_compute, "db_match": db, "argrows": argrows}


def run_offline_allaction(abox, db_basis_only=True):
    files = glob.glob("C:/tmp/traj/*_banking.json")
    arm_ct = Counter()
    unmet = Counter()
    term_excl = Counter()
    n = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            if db_basis_only and tuple(ri.get("reward_basis") or []) != ("DB",):
                continue
            # infra 종료 제외(능력 ceiling·[[08]])
            tr = str(s.get("termination_reason"))
            if tr == "too_many_errors":
                term_excl["too_many_errors(infra 제외)"] += 1
                continue
            n += 1
            r = classify_sim_allaction(s, abox)
            arm_ct[r["arm"]] += 1
            unmet["coverage(미호출)"] += r["n_cov"]
            unmet["args(호출·오답)"] += r["n_arg"]
            unmet["└ 그중 ABox-compute 사정권"] += r["n_arg_compute"]
    return arm_ct, unmet, term_excl, n, len(files)


# ══════════════════════════════════════════════════════════════════════════════
#  ★(c) 전-액션 per-step DAG 컨트롤러 (2026-07-16·사용자 지시)
#  균일 loop: gold DAG의 매 스텝을 연산 분류{FIND/COVERAGE/COMPUTE/GET-⋈/ASK/F3} →
#  결정론 closable 판정 + operator plan 생성. H_min = 전 스텝 닫힐 때까지.
#  라이브 배선(t2_eplan_patch 진화)의 결정론 controller 로직·유료 실행은 별도([[09]]).
# ══════════════════════════════════════════════════════════════════════════════

# 도메인일반 명명-convention만 엔진 보유(필드→op 리터럴은 ABox field_ops·리뷰 ❶ 반영·[[05]]).
_READ_PREFIX = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)  # API verb convention
_PROCEDURAL = re.compile(r"(^log_|_verification$|^kb_search|^kb_|^search_|^shell$|discoverable|transfer_to_human|give_)", re.I)  # harness meta-tool
# 연산 tier (리뷰 ❺): hard=강제가능(FIND read·COMPUTE·GET-⋈) · soft=write-emit 리마인더만([[16]]/[[07]]) · inner=의미(F3/gather)
_OP_TIER = {"FIND": "hard", "COMPUTE": "hard", "GET-xmatch": "hard",
            "COVERAGE": "soft", "F3-enum": "inner-B", "F3-judgment": "inner-hard", "GATHER": "inner-hard"}
_TIER_RANK = {"hard": 0, "soft": 1, "inner-B": 2, "inner-hard": 3}


def _field_op(field, tool_family, abox):
    """틀린 필드 → 연산 타입 (전부 ABox field_ops·엔진 리터럴0·[[05]]·리뷰 ❶)."""
    fo = abox.get("field_ops") or {}
    if field in set(fo.get("judgment", [])):
        return "F3-judgment"
    if field in set(fo.get("compute", [])) or field in compute_field_names(tool_family, abox):
        return "COMPUTE"
    if field in set(fo.get("id_ref", [])):
        return "GET-xmatch"
    if field in set(fo.get("enum", [])):
        return "F3-enum"
    return "GATHER"


def _write_worst_op(tf, gold_args, uncalled, abox):
    """write 스텝의 최악 연산(리뷰 ❷: uncalled write도 gold_args 검사).
    uncalled면 COVERAGE(write-emit·soft) 후보에 포함·args 중 최악과 tier 비교."""
    cand = ["COVERAGE"] if uncalled else []
    for field in gold_args:
        if field == "transaction_id":
            continue
        cand.append(_field_op(field, tf, abox))
    if not cand:
        cand = ["COVERAGE"]
    return max(cand, key=lambda o: _TIER_RANK[_OP_TIER[o]])


def dag_plan(s, abox):
    """한 sim의 gold DAG를 per-step walk → (tool,op,tier) 스텝 + 다중 closure 판정.
    tier: hard(강제)·soft(write-emit 리마인더)·inner-B(F3-enum)·inner-hard(judgment/gather)."""
    ri = s.get("reward_info") or {}
    calls = _agent_called_families(s)
    steps = []
    for ac in (ri.get("action_checks") or []):
        a = ac.get("action") or {}
        outer = _nd(a.get("arguments"))
        atn = outer.get("agent_tool_name", "")
        if not atn or "arguments" not in outer:
            continue
        tf = _fam(atn)
        if _PROCEDURAL.search(tf):
            continue
        met = ac.get("action_reward")
        if met is None:
            met = 1.0 if ac.get("action_match") else 0.0
        if float(met) >= 1.0:
            continue
        rd = bool(_READ_PREFIX.match(tf))
        gold_args = _nd(outer.get("arguments"))
        uncalled = (tf not in calls) or (not calls.get(tf))
        if rd:
            op = "FIND"                            # read = 강제열거(hard)·값판정 생략
        else:
            op = _write_worst_op(tf, gold_args, uncalled, abox)  # ❷ uncalled도 args 분석
        steps.append((tf, op, _OP_TIER[op]))
    # 리뷰 ❺❼ 다중 closure: hard-only / hard+soft / +B(F3-enum) 상한
    tiers = [t for (_, _, t) in steps]
    n = len(steps)
    hard_only = n > 0 and all(t == "hard" for t in tiers)
    hard_soft = n > 0 and all(t in ("hard", "soft") for t in tiers)
    with_B = n > 0 and all(t in ("hard", "soft", "inner-B") for t in tiers)
    residual = sorted({op for (_, op, t) in steps if t in ("inner-B", "inner-hard")})
    return {"steps": steps, "n_steps": n, "hard_only": hard_only,
            "hard_soft": hard_soft, "with_B": with_B, "residual_ops": residual}


def run_dag_replay(abox):
    """오프라인 DAG replay: 3 상한(hard-only/hard+soft/A+B) — 리뷰 ❺❼ 반영."""
    files = glob.glob("C:/tmp/traj/*_banking.json")
    verdict = Counter(); res_op = Counter(); n = 0; blind = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            if tuple(ri.get("reward_basis") or []) != ("DB",):
                continue
            if str(s.get("termination_reason")) == "too_many_errors":
                continue
            n += 1
            p = dag_plan(s, abox)
            if p["n_steps"] == 0:
                verdict["Blind(pure-DB)"] += 1; blind += 1
                continue
            # 3 상한(중첩·리뷰 ❺❼): hard-only ⊆ hard+soft ⊆ +B
            if p["hard_only"]:
                verdict["hard-only(강제가능·FIND/COMPUTE/GET)"] += 1
            elif p["hard_soft"]:
                verdict["+soft(COVERAGE write-emit 리마인더)"] += 1
            elif p["with_B"]:
                verdict["+B(F3-enum 스킬 가정)"] += 1
            else:
                verdict["잔여(judgment/gather-user)"] += 1
                for op in p["residual_ops"]:
                    res_op[op] += 1
    return verdict, res_op, n, blind, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dag", action="store_true", help="(c) per-step DAG 컨트롤러 오프라인 replay")
    ap.add_argument("--dispute-only", action="store_true", help="폐기된 dispute 부분모델(provenance)")
    ap.add_argument("--all-basis", action="store_true", help="DB-basis 외 sim도 포함")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    abox = load_abox()
    if a.dag:
        verdict, res_op, n, blind, nfiles = run_dag_replay(abox)
        obs = n - blind
        ho = verdict["hard-only(강제가능·FIND/COMPUTE/GET)"]
        hs = ho + verdict["+soft(COVERAGE write-emit 리마인더)"]
        ab = hs + verdict["+B(F3-enum 스킬 가정)"]
        print("=== (c) per-step DAG 컨트롤러 오프라인 상한 3분해 (DB-basis 실패 %d·%d궤적·리뷰 ❶❷❺❼ 반영) ===" % (n, nfiles))
        for k in ("hard-only(강제가능·FIND/COMPUTE/GET)", "+soft(COVERAGE write-emit 리마인더)",
                  "+B(F3-enum 스킬 가정)", "잔여(judgment/gather-user)", "Blind(pure-DB)"):
            print("  %-38s %5d (%.1f%%)" % (k, verdict[k], 100 * verdict[k] / max(n, 1)))
        print("  " + "─" * 44)
        print("  ★HARD-only 상한(강제가능·[[16]] 준수) = %.1f%% (관측가능 %.1f%%)" % (100 * ho / max(n, 1), 100 * ho / max(obs, 1)))
        print("  ★+SOFT 상한(COVERAGE=write-emit 리마인더·soft-bet·[[07]] 불확실) = %.1f%% (관측 %.1f%%)" % (100 * hs / max(n, 1), 100 * hs / max(obs, 1)))
        print("  ★A+B 상한(F3-enum=Track B 스킬 가정) = %.1f%% (관측 %.1f%%) ← frontier(gpt55 pass 37.4%%) 비교 타깃(❼)" % (100 * ab / max(n, 1), 100 * ab / max(obs, 1)))
        print("  잔여(judgment/gather-user):", dict(res_op.most_common()))
        print("  [[08]]/리뷰: HARD-only=진짜 결정론 실현 하한·SOFT=write강제금지라 soft-bet(❺)·이 3숫자가 정직한 봉투.")
        print("        상한=gold-informed 기회(❸)·라이브 실현<상한·gold-free coverage 게이트(❹)가 실현성 관문.")
        return
    if a.dispute_only:
        summary, gap_cause, nfiles, nstates = run_offline(abox)
        print("=== [폐기·provenance] dispute-only 부분모델 (%d 궤적·%d dispute-sim) ===" % (nfiles, nstates))
        for arm in ("COMPUTE", "LOAD"):
            npass, nn = summary[arm]
            print("  %-8s dispute-attributable pass %4d/%d = %.1f%%" % (arm, npass, nn, 100 * npass / max(nn, 1)))
        print("  ⚠ 이 수치는 reward(DB) 아닌 dispute action_checks proxy·실패의 ~20%만 봄(FORENSIC 참조).")
        return
    arm_ct, unmet, term_excl, n, nfiles = run_offline_allaction(abox, db_basis_only=not a.all_basis)
    scope = "DB-basis" if not a.all_basis else "전-basis"
    print("=== banking E-PLAN 전-액션 오프라인 재앵커 (%d 궤적·%s 실패 sim %d·infra제외) ===" % (nfiles, scope, n))
    print("  [term 제외] %s" % dict(term_excl))
    print("\n  미충족 gold-액션(args-row) 분해:")
    tot = unmet["coverage(미호출)"] + unmet["args(호출·오답)"]
    for k in ("coverage(미호출)", "args(호출·오답)", "└ 그중 ABox-compute 사정권"):
        base = tot if not k.startswith("└") else unmet["args(호출·오답)"]
        print("    %-26s %6d (%.1f%%)" % (k, unmet[k], 100 * unmet[k] / max(base, 1)))
    print("\n  sim arm 분류 (아우터 loop coverage 레버 사정권):")
    for k, v in arm_ct.most_common():
        print("    %-42s %5d (%.1f%%)" % (k, v, 100 * v / max(n, 1)))
    cov = arm_ct["coverage-only(아우터loop 강제열거 사정권)"]
    mix = arm_ct["coverage+args 혼합"]
    print("\n  ★coverage 레버 사정권: coverage-only %.1f%% (강제열거 상한) + 혼합 %.1f%% (부분)"
          % (100 * cov / max(n, 1), 100 * mix / max(n, 1)))
    print("  ★잔여: args-only(inner router)·over-action(all-met DB실패)·pure-DB(사각지대·action-check 없음).")
    print("  [[08]] proxy ~90% tight(FORENSIC §3.5)·pure-DB blind 860 = 이 오프라인 상한의 근본 한계.")


if __name__ == "__main__":
    main()
