#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""banking floor(bankxfer_floor_bank_t4·trial 0/1=nt2 정본) 실패 전수 → retail 레버 스택 fit 1차 분류기.

레버 발화조건(기계 판정·BANKING_FLOOR_LEVER_FIT_2026_07_11.md §0 기준):
  PERARG    : 실패-원인 write의 hint-키 인자값이 호출-시점 문맥(user발화+tool출력)에 부재(=날조) → regen 발화.
              engine과 동일 규칙: substring lower + '#'접두 strip(≥4자) (t2_gate_patch._ctx_has).
  GATE_GB1  : banking A2 유일 게이트(auth: log_verification 선행). 발화 = gold에 log_verification 있는데
              미실행/후행인 채 gated 도구(call_discoverable_agent_tool 비면제 or 직접 4도구) 시도.
  CALC_EXT  : arg-diff가 수치류이고 gold값이 문맥 비-verbatim(=유도값) → 계산형 후보(정독 확정 필요).
  DISAMB    : 쓴 값도 gold값도 둘 다 호출-시점 문맥에 실재(|C|>=2) → silent P-B 후보.
  P2_DEFAULT: banking A2에 default_specs 없음 → 구조적 비발화(0 고정·계상만).
  EPLAN_L1  : gold 반복-write(N>=2)인데 그 write 계열 0회 & 열거자(list-tool) 미호출.
  EPLAN_L2  : N>M & 열거자 호출됨(목록엔 있음) & 일부 entity 미처리.
  EPLAN_WALK: N>M 인 채 user_stop 종료(재확인 리마인더 개연).
  NOTICE/EXCL: banking A2 notice 미인스턴스 → 구조적 비발화. EXCL은 동일-key 재-write 순서로 근사.
  RESIDUAL  : (a) reach/조립(missing>=1/3) (b) semantic (c) NL (d) user-sim/harness (e) 신규 — 정독으로 확정.

[[08]] 정독-검증 후 v2 추가 플래그 (12+ 궤적 전문 정독이 스크립트 v1 판정을 정정):
  PERARG_ALLCALL : v1은 gold-매칭 argdiff만 봤음 → 엔진은 '모든' 호출을 검사. 특히
                   agent_tool_name/discoverable_tool_name('name' 힌트 매칭!)의 도구명 날조 = 현행 엔진이 잡음.
  FAB_TOOLNAME   : 문맥에 없는(=KB로 발견 안 한) 발견가능 도구명을 지어내 호출 (task_012 set_travel_notification,
                   task_016 investigate_referral_status, task_096 link_checking_savings_accounts 등).
  LOGV_TIME_FAB  : log_verification.time_verified 날조(get_current_time 미호출·과거시각 지어냄, task_043/065 등).
                   DB all-or-nothing에서 단독으로도 치명. 현행 힌트('time' 미포함) 사거리 밖 —
                   A2 identifying_arg_types 확장(ABox-only)으로 PERARG 사거리에 들어옴 + producer=get_current_time.
  EARLY_TRANSFER : gold에 없는 transfer 실행(조기 이관·task_082). retail G_EXHAUST(exhaust_before_escalate) kind로
                   닫힘 후보 — banking A2 미인스턴스(확장 필요).

주의: 발화≠복구. per-diff로 fired(발화확실)와 gold-in-ctx(복구개연)를 분리 기록.
usage: py -3 banking_lever_fit_census.py [results.json.gz] [out.json]
"""
import sys, os, io, json, gzip, re
from collections import Counter, defaultdict

GZ = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\bankxfer_floor_bank_t4.results.json.gz"
OUT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\banking_lever_fit_percase.json"

# engine과 동일 (t2_gate_patch.DEFAULT_ARG_HINTS)
HINTS = ("email", "name", "zip", "user_id", "order_id", "username", "id",
         "payment", "address", "phone", "item", "reservation")
# banking A2 GB1 (banking_knowledge.gate.json)
GB1_APPLIES = {"change_user_email", "get_referrals_by_user",
               "get_credit_card_transactions_by_user", "get_credit_card_accounts_by_user",
               "call_discoverable_agent_tool"}
GB1_EXEMPT_INNER = {"initial_transfer_to_human_agent_0218", "initial_transfer_to_human_agent_1822",
                    "emergency_credit_bureau_incident_transfer_1114"}
# 읽기(비-write) 도구: 상태 불변
READ_TOOLS = {"KB_search", "get_user_information_by_name", "get_user_information_by_user_id",
              "get_user_information_by_email", "get_user_information_by_phone",
              "get_current_time", "get_referrals_by_user",
              "get_credit_card_transactions_by_user", "get_credit_card_accounts_by_user",
              "search_available_tools", "list_available_tools"}
# 열거자(list-tool) 후보: E-PLAN 판정용 (banking의 get_user_details 대응물)
ENUMERATORS = {"get_credit_card_transactions_by_user", "get_credit_card_accounts_by_user",
               "get_referrals_by_user"}
NUMSTR = re.compile(r"^-?[\d,]+(\.\d+)?\s*[a-zA-Z%$]*$")


def parse_maybe_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def resolve(name, args):
    """discoverable 4도구는 내부도구명으로 키 해석. returns (key, inner_args_dict)."""
    a = parse_maybe_json(args) if args is not None else {}
    if not isinstance(a, dict):
        a = {}
    if name in ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool"):
        inner = a.get("agent_tool_name")
        ia = parse_maybe_json(a.get("arguments", {})) or {}
        return "%s:%s" % (name, inner), (ia if isinstance(ia, dict) else {})
    if name in ("give_discoverable_user_tool", "call_discoverable_user_tool"):
        inner = a.get("discoverable_tool_name")
        ia = parse_maybe_json(a.get("arguments", {})) or {}
        return "%s:%s" % (name, inner), (ia if isinstance(ia, dict) else {})
    return name, a


def is_write_key(key):
    base = key.split(":")[0]
    if base in READ_TOOLS:
        return False
    if base in ("unlock_discoverable_agent_tool", "give_discoverable_user_tool"):
        return True   # 상태(unlock/give) 변경 — 체인 필수 단계
    return True


def canon(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if NUMSTR.match(s.replace(",", "")) and re.match(r"^-?\d", s):
            try:
                return float(re.sub(r"[^\d.\-]", "", s))
            except Exception:
                pass
        return s.lower()
    if isinstance(v, list):
        return tuple(canon(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, canon(x)) for k, x in v.items()))
    return v


def ctx_has(val, ctx):
    """engine _ctx_has 동일: lower substring + '#'접두 strip 재시도. 수치는 int표기 변형도 시도."""
    s = str(val).strip()
    if not s:
        return True
    if s.lower() in ctx:
        return True
    t = s.lstrip("#")
    if t != s and len(t) >= 4 and t.lower() in ctx:
        return True
    # 수치 표기 변형(6300.0 vs 6300 / 1,020)
    try:
        f = float(s.replace(",", ""))
        for cand in ("%d" % f if f == int(f) else None, "%.2f" % f, "{:,}".format(int(f)) if f == int(f) else None):
            if cand and cand in ctx:
                return True
    except Exception:
        pass
    return False


def field_class(k, gv, ev):
    if isinstance(gv, bool) or isinstance(ev, bool):
        return "BOOL"
    def isnum(v):
        c = canon(v)
        return isinstance(c, float)
    if isnum(gv) and isnum(ev):
        return "NUMERIC"
    if re.search(r"(^|_)(id|ids)$", k.lower()):
        return "ID"
    sv = str(gv)
    if re.match(r"^\d{4}-\d{2}-\d{2}", sv) or "date" in k.lower() or "time" in k.lower():
        return "DATE"
    if isinstance(gv, str) and len(gv) <= 40 and " " not in gv.strip():
        return "ENUM"
    return "TEXT"


def hinted(k):
    return any(h in k.lower() for h in HINTS)


def build_events(sim):
    """시간순 이벤트: ('ctx', text) = user발화/tool출력, ('call', requestor, key, args, name, raw_args)."""
    ev = []
    for m in sim.get("messages", []):
        role = m.get("role")
        content = m.get("content")
        if role == "user" and isinstance(content, str) and content:
            ev.append(("ctx", content))
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name")
            key, ia = resolve(nm, tc.get("arguments"))
            raw = parse_maybe_json(tc.get("arguments")) or {}
            ev.append(("call", tc.get("requestor") or role, key, ia, nm,
                       raw if isinstance(raw, dict) else {}))
        if role == "tool" and isinstance(content, str) and content:
            ev.append(("ctx", content))
    return ev


def analyze_sim(sim, task):
    ec = (task.get("evaluation_criteria") or {})
    gold_raw = ec.get("actions") or []
    gold = []
    for a in gold_raw:
        key, ia = resolve(a["name"], a.get("arguments"))
        gold.append({"key": key, "args": ia, "requestor": a.get("requestor"), "id": a.get("action_id")})

    events = build_events(sim)
    calls = []          # executed, ctx-시점 포함
    ctx_parts = []
    for e in events:
        if e[0] == "ctx":
            ctx_parts.append(e[1].lower())
        else:
            calls.append({"requestor": e[1], "key": e[2], "args": e[3],
                          "name": e[4], "raw": e[5],
                          "ctx": " ".join(ctx_parts)})
    full_ctx = " ".join(ctx_parts)

    # ---- greedy 매칭: exact → name-nearest ----
    left = list(range(len(calls)))
    missing, argdiffs, matched = [], [], []
    for g in gold:
        cg = {k: canon(v) for k, v in g["args"].items()}
        hit = None
        for i in left:
            c = calls[i]
            if c["key"] == g["key"] and {k: canon(v) for k, v in c["args"].items()} == cg:
                hit = i
                break
        if hit is not None:
            left.remove(hit)
            matched.append(g["key"])
            continue
        cands = [i for i in left if calls[i]["key"] == g["key"]]
        if not cands:
            missing.append(g)
            continue
        best, bestdiff = None, None
        for i in cands:
            ce = {k: canon(v) for k, v in calls[i]["args"].items()}
            diff = {k for k in set(cg) | set(ce) if cg.get(k) != ce.get(k)}
            if best is None or len(diff) < len(bestdiff):
                best, bestdiff = i, diff
        left.remove(best)
        c = calls[best]
        fields = []
        for k in sorted(bestdiff):
            gv, evv = g["args"].get(k), c["args"].get(k)
            if k not in cg:
                fields.append({"field": k, "cls": "EXTRA_FIELD", "exec": evv})
                continue
            if k not in {kk: canon(v) for kk, v in c["args"].items()}:
                fields.append({"field": k, "cls": "MISSING_FIELD", "gold": gv})
                continue
            fc = field_class(k, gv, evv)
            fields.append({
                "field": k, "cls": fc, "gold": gv, "exec": evv,
                "hinted": hinted(k),
                "exec_in_ctx": ctx_has(evv, c["ctx"]),
                "gold_in_ctx": ctx_has(gv, c["ctx"]),
                "gold_in_full_ctx": ctx_has(gv, full_ctx),
            })
        argdiffs.append({"key": g["key"], "requestor": g["requestor"], "fields": fields})

    gold_keys = set(g["key"] for g in gold)
    extra_writes = [c["key"] for i, c in enumerate(calls)
                    if i in left and is_write_key(c["key"]) and c["key"] not in gold_keys]

    # ---- 레버 판정 ----
    flags = {}
    ngold = len(gold)
    nmiss = len(missing)
    miss_keys = [g["key"] for g in missing]

    # GATE_GB1
    lv_in_gold = any(g["key"] == "log_verification" for g in gold)
    lv_missing = "log_verification" in miss_keys
    lv_called_idx = next((i for i, c in enumerate(calls) if c["key"] == "log_verification"), None)
    gated_attempt_idx = next(
        (i for i, c in enumerate(calls)
         if c["requestor"] == "assistant"
         and (c["key"].split(":")[0] in GB1_APPLIES)
         and not (c["key"].startswith("call_discoverable_agent_tool:")
                  and c["key"].split(":", 1)[1] in GB1_EXEMPT_INNER)),
        None)
    gate_fires = (lv_in_gold
                  and gated_attempt_idx is not None
                  and (lv_called_idx is None or lv_called_idx > gated_attempt_idx))
    flags["GATE_GB1"] = {
        "fires": bool(gate_fires),
        "lv_in_gold": lv_in_gold, "lv_missing": lv_missing,
        "closure_plausible": bool(gate_fires and set(miss_keys) <= {"log_verification"} and not argdiffs
                                  and not extra_writes),
    }

    # per-field 레버 (agent-측 write의 arg-diff만 — user측 write는 별도 계상)
    perarg_f, perarg_rec, disamb, calc, calc_sel = [], [], [], [], []
    user_side_diff = []
    for ad in argdiffs:
        agent_side = (ad["requestor"] == "assistant")
        for f in ad["fields"]:
            if f["cls"] in ("EXTRA_FIELD", "MISSING_FIELD"):
                continue
            tag = "%s.%s" % (ad["key"], f["field"])
            if not agent_side:
                user_side_diff.append(tag)
                continue
            if f["hinted"] and len(str(f["exec"]).strip()) >= 4 and not f["exec_in_ctx"]:
                perarg_f.append(tag)
                if f["gold_in_ctx"]:
                    perarg_rec.append(tag)
            if f["exec_in_ctx"] and f["gold_in_ctx"]:
                disamb.append(tag)
            if f["cls"] == "NUMERIC":
                (calc_sel if f["gold_in_full_ctx"] else calc).append(tag)
    flags["PERARG"] = {"fired_fields": perarg_f, "recovery_plausible_fields": perarg_rec}
    flags["DISAMB"] = {"candidate_fields": disamb}
    flags["CALC_EXT"] = {"derived_numeric_fields": calc, "selection_numeric_fields": calc_sel}
    flags["USER_SIDE_ARGDIFF"] = user_side_diff
    flags["P2_DEFAULT"] = {"fires": False, "why": "banking A2 default_specs 부재(구조적 비발화)"}
    flags["NOTICE"] = {"fires": False, "why": "banking A2 notice 미인스턴스"}

    # ---- v2: 엔진-동형 PERARG 전-호출 스캔 (정독 정정 [[08]]) ----
    # 엔진은 모든 assistant 호출의 raw args를 검사. agent_tool_name/discoverable_tool_name도
    # 'name' 힌트에 걸리므로 도구명 날조가 현행 스택 사거리 안.
    gold_keys_pre = set()
    for g in gold:
        gold_keys_pre.add(g["key"])
    perarg_all, fab_toolname = [], []
    logv_time_fab = False
    gold_lv_args = next((g["args"] for g in gold if g["key"] == "log_verification"), None)
    for c in calls:
        if c["requestor"] != "assistant":
            continue
        # (a) raw-arg 힌트 스캔 (엔진 _provenance_deny 동형)
        for k, v in (c["raw"] or {}).items():
            if k == "arguments":
                continue  # inner args는 아래 (b)와 gold-diff 분석이 담당
            if not hinted(k):
                continue
            s = str(v).strip()
            if len(s) >= 4 and not ctx_has(s, c["ctx"]):
                perarg_all.append("%s.%s=%s" % (c["name"], k, s[:40]))
                if k in ("agent_tool_name", "discoverable_tool_name") and c["key"] not in gold_keys_pre:
                    fab_toolname.append(s[:60])
        # (b) log_verification.time_verified 날조 (힌트 밖 — A2 identifying_arg_types 확장 표적)
        if c["name"] == "log_verification" and gold_lv_args:
            tv = str((c["args"] or {}).get("time_verified", "")).strip()
            gtv = str(gold_lv_args.get("time_verified", "")).strip()
            if tv and gtv and tv != gtv and not ctx_has(tv, c["ctx"]):
                logv_time_fab = True
    flags["PERARG_ALLCALL"] = {"fired": perarg_all}
    flags["FAB_TOOLNAME"] = sorted(set(fab_toolname))
    flags["LOGV_TIME_FAB"] = logv_time_fab
    # v2: 조기/과잉 이관 (retail G_EXHAUST kind로 닫힘 후보·banking A2 미인스턴스)
    tr_gold = any(g["key"] in ("transfer_to_human_agents", "request_human_agent_transfer") for g in gold)
    tr_exec = any(c["key"] in ("transfer_to_human_agents", "request_human_agent_transfer") for c in calls)
    flags["EARLY_TRANSFER"] = bool(tr_exec and not tr_gold)

    # E-PLAN: gold 반복-write 계열
    gmult = Counter(g["key"] for g in gold if is_write_key(g["key"]))
    emult = Counter(c["key"] for c in calls if is_write_key(c["key"]))
    ep = []
    enum_called = any(c["key"] in ENUMERATORS for c in calls)
    for k, N in gmult.items():
        if N < 2:
            continue
        M = emult.get(k, 0)
        if M >= N:
            continue
        ep.append({"key": k, "N": N, "M": M})
    eplan_l1 = bool(ep) and not enum_called and all(e["M"] == 0 for e in ep)
    eplan_l2 = bool(ep) and enum_called
    eplan_walk = bool(ep) and sim.get("termination_reason") == "user_stop"
    flags["EPLAN"] = {"repeat_write_gaps": ep, "enumerator_called": enum_called,
                      "L1": eplan_l1, "L2": eplan_l2, "walk": eplan_walk}

    # EXCLUSIVITY 근사: 동일 key write 초과실행
    over = [{"key": k, "N": gmult.get(k, 0), "M": m} for k, m in emult.items()
            if m > gmult.get(k, 0) and k in gold_keys]
    flags["OVER_EXEC"] = over

    # ---- 1차 primary 버킷 ----
    miss_frac = (nmiss / ngold) if ngold else 0.0
    if ngold == 0:
        primary = "no_gold(NL/knowledge형)"
    elif miss_frac >= 1 / 3:
        primary = "REACH_ASSEMBLY(미실행지배)"
    elif nmiss == 0 and not argdiffs and extra_writes:
        primary = "EXTRA_WRITE_ONLY"
    elif nmiss == 0 and not argdiffs and not extra_writes:
        primary = "ALL_MATCH(순서/env/유저측/채점)"
    elif argdiffs and nmiss == 0:
        primary = "ARGDIFF(완주-후-불일치)"
    else:
        primary = "MIXED(부분미실행+argdiff)"

    return {
        "task_id": sim["task_id"], "trial": sim["trial"],
        "termination": sim.get("termination_reason"),
        "ngold": ngold, "n_missing": nmiss, "missing_keys": miss_keys,
        "miss_frac": round(miss_frac, 3),
        "argdiffs": argdiffs, "extra_writes": extra_writes,
        "n_agent_calls": sum(1 for c in calls if c["requestor"] == "assistant"),
        "flags": flags, "primary": primary,
        "db_match": (sim.get("reward_info") or {}).get("db_check", {}) and
                    (sim["reward_info"].get("db_check") or {}).get("db_match"),
        "reward_basis": ec.get("reward_basis"),
    }


def main(gz, out):
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        d = json.load(f)
    tasks = {t["id"]: t for t in d["tasks"]}
    sims = [s for s in d["simulations"] if s.get("trial") in (0, 1)]
    infra = [s for s in sims if s.get("reward_info") is None]
    fails = [s for s in sims if s.get("reward_info") and s["reward_info"].get("reward") == 0.0]
    passes = [s for s in sims if s.get("reward_info") and s["reward_info"].get("reward") == 1.0]
    print("n=%d pass=%d fail=%d infra(reward None·제외계상)=%d" % (len(sims), len(passes), len(fails), len(infra)))

    recs = [analyze_sim(s, tasks[s["task_id"]]) for s in fails]

    # ---- 집계 ----
    print("\n#### primary buckets (170 fails) ####")
    for k, v in Counter(r["primary"] for r in recs).most_common():
        print("  %-36s %3d (%.1f%%)" % (k, v, 100 * v / len(recs)))

    print("\n#### lever fire counts (sim 단위·다중라벨) ####")
    def cnt(pred):
        n = sum(1 for r in recs if pred(r))
        return "%3d (%.1f%%)" % (n, 100 * n / len(recs))
    print("  GATE_GB1 fires          :", cnt(lambda r: r["flags"]["GATE_GB1"]["fires"]))
    print("  GATE_GB1 closure-plaus  :", cnt(lambda r: r["flags"]["GATE_GB1"]["closure_plausible"]))
    print("  PERARG fired            :", cnt(lambda r: r["flags"]["PERARG"]["fired_fields"]))
    print("  PERARG fired+recov      :", cnt(lambda r: r["flags"]["PERARG"]["recovery_plausible_fields"]))
    print("  DISAMB candidate        :", cnt(lambda r: r["flags"]["DISAMB"]["candidate_fields"]))
    print("  CALC derived-numeric    :", cnt(lambda r: r["flags"]["CALC_EXT"]["derived_numeric_fields"]))
    print("  CALC selection-numeric  :", cnt(lambda r: r["flags"]["CALC_EXT"]["selection_numeric_fields"]))
    print("  EPLAN L1                :", cnt(lambda r: r["flags"]["EPLAN"]["L1"]))
    print("  EPLAN L2                :", cnt(lambda r: r["flags"]["EPLAN"]["L2"]))
    print("  EPLAN walk(user_stop)   :", cnt(lambda r: r["flags"]["EPLAN"]["walk"]))
    print("  USER-side argdiff       :", cnt(lambda r: r["flags"]["USER_SIDE_ARGDIFF"]))
    print("  EXTRA writes            :", cnt(lambda r: r["extra_writes"]))
    print("  P2_DEFAULT / NOTICE     : 구조적 0 (banking A2 미인스턴스)")
    print("  --- v2 (정독-정정 후) ---")
    print("  PERARG all-call fired   :", cnt(lambda r: r["flags"]["PERARG_ALLCALL"]["fired"]))
    print("  FAB_TOOLNAME(도구명날조):", cnt(lambda r: r["flags"]["FAB_TOOLNAME"]))
    print("  LOGV_TIME_FAB           :", cnt(lambda r: r["flags"]["LOGV_TIME_FAB"]))
    print("  EARLY_TRANSFER          :", cnt(lambda r: r["flags"]["EARLY_TRANSFER"]))
    only_tv = sum(1 for r in recs if r["flags"]["LOGV_TIME_FAB"] and r["n_missing"] == 0
                  and not r["extra_writes"]
                  and all(all(f["field"] == "time_verified" or f["cls"] in ()
                              for f in ad["fields"]) for ad in r["argdiffs"]))
    print("  LOGV_TIME_FAB이 유일 diff:", only_tv)

    # ---- per-sim 폐쇄(closure) 상한 — all-or-nothing DB라 '모든' blocking item이 커버돼야 sim이 닫힘 ----
    def closure_tier(r):
        f = r["flags"]
        blockers_uncov_t1, blockers_uncov_t2 = 0, 0
        # 미실행 gold: reach — 어떤 레버도 못 채움 (gate-강제 lv 예외는 fires 조건 필요)
        for k in r["missing_keys"]:
            if k == "log_verification" and f["GATE_GB1"]["fires"]:
                continue
            blockers_uncov_t1 += 1
            blockers_uncov_t2 += 1
        # arg-diff 필드
        for ad in r["argdiffs"]:
            agent_side = (ad["requestor"] == "assistant")
            for fd in ad["fields"]:
                if fd["cls"] in ("EXTRA_FIELD",):
                    continue  # 잉여 필드는 env가 무시할 개연 — 비계상(관대)
                cov1 = cov2 = False
                if agent_side and fd["cls"] not in ("MISSING_FIELD",):
                    perarg_hit = fd.get("hinted") and not fd.get("exec_in_ctx") and \
                        (fd.get("gold_in_ctx") or fd.get("gold_in_full_ctx"))
                    disamb_hit = fd.get("exec_in_ctx") and fd.get("gold_in_ctx")
                    cov1 = bool(perarg_hit or disamb_hit)
                    tv_hit = (ad["key"] == "log_verification" and fd["field"] == "time_verified")
                    calc_hit = (fd["cls"] == "NUMERIC" and fd.get("gold_in_full_ctx"))
                    cov2 = bool(cov1 or tv_hit or calc_hit)
                if not cov1:
                    blockers_uncov_t1 += 1
                if not cov2:
                    blockers_uncov_t2 += 1
        # 과잉 write
        for k in r["extra_writes"]:
            is_tr = k in ("transfer_to_human_agents", "request_human_agent_transfer")
            if not is_tr:
                blockers_uncov_t1 += 1
                blockers_uncov_t2 += 1
            elif not f["EARLY_TRANSFER"]:
                blockers_uncov_t1 += 1
                blockers_uncov_t2 += 1
            else:
                blockers_uncov_t1 += 1  # T1: exhaust 미인스턴스 → 미커버
        return blockers_uncov_t1 == 0, blockers_uncov_t2 == 0
    t1 = sum(1 for r in recs if closure_tier(r)[0])
    t2 = sum(1 for r in recs if closure_tier(r)[1])
    print("\n#### per-sim 폐쇄 상한 (발화+복구개연이 '모든' blocking diff를 커버·정직 상한) ####")
    print("  T1 현행 스택(GB1+prov+DISAMB)     : %d/%d (%.1f%%)" % (t1, len(recs), 100 * t1 / len(recs)))
    print("  T2 +A2확장(time힌트·calc·exhaust) : %d/%d (%.1f%%)" % (t2, len(recs), 100 * t2 / len(recs)))

    print("\n#### missing key 유형 (REACH 내용물) ####")
    mk = Counter()
    for r in recs:
        for k in r["missing_keys"]:
            mk[k.split(":")[0]] += 1
    for k, v in mk.most_common():
        print("  %-40s %d" % (k, v))

    print("\n#### per-task 지도 (task: trial별 primary·주요 flag) ####")
    bytask = defaultdict(list)
    for r in recs:
        f = r["flags"]
        tags = []
        if f["GATE_GB1"]["fires"]: tags.append("GATE")
        if f["PERARG"]["fired_fields"]: tags.append("PERARG")
        if f["DISAMB"]["candidate_fields"]: tags.append("DISAMB")
        if f["CALC_EXT"]["derived_numeric_fields"]: tags.append("CALCd")
        if f["CALC_EXT"]["selection_numeric_fields"]: tags.append("CALCs")
        if f["EPLAN"]["L1"]: tags.append("EPL1")
        if f["EPLAN"]["L2"]: tags.append("EPL2")
        if f["EPLAN"]["walk"]: tags.append("EPwalk")
        if f["USER_SIDE_ARGDIFF"]: tags.append("USERarg")
        bytask[r["task_id"]].append("t%s:%s[%s] miss%d/%d" % (
            r["trial"], r["primary"].split("(")[0], ",".join(tags) or "-", r["n_missing"], r["ngold"]))
    for tid in sorted(bytask):
        print("  %-9s %s" % (tid, " | ".join(bytask[tid])))

    json.dump({"summary": {"n": len(sims), "pass": len(passes), "fail": len(fails), "infra": len(infra)},
               "records": recs}, io.open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nWROTE", out)


if __name__ == "__main__":
    gz = sys.argv[1] if len(sys.argv) > 1 else GZ
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    main(gz, out)
