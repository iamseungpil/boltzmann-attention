#!/usr/bin/env python
"""E-COMP 사전 census — prov arm(456 sims) 실패 전수를 3축(관찰->기능->레버)으로 분류하고
각 버킷의 레버-도달가능성을 궤적에서 직접 측정 ([[08]] 전수 포렌식·무료).

관찰 버킷 (DB-기준·dbonly_forensic 계열·[[48]] 서술형 이름만):
  NL_ONLY          db는 맞는데 reward<1 (NL/communicate 채점)
  ZERO_WRITE_ATT   성공 write 0·시도는 함(전부 에러)
  ZERO_WRITE_NEV   성공 write 0·시도조차 없음
  WRONG_REF_ORDER  같은 도구·order_id 불일치 (참조매칭)
  WRONG_ITEMS      item_ids/new_item_ids 불일치 (변형/집합 선택)
  WRONG_PAYMENT    payment_method_id 불일치
  WRONG_ADDRESS    주소 필드 불일치
  WRONG_OP         gold와 다른 write 도구 사용(교환<->반품 등)
  OVER_ACTION      gold에 없는 write 실행
  MISSED_WRITE     gold write 일부 미실행(부분 커버리지)
  OTHER_ARG        기타 인자 불일치

레버-도달가능성 (버킷과 직교·per-sim 플래그):
  disamb: 잘못 쓴 인자에 대해 gold 값이 그 write *이전* 문맥(도구출력/유저발화)에 실재
          AND 쓴 값도 실재 (=|C|>=2 same-type = T2_DISAMB 발화 조건 근사)
  gate_constraints: 인자만으로 결정가능한 제약위반 write 시도 존재
          (equal_len: item_ids != new_item_ids 길이 / disjoint: 두 리스트 교집합)
  gate_retry: 동일 write 도구 3+회 에러 (RETRY/steer 게이트 대상 근사)
  fab_residual: 쓴 인자값이 이전 문맥에 부재 (prov 잔존 날조 근사·identifying만)

usage: ecomp_fail_census.py --results <results.json|.gz> --tasks <tasks.json> [--a2 <retail.gate.json>]
"""
import argparse, gzip, json, os, sys
from collections import Counter, defaultdict

ID_HINTS = ("order_id", "item_ids", "new_item_ids", "payment_method_id", "user_id")


def load_json(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def args_of(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def flat_vals(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from flat_vals(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from flat_vals(x)
    else:
        yield v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--a2", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2", "retail.gate.json"))
    ap.add_argument("--dump", default=None, help="per-sim jsonl dump path")
    a = ap.parse_args()

    a2 = load_json(a.a2)
    WRITE_TOOLS = sorted({t for g in a2["gates"] if g.get("kind") == "confirm" for t in g.get("applies_to", [])})
    tasks = {t["id"]: t for t in load_json(a.tasks)}
    sims = load_json(a.results)["simulations"]

    buckets = Counter()
    lever = defaultdict(Counter)   # bucket -> lever flags count
    per_sim_rows = []
    n_fail = 0

    for s in sims:
        ri = s.get("reward_info") or {}
        r = ri.get("reward")
        if r is None or r >= 1:
            continue
        n_fail += 1
        tid = str(s.get("task_id"))
        task = tasks.get(tid) or tasks.get("%s" % tid) or {}
        gold = [(x.get("name"), args_of(x.get("arguments")))
                for x in ((task.get("evaluation_criteria") or {}).get("actions") or [])
                if x.get("requestor", "assistant") == "assistant" and x.get("name") in WRITE_TOOLS]

        msgs = s.get("messages") or []
        res_by_id = {m.get("id"): m for m in msgs if m.get("role") == "tool"}
        # 시간순 write 시도/성공 + 각 시점 이전 문맥
        ctx_parts = []
        exec_writes, attempted, err_counts = [], [], Counter()
        constraints_viol = False
        for m in msgs:
            role = m.get("role")
            c = m.get("content")
            if role == "user" and isinstance(c, str):
                ctx_parts.append(c.lower())
            for tc in (m.get("tool_calls") or []) if role == "assistant" else []:
                nm, ar = tc.get("name"), args_of(tc.get("arguments"))
                if nm in WRITE_TOOLS:
                    attempted.append(nm)
                    v1, v2 = ar.get("item_ids"), ar.get("new_item_ids")
                    if isinstance(v1, list) and isinstance(v2, list):
                        if len(v1) != len(v2) or (set(map(str, v1)) & set(map(str, v2))):
                            constraints_viol = True
                    tm = res_by_id.get(tc.get("id"))
                    ok = tm is not None and not tm.get("error")
                    if ok:
                        exec_writes.append((nm, ar, " ".join(ctx_parts)))
                    else:
                        err_counts[nm] += 1
            if role == "tool" and isinstance(c, str) and not m.get("error"):
                ctx_parts.append(c.lower())

        db = (ri.get("db_check") or {})
        db_ok = db.get("db_match") if isinstance(db, dict) else None

        flags = {"disamb": False, "gate_constraints": constraints_viol,
                 "gate_retry": any(v >= 3 for v in err_counts.values()), "fab_residual": False}
        bucket = None

        if db_ok is True:
            bucket = "NL_ONLY"
        elif not exec_writes and gold:
            bucket = "ZERO_WRITE_ATT" if attempted else "ZERO_WRITE_NEV"
        elif exec_writes:
            gold_by_tool = defaultdict(list)
            for nm, ar in gold:
                gold_by_tool[nm].append(ar)
            extra = [w for w in exec_writes if w[0] not in gold_by_tool]
            wrong_fields = []
            for nm, ar, ctx in exec_writes:
                cands = gold_by_tool.get(nm) or []
                if not cands:
                    continue
                best = min(cands, key=lambda g: sum(1 for k in set(g) | set(ar)
                                                    if str(g.get(k)) != str(ar.get(k))))
                for k in set(best) | set(ar):
                    if str(best.get(k)) == str(ar.get(k)):
                        continue
                    wrong_fields.append(k)
                    gv, av = best.get(k), ar.get(k)
                    gold_in = all(str(x).lower() in ctx for x in flat_vals(gv) if x is not None and len(str(x)) >= 4)
                    act_in = all(str(x).lower() in ctx for x in flat_vals(av) if x is not None and len(str(x)) >= 4)
                    if gold_in and act_in:
                        flags["disamb"] = True
                    if not act_in and any(h in k for h in ID_HINTS):
                        flags["fab_residual"] = True
            missed = []
            for nm, cands in gold_by_tool.items():
                done = sum(1 for w in exec_writes if w[0] == nm)
                if done < len(cands):
                    missed.append(nm)
            if extra:
                bucket = "OVER_ACTION"
            elif "order_id" in wrong_fields:
                bucket = "WRONG_REF_ORDER"
            elif any(k in ("item_ids", "new_item_ids") for k in wrong_fields):
                bucket = "WRONG_ITEMS"
            elif any("payment" in k for k in wrong_fields):
                bucket = "WRONG_PAYMENT"
            elif any(k in ("address1", "address2", "city", "state", "zip", "country") for k in wrong_fields):
                bucket = "WRONG_ADDRESS"
            elif wrong_fields:
                bucket = "OTHER_ARG"
            elif missed:
                bucket = "MISSED_WRITE"
            elif set(w[0] for w in exec_writes) != set(gold_by_tool):
                bucket = "WRONG_OP"
            else:
                bucket = "ARGS_MATCH_DB_FAIL"  # 순서/수량 미묘 or C26류
        else:
            bucket = "NO_GOLD_WRITE_DB_FAIL"

        buckets[bucket] += 1
        for k, v in flags.items():
            if v:
                lever[bucket][k] += 1
        per_sim_rows.append({"task_id": tid, "trial": s.get("trial"), "bucket": bucket, **flags,
                             "term": s.get("termination_reason")})

    print("n_sims=%d n_fail=%d" % (len(sims), n_fail))
    print("%-22s %5s %6s | %7s %9s %6s %6s" % ("bucket", "n", "%fail", "disamb", "gate_cons", "retry", "fabres"))
    for b, n in buckets.most_common():
        lv = lever[b]
        print("%-22s %5d %5.1f%% | %7d %9d %6d %6d" % (
            b, n, 100.0 * n / max(n_fail, 1), lv["disamb"], lv["gate_constraints"],
            lv["gate_retry"], lv["fab_residual"]))
    tot = Counter()
    for b in lever.values():
        tot.update(b)
    print("TOTAL lever-reachable: disamb=%d gate_cons=%d retry=%d fab_res=%d (of %d fails; sims=456 => 1pp≈4.56)"
          % (tot["disamb"], tot["gate_constraints"], tot["gate_retry"], tot["fab_residual"], n_fail))
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            for row in per_sim_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("dumped:", a.dump)


if __name__ == "__main__":
    main()
