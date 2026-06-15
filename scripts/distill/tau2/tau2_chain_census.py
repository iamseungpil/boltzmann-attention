#!/usr/bin/env python
"""τ² 전수 체인-완성 census (2026-06-15) — "gather 늘었는데 왜 pass 0?"의 진짜 병목 단계 규명.

각 궤적이 표준 체인 어디서 죽나:
  auth(find_user) → gather(get_user_details=주문/결제목록) → real-order(get_order_details 진짜 id)
  → write(exchange/return/modify) → write_ok(gold 매치)
+ ★핵심: gather 했는데도 *여전히 날조*하나? (get_user_details 출력에 진짜 order_id가 있는데
  get_order_details에 fabricated id 씀 = 추출/select 실패 = P2b 더 깊은 층 vs P4).

Usage: tau2_chain_census.py <sim_dir> [--show]
"""
import argparse, json, os, re
from collections import Counter

AUTH = ("find_user", "get_user_id", "find_customer")
ERR_RE = re.compile(r"\berror\b|not found|not exist|invalid|policy gate|requires|denied", re.I)


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    return json.load(open(path, encoding="utf-8"))


def is_err(o):
    return o is None or bool(ERR_RE.search(str(o)))


def calls_of(msgs):
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                nm = fn.get("name") or ""
                ar = fn.get("arguments")
                if isinstance(ar, str):
                    try:
                        ar = json.loads(ar)
                    except Exception:
                        ar = {}
                nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
                outp = str(nxt.get("content")) if nxt.get("role") == "tool" else None
                out.append((nm, ar or {}, outp))
    return out


def analyze(sim):
    ri = sim.get("reward_info") or {}
    reward = ri.get("reward", 0.0)
    term = sim.get("termination_reason")
    calls = calls_of(sim.get("messages") or [])

    auth_ok = gather_ok = order_ok = write_attempt = False
    seen_output = ""  # 누적 tool 출력(=진짜값 풀)
    gather_then_fab = 0  # gather 후에도 날조 order_id
    n_fab_order = n_real_order = 0
    WRITE = ("exchange_", "return_", "modify_", "cancel_", "place_", "update_")

    for nm, ar, outp in calls:
        err = is_err(outp)
        if any(nm.startswith(p) for p in AUTH) and not err:
            auth_ok = True
        if nm == "get_user_details" and not err:
            gather_ok = True
        if nm == "get_order_details":
            oid = str(ar.get("order_id", ""))
            grounded = oid and oid.lower() in seen_output.lower()
            if grounded:
                n_real_order += 1
                if not err:
                    order_ok = True
            else:
                n_fab_order += 1
                if gather_ok:  # gather 했는데도 날조 = 추출 실패
                    gather_then_fab += 1
        if any(nm.startswith(p) for p in WRITE):
            write_attempt = True
        if outp:
            seen_output += " " + str(outp)

    write_ok = any(a.get("tool_type") == "write" and a.get("action_match")
                   for a in (ri.get("action_checks") or []))

    if reward and reward > 0:
        stall = "PASS"
    elif not auth_ok:
        stall = "1_never_auth"
    elif not gather_ok:
        stall = "2_auth_no_gather"
    elif not order_ok:
        stall = "3_gather_no_realorder"  # ★gather했으나 진짜 order 못 얻음(날조 지속/추출실패)
    elif not write_attempt:
        stall = "4_order_no_write"
    elif not write_ok:
        stall = "5_write_failed"
    else:
        stall = "6_other"

    return {
        "task": sim.get("task_id"), "stall": stall, "term": term,
        "auth": auth_ok, "gather": gather_ok, "order": order_ok,
        "write_try": write_attempt, "write_ok": write_ok,
        "fab_order": n_fab_order, "real_order": n_real_order,
        "gather_then_fab": gather_then_fab, "n_calls": len(calls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    sims = load(a.path)["simulations"]
    rows = [analyze(s) for s in sims]
    rows.sort(key=lambda x: x["stall"])

    print("=== τ² CHAIN CENSUS: %s (n=%d) ===" % (a.path, len(rows)))
    if a.show:
        for x in rows:
            print("  task %-5s %-22s auth=%d gather=%d order=%d write=%d/%d | fab_ord=%d real_ord=%d g→fab=%d | %s"
                  % (str(x["task"]), x["stall"], x["auth"], x["gather"], x["order"],
                     x["write_try"], x["write_ok"], x["fab_order"], x["real_order"],
                     x["gather_then_fab"], x["term"]))

    print("\n=== ★병목 단계 분포 (어디서 죽나) ===")
    for s, n in sorted(Counter(x["stall"] for x in rows).items()):
        print("  %-24s %d" % (s, n))

    print("\n=== 체인 통과율 ===")
    n = len(rows)
    print("  auth_ok        : %d/%d" % (sum(x["auth"] for x in rows), n))
    print("  gather_ok      : %d/%d" % (sum(x["gather"] for x in rows), n))
    print("  real_order_ok  : %d/%d" % (sum(x["order"] for x in rows), n))
    print("  write_attempt  : %d/%d" % (sum(x["write_try"] for x in rows), n))
    print("  write_ok(PASS) : %d/%d" % (sum(x["write_ok"] for x in rows), n))

    print("\n=== ★gather 생산성 (gather가 추출로 이어지나) ===")
    gathered = [x for x in rows if x["gather"]]
    g_then_fab = sum(1 for x in gathered if x["gather_then_fab"] > 0)
    g_to_order = sum(1 for x in gathered if x["order"])
    print("  gather한 태스크: %d" % len(gathered))
    print("    그중 real-order 추출 성공: %d" % g_to_order)
    print("    그중 gather 후에도 order_id 날조(추출 실패): %d" % g_then_fab)
    tot_fab = sum(x["fab_order"] for x in rows)
    tot_real = sum(x["real_order"] for x in rows)
    print("  전체 get_order_details: 날조 %d / 진짜 %d (진짜비율 %.0f%%)"
          % (tot_fab, tot_real, 100 * tot_real / max(1, tot_fab + tot_real)))


if __name__ == "__main__":
    main()
