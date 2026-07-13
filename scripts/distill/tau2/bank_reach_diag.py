# -*- coding: utf-8 -*-
"""banking 지배잔여 진단: 실패 sim이 '어디서' 막히나 = GET(발견/조립) vs 그 외.
gold action 수 대비 실제 실행된 action 수(미실행율)·KB검색 성공여부·transfer 조기여부."""
import gzip, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                 "sim_results", "bankxfer_floor_bank_t4.results.json.gz"))
WRITE_HINT = ("unlock_", "call_discoverable", "apply_", "submit_", "change_", "log_", "give_")


def main():
    d = json.load(gzip.open(P))
    sims = d["simulations"]
    tasks = {str(t.get("id")): t for t in (d.get("tasks") or [])}
    n = 0; fails = 0
    reach = 0        # gold action 절반 미만 실행 = 조립 미완
    argdiff = 0      # gold 대부분 실행했으나 인자 불일치
    early_xfer = 0   # gold엔 없는 transfer 실행
    kb_fail = 0      # KB_search 에러 존재
    for x in sims:
        db = bool(((x.get("reward_info") or {}).get("db_check") or {}).get("db_match"))
        if db:
            continue
        n += 1; fails += 1
        tid = str(x["task_id"])
        gold = (tasks.get(tid, {}).get("evaluation_criteria") or {}).get("actions") or []
        gold_names = [a.get("name") for a in gold]
        gold_writes = [g for g in gold_names if any(h in (g or "") for h in WRITE_HINT)]
        # 실제 실행된 action(성공 tool 결과 있는 assistant tool_call)
        exec_names = []
        kberr = False
        for m in (x.get("messages") or []):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    exec_names.append(tc.get("name"))
            if m.get("role") == "tool" and m.get("error") and "OPENAI" in str(m.get("content") or ""):
                kberr = True
        exec_writes = [e for e in exec_names if any(h in (e or "") for h in WRITE_HINT)]
        if kberr:
            kb_fail += 1
        # 분류
        if gold_writes and len(exec_writes) < 0.5 * len(gold_writes):
            reach += 1
        elif "transfer_to_human_agents" in exec_names and "transfer_to_human_agents" not in gold_names:
            early_xfer += 1
        else:
            argdiff += 1
    print("banking floor 실패 %d개 분류(gold write 대비 실행율):" % fails)
    print("  REACH(조립 미완·gold write 절반↓ 실행): %d (%.0f%%)" % (reach, 100 * reach / fails))
    print("  ARGDIFF(대부분 실행·인자 불일치):      %d (%.0f%%)" % (argdiff, 100 * argdiff / fails))
    print("  EARLY_TRANSFER(gold 없는 transfer):    %d (%.0f%%)" % (early_xfer, 100 * early_xfer / fails))
    print("  KB_search 에러 존재 sim:               %d (%.0f%%)" % (kb_fail, 100 * kb_fail / fails))


if __name__ == "__main__":
    main()
