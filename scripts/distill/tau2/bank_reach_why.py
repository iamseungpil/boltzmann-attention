# -*- coding: utf-8 -*-
"""REACH(조립 미완) 실패가 '왜' 멈추나 — 능력(learn) vs 판단(우리 루프 abstain/select) 판별.
REACH sim의 종료직전 몇 스텝을 분류:
  (A) 조기 transfer/포기 = "다음 진행 vs 중단" 오판 = on_error/abstain 판단 (우리 루프)
  (B) user가 STOP/이탈로 종결 = 판단 기회 소진
  (C) max_steps = 무한루프/thrash (판단 실패의 일종)
  (D) 순수 텍스트로 끝(행동 안 하고 대화만) = 계획/실행 미개시 (learn 근접)
  + 종료 직전 도구실패(에러)에 어떻게 반응했나(재시도 vs 포기)."""
import gzip, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                 "sim_results", "bankxfer_floor_bank_t4.results.json.gz"))
WRITE_HINT = ("unlock_", "call_discoverable", "apply_", "submit_", "change_", "log_")


def main():
    d = json.load(gzip.open(P))
    sims = d["simulations"]
    tasks = {str(t.get("id")): t for t in (d.get("tasks") or [])}
    cat = {"transfer_give_up": 0, "user_stop": 0, "max_steps": 0, "text_only_no_action": 0, "other": 0}
    err_then_giveup = 0; err_then_retry = 0; had_err = 0
    reach_n = 0
    for x in sims:
        db = bool(((x.get("reward_info") or {}).get("db_check") or {}).get("db_match"))
        if db:
            continue
        tid = str(x["task_id"])
        gold = (tasks.get(tid, {}).get("evaluation_criteria") or {}).get("actions") or []
        gold_w = [a.get("name") for a in gold if any(h in (a.get("name") or "") for h in WRITE_HINT)]
        msgs = x.get("messages") or []
        exec_w = [tc.get("name") for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
                  for tc in m["tool_calls"] if any(h in (tc.get("name") or "") for h in WRITE_HINT)]
        if not (gold_w and len(exec_w) < 0.5 * len(gold_w)):
            continue   # REACH만
        reach_n += 1
        term = str(x.get("termination_reason") or "")
        # 마지막 assistant 행동 유형
        last_assts = [m for m in msgs if m.get("role") == "assistant"]
        did_transfer = any((tc.get("name") or "").startswith(("transfer", "request_human"))
                           for m in last_assts if m.get("tool_calls") for tc in m["tool_calls"])
        n_tool_calls = sum(1 for m in last_assts if m.get("tool_calls"))
        # 종료 직전 에러 반응
        errs = [i for i, m in enumerate(msgs) if m.get("role") == "tool" and m.get("error")]
        if errs:
            had_err += 1
            last_err_i = errs[-1]
            after = [m for m in msgs[last_err_i + 1:] if m.get("role") == "assistant" and m.get("tool_calls")]
            if after:
                err_then_retry += 1
            else:
                err_then_giveup += 1
        # 분류
        if "max_step" in term:
            cat["max_steps"] += 1
        elif did_transfer and "transfer_to_human_agents" not in [a.get("name") for a in gold]:
            cat["transfer_give_up"] += 1
        elif n_tool_calls == 0:
            cat["text_only_no_action"] += 1
        elif "user" in term.lower() or "stop" in term.lower():
            cat["user_stop"] += 1
        else:
            cat["other"] += 1
    print("REACH 실패 %d개 — 종료 양상(왜 멈췄나):" % reach_n)
    for k, v in cat.items():
        print("  %-22s %d (%.0f%%)" % (k, v, 100 * v / max(1, reach_n)))
    print()
    print("종료 직전 도구-에러 반응(에러 있던 %d sim 중):" % had_err)
    print("  에러 후 재시도(행동 지속): %d (%.0f%%)" % (err_then_retry, 100 * err_then_retry / max(1, had_err)))
    print("  에러 후 포기(행동 중단):   %d (%.0f%%)" % (err_then_giveup, 100 * err_then_giveup / max(1, had_err)))
    print()
    print("해석 축: transfer_give_up+에러후포기 = '중단 오판(우리 루프 abstain/on_error)'")
    print("         text_only_no_action = '계획/실행 미개시(learn 근접)'")


if __name__ == "__main__":
    main()
