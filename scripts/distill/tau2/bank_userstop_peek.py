# -*- coding: utf-8 -*-
"""REACH+user_stop sim의 종료 직전 대화를 봐서 '왜 user가 멈췄나' 판별:
  (A) 에이전트가 '완료' 선언(미완인데) → user 수락 = '언제 끝인가' 오판(루프 terminal)
  (B) 에이전트가 되묻기/스톨 → user 이탈 = over-ask/stall
  (C) 에이전트가 못한다고 함 → user 포기 = abstain 오판
샘플 6개의 마지막 assistant 2발화 + 마지막 user 발화 인용."""
import gzip, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                 "sim_results", "bankxfer_floor_bank_t4.results.json.gz"))
WRITE_HINT = ("unlock_", "call_discoverable", "apply_", "submit_", "change_", "log_")


def main():
    d = json.load(gzip.open(P))
    tasks = {str(t.get("id")): t for t in (d.get("tasks") or [])}
    shown = 0
    for x in d["simulations"]:
        if shown >= 6:
            break
        if ((x.get("reward_info") or {}).get("db_check") or {}).get("db_match"):
            continue
        term = str(x.get("termination_reason") or "")
        if "user" not in term.lower() and "stop" not in term.lower():
            continue
        tid = str(x["task_id"])
        gold = (tasks.get(tid, {}).get("evaluation_criteria") or {}).get("actions") or []
        gold_w = [a.get("name") for a in gold if any(h in (a.get("name") or "") for h in WRITE_HINT)]
        msgs = x.get("messages") or []
        exec_w = [tc.get("name") for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")
                  for tc in m["tool_calls"] if any(h in (tc.get("name") or "") for h in WRITE_HINT)]
        if not (gold_w and len(exec_w) < 0.5 * len(gold_w)):
            continue
        shown += 1
        print("=" * 70)
        print("TASK %s  term=%s  gold_writes=%d exec_writes=%d" % (tid, term, len(gold_w), len(exec_w)))
        # 마지막 3 non-tool 메시지
        tail = [m for m in msgs if m.get("role") in ("assistant", "user")][-4:]
        for m in tail:
            c = m.get("content")
            if not isinstance(c, str):
                tcs = m.get("tool_calls") or []
                c = "TOOLCALL " + ",".join(t.get("name") for t in tcs) if tcs else "(none)"
            print("  [%s] %s" % (m.get("role"), c[:220].replace("\n", " ")))


if __name__ == "__main__":
    main()
