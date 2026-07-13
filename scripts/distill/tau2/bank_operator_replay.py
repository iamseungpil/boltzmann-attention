# -*- coding: utf-8 -*-
"""오프라인 replay: resolve_operator(operator-provenance)를 기존 banking floor 데이터에 적용.
질문(U1 작동적-전이): 도구명 날조(banking 35.9% 보고)가 (a) 실제로 얼마나 발화하나
(b) 실패 sim과 연관되나 (c) fab-sim 중 그것만 고치면 닫힐 여지가 있나. gpt-4.1 0원."""
import gzip, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_resolve as R

P = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results",
                 "bankxfer_floor_bank_t4.results.json.gz")
P = os.path.abspath(P)


class M:
    def __init__(s, role, content, error=False):
        s.role, s.content, s.error = role, content, error


OPSPEC = {"kind": "operator", "arg": "agent_tool_name",
          "name_pattern": "[a-z_]+_[0-9]{4}", "operator_resolution": "discoverable"}
DISCO_TOOLS = ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool")


def main():
    d = json.load(gzip.open(P))
    sims = d["simulations"]
    tot_calls = fab_calls = 0
    sims_with_fab = {}
    for x in sims:
        tid = str(x["task_id"]); tr = x.get("trial")
        db = bool(((x.get("reward_info") or {}).get("db_check") or {}).get("db_match"))
        msgs = []
        fabbed = False
        for m in (x.get("messages") or []):
            role = m.get("role")
            msgs.append(M(role, m.get("content"), m.get("error", False)))
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc.get("name") in DISCO_TOOLS:
                        args = tc.get("arguments") or {}
                        if isinstance(args, str):
                            try: args = json.loads(args)
                            except Exception: args = {}
                        tot_calls += 1
                        if R.resolve_operator(OPSPEC, args, msgs).get("status") == "deny":
                            fab_calls += 1; fabbed = True
        if fabbed:
            sims_with_fab[(tid, tr)] = db
    n = len(sims)
    npass = sum(1 for x in sims if ((x.get("reward_info") or {}).get("db_check") or {}).get("db_match"))
    fabfail = sum(1 for v in sims_with_fab.values() if not v)
    fabpass = sum(1 for v in sims_with_fab.values() if v)
    print("banking floor: n=%d pass=%d (%.3f)" % (n, npass, npass / n))
    print("discoverable-tool calls: %d, operator-fab deny: %d (%.1f%%)"
          % (tot_calls, fab_calls, 100 * fab_calls / max(1, tot_calls)))
    print("sims with >=1 operator-fab: %d/%d (%.1f%%)" % (len(sims_with_fab), n, 100 * len(sims_with_fab) / n))
    print("  fab-sim 중 fail=%d pass=%d (fab이 실패와 연관되나)" % (fabfail, fabpass))
    print("  ★상한: fab-fail sim=%d 이 operator-prov의 최대 타깃(단 sim-폐쇄=전 blocker 커버 필요)" % fabfail)


if __name__ == "__main__":
    main()
