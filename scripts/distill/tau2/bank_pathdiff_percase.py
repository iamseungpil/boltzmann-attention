# -*- coding: utf-8 -*-
"""banking per-case 경로 대조 (gold 시퀀스 ↔ 우리 호출 시퀀스). [[08]] 정독용.
용법: python bank_pathdiff_percase.py <task_id> [floor.gz]"""
import gzip, json, sys

TID = sys.argv[1]
P = sys.argv[2] if len(sys.argv) > 2 else \
    "C:/workspace/ba-frft/reports/facet_rft_2026/sim_results/bankxfer_floor_bank_t4.results.json.gz"
d = json.load(gzip.open(P))
tasks = {str(t.get("id")): t for t in (d.get("tasks") or [])}
t = tasks[TID]

print("###", TID, "| purpose:", (t.get("description") or {}).get("purpose", "")[:200] if isinstance(t.get("description"), dict) else str(t.get("description"))[:200])
ec = t.get("evaluation_criteria") or {}
print("\n=== GOLD actions (요구 시퀀스) ===")
for a in (ec.get("actions") or []):
    args = a.get("arguments") or {}
    key = args.get("agent_tool_name") or args.get("card_type") or args.get("transaction_type") or ""
    print("  [%s] %s(%s) key=%s" % (a.get("requestor"), a.get("name"),
          ",".join(sorted(args.keys())), key))
print("  reward_basis:", ec.get("reward_basis"))

# 첫 실패 trial 하나 정독
sims = [x for x in d["simulations"] if str(x["task_id"]) == TID]
s = None
for x in sims:
    if (x.get("reward_info") or {}).get("reward") not in (1.0, None):
        s = x; break
s = s or sims[0]
ri = s.get("reward_info") or {}
print("\n=== OUR sim (reward=%s db_match=%s) 호출 시퀀스 ===" % (
    ri.get("reward"), ((ri.get("db_check") or {}).get("db_match"))))
for m in s.get("messages") or []:
    r = m.get("role")
    for tc in (m.get("tool_calls") or []):
        a = tc.get("arguments") or {}
        key = a.get("agent_tool_name") or a.get("card_type") or a.get("query") or a.get("transaction_type") or ""
        print("  [%s] %s  key=%s" % (r, tc.get("name"), str(key)[:80]))
print("\n=== action_checks (gold별 매치) ===")
for ac in (ri.get("action_checks") or []):
    a = ac.get("action") or {}
    print("  %s match=%s args=%s" % (a.get("name"), ac.get("action_match"),
          json.dumps(a.get("arguments") or {}, ensure_ascii=False)[:120]))
