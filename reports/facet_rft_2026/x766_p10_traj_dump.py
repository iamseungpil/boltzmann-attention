# x766b — dump one task_010 simulation trajectory verbatim (read-only).
import gzip, json, os, sys

ROOT = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results"
fn = sys.argv[1]
seed = int(sys.argv[2])
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
hi = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9

with gzip.open(os.path.join(ROOT, fn), "rb") as fh:
    obj = json.load(fh)

for s in obj["simulations"]:
    if s.get("task_id") != "task_010":
        continue
    if s.get("seed") != seed:
        continue
    print("### FILE=%s sim_id=%s trial=%s seed=%s reward=%s nmsgs=%d term=%s"
          % (fn, s.get("id"), s.get("trial"), s.get("seed"),
             (s.get("reward_info") or {}).get("reward"), len(s.get("messages") or []),
             s.get("termination_reason")))
    ri = s.get("reward_info") or {}
    print("### reward_breakdown=%s" % json.dumps({k: v for k, v in ri.items() if k != "reward_breakdown"}, ensure_ascii=False)[:3000])
    for i, m in enumerate(s.get("messages") or []):
        if i < lo or i > hi:
            continue
        role = m.get("role")
        turn = m.get("turn_idx", m.get("turn", "?"))
        print("\n" + "=" * 90)
        print("[msg %d] role=%s turn_idx=%s tool_call_id=%s" % (i, role, turn, m.get("tool_call_id")))
        if m.get("content"):
            print("--- content ---")
            print(m["content"])
        tc = m.get("tool_calls")
        if tc:
            print("--- tool_calls ---")
            for c in tc:
                print(json.dumps(c, ensure_ascii=False))
        for k in ("name", "requestor", "error"):
            if m.get(k) is not None:
                print("--- %s: %s" % (k, m.get(k)))
    break
