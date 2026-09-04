import gzip, json, sys
p = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else "submit_referral"
with gzip.open(p, 'rt', encoding='utf-8') as f:
    d = json.load(f)
for si, s in enumerate(d.get("simulations", [])):
    print("=== SIM", si, s.get("task_id"), "reward=", (s.get("reward_info") or {}).get("reward"))
    for i, m in enumerate(s.get("messages", [])):
        c = m.get("content") or ""
        tc = m.get("tool_calls") or []
        blob = c + json.dumps(tc, ensure_ascii=False)
        if pat in blob:
            print("--- msg", i, "role=", m.get("role"), "requestor=", m.get("requestor"))
            print(blob[:6000])
            print()
