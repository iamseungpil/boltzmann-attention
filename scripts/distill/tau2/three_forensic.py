import gzip, json, re
GZ="/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/bank_all97_nt1_v2_20260718.results.json.gz"
raw=json.load(gzip.open(GZ))
sims={s["task_id"]:s for s in raw["simulations"]}
TASKS=["task_058","task_070","task_075"]
for tid in TASKS:
    s=sims[tid]; msgs=s.get("messages") or []
    print("\n"+"="*100)
    print("### %s  nmsg=%d  term=%s"%(tid,len(msgs),s.get("termination_reason")))
    # gold actions
    ri=s.get("reward_info") or {}
    ac=ri.get("action_checks") or []
    print("--- GOLD actions (match?) ---")
    for a in ac:
        act=a["action"]; nm=act["name"]
        inner=""
        if nm in ("unlock_discoverable_agent_tool","call_discoverable_agent_tool"):
            inner="→"+str((act.get("arguments") or {}).get("agent_tool_name"))
        print("   %-6s %s%s  args=%s"%("OK" if a["action_match"] else "MISS", nm, inner,
                                       json.dumps(act.get("arguments"))[:100]))
    print("--- TRAJECTORY (tool calls + key results + assistant text) ---")
    for i,m in enumerate(msgs):
        role=m.get("role"); c=(m.get("content") or "")
        for tc in (m.get("tool_calls") or []):
            nm=str(tc.get("name")); a=tc.get("arguments") or {}
            if not isinstance(a,dict):
                try: a=json.loads(a)
                except Exception: a={}
            extra=""
            if nm in ("unlock_discoverable_agent_tool","call_discoverable_agent_tool"):
                extra=" →"+str(a.get("agent_tool_name"))
            print("  %3d [TOOL] %s%s %s"%(i,nm,extra,json.dumps(a)[:110]))
        if role=="tool" and c:
            t=c.strip().replace("\n"," ")
            if re.search(r"error|not found|unknown|invalid|fail", t, re.I):
                print("  %3d [RESULT-ERR] %s"%(i,t[:150]))
        if role=="assistant" and c.strip():
            print("  %3d [asst] %s"%(i,c.strip().replace("\n"," ")[:130]))
        if role=="user" and c.strip():
            print("  %3d [USER] %s"%(i,c.strip().replace("\n"," ")[:130]))
