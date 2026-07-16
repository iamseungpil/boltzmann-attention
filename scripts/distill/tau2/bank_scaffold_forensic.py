# -*- coding: utf-8 -*-
"""bank_scaffold_forensic.py — 43태스크 전수 per-step 분해 + E-PLAN scaffold 개입 결정론 재구성.

목적(사용자 지시 2026-07-16): floor vs fullstack(gate1+E-PLAN walk+L1/L2 deny) 43 궤적을
per-step으로 분해하고 "어떤 scaffold가 막았는지" 기전 확정.

방법([[08]]·[[09]] 무료): scaffold 개입(L1/L2 deny·CP5 walk)은 생성-레벨이라 저장 궤적에
안 남음 → 실제 결정론 함수(t2_eplan_patch)를 저장 궤적의 도구호출 시퀀스에 재통과시켜
각 write 시도에서 deny 발화 여부, stop에서 walk 발화 여부를 재구성.

fs 런 플래그: T2_EPLAN=1 T2_EPLAN_WALK=1 (examined_safe/reads_only/replan OFF).
"""
import json, os, re, sys, io, gzip
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

D = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
sys.path.insert(0, D)
# fs 런 플래그 재현(재구성 결정론에 영향)
os.environ["T2_EPLAN"] = "1"
os.environ["T2_EPLAN_WALK"] = "1"
os.environ.pop("T2_EPLAN_EXAMINED_SAFE", None)
os.environ.pop("T2_EPLAN_READS_ONLY", None)
os.environ.pop("T2_EPLAN_REPLAN", None)

import t2_eplan_patch as ep

DOM = "banking_knowledge"
SPEC = ep.load_eplan_spec(DOM)
GATE_WT = ep.load_write_tools(DOM)               # confirm-gate write set (banking=∅)
EP_WT = set(SPEC.get("write_tools") or ())        # eplan dispute write set
WT_ALL = GATE_WT | EP_WT
DISP = SPEC.get("dispatch_tool")
DKEY = SPEC.get("dispatch_name_key", "agent_tool_name")
DARG = SPEC.get("dispatch_args_key", "arguments")
EKEY = SPEC.get("entity_key")
IKEY = SPEC.get("items_key", "item_ids")
_fam = lambda n: re.sub(r"_\d+$", "", str(n or ""))

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    d = json.load(op(path, "rt", encoding="utf-8"))
    return {s.get("task_id"): s for s in d.get("simulations", [])}

def unwrap(nm, ar):
    """dispatch면 nested {name,args}로 풀어 실제 도구명/인자."""
    if isinstance(ar, str):
        try: ar = json.loads(ar)
        except Exception: ar = {}
    ar = ar if isinstance(ar, dict) else {}
    if DISP and nm == DISP:
        enm = _fam(ar.get(DKEY, ""))
        inner = ar.get(DARG)
        if isinstance(inner, str):
            try: inner = json.loads(inner)
            except Exception: inner = {}
        return enm, (inner if isinstance(inner, dict) else {})
    return nm, ar

def is_ep_write(nm):
    return nm in WT_ALL or _fam(nm) in WT_ALL

def tool_calls(m):
    return m.get("tool_calls") or []

def analyze_task(tid, s, arm):
    msgs = s.get("messages") or []
    r = (s.get("reward_info") or {}).get("reward")
    term = s.get("termination_reason")
    events = []              # per-step scaffold-relevant events
    gate_blocks = []         # POLICY GATE blocks visible in committed tool responses
    all_writes = []          # every write attempt (unwrapped)
    all_tool_names = []
    for i, m in enumerate(msgs):
        role = m.get("role")
        c = m.get("content")
        # committed gate-policy block (tool error content)
        if role == "tool" and isinstance(c, str) and ("POLICY GATE" in c or "blocked by policy" in c):
            gm = re.search(r"POLICY GATE (\w+)", c)
            gate_blocks.append(gm.group(1) if gm else "?")
        if role != "assistant":
            continue
        for tc in tool_calls(m):
            nm0 = tc.get("name") or ""
            enm, ear = unwrap(nm0, tc.get("arguments"))
            all_tool_names.append(enm)
            if is_ep_write(enm):
                # 이 write 시도 직전까지의 ledger로 discovery_precondition 재구성
                led = ep.build_ledger_from_messages(msgs[:i], SPEC, GATE_WT)
                ent = ear.get(EKEY)
                items = ear.get(IKEY) or ()
                deny = ep.discovery_precondition(led, SPEC, enm, items, ent)
                all_writes.append((enm, ep._norm(ent) if ent else None, bool(deny)))
                if deny:
                    kind = "L1(list-first)" if "list the customer" in deny or "MULTIPLE records" in deny else "L2(read-siblings)"
                    events.append(("DENY@%d" % i, enm, ep._norm(ent) if ent else "?", kind))
    # stop-time walk 재구성 (user_stop 일 때만·fs 조건)
    walk = None
    if term and "user_stop" in str(term).lower():
        led = ep.build_ledger_from_messages(msgs, SPEC, GATE_WT)
        unexamined = sorted(led.listed - led.examined)
        n = ep.walk_required_n(led, unexamined)
        mm = len({e["entity"] for e in led.executed})
        if n <= 1 or n <= mm:
            walk = ("no-gap", n, mm, len(unexamined))
        elif ep.qty_item_covered(led, n):
            walk = ("suppressed(item-covered)", n, mm, len(unexamined))
        else:
            walk = ("FIRE", n, mm, len(unexamined))
    return dict(tid=tid, arm=arm, r=r, term=term, events=events,
                gate_blocks=gate_blocks, writes=all_writes, walk=walk,
                ntool=len(all_tool_names), tools=all_tool_names)

def main():
    base = "/home/woori/scratch/tau2-bench/data/simulations"
    fl = load(base + "/bank_floor_bm/results.json")
    fs = load(base + "/bank_fs_bm/results.json")
    common = sorted(set(fl) & set(fs), key=lambda t: int(re.sub(r"\D", "", t) or 0))
    print("SPEC eplan write_tools:", sorted(EP_WT))
    print("dispatch_tool:", DISP, "| entity_key:", EKEY)
    print("=" * 100)
    # 집계
    agg = dict(fl_deny=0, fs_deny=0, fl_walkfire=0, fs_walkfire=0,
               fl_gate=0, fs_gate=0, regressions=[], ep_applicable=0)
    for t in common:
        A = analyze_task(t, fl[t], "floor")
        B = analyze_task(t, fs[t], "fs")
        fl_w = len(A["writes"]); fs_w = len(B["writes"])
        fl_d = sum(1 for _,_,d in A["writes"] if d)
        fs_d = sum(1 for _,_,d in B["writes"] if d)
        agg["fl_deny"] += fl_d; agg["fs_deny"] += fs_d
        agg["fl_gate"] += len(A["gate_blocks"]); agg["fs_gate"] += len(B["gate_blocks"])
        if A["walk"] and A["walk"][0] == "FIRE": agg["fl_walkfire"] += 1
        if B["walk"] and B["walk"][0] == "FIRE": agg["fs_walkfire"] += 1
        if fl_w or fs_w: agg["ep_applicable"] += 1
        regr = (A["r"] == 1.0 and B["r"] != 1.0)
        if regr: agg["regressions"].append(t)
        flag = " <<< REGRESSION" if regr else ""
        print("\n### %s  floor r=%s | fs r=%s%s" % (t, A["r"], B["r"], flag))
        for lbl, X in [("floor", A), ("fs", B)]:
            wsum = ", ".join("%s@%s%s" % (w, e, "[DENY]" if d else "") for w,e,d in X["writes"]) or "(no ep-write)"
            gb = (" gate-blocks=%s" % X["gate_blocks"]) if X["gate_blocks"] else ""
            wk = (" walk=%s(n=%d,m=%d,unex=%d)" % X["walk"]) if X["walk"] else " walk=n/a"
            print("   %-5s term=%s nt=%d ep-writes=[%s]%s%s"
                  % (lbl, X["term"], X["ntool"], wsum, gb, wk))
            for ev in X["events"]:
                print("       %s %s entity=%s → %s" % ev)
            # 비-ep write(apply/submit_referral 등) — floor에서 실행됐고 fs에서 없으면 표시
        # 비-ep write 대조(회귀 기전 분류)
        flnon = [n for n in A["tools"] if _is_action(n)]
        fsnon = [n for n in B["tools"] if _is_action(n)]
        missing = [n for n in set(flnon) if n not in set(fsnon)]
        if missing:
            print("   >> floor가 실행한 非-ep 액션 중 fs에 없음: %s" % sorted(set(missing)))
    print("\n" + "=" * 100)
    print("=== 집계 ===")
    print("E-PLAN dispute-write가 등장한 태스크: %d/43" % agg["ep_applicable"])
    print("dispute-write DENY (재구성): floor %d · fs %d" % (agg["fl_deny"], agg["fs_deny"]))
    print("CP5 walk FIRE (재구성): floor %d · fs %d" % (agg["fl_walkfire"], agg["fs_walkfire"]))
    print("gate POLICY 블록(궤적 committed): floor %d · fs %d" % (agg["fl_gate"], agg["fs_gate"]))
    print("회귀(floor pass→fs fail): %d %s" % (len(agg["regressions"]), agg["regressions"]))

_READ = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check|KB_)", re.I)
_PROC = re.compile(r"(^log_|_verification$|^kb_|^shell$|discoverable|transfer_to_human|^give_|^unlock_|get_current_time)", re.I)
def _is_action(n):
    f = _fam(n)
    return bool(f) and not _READ.match(f) and not _PROC.search(f)

if __name__ == "__main__":
    main()
