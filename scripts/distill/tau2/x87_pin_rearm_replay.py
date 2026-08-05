# -*- coding: utf-8 -*-
"""After the procedure names a step, does the model take it — and would holding the pin help?

The pin is the strongest lever in this design: it narrows `tool_choice` to one value, which
is as close as the engine gets to choosing the model's next action. The case for making it
sticky rested on `task_051` — "the deny named the step and the model never called it" — and
that case collapsed: the deny was never delivered, because the regeneration loop's break
guard did not list `proc_fb` (F19). So the question has to be asked again on runs where the
message actually arrived.

  followed     the named tool is called within N assistant turns of the deny
  repeated     the same (tool, missing) deny recurs — the loop `task_048` spent eight
               rounds in, which one-shot pinning cannot end
  would_hold   turns the pin would have stayed armed had it been rearmed

A run from before F19 cannot answer this: its denies were printed, not sent. The script
says so rather than reporting a number that means nothing.

Free: reads persisted trajectories and logs. No model calls.

  usage: x87_pin_rearm_replay.py [tag_glob] [N]
"""

import collections
import glob
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_procedure as P  # noqa: E402
from x50_says_not_does import SIM  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "bank_smk_gpu*_20260805h"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
# F19(=deny 전달 배선) 이전 런은 이 질문에 답할 수 없다. 태그로 구분한다.
PRE_F19 = ("20260805a", "20260805b", "20260805c", "20260805d", "20260805e",
           "20260805f", "20260805g")


def a2_procedures(domain="banking_knowledge"):
    import gate_interpreter as GI
    return (GI.load_domain_a2(domain) or {}).get("procedures") or []


def args_of(tc):
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def exact_name(tc):
    nm = str(tc.get("name") or "")
    if nm.startswith("call_"):
        a = args_of(tc)
        inner = (a.get("agent_tool_name") or a.get("user_tool_name")
                 or a.get("discoverable_tool_name") or "")
        if inner:
            return str(inner)
    return nm


def errored(m):
    return bool(m.get("error")) or str(m.get("content") or "").lstrip().startswith("Error:")


def replay(sim, procs):
    """deny가 성립한 자리마다, 그 뒤 N턴 안에 지목된 도구가 불렸는지."""
    executed, pending, out = set(), {}, []
    open_denies = []                      # (turn_idx, target_tool, node_id)
    turn = 0
    for m in sim.get("messages") or []:
        if m.get("role") == "tool":
            nm = pending.pop(m.get("id") or m.get("tool_call_id"), None)
            if nm and not errored(m):
                executed.add(nm)
            continue
        for tc in (m.get("tool_calls") or []):
            pending[tc.get("id")] = exact_name(tc)
        if m.get("role") != "assistant":
            continue
        turn += 1
        called = {exact_name(tc) for tc in (m.get("tool_calls") or [])}
        for d in open_denies:
            if d["target"] in called:
                d["followed_in"] = min(d.get("followed_in", 99), turn - d["turn"])
        # 이 턴의 호출들이 deny를 유발했을까 — 엔진과 같은 술어로 판정
        for tc in (m.get("tool_calls") or []):
            dc = P.decide(procs, exact_name(tc), args_of(tc), executed)
            if dc.get("verdict") != "deny" or not dc.get("missing"):
                continue
            nid = dc["missing"][0]
            proc = P.find_procedure(procs, exact_name(tc), executed)
            node = next((n for n in ((proc or {}).get("nodes") or [])
                         if n.get("id") == nid), None)
            tools = P._tools_of(node) if node else []
            if len(tools) != 1:
                continue                  # 핀은 단일값일 때만 걸린다
            open_denies.append({"turn": turn, "target": tools[0], "node": nid,
                                "caller": exact_name(tc)})
    for d in open_denies:
        d["followed"] = d.get("followed_in", 99) <= N
        out.append(d)
    return out


procs = a2_procedures()
files = sorted(glob.glob(os.path.join(SIM, TAG + ".results.json.gz")))
if not files:
    raise SystemExit("no runs matched %s" % TAG)
stale = [f for f in files if any(t in f for t in PRE_F19)]
if stale:
    print("⚠ F19(deny 전달 배선) **이전** 런이 섞여 있다 — 그 런의 deny는 인쇄만 됐고")
    print("  모델에게 간 적이 없다. 아래 수치를 'pin이 필요하다'의 근거로 쓸 수 없다:")
    for f in stale:
        print("    %s" % os.path.basename(f))
    print()

tally = collections.Counter()
rep = collections.Counter()
rows = []
for p in files:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        ds = replay(s, procs)
        if not ds:
            continue
        sid = "%s/t%s" % (s.get("task_id"), s.get("trial"))
        for d2 in ds:
            tally["deny"] += 1
            tally["followed"] += 1 if d2["followed"] else 0
            rep[(sid, d2["node"], d2["target"])] += 1
        for key, c in rep.items():
            pass
        rows.append((sid, len(ds), sum(1 for x in ds if x["followed"])))

n = tally["deny"] or 1
worst = rep.most_common(6)
print("tag %s · N=%d(이행 판정 창)" % (TAG, N))
print("  deny 성립 지점 **%d**  ·  그 중 N턴 내 이행 **%d = %.1f%%**"
      % (tally["deny"], tally["followed"], 100.0 * tally["followed"] / n))
print("  같은 (sim,단계,도구)로 **반복된** deny 상위:")
for (sid, nid, tool), c in worst:
    print("    %-16s %-20s %-38s %d회%s" % (sid, nid, tool, c, "  ← 1회 핀으로는 못 끝낸다" if c >= 3 else ""))
print()
print("  sim별: %s" % ", ".join("%s %d/%d" % (a, c, b) for a, b, c in rows[:12]))
print()
print("  판정 규칙(사전등록): 이행률이 낮고 **반복 ≥3인 자리가 있으면** sticky를 켤 근거가 된다.")
print("                      이행률이 높으면 1회 핀으로 충분하므로 **켜지 않는다**([[10]] 최소 개입).")
