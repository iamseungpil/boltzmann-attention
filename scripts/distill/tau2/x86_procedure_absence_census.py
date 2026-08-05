# -*- coding: utf-8 -*-
"""If the engine spoke when the procedure went quiet, how often would it point somewhere wrong?

A procedure the agent entered and then stopped walking is the shape behind the closure
cluster: `task_048` entered, called nothing from the declaration for thirteen turns, and
transferred. A checklist that says where the walk stopped is only worth wiring if it
would point at steps the task actually needed — a gate that names steps gold never asked
for buys completion by selling scope, which is the trade the first principle says every
lever makes ([[19]], lighthouse §1.3).

So this replays the persisted trajectories against the same declaration and the same
pure functions the engine would use — `active_procedures`, `checklist`, `next_step` —
and counts four things:

  fire        turns where the absence condition holds (active ∧ unmet ∧ K quiet turns)
  unique      of those, how often exactly one node is ready — the only case the design
              lets the engine point at ("▶"). Ties must be listed, never chosen ([[10]])
  in-gold     whether the tool the checklist would name is one gold's own actions call
  locked      whether it is a discoverable tool never unlocked in that run — the case
              where naming the tool is not enough and the unlock has to be said

Free: reads persisted trajectories and the declaration. No model calls.

  usage: x86_procedure_absence_census.py [arm] [K] [domain]
"""

import collections
import glob
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_procedure as P  # noqa: E402
from x50_says_not_does import ARMS, SIM  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "N97B"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DOMAIN = sys.argv[3] if len(sys.argv) > 3 else "banking_knowledge"
CAP = int(os.environ.get("X86_CAP", "2"))          # 설계의 sim당 상한과 같게 둔다
UNLOCK = "unlock_discoverable_agent_tool"


def a2_procedures(domain):
    import gate_interpreter as GI
    return (GI.load_domain_a2(domain) or {}).get("procedures") or []


def write_tools(domain):
    """env가 WRITE로 표시한 도구 이름 — gold-밖 지목의 **대가**를 가르는 축.

    gold이 요구하지 않은 read를 하나 더 부르는 것은 토큰을 쓰고, write를 하나 더 부르는 것은
    DB를 바꾼다. 후자만이 등대 §1.3이 말하는 over-action이다. env를 못 읽으면 빈 집합을
    돌려주고 그 줄은 계량에서 빠진다(추측하지 않는다).
    """
    try:
        from tau2.environment.toolkit import TOOL_TYPE_ATTR, ToolType
        mod = __import__("tau2.domains.%s.tools" % domain, fromlist=["*"])
    except Exception:
        return None
    out = set()
    for cls_name in dir(mod):
        cls = getattr(mod, cls_name)
        if not isinstance(cls, type):
            continue
        for n in dir(cls):
            if getattr(getattr(cls, n, None), TOOL_TYPE_ATTR, None) == ToolType.WRITE:
                out.add(n)
    return out


def args_of(tc):
    a = tc.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def exact_name(tc):
    """엔진 `_exact_tool_name`의 거울: 디스패처는 인자에 실린 실제 이름으로 센다."""
    nm = str(tc.get("name") or "")
    if nm.startswith("call_"):
        a = args_of(tc)
        inner = (a.get("agent_tool_name") or a.get("user_tool_name")
                 or a.get("discoverable_tool_name") or "")
        if inner:
            return str(inner)
    return nm


def errored(m):
    txt = str(m.get("content") or "").lstrip()
    return bool(m.get("error")) or txt.startswith("Error:")


def gold_tools(sim):
    out = set()
    for c in (sim.get("reward_info") or {}).get("action_checks") or []:
        act = c.get("action") or {}
        out.add(exact_name(act) if act.get("name") else "")
        a = act.get("arguments") if isinstance(act.get("arguments"), dict) else {}
        for k in ("agent_tool_name", "user_tool_name", "discoverable_tool_name"):
            if a.get(k):
                out.add(str(a[k]))
    return {x for x in out if x}


def walk(sim, procs):
    """(fire, unique, in_gold, locked, detail) — 한 sim에서 부재 트리거가 섰을 자리들."""
    gold = gold_tools(sim)
    executed, unlocked, pending = set(), set(), {}
    quiet = collections.Counter()          # proc id → 연속 조용한 assistant 턴 수
    fires, fired = [], 0
    for m in sim.get("messages") or []:
        role = m.get("role")
        tcs = m.get("tool_calls") or []
        if role == "tool":
            # 엔진과 같은 규약: 에러로 돌아온 호출은 그 단계를 수행하지 않았다.
            nm = pending.pop(m.get("id") or m.get("tool_call_id"), None)
            if nm and not errored(m):
                executed.add(nm)
            continue
        for tc in tcs:
            pending[tc.get("id")] = exact_name(tc)
            if tc.get("name") == UNLOCK:
                # unlock은 성사 여부를 따로 보지 않는다 — 048은 unlock 자체는 성공시켰고
                # **다른 도구를** 풀었다. 여기서 재는 것은 "이 이름이 풀린 적 있나"다.
                unlocked.add(str(args_of(tc).get("agent_tool_name") or ""))
        if role != "assistant":
            continue

        called = {exact_name(tc) for tc in tcs}
        for p in P.active_procedures(procs, executed):
            pid = p.get("id")
            nodes = {t for n in (p.get("nodes") or []) for t in (P._tools_of(n) or [])}
            if called & nodes:
                quiet[pid] = 0
                continue
            quiet[pid] += 1
            if quiet[pid] < K or fired >= CAP:
                continue
            cands, uniq = P.next_step(p, executed)
            if not cands:
                continue
            quiet[pid] = 0                  # 발화하면 창을 다시 연다(연속 도배 방지)
            fired += 1
            tools = [t for n in cands for t in (P._tools_of(n) or [])]
            fires.append({
                "procedure": pid,
                "unique": uniq,
                "nodes": [n.get("id") for n in cands],
                "tools": tools,
                "in_gold": bool(set(tools) & gold),
                "locked": [t for t in tools if t not in unlocked],
                "done": sum(1 for _, _, d in P.checklist(p, executed) if d is True),
                "total": len(p.get("nodes") or []),
            })
    return fires


procs = a2_procedures(DOMAIN)
WRITES = write_tools(DOMAIN)
if not procs:
    raise SystemExit("no procedures declared for %s" % DOMAIN)
print("선언 절차 %d종: %s\n" % (len(procs), ", ".join(p.get("id") for p in procs)))

tally = collections.Counter()
by_proc = collections.Counter()
sims_fired, examples = set(), []
files = sorted(glob.glob(os.path.join(SIM, ARMS[ARM] + "*.results.json.gz")))
if not files:
    raise SystemExit("no runs matched arm %s" % ARM)
for p in files:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    for s in (d.get("simulations") if isinstance(d, dict) else d):
        tally["sim"] += 1
        fs = walk(s, procs)
        if not fs:
            continue
        sims_fired.add((s.get("task_id"), s.get("trial")))
        for fr in fs:
            tally["fire"] += 1
            by_proc[fr["procedure"]] += 1
            tally["unique"] += 1 if fr["unique"] else 0
            tally["in_gold"] += 1 if fr["in_gold"] else 0
            tally["locked"] += 1 if fr["locked"] else 0
            if not fr["in_gold"] and WRITES is not None:
                tally["outside_write" if (set(fr["tools"]) & WRITES) else "outside_read"] += 1
            if len(examples) < 12:
                examples.append(((s.get("task_id"), s.get("trial")), fr))

n = tally["fire"] or 1
print("arm %s · K=%d · sim당 상한 %d · sim %d" % (ARM, K, CAP, tally["sim"]))
print("  발화 **%d회** / **%d sim**(%.0f%%)"
      % (tally["fire"], len(sims_fired), 100.0 * len(sims_fired) / max(tally["sim"], 1)))
print("  절차별: %s" % ", ".join("%s %d" % kv for kv in by_proc.most_common()))
print()
print("  ▶NEXT가 **유일**했던 비율      %3d/%3d = %5.1f%%   (동렬이면 목록만·[[10]])"
      % (tally["unique"], tally["fire"], 100.0 * tally["unique"] / n))
print("  지목 도구가 **gold에 있음**    %3d/%3d = %5.1f%%   ← 높을수록 Δover-action 낮다"
      % (tally["in_gold"], tally["fire"], 100.0 * tally["in_gold"] / n))
print("  지목 도구가 **미-unlock**      %3d/%3d = %5.1f%%   ← unlock 힌트가 필요한 몫"
      % (tally["locked"], tally["fire"], 100.0 * tally["locked"] / n))
print()
print("  사전등록 게이트: gold-밖 지목 = %.1f%% (30%% 초과면 K 상향 후 재계량)"
      % (100.0 * (n - tally["in_gold"]) / n))
if WRITES is None:
    print("  ⚠env 미탑재 — gold-밖의 read/write 분해 불가(리모트에서 재실행할 것)")
else:
    print("  ★gold-밖의 성질: **write %d** · read %d  — over-action은 write 쪽만이다"
          % (tally["outside_write"], tally["outside_read"]))
print("\n  예시:")
for sid, fr in examples:
    print("    %-18s %-30s %d/%d done · next=%s%s%s"
          % ("%s/t%s" % sid, fr["procedure"], fr["done"], fr["total"],
             ",".join(fr["nodes"]), "" if fr["unique"] else " (동렬)",
             "" if fr["in_gold"] else "  ⚠gold-밖"))
