# -*- coding: utf-8 -*-
"""When we named the next tool and it was never called, what else were we saying at that moment?

`x96`/the smoke forensics established that twelve of sixteen never-called gold tools were
already on screen — retrieved text named them, and our own procedure and follow-up messages
named them again. The reading that follows is not "the model ignored us"; that reading has
been wrong every time it was checked ([[55]]). `task_048` shows why: three turns after the
procedure listed six steps by tool name, the engine also said *"resolve the flagged call(s)
first; do not call this tool yet"*. Both are ours.

So this rebuilds the turn axis per simulation and asks, for each named-but-uncalled tool,
what the same window contained:

  POINT_AT   an instruction naming that tool as the thing to do
  BLOCK      an instruction forbidding a call right now ("do not call this tool yet")
  REDIRECT   an instruction pointing the turn at something else (search, transfer, ask)

A window with POINT_AT and BLOCK/REDIRECT together is a contradiction the model actually
received — the same shape as the search/no-search pair that explained 048 before.

The marker table is declared here and printed, because we author the strings it classifies.

  usage: x101_named_uncalled_timeline.py <tag> [task ...]
"""

import collections
import glob
import hashlib
import io
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

TAG = sys.argv[1] if len(sys.argv) > 1 else "20260805n"
WANT = set(sys.argv[2:])
LOGD = os.environ.get("T2_LOGD", "/home/woori/scratch/logs")
SIMD = os.environ.get("T2_SIMD", "/home/woori/scratch/tau2-bench/data/simulations")

BLOCK = [r"do not call this tool yet", r"do NOT retry this tool now", r"do not proceed",
         r"Do NOT use placehold", r"resolve the flagged call", r"cannot be unlocked",
         r"was not provided by the user nor returned by any tool", r"Do NOT claim"]
REDIRECT = [r"search the knowledge base", r"KB_search", r"TRANSFER NOTICE",
            r"transfer_to_human_agents", r"Stop re-asking the customer", r"tell the customer",
            r"ask the user for", r"send the user exactly this message"]


def hits(text, pats):
    return [p for p in pats if re.search(p, text or "")]


def sims():
    out = []
    for f in sorted(glob.glob(os.path.join(SIMD, "bank_smk_gpu*_%s" % TAG, "results.json"))):
        out.extend(json.load(io.open(f, encoding="utf-8")).get("simulations") or [])
    return out


def fingerprint(sim):
    for m in sim.get("messages") or []:
        if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
            return hashlib.sha1(m["content"].strip().encode("utf-8")).hexdigest()[:12]
    return None


def sidecar():
    p = os.path.join(LOGD, "fb_%s.jsonl" % TAG)
    by = collections.defaultdict(list)
    if not os.path.exists(p):
        return by
    for line in io.open(p, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("text"):
            by[r.get("sim")].append(r)
    for v in by.values():
        v.sort(key=lambda r: r.get("turn") or 0)
    return by


def inner(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    a = a if isinstance(a, dict) else {}
    return (a.get("agent_tool_name") or a.get("discoverable_tool_name")
            or a.get("user_tool_name"))


SC = sidecar()
tally = collections.Counter()
print("표지 — BLOCK %d개 · REDIRECT %d개 (전부 우리가 쓴 문구)" % (len(BLOCK), len(REDIRECT)))
for s in sims():
    tid = s.get("task_id")
    if (WANT and tid not in WANT) or (s.get("reward_info") or {}).get("reward") == 1.0:
        continue
    fp = fingerprint(s)
    recs = SC.get(fp, [])
    if not recs:
        print("\n== %s — 사이드카 없음(분석 불가)" % tid)
        continue

    # ① 턴 축: 에이전트가 무엇을 불렀나 / 문맥에 어떤 이름이 떠 있었나
    called, first_seen, calls_at = collections.Counter(), {}, collections.defaultdict(list)
    gold = collections.Counter()
    for c in ((s.get("reward_info") or {}).get("action_checks") or []):
        a = c.get("action") or {}
        gold[inner(a.get("arguments")) or a.get("name")] += 1
    # ★시계 통일(2026-08-05 계기 교정): 사이드카의 `turn`은 **메시지 개수**(`len(messages)`)다.
    #   어시스턴트 턴을 세면 두 축이 2배쯤 어긋나 창(窓)이 통째로 밀린다.
    pend = {}
    for turn, m in enumerate(s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = inner(tc.get("arguments")) or tc.get("name")
            called[nm] += 1
            calls_at[turn].append(nm)
            pend[tc.get("id")] = nm
        if m.get("role") == "tool":
            c = str(m.get("content") or "")
            for g in gold:
                if g and g in c and g not in first_seen:
                    first_seen[g] = turn
    missing = [g for g in gold if not called.get(g)]

    print("\n" + "=" * 78)
    print("== %s reward=%s · gold %d종 · 미호출 %d종 · 지시 %d건"
          % (tid, (s.get("reward_info") or {}).get("reward"), len(gold), len(missing), len(recs)))

    # ② 미호출 도구별로 "이름이 뜬 시점 이후" 우리가 무엇을 말했나
    for g in missing:
        seen_at = first_seen.get(g)
        pt = [r for r in recs if g in str(r.get("text") or "")]
        if seen_at is None and not pt:
            tally["never_named"] += 1
            continue
        cands = [t for t in ([seen_at] + [r.get("turn") for r in pt]) if t is not None]
        start = min(cands) if cands else 0
        after = [r for r in recs if (r.get("turn") or 0) >= start]
        nb = sum(1 for r in after if hits(str(r.get("text") or ""), BLOCK))
        nr = sum(1 for r in after if hits(str(r.get("text") or ""), REDIRECT))
        tally["named"] += 1
        tally["named_with_block"] += 1 if nb else 0
        tally["named_with_redirect"] += 1 if nr else 0
        print("  ▸ %-44s 이름 등장 t%s · 이후 우리 지시 %d건 (BLOCK %d · REDIRECT %d)"
              % (g[:44], start, len(after), nb, nr))
        for r in after[:14]:
            t = " ".join(str(r.get("text") or "").split())
            mark = ("B" if hits(t, BLOCK) else " ") + ("R" if hits(t, REDIRECT) else " ") + \
                   ("N" if g in t else " ")
            print("      t%-3s [%s] %-12s %s" % (r.get("turn"), mark,
                                                 str(r.get("channel"))[:12], t[:150]))
        acted = [t for t in sorted(calls_at) if t >= start and g in calls_at[t]]
        print("      → 그 뒤 이 도구 호출: %s" % (acted or "없음"))

print("\n" + "=" * 78)
print("합계: %s" % dict(tally))
print("이름이 떠 있던 미호출 중 **이후에 BLOCK을 함께 받은 비율**: %d/%d"
      % (tally["named_with_block"], tally["named"] or 1))
print("⚠이 비율은 창이 길면 자동으로 1이 된다 — 아래 **같은 턴** 통계와 **호출된 도구 대조**로만 읽을 것.")

# ── 부정 통제: 같은 턴에서 지목과 금지가 함께 왔는가 · 결국 호출된 도구와 비교 ──────────
print("\n" + "=" * 78)
print("② 같은 턴 동시발생 (지목 N ∧ 금지 B) — 결국 호출된 gold 도구를 대조군으로")
ctl = collections.Counter()
rows = []
for s in sims():
    if (WANT and s.get("task_id") not in WANT) or (s.get("reward_info") or {}).get("reward") == 1.0:
        continue
    recs = SC.get(fingerprint(s), [])
    if not recs:
        continue
    by_turn = collections.defaultdict(list)
    for r in recs:
        by_turn[r.get("turn") or 0].append(str(r.get("text") or ""))
    called, calls_at, gold = collections.Counter(), collections.defaultdict(list), collections.Counter()
    for c in ((s.get("reward_info") or {}).get("action_checks") or []):
        a = c.get("action") or {}
        gold[inner(a.get("arguments")) or a.get("name")] += 1
    for turn, m in enumerate(s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = inner(tc.get("arguments")) or tc.get("name")
            called[nm] += 1
            calls_at[nm].append(turn)
    for g in gold:
        point_turns = [t for t, txts in by_turn.items() if any(g in x for x in txts)]
        if not point_turns:
            continue
        # 호출된 도구는 **첫 호출 전까지의** 지목만 센다(사후 문구는 원인이 될 수 없다)
        cutoff = min(calls_at[g]) if called.get(g) else 10 ** 6
        pts = [t for t in point_turns if t <= cutoff]
        if not pts:
            continue
        both = sum(1 for t in pts if any(hits(x, BLOCK) for x in by_turn[t]))
        red = sum(1 for t in pts if any(hits(x, REDIRECT) for x in by_turn[t]))
        grp = "호출됨" if called.get(g) else "미호출"
        ctl[grp + "_도구"] += 1
        ctl[grp + "_지목턴"] += len(pts)
        ctl[grp + "_지목∧금지턴"] += both
        ctl[grp + "_지목∧전환턴"] += red
        rows.append((grp, s.get("task_id"), g[:40], len(pts), both, red))
for grp in ("미호출", "호출됨"):
    n, tt = ctl[grp + "_도구"], ctl[grp + "_지목턴"]
    print("  %-5s 도구 %2d종 · 지목된 턴 %3d · 그중 같은 턴에 금지 %3d (%.0f%%) · 전환 %3d (%.0f%%)"
          % (grp, n, tt, ctl[grp + "_지목∧금지턴"], 100.0 * ctl[grp + "_지목∧금지턴"] / (tt or 1),
             ctl[grp + "_지목∧전환턴"], 100.0 * ctl[grp + "_지목∧전환턴"] / (tt or 1)))
for r in sorted(rows):
    print("     %-5s %-10s %-40s 지목턴%3d 금지%3d 전환%3d" % r)
print("  ⇒ 호출된 도구도 같은 턴에 금지를 받는다 — **동시발생만으로는 원인이 아니다**(대조군 7건 중")
print("     6건이 `log_verification`이라 동질 비교도 아니다). 아래 ③이 실제 구분자다.")

# ── ③ 요구 대상이 턴마다 바뀌는가 — 통과 sim을 대조군으로 ────────────────────────────
print("\n" + "=" * 78)
print("③ 우리가 **한 번에 하나를 요구하는가** (턴별 지목 대상의 이동)")
TOOLS = set()
tp = os.environ.get("T2_TOOLS_PY",
                    "/home/woori/scratch/tau2-bench/src/tau2/domains/banking_knowledge/tools.py")
if os.path.exists(tp):
    import ast
    for node in ast.walk(ast.parse(io.open(tp, encoding="utf-8").read())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            TOOLS.add(node.name)
print("  도구 이름 사전 %d개(환경 소스에서 추출)" % len(TOOLS))
print("  %-10s %-6s %-6s %-7s %-7s %s" % ("task", "결과", "지목턴", "대상수", "전환", "최장연속"))
for s in sims():
    recs = SC.get(fingerprint(s), [])
    if not recs:
        continue
    rew = (s.get("reward_info") or {}).get("reward")
    by_turn = collections.defaultdict(set)
    for r in recs:
        t = str(r.get("text") or "")
        for nm in TOOLS:
            if nm in t:
                by_turn[r.get("turn") or 0].add(nm)
    seq = [(t, by_turn[t]) for t in sorted(by_turn) if by_turn[t]]
    if not seq:
        continue
    switches = sum(1 for i in range(1, len(seq)) if seq[i][1] != seq[i - 1][1])
    runs, best = collections.Counter(), 0
    for nm in {x for _, v in seq for x in v}:
        cur = 0
        for _, v in seq:
            cur = cur + 1 if nm in v else 0
            runs[nm] = max(runs[nm], cur)
    best = max(runs.values()) if runs else 0
    print("  %-10s %-6s %-6d %-7d %-7d %d (%s)"
          % (s.get("task_id"), "PASS" if rew == 1.0 else "fail", len(seq),
             len({x for _, v in seq for x in v}), switches, best,
             max(runs, key=runs.get)[:28] if runs else "-"))

# ── ④ **언제** 지목했나 — 지목 시점의 위치(대화 길이 대비)와 남은 여유 ────────────────
print()
print("=" * 78)
print("④ 지목 시점 (사이드카 turn = 메시지 개수 기준)")
pos = {"미호출": [], "호출됨": []}
late = collections.Counter()
for s in sims():
    recs = SC.get(fingerprint(s), [])
    if not recs or (WANT and s.get("task_id") not in WANT):
        continue
    n = len(s.get("messages") or [])
    called, calls_at, gold = collections.Counter(), collections.defaultdict(list), collections.Counter()
    for c in ((s.get("reward_info") or {}).get("action_checks") or []):
        a = c.get("action") or {}
        gold[inner(a.get("arguments")) or a.get("name")] += 1
    for turn, m in enumerate(s.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = inner(tc.get("arguments")) or tc.get("name")
            called[nm] += 1
            calls_at[nm].append(turn)
    for g in gold:
        pts = sorted(r.get("turn") or 0 for r in recs if g in str(r.get("text") or ""))
        if not pts:
            continue
        grp = "호출됨" if called.get(g) else "미호출"
        first = pts[0]
        pos[grp].append((first / float(n or 1), n - first, s.get("task_id"), g))
        if first / float(n or 1) > 0.5:
            late[grp] += 1
for grp in ("미호출", "호출됨"):
    v = pos[grp]
    if not v:
        continue
    frac = sorted(x[0] for x in v)
    rem = sorted(x[1] for x in v)
    print("  %-5s n=%2d · 첫 지목 위치 중앙값 %.2f (대화의 몇 %% 지점) · 남은 메시지 중앙값 %d · 후반부(>50%%) %d건"
          % (grp, len(v), frac[len(frac) // 2], rem[len(rem) // 2], late[grp]))
print("  ── 늦게 지목된 것들(위치 > 0.5) ──")
for grp in ("미호출", "호출됨"):
    for f, r, tid, g in sorted(pos[grp], reverse=True):
        if f > 0.5:
            print("     %-5s %-10s %-42s 위치 %.2f · 남은 메시지 %d" % (grp, tid, g[:42], f, r))
