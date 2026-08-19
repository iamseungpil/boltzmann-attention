# -*- coding: utf-8 -*-
r"""x422 — **태스크별·스텝별 원인 확정** (사용자 지시 2026-08-20: *"계속 포렌식에서 정확한 원인을 못찾고 있다"*)

두 런(t7326 오염 · t7328 재베이스라인)의 80 sim 을 같은 자로 잰다.

## 자 (순서를 지킨다 · [[69]] 원인 확정 4단)
  ①채점 단위   reward · reward_basis
  ②변이 집합   gold ↔ 성공 = MISSING · WRONGARG · EXTRA · BLOCKED   (`t2_forensic.mutation_diff`)
  ③온셋 스텝   결손 변이마다 **궤적의 어느 자리에서 갈렸는가**
  ④우리 층     그 자리에서 우리 계기가 무엇을 말했나 (trace/fb 조인 · [[55]] 우리 배관 먼저)

## 온셋의 정의 (이것이 앞 포렌식과 다른 점)
앞선 포렌식들은 *"실패 시작지점"* 을 산문으로 골랐다. 여기서는 **결손 변이마다** 기계적으로 잡는다:
  · 그 도구를 **시도했나** → 시도했는데 막혔다면 온셋 = 그 호출 메시지, 원인 = 거절 주체(우리/환경)
  · 시도 안 했다면 → 그 도구 **이름을 산문에 올린 첫 메시지**(= 알고도 안 함 · knowing-doing)
                     없으면 → 마지막 어시스턴트 메시지(= 이름조차 안 나옴 · 미도달/미선택)
  · WRONGARG 는 그 호출 메시지가 온셋이고, 값의 **출처**를 궤적에서 되짚는다
    (앞선 도구-결과에 축자로 있었나 / 대화에 있었나 / 어디에도 없나 = 날조)

사용: py -3 x422_perstep_cause.py            # 요약표
      py -3 x422_perstep_cause.py task_073   # 태스크 한정 상세
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F  # noqa: E402

RUNS = [("t7326", ["bank_t7326_halfA_20260819q", "bank_t7326_halfB_20260819q"]),
        ("t7328", ["bank_t7328_halfA_20260819r", "bank_t7328_halfB_20260819r2"])]
SUF = ".results.json.gz"


def prose_first_mention(sim, name):
    """도구 이름을 **산문에** 처음 올린 어시스턴트 메시지 index (호출은 세지 않는다)."""
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        if name in str(m.get("content") or ""):
            return i
    return None


def last_assistant(sim):
    idx = None
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") == "assistant":
            idx = i
    return idx


def value_source(sim, upto, value):
    """값이 그 자리 **이전에** 어디서 왔는가 — 도구-결과 / 손님 발화 / 우리 층 / 없음(날조)."""
    v = str(value)
    if not v or v in ("True", "False", "None"):
        return "-"
    src = []
    for i, m in enumerate(sim.get("messages") or []):
        if upto is not None and i >= upto:
            break
        body = str(m.get("content") or "")
        if v not in body:
            continue
        r = m.get("role")
        src.append("tool" if r == "tool" else ("user" if r == "user" else "assistant"))
    if "tool" in src:
        return "tool-result"
    if "user" in src:
        return "user-said"
    if "assistant" in src:
        return "self-said"
    return "NOWHERE"


def onset_rows(sim):
    """결손 변이 하나마다 온셋 행을 만든다."""
    d = F.mutation_diff(sim)
    rows = []
    bykey = {}
    for b in d["blocked"]:
        bykey.setdefault(b["name"], []).append(b)
    for g in d["missing"]:
        blk = bykey.get(g["name"]) or []
        same = [b for b in blk if b["key"] == g["key"]]
        if same:
            b = same[0]
            rows.append({"kind": "MISSING", "sub": "BLOCKED-" + (b["deny"] or "?"),
                         "tool": g["name"], "onset": b["msg_i"], "detail": b["marker"],
                         "gold_args": g["args"]})
        elif blk:
            b = blk[0]
            rows.append({"kind": "MISSING", "sub": "BLOCKED-other-args-" + (b["deny"] or "?"),
                         "tool": g["name"], "onset": b["msg_i"], "detail": b["marker"],
                         "gold_args": g["args"]})
        else:
            said = prose_first_mention(sim, g["name"])
            if said is not None:
                rows.append({"kind": "MISSING", "sub": "NAMED-NOT-CALLED", "tool": g["name"],
                             "onset": said, "detail": "이름은 산문에 있었다", "gold_args": g["args"]})
            else:
                rows.append({"kind": "MISSING", "sub": "NEVER-NAMED", "tool": g["name"],
                             "onset": last_assistant(sim), "detail": "이름이 궤적에 없다",
                             "gold_args": g["args"]})
    gold_by_name = {}
    for g in d["gold"]:
        gold_by_name.setdefault(g["name"], []).append(g)
    for w in d["wrongarg"]:
        gs = gold_by_name.get(w["name"]) or []
        diffs = []
        for g in gs[:1]:
            keys = set(g["args"]) | set(w["args"])
            for k in sorted(keys):
                gv, wv = g["args"].get(k), w["args"].get(k)
                if str(gv) != str(wv):
                    diffs.append("%s: gold=%s ours=%s [%s]"
                                 % (k, gv, wv, value_source(sim, w["msg_i"], wv)))
        rows.append({"kind": "WRONGARG", "sub": "ARG-MISMATCH", "tool": w["name"],
                     "onset": w["msg_i"], "detail": " · ".join(diffs) or "?", "gold_args": None})
    for u in d.get("dup") or []:
        rows.append({"kind": "DUP", "sub": "REPEATED-GOLD-CALL", "tool": u["name"],
                     "onset": u["msg_i"], "detail": json.dumps(u["args"], ensure_ascii=False)[:110],
                     "gold_args": None})
    for e in d["extra"]:
        rows.append({"kind": "EXTRA", "sub": "NOT-IN-GOLD", "tool": e["name"],
                     "onset": e["msg_i"], "detail": json.dumps(e["args"], ensure_ascii=False)[:110],
                     "gold_args": None})
    return d, rows


def main():
    want = [a for a in sys.argv[1:] if a.startswith("task_")]
    out = {}
    tally = collections.Counter()
    for run, tags in RUNS:
        for tag in tags:
            tr = F.trace(tag)
            fb = F.sidecar_rows(tag)
            trace_by = collections.defaultdict(list)
            for r in tr:
                trace_by[r.get("sim")].append(r)
            fb_by = collections.defaultdict(list)
            for r in fb:
                fb_by[r.get("simtag") or r.get("sim")].append(r)
            for sim in F.sims(tag, SUF):
                t = F.task_id(sim)
                if want and t not in want:
                    continue
                ri = sim.get("reward_info") or {}
                rw = ri.get("reward")
                d, rows = onset_rows(sim)
                key = "%s|%s|t%s" % (run, t, sim.get("trial"))
                out[key] = {"run": run, "task": t, "trial": sim.get("trial"), "tag": tag,
                            "simtag": F.simtag(sim), "reward": rw,
                            "basis": ri.get("reward_basis"), "term": F.term_reason(sim),
                            "n_msgs": len(sim.get("messages") or []),
                            "gold_n": len(d["gold"]), "done_n": len(d["done"]),
                            "blocked_n": len(d["blocked"]), "rows": rows,
                            "trace_n": len(trace_by.get(F.simtag(sim)) or []),
                            "fb_n": len(fb_by.get(F.simtag(sim)) or [])}
                for r in rows:
                    tally[(r["kind"], r["sub"])] += 1
    p = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "x422_perstep_cause.json")
    with io.open(os.path.abspath(p), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("=" * 108)
    print("x422 · sim %d · 결손행 %d" % (len(out), sum(tally.values())))
    print("=" * 108)
    for (k, s), n in tally.most_common():
        print("  %-9s %-28s %3d" % (k, s, n))
    print("\n%-6s %-9s %-3s %-6s %-7s %s" % ("run", "task", "t", "reward", "basis", "결손"))
    for key in sorted(out):
        r = out[key]
        b = ",".join(r["basis"] or [])
        lab = "·".join("%s/%s" % (x["kind"][:4], x["sub"]) for x in r["rows"]) or "clean"
        print("%-6s %-9s %-3s %-6s %-7s %s" % (r["run"], r["task"], r["trial"], r["reward"], b, lab[:70]))
    print("\n→ %s" % os.path.abspath(p))
    return 0


sys.exit(main())
