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


def turn_of(sim, msg_i):
    """메시지 index → 벤치가 찍은 `turn_idx`(우리 계기 로그의 `turn` 과 같은 축)."""
    ms = sim.get("messages") or []
    if msg_i is None or msg_i >= len(ms):
        return None
    return ms[msg_i].get("turn_idx")


def arrival(sim, name):
    """도구 이름이 **도구-결과로 도착한** 첫 메시지 index (KB 문서·unlock 목록 등). 없으면 None.

    이것이 knowing-doing 판정의 분모다 — 도착하지 않은 이름을 안 불렀다고 모델 결손이라 부르면
    우리 전달층의 결함을 모델에 떠넘기게 된다([[55]] 우리 배관 먼저)."""
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "tool":
            continue
        if name in str(m.get("content") or ""):
            return i
    return None


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


def value_arrived(sim, value, upto=None):
    """gold 값이 **도구-결과로 도착했는가** · 첫 도착 메시지 index. 없으면 None.

    WRONGARG 를 능력 결손으로 부르려면 먼저 이것을 봐야 한다 — 옳은 값이 궤적에 온 적이 없으면
    그것은 선택 결손이 아니라 **전달 결손**이다([[55]])."""
    v = str(value)
    if not v or len(v) < 2:
        return None
    for i, m in enumerate(sim.get("messages") or []):
        if upto is not None and i >= upto:
            break
        if m.get("role") != "tool":
            continue
        if v in str(m.get("content") or ""):
            return i
    return None


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
    """결손 변이 하나마다 **사다리**로 원인을 가른다 — 전달→지목→호출→인자.

    ladder(도달했나 → 이름을 말했나 → 불렀나 → 막혔나 → 인자가 맞나)는 각 칸이 궤적에서
    기계적으로 판정되고, 칸마다 처방 축이 다르다:
        DELIVERY-MISS      이름이 궤적에 **도착조차 안 했다**            → 검색·전달(우리 층)
        ARRIVED-NOT-NAMED  도착했는데 산문에도 안 올렸다                 → 선택
        NAMED-NOT-CALLED   이름을 말해 놓고 안 불렀다                    → 이행(knowing-doing)
        BLOCKED-ours/env   불렀는데 거절당했다                           → 게이트(우리)·환경
        WRONGARG           불렀는데 값이 다르다                          → operand
        DUP/EXTRA          gold 밖 변이·중복                             → 범위·중복
    """
    d = F.mutation_diff(sim)
    rows = []
    blk_by = {}
    for b in d["blocked"]:
        blk_by.setdefault(b["name"], []).append(b)

    def row(kind, sub, tool, msg_i, detail, extra=None):
        r = {"kind": kind, "sub": sub, "tool": tool, "msg_i": msg_i,
             "turn": turn_of(sim, msg_i), "detail": detail}
        if extra:
            r.update(extra)
        return r

    for g in d["missing"]:
        nm = g["name"]
        blk = blk_by.get(nm) or []
        same = [b for b in blk if b["key"] == g["key"]]
        arr = arrival(sim, nm)
        said = prose_first_mention(sim, nm)
        if same:
            b = same[0]
            rows.append(row("MISSING", "BLOCKED-" + (b["deny"] or "?"), nm, b["msg_i"],
                            (b["marker"] or "")[:90], {"gold_args": g["args"]}))
        elif blk:
            b = blk[0]
            rows.append(row("MISSING", "TRIED-OTHER-ARGS-BLOCKED-" + (b["deny"] or "?"), nm,
                            b["msg_i"], (b["marker"] or "")[:90], {"gold_args": g["args"]}))
        elif said is not None:
            rows.append(row("MISSING", "NAMED-NOT-CALLED", nm, said,
                            "도착 msg=%s · 지목 msg=%s" % (arr, said), {"gold_args": g["args"]}))
        elif arr is not None:
            rows.append(row("MISSING", "ARRIVED-NOT-NAMED", nm, arr,
                            "도착 msg=%s · 지목 없음" % arr, {"gold_args": g["args"]}))
        else:
            rows.append(row("MISSING", "DELIVERY-MISS", nm, last_assistant(sim),
                            "이름이 궤적에 한 번도 안 왔다", {"gold_args": g["args"]}))

    gold_by_name = {}
    for g in d["gold"]:
        gold_by_name.setdefault(g["name"], []).append(g)
    for w in d["wrongarg"]:
        diffs = []
        for g in (gold_by_name.get(w["name"]) or [])[:1]:
            for k in sorted(set(g["args"]) | set(w["args"])):
                gv, wv = g["args"].get(k), w["args"].get(k)
                if str(gv) != str(wv):
                    ga = value_arrived(sim, gv, w["msg_i"])
                    diffs.append("%s: gold=%s%s ours=%s [%s]"
                                 % (k, str(gv)[:40], "" if ga is None else "(도착msg%d)" % ga,
                                    str(wv)[:40], value_source(sim, w["msg_i"], wv)))
        rows.append(row("WRONGARG", "ARG-MISMATCH", w["name"], w["msg_i"], " · ".join(diffs) or "?"))
    for u in d.get("dup") or []:
        rows.append(row("DUP", "REPEATED-GOLD-CALL", u["name"], u["msg_i"],
                        json.dumps(u["args"], ensure_ascii=False)[:110]))
    for e in d["extra"]:
        rows.append(row("EXTRA", "NOT-IN-GOLD", e["name"], e["msg_i"],
                        json.dumps(e["args"], ensure_ascii=False)[:110]))
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
                            "transfer": F.transfer_msg_index(sim) is not None,
                            "gold_n": len(d["gold"]), "done_n": len(d["done"]),
                            "blocked_n": len(d["blocked"]), "rows": rows,
                            "trace_n": len(trace_by.get(F.simtag(sim)) or []),
                            "fb_n": len(fb_by.get(F.simtag(sim)) or [])}
                st = F.simtag(sim)
                trs = trace_by.get(st) or []
                fbs = fb_by.get(st) or []
                for r in rows:
                    tally[(r["kind"], r["sub"])] += 1
                    t = r.get("turn")
                    if t is None:
                        r["ours_marks"], r["ours_says"] = [], []
                        continue
                    r["ours_marks"] = sorted({x.get("mark") for x in trs
                                              if x.get("turn") in (t, t - 1) and x.get("mark")})
                    r["ours_says"] = sorted({x.get("channel") for x in fbs
                                             if x.get("turn") in (t, t - 1) and x.get("channel")})
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
        lab = "·".join("%s/%s@t%s" % (x["kind"][:4], x["sub"], x.get("turn")) for x in r["rows"]) or "clean"
        print("%-6s %-9s %-3s %-6s %-7s %s" % (r["run"], r["task"], r["trial"], r["reward"], b, lab[:88]))
    print("\n→ %s" % os.path.abspath(p))
    return 0


sys.exit(main())
