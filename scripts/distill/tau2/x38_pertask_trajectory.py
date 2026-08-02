#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x38: 실패 궤적을 **태스크별로 펼쳐** 분기점을 짚는다 (무료·로컬·영속 데이터).

사용자 지시(2026-08-02): *"per step 실패 궤적 포렌식을 좀 더 정밀하게 각 태스크 별로 분해하고 알려달라."*
x37은 기전 군집까지만 냈다. 여기서는 sim마다 **실제 호출 열·엔진 피드백·손님 발화·종결 행동**을
같이 놓고, "어디서 갈라졌는가"를 근거와 함께 적는다.

각 태스크마다 출력:
  ⑴ 채점 기준(`reward_basis`)과 gold 액션 목록(이름 + 핵심 인자)
  ⑵ **압축 호출 열**(연속 중복은 ×N으로 접음 · requestor 표시 · 오류 반환은 !로)
  ⑶ **미충족 gold**(x37의 의미 대조 재사용) — 무엇이 없었나 / 무엇이 달랐나
  ⑷ **서사↔행동 괴리**: 에이전트 본문이 gold 도구 이름을 말했는데 호출은 안 했나(027형)
  ⑸ 궤적 끝 5스텝 · 마지막 손님 발화 · 마지막 에이전트 본문
  ⑹ 그 sim에서 뜬 **엔진 표면화 마커** 계수(무엇을 이미 말해줬는가)

주의: 여기서 만드는 라벨은 **관찰 기술**이지 원인 단정이 아니다([[08]]). 원인은 근거를 읽고 사람이 쓴다.
"""
import argparse
import collections
import gzip
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--simdir", default=os.path.join(HERE, "..", "..", "..",
                                                 "reports", "facet_rft_2026", "sim_results"))
ap.add_argument("--tags", default="bank_qp32p1_gpu0_20260802,bank_qp32p1_gpu1_20260802")
ap.add_argument("--out", default="")
A = ap.parse_args()

MARKERS = ["[DUPLICATE-READ]", "[GROUNDING WARNING]", "[GUIDANCE]", "[coverage]", "[quote-pin]",
           "[T2_", "★FEEDBACK", "NOT_VERIFIED", "[UNAVAILABLE]"]
DISC_RE = re.compile(r"\b([a-z_]{4,}_\d{3,4})\b")


def norm(v):
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{":
            try:
                return norm(json.loads(s))
            except Exception:
                pass
        return " ".join(s.split()).lower()
    if isinstance(v, dict):
        return {str(k).lower(): norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, (int, float)):
        return float(v)
    return v


def seq_of(sim):
    byid = {m.get("id"): str(m.get("content") or "")
            for m in (sim.get("messages") or []) if (m.get("role") or "") == "tool"}
    out = []
    for m in (sim.get("messages") or []):
        role = m.get("role") or ""
        if role == "user" and (m.get("content") or "").strip():
            out.append({"k": "user", "text": str(m.get("content"))})
        if role in ("assistant", "user"):
            if role == "assistant" and (m.get("content") or "").strip():
                out.append({"k": "say", "text": str(m.get("content"))})
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                a = fn.get("arguments", tc.get("arguments"))
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {"__raw": a}
                out.append({"k": "call", "name": str(fn.get("name") or ""),
                            "args": a if isinstance(a, dict) else {},
                            "req": tc.get("requestor") or role,
                            "out": byid.get(tc.get("id"), "")})
    return out


def brief(args, n=3):
    """호출 인자를 짧게 — 도구 이름/식별자 같은 판별 인자를 우선."""
    if not args:
        return ""
    pref = [k for k in ("agent_tool_name", "user_tool_name", "tool_name", "transaction_id",
                        "user_id", "reason", "query", "new_rewards_earned") if k in args]
    keys = pref + [k for k in args if k not in pref]
    parts = []
    for k in keys[:n]:
        v = args.get(k)
        s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
        parts.append("%s=%s" % (k, s[:38]))
    return " ".join(parts)


def compress(seq):
    """연속 동일 (도구,인자) 를 ×N 으로 접는다."""
    out = []
    for e in seq:
        if e["k"] != "call":
            out.append(e)
            continue
        key = (e["name"], json.dumps(e["args"], sort_keys=True, ensure_ascii=False))
        if out and out[-1].get("k") == "call" and out[-1].get("_key") == key:
            out[-1]["_n"] += 1
            continue
        e = dict(e)
        e["_key"], e["_n"] = key, 1
        out.append(e)
    return out


SIMS = []
for tag in A.tags.split(","):
    p = os.path.join(A.simdir, tag.strip() + ".results.json.gz")
    if not os.path.exists(p):
        continue
    d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    for s in d.get("simulations", []):
        SIMS.append((tag.strip(), s))
fails = [(t, s) for t, s in SIMS
         if ((s.get("reward_info") or {}).get("reward") or 0) < 1]

buf = []


def P(x=""):
    buf.append(x)
    print(x)


P("# qp32p1 실패 궤적 — 태스크별 분해 (x38)")
P()
P("실패 %d / 전체 %d · 입력 = %s" % (len(fails), len(SIMS), A.tags))
P()
for tag, s in sorted(fails, key=lambda x: str(x[1].get("task_id"))):
    ri = s.get("reward_info") or {}
    acts = ri.get("action_checks") or []
    seq = seq_of(s)
    calls = [e for e in seq if e["k"] == "call"]
    csq = compress(seq)
    # 미충족 gold (x37과 같은 의미 대조)
    unmet = []
    used = set()
    for ac in acts:
        a = ac.get("action") or {}
        nm, want, req = a.get("name"), (a.get("arguments") or {}), a.get("requestor")
        keys = a.get("compare_args")
        keys = keys if keys is not None else list(want.keys())
        cand = [(i, c) for i, c in enumerate(calls)
                if (c["name"] == nm or c["name"].startswith(str(nm) + "_"))
                and (req is None or c["req"] == req)]
        hit = None
        for i, c in cand:
            if i in used:
                continue
            if all(norm(c["args"].get(k)) == norm(want.get(k)) for k in keys):
                hit, _ = c, used.add(i)
                break
        if hit is None:
            why = "호출 자체 없음" if not cand else "인자 불일치"
            d = ""
            if cand:
                c0 = cand[0][1]
                dd = ["%s: gold=%s / got=%s" % (k, json.dumps(want.get(k), ensure_ascii=False)[:46],
                                                json.dumps(c0["args"].get(k), ensure_ascii=False)[:46])
                      for k in keys if norm(c0["args"].get(k)) != norm(want.get(k))]
                d = " ; ".join(dd[:2])
            unmet.append((nm, brief(want, 2), why, d))
    # 서사↔행동 괴리
    said = set()
    for e in seq:
        if e["k"] == "say":
            said |= set(DISC_RE.findall(e["text"]))
    called = set()
    for c in calls:
        called.add(c["name"])
        for v in c["args"].values():
            if isinstance(v, str):
                called |= set(DISC_RE.findall(v))
    said_not_called = sorted(said - called)
    marks = collections.Counter()
    for c in calls:
        for m in MARKERS:
            if m in (c["out"] or ""):
                marks[m] += 1

    P("## %s  [%s]" % (s.get("task_id"), tag[-4:]))
    P("- 기준 `%s` %s · 종료 `%s` · 호출 %d · 메시지 %d · 소요 %ds"
      % (",".join(ri.get("reward_basis") or []) or "—", ri.get("reward_breakdown") or {},
         s.get("termination_reason"), len(calls), len(s.get("messages") or []),
         int(s.get("duration") or 0)))
    if acts:
        P("- gold 액션 %d: %s" % (len(acts), " | ".join(
            "%s(%s)" % ((x.get("action") or {}).get("name"), brief((x.get("action") or {}).get("arguments") or {}, 2))
            for x in acts[:6])))
    P("- **미충족 %d**:" % len(unmet))
    for nm, wa, why, d in unmet[:6]:
        P("    - `%s`(%s) — %s%s" % (nm, wa, why, (" · " + d) if d else ""))
    P("- 호출 열(압축): %s" % " → ".join(
        "%s%s%s" % (e["name"], "×%d" % e["_n"] if e["_n"] > 1 else "",
                    "" if e["req"] == "assistant" else "[u]")
        for e in csq if e["k"] == "call")[:900])
    if said_not_called:
        P("- ⚠**말했는데 안 부른 도구**: %s" % ", ".join(said_not_called[:6]))
    if marks:
        P("- 엔진 표면화: %s" % ", ".join("%s×%d" % (k, v) for k, v in marks.most_common(5)))
    lastu = [e["text"] for e in seq if e["k"] == "user"]
    lasta = [e["text"] for e in seq if e["k"] == "say"]
    if lastu:
        P("- 마지막 손님: %s" % " ".join(lastu[-1].split())[:220])
    if lasta:
        P("- 마지막 에이전트: %s" % " ".join(lasta[-1].split())[:220])
    P()

if A.out:
    open(A.out, "w", encoding="utf-8").write("\n".join(buf))
    print("→ %s" % A.out, file=sys.stderr)
