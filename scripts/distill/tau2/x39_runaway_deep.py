#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x39: qp32p1 폭주 3건(022·026·027) **문구 단위** 정밀 분석 (무료·로컬).

사용자 지시(2026-08-02): *"폭주 3건 다시 정밀 분석하라. 각 문구 다 살펴보라."*

x37/x38은 기전 라벨까지만 냈다. 여기서는 **에이전트가 반복 직전에 실제로 읽은 텍스트**를 전부 펼친다:
  ⑴ 루프 진입점 탐지(꼬리 주기) → 진입 전후 창을 **전문**으로 출력
  ⑵ 그 sim에 등장한 **엔진/도구 문구를 축자로 유일화**(어떤 지시가 몇 번 떴는가)
  ⑶ 에이전트 본문(narration) ↔ 실제 호출 대조 — 말한 도구를 불렀는가
  ⑷ 문맥 팽창 회계 — 어떤 메시지가 창을 먹었나(022형 ctx 초과용)
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
ap.add_argument("--tasks", default="task_022,task_026,task_027")
ap.add_argument("--win", type=int, default=6, help="루프 진입 전후 출력 창")
ap.add_argument("--full", type=int, default=1200, help="문구 전문 출력 상한 문자")
A = ap.parse_args()

DISC_RE = re.compile(r"\b([a-z_]{4,}_\d{3,4})\b")


def load_sims():
    out = {}
    for tag in A.tags.split(","):
        p = os.path.join(A.simdir, tag.strip() + ".results.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt", encoding="utf-8"))
        for s in d.get("simulations", []):
            out[str(s.get("task_id"))] = s
    return out


def steps_of(sim):
    byid = {m.get("id"): str(m.get("content") or "")
            for m in (sim.get("messages") or []) if (m.get("role") or "") == "tool"}
    out = []
    for m in (sim.get("messages") or []):
        role = m.get("role") or ""
        if role == "tool":
            continue
        txt = str(m.get("content") or "")
        if role == "user" and txt.strip():
            out.append({"k": "USER", "text": txt})
        if role == "assistant" and txt.strip():
            out.append({"k": "SAY", "text": txt})
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
            out.append({"k": "CALL", "name": str(fn.get("name") or ""),
                        "args": a if isinstance(a, dict) else {},
                        "req": tc.get("requestor") or role,
                        "out": byid.get(tc.get("id"), "")})
    return out


SIMS = load_sims()
for task in A.tasks.split(","):
    task = task.strip()
    sim = SIMS.get(task)
    if sim is None:
        print("⚠ %s 없음" % task)
        continue
    st = steps_of(sim)
    calls = [i for i, e in enumerate(st) if e["k"] == "CALL"]
    print("\n" + "=" * 100)
    print("# %s — 종료 %s · 메시지 %d · 호출 %d · %ds"
          % (task, sim.get("termination_reason"), len(sim.get("messages") or []),
             len(calls), int(sim.get("duration") or 0)))

    # ── ⑵ 문구 카탈로그 (도구 반환 텍스트를 선두 80자로 유일화) ────────────────
    cat = collections.Counter()
    first = {}
    for i in calls:
        o = st[i]["out"] or ""
        key = " ".join(o.split())[:80]
        if not key:
            continue
        cat[key] += 1
        first.setdefault(key, o)
    print("\n## 도구 반환 문구 (유일화 · 상위 8)")
    for k, n in cat.most_common(8):
        print("  ×%-3d %s" % (n, k))

    # ── ⑶ 서사 ↔ 행동 ─────────────────────────────────────────────────────────
    said, called = collections.Counter(), set()
    for e in st:
        if e["k"] == "SAY":
            for t in DISC_RE.findall(e["text"]):
                said[t] += 1
        if e["k"] == "CALL":
            called.add(e["name"])
            for v in e["args"].values():
                if isinstance(v, str):
                    called |= set(DISC_RE.findall(v))
    gap = {t: n for t, n in said.items() if t not in called}
    if gap:
        print("\n## ⚠서사↔행동 괴리 — 본문에서 지목했으나 **한 번도 안 부른** 도구")
        for t, n in sorted(gap.items(), key=lambda x: -x[1]):
            print("  %s — 본문 언급 %d회" % (t, n))

    # ── ⑴ 루프 진입점 + 창 전문 ───────────────────────────────────────────────
    names = [st[i]["name"] for i in calls]
    keys = [(st[i]["name"], json.dumps(st[i]["args"], sort_keys=True, ensure_ascii=False))
            for i in calls]
    onset = None
    for j in range(len(keys)):
        if keys[j:j + 3].count(keys[j]) == 3 and len(keys[j:j + 3]) == 3:
            onset = j
            break
    if onset is None:
        rep = collections.Counter(keys)
        if rep and rep.most_common(1)[0][1] >= 3:
            k0 = rep.most_common(1)[0][0]
            onset = keys.index(k0)
    if onset is not None:
        lo = max(0, calls[onset] - A.win)
        hi = min(len(st), calls[min(onset + A.win, len(calls) - 1)] + 1)
        print("\n## 루프 진입 창 (호출 #%d 부근 · 전문)" % onset)
        for e in st[lo:hi]:
            if e["k"] == "CALL":
                print("\n  ▶ CALL %s [%s] args=%s"
                      % (e["name"], e["req"],
                         json.dumps(e["args"], ensure_ascii=False)[:200]))
                print("    ← %s" % " ".join((e["out"] or "").split())[:A.full])
            else:
                print("\n  ◆ %s: %s" % (e["k"], " ".join(e["text"].split())[:A.full]))
    else:
        print("\n## 루프 진입점 미검출(정확 반복 3회 없음)")

    # ── ⑷ 문맥 팽창 회계 ─────────────────────────────────────────────────────
    print("\n## 문맥 팽창 회계 (문자 수 기준 상위 기여)")
    contrib = collections.Counter()
    for e in st:
        if e["k"] == "CALL":
            contrib["tool:" + e["name"]] += len(e["out"] or "")
        else:
            contrib[e["k"].lower()] += len(e["text"])
    total = sum(contrib.values())
    print("  총 %d자" % total)
    for k, v in contrib.most_common(7):
        print("   %-46s %8d  (%4.1f%%)" % (k, v, 100.0 * v / max(total, 1)))
