#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x45: **동등성 재생 하네스** v0 — 통합(거버너·버스)의 make-or-break 계측기.

정본 = `LEVER_CONSOLIDATION_DESIGN_2026_08_02` §5. 역할:
  ① 영속 궤적 전수에서 **엔진-저작 개입**(스텁·노트·deny·경고)을 순서 보존으로 추출 → **스냅샷**
  ② 스냅샷 두 개를 비교 — 수용 기준 = 불일치 0 or 전건이 화이트리스트(각 1줄 사유)
  통합 엔진은 같은 궤적 입력에서 **같은 지점·같은 문구·같은 순서**의 개입을 재생해야 교체 자격.

v0 범위(정직): 궤적에 **남은** 개입만 본다 — stderr-only 신호(beat 등)와 생성-레벨 regen의 산물은
  발화 지점까지만(§5 한계 그대로). 추출기 자체가 기준이므로 마커 목록은 이 파일이 정본이다.

용법:
  py -3 x45_equivalence.py snapshot --glob "bank_qp32*.results.json.gz" --out base.jsonl.gz
  py -3 x45_equivalence.py compare  --a base.jsonl.gz --b new.jsonl.gz [--whitelist wl.json]
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

# ── 개입 마커 정본 (궤적-가시 · 시작 토큰 → 종료 = 다음 마커/빈 줄/끝) ────────────
MARKERS = [
    "[DUPLICATE-READ]", "[NEAR-DUPLICATE-READ]", "[REPEAT-CAP]", "[GROUNDING WARNING]",
    "[coverage]", "[quote-pin]", "[axis]", "[GUIDANCE]", "[UNAVAILABLE]", "[TERMINAL]",
    "[BYREF]", "★FEEDBACK", "Error: [POLICY GATE", "Error: [UNKNOWN-TOOL REPEAT]",
    "Error: [PROVENANCE]", "[T2_TERM_GRANT", "[PLAN]", "[CHAIN]", "NOT_VERIFIED —",
]
_M_RE = re.compile("|".join(re.escape(m) for m in MARKERS))


def spans_of(text):
    """한 텍스트에서 (마커, 축자 스팬) 순서 보존 추출. 스팬 = 마커부터 다음 마커/이중개행/끝."""
    t = str(text or "")
    hits = [(m.start(), m.group(0)) for m in _M_RE.finditer(t)]
    out = []
    for i, (pos, mk) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(t)
        seg = t[pos:end]
        cut = seg.find("\n\n")
        if cut > 0:
            seg = seg[:cut]
        out.append((mk, seg.strip()))
    return out


def sim_snapshot(sim):
    """sim 하나 → 개입 레코드 목록(호출 순서·마커·스팬 해시·머리 120자)."""
    byid = {m.get("id"): str(m.get("content") or "")
            for m in (sim.get("messages") or []) if (m.get("role") or "") == "tool"}
    recs = []
    ci = 0
    for m in (sim.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            out = byid.get(tc.get("id"), "")
            for mk, seg in spans_of(out):
                recs.append({"ci": ci, "tool": str(fn.get("name") or ""), "marker": mk,
                             "h": hashlib.sha1(seg.encode("utf-8")).hexdigest()[:16],
                             "head": " ".join(seg.split())[:120]})
            ci += 1
    return recs


def cmd_snapshot(a):
    files = sorted(glob.glob(os.path.join(SIMDIR, a.glob)))
    n_sim = n_rec = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as w:
        for f in files:
            try:
                d = json.load(gzip.open(f, "rt", encoding="utf-8"))
            except Exception:
                continue
            tag = os.path.basename(f).replace(".results.json.gz", "")
            for s in d.get("simulations", []):
                recs = sim_snapshot(s)
                w.write(json.dumps({"tag": tag, "task": s.get("task_id"),
                                    "trial": s.get("trial"), "recs": recs},
                                   ensure_ascii=False) + "\n")
                n_sim += 1
                n_rec += len(recs)
    print("스냅샷: %d파일 → sim %d · 개입 레코드 %d → %s" % (len(files), n_sim, n_rec, a.out))
    mk = collections.Counter()
    for ln in gzip.open(a.out, "rt", encoding="utf-8"):
        for r in json.loads(ln)["recs"]:
            mk[r["marker"]] += 1
    for k, v in mk.most_common():
        print("  %-28s %6d" % (k, v))


def _load(p):
    out = {}
    for ln in gzip.open(p, "rt", encoding="utf-8"):
        d = json.loads(ln)
        out[(d["tag"], str(d["task"]), str(d.get("trial")))] = d["recs"]
    return out


def cmd_compare(a):
    A, B = _load(a.a), _load(a.b)
    wl = set()
    if a.whitelist and os.path.exists(a.whitelist):
        wl = {tuple(x[:2]) for x in json.load(open(a.whitelist, encoding="utf-8"))}
    keys = sorted(set(A) | set(B))
    n_ok = n_diff = 0
    diffs = collections.Counter()
    for k in keys:
        ra, rb = A.get(k, []), B.get(k, [])
        if [(r["ci"], r["marker"], r["h"]) for r in ra] == \
           [(r["ci"], r["marker"], r["h"]) for r in rb]:
            n_ok += 1
            continue
        sa = {(r["ci"], r["marker"], r["h"]) for r in ra}
        sb = {(r["ci"], r["marker"], r["h"]) for r in rb}
        onlyA, onlyB = sa - sb, sb - sa
        real = [(x, "A만") for x in onlyA if (x[1], "A") not in wl] + \
               [(x, "B만") for x in onlyB if (x[1], "B") not in wl]
        if not real:
            n_ok += 1
            continue
        n_diff += 1
        for (ci, mk, h), side in real[:4]:
            diffs[(mk, side)] += 1
        if n_diff <= 6:
            print("✗ %s/%s t%s — A만 %d · B만 %d" % (k[0][:28], k[1], k[2], len(onlyA), len(onlyB)))
    print("=" * 72)
    print("동등 %d / 불일치 %d  (수용 기준: 불일치 0 or 전건 화이트리스트)" % (n_ok, n_diff))
    for (mk, side), n in diffs.most_common(10):
        print("  %-28s %-4s %d" % (mk, side, n))
    sys.exit(1 if n_diff else 0)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s1 = sub.add_parser("snapshot")
s1.add_argument("--glob", default="*.results.json.gz")
s1.add_argument("--out", required=True)
s2 = sub.add_parser("compare")
s2.add_argument("--a", required=True)
s2.add_argument("--b", required=True)
s2.add_argument("--whitelist", default="")
a = ap.parse_args()
{"snapshot": cmd_snapshot, "compare": cmd_compare}[a.cmd](a)
