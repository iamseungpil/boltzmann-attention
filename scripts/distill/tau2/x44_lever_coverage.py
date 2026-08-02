#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x44: **레버 커버리지 감사** — 한 런에서 모든 레버의 turn-on 여부 × 발화/비발화를 표로.

사용자 지시(2026-08-02): *"다음 실험에서 모든 레버 발화 비발화 여부 확인하고 turn on 여부도 확인하라."*
배경 = wrap(`T2_FN_ISOLATE`)이 설계·구현 완료인 채 **한 번도 발화한 적 없던** 사고(x43).

방법(기계 산출·손 목록 금지 — x20 교훈):
  · ON/OFF = 발사 시 덤프된 env 파일(`env | grep ^T2_`)에서 읽는다(러너가 남김).
  · 태그→플래그 대응 = 코드 정적 스캔(태그 print 주변 ±40행의 environ.get — x43 휴리스틱).
  · 발화 = 이 런의 로그에서 태그 계수.
  판정: ON+발화=정상 / **ON+무발화=감사 대상**(트리거 부재 vs dark 구분은 사람이) /
       OFF+발화=**오염**(플래그 우회 경로) / OFF+무발화=정상.
  무태그 레버(print 없음)는 ON/OFF만 보고하고 '관측불가'로 정직하게 표기.

용법: py -3 x44_lever_coverage.py --env <env덤프…> --log <로그…> [--json out]
"""
import argparse
import collections
import glob
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
ap.add_argument("--env", nargs="+", required=True, help="env 덤프 파일(들)")
ap.add_argument("--log", nargs="+", required=True, help="런 로그 파일(들·.gz 가능)")
ap.add_argument("--json", default="")
A = ap.parse_args()

# ── 코드 정적 스캔: 플래그 전수 + 태그 대응 ──────────────────────────────────
ENGINE = [f for f in glob.glob(os.path.join(HERE, "t2_*.py")) +
          [os.path.join(HERE, "gate_interpreter.py")] if os.path.isfile(f)]
FLAG_RE = re.compile(r'environ(?:\.get)?\(\s*"(T2_[A-Z0-9_]+)"')
TAG_RE = re.compile(r'"(\[T2_[A-Z0-9_]+)')
all_flags, tag_flags = set(), collections.defaultdict(set)
flag_tags = collections.defaultdict(set)
for f in ENGINE:
    lines = open(f, encoding="utf-8").read().splitlines()
    for i, ln in enumerate(lines):
        for m in FLAG_RE.finditer(ln):
            all_flags.add(m.group(1))
        for m in TAG_RE.finditer(ln):
            tag = m.group(1) + "]"
            lo, hi = max(0, i - 40), min(len(lines), i + 5)
            for fm in FLAG_RE.finditer("\n".join(lines[lo:hi])):
                tag_flags[tag].add(fm.group(1))
                flag_tags[fm.group(1)].add(tag)
# [axis] 계열(축-레버 공용 태그)
for fl in ("T2_TOOL_CHANNEL", "T2_TERMINAL_TURN", "T2_FIT_DIFF", "T2_SCALAR_ARRAY"):
    flag_tags[fl].add("[T2_AXIS]")
flag_tags["T2_REPEAT_CAP"].add("[REPEAT-CAP")
flag_tags["T2_FN_ISOLATE"].add("[T2_FN_ISOLATE]")

# ── env 덤프 → ON/OFF ────────────────────────────────────────────────────────
envmap = {}
for p in A.env:
    for ln in open(p, encoding="utf-8", errors="replace"):
        if "=" in ln and ln.startswith("T2_"):
            k, v = ln.rstrip("\n").split("=", 1)
            envmap[k] = v
print("env 덤프 %d파일 · T2_* 설정 %d개 · 코드가 읽는 플래그 %d개"
      % (len(A.env), len(envmap), len(all_flags)))

# ── 로그 → 태그 계수 ─────────────────────────────────────────────────────────
tagcount = collections.Counter()
txt_all = []
for p in A.log:
    op = gzip.open if p.endswith(".gz") else open
    try:
        txt_all.append(op(p, "rt", encoding="utf-8", errors="replace").read())
    except Exception as e:
        print("⚠로그 읽기 실패 %s: %r" % (p, e))
TXT = "\n".join(txt_all)
alltags = set(tag_flags.keys()) | {t for ts in flag_tags.values() for t in ts}
for tag in alltags:
    n = TXT.count(tag)
    if n:
        tagcount[tag] = n

# ── 판정 표 ──────────────────────────────────────────────────────────────────
rows = []
for fl in sorted(all_flags):
    raw = envmap.get(fl)
    on = raw not in (None, "", "0")
    tags = sorted(flag_tags.get(fl) or [])
    fired = sum(tagcount.get(t, 0) for t in tags)
    if not tags:
        verdict = "관측불가(무태그)" if on else "OFF·무태그"
    elif on and fired:
        verdict = "정상(ON·발화)"
    elif on and not fired:
        verdict = "★ON인데 무발화 — 감사"
    elif not on and fired:
        verdict = "⚠OFF인데 발화 — 오염"
    else:
        verdict = "OFF"
    rows.append({"flag": fl, "value": raw, "on": on, "tags": tags,
                 "fired": fired, "verdict": verdict})

order = {"⚠OFF인데 발화 — 오염": 0, "★ON인데 무발화 — 감사": 1, "관측불가(무태그)": 2,
         "정상(ON·발화)": 3, "OFF": 4, "OFF·무태그": 5}
rows.sort(key=lambda r: (order.get(r["verdict"], 9), r["flag"]))
print("=" * 96)
print("%-30s %-8s %-8s %s" % ("flag", "설정값", "발화수", "판정"))
cur = None
for r in rows:
    if r["verdict"] != cur:
        cur = r["verdict"]
        print("── %s ──" % cur)
    print("%-30s %-8s %-8d %s" % (r["flag"], (r["value"] or "-")[:8], r["fired"],
                                  ",".join(r["tags"])[:44]))
n_bad = sum(1 for r in rows if r["verdict"].startswith(("★", "⚠")))
n_blind = sum(1 for r in rows if r["verdict"] == "관측불가(무태그)")
print("=" * 96)
print("요약: 감사 대상(ON·무발화) %d · 오염(OFF·발화) %d · 관측불가 %d / 전체 %d"
      % (sum(1 for r in rows if r["verdict"].startswith("★")),
         sum(1 for r in rows if r["verdict"].startswith("⚠")), n_blind, len(rows)))
if A.json:
    json.dump(rows, open(A.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s" % A.json)
sys.exit(2 if n_bad else 0)
