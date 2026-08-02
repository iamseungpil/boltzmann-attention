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

# ── 코드 정적 스캔: 플래그 전수 + **증거 등급별** 태그 대응 (r2 정제·2026-08-02) ──
# ★r1 결함(첫 라이브 감사가 스스로 적발): ±40행 휴리스틱이 형제 플래그를 공용 태그에 묶어
#   "OFF인데 발화" 오탐 19건(T2_DECLFIRST_ENFORCE=0인데 [T2_DECLFIRST] 발화 등)을 냈고,
#   파라미터형(*_CAP·*_MIN 등)을 on/off처럼 판정했다. r2 = 증거 3등급 분리:
#   ①정확(beat `[T2_LEVER] <플래그>` · 태그명==플래그명 · 검증된 소수 쌍) → ★/⚠ 판정 자격
#   ②공유(휴리스틱 근접 태그) → 참고만 · 오염/무발화 판정 금지(귀속 불가)
#   ③파라미터형(`== "1"` 비교 부재) → 값 보고만(발화 개념 없음)
ENGINE = [f for f in glob.glob(os.path.join(HERE, "t2_*.py")) +
          [os.path.join(HERE, "gate_interpreter.py")] if os.path.isfile(f)]
FLAG_RE = re.compile(r'environ(?:\.get)?\(\s*"(T2_[A-Z0-9_]+)"')
ONOFF_RE = re.compile(r'environ(?:\.get)?\(\s*"(T2_[A-Z0-9_]+)"[^)]*\)\s*==\s*"1"')
TAG_RE = re.compile(r'"(\[T2_[A-Z0-9_]+)')
all_flags, onoff_flags = set(), set()
exact_tags = collections.defaultdict(set)
shared_tags = collections.defaultdict(set)
for f in ENGINE:
    src = open(f, encoding="utf-8").read()
    lines_ = src.splitlines()
    for m in FLAG_RE.finditer(src):
        all_flags.add(m.group(1))
    for m in ONOFF_RE.finditer(src):
        onoff_flags.add(m.group(1))
    for i, ln in enumerate(lines_):
        for m in TAG_RE.finditer(ln):
            tag = m.group(1) + "]"
            lo, hi = max(0, i - 40), min(len(lines_), i + 5)
            for fm in FLAG_RE.finditer("\n".join(lines_[lo:hi])):
                shared_tags[fm.group(1)].add(tag)
for fl in list(all_flags):
    exact_tags[fl].add("[T2_LEVER] " + fl)          # beat(정확명)
    nm = "[" + fl + "]"
    exact_tags[fl].add(nm)                          # 태그명==플래그명
    shared_tags[fl].discard(nm)
_CURATED = {  # 이름은 다르지만 코드 정독으로 1:1 확정한 쌍만(늘릴 땐 반드시 코드 확인·grep 축자)
    "T2_TOOL_CHANNEL": {"[T2_AXIS]"}, "T2_TERMINAL_TURN": {"[T2_AXIS]"},
    "T2_FIT_DIFF": {"[T2_AXIS]"}, "T2_SCALAR_ARRAY": {"[T2_AXIS]"},
    "T2_REPEAT_CAP": {"[REPEAT-CAP"},
    # 이름-변형 쌍(2026-08-02 grep 확정): 플래그와 태그의 언더스코어/표기 차이
    "T2_CLAIM_PROV": {"[T2_CLAIMPROV]"},
    "T2_WRITE_PROV": {"[T2_WRITEPROV]"},
    "T2_QUOTE_PIN": {"[T2_SG_ISOLATE] quote-pin"},   # 판정 발화(비-pass시)·pass만이면 무발화가 정상
    "T2_SG_GROUND": {"[GROUNDING WARNING]"},
    "T2_GROUND_HDR": {"[GROUNDING WARNING]"},        # 트리거 공유(발화=경고 자체·HDR는 문구 형태)
    "T2_TOOL_SIGNATURE_OBSERVE": {"[T2_TOOL_SIGNATURE]"},  # observe 모드가 본체 태그로 발화
    "T2_ARG_SCHEMA": {"[T2_ARGSCHEMA]"},                   # 언더스코어 변형(dark 감사서 확정)
}
for fl, ts in _CURATED.items():
    exact_tags[fl] |= ts
# ⚠[T2_AXIS]는 4레버 공용 — 발화>0은 존재 증명 수준이고 어느 레버인지는 로그 본문으로(정직 표기).

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
alltags = ({t for ts in exact_tags.values() for t in ts}
           | {t for ts in shared_tags.values() for t in ts})
for tag in alltags:
    n = TXT.count(tag)
    if n:
        tagcount[tag] = n

# ── 판정 표 ──────────────────────────────────────────────────────────────────
rows = []
for fl in sorted(all_flags):
    raw = envmap.get(fl)
    on = raw not in (None, "", "0")
    ex = sorted(exact_tags.get(fl) or [])
    sh = sorted((shared_tags.get(fl) or set()) - set(ex))
    ex_fired = sum(tagcount.get(t, 0) for t in ex)
    sh_fired = sum(tagcount.get(t, 0) for t in sh)
    if fl not in onoff_flags:
        verdict = "파라미터(값 보고만)" if on else "파라미터·미설정"
    elif on and ex_fired:
        verdict = "정상(ON·발화·정확증거)"
    elif on and sh_fired:
        verdict = "정상?(ON·공유증거만 — 귀속 불확실)"
    elif on:
        verdict = "★ON인데 무발화 — 감사"
    elif ex_fired:
        # ★형제-ON 귀속(r2b): 같은 정확 태그를 가진 **켜진** 형제가 있으면 발화는 그쪽 것이다
        #   (예: SIG=0·OBS=1에서 [T2_TOOL_SIGNATURE]는 OBSERVE의 발화). 진짜 오염만 ⚠로.
        _sib = [g for g in all_flags if g != fl
                and (exact_tags.get(g) or set()) & set(ex)
                and envmap.get(g) not in (None, "", "0")]
        verdict = ("공유태그(형제 %s 귀속)" % _sib[0]) if _sib else "⚠OFF인데 발화 — 오염(정확증거)"
    else:
        verdict = "OFF"
    rows.append({"flag": fl, "value": raw, "on": on, "tags": ex, "shared": sh,
                 "fired": ex_fired, "shared_fired": sh_fired, "verdict": verdict})

order = {"⚠OFF인데 발화 — 오염(정확증거)": 0, "★ON인데 무발화 — 감사": 1,
         "정상?(ON·공유증거만 — 귀속 불확실)": 2, "정상(ON·발화·정확증거)": 3,
         "파라미터(값 보고만)": 4, "OFF": 5, "파라미터·미설정": 6}
# (공유태그(형제…) 판정은 order 미등재 → 말미 출력)
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
print("=" * 96)
print("요약: ★ON·무발화 %d · ⚠오염(정확증거) %d · 공유증거만 %d · 정상 %d · 파라미터 %d / 전체 %d"
      % (sum(1 for r in rows if r["verdict"].startswith("★")),
         sum(1 for r in rows if r["verdict"].startswith("⚠")),
         sum(1 for r in rows if r["verdict"].startswith("정상?")),
         sum(1 for r in rows if r["verdict"].startswith("정상(")),
         sum(1 for r in rows if r["verdict"].startswith("파라미터")), len(rows)))
if A.json:
    json.dump(rows, open(A.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s" % A.json)
sys.exit(2 if n_bad else 0)
