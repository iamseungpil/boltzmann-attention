#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x43: **dark 레버 전수 감사** — 구현돼 있는데 라이브에서 발화한 적 없는 레버 찾기.

동기(사용자 지시 2026-08-02): wrap 기능서브(`T2_FN_ISOLATE`)가 설계·리뷰·구현 완료 상태로
**한 번도 발화한 적이 없었다**(플래그 미설정 + A2 `wraps`에 실사용 도구명 부재). 같은 종류의
dark가 더 있는지 **기계로** 전수 조사한다(손 목록 금지 — x20 교훈).

방법:
  ① 코드 전수에서 stderr **태그**(`[T2_...]`·`[axis]`류 print)를 추출 = "정의된 발화 신호"
  ② 영속 라이브 로그(sim_results/*.log.gz) 전수에서 태그 관측 계수 = "실제 발화"
  ③ 정의됐는데 관측 0 = **dark 후보** → 소속 파일·게이트 플래그·최근 런(qp32p{1,2}) 한정 계수 병기
  ④ x20(플래그 회계)와 교차: ON인데 태그 관측 0 = 최우선 감사 대상

한계(정직): 태그 없는 레버(print를 안 하는 조용한 레버)는 이 방법으로 못 본다 — 그런 레버는
A2 선언 소비 여부로 별도 확인해야 하며, 목록만 남긴다.
"""
import collections
import glob
import gzip
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")

# ── ① 코드에서 태그 추출 ─────────────────────────────────────────────────────
ENGINE = [f for f in glob.glob(os.path.join(HERE, "t2_*.py")) +
          [os.path.join(HERE, "gate_interpreter.py")] if os.path.isfile(f)]
TAG_RE = re.compile(r'"(\[T2_[A-Z0-9_]+[^"\]]*\])')
tag_src = {}                    # tag -> set(files)
flag_near = collections.defaultdict(set)   # tag -> flags in ±40 lines
for f in ENGINE:
    src = open(f, encoding="utf-8").read()
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        for m in TAG_RE.finditer(ln):
            tag = m.group(1)
            tag_src.setdefault(tag, set()).add(os.path.basename(f))
            lo, hi = max(0, i - 40), min(len(lines), i + 5)
            for fm in re.finditer(r'environ(?:\.get)?\(\s*"(T2_[A-Z0-9_]+)"', "\n".join(lines[lo:hi])):
                flag_near[tag].add(fm.group(1))
print("정의된 stderr 태그 %d종 (엔진 %d파일)" % (len(tag_src), len(ENGINE)))

# ── ② 라이브 로그 전수 관측 ──────────────────────────────────────────────────
obs_all = collections.Counter()
obs_recent = collections.Counter()
RECENT = ("qp32p1", "qp32p2", "qpnt2", "y2c")
logs = sorted(glob.glob(os.path.join(SIMDIR, "*.log.gz")))
for f in logs:
    recent = any(r in os.path.basename(f) for r in RECENT)
    try:
        txt = gzip.open(f, "rt", encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    for tag in tag_src:
        n = txt.count(tag)
        if n:
            obs_all[tag] += n
            if recent:
                obs_recent[tag] += n
print("라이브 로그 %d개 스캔 (최근 런 = %s)\n" % (len(logs), "/".join(RECENT)))

# ── ③④ 판정 ─────────────────────────────────────────────────────────────────
# go_stack + axis32 러너의 ON 플래그
onflags = set()
for sh in ("go_stack.sh", "run_axis32_chain.sh"):
    p = os.path.join(HERE, sh)
    if os.path.exists(p):
        for m in re.finditer(r"export\s+(T2_[A-Z0-9_]+)=([^\s]+)", open(p, encoding="utf-8").read()):
            if m.group(2) not in ("0", '""'):
                onflags.add(m.group(1))

rows = []
for tag, files in sorted(tag_src.items()):
    flags = flag_near.get(tag) or set()
    on = bool(flags & onflags) or not flags     # 플래그 없는 태그 = 상시 경로
    rows.append((tag, obs_all.get(tag, 0), obs_recent.get(tag, 0),
                 ",".join(sorted(flags)) or "(무플래그·상시)", on, ",".join(sorted(files))))

print("=" * 100)
print("★DARK 후보 — 태그 정의됨 · **전 로그 관측 0** · 소속 플래그가 ON(또는 상시)")
dark = [r for r in rows if r[1] == 0 and r[4]]
for tag, a, rct, flags, on, files in dark:
    print("  %-46s flags=%-34s %s" % (tag, flags[:34], files))
print("  소계 %d종" % len(dark))

print("\n반(半)-dark — 과거엔 발화했는데 **최근 런(qp32/qpnt2/y2c) 관측 0**")
half = [r for r in rows if r[1] > 0 and r[2] == 0 and r[4]]
for tag, a, rct, flags, on, files in half:
    print("  %-46s 과거 %5d회 · flags=%s" % (tag, a, flags[:40]))
print("  소계 %d종" % len(half))

print("\n정상 발화 상위 12 (참고)")
for tag, n in obs_recent.most_common(12):
    print("  %-46s 최근 %6d회" % (tag, n))

print("\n⚠이 방법의 사각(태그 없는 레버) — A2 선언 소비로 별도 확인 필요:")
noprint = sorted(f2 for f2 in
                 {fl for fls in flag_near.values() for fl in fls} ^
                 {m.group(1) for f in ENGINE
                  for m in re.finditer(r'environ(?:\.get)?\(\s*"(T2_[A-Z0-9_]+)"',
                                       open(f, encoding="utf-8").read())})
print("  " + ", ".join(noprint[:20]))
