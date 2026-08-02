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


# ═══════════════════════════════════════════════════════════════════════════
# ★설치-경로 감사 (2026-08-02 신설 · ARG_SCHEMA 死코드 사고에서 상설화)
#   판별식: go_stack ON 플래그 × **승자-설치자 span 내 참조 수 == 0** → 死.
#   배경: `LLMAgent._generate_next_message` 할당은 3곳(patched·gen_gated·unified)이고
#   러너(t2_run_gated.py)는 `if _unified: … elif …`로 **정확히 하나만** 설치한다.
#   패치 파일 안에 코드가 있어도 승자 span 밖이면 실행되지 않는다(WEV·ARG_SCHEMA 실사고 2건).
# ═══════════════════════════════════════════════════════════════════════════
def _install_path_audit(gate_path=None, go_path=None):
    import re as _re
    import os as _os
    here = _os.path.dirname(_os.path.abspath(__file__))
    gate_path = gate_path or _os.path.join(here, "t2_gate_patch.py")
    go_path = go_path or _os.path.join(here, "go_stack.sh")
    try:
        src = open(gate_path, encoding="utf-8").read().splitlines()
        go = open(go_path, encoding="utf-8").read()
    except OSError as e:
        print("\n[설치-경로 감사] 스킵(파일 없음): %r" % e)
        return []
    # 설치자 = _generate_next_message 를 할당하는 지점들 → 각 할당의 소속 함수 span 추정
    assigns = [i + 1 for i, l in enumerate(src)
               if "_generate_next_message" in l and "=" in l and "def " not in l]
    tops = [i + 1 for i, l in enumerate(src) if l.startswith("def ")]
    spans = {}
    for a in assigns:
        start = max([t for t in tops if t <= a], default=1)
        end = min([t for t in tops if t > a], default=len(src))
        spans[start] = (start, end, src[start - 1].split("(")[0].replace("def ", "").strip())
    # 승자 = 러너가 _unified 조건에서 부르는 installer (관례상 이름에 'unified')
    winner = None
    for s, (lo, hi, nm) in spans.items():
        if "unified" in nm:
            winner = (lo, hi, nm)
    if winner is None:
        print("\n[설치-경로 감사] 승자 설치자(unified) 미발견 — 러너 라우팅 수동 확인 필요")
        return []
    lo, hi, nm = winner
    on = set(_re.findall(r"(T2_[A-Z0-9_]+)=1", go))
    tops = [i + 1 for i, l in enumerate(src) if l.startswith("def ")]

    def _enclosing(ln):
        s = max([t for t in tops if t <= ln], default=1)
        return src[s - 1].split("(")[0].replace("def ", "").strip()

    # ⚠오탐 보정(2026-08-02 실측): 승자 span **밖**에 정의됐어도 span **안에서 호출되는 헬퍼**면 생존.
    #   (초판이 T2_READ_DEDUP=_install_regen_exec·T2_REGEN_BUDGET=_regen_budget_ok을 死로 오판)
    #   또한 러너(t2_run_gated.py)가 소비하는 라우팅 플래그도 死가 아니다(T2_PROV_REGEN).
    try:
        runner = open(_os.path.join(here, "t2_run_gated.py"), encoding="utf-8").read()
    except OSError:
        runner = ""
    dead_on = []
    for f in sorted(on):
        refs = sum(1 for i in range(lo - 1, min(hi, len(src))) if f in src[i])
        if refs or f not in "\n".join(src):
            continue
        if f in runner:                       # 러너 소비 = 라우팅 플래그
            continue
        alive_via_helper = False
        for i, l in enumerate(src):
            if f in l:
                nm2 = _enclosing(i + 1)
                if nm2 and any(_re.search(r"\b%s\s*\(" % _re.escape(nm2), src[j])
                               for j in range(lo - 1, min(hi, len(src)))):
                    alive_via_helper = True
                    break
        if not alive_via_helper:
            dead_on.append(f)
    print("\n★설치-경로 감사 — 승자 설치자 = %s (라인 %d~%d)" % (nm, lo, hi))
    if dead_on:
        print("  ☠DEAD-ON (go_stack ON인데 승자 span 참조 0) %d종:" % len(dead_on))
        for f in dead_on:
            print("    %s" % f)
        print("  ⇒ 이 플래그들은 켜져 있어도 **실행되지 않는다** — 이설 or OFF 선언 필요")
    else:
        print("  DEAD-ON 0종 — ON 플래그 전부 승자 경로에서 참조됨")
    return dead_on


_install_path_audit()
