# -*- coding: utf-8 -*-
"""x480 — task_093 정밀 포렌식 (원인 확정) · 오프라인·모델 0·env 0.

★왜 (2026-08-22 사용자 지시 *"093 정밀 포렌식하여 원인 확정하라"*):
  093 은 네 런 연속 reward 0.0 인데 **하류 증상이 매번 다르다** —
    t7337  격리 서브가 0.0 복사 → 폐기 → 폴백 30.0
    t7338  서브 0.0 → 도구 None → WEV livelock (34분 소모)
    t7341  서브 -1 → 폴백 32.999… → WEV deny 10
    t7342  그 계산 도구에 도달조차 안 함
  증상이 갈리는데 결과가 늘 같다면 근인은 **더 상류**에 있다. 이 프로브는 네 궤적을
  같은 자로 재고 **공통분모**를 찾는다.

채점 단위는 `reward` 뿐이고([[69]]) 실패 단위는 **변이 집합**(MISSING/WRONGARG/EXTRA)이다 —
정본 `t2_forensic.mutation_diff` 를 쓴다([[67]] 사본 금지).
⚠gold 는 **진단에만** 쓴다. 여기서 얻은 것으로 A2 를 쓰면 [[23]] 위반이다([[69]] 축자).

실행: py -3 x480_093_forensic.py
"""
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

import t2_forensic as F                                          # noqa: E402

TAGS = [
    ("t7337", "bank_t7337_smoke_20260822"),
    ("t7338", "bank_t7338_smoke_20260822"),
    ("t7341a", "bank_t7341_smoke_20260822_a"),
    ("t7342a", "bank_t7342_smoke_20260822_a"),
]
TASK = "task_093"


def load_093(tag):
    """그 태그의 093 sim 들 (없으면 빈 리스트)."""
    out = []
    for suf in ("_results.json.gz", ".results.json.gz"):
        try:
            for s in F.sims(tag, suffix=suf):
                if F.task_id(s) == TASK:
                    out.append(s)
            if out:
                return out
        except Exception:
            continue
    return out


print("=" * 78)
print("x480 — task_093 정밀 포렌식")
print("=" * 78)

found = {}
for label, tag in TAGS:
    sims = load_093(tag)
    if sims:
        found[label] = sims
    print("  %-8s %s → 093 sim %d개" % (label, tag, len(sims)))

if not found:
    print("\n⛔093 sim 을 하나도 못 읽었다 — 경로/접미사 확인 필요")
    sys.exit(1)

# ── ① reward 와 변이 집합 ────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("① reward · 변이 집합 (실패 단위 = MISSING / WRONGARG / EXTRA)")
print("=" * 78)
diffs = {}
for label, sims in found.items():
    for s in sims:
        ri = s.get("reward_info") or {}
        print("\n[%s] reward=%s  term=%s  basis=%s"
              % (label, ri.get("reward"), F.term_reason(s), F.reward_basis(s)))
        try:
            d = F.mutation_diff(s)
            diffs[label] = d
            for k in ("missing", "wrongarg", "extra"):
                v = d.get(k) or []
                if v:
                    print("   %-9s %d" % (k.upper(), len(v)))
                    for it in v[:6]:
                        print("      -", str(it)[:150])
        except Exception as e:
            print("   ⚠mutation_diff 실패: %r" % (e,))

# ── ② 공통분모 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("② 네 런의 공통분모 — 늘 빠지는 것 / 늘 틀리는 것")
print("=" * 78)


def keyset(d, k):
    return {str(x) for x in (d.get(k) or [])}


if diffs:
    labels = list(diffs)
    for k in ("missing", "wrongarg", "extra"):
        sets = [keyset(diffs[l], k) for l in labels]
        common = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        print("\n  %s — 공통 %d / 합집합 %d" % (k.upper(), len(common), len(union)))
        for c in sorted(common)[:10]:
            print("     ★공통:", c[:150])
        only = union - common
        for o in sorted(only)[:6]:
            where = [l for l in labels if o in keyset(diffs[l], k)]
            print("      (%s만):" % ",".join(where), o[:120])

# ── ③ 궤적 행동 시퀀스 ───────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("③ 실제 행동 시퀀스 (mutating 만) — 어디서 갈리나")
print("=" * 78)
for label, sims in found.items():
    for s in sims:
        try:
            att = F.attempted_mutations(s)
            print("\n[%s] 시도한 변이 %d건" % (label, len(att)))
            for a in att[:12]:
                print("   ", str(a)[:150])
        except Exception as e:
            print("[%s] ⚠attempted_mutations 실패: %r" % (label, e))

# ── ④ gold 가 요구하는 것 ────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("④ gold 가 요구하는 변이 (진단 전용·[[23]] A2 저작 금지)")
print("=" * 78)
for label, sims in list(found.items())[:1]:
    for s in sims:
        try:
            g = F.gold_mutations(s)
            print("  gold 변이 %d건" % len(g))
            for x in g:
                print("   ", str(x)[:170])
        except Exception as e:
            print("  ⚠gold_mutations 실패: %r" % (e,))

print("\n완료.")
