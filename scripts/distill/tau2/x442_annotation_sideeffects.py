# -*- coding: utf-8 -*-
r"""x442 — **주석을 한 칸 더 주면 무엇을 파나**: 이미 배포된 유사물로 재는 오프라인 선계량

## 왜 (사용자 지시 2026-08-20 밤 *"2부터 하라"*)
C562 가 *"후보별 값을 주면 0.38 → 0.98"* 을 냈다. 그러나 같은 자리에서 `present` 는 pass 를 사고
**미조회 날조 5.6% → 10.4% · 조회 2.62 → 0.48** 을 팔았다(C31/C34/C43). ⇒ 배선 전에 **부작용을 먼저**
재야 한다(등대 §1.3 Δspurious ≤ 0 · [[70]]). 유료 런 없이 재는 길은 하나뿐이다 —
**이미 배포된 가장 가까운 유사물**을 쓰는 것.

    유사물 = `check_card_application_fit` 의 `rate_for('<범주>')` 주석(C204/D6).
    이 주석은 모델이 **선택적 인자 `spend_category` 를 줄 때만** 붙는다 ⇒ 같은 코퍼스 안에
    **주석이 붙은 sim** 과 **안 붙은 sim** 이 공존한다(자연실험).

## 무엇을 재나 (전부 정본 재사용 · 새 술어 0)
    ⒜ 날조   = 엔진 `t2_gate_patch._first_fab_call` **그대로**(ctx = user+tool 텍스트만·[[03b]])
    ⒝ 조회   = write 가 아닌 도구 호출 수 (write 집합은 `t2_forensic.write_tools` = gold 채점표에서 도출)
    ⒞ 쓰기   = write 도구 호출 수
    ⒟ reward = sim 단위 성적([[69]])
전부 **fit 호출 이후**만 센다 — 주석은 그 뒤에만 영향을 줄 수 있다.

## ⛔이것은 인과가 아니다
`spend_category` 를 줄지는 **모델이 고른다** ⇒ 선택 효과가 있다(범주를 말한 손님이 있는 태스크에
몰린다). 그래서 ⑴전체 ⑵**태스크 내** ⑶**태그 내** 세 층으로 낸다([[08]]). 층을 넘어 읽지 말 것.

사용: py -3 x442_annotation_sideeffects.py
"""
import collections
import glob
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
from t2_gate_patch import _first_fab_call, DEFAULT_ARG_HINTS  # noqa: E402

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
FIT = "check_card_application_fit"


class _TC(object):
    """엔진 술어가 기대하는 최소 인터페이스 — `bank_readpath_writepath._TC` 와 같은 shim."""

    def __init__(self, name, args):
        self.name = name
        self.arguments = args


def tags():
    base = os.path.abspath(os.path.join(REP, "sim_results"))
    return sorted({os.path.basename(p).replace(".results.json.gz", "")
                   for p in glob.glob(os.path.join(base, "*.results.json.gz"))})


class _Msg(object):
    """엔진 술어가 기대하는 최소 인터페이스(`.tool_calls`) — `bank_readpath_writepath._TC` 와 같은 shim."""

    def __init__(self, tcs):
        self.tool_calls = tcs


def fab_count(tcs, ctx):
    """그 메시지의 날조 인자 수 — 엔진 함수를 exclude 누적으로 반복 호출(정본 방식·새 술어 0)."""
    n, excl = 0, set()
    while n <= 40:
        hit = _first_fab_call(_Msg(tcs), ctx, DEFAULT_ARG_HINTS, exclude=frozenset(excl))
        if hit is None:
            break
        tc, k, s = hit
        n += 1
        excl.add((id(tc), k, s))
    return n



def scan(sim, wt):
    """fit 호출 **이후**의 조회·쓰기·날조를 센다. 주석 유무는 그 호출의 인자로 가른다."""
    msgs = sim.get("messages") or []
    first, cat = None, False
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            if F.nameof(tc) != FIT:
                continue
            if first is None:
                first = i
                a = F.argsof(tc) or {}
                cat = bool(str(a.get("spend_category") or "").strip())
    if first is None:
        return None
    reads = writes = fabs = 0
    ctx = []
    for i, m in enumerate(msgs):
        role = m.get("role") or ""
        if role in ("user", "tool"):
            ctx.append(str(m.get("content") or ""))
        if i <= first:
            continue
        tcs = [_TC(F.nameof(tc), F.argsof(tc)) for tc in (m.get("tool_calls") or [])]
        for tc in (m.get("tool_calls") or []):
            nm = F.nameof(tc)
            if nm in wt:
                writes += 1
            else:
                reads += 1
        if role == "assistant" and tcs:
            fabs += fab_count(tcs, "\n".join(ctx))
    return {"cat": cat, "reads": reads, "writes": writes, "fabs": fabs,
            "reward": (sim.get("reward_info") or {}).get("reward"),
            "task": F.task_id(sim)}


def show(label, rows):
    if not rows:
        return
    g = collections.defaultdict(list)
    for r in rows:
        g["주석 O" if r["cat"] else "주석 X"].append(r)
    print("  %s (sim %d)" % (label, len(rows)))
    for k in ("주석 O", "주석 X"):
        v = g.get(k) or []
        if not v:
            continue
        n = float(len(v))
        print("     %-6s n=%-4d 조회 %5.2f · 쓰기 %4.2f · 날조 %4.2f · pass %.2f"
              % (k, len(v), sum(x["reads"] for x in v) / n, sum(x["writes"] for x in v) / n,
                 sum(x["fabs"] for x in v) / n,
                 sum(1 for x in v if x["reward"] == 1.0) / n))


def main():
    rows = []
    for t in tags():
        try:
            sims = F.sims(t, ".results.json.gz")
        except Exception:
            continue
        if not sims:
            continue
        wt = F.write_tools({"simulations": sims})
        for s in sims:
            r = scan(s, wt)
            if r:
                r["tag"] = t
                rows.append(r)
    print("=" * 100)
    print("x442 · 배포된 주석(rate_for)의 자연실험 · fit 호출 sim %d" % len(rows))
    print("⛔인과 아님 — `spend_category` 를 줄지는 모델이 고른다(선택 효과). 층을 넘어 읽지 말 것.")
    print("=" * 100)
    show("① 전체", rows)
    print()
    print("② 태스크 내 (양쪽 팔이 다 있는 태스크만)")
    bytask = collections.defaultdict(list)
    for r in rows:
        bytask[r["task"]].append(r)
    for task in sorted(bytask):
        v = bytask[task]
        if len({x["cat"] for x in v}) < 2:
            continue
        show("   %s" % task, v)
    print()
    print("③ 태그 내 (양쪽 팔이 다 있는 태그만·상위 6)")
    bytag = collections.defaultdict(list)
    for r in rows:
        bytag[r["tag"]].append(r)
    shown = 0
    for tag in sorted(bytag, key=lambda k: -len(bytag[k])):
        v = bytag[tag]
        if len({x["cat"] for x in v}) < 2:
            continue
        show("   %s" % tag, v)
        shown += 1
        if shown >= 6:
            break
    p = os.path.abspath(os.path.join(REP, "x442_annotation_sideeffects.json"))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
