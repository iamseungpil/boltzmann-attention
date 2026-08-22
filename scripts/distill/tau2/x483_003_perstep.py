# -*- coding: utf-8 -*-
"""x483 — task_003 회귀의 **per-step 궤적 추적** (2026-08-22·오프라인·모델 0·env 0)

## 왜
003 은 t7336 이전부터 10 런 넘게 pass 하던 태스크인데(사용자 확인) t7345 에서 0.0 이다.
회귀는 확정이고, 여기서 하는 일은 **어느 스텝에서 무엇이 갈렸는가**를 턴 단위로 못박는 것이다.
집계·발화 카운트에서 결론으로 가지 않는다([[08]]).

## 방법
두 궤적(t7336 = pass · t7345 = fail)의 **행동 시퀀스**(assistant 의 tool_call 이름)를 정렬해
**첫 불일치 턴**을 찾고, 그 전후를 양쪽 다 원문으로 편다 — 우리 층이 그 턴에 무엇을 말했는지
(사이드카·로그 마크)까지 붙인다.

실행: py -3 x483_003_perstep.py [--task task_003] [--ctx 3]
"""
import argparse
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
BASE = os.path.join(REP, "sim_results")
SCRATCH = "/home/woori/scratch/tau2-bench/data/simulations"


def load_sim(path, task, gz=True):
    try:
        if gz:
            d = json.load(gzip.open(path, "rt", encoding="utf-8"))
        else:
            d = json.load(io.open(path, encoding="utf-8"))
    except Exception as e:
        print("  ⚠못 읽음 %s: %r" % (path, e))
        return None
    for s in (d.get("simulations") or []):
        if s.get("task_id") == task:
            return s
    return None


def args_of(tc):
    a = tc.get("arguments") or {}
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {"_raw": a[:80]}
    return a if isinstance(a, dict) else {}


def label(tc):
    a = args_of(tc)
    inner = str(a.get("agent_tool_name") or a.get("discoverable_tool_name") or "")
    return (tc.get("name") or "?") + ("/" + inner if inner else "")


def steps(sim):
    """턴 단위 스텝: (idx, role, 요약, 원문일부)."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        r = m.get("role")
        tcs = m.get("tool_calls") or []
        if r == "assistant":
            if tcs:
                out.append((i, "assistant→tool", " + ".join(label(t) for t in tcs),
                            json.dumps([args_of(t) for t in tcs], ensure_ascii=False)[:260]))
            else:
                out.append((i, "assistant→user", "(산문)", str(m.get("content") or "")[:260]))
        elif r == "user":
            out.append((i, "user", "(발화)", str(m.get("content") or "")[:260]))
        elif r == "tool":
            out.append((i, "tool", "(결과)", str(m.get("content") or "")[:260]))
    return out


def action_seq(sim):
    """행동 시퀀스 = assistant 의 tool_call 라벨만(정렬 축)."""
    seq = []
    for m in (sim.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        for t in (m.get("tool_calls") or []):
            seq.append(label(t))
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_003")
    ap.add_argument("--ctx", type=int, default=3, help="갈림점 전후 몇 스텝을 펼칠까")
    a = ap.parse_args()

    pass_p = os.path.join(BASE, "bank_t7336_halfA_20260821b.results.json.gz")
    fail_p = os.path.join(SCRATCH, "bank_t7345_halfA_20260822", "results.json")

    P = load_sim(pass_p, a.task, gz=True)
    F = load_sim(fail_p, a.task, gz=False)
    if not P or not F:
        raise SystemExit("궤적을 못 읽었다 (pass=%s fail=%s)" % (bool(P), bool(F)))

    rp = (P.get("reward_info") or {}).get("reward")
    rf = (F.get("reward_info") or {}).get("reward")
    print("=" * 100)
    print("x483 · %s · t7336 reward=%s (PASS)  ↔  t7345 reward=%s (FAIL)" % (a.task, rp, rf))
    print("=" * 100)

    sp, sf = action_seq(P), action_seq(F)
    print("\n[행동 시퀀스] t7336 %d개 · t7345 %d개" % (len(sp), len(sf)))
    n = min(len(sp), len(sf))
    div = None
    for i in range(n):
        if sp[i] != sf[i]:
            div = i
            break
    if div is None and len(sp) != len(sf):
        div = n
    if div is None:
        print("  행동 시퀀스가 완전히 같다 — 갈린 것은 **인자나 본문**이다(아래 스텝 비교로).")
    else:
        print("  ★첫 불일치 = %d 번째 행동" % div)
        lo = max(0, div - a.ctx)
        hi = min(max(len(sp), len(sf)), div + a.ctx + 1)
        print("\n  %-4s %-46s %-46s" % ("#", "t7336 (PASS)", "t7345 (FAIL)"))
        for i in range(lo, hi):
            x = sp[i] if i < len(sp) else "—"
            y = sf[i] if i < len(sf) else "—"
            mark = "  ←★" if i == div else ""
            print("  %-4d %-46s %-46s%s" % (i, x[:46], y[:46], mark))

    # ── 스텝 원문: 갈림 근처를 양쪽 다 편다 ──────────────────────────────────
    for name, sim in (("t7336 (PASS)", P), ("t7345 (FAIL)", F)):
        st = steps(sim)
        print("\n" + "─" * 100)
        print("[%s] 스텝 %d개 — 앞 %d 스텝" % (name, len(st), min(len(st), 22)))
        print("─" * 100)
        for (i, role, summ, body) in st[:22]:
            print("  [%3d] %-14s %s" % (i, role, summ))
            if body and role in ("assistant→tool", "tool", "assistant→user"):
                print("        %s" % body.replace("\n", " | ")[:200])

    # ── 결정적 차이 요약 ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("[요약] 도구 호출 횟수")
    from collections import Counter
    cp, cf = Counter(sp), Counter(sf)
    keys = sorted(set(cp) | set(cf))
    print("  %-52s %6s %6s" % ("도구", "t7336", "t7345"))
    for k in keys:
        if cp.get(k, 0) != cf.get(k, 0):
            print("  %-52s %6d %6d  ←차이" % (k[:52], cp.get(k, 0), cf.get(k, 0)))
        else:
            print("  %-52s %6d %6d" % (k[:52], cp.get(k, 0), cf.get(k, 0)))


if __name__ == "__main__":
    main()
