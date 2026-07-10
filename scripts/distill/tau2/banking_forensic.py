#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""banking 저-pass 근본원인 전수 포렌식 ([[08]]).
(1) 모델별 pass·종료사유 분포 (2) 태스크-레벨 교차표(17모델 중 몇이 통과 — universal-fail 식별)
(3) horizon(gold act 수)·discoverable-도구 요구와 pass의 교차 (4) universal-fail 태스크 gold 구조 요약.
usage: py -3 banking_forensic.py C:\\tmp\\traj"""
import sys, glob, os, json, io
from collections import Counter, defaultdict
import statistics
import fine_function_decomp as F

def main(d):
    files = sorted(glob.glob(os.path.join(d, "*_banking.json")))
    print(f"models: {len(files)}")
    task_pass = defaultdict(list)   # task_id -> [(model, passed)]
    task_gold = {}                  # task_id -> gold acts (from first sim seen)
    term_all = Counter()
    print("\n#### (1) per-model pass & termination ####")
    print(f"{'model':16}{'n':>5}{'pass':>7}  top terminations")
    for f in files:
        lbl = os.path.basename(f).replace("_banking.json", "")
        data = json.load(io.open(f, encoding="utf-8"))
        sims = data.get("simulations") or data
        npass = 0
        terms = Counter()
        for s in sims:
            ok = not F.is_fail(s)
            npass += ok
            tid = str(s.get("task_id"))
            task_pass[tid].append((lbl, ok))
            terms[s.get("termination_reason", "?")] += 1
            if tid not in task_gold:
                try:
                    task_gold[tid] = F.gold_acts(s)
                except Exception:
                    task_gold[tid] = None
        term_all.update(terms)
        top = " ".join(f"{k}:{v}" for k, v in terms.most_common(3))
        print(f"{lbl:16}{len(sims):>5}{npass/len(sims):>7.3f}  {top}")
    print("\npooled termination:", dict(term_all.most_common(6)))

    # (2) task-level cross table
    print("\n#### (2) task-level: how many of 17 models pass each task ####")
    hist = Counter()
    unifail, nearfail, easy = [], [], []
    for tid, lst in task_pass.items():
        k = sum(1 for _, ok in lst if ok)
        n = len(lst)
        hist[k] += 1
        if k == 0: unifail.append(tid)
        elif k <= 2: nearfail.append(tid)
        elif k >= n - 1: easy.append(tid)
    for k in sorted(hist):
        print(f"  pass by {k:>2} models: {hist[k]:>3} tasks")
    ntask = len(task_pass)
    print(f"tasks={ntask} · universal-fail(0/17)={len(unifail)} ({100*len(unifail)/ntask:.0f}%) · ≤2/17={len(unifail)+len(nearfail)} ({100*(len(unifail)+len(nearfail))/ntask:.0f}%) · ≥16/17={len(easy)}")

    # (3) horizon & discoverable cross
    print("\n#### (3) pass-rate vs gold-action horizon ####")
    def bucket(g):
        if g is None: return "?"
        L = len(g)
        return "1-3" if L <= 3 else ("4-6" if L <= 6 else ("7-9" if L <= 9 else "10+"))
    agg = defaultdict(lambda: [0, 0])  # bucket -> [pass, total]
    disc = defaultdict(lambda: [0, 0]) # requires unlock/discover tool in gold?
    for tid, lst in task_pass.items():
        g = task_gold.get(tid)
        b = bucket(g)
        p = sum(1 for _, ok in lst if ok); n = len(lst)
        agg[b][0] += p; agg[b][1] += n
        req = False
        if g:
            names = " ".join(n0.lower() for n0, _ in g)
            req = ("unlock" in names) or ("discover" in names) or ("request_" in names)
        disc[req][0] += p; disc[req][1] += n
    for b in ("1-3", "4-6", "7-9", "10+", "?"):
        if agg[b][1]:
            print(f"  gold {b:>4} acts: pass {agg[b][0]/agg[b][1]:.3f}  (n={agg[b][1]})")
    for r in (False, True):
        if disc[r][1]:
            print(f"  gold {'has' if r else 'no '} unlock/discover/request: pass {disc[r][0]/disc[r][1]:.3f} (n={disc[r][1]})")

    # (4) universal-fail gold structure
    print("\n#### (4) universal-fail tasks — gold structure ####")
    L = []
    for tid in unifail:
        g = task_gold.get(tid) or []
        names = [n0 for n0, _ in g]
        L.append((tid, len(g), names))
    L.sort(key=lambda x: -x[1])
    for tid, n, names in L[:15]:
        print(f"  {tid:>10} gold={n:>2}: {', '.join(names[:8])}{' …' if n > 8 else ''}")
    hor_uni = [n for _, n, _ in L if n]
    hor_all = [len(g) for g in task_gold.values() if g]
    if hor_uni:
        print(f"\nuniversal-fail horizon med={statistics.median(hor_uni)} vs all-task med={statistics.median(hor_all)}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\tmp\traj")
