#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x772 — 전사-서브 레버군(SG 4종)의 **캠페인 정본 집합 재구성 + 발화 귀속** (2026-09-05).

⛔판정하지 않는다. 세기만 한다([[62]]·[[77]]).

무엇을 하나
  ⑴ 캠페인 창(2026-09-03T13 ~ 09-05T06) 안의 태그 전부에서 **태스크당 최신 sim** 을 골라
     97/51/46 을 재현하고, x768 `pairs.txt` 의 46 핀과 **완전 일치**하는지 검산한다.
  ⑵ 그 정본 sim 이 어느 태그에 있는지 찍어, 로그의 `[sim=task_NNN#...]` 발화가
     **정본 sim 의 것인지** 대조할 수 있게 한다.
  ⑶ 관심 태스크(전사 서브가 도는 태스크)의 reward 를 함께 찍는다.

사용(리모트): /home/woori/venvs/seka_env/bin/python x772_sgfamily_fire.py
"""
import io, json, os, sys, time

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LOGDIR = "/home/woori/scratch/logs"
LO = time.mktime(time.strptime("2026-09-03 13:00", "%Y-%m-%d %H:%M"))
HI = time.mktime(time.strptime("2026-09-05 06:00", "%Y-%m-%d %H:%M"))

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

best = {}   # task -> (end_time, tag, sim_id, reward)
tags_seen = 0
for tag in sorted(os.listdir(SIMROOT)):
    p = os.path.join(SIMROOT, tag, "results.json")
    if not os.path.exists(p):
        continue
    st = os.path.getmtime(p)
    if not (LO <= st <= HI):
        continue
    if not tag.startswith("bank_"):
        continue
    tags_seen += 1
    try:
        d = json.load(io.open(p, encoding="utf-8"))
    except Exception as e:
        print("LOADFAIL %s %r" % (tag, e))
        continue
    for s in (d.get("simulations") or []):
        tid = s.get("task_id")
        if not tid:
            continue
        et = s.get("end_time") or s.get("start_time") or ""
        rw = ((s.get("reward_info") or {}).get("reward"))
        cur = best.get(tid)
        if cur is None or et > cur[0]:
            best[tid] = (et, tag, s.get("id"), rw)

fails = sorted(t for t, v in best.items() if not v[3])
passes = sorted(t for t, v in best.items() if v[3])
print("TAGS %d · TASKS %d · PASS %d · FAIL %d" % (tags_seen, len(best), len(passes), len(fails)))

pinp = "/home/woori/scratch/x768/pairs.txt"
if os.path.exists(pinp):
    pins = {}
    for ln in io.open(pinp, encoding="utf-8"):
        a = ln.split()
        if len(a) == 3:
            pins[a[1]] = (a[0], a[2])
    same = sum(1 for t in pins if t in best and best[t][1] == pins[t][0] and best[t][2] == pins[t][1])
    print("PIN CHECK: pairs=%d · 46핀 중 태그·simid 완전일치 %d · 실패집합 동일=%s"
          % (len(pins), same, sorted(pins) == fails))
    for t in sorted(pins):
        if t not in best:
            print("  MISSING-IN-RECON %s" % t)
        elif (best[t][1], best[t][2]) != pins[t]:
            print("  DIFFPIN %s recon=%s/%s pairs=%s/%s" % (t, best[t][1], best[t][2][:8],
                                                            pins[t][0], pins[t][1][:8]))

WATCH = ["task_017", "task_018", "task_019", "task_020", "task_021", "task_022", "task_023",
         "task_024", "task_025", "task_026", "task_027", "task_028", "task_029",
         "task_059", "task_063", "task_064", "task_065", "task_066", "task_067", "task_068",
         "task_072", "task_073", "task_074", "task_093", "task_094", "task_095", "task_096",
         "task_097"]
print("")
print("%-10s %-40s %-10s %s" % ("task", "canonical tag", "sim", "reward"))
for t in WATCH:
    v = best.get(t)
    if not v:
        print("%-10s (정본 집합 밖)" % t)
        continue
    print("%-10s %-40s %-10s %s" % (t, v[1], (v[2] or "")[:8], v[3]))
