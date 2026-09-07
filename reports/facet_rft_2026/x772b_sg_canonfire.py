#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x772b — SG 레버군 발화를 **캠페인 정본 sim 에만** 귀속시킨다 (2026-09-05 · GPU 0).

정본 = x768 규칙(태스크당 최신 sim). 실패 46 은 `pairs.txt` 를 권위로 덮어쓴다.
로그 줄의 `[sim=task_NNN#sSEED]` 는 seed 라 uuid 가 아니다 ⇒ **(태그, 태스크)** 로 맞춘다.
⛔판정하지 않는다. 세기만 한다.
"""
import collections, io, json, os, re, sys, time

SIMROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LOGDIR = "/home/woori/scratch/logs"
LO = time.mktime(time.strptime("2026-09-03 13:00", "%Y-%m-%d %H:%M"))
HI = time.mktime(time.strptime("2026-09-05 03:30", "%Y-%m-%d %H:%M"))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

best = {}
for tag in sorted(os.listdir(SIMROOT)):
    p = os.path.join(SIMROOT, tag, "results.json")
    if not (tag.startswith("bank_") and os.path.exists(p)):
        continue
    if not (LO <= os.path.getmtime(p) <= HI):
        continue
    try:
        d = json.load(io.open(p, encoding="utf-8"))
    except Exception:
        continue
    for s in (d.get("simulations") or []):
        tid = s.get("task_id")
        et = s.get("end_time") or s.get("start_time") or ""
        if not tid:
            continue
        if tid not in best or et > best[tid][0]:
            best[tid] = (et, tag, s.get("id"), ((s.get("reward_info") or {}).get("reward")))

pins = {}
for ln in io.open("/home/woori/scratch/x768/pairs.txt", encoding="utf-8"):
    a = ln.split()
    if len(a) == 3:
        pins[a[1]] = a[0]
        best[a[1]] = ("PIN", a[0], a[2], 0.0)

canon = {t: v[1] for t, v in best.items()}
rew = {t: v[3] for t, v in best.items()}
FAIL = set(pins)
print("정본 태스크 %d · 실패핀 %d · 통과 %d"
      % (len(canon), len(FAIL), sum(1 for t in rew if rew[t])))

MARK = [("PROMPT_V2", r"\[T2_SG_PROMPT_V2\]"),
        ("RECORD_ORDER", r"\[T2_SG_RECORD_ORDER\]"),
        ("ROW_COUNT", r"\[T2_SG_ROW_COUNT\]"),
        ("REQREADS_CANON", r"\[T2_SG_REQREADS\] .*(정본 입구 채택)"),
        ("REQREADS_DENY", r"\[T2_SG_REQREADS\] .* denied"),
        ("CLOSE_SELF_ON", r"형태=자기완결"),
        ("CLOSE_SELF_OFF", r"형태=덧붙임"),
        ("ISO_FETCH", r"\[T2_SG_ISOLATE\] fetch "),
        ("ISO_OPSIZE", r"\[T2_SG_ISOLATE\] operand-size "),
        ("ISO_SKIP", r"호출자가 operand 를 전부 채웠다"),
        ("ISO_RATEFIX", r"\[T2_SG_ISOLATE\] get_reward_discrepancies: ")]
RE_SIM = re.compile(r"^\[sim=(task_\d+)#")
hits = collections.defaultdict(lambda: collections.Counter())
noncanon = collections.defaultdict(lambda: collections.Counter())
samples = collections.defaultdict(list)

for fn in sorted(os.listdir(LOGDIR)):
    if not (fn.startswith("bank_") and fn.endswith(".log")) or fn.endswith("_driver.log"):
        continue
    fp = os.path.join(LOGDIR, fn)
    if not (LO <= os.path.getmtime(fp) <= HI + 7200):
        continue
    tag = fn[:-4]
    try:
        fh = io.open(fp, encoding="utf-8", errors="replace")
    except Exception:
        continue
    with fh:
        for ln in fh:
            m = RE_SIM.match(ln)
            if not m:
                continue
            tid = m.group(1)
            for name, pat in MARK:
                if re.search(pat, ln):
                    if canon.get(tid) == tag:
                        hits[name][tid] += 1
                        if len(samples[name]) < 6:
                            samples[name].append("%s %s | %s" % (tid, tag, ln.strip()[:150]))
                    else:
                        noncanon[name][tid] += 1

for name, _ in MARK:
    h = hits[name]
    inf = {t: c for t, c in h.items() if t in FAIL}
    inp = {t: c for t, c in h.items() if t not in FAIL}
    print("")
    print("== %s : 정본 발화 %d 줄 / %d 태스크" % (name, sum(h.values()), len(h)))
    print("   실패46 내: %s" % (sorted(inf.items()) or "없음"))
    print("   통과51 내: %s" % (sorted(inp.items()) or "없음"))
    if noncanon[name]:
        print("   (비정본 sim: %s)" % sorted(noncanon[name].items()))
    for s in samples[name][:3]:
        print("   ex  %s" % s)
