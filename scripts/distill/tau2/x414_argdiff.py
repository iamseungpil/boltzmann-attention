# -*- coding: utf-8 -*-
r"""x414 - 미매치 gold 중 **호출은 한 것**의 인자 대조 (gold vs 실제) — 태스크별 실패 시작지점

x400 의 MISCALLED 55건이 최대 덩어리다. 그 55건이 *어느 인자에서* 갈렸는지를 축자로 나란히 놓는다.
집계 없음 - 태스크별로 읽기 위한 것.
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

import t2_forensic as F
import x396_saying_vs_doing as C


def flat(a):
    a = F.norm_args(a)
    if isinstance(a, dict) and isinstance(a.get("arguments"), dict):
        a = a["arguments"]
    if isinstance(a, dict) and isinstance(a.get("arguments"), str):
        try:
            a = json.loads(a["arguments"])
        except Exception:
            pass
    return a if isinstance(a, dict) else {"_": a}


def main():
    want = [x for x in sys.argv[1:] if x.startswith("task_")]
    for tag in C.TAGS:
        for sim in F.scored(tag, C.SUF):
            if ((sim.get("reward_info") or {}).get("reward") or 0) >= 1.0:
                continue
            t = F.task_id(sim)
            if want and t not in want:
                continue
            calls = {}
            for m, tc in F.calls(sim):
                a = F.argsof(tc)
                nm = str(F.inner_name(a) or F.nameof(tc))
                calls.setdefault(nm, []).append(flat(a))
            shown = False
            for g in C.gold_rows(sim):
                if g["match"] or g["name"] not in calls:
                    continue
                if not shown:
                    print("=" * 110)
                    print("### %s t%s  reward=%s" % (t, sim.get("trial"),
                                                     (sim.get("reward_info") or {}).get("reward")))
                    shown = True
                gd = flat(g["args"])
                print("  ★%s  (실제 호출 %d회)" % (g["name"], len(calls[g["name"]])))
                for act in calls[g["name"]][:3]:
                    keys = sorted(set(list(gd.keys()) + list(act.keys())))
                    for k in keys:
                        gv, av = gd.get(k, "∅"), act.get(k, "∅")
                        mark = "  " if str(gv) == str(av) else "≠≠"
                        print("     %s %-32s gold=%-34s 실제=%s"
                              % (mark, k[:32], str(gv)[:34], str(av)[:44]))
                    print("     " + "-" * 92)
    return 0


sys.exit(main())
