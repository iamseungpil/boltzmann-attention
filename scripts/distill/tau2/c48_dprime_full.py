#!/usr/bin/env python3
"""C48 - D'' : D' + 검증기 소진 시 GET 강제 폴백 + Δspurious 측정.

두 모집단에 같은 레버를 건다:
  FAB   : 원 궤적에서 *날조가 일어난* 결정점 (30)      → 목표: 날조 0
  CLEAN : 원 궤적에서 *정상 write* 였던 결정점 (30)    → 목표: 파손 0 (Δspurious ≤ 0)

파손(break) 정의 = CLEAN 지점에서 D''가 gold 값을 잃는 것:
   · FIND를 골랐는데 값이 gold와 다름  → break
   · 날조(FIND 값이 문맥에 없음)        → break
   · GET/INFER                          → 값을 잃지 않음(회복 가능) = break 아님
"""
import gzip
import json
import random
import sys
from collections import Counter

sys.path.insert(0, "/home/woori/scratch")
import c47_dprime as D  # noqa: E402
from e11a_isolated_probe import find_violations  # noqa: E402

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"
WRITE = {"return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
         "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
         "modify_user_address", "place_order"}
ARGS = ("new_item_ids", "payment_method_id", "address1", "item_ids", "order_id")


def clean_points(sims, n, seed=0):
    """원 궤적서 grounded 였던 write 인자 결정점 (gold=FIND/GET·NO-WRITE 제외)."""
    out = []
    for sim in sims:
        if sim.get("termination_reason") != "user_stop":
            continue
        for i, m in enumerate(sim.get("messages", [])):
            if m.get("role") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if tc.get("name") not in WRITE:
                    continue
                a = tc.get("arguments") or {}
                pre = D.prefix_txt(sim, i)
                for key in ARGS:
                    if key not in a:
                        continue
                    v = a[key]
                    v0 = v[0] if isinstance(v, list) and v else v
                    if v0 is None or D.norm(v0) not in pre:      # 원 호출이 grounded 였던 것만
                        continue
                    gold, gv = D.gold_label(sim, tc, key, i)
                    if gold not in ("FIND", "GET"):
                        continue
                    out.append((sim, i, tc, key, gv, gold))
                    break
    random.Random(seed).shuffle(out)
    return out[:n]


def run(sim, idx, tc, key, max_tries=3, fallback=True):
    fb = None
    ch, d = None, {}
    for attempt in range(max_tries):
        try:
            txt = D.chat(D.build(sim, idx, key, tc.get("name"), fb))
        except Exception:
            txt = ""
        ch, d = D.parse(txt)
        ok, msg = D.verify(ch, d, sim, idx, key)
        if ok:
            return ch, d, attempt + 1, False
        fb = (txt.strip()[:200], msg)
    if fallback:                                   # ★검증기 소진 → GET 강제
        prods = D.PRODUCER.get(key, [])
        if prods:
            return "GET", {"tool": prods[0]}, max_tries, True
    return ch, d, max_tries, False


def main():
    sims = json.load(gzip.open(SIM + "fl32b_floor_retail_t4.results.json.gz"))["simulations"]
    fab_pts = [(s, i, tc, k, None, None) for (s, i, tc, k, v, w) in find_violations(sims, 30)]
    cln_pts = clean_points(sims, 30)
    print("FAB %d · CLEAN %d" % (len(fab_pts), len(cln_pts)), flush=True)

    for name, pts in (("FAB", fab_pts), ("CLEAN", cln_pts)):
        stat = Counter()
        breaks = []
        for j, (sim, idx, tc, key, gv, gold) in enumerate(pts):
            if gold is None:
                gold, gv = D.gold_label(sim, tc, key, idx)
            ch, d, tries, forced = run(sim, idx, tc, key)
            stat["forced_GET"] += forced
            stat["tries=%d" % tries] += 1
            pre = D.prefix_txt(sim, idx)
            fab = (ch == "FIND" and D.norm(str(d.get("value", ""))) not in pre)
            wrong_copy = (ch == "FIND" and not fab and gv is not None
                          and D.norm(str(d.get("value", ""))) != D.norm(str(gv)))
            stat["FAB(날조)" if fab else ("FIND-wrong(값 오선택)" if wrong_copy else str(ch))] += 1
            stat["gold=%s" % gold] += 1
            if name == "CLEAN" and (fab or wrong_copy):
                breaks.append((str(sim.get("task_id")), sim.get("trial"), key, gold,
                               str(d.get("value"))[:34], str(gv)[:34]))
            if (j + 1) % 10 == 0:
                print("  [%s] ..%d/%d" % (name, j + 1, len(pts)), flush=True)
        n = len(pts)
        fabn = stat["FAB(날조)"]
        print("\n### %s  n=%d" % (name, n))
        print("   " + str(dict(stat)))
        print("   날조 %d (%.0f%%) · GET강제폴백 %d" % (fabn, 100.0 * fabn / max(n, 1), stat["forced_GET"]))
        if name == "CLEAN":
            print("   ★Δspurious: 파손 %d / %d = %.1f%%" % (len(breaks), n, 100.0 * len(breaks) / max(n, 1)))
            for b in breaks:
                print("      t%s tr%s %s gold=%s  D''값=%r  gold값=%r" % b)


if __name__ == "__main__":
    main()
