#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""nlnum_offline_census.py — NL-NUM-PROV V0 (무료·오프라인) 가상-fire census.

레버1(t47 처방) 착수 게이트: comp gz 456 전수 replay —
  각 assistant 발화에 대해 술어(_unverified_amounts·라이브와 동일 코드)를
  이전 문맥(user/tool 텍스트·콤마 정규화) 기준으로 판정(개입 없음·관측만).
보고: passing sim(reward>=1)에서 발화했을 뻔한 수(over-fire) vs failing sim.
      + t5c_aprime1 t47 궤적서 $928.13 포착 재현.

usage: PYTHONIOENCODING=utf-8 py -3 nlnum_offline_census.py --results <gz> [--task N]
"""
import argparse, gzip, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 술어 = 라이브 코드 그대로 임포트 (census-라이브 동형성 보장)
from t2_gate_patch import _unverified_amounts  # noqa: E402


def load_sims(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)["simulations"]


def virtual_fires(sim):
    """궤적 1건 replay: [(msg_idx, [amounts])] — assistant 발화별 미검증 금액.
    ctx = 이전 user/tool 텍스트만(라이브 _ctx_from_messages 동형·assistant 제외)."""
    ctx_parts = []
    fires = []
    for i, m in enumerate(sim.get("messages") or []):
        role, c = m.get("role"), m.get("content")
        if role == "assistant" and isinstance(c, str) and c.strip():
            ctx_num = " ".join(ctx_parts).lower().replace(",", "")
            bad = _unverified_amounts(c, ctx_num)
            if bad:
                fires.append((i, bad))
        if role in ("user", "tool") and c is not None:
            ctx_parts.append(c if isinstance(c, str) else str(c))
    return fires


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--task", default=None, help="단일 task 상세 출력")
    ap.add_argument("--show", type=int, default=6, help="passing-fire 샘플 출력 수")
    a = ap.parse_args()
    sims = load_sims(a.results)

    if a.task is not None:
        for s in [x for x in sims if str(x.get("task_id")) == str(a.task)]:
            r = (s.get("reward_info") or {}).get("reward")
            fires = virtual_fires(s)
            print("task %s trial %s reward=%s : %d firing utterance(s)"
                  % (s.get("task_id"), s.get("trial"), r, len(fires)))
            for i, bad in fires:
                snippet = (s["messages"][i].get("content") or "")[:160].replace("\n", " ")
                print("   msg[%d] %s | %r" % (i, bad, snippet))
        return

    stats = {True: Counter(), False: Counter()}   # key=passing
    firing_pass_sims = []
    n_pass = n_fail = 0
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        if r is None:
            continue
        ok = r >= 1
        n_pass += ok
        n_fail += (not ok)
        fires = virtual_fires(s)
        if fires:
            stats[ok]["sims"] += 1
            stats[ok]["utterances"] += len(fires)
            stats[ok]["amounts"] += sum(len(b) for _, b in fires)
            if ok:
                firing_pass_sims.append((s, fires))
    print("== NL-NUM-PROV virtual-fire census: %s ==" % os.path.basename(a.results))
    print("sims: pass=%d fail=%d" % (n_pass, n_fail))
    for ok, tag in ((True, "PASSING (over-fire 상한)"), (False, "FAILING (표적 포함)")):
        c = stats[ok]
        denom = n_pass if ok else n_fail
        print("  %s: firing sims=%d/%d (%.1f%%) · firing utterances=%d · amounts=%d"
              % (tag, c["sims"], denom, 100.0 * c["sims"] / max(denom, 1),
                 c["utterances"], c["amounts"]))
    print("\n-- passing-sim fire 샘플 (성격 판정용·최대 %d) --" % a.show)
    for s, fires in firing_pass_sims[:a.show]:
        i, bad = fires[0]
        snippet = (s["messages"][i].get("content") or "")[:200].replace("\n", " ")
        print("task %s trial %s: %s | %r" % (s.get("task_id"), s.get("trial"), bad, snippet))


if __name__ == "__main__":
    main()
