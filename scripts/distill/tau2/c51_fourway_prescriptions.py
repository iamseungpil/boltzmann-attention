#!/usr/bin/env python3
"""C51 — 출처 선언 4지선다(GET/FIND/INFER/ASK)를 세 처방으로 얼마나 여는가.

정본 실험 (설계서 = reports/facet_rft_2026/FOURWAY_PRESCRIPTION_COMPARISON_DESIGN_2026_07_11.md).
선행 = C42(짧은 clean이면 base도 완벽·gradient 0) · C43(정박치환) · C44/C45/C47/C48(출처선언+검증기 67->0%).

결정점 = write 직전 인자 하나. "이 인자값의 출처는? {GET·FIND·INFER·ASK}" + 갈래별 실행.
모집단(둘 다 tau2 retail 실 궤적 fl32b_floor·in-vivo·오염):
  FAB   : 원 궤적에서 *날조가 난* 결정점 30 (긴 문맥·근접-오답 정박·C43)
  CLEAN : 원 궤적에서 *grounded* 였던 결정점 30 (Δspurious 대조)

세 처방 arm (+base) — 엔진·A2(PRODUCER 매핑)는 세 arm 공통([[05]]):
  base   : 4지선다만 제시 · 단일 호출 · 검증기/재발화 없음 (c47.build 1회)
  prompt : base + 강화 규칙문 + 도메인-일반 예시 · 단일 호출 · 검증기 없음 ([[42]] 천장 검정)
  loop   : 결정론 controller — producer 있으면 ASK 금지·FIND는 문맥실재 검증·소진 시 GET 강제 (c48.run = D'')
  learn  : 설계서 전용 (유료/데이터 게이트 · C42 무-gradient → D7 정박합성 타당성 게이트 선통과 필요)

지표 (arm별):
  (a) 4지선다 정확도 : 선언한 출처 == gold 라벨  (판정가능 = gold in {GET,FIND,ASK})
  (b) 최종 인자 : 날조율(FIND 값이 문맥에 없음) · FIND-exact(값==gold) · FIND-wrong(문맥엔 있으나 !=gold=⋈ 경계)
  (c) GET-before-ASK 준수 : producer 존재 지점서 ASK 를 *안* 고른 비율 (t17형 누수 측정)

JSONL(per-case)은 리모트에만 영속 (스크립트만 커밋). 규율 [[08]]/[[05]]/[[09]].

Run (리모트):
  python3 c51_fourway_prescriptions.py --arm base,prompt,loop --n 30 \
      --out /home/woori/.../sim_results/c51_fourway_prescriptions.jsonl
"""
import argparse
import gzip
import json
import sys
from collections import Counter

sys.path.insert(0, "/home/woori/scratch")
sys.path.insert(0, "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2")

import c47_dprime as D            # noqa: E402  build/parse/verify/gold_label/chat/PRODUCER/prefix_txt/norm/obtainable
import c48_dprime_full as F       # noqa: E402  clean_points/run(=loop)/WRITE
from e11a_isolated_probe import find_violations  # noqa: E402

SIM = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/sim_results/"

# prompt arm — 강화 규칙 + 도메인-일반 예시 (retail gold 값 주입 0 · [[05]]).
STRONG_RULE = (
    "\n\nCRITICAL SOURCING RULES — apply before choosing:\n"
    "1. NEVER invent, guess, or copy an example/placeholder value for an argument.\n"
    "2. If ANY tool can return this value (a getter/lookup exists), you MUST choose GET. "
    "Never ASK the user for something the system can look up.\n"
    "3. Choose FIND only when the exact value already appears verbatim in the user's words "
    "or in a previous tool output above.\n"
    "4. Choose ASK only as a LAST RESORT: no tool can produce it and it is not in the context.\n"
    "Illustrative (not domain-specific):\n"
    "  - You need an internal record id you don't have yet, but a lookup tool returns it -> GET.\n"
    "  - The user already stated the value earlier -> FIND.\n"
    "  - The value must be derived from other values above (e.g. the cheapest option) -> INFER.\n"
    "  - Only the user knows it and no tool returns it (a value they have not stated) -> ASK.\n"
)


def build_base(sim, idx, key, tcname):
    return D.build(sim, idx, key, tcname)          # AI + policy + 히스토리 + 4지선다, 단일


def build_prompt(sim, idx, key, tcname):
    msgs = D.build(sim, idx, key, tcname)
    msgs[0]["content"] = msgs[0]["content"] + STRONG_RULE   # system 에 강화 규칙 주입
    return msgs


def classify_value(ch, d, gv, prefix):
    """(b) 최종 인자 분류."""
    if ch == "FIND":
        v = D.norm(str(d.get("value", "")))
        if not v or v not in prefix:
            return "fab"                     # 문맥에 없는 값 = 날조
        if gv is not None and v == D.norm(str(gv)):
            return "find_exact"
        return "find_wrong"                  # 문맥엔 있으나 gold 아님 = ⋈ 오선택
    if ch == "GET":
        return "get"
    if ch == "INFER":
        return "infer"
    if ch == "ASK":
        return "ask"
    return "none"


def run_point(arm, sim, idx, tc, key):
    """arm 정책 실행 → (choice, detail, tries, forced_GET)."""
    tcname = tc.get("name")
    if arm == "loop":
        ch, d, tries, forced = F.run(sim, idx, tc, key)   # 검증기+재발화+GET강제폴백
        return ch, d, tries, forced
    # base / prompt = 단일 호출
    builder = build_base if arm == "base" else build_prompt
    try:
        txt = D.chat(builder(sim, idx, key, tcname))
    except Exception:
        txt = ""
    ch, d = D.parse(txt)
    return ch, d, 1, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="base,prompt,loop")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tag", default="fl32b_floor_retail_t4")
    ap.add_argument("--out", default=SIM + "c51_fourway_prescriptions.jsonl")
    a = ap.parse_args()
    arms = a.arm.split(",")

    sims = json.load(gzip.open(SIM + a.tag + ".results.json.gz"))["simulations"]
    fab_pts = [(s, i, tc, k) for (s, i, tc, k, v, w) in find_violations(sims, a.n)]
    cln_raw = F.clean_points(sims, a.n)
    cln_pts = [(s, i, tc, k) for (s, i, tc, k, v, w) in cln_raw]
    print("FAB %d · CLEAN %d · arms=%s" % (len(fab_pts), len(cln_pts), arms), flush=True)

    fout = open(a.out, "w", encoding="utf-8")
    # agg[pop][arm] = Counter
    agg = {p: {arm: Counter() for arm in arms} for p in ("FAB", "CLEAN")}
    per_case = {p: [] for p in ("FAB", "CLEAN")}

    for pop, pts in (("FAB", fab_pts), ("CLEAN", cln_pts)):
        for j, (sim, idx, tc, key) in enumerate(pts):
            gold, gv = D.gold_label(sim, tc, key, idx)
            prefix = D.prefix_txt(sim, idx)
            has_producer = bool(D.PRODUCER.get(key))
            row = {"pop": pop, "task": str(sim.get("task_id")), "trial": sim.get("trial"),
                   "arg": key, "gold": gold, "gold_val": (str(gv)[:40] if gv is not None else None),
                   "has_producer": has_producer}
            for arm in arms:
                ch, d, tries, forced = run_point(arm, sim, idx, tc, key)
                vclass = classify_value(ch, d, gv, prefix)
                payload = str(d.get("tool") or d.get("value") or d.get("question") or "")[:50]
                row[arm] = {"choice": ch, "vclass": vclass, "tries": tries,
                            "forced_GET": forced, "payload": payload}
                c = agg[pop][arm]
                c["n"] += 1
                if gold in ("GET", "FIND", "ASK"):
                    c["judgeable"] += 1
                    if ch == gold:
                        c["acc_correct"] += 1
                c[vclass] += 1
                c["choice_" + str(ch)] += 1
                if has_producer:
                    c["prod_pts"] += 1
                    if ch != "ASK":
                        c["get_before_ask_ok"] += 1
                c["forced_GET"] += int(forced)
                c["tries_sum"] += tries
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            per_case[pop].append(row)
            if (j + 1) % 10 == 0:
                print("  [%s] ..%d/%d" % (pop, j + 1, len(pts)), flush=True)
    fout.close()

    # ---- 집계표 ----
    for pop in ("FAB", "CLEAN"):
        print("\n================ %s (n=%d) ================" % (pop, agg[pop][arms[0]]["n"]))
        hdr = ("%-7s %6s %6s %6s %8s %9s %9s %9s %7s %6s" %
               ("arm", "acc(a)", "fab", "GET", "ASK", "find_ex", "find_wr", "getbfAsk", "forced", "tries"))
        print(hdr)
        for arm in arms:
            c = agg[pop][arm]
            n = c["n"] or 1
            judg = c["judgeable"] or 1
            acc = c["acc_correct"] / judg
            prodp = c["prod_pts"] or 1
            gba = c["get_before_ask_ok"] / prodp
            print("%-7s %6.2f %6d %6d %8d %9d %9d %9.2f %7d %6.1f" %
                  (arm, acc, c["fab"], c["choice_GET"], c["choice_ASK"],
                   c["find_exact"], c["find_wrong"], gba, c["forced_GET"],
                   c["tries_sum"] / n))
        print("  (acc=판정가능 %d건 기준 · fab/GET/ASK/find_* = 선택 건수 · getbfAsk=producer지점서 non-ASK 비율)"
              % agg[pop][arms[0]]["judgeable"])

    # ---- per-case 뒤집힘 (base vs loop) ----
    if "base" in arms and "loop" in arms:
        print("\n=== per-case: base→loop 뒤집힌 결정 (FAB) ===")
        for r in per_case["FAB"]:
            b, lp = r["base"], r["loop"]
            if b["choice"] != lp["choice"] or b["vclass"] != lp["vclass"]:
                print("  t%s tr%s %-16s gold=%-5s | base=%-5s/%-10s -> loop=%-5s/%-10s (forced=%s tries=%s)"
                      % (r["task"], r["trial"], r["arg"], r["gold"],
                         b["choice"], b["vclass"], lp["choice"], lp["vclass"],
                         lp["forced_GET"], lp["tries"]))
    if "base" in arms and "prompt" in arms:
        print("\n=== per-case: base vs prompt 차이 (FAB · [[42]] 천장 점검) ===")
        diff = 0
        for r in per_case["FAB"]:
            b, p = r["base"], r["prompt"]
            if b["choice"] != p["choice"] or b["vclass"] != p["vclass"]:
                diff += 1
                print("  t%s tr%s %-16s gold=%-5s | base=%-5s/%-9s -> prompt=%-5s/%-9s"
                      % (r["task"], r["trial"], r["arg"], r["gold"],
                         b["choice"], b["vclass"], p["choice"], p["vclass"]))
        print("  base→prompt 바뀐 결정 %d / %d" % (diff, len(per_case["FAB"])))

    print("\nJSONL -> %s" % a.out)


if __name__ == "__main__":
    main()
