# -*- coding: utf-8 -*-
"""x499 — 반증 7종의 집계. **승격 가능한 것만** 남긴다([[31]] 규칙 6).

원인-스텝 포렌식(Trace 8)의 반증 팔 산출을 한 표로 모은다:
  · 우리-층 주장 판정(CONFIRMED / REFUTED / PLAUSIBLE)
  · 닫힌-술어 후보의 CLOSED·TARGET·COST 3답
  · 반증자가 **새로 찾은** 우리-층 결손(분석가가 과소 귀속한 것)
  · 못 가른 것(다음 계측 대상)

⚠집계는 원인을 말하지 않는다([[08]]) — 이 표는 **무엇이 반증을 통과했나**의 명부일 뿐이고,
   각 항목의 근거는 원본 `refute_*.json` 의 `why` 축자에 있다.

    py -3 x499_refute_rollup.py <refute_dir>
"""
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                   "x499_refute_rollup.json")


def norm_verdict(v):
    v = str(v or "").upper()
    for k in ("REFUTED", "CONFIRMED", "PLAUSIBLE"):
        if k in v:
            return k
    return "UNKNOWN"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(d, "refute_*.json")))
    if not files:
        print("refute_*.json 없음: %s" % d)
        return 1
    roll = {"sources": [], "by_verdict": {}, "traces": []}
    tot = {"CONFIRMED": 0, "REFUTED": 0, "PLAUSIBLE": 0, "UNKNOWN": 0}
    for f in files:
        r = json.load(io.open(f, encoding="utf-8"))
        roll["sources"].append(os.path.basename(f))
        vs = r.get("verdicts") or []
        cnt = {"CONFIRMED": 0, "REFUTED": 0, "PLAUSIBLE": 0, "UNKNOWN": 0}
        for v in vs:
            k = norm_verdict(v.get("verdict"))
            cnt[k] += 1
            tot[k] += 1
        roll["traces"].append({
            "index": r.get("index"), "tool": r.get("tool"),
            "counts": cnt,
            "cause": r.get("cause"),
            "surviving_our_layer": r.get("surviving_our_layer") or [],
            "surviving_predicates": r.get("surviving_predicates") or [],
            "verdicts": vs,
            "chain_note": r.get("chain_note"),
        })
        print("=" * 78)
        print("TRACE %s  %s" % (r.get("index"), str(r.get("tool"))[:56]))
        print("  판정  CONFIRMED %d · REFUTED %d · PLAUSIBLE %d"
              % (cnt["CONFIRMED"], cnt["REFUTED"], cnt["PLAUSIBLE"]))
        for s in (r.get("surviving_our_layer") or []):
            print("  [생존·우리층] %s" % str(s).replace("\n", " ")[:150])
        for s in (r.get("surviving_predicates") or []):
            print("  [생존·술어]  %s" % str(s).replace("\n", " ")[:150])
    roll["by_verdict"] = tot
    print("\n" + "=" * 78)
    print("합계  CONFIRMED %d · REFUTED %d · PLAUSIBLE %d · UNKNOWN %d  (trace %d)"
          % (tot["CONFIRMED"], tot["REFUTED"], tot["PLAUSIBLE"], tot["UNKNOWN"], len(files)))
    print("생존 우리-층 %d · 생존 술어 %d"
          % (sum(len(t["surviving_our_layer"]) for t in roll["traces"]),
             sum(len(t["surviving_predicates"]) for t in roll["traces"])))
    io.open(OUT, "w", encoding="utf-8").write(json.dumps(roll, ensure_ascii=False, indent=1) + "\n")
    print("→ %s" % os.path.basename(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
