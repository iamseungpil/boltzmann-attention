#!/usr/bin/env python3
"""P1 step-0 prereg (TB_SCALE v3 §2-P1): read the 32B base census, predict held-out
MM Δedge from the v3 mechanism, and freeze the prediction BEFORE training starts.

Mechanism (결과문서 §8): Δ ≈ index-correction gain − vocab interference(~5pp),
gain ∝ base self-reference rate (nself/ex). Anchors (net Δ, sub500 same-id):
  7B  nself 0.218 -> Δ −0.8  => gain  4.2  (slope 19.3)
  14B nself 0.478 -> Δ +4.3  => gain  9.3  (slope 19.5)
The two anchors give a near-identical slope, so we use their mean (19.4) through
the origin: Δpred(32B) = 19.4 * nself_32B − 5.0   (edge-F1 points).

Usage: tb_p1_prereg.py <census_md> <out_json>
"""
import json
import sys
from datetime import datetime, timezone

INTERFERENCE_PP = 5.0
ANCHORS = {"7B": {"nself": 0.218, "delta": -0.8}, "14B": {"nself": 0.478, "delta": 4.3}}

census_path, out_path = sys.argv[1], sys.argv[2]
lines = open(census_path).read().splitlines()
agg = None
for i, l in enumerate(lines):
    if l.startswith("## aggregate"):
        assert lines[i + 1].startswith("A: ")
        agg = json.loads(lines[i + 1][3:])
        break
assert agg is not None, "aggregate block not found in census output"

slopes = [(a["delta"] + INTERFERENCE_PP) / a["nself"] for a in ANCHORS.values()]
slope = sum(slopes) / len(slopes)
nself = agg["nself"]
pred = slope * nself - INTERFERENCE_PP

out = {
    "experiment": "P1 gold-SFT LODO_mm @ Qwen2.5-32B (COWORKER_REQUEST_TB_SCALE v3)",
    "registered_utc": datetime.now(timezone.utc).isoformat(),
    "base_census_aggregate": agg,
    "mechanism": "delta_edge ~= slope*nself - interference (결과문서 §8)",
    "anchors": ANCHORS,
    "slope_per_anchor": dict(zip(ANCHORS, slopes)),
    "slope_used": slope,
    "interference_pp": INTERFERENCE_PP,
    "nself_32b_base": nself,
    "valid_frac_32b_base": agg.get("valid_frac"),
    "predicted_delta_edge_mm_sub500": round(pred, 2),
    "verdict_rule": ("①pred 적중=기제 확립 ②초과=인덱스 외 추가 전이(용량 부분 부활) "
                     "③미달=32B 고유 변수 재조사 (v3 §2-P1)"),
}
json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
print(f"[prereg] nself={nself:.3f} -> predicted Δedge(MM sub500) = {pred:+.2f}  -> {out_path}")
