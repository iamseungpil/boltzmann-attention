#!/usr/bin/env python
"""ReST S1 replay 믹스 — retail keep 궤적 + 일반 tool-use replay → forgetting-통제 학습셋.

설계=REST_INTERNALIZE_DESIGN §5: narrow-only가 fact_*/solo_* 파탄 원인 → replay 혼합.
- keep = sft_rest_s0_retail.jsonl(실 retail discipline·318).
- replay = 일반 tool-use(tbnfc daily + tb_all·비-retail·일반능력 보존). airline은 *제외*(전이/held-out 보존).
- supervise="assistant" 통일(replay 원본엔 없음).
Usage: rest_s0_mix_replay.py <retail.jsonl> <out.jsonl> <general_per_retail> [seed]
"""
import json, sys, random

RETAIL = sys.argv[1]
OUT = sys.argv[2]
RATIO = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0   # general per 1 retail
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
POOL = ["/home/woori/scratch/fc_build/tbnfc_lodo_daily_train.jsonl",
        "/home/woori/scratch/fc_build/tb_all_v4.jsonl"]

random.seed(SEED)

def load(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

retail = load(RETAIL)
for r in retail:
    r.setdefault("supervise", "assistant")

pool = []
for p in POOL:
    recs = load(p)
    for r in recs:
        r["supervise"] = "assistant"      # 통일
        r.setdefault("_meta", {})
        r["_meta"]["replay"] = True
    pool.extend(recs)
random.shuffle(pool)

n_general = int(len(retail) * RATIO)
general = pool[:n_general]

mixed = retail + general
random.shuffle(mixed)
with open(OUT, "w") as f:
    for r in mixed:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[mix] retail={len(retail)} + general_replay={len(general)} (ratio 1:{RATIO}) "
      f"= {len(mixed)} total → {OUT}")
print(f"[mix] general pool available={len(pool)} · seed={SEED}")
