#!/usr/bin/env bash
# D0 sweep: assumption fixes (가)(나)(마) applied, Qwen2.5-7B-Instruct
# query_mode=native, gt_mode=unique, +nl_sysless condition
# Output: reports/facet_rft_2026/step1_d0/

set -euo pipefail

PYTHON=/home/woori/venvs/seka_env/bin/python3.12
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRIPT=$REPO/scripts/rank_replaceability/facet_eval.py
OUT=$REPO/reports/facet_rft_2026/step1_d0
MODEL=Qwen/Qwen2.5-7B-Instruct
CONDITIONS=nl_full,nl_with_desc,facet_full,facet_compact,list_only,nl_sysless,list_anon,facet_anon,noprompt

mkdir -p "$OUT"
LOG=$OUT/sweep.log
exec > >(tee -a "$LOG") 2>&1

echo "=== D0 sweep start $(date) ==="
echo "query_mode=native  gt_mode=unique  +nl_sysless"

# --- retail N=114 ---
echo "[1/3] retail N=114 $(date)"
$PYTHON $SCRIPT \
    --model $MODEL \
    --schema $REPO/data/facet_schemas/tau2_retail.yaml \
    --task tau2_retail \
    --max-samples 114 \
    --device cuda:0 \
    --conditions $CONDITIONS \
    --d0 \
    --out $OUT/qwen_retail_n114_d0.json

# --- telecom N=256 ---
echo "[2/3] telecom N=256 $(date)"
$PYTHON $SCRIPT \
    --model $MODEL \
    --schema $REPO/data/facet_schemas/tau2_telecom.yaml \
    --task tau2_telecom \
    --max-samples 256 \
    --device cuda:0 \
    --conditions $CONDITIONS \
    --d0 \
    --out $OUT/qwen_telecom_n256_d0.json

# --- airline N=50 ---
echo "[3/3] airline N=50 $(date)"
$PYTHON $SCRIPT \
    --model $MODEL \
    --schema $REPO/data/facet_schemas/tau2_airline.yaml \
    --task tau2_airline \
    --max-samples 50 \
    --device cuda:0 \
    --conditions $CONDITIONS \
    --d0 \
    --out $OUT/qwen_airline_n50_d0.json

echo "=== D0 sweep complete $(date) ==="

# Summary
$PYTHON - <<'PYEOF'
import json, os, glob

OUT = "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/step1_d0"
conds = ["nl_full", "nl_with_desc", "facet_full", "facet_compact", "list_only", "nl_sysless", "list_anon", "facet_anon"]

for fpath in sorted(glob.glob(f"{OUT}/qwen_*.json")):
    with open(fpath) as f:
        d = json.load(f)
    task = d.get("task", os.path.basename(fpath))
    n = d.get("n_samples", "?")
    print(f"\n=== {task} N={n} (D0: qmode={d.get('d0_query_mode')} gtmode={d.get('d0_gt_mode')}) ===")
    s = d.get("summary", {})
    print(f"  {'cond':<20} {'F1':>6} {'[lo':>6} {'hi]':>6} {'tok':>5}")
    for c in conds:
        if c not in s:
            continue
        cs = s[c]
        print(f"  {c:<20} {cs['f1_mean']:>6.3f} {cs['f1_ci95_lo']:>6.3f} {cs['f1_ci95_hi']:>6.3f} {cs['prompt_len_mean']:>5.0f}")
PYEOF
