#!/usr/bin/env bash
# V-bias smoke test on Subtask4 N=20 (Qwen-Instruct).
# Waits for Mistral-Instruct Wave 3b (PID 810304) to finish on GPU1, then runs.
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env

MODEL="Qwen/Qwen2.5-7B-Instruct"
BONT="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
LOG_DIR=logs/vbias_smoke
OUT_DIR=reports/vbias_smoke
mkdir -p "$LOG_DIR" "$OUT_DIR"

# --- Wait for Mistral-Instruct to finish to free up GPU1 memory ---
WAIT_PID=810304
echo "[vbias] waiting for Mistral-Instruct PID $WAIT_PID at $(date)" >> "$LOG_DIR/chain.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
echo "[vbias] Mistral-Instruct exited at $(date)" >> "$LOG_DIR/chain.log"

# --- V-bias-only smoke N=20 ---
# Test: does V-side alone do anything on multi-tool?
for AV in 0.1 0.3 0.5; do
  python scripts/ocq/eval_metatool_subtask4.py \
    --model "$MODEL" --device cuda:1 \
    --methods "no_steer" "ocq_vbias_a${AV}" \
    --b-ont "$BONT" \
    --max-samples 20 \
    --out "$OUT_DIR/smoke_vbias_av${AV}.json" \
    > "$LOG_DIR/smoke_vbias_av${AV}.log" 2>&1
done

# --- K+V combined smoke N=20 ---
# Test: KQV hybrid intuition (K marks facet, V amplifies content)
for AK in 0.1 0.3; do
  for AV in 0.1 0.3 0.5; do
    python scripts/ocq/eval_metatool_subtask4.py \
      --model "$MODEL" --device cuda:1 \
      --methods "no_steer" "ocq_kvbias_a${AK}_v${AV}" \
      --b-ont "$BONT" \
      --max-samples 20 \
      --out "$OUT_DIR/smoke_kvbias_ak${AK}_av${AV}.json" \
      > "$LOG_DIR/smoke_kvbias_ak${AK}_av${AV}.log" 2>&1
  done
done

echo "[vbias] all smoke done at $(date)" >> "$LOG_DIR/chain.log"

# Summary
python3 -c "
import json, glob
print('=== V-bias smoke summary ===')
for p in sorted(glob.glob('$OUT_DIR/*.json')):
    d = json.load(open(p))
    for r in d['results']:
        m = r['macro']
        print(f\"{p.split('/')[-1]:40s} {r['method']:28s} F1={m['F1']:.3f} F0.5={m['F_0.5']:.3f} EU={m['EU']:.3f} Jac={m['Jaccard']:.3f} Exact={m['Exact']:.3f} rec={m['recall']:.3f}\")
" >> "$LOG_DIR/summary.log" 2>&1
