#!/usr/bin/env bash
# GPU1 PM wave 2026-04-15 — runs after current Q-coverage full sweep (PID 1885654)
# Order: Q-coverage extended sweep → null-controls → Subtask1 full → R6 ext →
#        Mistral-Instruct null-control → Llama Q-coverage smoke
set -u

REPO=/home/woori/workspace_common/boltzmann-attention
cd "$REPO"
source /home/woori/workspace_common/CDP/poc/set.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEV="cuda:1"
QWEN="Qwen/Qwen2.5-7B-Instruct"
LLAMA="NousResearch/Meta-Llama-3.1-8B-Instruct"
MISTRAL_INST="mistralai/Mistral-7B-Instruct-v0.3"
BONT_QWEN="external/SEKA/seka_projections/ontology-qwen25-7b-metatool/B_ont.pt"
BONT_QWEN_RAND="external/SEKA/seka_projections/ontology-qwen25-7b-metatool-random/B_ont.pt"
BONT_QWEN_SHUF="external/SEKA/seka_projections/ontology-qwen25-7b-metatool-featshuffle/B_ont.pt"
BONT_LLAMA="external/SEKA/seka_projections/ontology-llama31-8b-metatool/B_ont.pt"
BONT_MISTRAL_PAD="external/SEKA/seka_projections/ontology-mistral-7b-v03-metatool-skipL0-padmax/B_ont.pt"

LOG=logs/wave_2026_04_15_pm/gpu1
OUT=reports/wave_2026_04_15_pm/gpu1
mkdir -p "$LOG" "$OUT"

WAIT_PID=1885654
echo "[g1] waiting for PID $WAIT_PID at $(date)" >> "$LOG/chain.log"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[g1] $WAIT_PID done at $(date), starting wave" >> "$LOG/chain.log"

run_st4() {
  local TAG=$1; local METHODS=$2; local BONT=$3; local MODEL=$4; local SAMPLES=${5:-0}
  local extra=""
  if [ "$SAMPLES" != "0" ]; then extra="--max-samples $SAMPLES"; fi
  echo "[g1] $TAG start $(date)" >> "$LOG/chain.log"
  python scripts/ocq/eval_metatool_subtask4.py \
      --model "$MODEL" --device "$DEV" \
      --methods $METHODS --b-ont "$BONT" $extra \
      --out "$OUT/$TAG.json" \
      > "$LOG/$TAG.log" 2>&1
  echo "[g1] $TAG done $(date) rc=$?" >> "$LOG/chain.log"
}

run_st1() {
  local TAG=$1; local METHODS=$2; local BONT=$3; local MODEL=$4
  echo "[g1] $TAG start $(date)" >> "$LOG/chain.log"
  python scripts/ocq/eval_metatool_subtask1.py \
      --model "$MODEL" --device "$DEV" \
      --methods $METHODS --b-ont "$BONT" \
      --max-new-tokens 32 \
      --out "$OUT/$TAG.json" \
      > "$LOG/$TAG.log" 2>&1
  echo "[g1] $TAG done $(date) rc=$?" >> "$LOG/chain.log"
}

# === Wave 1: Q-coverage extended β sweep on Subtask4 full 497 ===
run_st4 "qwen_st4_qbias_extended" \
        "no_steer ocq_qbias_b-0.05 ocq_qbias_b-0.15 ocq_qbias_b-0.2 ocq_qbias_b-0.7" \
        "$BONT_QWEN" "$QWEN"

# === Wave 2: Q-coverage NULL-CONTROL falsifiability test (Thm 6.17 specificity) ===
# If Q-coverage works with random/featshuffle B_ont too → Q-coverage is generic
# (subspace-projection) not ontology-specific; downgrades the contribution.
run_st4 "qwen_st4_qbias_b-0.1_random_bont" \
        "no_steer ocq_qbias_b-0.1" \
        "$BONT_QWEN_RAND" "$QWEN"

run_st4 "qwen_st4_qbias_b-0.1_featshuffle_bont" \
        "no_steer ocq_qbias_b-0.1" \
        "$BONT_QWEN_SHUF" "$QWEN"

# === Wave 3: QKV at small magnitudes (probe α_coupling boundary) ===
run_st4 "qwen_st4_qkv_small_smoke" \
        "no_steer ocq_qkv_a0.05_v0_q-0.1 ocq_qkv_a0.1_v0.1_q-0.1 ocq_qkv_a0.05_v0.05_q-0.05 ocq_qkv_a0_v0.1_q-0.1 ocq_qkv_a0.1_v0_q-0.1" \
        "$BONT_QWEN" "$QWEN" 20

# === Wave 4: Q-coverage on Subtask1 full 995 (single-tool effect check) ===
run_st1 "qwen_st1_qbias_full995" \
        "no_steer ocq_qbias_b-0.1 ocq_qbias_b-0.3 ocq_bias_a0.3" \
        "$BONT_QWEN" "$QWEN"

# === Wave 5: Mistral-Instruct null-control on Subtask4 (chat-template hedging diagnosis) ===
# Tests whether Mistral-Instruct null-control still produces +60pp gap (would
# isolate the −2.92pp single-tool result as hedging, not mechanism failure).
run_st4 "mistral_inst_st4_kbias_smoke" \
        "no_steer ocq_bias_a0.3" \
        "$BONT_MISTRAL_PAD" "$MISTRAL_INST" 20

# === Wave 6: Llama-3.1-8B-Instruct Subtask4 cross-model verification ===
# Smoke first; if positive, queue full 497 in next wave manually.
run_st4 "llama_inst_st4_kbias_qbias_smoke" \
        "no_steer ocq_bias_a0.3 ocq_qbias_b-0.1" \
        "$BONT_LLAMA" "$LLAMA" 20

# === Wave 7: R6 MMLU expand (more α cells around 0.2 sweet-spot) ===
echo "[g1] r6_extend start $(date)" >> "$LOG/chain.log"
for ALPHA in 0.15 0.25 0.4; do
  python scripts/ocq/eval_metatool_subtask1.py \
      --model "$QWEN" --device "$DEV" \
      --dataset /tmp/MetaTool/dataset/mmlu_subset_n1000.json 2>/dev/null \
      --methods no_steer ocq_bias_a${ALPHA} \
      --b-ont "$BONT_QWEN" \
      --out "$OUT/r6_flat_a${ALPHA}.json" \
      > "$LOG/r6_flat_a${ALPHA}.log" 2>&1 || true
done
echo "[g1] r6_extend done $(date)" >> "$LOG/chain.log"

# === Final summary ===
python3 -c "
import json, glob, os
print('=== GPU1 PM wave summary ===')
for p in sorted(glob.glob('$OUT/*.json')):
    try:
        d = json.load(open(p))
        for r in d.get('results', []):
            m = r.get('macro', r)
            if isinstance(m, dict) and 'F1' in m:
                print(f\"{os.path.basename(p):45s} {r['method']:30s} F1={m['F1']:.3f} rec={m.get('recall',0):.3f}\")
            elif 'top1_accuracy' in r:
                print(f\"{os.path.basename(p):45s} {r['method']:30s} top1={r['top1_accuracy']*100:.2f}%\")
            elif 'accuracy' in r:
                print(f\"{os.path.basename(p):45s} {r['method']:30s} acc={r['accuracy']*100:.2f}%\")
    except Exception as e:
        print(f'{p}: ERROR {e}')
" >> "$LOG/SUMMARY.txt" 2>&1
echo "[g1] WAVE COMPLETE $(date)" >> "$LOG/chain.log"
