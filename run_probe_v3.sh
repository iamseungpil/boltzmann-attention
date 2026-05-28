#!/bin/bash
# Phase 0 v3 probing — 12종 관계, A7 실증 비교 데이터 생성
cd /home/woori/workspace_common/boltzmann-attention-pi
PYTHON=/home/woori/venvs/seka_env/bin/python3.12
REPORT=reports/facet_rft_2026/phase0_probing
LOG=$REPORT/probe_v3_run.log

echo "=== Phase 0 v3 시작: $(date) ===" | tee "$LOG"

# 1. 온톨로지 v2 서버로 복사
cp scripts/ontology/tau2_telecom_ontology.py scripts/ontology/tau2_telecom_ontology_v2.py 2>/dev/null || true

# 2. 대조쌍 v3 생성
echo "--- 대조쌍 v3 생성 ---" | tee -a "$LOG"
$PYTHON scripts/ontology/generate_contrast_pairs_v3.py \
  --tau2  /home/woori/workspace_common/boltzmann-attention/external/tau2-bench/data/tau2/domains/telecom/tasks.json \
  --n-binary 12 \
  --n-unary  25 \
  --out   $REPORT/contrast_pairs_v3.json \
  2>&1 | tee -a "$LOG"

# 3. Probing v3 실행
echo "--- Probing v3 실행 ---" | tee -a "$LOG"
$PYTHON scripts/ontology/probe_ontology_v3.py \
  --pairs  $REPORT/contrast_pairs_v3.json \
  --model  Qwen/Qwen2.5-7B-Instruct \
  --device cuda:0 \
  --batch-size 12 \
  --max-length 300 \
  --out-dir $REPORT/ \
  2>&1 | tee -a "$LOG"

echo "=== 완료: $(date) ===" | tee -a "$LOG"
