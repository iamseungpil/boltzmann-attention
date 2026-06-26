#!/bin/bash
# node_run_a2_census_p4.sh — P4 (요청서 §10): A2-컴파일러 크기 하한 census, 추론-only.
# arms (사전예측 §10 표에 동결, 2026-06-12):
#   P4a base-72B           TP4  예측 gate_recall ≥0.8 ∧ applies_F1 ≥0.7 (Fable-5 reference 근접)
#   P4b base-32B           TP2  예측 중간대 0.5~0.8 = 하한 후보 구간
#   P4c base-235B-A22B-INT4 TP4 예측 ≈72B ±0.05 (크기 포화 통제; non-thinking 고정)
# 입력 전부 repo 내장 (specs/: airline_policy.md·airline_tool_catalog.json·*_gate_spec_fable5.json)
# — 외부 클론·다운로드 불요. 7B/14B는 Track A가 로컬 수행.
# 멱등: per-arm done-marker. 산출물 = 생성 spec JSON(원본) + 점수 stdout → trackb_raw push.
set -x
export HF_HUB_CACHE=/scratch/hf_cache
R=/scratch/boltzmann-attention
VLLM=/scratch/venvs/sop_env/bin/vllm
IP=/scratch/venvs/tb_env/bin/python
PORT=${P4_PORT:-8500}
SPECS=$R/scripts/distill/tau2/specs
OUT=/scratch/taskbench_runs/p4_a2_census
RAW=$R/reports/facet_rft_2026/trackb_raw/p4_a2_census
mkdir -p $OUT /scratch/logs
cd $R && git pull --ff-only

kill_gpus() {
  for g in $(echo $1 | tr , ' '); do
    for p in $(nvidia-smi --id=$g --query-compute-apps=pid --format=csv,noheader); do
      kill -9 $p 2>/dev/null; done; done
  sleep 10
}

run_arm() { # name hf_model gpus tp
  local NAME=$1 MODEL=$2 GPUS=$3 TP=$4
  [ -f $OUT/done_$NAME ] && { echo "SKIP $NAME (done)"; return; }
  kill_gpus $GPUS
  CUDA_VISIBLE_DEVICES=$GPUS setsid nohup $VLLM serve $MODEL --port $PORT \
    --tensor-parallel-size $TP --max-model-len 16384 \
    > /scratch/logs/vllm_p4_$NAME.log 2>&1 &
  local ok=0
  for i in $(seq 1 180); do
    curl -s localhost:$PORT/v1/models | grep -q model && ok=1 && break; sleep 10
  done
  [ $ok = 1 ] || { echo "SERVE_FAIL_$NAME"; return; }
  # sanity: response_format 실효 확인 (vllm 0.10.x silent-ignore 함정 — census parse가
  # brace-추출로 우아하게 강등되므로 진행하되 로그에 남김)
  curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL\", \"max_tokens\": 30, \"response_format\": {\"type\": \"json_object\"},
    \"messages\": [{\"role\": \"user\", \"content\": \"return {\\\"ok\\\":true}\"}]}" | head -c 200
  echo
  $IP $R/scripts/distill/tau2/t2_a2_size_census.py --target airline \
    --ref_dir $SPECS --policy $SPECS/airline_policy.md \
    --catalog $SPECS/airline_tool_catalog.json \
    --model ${NAME}:http://localhost:${PORT}/v1:${MODEL} \
    --out $OUT/spec 2>&1 | tee $OUT/score_$NAME.txt
  kill_gpus $GPUS
  touch $OUT/done_$NAME
}

run_arm qwen25_32b Qwen/Qwen2.5-32B-Instruct 0,1 2
run_arm qwen25_72b Qwen/Qwen2.5-72B-Instruct 0,1,2,3 4
run_arm qwen3_235b_int4 Qwen/Qwen3-235B-A22B-Instruct-2507-GPTQ-Int4 0,1,2,3 4

mkdir -p $RAW
cp $OUT/spec.*.json $OUT/score_*.txt $RAW/ 2>/dev/null
cd $R && git add reports/facet_rft_2026/trackb_raw/p4_a2_census && \
  git commit -m "P4 A2-compiler size census raw: generated specs + scores (32b/72b/235b-int4)" && git push
echo "P4_DONE $(date)"
