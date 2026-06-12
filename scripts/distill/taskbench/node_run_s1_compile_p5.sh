#!/bin/bash
# node_run_s1_compile_p5.sh — P5 (요청서 §11): S1 교사-풀 컴파일, 추론-only.
# 각 모델이 specs/s1_inputs/manifest.txt의 (도메인, 정책NL, A1카탈로그)를 GATE_SPEC으로
# 컴파일(모델당 도메인당 1콜). reference 없음 = --gen_only (채점 생략) — 수용 판정은
# Track A가 결정론 replay 검증(over/under-deny)으로 사후 수행, 통과분만 S1 학습셋 편입.
# 입력 전부 repo 내장. 멱등: per-arm done-marker. 산출물 = spec JSON 원본 -> trackb_raw push.
set -x
export HF_HUB_CACHE=/scratch/hf_cache
R=/scratch/boltzmann-attention
VLLM=/scratch/venvs/sop_env/bin/vllm
IP=/scratch/venvs/tb_env/bin/python
PORT=${P5_PORT:-8500}
SPECS=$R/scripts/distill/tau2/specs
S1=$SPECS/s1_inputs
OUT=/scratch/taskbench_runs/p5_s1_compile
RAW=$R/reports/facet_rft_2026/trackb_raw/p5_s1_compile
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
    > /scratch/logs/vllm_p5_$NAME.log 2>&1 &
  local ok=0
  for i in $(seq 1 180); do
    curl -s localhost:$PORT/v1/models | grep -q model && ok=1 && break; sleep 10
  done
  [ $ok = 1 ] || { echo "SERVE_FAIL_$NAME"; return; }
  while read -r DOM POL CAT; do
    [ -z "$DOM" ] && continue
    $IP $R/scripts/distill/tau2/t2_a2_size_census.py --target $DOM --gen_only \
      --ref_dir $SPECS --policy $S1/$POL --catalog $S1/$CAT \
      --model ${NAME}:http://localhost:${PORT}/v1:${MODEL} \
      --out $OUT/${DOM}_spec 2>&1 | tee -a $OUT/score_$NAME.txt
  done < $S1/manifest.txt
  kill_gpus $GPUS
  touch $OUT/done_$NAME
}

run_arm qwen25_32b Qwen/Qwen2.5-32B-Instruct 0,1 2
run_arm qwen25_72b Qwen/Qwen2.5-72B-Instruct 0,1,2,3 4
run_arm qwen3_235b_int4 Qwen/Qwen3-235B-A22B-Instruct-2507-GPTQ-Int4 0,1,2,3 4

mkdir -p $RAW
cp $OUT/*_spec.*.json $OUT/score_*.txt $RAW/ 2>/dev/null
cd $R && git add reports/facet_rft_2026/trackb_raw/p5_s1_compile && \
  git commit -m "P5 S1 teacher-compile raw: generated GATE_SPECs (manifest domains x 32b/72b/235b-int4)" && git push
echo "P5_DONE $(date)"
