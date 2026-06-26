#!/bin/bash
# 4hr 무인 cost-inclusive 배터리 (2026-06-23 외출·GPU1·GPU0의 32B nt=3 denoise와 병렬).
# 모든 런=duration 자동포착→tco_cost로 비용열. 값순서: deploy-retail nt=3 먼저, 그 후 cross-domain floor nt=3.
#  {7B,14B} × [ g15 retail(deploy·nt3) + airline/banking floor(nt3) ].  (32B=GPU0 denoise가 담당)
# 회수(아침): morning_tables.py / tco_table.py + tco_cost.py. 마커=TCO_OVERNIGHT_DONE.
set -u
GPU=1; PORT=8361
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
NT=3
LOG=$S/tco_overnight.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
MODELS=("n7b:Qwen/Qwen2.5-7B-Instruct" "n14b:Qwen/Qwen2.5-14B-Instruct")

run () {  # $1=save $2=domain $3=gate $4..=env
  local save=$1 dom=$2 gate=$3; shift 3
  echo "######## RUN $save dom=$dom gate=$gate ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate $gate --domain "$dom" \
    --agent_model "$CKPT" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials $NT --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date
}

for entry in "${MODELS[@]}"; do
  TAG=${entry%%:*}; CKPT=${entry#*:}
  echo "============ MODEL $TAG ============"; date
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$CKPT" --port $PORT \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
    > $S/vllm_tco_$TAG.log 2>&1 &
  ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$CKPT" && ok=1 && break; sleep 10; done
  [ $ok = 1 ] || { echo "SERVE_FAIL $TAG"; tail -40 $S/vllm_tco_$TAG.log; continue; }
  echo "SERVE_OK $TAG"; date
  # 값순서: deploy-retail nt3 먼저
  run ours_${TAG}_g15_retail_t3   retail            1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions
  run ours_${TAG}_floor_airline_t3 airline          0
  run ours_${TAG}_floor_bank_t3    banking_knowledge 0
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
done
echo "TCO_OVERNIGHT_DONE"; date
