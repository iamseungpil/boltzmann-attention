#!/bin/bash
# 밤샘 outline (2026-06-22 사용자: 아침에 테이블 확인). GPU1 전용(GPU0의 현 flow-disc run과 병렬).
# 검증된 인프라만(미검증 airline/banking 게이트 제외 = Stage 2). num_trials=1(breadth).
#  Table1 cross-domain floor: {7,14,32B}×{airline,banking} floor (retail floor 기존)
#  Table2 retail flow-discipline: {7,14,32B}×{g14(G1-G4),g15(G1-G5),g15retry(+retry)} (floor 기존)
# 회수(아침): grep -E "RESULT|ARM_DONE|ARM_FAIL|XDOMAIN_OVERNIGHT_DONE" $LOG
#   + t2_compliance.py --arm <save> --domain <d>  (bench+compliant)  + t2_failcensus.py (클래스Δ)
set -u
GPU=1; PORT=8361                       # ★GPU1 (GPU0 = 현 run)
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
TB=/home/woori/scratch/tau2-bench
NT="${1:-1}"
LOG=$S/overnight_xdomain.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2

MODELS=("n7b:Qwen/Qwen2.5-7B-Instruct" "n14b:Qwen/Qwen2.5-14B-Instruct" "n32int8:Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")

run () {  # $1=save $2=domain $3..=env
  local save=$1 dom=$2; shift 2
  echo "######## RUN $save dom=$dom env=$* ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate "${GATE:-1}" --domain "$dom" \
    --agent_model "$CKPT" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials $NT --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date
}

for entry in "${MODELS[@]}"; do
  TAG=${entry%%:*}; CKPT=${entry#*:}
  echo "============ MODEL $TAG ($CKPT) ============"; date
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$CKPT" --port $PORT \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
    > $S/vllm_overnight_$TAG.log 2>&1 &
  ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$CKPT" && ok=1 && break; sleep 10; done
  [ $ok = 1 ] || { echo "SERVE_FAIL $TAG"; tail -40 $S/vllm_overnight_$TAG.log; continue; }
  echo "SERVE_OK $TAG"; date

  # Table2: retail flow-discipline (gate=1)
  GATE=1 run ours_${TAG}_g14_retail      retail T2_GATE_KINDS=auth,confirm,ownership,notice
  GATE=1 run ours_${TAG}_g15_retail      retail T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions
  GATE=1 run ours_${TAG}_g15retry_retail retail T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions T2_RETRY_CONTROLLER=1 T2_RETRY_K=3
  # Table1: cross-domain floor (gate=0)
  GATE=0 run ours_${TAG}_floor_airline   airline
  GATE=0 run ours_${TAG}_floor_bank      banking_knowledge

  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 3
done
echo "XDOMAIN_OVERNIGHT_DONE"; date
