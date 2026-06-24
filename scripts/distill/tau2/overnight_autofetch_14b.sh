#!/bin/bash
# 밤샘 (2026-06-25 취침·GPU1): autofetch-σ arm 전환검증 — 14B scale point.
# H: FETCH-offload(present)가 *약한 모델*에 더 큰 레버인가(cheap-deploy 스토리). select_confirm 한계효과.
#   A: select-only(auth+select_confirm) vs floor(기존) · B: g14+select vs g14(기존 ours_n14b_g14_retail_t3).
# user-sim=gpt-4.1(COST GUARD). nt=3. 마커=AUTOFETCH_GPU1_DONE.
set -u
GPU=1; PORT=8362
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
M=Qwen/Qwen2.5-14B-Instruct
LOG=$S/overnight_autofetch_14b.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2

run () {  # $1=save $2..=env
  local save=$1; shift
  echo "######## RUN $save env=$* ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials 3 --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date
}

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
  > $S/vllm_autofetch_14b.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_autofetch_14b.log; exit 1; }
echo "SERVE_OK"; date

run ours_n14b_selectonly_retail_t3  T2_GATE_KINDS=auth,select_confirm
run ours_n14b_g14select_retail_t3   T2_GATE_KINDS=auth,confirm,ownership,notice,select_confirm

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "AUTOFETCH_GPU1_DONE"; date
