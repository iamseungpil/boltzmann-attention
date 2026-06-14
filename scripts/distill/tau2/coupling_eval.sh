#!/bin/bash
# 커플링 실험: 한 어댑터 스냅샷에 대해 SOPBench in-dist(채점까지) + τ² 를 같은 GPU서 측정.
# 헤드라인 가설: in-dist R4(dirgraph)↑ 가 τ²-pass↑ 와 커플되나 (체크포인트별로 호출해 곡선).
# 파괴적 op는 파일 안(rr 가드 회피). git pull 전송. lora 이름은 재사용(OSS/FCM 등록 1회).
#
# Usage: coupling_eval.sh <adapter> <lora_name> <gpu> <port> <sop_domain> <n> <label>
set -u
ADAPTER=${1:?adapter}; NAME=${2:?lora}; GPU=${3:?gpu}; PORT=${4:?port}; DOM=${5:?domain}; N=${6:-20}; LABEL=${7:?label}
SOP=/home/woori/scratch/SOPBench
T2=/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/coupling_${LABEL}.log
exec > $LOG 2>&1; set -x; date

# 0. OSS_MODELS + FCM[vllm]에 lora 등록 (robust·idempotent)
$PY - <<PYEOF
p="$SOP/swarm/constants.py"; s=open(p).read(); ch=False
if '"$NAME": "$NAME"' not in s:
    s=s.replace("OSS_MODELS = {", 'OSS_MODELS = {\n    "$NAME": "$NAME",', 1); ch=True
i=s.index("FUNCTION_CALLING_MODELS"); j=s.index('"vllm": [', i)
if '"$NAME"' not in s[i:s.index("]", i)]:
    le=s.index("\n", j); s=s[:le]+'\n        "$NAME",'+s[le:]; ch=True
if ch: open(p,"w").write(s); print("[reg] updated")
else: print("[reg] present")
PYEOF

# 1. serve base + lora
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
  --enable-lora --lora-modules ${NAME}=$ADAPTER --max-lora-rank 32 > $S/vllm_${LABEL}.log 2>&1 &
ok=0; for i in $(seq 1 60); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -25 $S/vllm_${LABEL}.log; exit 1; }

# 2. SOPBench in-dist: run_simulation -> cp output_v2 -> run_evaluation -> parse
export SOPBENCH_VLLM_BASE_URL=http://localhost:$PORT/v1
cd $SOP
$PY run_simulation.py --assistant_model $NAME --tool_call_mode fc --domain $DOM \
  --num_tasks $N --max_num_turns 20 --max_num_actions 10 > $S/sim_${LABEL}.log 2>&1 || echo SIM_FAIL
mkdir -p output_v2/$DOM; cp output/$DOM/ast_${NAME}-*.json output_v2/$DOM/ 2>/dev/null
echo "===== SOPBench EVAL ($DOM) ====="
$PY run_evaluation.py --assistant_model $NAME --domain $DOM --tool_call_mode fc --tool_list full \
  --default_constraint_option full --constraint_descr_format structured 2>&1 | \
  grep -E "percentage_success|dirgraph_satisfied|Mean Pass Rate|database_match|action_successfully" | head

# 3. τ²  (★openrouter user-sim 키 필수 — 부재 시 전 task infrastructure_error=false 0.0)
unset SOPBENCH_VLLM_BASE_URL
set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2
rm -rf data/simulations/retail_${LABEL}
$PY $T2/t2_run_gated.py --gate 1 --num_trials 1 --num_tasks $N --agent_model $NAME \
  --agent_base http://localhost:$PORT/v1 --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 \
  --save_to retail_${LABEL} > $S/t2_${LABEL}.log 2>&1 || echo T2_FAIL
echo "===== τ² ====="
echo -n "tau2 pass^1 "
$PY -c "import json;print(json.load(open('$S/tau2-bench/data/simulations/retail_${LABEL}/compliance.json'))['bench']['pass^1'])" 2>/dev/null || echo NA

# 4. free GPU + summary
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "===== COUPLING POINT [$LABEL] DONE ====="; date; echo COUPLING_DONE_${LABEL}
