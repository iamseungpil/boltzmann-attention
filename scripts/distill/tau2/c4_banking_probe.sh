#!/bin/bash
# banking_knowledge prior probe: base 7B 돌려 fetch-required-id 날조(schema-copy prior)가 있나 확인.
# autofetch 전이 테스트의 선결(prior 없으면 무의미). gate0·resolve0·순수 base.
# Usage: c4_banking_probe.sh <gpu> <port> [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; NT=${3:-50}
BASE=Qwen/Qwen2.5-7B-Instruct
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/c4_banking_probe.log; exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  > $S/vllm_bankprobe.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_bankprobe.log; exit 1; }
echo "[serve OK]"

set +x; source /home/woori/.openrouter_key; source /home/woori/.openai_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
echo "[keys] openrouter(user-sim)=${OPENROUTER_API_KEY:+SET} openai(embed)=${OPENAI_API_KEY:+SET}"
cd $S/tau2-bench; export PYTHONPATH=src:$T2
unset T2_PROVENANCE T2_AUTOFETCH T2_MAXPROMPT
rm -rf data/simulations/c4bank_base
$PY $T2/t2_run_gated.py --gate 0 --resolve 0 --domain banking_knowledge --num_trials 1 --num_tasks $NT \
  --agent_model "$BASE" --agent_base http://localhost:$PORT/v1 \
  --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to c4bank_base > $S/c4bank_base.log 2>&1 || echo RUN_FAIL
grep -oE "pass1=[0-9]+/[0-9]+" $S/c4bank_base.log | tail -1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== banking base done — inspect data/simulations/c4bank_base for id-fabrication ==="
date; echo C4_BANKING_PROBE_DONE
