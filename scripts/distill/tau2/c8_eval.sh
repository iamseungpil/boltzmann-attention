#!/bin/bash
# C8 recovery cheap-replication eval (설계=C8_RECOVERY_DESIGN_2026_06_21).
# 한 serve(base 7B)·retail n=114. 셀=pass^1 + too_many_errors 수 + failcensus.
# 격리: c8_gate vs c8_gate_retry (T2_RETRY_CONTROLLER 1개 차이). c8_floor=절대 baseline.
# GO 분리(리뷰): pass↑=recovery / too_many↓∧pass=→loop-only 경계.
# Usage: c8_eval.sh <gpu> <port> [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; NT=${3:-114}
BASE=Qwen/Qwen2.5-7B-Instruct
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/c8_eval.log; exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  > $S/vllm_c8.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_c8.log; exit 1; }
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$BASE\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "HC_FAIL: $hc"; exit 1; }; echo "[hc OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

cell () {  # $1=tag $2=gate $3=retry
  local TAG=$1 G=$2 RT=$3 SAVE=c8_$1
  [ "$RT" = "1" ] && export T2_RETRY_CONTROLLER=1 || unset T2_RETRY_CONTROLLER
  unset T2_PROVENANCE T2_AUTOFETCH   # C8 격리: provenance/autofetch off (recovery 단독 측정)
  rm -rf data/simulations/$SAVE
  echo "######## $TAG (gate=$G retry=$RT) ########"; date
  $PY $T2/t2_run_gated.py --gate $G --resolve 0 --domain retail --num_trials 1 --num_tasks $NT \
    --agent_model "$BASE" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/c8_${TAG}.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/c8_${TAG}.log | tail -1)
  local TME=$($PY - "$SAVE" <<'PY'
import json,sys
d=json.load(open(f"data/simulations/{sys.argv[1]}/results.json"))
s=d.get("simulations") or []
tme=sum(1 for x in s if x.get("termination_reason")=="too_many_errors")
print(f"too_many_errors={tme}/{len(s)}")
PY
)
  local F=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|B_wrong|F_error|D_no" | tr '\n' ' ')
  echo "C8_ROW $TAG pass=$P | $TME | $F"
}

cell floor      0 0
cell gate       1 0
cell gate_retry 1 1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== C8 SUMMARY ==="; grep C8_ROW $LOG
date; echo C8_EVAL_DONE
