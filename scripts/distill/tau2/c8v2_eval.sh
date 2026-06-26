#!/bin/bash
# C8-v2 = grounding(autofetch) 위에서 recovery(retry-controller)의 *한계가치* (paired).
# C8(격리·autofetch OFF) 재프레임: A의 처방=autofetch 확정(§35b) 후 "retry가 더 보태나?".
# arms: engine(gate1 prov1 af1) vs engine_retry(+retry1). 키스톤 A2-구동 scaffold 위.
# 판정: engine_retry pass > engine → recovery 잔여가치 / too_many↓∧pass= → grounding 위에서도 loop-only.
# GPU1 전용(C3 sweep=GPU0 무간섭). Usage: c8v2_eval.sh <gpu> <port> [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; NT=${3:-114}
BASE=Qwen/Qwen2.5-7B-Instruct
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/c8v2_eval.log; exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 10
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  > $S/vllm_c8v2.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_c8v2.log; exit 1; }
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$BASE\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "HC_FAIL: $hc"; exit 1; }; echo "[hc OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2
export T2_PROVENANCE=1 T2_AUTOFETCH=1   # grounding 항상 ON (재프레임)

cell () {  # $1=tag $2=retry
  local TAG=$1 RT=$2 SAVE=c8v2_$1
  [ "$RT" = "1" ] && export T2_RETRY_CONTROLLER=1 || unset T2_RETRY_CONTROLLER
  rm -rf data/simulations/$SAVE
  echo "######## $TAG (gate1 prov1 af1 retry=$RT) ########"; date
  $PY $T2/t2_run_gated.py --gate 1 --resolve 0 --domain retail --num_trials 1 --num_tasks $NT \
    --agent_model "$BASE" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/c8v2_${TAG}.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/c8v2_${TAG}.log | tail -1)
  local TME=$($PY - "$SAVE" <<'PY'
import json,sys
d=json.load(open(f"data/simulations/{sys.argv[1]}/results.json"))
s=d.get("simulations") or []
tme=sum(1 for x in s if x.get("termination_reason")=="too_many_errors")
print(f"too_many_errors={tme}/{len(s)}")
PY
)
  local F=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|B_wrong" | tr '\n' ' ')
  echo "C8V2_ROW $TAG pass=$P | $TME | $F"
}

cell engine       0
cell engine_retry 1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== C8-v2 SUMMARY ==="; grep C8V2_ROW $LOG
date; echo C8V2_EVAL_DONE
