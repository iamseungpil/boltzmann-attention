#!/bin/bash
# 날조금지 방법 비교 — 한 모델 × {none / nofab-prompt / prov-gate(ABox-swap) / prompt+gate} → τ² retail.
# 학습 0 방법들(프롬프트·게이트)의 비용대비 효과. SFT/ReST는 transfer_matrix가 담당. 셀마다 pass^1+order_id날조율.
# Usage: method_compare.sh <gpu> <port> <hf_model> <tag> [num_tasks] [extra_serve]
set -u
GPU=${1:?}; PORT=${2:?}; MODEL=${3:?}; TAG=${4:?}; NT=${5:-50}; EXTRA=${6:-}
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; NOFAB=$T2/a2/NOFAB_PROMPT.txt
LOG=$S/method_compare_$TAG.log; exec > $LOG 2>&1; set -x; date; echo "MODEL=$MODEL"

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $MODEL --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager $EXTRA \
  > $S/vllm_mc_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$MODEL" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_mc_$TAG.log; exit 1; }
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "HC_FAIL: $hc"; exit 1; }; echo "[hc OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

cell () {  # $1=name $2=gate $3=prov(0/1) $4=prompt(0/1)
  local NM=$1 G=$2 PV=$3 PR=$4 SAVE=mc_${TAG}_$1 RARG=""
  [ "$PV" = "1" ] && export T2_PROVENANCE=1 || unset T2_PROVENANCE
  [ "$PR" = "1" ] && RARG="--rules_prompt $NOFAB"
  rm -rf data/simulations/$SAVE
  echo "######## CELL $TAG/$NM (gate=$G prov=$PV prompt=$PR) ########"; date
  $PY $T2/t2_run_gated.py --gate $G --resolve 0 --domain retail --num_trials 1 --num_tasks $NT \
    $RARG --agent_model "$MODEL" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/mc_${TAG}_${NM}.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/mc_${TAG}_${NM}.log | tail -1)
  local F=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|실패 " | tr '\n' ' ')
  echo "MC_ROW $TAG $NM $P | $F"
}

cell none   0 0 0
cell prompt 0 0 1
cell gate   1 1 0
cell pg     1 1 1

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== METHOD_COMPARE $TAG DONE ==="; grep MC_ROW $LOG; date; echo MC_DONE_$TAG
