#!/bin/bash
# 전이 매트릭스 — 학습모델 × {TBox-only / +ABox-swap(provenance gate+resolve)} → τ² retail.
# 질문: 추상학습 provenance가 실 retail로 (1)모델만으로 전이하나 (2)ABox-swap scaffold가 메우나.
# 한 serve(base+LoRAs)서 모델×조건 순차. 셀마다 pass^1 + order_id날조율(failcensus A%).
# Usage: transfer_matrix.sh <gpu> <port> <half:A|B> [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; HALF=${3:?}; NT=${4:-50}
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; BASE=Qwen/Qwen2.5-7B-Instruct
AD=$S/adapters; FR=$S/sft_runs
# half별 모델: tag=path (base는 lora 없음)
if [ "$HALF" = "A" ]; then
  LORAS="solo_sts=$AD/qwen7b_solo_sts solo_cfb_mid=$AD/qwen7b_solo_cfb_mid"
  MODELS="base solo_sts solo_cfb_mid"
else
  LORAS="fact_full=$FR/fact_FULL fact_prov=$FR/fact_A_prov solo_cons=$AD/qwen7b_solo_cons"
  MODELS="fact_full fact_prov solo_cons"
fi
LOG=$S/transfer_matrix_$HALF.log; exec > $LOG 2>&1; set -x; date

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  --enable-lora --max-lora-rank 64 --lora-modules $LORAS > $S/vllm_mtx_$HALF.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_mtx_$HALF.log; exit 1; }
hc=$(curl -s localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$BASE\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3}" 2>/dev/null)
echo "$hc" | grep -q '"content"' || { echo "HEALTHCHECK_FAIL: $hc"; exit 1; }
echo "[hc OK]"

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

run_cell () {  # $1=model(served name) $2=cond(tbox|abox)
  local M=$1 C=$2 SAVE GATE RES MS
  MS=$M; [ "$M" = "base" ] && MS=$BASE   # base 셀 = 실제 base 모델명(LoRA 미적용)
  SAVE=mtx_${M}_${C}
  if [ "$C" = "abox" ]; then GATE=1; RES=1; export T2_PROVENANCE=1; else GATE=0; RES=0; unset T2_PROVENANCE; fi
  rm -rf data/simulations/$SAVE
  echo "######## CELL $M / $C (gate=$GATE resolve=$RES prov=${T2_PROVENANCE:-0}) ########"; date
  $PY $T2/t2_run_gated.py --gate $GATE --resolve $RES --domain retail --num_trials 1 --num_tasks $NT \
    --agent_model "$MS" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/mtx_${M}_${C}.log 2>&1 || echo "FAIL"
  local PASS=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/mtx_${M}_${C}.log | tail -1)
  local FAB=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|실패" | tr '\n' ' ')
  echo "RESULT_ROW $M $C $PASS | $FAB"
}

for M in $MODELS; do for C in tbox abox; do run_cell $M $C; done; done

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== MATRIX $HALF DONE ==="; grep RESULT_ROW $LOG; date; echo MATRIX_DONE_$HALF
