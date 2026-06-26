#!/bin/bash
# fetch-first 규칙 × 전 레버 곡선 eval (C4_LEARN_FETCHFIRST_CROSSOVER §2·RULE_LEVER_PROGRAM).
# base+LoRA 한 serve(--enable-lora). y=A_notfound(격리·failcensus_deep) + 보조 pass^1.
# arms: base / prompt(DG) / skill / scaffold-deny / scaffold-perform(=engine autofetch) / learn / learn-pure.
#   grammar=N/A(fetch-first=행동규율). scale(14/32B)=별 serve(이 드라이버 밖).
# Usage: c4_fetchfirst_eval.sh <gpu> <port> [domain] [num_tasks]
set -u
GPU=${1:?}; PORT=${2:?}; DOM=${3:-retail}; NT=${4:-114}
BASE=Qwen/Qwen2.5-7B-Instruct
ADAPTER=/home/woori/scratch/adapters/qwen7b_c4_fetchfirst
R=/home/woori/workspace_common/boltzmann-attention-pi; T2=$R/scripts/distill/tau2; A2=$T2/a2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch
LOG=$S/c4_ff_eval_${DOM}.log; exec > $LOG 2>&1; set -x; date
[ -d $ADAPTER ] || { echo "ADAPTER_MISSING $ADAPTER"; exit 1; }

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 12
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve $BASE --port $PORT \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --enforce-eager \
  --enable-lora --max-lora-rank 16 --lora-modules c4ff=$ADAPTER > $S/vllm_c4ff_${DOM}.log 2>&1 &
ok=0; for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$BASE" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo SERVE_FAIL; tail -40 $S/vllm_c4ff_${DOM}.log; exit 1; }
echo "[serve OK]"; curl -s localhost:$PORT/v1/models | grep -o '"id":"[^"]*"'

set +x; source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())"); set -x
cd $S/tau2-bench; export PYTHONPATH=src:$T2

cell () {  # $1=tag $2=model $3=gate $4=prov $5=autofetch $6=promptfile(or -)
  local TAG=$1 MODEL=$2 G=$3 PV=$4 AF=$5 PF=$6 SAVE=c4ff_${DOM}_$1 RARG=""
  [ "$PV" = "1" ] && export T2_PROVENANCE=1 || unset T2_PROVENANCE
  [ "$AF" = "1" ] && export T2_AUTOFETCH=1 || unset T2_AUTOFETCH
  unset T2_RETRY_CONTROLLER
  [ "$PF" != "-" ] && RARG="--rules_prompt $A2/$PF"
  rm -rf data/simulations/$SAVE
  echo "######## $TAG (model=$MODEL gate=$G prov=$PV af=$AF prompt=$PF) ########"; date
  $PY $T2/t2_run_gated.py --gate $G --resolve 0 --domain $DOM --num_trials 1 --num_tasks $NT \
    $RARG --agent_model "$MODEL" --agent_base http://localhost:$PORT/v1 \
    --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 --save_to $SAVE > $S/c4ff_${DOM}_${TAG}.log 2>&1 || echo FAIL
  local P=$(grep -oE "pass1=[0-9]+/[0-9]+" $S/c4ff_${DOM}_${TAG}.log | tail -1)
  local F=$($PY $T2/t2_failcensus_deep.py data/simulations/$SAVE 2>/dev/null | grep -E "A_notfound|B_wrong" | tr '\n' ' ')
  local M=$($PY $T2/c4_prompt_mechanism.py data/simulations/$SAVE 2>/dev/null | grep M_MECH_ROW)
  echo "C4FFROW $DOM $TAG pass=$P | $F"
  echo "C4FF_MECH $DOM $TAG | $M"
}

#    tag             model  gate prov af  prompt
cell base            $BASE  0    0    0   -
cell prompt          $BASE  0    0    0   C4_FETCHFIRST_DG.txt
cell skill           $BASE  0    0    0   C4_SKILL.txt
cell fewshot-dg      $BASE  0    0    0   C4_FEWSHOT_DG.txt
cell fewshot-retail  $BASE  0    0    0   C3_FEWSHOT.txt
cell scaffold-deny   $BASE  1    1    0   -
cell scaffold-perform $BASE 1    1    1   -
cell learn           c4ff   1    1    0   -
cell learn-pure      c4ff   0    0    0   -

for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
echo "=== C4 FETCH-FIRST CURVE ($DOM) ==="; grep "C4FFROW $DOM" $LOG
date; echo C4FF_EVAL_DONE
