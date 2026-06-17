#!/bin/bash
# τ² retail END-TO-END (gated compliant-pass^1, user-sim=gpt-4.1) overnight anchor, AFTER the GPU jobs
# finish. Measures the FULL agentic task (multi-turn, all args, flow), not the offline selection.
#   - base 7B            -> re-anchor 0.17 on current harness/cases.
#   - MD_route_ep1       -> does the 7-op op-IR routing SFT help/hurt as a NATIVE agent (format caveat)?
#   - MD_widesubst_w4    -> does wide-substitute SFT change end-to-end?
# CAVEAT: our LoRAs were SFT'd to emit op-IR JSON, not native tool-calls; running them as native agents
# is a diagnostic of as-is transfer, NOT the offload-integrated test (that needs op-IR->resolver wiring
# in the agent loop = separate attended work). Cost-bounded: num_tasks=40, num_trials=1.
set -u
R=/home/woori/workspace_common/boltzmann-attention-pi
T2=$R/scripts/distill/tau2
PY=/home/woori/venvs/seka_env/bin/python; VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm
S=/home/woori/scratch; ADP=$S/depth/c8/adapters
GPU=0; PORT=8366; NT=40
LOG=$S/tau2_e2e_overnight.log; exec > $LOG 2>&1; set -x; date

# 1. wait for every GPU overnight job to finish (no contention)
for J in "[w]idth_ladder_local" "[w]iden_retrain.sh" "[c]8_5op_reeval.sh"; do
  while pgrep -f "$J" >/dev/null; do echo "waiting $J"; date; sleep 120; done
done
echo "all GPU jobs done"; date; sleep 30
cd $R && git pull --ff-only 2>&1 | tail -1
source /home/woori/.openrouter_key; export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
kg(){ for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5; }

run_e2e(){  # $1=name $2=adapter_dir(empty=base) $3=save_to
  local NAME=$1 ADAPTER=$2 SAVE=$3
  kg
  if [ -n "$ADAPTER" ]; then
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
      --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 \
      --enable-lora --lora-modules ${NAME}=$ADAPTER --max-lora-rank 64 > $S/vllm_e2e_${NAME}.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port $PORT \
      --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 > $S/vllm_e2e_${NAME}.log 2>&1 &
  fi
  ok=0; for i in $(seq 1 90); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$NAME" && ok=1 && break; sleep 10; done
  [ $ok = 1 ] || { echo "SERVE_FAIL $NAME"; tail -20 $S/vllm_e2e_${NAME}.log; return; }
  cd $S/tau2-bench; export PYTHONPATH=src:$T2; rm -rf data/simulations/$SAVE
  $PY $T2/t2_run_gated.py --gate 1 --num_trials 1 --num_tasks $NT --agent_model "$NAME" \
    --agent_base http://localhost:$PORT/v1 --user_llm "openrouter/openai/gpt-4.1" --user_temp 0.0 \
    --save_to $SAVE || echo "RUN_FAIL $NAME"
  cd $R; kg
  echo -n "### $NAME pass^1 = "
  $PY -c "import json;print(json.load(open('$S/tau2-bench/data/simulations/$SAVE/compliance.json'))['bench']['pass^1'])" 2>/dev/null || echo NA
}

run_e2e "Qwen/Qwen2.5-7B-Instruct" "" retail_base_anchor
[ -d $ADP/MD_route_ep1 ]     && run_e2e MD_route_ep1     $ADP/MD_route_ep1     retail_md_route
[ -d $ADP/MD_widesubst_w4 ]  && run_e2e MD_widesubst_w4  $ADP/MD_widesubst_w4  retail_md_widesubst

# ---- GBW headroom autopsy: classify base failures into wrong_write (=GBW, content/offload-addressable)
# vs FLOW (fab_auth/over_ask/agent_collapse/no_write). wrong_write count = max e2e gain from the content
# fix ALONE (answers: does fixing GBW alone improve e2e, and by how much?).
cd $S/tau2-bench; export PYTHONPATH=src:$T2
for SV in retail_base_anchor retail_md_route retail_md_widesubst; do
  D=$S/tau2-bench/data/simulations/$SV
  [ -d "$D" ] || continue
  echo "===== GBW AUTOPSY $SV ====="
  $PY $T2/tau2_autopsy.py "$D" 2>&1 | tee "$D/autopsy.txt" | tail -25
done
cd $R
echo "TAU2_E2E_OVERNIGHT_DONE (base 0.17 / frontier 0.81 ref)"; date