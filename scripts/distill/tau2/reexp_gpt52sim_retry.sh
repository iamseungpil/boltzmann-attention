#!/bin/bash
# Targeted re-run of the context-contaminated tasks ONLY, at 32768 ctx, reusing the
# already-running 32B server. Merge with the clean 16k-run sims for a full result.
# usage: reexp_gpt52sim_retry.sh <PORT> <MODEL> <TAG> <TASKIDS-comma>
set -u
PORT=$1; M=$2; TAG=$3; TASKIDS=$4
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/gpt52sim_${TAG}.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" || { echo "SERVE_NOT_UP — abort"; exit 1; }
echo "SERVE_OK (reusing running 32768 server)"; date
cd $TB; rm -rf "$TB/data/simulations/${TAG}"
env T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints \
    T2_PRESENT_READS=1 T2_PRESENT_NESTED=1 T2_CALC=1 PYTHONPATH=src:$T2 \
  $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --num_trials 4 --max_concurrency 6 --task_ids "$TASKIDS" --save_to "$TAG" || echo "RETRY_FAIL"
echo "RETRY_DONE"; date
RES=$TB/data/simulations/${TAG}/results.json
PERSIST=$REPO/reports/facet_rft_2026/sim_results; mkdir -p $PERSIST
if [ -f "$RES" ]; then
  gzip -c "$RES" > $PERSIST/${TAG}.results.json.gz
  cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
  git add -f $PERSIST/${TAG}.results.json.gz
  git commit -q -m "persist gpt-5.2-sim 32k targeted retry: ${TAG}" 2>/dev/null
  for t in 1 2 3; do git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && { echo "PERSISTED_${TAG}"; break; }; sleep 5; done
fi
echo "GPT52SIM_RETRY_${TAG}_DONE"; date
