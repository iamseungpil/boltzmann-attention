#!/bin/bash
# gpt-5.2 user-sim measurement of assembled 32B+scaffold (2026-07-06·user-authorized paid run).
# Purpose: fill the gpt-5.2-user-sim comparison (deck conclusion pages 2&4) with OUR point,
# on the SAME (harder) user-sim as the frontier gpt-5.2-sim leaderboard. Agent = local 32B (free);
# ONLY the gpt-5.2 user-sim/judge is billed (OpenRouter). SMOKE-first then full nt=4 ([[30]]).
# COST GUARD note: guard only refuses Claude/Anthropic; gpt-5.2 (OpenAI) is allowed as-is.
# usage: reexp_gpt52sim.sh <GPU> <PORT> <MODEL> <TAG>
set -u
GPU=$1; PORT=$2; M=$3; TAG=$4
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/gpt52sim_$TAG.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
$PY $T2/test_assembled_run.py || { echo "GATE_FAIL — abort"; exit 1; }
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
USIM=openrouter/openai/gpt-5.2
GATES="T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints T2_PRESENT_READS=1 T2_PRESENT_NESTED=1 T2_CALC=1"
run () { local save=$1; local nt=$2; local ntask=$3; shift 3
  echo "######## RUN $save nt=$nt ntask=$ntask ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  local extra=""; [ "$ntask" != "0" ] && extra="--num_tasks $ntask"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm $USIM --user_temp 0.0 \
    --num_trials $nt --max_concurrency 6 --save_to "$save" $extra || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date; }
# ---- serve 32B-int8 on GPU (kills existing procs on that GPU first) ----
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT --enable-auto-tool-choice \
  --tool-call-parser hermes --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.95 \
  > $S/vllm_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_$TAG.log; exit 1; }
echo "SERVE_OK"; date
# ---- SMOKE (10 tasks, nt=1): verify gpt-5.2 user-sim end-to-end + mechanism before the full paid run ----
run ${TAG}_smoke 1 10 $GATES
SMK=$TB/data/simulations/${TAG}_smoke/results.json
$PY -c "import json;d=json.load(open('$SMK'));s=d.get('simulations',d.get('results',[]));n=len(s);r=sum((x.get('reward_info') or {}).get('reward')==1.0 for x in s);print('SMOKE_SIMS',n,'pass',r);assert n>=8,'smoke too few sims'" || { echo "SMOKE_FAIL — abort full run"; exit 1; }
echo "SMOKE_OK"; date
# ---- FULL nt=4 (114 tasks) to match frontier retail pass^4 ----
run ${TAG}_retail_t4 4 0 $GATES
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
# ---- persist (gitignore bypass; distinct tag; no wipe) ----
RES=$TB/data/simulations/${TAG}_retail_t4/results.json
PERSIST=$REPO/reports/facet_rft_2026/sim_results; mkdir -p $PERSIST
if [ -f "$RES" ]; then
  gzip -c "$RES" > $PERSIST/${TAG}_retail_t4.results.json.gz
  cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
  git add -f $PERSIST/${TAG}_retail_t4.results.json.gz
  git commit -q -m "persist gpt-5.2-sim run: ${TAG}_retail_t4 (paid·user-authorized)" 2>/dev/null
  for try in 1 2 3; do git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && { echo "PERSISTED_${TAG}"; break; }; sleep 5; done
else echo "PERSIST_SKIP_NO_RESULTS"; fi
echo "GPT52SIM_${TAG}_DONE"; date
