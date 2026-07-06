#!/bin/bash
# ★replay-safe 게이트 스모크 (REPLAY_SAFE_GATE_DESIGN §7.1) — T2_GATE_REGEN=1.
# 목적: assembled config + 생성-레벨 regen 게이트로 (1)infra_error=0(replay 안 깨짐·R8 종단 포함)
#   (2)게이트 실발화·compliance 유지 검증. 크래시했던 task 재실행(num_tasks 40 covers 19/22/27/28/31/33/34/36/37).
# 사용: smoke_regen.sh <GPU> <PORT> <MODEL> <TAG> <NUM_TASKS>
set -u
GPU=$1; PORT=$2; M=$3; TAG=$4; NTASK=${5:-40}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/smoke_regen_$TAG.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT --enable-auto-tool-choice \
  --tool-call-parser hermes --max-model-len 16384 --enforce-eager --gpu-memory-utilization 0.92 \
  > $S/vllm_smoke_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 150); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_smoke_$TAG.log; exit 1; }
echo "SERVE_OK"; date
SAVE=${TAG}_smoke_retail_t1
cd $TB; rm -rf "$TB/data/simulations/$SAVE"
env T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 \
    T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints \
    T2_PRESENT_READS=1 T2_PRESENT_NESTED=1 T2_CALC=1 PYTHONPATH=src:$T2 \
  $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials 1 --num_tasks $NTASK --max_concurrency 8 --save_to "$SAVE" || echo "ARM_FAIL $SAVE"
echo "ARM_DONE $SAVE"; date
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
# 즉석 요약 (infra/termination/reward)
$PY - <<PYEOF
import json
from collections import Counter
sims=json.load(open("$TB/data/simulations/$SAVE/results.json"))["simulations"]
term=Counter(s.get("termination_reason") for s in sims)
none=sum(1 for s in sims if (s.get("reward_info") or {}).get("reward") is None)
ok=sum(1 for s in sims if (s.get("reward_info") or {}).get("reward")==1)
print("SMOKE_SUMMARY n=%d reward_none(infra)=%d reward1=%d" % (len(sims),none,ok))
print("termination:",dict(term))
print("infra task_ids:", sorted(s.get("task_id") for s in sims if (s.get("reward_info") or {}).get("reward") is None))
PYEOF
echo "SMOKE_REGEN_${TAG}_DONE"; date
