#!/bin/bash
# ★floor(base·게이트無) nt=4 재런 — pass^1..4 완성용(same-scale 비교·리더보드 프로토콜).
#   --gate 0 = 순수 base(scaffold/regen 없음)·nt=4·32768·공식 compute_metrics.
# 사용: reexp_floor.sh <GPU> <PORT> <MODEL> <TAG>
set -u
GPU=$1; PORT=$2; M=$3; TAG=$4
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/reexp_floor_$TAG.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT --enable-auto-tool-choice \
  --tool-call-parser hermes --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.92 \
  > $S/vllm_floor_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 180); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_floor_$TAG.log; exit 1; }
echo "SERVE_OK"; date
SAVE=${TAG}_floor_retail_t4
cd $TB; rm -rf "$TB/data/simulations/$SAVE"
$PY $T2/t2_run_gated.py --gate 0 --domain retail \
  --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  --num_trials 4 --max_concurrency 8 --save_to "$SAVE" || echo "ARM_FAIL $SAVE"
echo "ARM_DONE $SAVE"; date
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
RES=$TB/data/simulations/$SAVE/results.json
$PY - <<PYEOF
import json
from collections import Counter
s=json.load(open("$RES"))["simulations"]
inf=sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
print("RESULT_${TAG} n=%d infra=%d term=%s" % (len(s),inf,dict(Counter(x.get("termination_reason") for x in s))))
import sys; sys.path.insert(0,"src")
from tau2.metrics.agent_metrics import compute_metrics
from tau2.data_model.simulation import Results
m=compute_metrics(Results.model_validate(json.load(open("$RES"))))
print("OFFICIAL_${TAG} pass_hat_ks:", m.pass_hat_ks, "avg_reward:", round(m.avg_reward,4))
PYEOF
PERSIST=$REPO/reports/facet_rft_2026/sim_results; mkdir -p $PERSIST
if [ -f "$RES" ]; then
  gzip -c "$RES" > $PERSIST/${TAG}_floor_retail_t4.results.json.gz
  cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
  git add -f $PERSIST/${TAG}_floor_retail_t4.results.json.gz
  git commit -q -m "persist sim results: ${TAG}_floor_retail_t4 (base nt4·auto)" 2>/dev/null
  for try in 1 2 3; do git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && { echo "PERSISTED_${TAG}"; break; }; sleep 5; done
fi
echo "REEXP_FLOOR_${TAG}_DONE"; date
