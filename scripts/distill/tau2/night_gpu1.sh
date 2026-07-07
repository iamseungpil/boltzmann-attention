#!/bin/bash
# Overnight GPU1 chain (PAID·user-approved 2026-07-07): after QwQ isolated probe finishes, run
# QwQ-32B agentic FLOOR (--gate 0) nt=4 retail with gpt-4.1 user-sim. SMOKE-GATED: nt=1/12task first;
# only launch the paid nt=4 full-run if smoke has low infra (guards against thinking-context overflow).
# QwQ floor vs Qwen2.5-32B floor (0.557 bench / 0.509 compliant) = does agentic thinking help end-to-end.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
GPU=1; PORT=8141; M=Qwen/QwQ-32B-AWQ
LOG=$S/night_gpu1.log
exec > $LOG 2>&1; set -x; date
cd $REPO
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
# 1. wait for QwQ isolated probe to finish (max ~2h)
for i in $(seq 1 240); do [ -f $S/qwq_full_end ] && break; sleep 30; done
echo "=== QwQ isolated probe done ==="; date
# 2. ensure QwQ served on 8141 (reuse; re-serve if down)
if ! curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M"; then
  echo "QwQ down -> re-serving"
  for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT --enable-auto-tool-choice \
    --tool-call-parser hermes --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.92 \
    > $S/vllm_qwq_reserve.log 2>&1 &
  ok=0; for i in $(seq 1 180); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
  [ $ok = 1 ] || { echo "QWQ_SERVE_FAIL"; touch $S/night_gpu1_end; exit 1; }
fi
echo "=== QwQ SERVE_OK for agentic ==="; date
# 3. SMOKE (PAID small): nt=1, 12 tasks, floor, gpt-4.1 user-sim
cd $TB; rm -rf "$TB/data/simulations/qwq_smoke"
T0=$(date +%s)
$PY $T2/t2_run_gated.py --gate 0 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials 1 --num_tasks 12 --max_concurrency 6 \
  --save_to qwq_smoke || echo "SMOKE_RUN_ERR"
T1=$(date +%s); echo "=== SMOKE elapsed $((T1-T0))s ==="; date
# 4. smoke verdict: OK iff infra rate <0.4 and >=6 valid (agent acted, tool-calls parsed, no mass overflow)
SRES=$TB/data/simulations/qwq_smoke/results.json
VERDICT=$($PY - <<PYEOF
import json
try:
    s=json.load(open("$SRES"))["simulations"]; n=len(s)
    inf=sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None); valid=n-inf
    from collections import Counter
    print("SMOKE_STATS n=%d infra=%d valid=%d term=%s"%(n,inf,valid,dict(Counter(x.get("termination_reason") for x in s))))
    print("VERDICT_OK" if (n>0 and inf/n<0.4 and valid>=6) else "VERDICT_FAIL")
except Exception as e:
    print("VERDICT_FAIL exc",repr(e))
PYEOF
)
echo "$VERDICT"
if ! echo "$VERDICT" | grep -q VERDICT_OK; then
  echo "=== QwQ agentic SMOKE FAILED -> abort paid full-run (budget protected) ==="; date
  touch $S/night_gpu1_end; echo "NIGHT_GPU1_DONE (smoke-fail, no full)"; exit 0
fi
echo "=== SMOKE PASSED -> launch paid nt=4 full ==="; date
# 5. FULL nt=4 floor (PAID)
SAVE=qwq32b_floor_retail_t4
cd $TB; rm -rf "$TB/data/simulations/$SAVE"
$PY $T2/t2_run_gated.py --gate 0 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials 4 --max_concurrency 8 --save_to "$SAVE" || echo "ARM_FAIL $SAVE"
echo "=== FULL done ==="; date
# 6. metrics + persist (even if partial)
RES=$TB/data/simulations/$SAVE/results.json
[ -f "$RES" ] && $PY - <<PYEOF
import json,sys
from collections import Counter
s=json.load(open("$RES"))["simulations"]
inf=sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
print("RESULT_qwq32b n=%d infra=%d term=%s"%(len(s),inf,dict(Counter(x.get("termination_reason") for x in s))))
sys.path.insert(0,"src")
from tau2.metrics.agent_metrics import compute_metrics
from tau2.data_model.simulation import Results
m=compute_metrics(Results.model_validate(json.load(open("$RES"))))
print("OFFICIAL_qwq32b pass_hat_ks:",m.pass_hat_ks,"avg_reward:",round(m.avg_reward,4))
PYEOF
PERSIST=$REPO/reports/facet_rft_2026/sim_results; mkdir -p $PERSIST
if [ -f "$RES" ]; then
  gzip -c "$RES" > $PERSIST/qwq32b_floor_retail_t4.results.json.gz
  cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
  git add -f $PERSIST/qwq32b_floor_retail_t4.results.json.gz
  git commit -q -m "persist: qwq32b_floor_retail_t4 (QwQ agentic floor nt4·paid·auto)" 2>/dev/null
  for try in 1 2 3; do git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && { echo PERSISTED_qwq32b; break; }; sleep 5; done
fi
touch $S/night_gpu1_end
echo "NIGHT_GPU1_DONE"; date
