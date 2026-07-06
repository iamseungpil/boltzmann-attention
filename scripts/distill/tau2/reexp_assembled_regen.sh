#!/bin/bash
# ★replay-safe 게이트 full 재런 (REPLAY_SAFE_GATE_DESIGN §7.2·리더보드-동일).
#   apply_gate_regen(생성-레벨 게이트·replay-clean) + nt=4(공식 프로토콜) + max-model-len 32768
#   (ContextWindow 방지·스모크 task20 교정). 공식 compute_metrics로 pass^1..4.
# 사용: reexp_assembled_regen.sh <GPU> <PORT> <MODEL> <TAG>
set -u
GPU=$1; PORT=$2; M=$3; TAG=$4
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
VLLM=/home/woori/venvs/tau2_vllm_env/bin/vllm; S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/reexp_regen_$TAG.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
$PY $T2/test_assembled_run.py || { echo "GATE_FAIL — abort"; exit 1; }
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
run () { local save=$1; shift
  echo "######## RUN $save env=$* ########"; date
  cd $TB; rm -rf "$TB/data/simulations/$save"
  env "$@" PYTHONPATH=src:$T2 $PY $T2/t2_run_gated.py --gate 1 --domain retail \
    --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
    --num_trials 4 --max_concurrency 8 --save_to "$save" || echo "ARM_FAIL $save"
  echo "ARM_DONE $save"; date; }
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $VLLM serve "$M" --port $PORT --enable-auto-tool-choice \
  --tool-call-parser hermes --max-model-len 32768 --enforce-eager --gpu-memory-utilization 0.92 \
  > $S/vllm_regen_$TAG.log 2>&1 &
ok=0; for i in $(seq 1 180); do curl -s localhost:$PORT/v1/models 2>/dev/null | grep -q "$M" && ok=1 && break; sleep 10; done
[ $ok = 1 ] || { echo "SERVE_FAIL"; tail -40 $S/vllm_regen_$TAG.log; exit 1; }
echo "SERVE_OK"; date
run ${TAG}_regen_retail_t4 \
  T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 \
  T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints \
  T2_PRESENT_READS=1 T2_PRESENT_NESTED=1 T2_CALC=1
for p in $(nvidia-smi --id=$GPU --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
# 즉석 요약 (공식 compute_metrics pass^1..4 + infra)
RES=$TB/data/simulations/${TAG}_regen_retail_t4/results.json
$PY - <<PYEOF
import json
from collections import Counter
s=json.load(open("$RES"))["simulations"]
term=Counter(x.get("termination_reason") for x in s)
inf=[int(x.get("task_id")) for x in s if (x.get("reward_info") or {}).get("reward") is None]
print("RESULT_${TAG} n=%d infra=%d termination=%s" % (len(s),len(inf),dict(term)))
print("infra_task_ids:", sorted(set(inf)))
try:
    import sys; sys.path.insert(0,"src")
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.data_model.simulation import Results
    m=compute_metrics(Results.model_validate(json.load(open("$RES"))))
    print("OFFICIAL_${TAG} pass_hat_ks:", m.pass_hat_ks, "avg_reward:", round(m.avg_reward,4))
except Exception as e:
    print("compute_metrics err:", repr(e)[:200])
PYEOF
# 영속화 (gitignore 우회·소실방지)
PERSIST=$REPO/reports/facet_rft_2026/sim_results; mkdir -p $PERSIST
if [ -f "$RES" ]; then
  gzip -c "$RES" > $PERSIST/${TAG}_regen_retail_t4.results.json.gz
  cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
  git add -f $PERSIST/${TAG}_regen_retail_t4.results.json.gz
  git commit -q -m "persist sim results: ${TAG}_regen_retail_t4 (replay-safe gate·nt4·auto)" 2>/dev/null
  for try in 1 2 3; do git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && { echo "PERSISTED_${TAG}"; break; }; sleep 5; done
else echo "PERSIST_SKIP_NO_RESULTS_${TAG}"; fi
echo "REEXP_REGEN_${TAG}_DONE"; date
