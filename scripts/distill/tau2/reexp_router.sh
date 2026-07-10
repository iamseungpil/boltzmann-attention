#!/bin/bash
# ★T5-B 라우터 arm: floor+prov(C53 invocation 정확 재현) + T2_DISAMB=1 (E-AMB T5).
#   기준선 = prov_e2e_retail_t4 (C53·0.580). GO: pass↑ ∧ over-block 0 ∧ Δspurious≤0 ∧ tme 폭증 없음.
#   사용: reexp_router.sh <smoke|full> <TAG>   (기동 중인 8140 Qwen32B 재사용·serve 없음)
set -u
MODE=$1; TAG=$2
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
LOG=$S/reexp_router_$TAG.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_DISAMB=1 T2_GATE_KINDS=__none__
unset T2_PROV_GROUND T2_PRESENT_READS T2_PRESENT_NESTED T2_AUTOFETCH T2_CALC T2_GATE_REGEN T2_PROV_BADWORDS
curl -s localhost:8140/v1/models | grep -q Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 || { echo SERVE_MISSING; exit 1; }
if [ "$MODE" = smoke ]; then EXTRA="--num_tasks 10 --num_trials 1"; SAVE=${TAG}_smoke
else EXTRA="--num_trials 4"; SAVE=${TAG}_retail_t4; fi
cd $TB; rm -rf "$TB/data/simulations/$SAVE"
$PY $T2/t2_run_gated.py --gate 1 --domain retail \
  --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:8140/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  $EXTRA --max_concurrency 8 --save_to "$SAVE" || echo "ARM_FAIL $SAVE"
echo "ARM_DONE $SAVE"; date
echo "DISAMB_FIRED=$(grep -c 'T2_DISAMB. fired' $LOG)" ; echo "DISAMB_SWITCHED=$(grep -c 'T2_DISAMB. switched' $LOG)"
RES=$TB/data/simulations/$SAVE/results.json
$PY - <<PYEOF
import json
from collections import Counter
d=json.load(open("$RES")); s=d["simulations"]
inf=sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
tme=sum(1 for x in s if x.get("termination_reason")=="too_many_errors")
print("RESULT_${TAG} n=%d infra=%d tme=%d term=%s" % (len(s),inf,tme,dict(Counter(x.get("termination_reason") for x in s))))
rw=[(x.get("reward_info") or {}).get("reward") for x in s]
rw=[r for r in rw if r is not None]
print("avg_reward %.4f (n=%d)" % (sum(rw)/max(len(rw),1), len(rw)))
import sys; sys.path.insert(0,"src")
try:
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.data_model.simulation import Results
    m=compute_metrics(Results.model_validate(d))
    print("OFFICIAL_${TAG} pass_hat_ks:", m.pass_hat_ks, "avg_reward:", round(m.avg_reward,4))
except Exception as e:
    print("metrics_err", type(e).__name__, e)
PYEOF
PERSIST=$REPO/reports/facet_rft_2026/sim_results; mkdir -p $PERSIST
if [ -f "$RES" ] && [ "$MODE" = full ]; then
  gzip -c "$RES" > $PERSIST/${SAVE}.results.json.gz
  cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
  git add -f $PERSIST/${SAVE}.results.json.gz
  git commit -q -m "persist sim results: ${SAVE} (router=prov+disamb nt4 auto)" 2>/dev/null
  for try in 1 2 3; do git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && { echo "PERSISTED_${TAG}"; break; }; sleep 10; done
fi
echo ALL_DONE; date
