#!/bin/bash
# Step 3 (2026-07-08): QwQ agentic floor + reasoning-parser, nt=4 FULL retail, gpt-4.1 user-sim.
# Definitive pass^1..4 vs base fl32b_floor (0.557/0.411/0.358/0.333). QwQ already served on 8141 w/ reasoning-parser.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench; PORT=8141; M=Qwen/QwQ-32B-AWQ
LOG=$S/reexp_qwq_rparser_nt4.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --ff-only 2>&1 | tail -1
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s localhost:$PORT/v1/models | grep -q "$M" || { echo "QWQ_NOT_SERVED"; exit 1; }
echo "SERVE_OK (reasoning-parser)"; date
SAVE=qwq_rparser_floor_t4
cd $TB; rm -rf "$TB/data/simulations/$SAVE"
$PY $T2/t2_run_gated.py --gate 0 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials 4 --max_concurrency 8 --save_to "$SAVE" || echo "ARM_FAIL"
echo "ARM_DONE"; date
RES=$TB/data/simulations/$SAVE/results.json
[ -f "$RES" ] && $PY - <<PYEOF
import json,sys
from collections import Counter
s=json.load(open("$RES"))["simulations"]
inf=sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
WRITE={"return_delivered_order_items","exchange_delivered_order_items","cancel_pending_order","modify_pending_order_items","modify_pending_order_address","modify_pending_order_payment","modify_user_address","place_order"}
def tn(x): return [tc.get("name") for m in x.get("messages",[]) if m.get("role")=="assistant" for tc in (m.get("tool_calls") or [])]
LEAK=["</tool_call>","<|im_start|>","<|im_end|>","<tool_call>"]
def leak(x): return any(any(k in (m.get("content") or "") for k in LEAK) for m in x.get("messages",[]) if m.get("role")=="assistant")
zeros=[x for x in s if (x.get("reward_info") or {}).get("reward")==0]
noexec=sum(1 for x in zeros if not [n for n in tn(x) if n in WRITE])
print("RESULT_qwqrp_nt4 n=%d infra=%d zero=%d noexec=%d leak=%d"%(len(s),inf,len(zeros),noexec,sum(1 for x in s if leak(x))))
sys.path.insert(0,"src")
from tau2.metrics.agent_metrics import compute_metrics
from tau2.data_model.simulation import Results
m=compute_metrics(Results.model_validate(json.load(open("$RES"))))
print("OFFICIAL_qwqrp pass_hat_ks:",m.pass_hat_ks,"avg_reward:",round(m.avg_reward,4))
PYEOF
PERSIST=$REPO/reports/facet_rft_2026/sim_results; mkdir -p $PERSIST
if [ -f "$RES" ]; then
  gzip -c "$RES" > $PERSIST/${SAVE}.results.json.gz
  cd $REPO && git pull --rebase --autostash -q origin facet-rft-2026 2>/dev/null
  git add -f $PERSIST/${SAVE}.results.json.gz
  git commit -q -m "persist: ${SAVE} (QwQ reasoning-parser floor nt4·Step3 definitive·auto)" 2>/dev/null
  for t in 1 2 3; do git pull --rebase --autostash -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && { echo PERSISTED; break; }; sleep 5; done
fi
touch $S/qwqrp_nt4_end; echo "STEP3_DONE"; date
