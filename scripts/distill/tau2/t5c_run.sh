#!/bin/bash
# T5-C COMP+D-v2 파라미터화 러너 (단계적 nt=1-사이클 프로토콜·§7c).
#   silent 전 레버(P-A GROUND·P-B subcall·P-C rescue) + 게이트6종 + calc/nested.
# usage: t5c_run.sh <TAG> <TASK_IDS|ALL> <NT>
#   예: t5c_run.sh v25c 0,17,40,47,61,95 1   |  t5c_run.sh b13c1 <13목록> 1  |  t5c_run.sh full_c1 ALL 1
set -u
TAG=${1:?tag}; TASK_IDS=${2:?tasks|ALL}; NT=${3:-1}
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"; PORT=8140
LOG=$S/t5c_${TAG}.log
exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING; exit 1; }
# COMP+D-v2 (rev4: constraints 유지·silent)
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PRESENT_NESTED=1 T2_CALC=1
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue T2_GROUND=1
export T2_DISAMB=1 T2_DISAMB_MODE=subcall
unset T2_PRESENT_READS T2_AUTOFETCH T2_PROV_BADWORDS
SAVE=t5c_${TAG}
cd $TB; rm -rf "$TB/data/simulations/$SAVE"
TIDARG=""; [ "$TASK_IDS" != "ALL" ] && TIDARG="--task_ids $TASK_IDS"
$PY $T2/t2_run_gated.py --gate 1 --domain retail \
  --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  --num_trials $NT --max_concurrency 8 --save_to "$SAVE" $TIDARG || echo "ARM_FAIL $TAG"
echo "T5C_RUN_DONE $TAG"; date
echo "== lever fire =="; grep -cE "T2_DISAMB.*subcall" $LOG; grep -cE "SUBCALL switch" $LOG; grep -cE "rescue pass-through" $LOG; grep -cE "R8 strip" $LOG
RES=$TB/data/simulations/$SAVE/results.json
$PY - "$RES" "$TAG" <<'PYEOF'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1])); s = d["simulations"]; tag = sys.argv[2]
WR = ("modify", "exchange", "return", "cancel")
inf = sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
nowrite = 0
for x in s:
    nw = sum(1 for m in (x.get("messages") or []) for tc in (m.get("tool_calls") or [])
             if any(w in (tc.get("name") or "") for w in WR))
    if nw == 0: nowrite += 1
try:
    import sys as _s; _s.path.insert(0, "src")
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.data_model.simulation import Results
    m = compute_metrics(Results.model_validate(d))
    pk = m.pass_hat_ks; ar = round(m.avg_reward, 4)
except Exception as e:
    pk = {}; ar = None; print("metrics_err", e)
db = sum(1 for x in s if isinstance((x.get("reward_info") or {}).get("db_check"), dict)
         and (x["reward_info"]["db_check"] or {}).get("db_match"))
print("T5C_SUMMARY %s n=%d infra=%d nowrite=%d db_pass=%d avg_reward=%s pass_hat=%s term=%s"
      % (tag, len(s), inf, nowrite, db, ar, pk, dict(Counter(x.get("termination_reason") for x in s))))
# per-task (6/13-샘플만 짧게)
if len(s) <= 20:
    for x in sorted(s, key=lambda z: (int(z["task_id"]), z.get("trial") or 0)):
        ri = x.get("reward_info") or {}
        dc = ri.get("db_check")
        print("  t%s tr%s r=%s db=%s" % (x["task_id"], x.get("trial"),
              ri.get("reward"), (dc.get("db_match") if isinstance(dc, dict) else None)))
PYEOF
gzip -c "$RES" > $REPO/reports/facet_rft_2026/sim_results/${SAVE}.results.json.gz
cd $REPO
git add -f reports/facet_rft_2026/sim_results/${SAVE}.results.json.gz
git commit -q -m "persist sim results: ${SAVE} (T5-C staged, auto)" 2>/dev/null
git pull --rebase -q origin facet-rft-2026 2>/dev/null
git push -q origin facet-rft-2026 && echo "PERSISTED_${TAG}"
echo "T5C_ALLDONE_${TAG}"; date
