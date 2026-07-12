#!/bin/bash
# E1 E-PLAN 활성화 SMOKE — v25c 스택 + write-cap + T2_EPLAN(+WALK). t0(무회귀)+t41(다중-coverage) 2 sims.
# GO = [T2_EPLAN] discovery/walk 발화 ∧ 크래시/infra 0 ∧ t0 무회귀.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"; PORT=8140
exec > $S/eplan_smoke.log 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING; exit 1; }
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PRESENT_NESTED=1 T2_CALC=1
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue T2_GROUND=1
export T2_DISAMB=1 T2_DISAMB_MODE=subcall
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2
export T2_EPLAN=1 T2_EPLAN_WALK=1   # ★E1
unset T2_PRESENT_READS T2_AUTOFETCH T2_PROV_BADWORDS
TASKS="0,41"   # t0=무회귀·t41=gold 4-write 중 1 MISSING(다중-coverage → E-PLAN discovery/walk 발화 기대)
cd $TB; rm -rf "$TB/data/simulations/eplan_smoke"
$PY $T2/t2_run_gated.py --gate 1 --domain retail \
  --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  --num_trials 1 --max_concurrency 2 --num_tasks 2 --save_to eplan_smoke --task_ids "$TASKS" || echo "ARM_FAIL eplan_smoke"
echo "SMOKE_RUN_DONE"; date
echo "== [T2_EPLAN] 발화 =="; grep -c "T2_EPLAN" $S/eplan_smoke.log || true
grep "T2_EPLAN" $S/eplan_smoke.log | head -5 || true
echo "== [T2_WRITE_CAP] =="; grep -c "T2_WRITE_CAP. capped" $S/eplan_smoke.log || true
$PY - "$TB/data/simulations/eplan_smoke/results.json" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); s=d["simulations"]
inf=sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
for x in sorted(s,key=lambda z:int(z["task_id"])):
    ri=x.get("reward_info") or {}; dc=ri.get("db_check")
    print("ESMOKE t%-4s r=%s db=%s term=%s"%(x["task_id"],ri.get("reward"),(dc.get("db_match") if isinstance(dc,dict) else None),x.get("termination_reason")))
print("ESMOKE_TOTAL n=%d infra=%d"%(len(s),inf))
PYEOF
echo "SMOKE_ALLDONE"; date
