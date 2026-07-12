#!/bin/bash
# B 개선 전제검증: COMP+D-v2+cap 스택이 36 trivial(COMP=1.000)을 회귀시키나. nt=1·36 sims.
# GO = db-pass ≈ 1.0 유지(DISAMB/E-PLAN over-fire로 인한 회귀 0). 회귀 시 그 레버 regression-safe화 필요.
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=/home/woori/scratch/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"; PORT=8140
exec > $S/trivial36_check.log 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING; exit 1; }
# b78c(COMP+D-v2) 스택 + write-cap
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PRESENT_NESTED=1 T2_CALC=1 T2_PRINCIPLE_DEFAULT=1
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue T2_GROUND=1
export T2_DISAMB=1 T2_DISAMB_MODE=subcall
export T2_EPLAN=1 T2_EPLAN_WALK=1
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2
unset T2_PRESENT_READS T2_AUTOFETCH T2_PROV_BADWORDS
TASKS="0,1,2,5,6,7,9,11,13,14,18,23,25,26,28,30,43,44,45,48,50,51,55,65,67,68,70,75,78,80,85,88,90,92,106,113"
cd $TB; rm -rf "$TB/data/simulations/trivial36_check"
$PY $T2/t2_run_gated.py --gate 1 --domain retail \
  --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 \
  --num_trials 1 --max_concurrency 8 --num_tasks 36 --save_to trivial36_check --task_ids "$TASKS" || echo "ARM_FAIL trivial36"
echo "RUN_DONE"; date
$PY - "$TB/data/simulations/trivial36_check/results.json" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); s=d["simulations"]
inf=sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
dbp=[1.0 if ((x.get("reward_info") or {}).get("db_check") or {}).get("db_match") else 0.0 for x in s]
fails=[x["task_id"] for x in s if not ((x.get("reward_info") or {}).get("db_check") or {}).get("db_match")]
print("TRIVIAL36 n=%d infra=%d db-pass=%.3f (COMP=1.000)"%(len(s),inf,sum(dbp)/len(dbp)))
print("회귀(db_fail) tasks:", sorted(fails,key=lambda z:int(z)))
PYEOF
cp "$TB/data/simulations/trivial36_check/results.json" /dev/null 2>/dev/null
gzip -c "$TB/data/simulations/trivial36_check/results.json" > $REPO/reports/facet_rft_2026/sim_results/trivial36_check.results.json.gz 2>/dev/null
cd $REPO; git add -f reports/facet_rft_2026/sim_results/trivial36_check.results.json.gz 2>/dev/null
git commit -q -m "persist: trivial36_check (36 회귀검증·auto)" 2>/dev/null
git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && echo "PERSISTED"
echo "ALLDONE"; date
