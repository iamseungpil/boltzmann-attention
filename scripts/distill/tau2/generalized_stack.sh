#!/bin/bash
# 일반화 스택 (GENERALIZED_SCAFFOLD_ARCHITECTURE §6): 가드+GET/FIND/ASK루프+coverage. override/트릭 OFF.
# ON: gate·prov-regen(rescue=지시루프)·calc·eplan(coverage,walk off)·cap. OFF: present·ground·disamb·principle·autofetch·eplan_walk.
# 용법: bash generalized_stack.sh <TAG> <TASKS|ALL> <NT> <PORT>
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TAG="${1:?tag}"; TASKS="${2:?tasks|ALL}"; NT="${3:-1}"; PORT="${4:-8141}"
LOG=$S/gen_${TAG}.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING_$PORT; exit 1; }
# ── 일반화 스택 env ──
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue
export T2_CALC=1
export T2_EPLAN=1
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2
unset T2_PRESENT_NESTED T2_PRESENT_READS T2_GROUND T2_DISAMB T2_DISAMB_MODE T2_PRINCIPLE_DEFAULT T2_AUTOFETCH T2_EPLAN_WALK T2_PROV_BADWORDS
cd $TB
TIDARG=""; [ "$TASKS" != "ALL" ] && TIDARG="--task_ids $TASKS"
rm -rf "$TB/data/simulations/gen_$TAG"
timeout 5400 $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 6 \
  --save_to "gen_$TAG" $TIDARG || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
# ── 레버-발화 검증 (스모크 핵심) ──
echo "== lever fire audit =="
echo "prov-regen(GET/FIND/ASK): $(grep -cE '\[T2_PROV\] regen fired' $LOG)"
echo "prov-rescue passthrough : $(grep -cE 'rescue pass-through' $LOG)"
echo "eplan coverage deny     : $(grep -cE '\[T2_EPLAN\].*deny' $LOG)"
echo "★override markers(0이어야): DISAMB=$(grep -cE 'T2_DISAMB' $LOG) GROUND=$(grep -cE 'T2_GROUND\] sub' $LOG) PRESENT_NESTED=$(grep -cE 'present.*nested|DISAMBIGUATION NOTE' $LOG) PRINCIPLE=$(grep -cE 'T2_PRINCIPLE' $LOG)"
# persist + summary (full 모드만)
if [ "$TASKS" = "ALL" ] || [ "$NT" -ge 2 ]; then
  gzip -c "$TB/data/simulations/gen_$TAG/results.json" > $REPO/reports/facet_rft_2026/sim_results/gen_$TAG.results.json.gz 2>/dev/null
  cd $REPO && git add -f reports/facet_rft_2026/sim_results/gen_$TAG.results.json.gz 2>/dev/null
  git commit -q -m "persist: generalized stack $TAG (auto)" 2>/dev/null
  git pull --rebase -q origin facet-rft-2026 2>/dev/null; git push -q origin facet-rft-2026 && echo PERSISTED
fi
$PY - "$TB/data/simulations/gen_$TAG/results.json" <<'PYEOF'
import json,sys,os
p=sys.argv[1]
if not os.path.exists(p): print("no results"); raise SystemExit
d=json.load(open(p)); s=d["simulations"]
inf=sum(1 for x in s if (x.get("reward_info") or {}).get("reward") is None)
db=[1 if ((x.get("reward_info") or {}).get("db_check") or {}).get("db_match") else 0 for x in s]
print("GEN %s: n=%d infra=%d db-pass=%.3f"%(sys.argv[1].split('/')[-2],len(s),inf,sum(db)/len(db) if db else 0))
fails=sorted(set(str(x["task_id"]) for x in s if not ((x.get("reward_info") or {}).get("db_check") or {}).get("db_match")),key=lambda z:int(z))
print("db_fail tasks:",fails)
PYEOF
echo ALLDONE; date
