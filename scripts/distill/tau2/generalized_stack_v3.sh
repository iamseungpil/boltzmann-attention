#!/bin/bash
# A1-v3 (A1_V3_DESIGN §9 반영·4 수정): A1-v2 + coverage완전화 + filter-scope + L4(fixed) + reads-only조건부.
#   = gate+prov(rescue)+calc+eplan+cap + examined-safe + reads-only(조건부·in-scope 허용)
#     + DISAMB-filter(order 전담·new_item 제외·코드수정) + T2_L4(제품별스코핑+floor-guard)
#     + T2_EPLAN_WALK(종료-시 coverage 완전화·A1-v2 지배실패 8건 겨냥).
# 용법: bash generalized_stack_v3.sh <TAG> <TASKS|ALL> <NT> <PORT>
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TAG="${1:?tag}"; TASKS="${2:?tasks|ALL}"; NT="${3:-1}"; PORT="${4:-8141}"
LOG=$S/genv3_${TAG}.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING_$PORT; exit 1; }
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue
export T2_CALC=1
export T2_EPLAN=1
export T2_EPLAN_EXAMINED_SAFE=1
export T2_EPLAN_READS_ONLY=1                     # #4 조건부(코드서 in-scope 허용)
export T2_EPLAN_WALK=1                           # ★#1 coverage 완전화(종료-시 gap 리마인더)
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2
export T2_DISAMB=1 T2_DISAMB_MODE=enumerate T2_DISAMB_ORDER=1   # filter(order 전담·new_item 제외=코드)
export T2_L4=1                                   # ★#3 L4(제품별스코핑+floor-guard·수정본)
unset T2_PROV_ADDR_FULL T2_PRESENT_NESTED T2_GROUND T2_PRINCIPLE_DEFAULT T2_AUTOFETCH T2_PROV_BADWORDS
echo "A1-v3: examined_safe reads_only(cond) EPLAN_WALK filter(order) L4(fixed) | prov=rescue cap=2"
cd $TB
TIDARG=""; [ "$TASKS" != "ALL" ] && TIDARG="--task_ids $TASKS"
rm -rf "$TB/data/simulations/genv3_$TAG"
timeout 14400 $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 10 \
  --save_to "genv3_$TAG" $TIDARG || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
RES="$TB/data/simulations/genv3_$TAG/results.json"
if [ -f "$RES" ]; then
  gzip -c "$RES" > $REPO/reports/facet_rft_2026/sim_results/genv3_$TAG.results.json.gz
  cd $REPO && git add -f reports/facet_rft_2026/sim_results/genv3_$TAG.results.json.gz && \
    git commit -q -m "persist A1-v3 $TAG (auto)" && git pull --rebase -q origin facet-rft-2026 && git push -q origin facet-rft-2026 && echo PERSISTED
fi
echo "== A1-v3 audit =="
echo "walk gap: $(grep -aE 'T2_EPLAN\] walk gap' $LOG|grep -avc '^+') | examined-safe: $(grep -aE 'examined-safe' $LOG|grep -avc '^+') | L4 sub: $(grep -aE 'T2_L4\] substituted' $LOG|grep -avc '^+') keep: $(grep -aF 'L4 keep' $LOG|grep -avc '^+') | FSUB new_item(0이어야): $(grep -aE 'FSUB\] substituted arg=new_item' $LOG|grep -avc '^+')"
$PY - "$RES" <<'PYEOF'
import json,sys,os
p=sys.argv[1]
if not os.path.exists(p): print("no results"); raise SystemExit
d=json.load(open(p)); s=d["simulations"]
from collections import defaultdict
by=defaultdict(list)
for x in s:
    db=((x.get("reward_info") or {}).get("db_check") or {}).get("db_match")
    by[str(x["task_id"])].append(1 if db else 0)
COV={"35","41","54","64","76","81","92","97"}   # A1-v2 coverage-miss 표적
allv=[v for vs in by.values() for v in vs]
print("★ 전체 db: %d/%d = %.3f (A1-v2=0.732 부분·A1=0.623)"%(sum(allv),len(allv),sum(allv)/max(1,len(allv))))
print("coverage 표적(35·41·54·64·76·81·92·97):", " ".join("t%s=%d/%d"%(k,sum(by[k]),len(by[k])) for k in sorted(COV,key=int) if k in by))
PYEOF
echo ALLDONE; date
