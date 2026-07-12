#!/bin/bash
# A1-v2 (A1_IMPROVEMENT_DESIGN_2026_07_13): A1 + 구현된·저위험 3 레버.
#   = A1(gate+prov-rescue+calc+eplan+cap) + examined-safe(harm-I) + reads-only(harm-III)
#     + DISAMB-filter(dotted-fixed·order선택). 제외: prov-addr(철회)·present(anti-drift).
# 목표: nt=1 재실험 db vs A1 0.623 + 표적 per-case(58·32·55·59·101) + help-8 무회귀.
# 용법: bash generalized_stack_v2.sh <TAG> <TASKS|ALL> <NT> <PORT>
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TAG="${1:?tag}"; TASKS="${2:?tasks|ALL}"; NT="${3:-1}"; PORT="${4:-8141}"
LOG=$S/genv2_${TAG}.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING_$PORT; exit 1; }
# ── A1 코어 ──
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue
export T2_CALC=1
export T2_EPLAN=1
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2
# ── ★A1-v2 개선 3 ──
export T2_EPLAN_EXAMINED_SAFE=1                  # (1) harm-I qty-conflation transfer(t58)
export T2_EPLAN_READS_ONLY=1                     # (2) harm-III 과-행동(t32)
export T2_DISAMB=1 T2_DISAMB_MODE=enumerate T2_DISAMB_ORDER=1   # (3) filter-substitute(dotted-fixed·order선택)
# ── 제외(철회/anti-drift) ──
unset T2_PROV_ADDR_FULL T2_PRESENT_NESTED T2_PRESENT_READS T2_GROUND T2_PRINCIPLE_DEFAULT T2_AUTOFETCH T2_EPLAN_WALK T2_PROV_BADWORDS
echo "A1-v2: examined_safe=1 reads_only=1 disamb=enumerate+order | prov=rescue eplan=1 cap=2 present=off"
cd $TB
TIDARG=""; [ "$TASKS" != "ALL" ] && TIDARG="--task_ids $TASKS"
rm -rf "$TB/data/simulations/genv2_$TAG"
timeout 14400 $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 10 \
  --save_to "genv2_$TAG" $TIDARG || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
RES="$TB/data/simulations/genv2_$TAG/results.json"
if [ -f "$RES" ]; then
  gzip -c "$RES" > $REPO/reports/facet_rft_2026/sim_results/genv2_$TAG.results.json.gz
  cd $REPO && git add -f reports/facet_rft_2026/sim_results/genv2_$TAG.results.json.gz && \
    git commit -q -m "persist A1-v2 $TAG (auto)" && git pull --rebase -q origin facet-rft-2026 && git push -q origin facet-rft-2026 && echo PERSISTED
fi
echo "== audit =="
echo "examined-safe 발화: $(grep -aE 'T2_EPLAN\] examined-safe' $LOG|grep -avc '^+') | eplan L2 deny: $(grep -aE 'T2_EPLAN\] L2 deny' $LOG|grep -avc '^+') | FSUB one: $(grep -aF '[T2_FEXEC] filter one' $LOG|grep -avc '^+') sub: $(grep -aE 'T2_FSUB\] substituted' $LOG|grep -avc '^+')"
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
TARGET={"58","32","55","59","101","83","69","92","112"}
HELP8={"21","23","33","42","52","94","108","110"}
allv=[v for vs in by.values() for v in vs]
print("★ 전체 db: %d/%d = %.3f (A1=0.623)"%(sum(allv),len(allv),sum(allv)/max(1,len(allv))))
print("표적(order/eplan해소 기대):", " ".join("t%s=%d/%d"%(k,sum(by[k]),len(by[k])) for k in sorted(TARGET,key=int) if k in by))
print("help-8(무회귀 확인):", " ".join("t%s=%d/%d"%(k,sum(by[k]),len(by[k])) for k in sorted(HELP8,key=int) if k in by))
PYEOF
echo ALLDONE; date
