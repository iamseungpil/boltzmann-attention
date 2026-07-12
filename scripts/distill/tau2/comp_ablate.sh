#!/bin/bash
# COMP 증분 ablation (2026-07-13·단일-요인) — COMP에서 한 factor씩 쌓아 eplan/present 인과 분리.
#   base = COMP−present = unified: gate + prov(FULL) + calc, present OFF, eplan OFF, cap OFF.
#   EPLAN=1 인자로 eplan만 추가 → arm1(EPLAN=0) vs arm2(EPLAN=1) = eplan 효과(단일-요인·재혼재 0).
#   COMP(present 포함·comp_retail_t4 nt4)와 arm1 비교 = present 효과.
# ★prov=full(COMP값·A1의 rescue 아님)·cap 없음(COMP값)·present off(양 arm 공통) = eplan만 변수.
# 용법: bash comp_ablate.sh <TAG> <TASKS|ALL> <NT> <PORT> <EPLAN:0|1>
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TAG="${1:?tag}"; TASKS="${2:?tasks|ALL}"; NT="${3:-2}"; PORT="${4:-8140}"; EPLAN="${5:-0}"; PROVM="${6:-full}"
LOG=$S/compabl_${TAG}.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING_$PORT; exit 1; }
# ── COMP 코어(present 제외) ──
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=$PROVM   # ★param6: full(COMP)|rescue(prov 격리 arm)
export T2_CALC=1
# ── present/cap/기타 개입 OFF(양 arm 공통) ──
unset T2_PRESENT_NESTED T2_PRESENT_READS T2_WRITE_CAP T2_GROUND T2_DISAMB T2_DISAMB_MODE T2_PRINCIPLE_DEFAULT T2_AUTOFETCH T2_EPLAN_WALK T2_EPLAN_REPLAN T2_PROV_BADWORDS T2_PROV_ADDR_FULL T2_EPLAN_EXAMINED_SAFE T2_DISAMB_ORDER
# ── ★유일 변수: eplan ──
if [ "$EPLAN" = "1" ]; then export T2_EPLAN=1; else unset T2_EPLAN; fi
echo "ARM config: EPLAN=$EPLAN prov=$PROVM present=off cap=off"
cd $TB
TIDARG=""; [ "$TASKS" != "ALL" ] && TIDARG="--task_ids $TASKS"
rm -rf "$TB/data/simulations/compabl_$TAG"
timeout 10800 $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 8 \
  --save_to "compabl_$TAG" $TIDARG || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
# ── 영속화([[30]]: 결과 gitignore→즉시 gz 커밋) ──
RES="$TB/data/simulations/compabl_$TAG/results.json"
if [ -f "$RES" ]; then
  gzip -c "$RES" > $REPO/reports/facet_rft_2026/sim_results/compabl_$TAG.results.json.gz
  cd $REPO && git add -f reports/facet_rft_2026/sim_results/compabl_$TAG.results.json.gz && \
    git commit -q -m "persist compabl $TAG (EPLAN=$EPLAN·auto)" && git push -q origin facet-rft-2026 && echo PERSISTED
fi
echo "== audit =="
echo "eplan L2 deny: $(grep -aE 'T2_EPLAN\] L2 deny' $LOG | grep -avc '^+') | L1: $(grep -aE 'T2_EPLAN\] L1 deny' $LOG | grep -avc '^+') | prov regen: $(grep -aE 'T2_PROV\] regen fired' $LOG | grep -avc '^+')"
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
for k in sorted(by,key=int):
    v=by[k]; print("  task %4s: %d/%d %s"%(k,sum(v),len(v),"".join(map(str,v))))
allv=[x for v in by.values() for x in v]
print("★ 전체 db: %d/%d (%.1f%%) · 태스크수 %d"%(sum(allv),len(allv),100*sum(allv)/max(1,len(allv)),len(by)))
PYEOF
echo ALLDONE; date
