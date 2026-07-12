#!/bin/bash
# A2 = 일반화 스택 + DISAMB-enumerate(≥2 후보→deny+"후보 전부 사용자에 나열하고 물어라·재추측금지").
# = frontier robust 방식(enumerate-and-ask) 재현. list-order 관례 불필요·사용자 내용선택.
# ★측정 대상: trivial 무회귀(over-ask Δspurious≤0) ∧ most-recent 개선. 용법: bash generalized_enum.sh <TAG> <TASKS|ALL> <NT> <PORT>
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TAG="${1:?tag}"; TASKS="${2:?tasks|ALL}"; NT="${3:-1}"; PORT="${4:-8140}"
LOG=$S/genenum_${TAG}.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING_$PORT; exit 1; }
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue
export T2_CALC=1
export T2_EPLAN=1
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2
export T2_DISAMB=1 T2_DISAMB_MODE=enumerate     # ★≥2 후보→후보 전부 나열+사용자에 물음(재추측 금지)
unset T2_PRESENT_NESTED T2_PRESENT_READS T2_GROUND T2_PRINCIPLE_DEFAULT T2_AUTOFETCH T2_EPLAN_WALK T2_PROV_BADWORDS
cd $TB
TIDARG=""; [ "$TASKS" != "ALL" ] && TIDARG="--task_ids $TASKS"
rm -rf "$TB/data/simulations/genenum_$TAG"
timeout 10800 $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 8 \
  --save_to "genenum_$TAG" $TIDARG || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
echo "== audit(비트레이스) =="
echo "DISAMB-enum fired(≥2→ask): $(grep -aE '\[T2_DISAMB\]' $LOG | grep -avc '^+') | ENUM feedback: $(grep -aoE 'DISAMBIGUATE-ASK' $LOG | wc -l)"
echo "★override subcall switch(0): $(grep -aE 'SUBCALL switch' $LOG | grep -avc '^+')"
$PY - "$TB/data/simulations/genenum_$TAG/results.json" <<'PYEOF'
import json,sys,os
p=sys.argv[1]
if not os.path.exists(p): print("no results"); raise SystemExit
d=json.load(open(p)); s=d["simulations"]
TRIV={"0","1","2","5","6","7","9","11","13","14","18","23","25","26","28","30","43","44","45","48","50","51","55","65","67","68","70","75","78","80","85","88","90","92","106","113"}
for x in sorted(s,key=lambda z:int(z["task_id"])):
    ri=x.get("reward_info") or {}; db=(ri.get("db_check") or {}).get("db_match")
    print("  [%s] task %4s db=%s"%("t" if str(x["task_id"]) in TRIV else "h",x["task_id"],db))
tr=[x for x in s if str(x["task_id"]) in TRIV]; hd=[x for x in s if str(x["task_id"]) not in TRIV]
def pr(g):
    v=[1 if ((x.get('reward_info') or {}).get('db_check') or {}).get('db_match') else 0 for x in g]
    return "%d/%d"%(sum(v),len(v)) if v else "-"
allv=[1 if ((x.get('reward_info') or {}).get('db_check') or {}).get('db_match') else 0 for x in s]
print("★ trivial(over-ask체크): %s | hard(개선체크): %s | 전체 db: %d/%d"%(pr(tr),pr(hd),sum(allv),len(allv)))
PYEOF
echo ALLDONE; date
