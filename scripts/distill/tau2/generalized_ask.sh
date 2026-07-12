#!/bin/bash
# A2 = 일반화 스택 + DISAMB-dialog(≥2 후보→deny+advise "사용자 의도 재확인"=ASK-advise·override 아님).
# = INFER→ASK 규칙의 안전 근사. "가장 최근"류=주문 ≥2→ASK로 해소 시도. 부작용(trivial 회귀) 체크.
# 용법: bash generalized_ask.sh <TAG> <TASKS|ALL> <NT> <PORT>
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TAG="${1:?tag}"; TASKS="${2:?tasks|ALL}"; NT="${3:-1}"; PORT="${4:-8140}"
LOG=$S/genask_${TAG}.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING_$PORT; exit 1; }
# ── 일반화 스택 + DISAMB-dialog(ASK-advise) ──
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue
export T2_CALC=1
export T2_EPLAN=1
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2
export T2_DISAMB=1 T2_DISAMB_MODE=dialog     # ★≥2 후보→deny+advise(ASK)·subcall-override 아님
unset T2_PRESENT_NESTED T2_PRESENT_READS T2_GROUND T2_PRINCIPLE_DEFAULT T2_AUTOFETCH T2_EPLAN_WALK T2_PROV_BADWORDS
cd $TB
TIDARG=""; [ "$TASKS" != "ALL" ] && TIDARG="--task_ids $TASKS"
rm -rf "$TB/data/simulations/genask_$TAG"
timeout 10800 $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 8 \
  --save_to "genask_$TAG" $TIDARG || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
echo "== lever audit(비트레이스) =="
echo "DISAMB-dialog fired(≥2→advise): $(grep -aE '\[T2_DISAMB\]|DISAMBIGUATE' $LOG | grep -avc '^+')"
echo "prov-regen: $(grep -aE '\[T2_PROV\] regen fired' $LOG | grep -avc '^+') | eplan deny: $(grep -aE '\[T2_EPLAN\].*deny' $LOG | grep -avc '^+')"
echo "★override(0이어야) subcall switch: $(grep -aE 'SUBCALL switch' $LOG | grep -avc '^+')"
$PY - "$TB/data/simulations/genask_$TAG/results.json" <<'PYEOF'
import json,sys,os
p=sys.argv[1]
if not os.path.exists(p): print("no results"); raise SystemExit
d=json.load(open(p)); s=d["simulations"]
TRIV={"0","1","2","5","6","7","9","11","13","14","18","23","25","26","28","30","43","44","45","48","50","51","55","65","67","68","70","75","78","80","85","88","90","92","106","113"}
print("per-task db (t=trivial·h=hard):")
for x in sorted(s,key=lambda z:int(z["task_id"])):
    ri=x.get("reward_info") or {}; db=(ri.get("db_check") or {}).get("db_match")
    tag="t" if str(x["task_id"]) in TRIV else "h"
    print("  [%s] task %4s db=%s"%(tag,x["task_id"],db))
tr=[x for x in s if str(x["task_id"]) in TRIV]; hd=[x for x in s if str(x["task_id"]) not in TRIV]
def pr(g):
    v=[1 if ((x.get('reward_info') or {}).get('db_check') or {}).get('db_match') else 0 for x in g]
    return "%d/%d"%(sum(v),len(v)) if v else "0/0"
print("★ trivial(부작용체크): %s  |  hard(개선체크): %s"%(pr(tr),pr(hd)))
print("  GO = trivial 무회귀(전부 pass) ∧ hard가 ASK로 개선")
PYEOF
echo ALLDONE; date
