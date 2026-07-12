#!/bin/bash
# L4 fexec-variants 격리 probe (A1_V3_IMPL §2d). base=gate+prov(rescue)+calc·L4만 추가(isolate).
# 표적: frontier 바닥 변형(0·15·20·79) + 변형선택(52·77·110) + trivial 무회귀(1·2·6·113).
# 측정: [T2_L4] 발화·det/form·치환·per-case db. R1(field-det 비율)·R5(FP)·I7(floor).
# 용법: bash l4_probe.sh <TAG> <PORT>
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TAG="${1:?tag}"; PORT="${2:-8140}"
TASKS="0,15,20,52,77,79,110,1,2,6,113"
LOG=$S/l4probe_${TAG}.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING_$PORT; exit 1; }
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue
export T2_CALC=1
export T2_L4=1                                   # ★격리: L4만 추가
unset T2_EPLAN T2_WRITE_CAP T2_DISAMB T2_DISAMB_MODE T2_GROUND T2_PRINCIPLE_DEFAULT T2_AUTOFETCH T2_PRESENT_NESTED T2_EPLAN_EXAMINED_SAFE T2_EPLAN_READS_ONLY T2_DISAMB_ORDER
echo "L4 probe: T2_L4=1 base=gate+prov(rescue)+calc (eplan/disamb/cap OFF)"
cd $TB
rm -rf "$TB/data/simulations/l4probe_$TAG"
timeout 5400 $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials 1 --max_concurrency 8 \
  --save_to "l4probe_$TAG" --task_ids "$TASKS" || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
echo "== L4 audit =="
echo "[T2_L4] fired: $(grep -aE '\[T2_L4\]' $LOG | grep -avc '^+') | substituted: $(grep -aE 'T2_L4\] substituted' $LOG|grep -avc '^+') | confirmed: $(grep -aE 'T2_L4\] confirmed' $LOG|grep -avc '^+') | reverted: $(grep -aE 'T2_L4\] reverted' $LOG|grep -avc '^+')"
echo "[T2_FEXEC] L4 det: $(grep -aE 'FEXEC\] L4 .*\(det\)' $LOG|grep -avc '^+') | form: $(grep -aE 'FEXEC\] L4 .*\(form\)' $LOG|grep -avc '^+')"
grep -aoE '\[T2_L4\] substituted arg=[a-z_]+ from=[^ ]+ to=[^ ]+' $LOG | grep -av '^+' | head -15
$PY - "$TB/data/simulations/l4probe_$TAG/results.json" <<'PYEOF'
import json,sys,os
p=sys.argv[1]
if not os.path.exists(p): print("no results"); raise SystemExit
d=json.load(open(p)); s=d["simulations"]
L4T={"0","15","20","52","77","79","110"}; TRIV={"1","2","6","113"}
for x in sorted(s,key=lambda z:int(z["task_id"])):
    db=((x.get("reward_info") or {}).get("db_check") or {}).get("db_match")
    tag="L4" if str(x["task_id"]) in L4T else ("triv" if str(x["task_id"]) in TRIV else "?")
    print("  [%s] task %4s db=%s"%(tag,x["task_id"],db))
l4=[x for x in s if str(x["task_id"]) in L4T]; tr=[x for x in s if str(x["task_id"]) in TRIV]
def pr(g):
    v=[1 if ((x.get('reward_info') or {}).get('db_check') or {}).get('db_match') else 0 for x in g]
    return "%d/%d"%(sum(v),len(v))
print("★ L4표적: %s | trivial(무회귀): %s"%(pr(l4),pr(tr)))
PYEOF
echo ALLDONE; date
