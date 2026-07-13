#!/bin/bash
# RESOLVE 선별 arm (사용자 지시 2026-07-13: v2 대비 측정-효과 있는 것만):
#   = v2-core + READALL(L2R·NT2 §4-1) + L10-membership(t94 인과 후보).
#   제외: L3(발화0)·G-noop(발화1·효과0)·L4-keep(v2에 L4 없음=무의미)·walk(폐기).
#   = A1-v2(gate+prov-rescue+calc+eplan+cap+examined-safe+reads-only+disamb-filter)
#     + L4a 치환 정지(T2_L4_MODE=keep·t58 회귀 제거) + L3 origin-prov(t97 확인-세탁 차단)
#     + CONSISTENCY(L10 멤버십 t35형 + G-noop t71형·예방) . walk=폐기(포렌식 확정).
# 용법: bash generalized_stack_v5.sh <TAG> <TASKS|ALL> <NT> <PORT>
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TAG="${1:?tag}"; TASKS="${2:?tasks|ALL}"; NT="${3:-1}"; PORT="${4:-8141}"; DOMAIN="${5:-retail}"
LOG=$S/genrz_${TAG}.log; exec > $LOG 2>&1; set -x; date
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
export T2_EPLAN_EXAMINED_SAFE=1
export T2_EPLAN_READS_ONLY=1
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2
export T2_DISAMB=1 T2_DISAMB_MODE=enumerate T2_DISAMB_ORDER=1
# ── ★v4 가드 (v3.2·PROBE_FORENSIC §4 step 0~4) ──
export T2_RESOLVE=1                       # ★통일 인터프리터(operator/membership/provenance 통합)
# ── 폐기·제외 ──
unset T2_EPLAN_WALK T2_EPLAN_REPLAN T2_PROV_ADDR_FULL T2_PRESENT_NESTED T2_PRESENT_READS T2_GROUND T2_PRINCIPLE_DEFAULT T2_AUTOFETCH T2_PROV_BADWORDS
echo "RESOLVE(선별): v2-core + READALL(cap2) + L10-membership only | L3/G-noop/L4 제외(v2 대비 효과 0 실측)"
cd $TB
TIDARG=""; [ "$TASKS" != "ALL" ] && TIDARG="--task_ids ${TASKS// /,}"   # 공백→콤마(파서=콤마 구분)
rm -rf "$TB/data/simulations/genrz_$TAG"
timeout 14400 $PY $T2/t2_run_gated.py --gate 1 --domain $DOMAIN --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 10 \
  --save_to "genrz_$TAG" $TIDARG || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
RES="$TB/data/simulations/genrz_$TAG/results.json"
if [ -f "$RES" ]; then
  gzip -c "$RES" > $REPO/reports/facet_rft_2026/sim_results/genrz_$TAG.results.json.gz
  cd $REPO && git add -f reports/facet_rft_2026/sim_results/genrz_$TAG.results.json.gz && \
    git commit -q -m "persist RESOLVE $TAG (auto)" && git pull --rebase -q origin facet-rft-2026 && git push -q origin facet-rft-2026 && echo PERSISTED
fi
echo "== audit =="
echo "RESOLVE deny: $(grep -aE 'T2_RESOLVE\] deny' $LOG|grep -avc '^+') | COV reminder: $(grep -aE 'T2_COV\] reminder' $LOG|grep -avc '^+') | READALL deny: $(grep -aE 'T2_READALL\] deny' $LOG|grep -avc '^+') | origin regen: $(grep -aE 'T2_PROV\] origin regen' $LOG|grep -avc '^+') | cons member: $(grep -aE 'T2_CONS\] membership deny' $LOG|grep -avc '^+') noop: $(grep -aE 'T2_CONS\] noop deny' $LOG|grep -avc '^+') | L4 would-sub(관측): $(grep -aE 'T2_L4\] keep-mode' $LOG|grep -avc '^+') sub(0이어야): $(grep -aE 'T2_L4\] substituted' $LOG|grep -avc '^+') | walk(0이어야): $(grep -aE 'walk gap' $LOG|grep -avc '^+')"
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
TARGET={"97","58","32","35","71"}        # v4 가드 표적 + 회귀 관찰
GUARD={"43","27","41","83"}              # Δspurious 가드(정당 주소·qty·정당 수정·selection)
allv=[v for vs in by.values() for v in vs]
print("★ 전체 db: %d/%d = %.3f (A1-v2=0.702 nt1)"%(sum(allv),len(allv),sum(allv)/max(1,len(allv))))
print("표적(가드 기대):", " ".join("t%s=%d/%d"%(k,sum(by[k]),len(by[k])) for k in sorted(TARGET,key=int) if k in by))
print("Δspurious 가드:", " ".join("t%s=%d/%d"%(k,sum(by[k]),len(by[k])) for k in sorted(GUARD,key=int) if k in by))
PYEOF
echo ALLDONE; date
