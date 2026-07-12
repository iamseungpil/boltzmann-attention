#!/bin/bash
# A2-v2 = A2(일반화 스택 + DISAMB-enumerate) + 3 교정 (2026-07-12 HANDOFF §6):
#   (1) filter-substitute(§6.1·dotted-path 수정) = ≥2 후보를 LLM-formalize→엔진 결정론 필터,
#       1?치환 : ≥2?열거-ASK : 0?재형식화. 32B directive 미준수 우회(advise→결정론).
#   (2) eplan EXAMINED-SAFE(§6.2) = 대상 record가 이미 검토됐으면 discovery-deny 생략
#       (A1 부작용: 정당 write까지 deny→read-루프→transfer, t21/32/42 순손실 3 교정).
#   (3) prov ADDR-FULL(§6.3) = 주소류 free-text 인자는 rescue 중립문 대신 full 문구(getter 명시)로
#       조회 유도(A1 부작용: 주소 날조 미검, t43/96 교정).
# ★측정: (a) filter one/many 발화(dotted 수정 효과) (b) trivial 무회귀 (c) hard 개선.
# 용법: bash generalized_enum_v2.sh <TAG> <TASKS|ALL> <NT> <PORT>
set -u
REPO=/home/woori/workspace_common/boltzmann-attention-pi
T2=$REPO/scripts/distill/tau2; PY=/home/woori/venvs/seka_env/bin/python
S=/home/woori/scratch; TB=$S/tau2-bench
M="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TAG="${1:?tag}"; TASKS="${2:?tasks|ALL}"; NT="${3:-1}"; PORT="${4:-8140}"
LOG=$S/genenumv2_${TAG}.log; exec > $LOG 2>&1; set -x; date
cd $REPO && git pull --rebase -q origin facet-rft-2026 2>/dev/null
source /home/woori/.openrouter_key
export SSL_CERT_FILE=$($PY -c "import certifi;print(certifi.where())")
export PYTHONPATH=src:$T2
curl -s --max-time 5 localhost:$PORT/v1/models | grep -q "$M" || { echo SERVE_MISSING_$PORT; exit 1; }
export T2_GATE_REGEN=1 T2_GATE_REGEN_K=1 T2_GATE_KINDS=auth,confirm,ownership,notice,preconditions,constraints
export T2_PROV_REGEN=1 T2_PROV_REGEN_K=4 T2_PROV_MODE=rescue
export T2_PROV_ADDR_FULL=1                       # ★(3) 주소류=full prov(getter 명시·§6.3)
export T2_CALC=1
export T2_EPLAN=1
export T2_EPLAN_EXAMINED_SAFE=1                  # ★(2) 검토된 대상 write는 discovery-deny 생략(§6.2)
export T2_WRITE_CAP=1 T2_WRITE_CAP_K=2
export T2_DISAMB=1 T2_DISAMB_MODE=enumerate     # ★(1) ≥2 후보→filter-substitute(dotted-path 수정·§6.1)
export T2_DISAMB_ORDER=1                         # ★order operand도 disamb 대상(71류)
unset T2_PRESENT_NESTED T2_PRESENT_READS T2_GROUND T2_PRINCIPLE_DEFAULT T2_AUTOFETCH T2_EPLAN_WALK T2_PROV_BADWORDS
cd $TB
TIDARG=""; [ "$TASKS" != "ALL" ] && TIDARG="--task_ids $TASKS"
rm -rf "$TB/data/simulations/genenumv2_$TAG"
timeout 10800 $PY $T2/t2_run_gated.py --gate 1 --domain retail --agent_model "$M" --agent_base http://localhost:$PORT/v1 \
  --user_llm openrouter/openai/gpt-4.1 --user_temp 0.0 --num_trials $NT --max_concurrency 8 \
  --save_to "genenumv2_$TAG" $TIDARG || echo "ARM_FAIL $TAG"
date; echo RUN_DONE
echo "== audit(비트레이스) =="
echo "DISAMB fired(≥2): $(grep -aE '\[T2_DISAMB\] fired' $LOG | grep -avc '^+')"
echo "★(1) FSUB: substituted=$(grep -aE 'T2_FSUB\] substituted' $LOG | grep -avc '^+') confirmed=$(grep -aE 'T2_FSUB\] confirmed' $LOG | grep -avc '^+') narrowed=$(grep -aE 'T2_FSUB\] narrowed' $LOG | grep -avc '^+') reverted=$(grep -aE 'FSUB switch reverted' $LOG | grep -avc '^+')"
echo "★(1) FEXEC filter: one=$(grep -aF '[T2_FEXEC] filter one' $LOG | grep -avc '^+') many=$(grep -aF '[T2_FEXEC] filter many' $LOG | grep -avc '^+') fallback=$(grep -aF '[T2_FEXEC] filter fallback' $LOG | grep -avc '^+') empty=$(grep -aF '[T2_FEXEC] filter empty' $LOG | grep -avc '^+')"
echo "★(2) EPLAN examined-safe(정당write 구제): $(grep -aE 'T2_EPLAN\] examined-safe' $LOG | grep -avc '^+') | L2 deny 잔여: $(grep -aE 'T2_EPLAN\] L2 deny' $LOG | grep -avc '^+')"
echo "★(3) PROV addr-full 발화: $(grep -aE 'T2_PROV\] regen fired' $LOG | grep -avc '^+') (addr 인자 포함)"
$PY - "$TB/data/simulations/genenumv2_$TAG/results.json" <<'PYEOF'
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
