#!/usr/bin/env bash
# t7330 — **formalize→calc 아키텍처의 첫 A/B** (사용자 승인 2026-08-19)
#
# ★사전등록 (런 시작 전에 못 박는다·결과 보고 바꾸지 않는다)
#   질문 : "LLM 이 formalize 하고 엔진이 그 값만 계산한다"가 **reward 를 사는가**
#   종점 : **reward 뿐**. gold 일치율·action_match 는 보지 않는다([[69]] §0).
#   설계 : 단일 변수 A/B. 변수 = `T2_SG_EXCLUDE=get_atm_fee_discrepancies` 하나.
#          A_off = 그 도구 제외(레버 없음) · B_on = 그대로(레버 있음)
#          나머지 스택·모델·시드·태스크·nt 전부 동일.
#   표적 : task_072 · task_073 · task_074 (ATM 수수료 환급 계열) × nt=3 = 팔당 9 sim
#   판정 : B_on 의 reward 합 − A_off 의 reward 합. 잡음 바닥이 ±1 이므로 |Δ|≤1 은 null 로 읽는다.
#
# ★왜 이 도구가 첫 표적인가 — 네 조건을 다 만족하는 유일한 현존 도구
#   ① 상수 출처가 정책 축자(`ATM_FEE_SCHEDULE_VERBATIM_2026_08_13.md`·gold 미접촉)
#   ② 엔진은 formalize 된 값만 계산(페어링·network 분류는 모델·엔진은 min/max/tier 산술)
#   ③ 엔진이 채점되는 인자를 안 만듦 — 2026-08-19 `${delta_total:.2f}` 제거로 확보
#   ④ 대조 팔 + reward 종점 — 이 런
#   3개월 174건 회고에서 ①~④를 모두 갖춘 측정은 **0건**이었다.
#
# ⚠유료(user-sim = openrouter gpt-5.2). 총 18 sim. GPU1(8141) 순차 — GPU0 은 t7328 halfB 사용 중.

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
SHA=$(cd "$REPO" && git rev-parse --short HEAD)

PIN="T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 \
T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 \
T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 \
T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 \
T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 \
T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0"

TASKS='task_072,task_073,task_074'
NT=3
PORT=8141

for f in "$LOG/bank_t7330_"*.log; do [ -e "$f" ] && { echo "[t7330] REFUSING: $f 존재" >&2; exit 1; }; done

echo "{\"tag\":\"t7330\",\"sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":$NT,\"arms\":[\"A_off(T2_SG_EXCLUDE=get_atm_fee_discrepancies)\",\"B_on\"],\"endpoint\":\"reward only\",\"why\":\"first A/B of the formalize->calc architecture with all four conditions satisfied\"}" \
  | tee "$LOG/bank_t7330.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  for ARM in A_off B_on; do
    TAG=bank_t7330_\${ARM}_20260819
    source ./go_stack.sh >/dev/null 2>&1
    export $PIN
    export GO_MAX_STEPS=150 GO_CONCURRENCY=1
    if [ \"\$ARM\" = A_off ]; then export T2_SG_EXCLUDE=get_atm_fee_discrepancies; else unset T2_SG_EXCLUDE; fi
    env | grep -E '^T2_SG_EXCLUDE|^T2_SCAFFOLD_GET' | sort > $LOG/env_\${TAG}.txt || true
    echo \"[t7330] === \$ARM 시작 · SG_EXCLUDE='\${T2_SG_EXCLUDE:-(unset)}' ===\"
    t2_launch \$TAG $PORT '$TASKS' $NT
    echo \"[t7330] \$ARM 완료 · 도구 발화=\$(grep -c 'get_atm_fee_discrepancies' $LOG/\$TAG.log || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    git add -f reports/facet_rft_2026/sim_results/\$TAG.results.json.gz reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  done
  echo '[t7330] 두 팔 완료'
" </dev/null > "$LOG/bank_t7330_chain.log" 2>&1 &

echo "[t7330] 기동 PID=$! · sha=$SHA · 18 sim · GPU1/$PORT"
echo "[t7330] 판독: grep -o 'Avg reward: [0-9.]*' $LOG/bank_t7330_{A_off,B_on}_20260819.log | tail -1"
