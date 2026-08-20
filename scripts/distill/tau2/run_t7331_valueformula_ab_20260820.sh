#!/usr/bin/env bash
# t7331 — **후보별 값 주석의 첫 라이브 A/B** (사용자 승인 2026-08-20)
#
# ★사전등록 (런 시작 전에 못 박는다·결과 보고 바꾸지 않는다)
#   질문 : 엔진이 후보별 **값**(요율 × 손님이 말한 금액 − 연회비)을 붙이면 **reward 를 사는가**
#   종점 : **reward 뿐**([[69]]). gold 일치율·action_match 는 성적이 아니다.
#   설계 : 단일 변수 A/B. 변수 = `T2_VALUE_FORMULA` 하나.
#          ctl  = unset (주석 0행 = 종전 스택과 **완전히 동일**·검정으로 고정)
#          base = ⒟안(기본 요율만·범주 분기 없음)
#   표적 : task_003 · task_024 · task_063 (fit 도구가 실제로 도는 카드축) × nt=3 = 팔당 9 sim
#   판정 : reward 합의 차. **판정선 ±4/40 (C483 잡음 바닥)** — 18 sim 규모에서는 차 2 미만이면 null.
#   동급 계측(사전 고정·C563 §4): ⒜ 조회 수(fit 이후) ⒝ 날조(엔진 `_provenance_deny`) ⒞ over-action(쓰기)
#                                  · Δspurious ≤ 0 (등대 §1.3) · 태스크별 부호표([[70]])
#
# ★왜 이 레버인가 — [[62]] 사다리를 전부 통과한 유일한 자리
#   격리 0.38 · 전달만 ✗ · 목적식 말해줘도 +2/48 · **값을 주면 0.98**(C562) ·
#   ⒟는 그 이득의 60% 를 지키며 오매핑 증폭이 **구조적으로 0**(C567) ·
#   부작용 선계량에서 present 형 손상 미재현(C563) · 금액 날조는 근거 검사로 차단(C565 수리)
#
# ⚠유료(user-sim = openrouter gpt-5.2). 스모크 1 + 본런 18 = **19 sim**. GPU1(8141) 순차.
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

TASKS='task_003,task_024,task_063'
NT=3
PORT=8141
MARK='documented_return_for_stated_spend'

for f in "$LOG/bank_t7331_"*.log; do [ -e "$f" ] && { echo "[t7331] REFUSING: $f 존재" >&2; exit 1; }; done

echo "{\"tag\":\"t7331\",\"sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":$NT,\"arms\":[\"ctl(unset)\",\"base(T2_VALUE_FORMULA=base)\"],\"endpoint\":\"reward only\",\"bar\":\"|d| < 2 = null (±4/40 noise floor)\",\"also\":[\"reads\",\"fabs\",\"over-action\"]}" \
  | tee "$LOG/bank_t7331.meta.json"

setsid bash -c "
  cd '$REPO/scripts/distill/tau2'
  source ./go_stack.sh >/dev/null 2>&1
  export $PIN
  export GO_MAX_STEPS=150 GO_CONCURRENCY=1

  # ── 0단계 스모크: 배선이 라이브에서 **발화**하는지부터([[30]]·[[67]] t2_liveness 0단계)
  export T2_VALUE_FORMULA=base
  SMK=bank_t7331_smoke_20260820
  echo '[t7331] === 스모크(1 sim·task_024·treat) ==='
  t2_launch \$SMK $PORT task_024 1
  N=\$(grep -c '$MARK' $LOG/\$SMK.log || echo 0)
  echo \"[t7331] 스모크 주석 발화 = \$N\"
  cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
  gzip -c '$SIMS/'\$SMK'/results.json' > reports/facet_rft_2026/sim_results/\$SMK.results.json.gz
  gzip -c $LOG/\$SMK.log > reports/facet_rft_2026/sim_results/\$SMK.log.gz
  git add -f reports/facet_rft_2026/sim_results/\$SMK.results.json.gz reports/facet_rft_2026/sim_results/\$SMK.log.gz
  cd '$REPO/scripts/distill/tau2'
  if [ \"\$N\" -eq 0 ]; then
    echo '[t7331] ⛔스모크 발화 0 — 본런을 돌리지 않는다(死배선에 돈을 쓰지 않는다)'
    exit 1
  fi

  for ARM in ctl base; do
    TAG=bank_t7331_\${ARM}_20260820
    if [ \"\$ARM\" = ctl ]; then unset T2_VALUE_FORMULA; else export T2_VALUE_FORMULA=base; fi
    env | grep -E '^T2_VALUE_FORMULA' | sort > $LOG/env_\${TAG}.txt || true
    echo \"[t7331] === \$ARM 시작 · T2_VALUE_FORMULA='\${T2_VALUE_FORMULA:-(unset)}' ===\"
    t2_launch \$TAG $PORT '$TASKS' $NT
    echo \"[t7331] \$ARM 완료 · 주석 발화=\$(grep -c '$MARK' $LOG/\$TAG.log || echo 0)\"
    cd '$REPO' && mkdir -p reports/facet_rft_2026/sim_results
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    git add -f reports/facet_rft_2026/sim_results/\$TAG.results.json.gz reports/facet_rft_2026/sim_results/\$TAG.log.gz
    cd '$REPO/scripts/distill/tau2'
  done
  echo '[t7331] 두 팔 완료'
" </dev/null > "$LOG/bank_t7331_chain.log" 2>&1 &

echo "[t7331] 기동 PID=$! · sha=$SHA · 스모크 1 + 본런 18 sim · GPU1/$PORT"
