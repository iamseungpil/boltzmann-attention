#!/usr/bin/env bash
# x411 — T2_FN_ISOLATE(policy_qa wrap) **liveness 스모크** ([[67]] 0단계 · 사용자 승인 2026-08-19)
#
# 왜: wrap 은 설계·리뷰·구현이 끝난 채 **두 번** dark 였다(2026-08-02 이름 불일치 ×2).
#     t7326·t7328 실측 발화 = 0회. 유료 A/B 를 돌리기 전에 **발화부터** 확인한다.
#     x408 이 재제시 효과를 확정했으므로(A_slice 0.83 ↔ C_neg 0.50·상승 13/하락 0) 남은 위험은
#     "레버가 또 안 켜지는 것" 하나다.
#
# 판정선 (결과 보기 전에 못 박는다):
#   PASS  : "[T2_FN_ISOLATE] ... quotes N 반환" 이 **1회 이상**
#   FAIL-A: 마커 0회                      -> 세 번째 dark 사고. t7329 런 금지
#   FAIL-B: "[T2_FN_ISOLATE] ... 폴백" 만  -> 폴백 사유별로 세서 원인 제거 후 재검
#
# 표적 2태스크 (x409/x410 이 지목한 실물):
#   task_016 : 'spend at least $750' 가 11,073자 결과의 **깊이 93.5%** 에 있었다
#   task_033 : 'in this exact order' 절차가 28KB 덤프 안에 있었다
#
# 스택 = t7328 과 **완전 동일 + T2_FN_ISOLATE=1 한 줄만**. 포트 8141(GPU1·t7328 은 GPU0).
# ⚠t7328 이 GPU0 에서 돌고 있으므로 t7328 런처의 pgrep 거부 가드는 의도적으로 뺐다.

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

TAG=bank_x411_fniso_smoke_20260819
TASKS='task_016,task_033'
PORT=8141
NT=1

if [ -e "$LOG/${TAG}.log" ]; then echo "[x411] SKIP: ${TAG}.log 존재" >&2; exit 0; fi
if [ -e "$SIMS/${TAG}" ]; then echo "[x411] REFUSING: $SIMS/${TAG} 잔존" >&2; exit 1; fi

echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"port\":$PORT,\"tasks\":\"$TASKS\",\"nt\":$NT,\"delta\":\"T2_FN_ISOLATE=1 only\",\"why\":\"liveness smoke of policy_qa wrap before t7329 A/B; wrap fired 0 times in t7326/t7328\"}" \
  | tee "$LOG/${TAG}.meta.json"

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export $PIN && export GO_MAX_STEPS=150 GO_CONCURRENCY=1 && \
  export T2_FN_ISOLATE=1 && \
  env | grep -E '^T2_FN_ISOLATE|^T2_SCAFFOLD_GET' | sort && \
  t2_launch $TAG $PORT '$TASKS' $NT" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &

echo "[x411] 기동 PID=$! port=$PORT tasks=$TASKS nt=$NT sha=$SHA"
echo "[x411] 판정: grep -c 'T2_FN_ISOLATE' $LOG/${TAG}.log  (0 이면 런 금지)"
