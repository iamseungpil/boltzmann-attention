#!/bin/bash
# t7300 — **`T2_MATERIAL_RESERVE` 나머지 편성**(2026-08-16·사용자 지시 *"3이 055·098 을 했으니 1은 나머지만"*).
#
# t7299 가 `055`(격리 24/24 ↔ 라이브 0/4 인 자리)와 `098`(불변 대조)을 **같은 sha·같은 플래그**로
# 이미 쟀다. 그래서 여기서는 **070·071 만** 돈다(16 → 8 sim/팔). 두 런의 055/098 과 070/071 은
# **동일 스택일 때만** 합칠 수 있다 ⇒ ⚠t7299 이후 엔진·A2 를 고치면 합산 근거가 사라진다.
# (아래 VERIFY 는 그 규율을 강제하지 못하므로 **sha 를 meta 에 남겨 사후 대조**한다.)
#
# 판정은 t7299 와 같은 사전 고정 순서: ⓐ배선 → **ⓑ1차 = 결정 자리 배달 수** → ⓒ선택(gold 클래스를
# 실제로 연 sim) → ⓓ성적(`reward`/`db_match`) → ⓔ부작용(over-action·지연).
# ⚠nt=4 · 잡음 바닥 ±4 ⇒ ⓒⓓ 인용 금지. 이 런도 **ⓑ 를 사러 간다**.
# ⚠t7299 의 ⓑ 가 안 열렸으면 이 런은 **돌리지 않는다**(그때는 기전 재규명이 먼저다).

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=4
TASKS=task_070,task_071
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_compute.py \
  scripts/distill/tau2/t2_scaffold_get.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7300] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_material_reserve.py test_material_bypass.py test_probe_canonical.py \
         test_now_selfcall.py test_provenance_nested.py test_no_undefined_names.py \
         test_decision_carry.py test_subcall_return_type.py test_a2_three_layer.py \
         test_operator_find.py test_route_trace.py; do
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7300] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7300] VERIFY OK"

if pgrep -f "[t]2_gap.py" >/dev/null || pgrep -f "[x]33[0-9]_" >/dev/null; then
  echo "[t7300] REFUSING: 무료 프로브가 8141 을 쓰는 중(팔을 동시에 못 띄운다)" >&2; exit 1
fi

launch () {
  NAME="$1"; PORT="$2"; RESERVE="$3"
  TAG="bank_t7300_${NAME}_20260816c"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7300] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7300] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7300] REFUSING: 포트 ${PORT} 사용 중" >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"reserve\":\"$RESERVE\",\"why\":\"C498 primary endpoint = material present at decision\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 \
           T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1 && \
    export T2_ACT_DEMAND=0 T2_MATERIAL_RESERVE=$RESERVE && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7300] $NAME(reserve=$RESERVE) → PID=$! port=$PORT"
}

launch ctl   8140 0
launch treat 8141 1
echo "[t7300] 기동 완료 · sha=$SHA · nt=$NT · tasks=$TASKS"
