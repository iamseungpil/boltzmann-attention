#!/bin/bash
# t7296 — **`T2_NOW_SELFCALL` pass 영향 실험**(2026-08-15·사용자 지시).
#
# 무엇을 재나. t7295 포렌식이 찾은 것: 검색 에이전트의 창은 A2 `action_tools` 푸시 **결정점**
# 에서만 열리는데, 071 은 세 sim 통틀어 그 줄이 **1개**였고 그 한 번이 시계보다 **앞**이었다
# (로그 2282 침묵 ↔ 시계 2424 · 나머지 2 sim 은 창이 아예 안 열림). 그래서 만료 제거 기계가
# 재료를 **한 번도** 못 냈다. `now 미확정` 은 arm b 침묵 80회 중 **1위 사유**다.
#
# 대상 2 태스크(x325 영향반경에서 ★열림-완전 = BLOCKED>0 ∧ DELIVERED=0):
#   task_071  BLOCKED  1 · DELIVERED 0 · pass 0/3   ← 만료 프로모션 배제가 태스크의 전부
#   task_072  BLOCKED 22 · DELIVERED 0 · pass 0/4   ← BLOCKED 최다 · B 군집(크레딧 미착지)과 겸함
#
# 격리 근거([[62]] ①②): `x248`·`x250`(n=8·프로덕션 경로) 두 축 **8/8**
#   (checking `Sky Blue` · savings `Gold Saver Account`). 부정통제 = 고지 없이 checking **0/8** ·
#   만료를 안 빼면 savings **0/8**. ⇒ 격리에선 모델이 이긴다 ⇒ 살 것은 **전달뿐**이다.
#
# 편성: 2 태스크 × nt=8 = **16 sim/arm** · 두 팔 동시(ctl=8140 · treat=8141).
#   두 팔의 차이는 **환경변수 하나**뿐이다(`T2_NOW_SELFCALL`). 그 외 전부 t7295 와 동일.
#
# 판정(사전 고정·이 순서로):
#   ⓐ배선  treat 에서 `[T2_NOW_SELFCALL]` 발화 > 0 ∧ `T2_SEARCH_AGENT ... group=` 전달 > 0
#           (071 은 t7295 에서 전달 **0**이었다). 여기서 실패면 성적을 읽지 않는다.
#   ⓑ레버  `dbdiff_task.py` 로 071 의 `accounts.data` 등급이 gold 로 가는가
#           (`Sky Blue` · `Gold Saver Account`).
#   ⓒ성적  **reward / db_match 로만**(C486: `action_match` 는 표기로 무너진다).
#   ⓓ부작용 지목 0 유지 · 중복 write 0 · ctl 이 t7295 와 같은 자리에 있는가(재현 대조).
#   ⚠n=8/태스크/팔 · 잡음 바닥 ±4/8(C483) ⇒ **성적 차이는 ≥5 만 인용**. 1차 종점은 성적이
#     아니라 **전달**(기계적·거의 결정론적)이다. 성적은 방향 지시로만 읽는다.

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=8
TASKS=task_071,task_072
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_compute.py \
  scripts/distill/tau2/t2_scaffold_get.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7296] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

# 이 수정의 유일한 정당화 = 부작용 0. 검정이 깨지면 런을 걸지 않는다.
PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python \
  test_now_selfcall.py >/dev/null 2>&1 || { echo "[t7296] REFUSING: test_now_selfcall FAIL" >&2; exit 1; }
for t in test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_subcall_canonical.py; do
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7296] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7296] VERIFY OK"

launch () {
  NAME="$1"; PORT="$2"; SELFCALL="$3"
  TAG="bank_t7296_${NAME}_20260815p"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7296] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7296] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7296] REFUSING: 포트 ${PORT} 사용 중" >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"now_selfcall\":\"$SELFCALL\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 && \
    export T2_NOW_SELFCALL=$SELFCALL && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7296] $NAME(selfcall=$SELFCALL) → PID=$! port=$PORT"
}

launch ctl   8140 0
launch treat 8141 1
echo "[t7296] 기동 완료 · sha=$SHA · nt=$NT · tasks=$TASKS"
