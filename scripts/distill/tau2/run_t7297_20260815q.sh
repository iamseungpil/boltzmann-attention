#!/bin/bash
# t7297 — **행동 촉구(`T2_ACT_DEMAND`) 라이브 A/B** (2026-08-15·사용자 지시 "1번부터 하고 2번 가라").
#
# 무엇을 재나. 오늘 격리가 knowing–doing 을 확정했다(C489): 같은 컷·같은 도구에서
#   이름 대기 **18/24** ↔ 실제 방출 **2/24**(부정통제 `D_EARLY` 0/24).
# 그리고 **열거 없는 행동 명령 한 줄**이 3런 연속 그 격차를 절반 메웠다:
#   x330 `C_EMIT_ASK` **11/24** · x331 `D_ASK` **13/24** · x332 `C_ASK` **16/24**
#   ↔ 같은 조건 기준선 **2 / 0 / 6**.
# 반례도 같이 쟀다: *"몇 개인지 세고 체크하고 처리하라"* = x332 `B_SELFLIST` **0/24**
#   (기준선보다 낮다) ⇒ **묘사를 시키면 묘사가 는다.** 그래서 문구는 열거를 담지 않는다.
#
# ⚠**이 레버는 라이브에서 한 번도 안 걸렸다.** t7296 은 *전달*만 봤다(그리고 null 이었다).
#   "프롬프트는 prior 를 못 이긴다"([[42]])도 라이브에서 검정된 적이 없다 — 그것을 여기서 잰다.
#   결과는 다음 단계(활성 스티어링)의 **대조군**이 된다.
#
# 편성: 4 태스크 × nt=5 = **20 sim/arm** · 두 팔 동시(ctl=8140 · treat=8141).
#   task_050·072·073 = **끝맺음이 떨어지는 자리**(x328·§6⒡: 마지막 gold 미호출 82%).
#     073·050 은 *노출됐는데 안 씀*(순수 채택 결손) · 072 는 unlock 까지 하고 안 부른다.
#   task_098 = **거동 불변 의무**(t7295 3/3 통과·이 레버가 건드리면 안 되는 자리·[[57]]).
#   두 팔의 차이는 **환경변수 하나**(`T2_ACT_DEMAND`). 전달 수리 2종은 **양 팔 모두 ON**
#   (t7296 에서 성적 null 이었으므로 교란이 아니라 공통 배경으로 고정한다).
#
# 판정(사전 고정·이 순서로):
#   ⓐ배선  treat 에서 `[T2_ACT_DEMAND] 행동 촉구` 발화 > 0 · ctl 0. 실패면 성적을 읽지 않는다.
#   ⓑ레버  **write 시도율** — gold write 도구를 실제로 호출한 sim 비율(호출 자체·성패 무관).
#           격리가 예측하는 1차 종점이 이것이다(pass 가 아니라 **실행 여부**).
#   ⓒ성적  `reward`/`db_match` 로만(C486).
#   ⓓ부작용 **over-action**: gold 에 없는 write(`dbdiff_task` 의 `ONLY-PRED`)가 늘면 손해다([[57]]).
#           098 은 **3/3 유지**여야 한다.
#   ⚠n=5/태스크/팔 · 잡음 바닥 ±4(C483) ⇒ **성적 차이는 ≥5 만 인용**. 1차 종점은 ⓑ.

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=5
TASKS=task_050,task_072,task_073,task_098
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_compute.py \
  scripts/distill/tau2/t2_scaffold_get.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7297] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_act_demand.py test_now_selfcall.py test_provenance_nested.py \
         test_no_undefined_names.py test_decision_carry.py test_subcall_return_type.py \
         test_a2_three_layer.py test_operator_find.py test_route_trace.py; do
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7297] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7297] VERIFY OK"

launch () {
  NAME="$1"; PORT="$2"; DEMAND="$3"
  TAG="bank_t7297_${NAME}_20260815q"
  if [ -e "$LOG/${TAG}.log" ]; then echo "[t7297] SKIP: ${TAG}.log 존재" >&2; return 0; fi
  if [ -e "$SIMS/${TAG}" ]; then echo "[t7297] REFUSING: $SIMS/${TAG} 잔존" >&2; return 1; fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7297] REFUSING: 포트 ${PORT} 사용 중" >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"act_demand\":\"$DEMAND\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 \
           T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1 && \
    export T2_ACT_DEMAND=$DEMAND && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7297] $NAME(demand=$DEMAND) → PID=$! port=$PORT"
}

launch ctl   8140 0
launch treat 8141 1
echo "[t7297] 기동 완료 · sha=$SHA · nt=$NT · tasks=$TASKS"
