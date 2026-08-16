#!/bin/bash
# t7298 — **C494 사전등록 예측의 검정**(2026-08-16·사용자 승인).
#
# 예측(원장 C494·격리 x335/x335b): 055 의 checking 선택 결손은 **능력이 아니라 우리 전달**이었다.
#   격리: 재료 없음 0/24 → 재료 전달 **24/24**(gold `Purple`) · 손님 축만 바꾸면 **24↔24 로 뒤집힘**
#   (선언 수치를 실제로 쓴다) · 재료 없으면 **카탈로그 밖 상품명 날조** 24/24.
#   라이브 대조(t7295·수리 前): 055 에서 `now 미확정 → 침묵` **12회** ↔ 실제 전달 **4회**.
#   수리 後(t7297 두 팔): `now 미확정` **0회** · 전달 **76/84회**.
# ⇒ 수리된 스택으로 다시 돌리면 **checking 선택 축은 통과해야 한다.**
#   통과 못 하면 남는 것은 **끝맺음**(C489·마지막 gold 미호출 82%)이다.
#
# 편성: `055·070·071` = 제거 축 표본(§22 감사로 **문서가 gold 를 지지**함을 확인한 셋.
#   **069 는 제외** — 거기서는 근거가 gold 뿐이라 표적으로 삼으면 [[23]] 위반).
#   `098` = **거동 불변 대조**(t7295 3/3 · t7297 5/5 · 여기서도 흔들리면 안 된다·[[57]]).
#   nt=4 × 4 태스크 = **16 sim · 단일 팔**(A/B 아님 — 처치는 이미 t7296/t7297 에 들어간 전달 수리이고
#   여기서 재는 것은 *예측의 적중* 이다).
#
# 플래그: t7297 **ctl 과 동일**(`T2_ACT_DEMAND=0`) — 촉구는 C492 에서 성적 null·over-action +6 이므로
#   켜지 않는다. ⇒ **t7297 ctl 이 이 편성의 스모크**다(같은 스택·같은 날·크래시 0).
#
# 판정(사전 고정·이 순서로):
#   ⓐ배선  055 궤적에서 `now 미확정` **0회** · `T2_SEARCH_AGENT` 실제 전달 > 0. 실패면 성적 안 읽는다.
#   ⓑ1차 종점 **checking 선택** — 055 에서 `Purple Account` 로 `open_bank_account_4821` 를 부른 sim 수
#     (pass 가 아니라 **선택의 적중**. 격리가 예측한 것은 이것이다).
#   ⓒ성적  `reward`/`db_match` 로만(C486·`action_match` 금지).
#   ⓓ부작용 over-action(`ONLY-PRED`) 증가 없음 · **098 불변**.
#   ⚠nt=4 · 잡음 바닥 ±4(C483) ⇒ 성적 차이 인용 금지. ⓑ가 1차 종점이고 여기선 **0 → 다수**가 예측이다.

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=4
TASKS=task_055,task_070,task_071,task_098
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_compute.py \
  scripts/distill/tau2/t2_scaffold_get.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7298] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

for t in test_now_selfcall.py test_provenance_nested.py test_no_undefined_names.py \
         test_decision_carry.py test_subcall_return_type.py test_a2_three_layer.py \
         test_operator_find.py test_route_trace.py; do
  PYTHONPATH=/home/woori/scratch/tau2-bench/src /home/woori/venvs/seka_env/bin/python "$t" \
    >/dev/null 2>&1 || { echo "[t7298] REFUSING: $t FAIL" >&2; exit 1; }
done
echo "[t7298] VERIFY OK"

TAG=bank_t7298_20260816a
PORT=8140
if [ -e "$LOG/${TAG}.log" ]; then echo "[t7298] REFUSING: ${TAG}.log 존재" >&2; exit 1; fi
if [ -e "$SIMS/${TAG}" ]; then echo "[t7298] REFUSING: $SIMS/${TAG} 잔존" >&2; exit 1; fi
if ps -eo cmd | grep -v grep | grep -q "t2_run_gated.py"; then
  echo "[t7298] REFUSING: 유료 런이 이미 돈다" >&2; exit 1
fi

echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"act_demand\":\"0\",\"why\":\"C494 prediction test\"}" \
  | tee "$LOG/${TAG}.meta.json"

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
         T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
         T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
         T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 \
         T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1 && \
  export T2_ACT_DEMAND=0 && \
  t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "[t7298] 기동 PID=$! · sha=$SHA · nt=$NT · tasks=$TASKS · port=$PORT"
