#!/bin/bash
# **072~075 재판정 2차** (t7276 전수 포렌식 후속·사용자 지시 2026-08-13 "다음 실험 4 태스크
# 모두 실험할 수 있게 준비").
# nt=1 · a={072,074}→8140 · b={073,075}→8141 · 동결.
#
# 탑재: t7276 구성(스택+STEP2 3종+FIX-1/2/3+FIX-5+(B2)) 그대로에 더해
#   + FIX-6(_slug_disp 하이픈 대문자화 — t7276 075 "Fee-free" 조용 통과의 우리층 수리·
#     test_slug_disp 6/6·플래그 없음=상시)
#   — FIX-4 최종 기각(x292b first-read 창 A_CUR 0/8 = 오표적 재현 실패·확률 사건 확정)
#   — T2_WRITE_DEDUP 보류(x294 A_ASIS 0/8 = 중복 재실행 재현 실패·저확률 사건) —
#     두 건 모두 라이브 재발률 관측만(부수 관측: x292b 8/8 log_verification 재호출 churn).
# 관측 포인트(t7276 포렌식 승계): ⓐ075 write 인자 표기(OFFICIAL-NAME deny 발화×결과)
#   ⓑ072/073 fee 도구 coverage 라인 뒤 재질의 여부(입력 커버리지 잔여) ⓒ072 중복 credit
#   재발 ⓓ074 첫 read 표적·계좌 열거 도달 ⓔ(B2) formalize 인자(usage 확정 전 추측 호출).
#
# usage: run_t7277_20260813y.sh
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=1
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_compute.py \
  scripts/distill/tau2/t2_scaffold_get.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7277] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
bad = []
for t in ("test_regen_break_guard.py", "test_no_undefined_names.py", "test_discovery_step2.py",
          "test_write_arg_enum.py", "test_decide_before_write.py", "test_route_trace.py",
          "test_a2_three_layer.py", "test_decision_carry.py", "test_decision_isolate.py",
          "test_axis_levers.py", "test_operator_find.py", "test_action_operator.py",
          "test_atm_fee_op.py", "test_checking_fee_totals.py", "test_c197_inputholes.py",
          "test_slug_disp.py"):
    if not os.path.exists(os.path.join(d, t)):
        continue
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s: %s" % (t, (r.stdout or "")[-140:]))
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

launch () {
  NAME="$1"; TASKS="$2"; PORT="$3"
  TAG="bank_t7277_${NAME}_20260813y"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7277] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7277] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7277] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7277_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7277] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074 8140
launch b task_073,task_075 8141
echo "[t7277] 기동 완료 · sha=$SHA · nt=$NT · a={072,074}→8140 · b={073,075}→8141"
