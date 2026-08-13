#!/bin/bash
# **fee-가족 재판정** (072/073/074·nt=1·양GPU) — get_atm_fee_discrepancies 첫 라이브.
# 델타 = (B) ATM-fee 결정론기(x288 A_DOCS 0/8 판정·정책 축자 요율·unit 9/9·3사본 동기).
# 읽을 것: ①도구 실호출 여부(발화×결과 짝) ②credit 금액이 도구 출력 기반으로 바뀌는가
#   ③073 gold 칸(073_8~10)·072(072_6~8)·074(074_9~12) 도달 ④보류 클래스(skipped) 표면화.
# usage: run_t7274w_20260813w.sh
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
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7274w] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
bad = []
for t in ("test_regen_break_guard.py", "test_no_undefined_names.py", "test_discovery_step2.py",
          "test_atm_fee_op.py",
          "test_write_arg_enum.py", "test_decide_before_write.py", "test_route_trace.py",
          "test_a2_three_layer.py", "test_decision_carry.py", "test_decision_isolate.py",
          "test_axis_levers.py", "test_operator_find.py", "test_action_operator.py"):
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
  TAG="bank_t7274w_${NAME}_20260813w"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7274w] SKIP: 로그 이미 있음." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7274w] REFUSING: 결과 디렉토리 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7274w] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7274w_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7274w] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074 8140
launch b task_073 8141
echo "[t7274w] 기동 완료 · sha=$SHA · nt=$NT · a={072,074}→8140 · b={073}→8141"
