#!/bin/bash
# **071 라이브 수리 검증** (사용자 지시 2026-08-13). STEP2 수리 3종(discoverable-한정·
# 레지스트리 폴백·REG_FB)의 원 표적 = 071 — x283/x283b 격리에서 t1 8/8·t3 6/8·t0 8/8.
# nt=4 · ON 팔만 · 동결 · **8141(GPU1)** — 사용자 지시 2026-08-13 "071은 GPU1에서·그러라고
# 한 개 비워둔 것" (t7273=8140 과 병행).
#
# 대조: 기준선 4열(11/24 계열)의 071 = 0/4·0/4·0/4·0/4 · q런 = 2/4 (폴백 이전 sha 67d0fd25).
# 읽을 것: ①071 pass ②[T2_DISCOVERY_STEP2] "레지스트리 폴백" 발화×결과 짝 ③over-block 0
#   ④t2/t3형(gold 완수 후 여분 write)은 이 레버 범위 밖 — 재발해도 회귀 아님(백로그 L6).
#
# usage: run_071verify_20260813t.sh
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=4
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[071v] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
fi

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
bad = []
for t in ("test_regen_break_guard.py", "test_no_undefined_names.py", "test_discovery_step2.py",
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

TAG="bank_071v_20260813t"
PORT=8141
TASKS="task_071"
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[071v] SKIP: 이미 있다." >&2; exit 0
fi
if [ -e "$SIMS/${TAG}" ]; then
  echo "[071v] REFUSING: 결과 디렉토리 잔존." >&2; exit 1
fi
if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
  echo "[071v] REFUSING: 포트 ${PORT} 사용 중." >&2; exit 1
fi
echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"071v_20260813\"}" \
  | tee "$LOG/${TAG}.meta.json"
setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
         T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
         T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
         T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
  t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "[071v] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log · sha=$SHA"
