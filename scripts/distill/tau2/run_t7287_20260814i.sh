#!/bin/bash
# **U1/U2 동작 확인 런 (nt=1·판정 아님·사용자 지시 2026-08-14)**.
#
# 탑재 = t7284 구성 + 이번 세션 수리 2종:
#   U1 `_dispatch_since_last_user` — 손님 발화 이후 발견형 디스패치가 **성공**했으면 보고 턴을
#      회피로 보지 않는다(073 t0 이 완료된 credit 3건을 3회 재실행한 자리·write 중복 36건 중
#      15건이 우리 문구 직후·검정 3/3).
#   U2 정체 카운터 **새 이름 리셋** — 회수 후보가 커지면 인자 변화로 보고 되돌린다
#      (x305: 노출 전 formalize none 8/8 · 노출 후 TARGET 8/8 · x304 그 문면 6/8·검정 4/4).
#
# 관측 포인트(성적 아님·nt=1 은 판정 도구가 아니다·C467):
#   ⓐ 073 credit 중복 실행이 사라지는가(같은 인자 성공 실행 2회+ = 0 이어야)
#   ⓑ `[DISCOVERY-REQUIRED]`/STEP2 가 **완료 후**에 나가는 사례가 사라지는가
#   ⓒ 087 형 태스크서 정체 캡이 새 이름 도착 시 리셋되어 옳은 이름이 푸시되는가
#   ⓓ Δspurious: 침묵이 필요한 자리(반복 요구)서 푸시가 늘지 않는가
#
# usage: run_t7287_20260814i.sh
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
  echo "[t7287] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
          "test_slug_disp.py", "test_ownership_fix.py",
          "test_claim_owner_recovery.py",
          "test_dispatch_history_guard.py", "test_resolve_cap_name_reset.py"):
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
  TAG="bank_t7287_${NAME}_20260814i"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7287] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7287] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7287] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7287_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7287] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074,task_087 8140
launch b task_073,task_075 8141
echo "[t7287] 기동 완료 · sha=$SHA · nt=$NT · a={072,074,087}→8140 · b={073,075}→8141"
