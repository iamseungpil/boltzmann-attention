#!/bin/bash
# **FIX-13b 사전(pre-draft) sync 서브 첫 라이브** (nt=1·판정 아님·사용자 지시).
#
# 무엇이 바뀌었나: 발화 자리를 **초안 이후 → 도구 결과 직후(초안 이전)** 로 옮겼다.
#   근거(측정): 종전 배선은 옳은 시점의 **27%만** 잡았고(감사 21 sim·t7288 32%), 나머지 73%에
#   끼어들면 그 턴 호출을 버리는데 그중 **33%가 버리면 안 되는 것**(write 23%=주로 log_verification·
#   새 read 10%·MISS 감사 101건). 사전 자리에는 **버릴 초안이 없다**.
#   한 근거당 1회·이미 실행된 도구는 후보에서 제외·검산 통과분만 전달(전부 검정 4/4).
#
# 관측 포인트(사전등록 `WRITE_SUB_TRIGGER_DESIGN_2026_08_14.md` §3):
#   A 시점: `[T2_WRITE_SUB] pre-draft 전달` 발화가 **근거 도착 턴**에 붙는가 · 공전 감소
#   B 부작용(하나라도 위반 시 되돌린다):
#     · 궤적 길이가 t7285 nt=4 범위의 1.3배 이내(072 ≤130·073 ≤113·074 ≤124·075 ≤59)
#     · **max_steps 0건** (t7288 에서 073 이 202msg·max_steps 로 죽었다)
#     · over-action(gold 밖 write 성공) 증가 0 · 서브 호출 수 ≤ t7288
#
# usage: run_t7289_20260814k.sh
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
  echo "[t7289] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
          "test_dispatch_history_guard.py", "test_resolve_cap_name_reset.py",
          "test_write_initiation_sub.py", "test_subcall_canonical.py",
          "test_write_sub_predraft.py"):
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
  TAG="bank_t7289_${NAME}_20260814k"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7289] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7289] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7289] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7289_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=2 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7289] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074 8140
launch b task_073,task_075 8141
echo "[t7289] 기동 완료 · sha=$SHA · nt=$NT · a={072,074,087}→8140 · b={073,075}→8141"
