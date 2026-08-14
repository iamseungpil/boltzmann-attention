#!/bin/bash
# **nt=2 판정 런** (2026-08-14 야간·사용자 지시 "nt=2 로 발사하라").
#
# t7291 이후 바뀐 것 — 레버 하나 + 배관 하나:
#   ⑴ **검산기 수리**: `grounded_calls` 가 중첩 계약 `{tool, arguments{...}}` 의 값을 못 읽어
#      **우리 형식을 지킨 write 제안을 100% 기각**했다(072 `제안 N건 → 통과 0건` ×8·양 런).
#      073 이 통과했던 자리는 모델이 우리 형식을 **어겼을 때**뿐이었다. 라이브 첫 검증이다.
#   ⑵ **ACTION-INDEX 출시**(`T2_ACTION_INDEX=1`): A3 `action_index` 43줄 1회 표면화.
#      x319·n=24(블록 8·8·8·잡음 바닥 ±4 밖): 도움 없음 **10/24** → 43줄 **24/24** ·
#      도구 설명 91종 23/24 · 이름만 91종 16/24 ⇒ 표면화가 열고 **의미가 이름보다 낫다**.
#
# 판정(사전 고정·nt=2 라 시행 2회):
#   배선  `t2_liveness` 에서 **T2_SEARCH_AGENT delivered>0 ∧ T2_ACTION_INDEX 발화>0**
#         (0 이면 나머지 수치는 판정에 쓰지 않는다 — 오늘 이걸 안 봐서 하루를 태웠다)
#   착수  072 의 `[T2_WRITE_SUB] 제안 N → 통과 M` 에서 **M>0** 이 나오는가(⑴의 직접 표적)
#   성적  072~075 · nt=2 · 073/075 통과 유지 · max_steps 0
#   ⚠nt=2 도 판정 도구로는 약하다(C467: 074 가 9/13↔0/13). **방향만** 읽고 확정은 nt=4 로.

set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

NT=2
LOG=/home/woori/scratch/logs
SIMS=/home/woori/scratch/tau2-bench/data/simulations
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_resolve.py scripts/distill/tau2/t2_compute.py \
  scripts/distill/tau2/t2_scaffold_get.py scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[t7292] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
          "test_write_sub_predraft.py", "test_subcall_return_type.py", "test_action_index.py",
          "test_grounded_calls_nested.py",
          "test_forensic_canonical.py", "test_omitted_rows_note.py", "test_bailout_axes.py"):
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
  TAG="bank_t7292_${NAME}_20260814p"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7292] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7292] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7292] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7292_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7292] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074 8140
launch b task_073,task_075 8141
echo "[t7292] 기동 완료 · sha=$SHA · nt=$NT · a={072,074,087}→8140 · b={073,075}→8141"
