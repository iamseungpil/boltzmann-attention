#!/bin/bash
# **alltools 복귀 + 오늘 두 레버 라이브 첫 판정** (2026-08-15·nt=2).
#
# t7293(shell 단독)은 기각됐다 — 0/8 통과·075 2/2→0/2·전 태스크 악화(C487). 빼기는 기계적으로
# 작동했으나(shell 51·bm25 0) **도구 채택 ≠ 검색 성능**이었다. `alltools` 로 되돌린다.
#
# 이 런에 처음 실리는 것 둘(둘 다 n=24·블록 8·8·8 로 사전 측정·잡음 바닥 ±4 밖):
#   ⑴ **지목 → 범위 표면화**(x322): A_REF **24/24** ↔ B_PINPOINT(현행 지목) **0/24** ↔
#      C_SCOPES(범위 표시) **24/24**. 우리 개입이 모델이 맞히던 것을 파괴하고 있었고,
#      073 의 중복 적립(C485)이 그 손실의 라이브 착지였다. 엔진은 이제 **고르지 않는다**.
#   ⑵ **환급 차감**(x323 + 072 원장 검산): 부과 24.00 − 환급 10.00 = **14.00 = gold**.
#      환급 축자를 줘도 0/24·정책 문면까지 줘도 0/24·엔진이 뺀 값 24/24 ⇒ 뺄셈 한 칸만 정당.
#
# 판정(사전 고정):
#   배선  `t2_liveness` 3축 — T2_ACTION_INDEX 발화>0 ∧ **도달 위험 0** (아니면 성적 판정 보류)
#   ⑴    `operator-scope` 발화>0 ∧ `operator-find`(지목) **0** ∧ 073 중복 write 0
#   ⑵    072 가 **14.00** 을 크레딧하는가(도구 반환 delta_total 로 확인)
#   성적  072~075 · nt=2 · **reward/db_match 로 판정**(C486: `gold N/M` 은 소수점 표기로 무너진다)
#   ⚠nt=2 는 방향용이다(C467) — 확정은 nt=4.

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
  echo "[t7294] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
          "test_grounded_calls_nested.py", "test_operator_find_executed.py",
          "test_rebate_netting.py", "test_atm_fee_op.py",
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
  TAG="bank_t7294_${NAME}_20260815a"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7294] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7294] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7294] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7294_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7294] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074 8140
launch b task_073,task_075 8141
echo "[t7294] 기동 완료 · sha=$SHA · nt=$NT · a={072,074,087}→8140 · b={073,075}→8141"
