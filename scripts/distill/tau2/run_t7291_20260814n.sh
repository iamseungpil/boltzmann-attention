#!/bin/bash
# **배선 수리 후 072 재측정** (2026-08-14 야간·원장 C482).
#
# 무엇이 바뀌었나 — 레버가 아니라 **배관**이다:
#   `sub_generate` 반환형 오용 8곳 수리. 그중 `t2_ledger.formalize_now` 사망이
#   `now` 를 영구 미해소로 만들어 **`T2_SEARCH_AGENT` 가 t7290 에서 10회 발화·10회 침묵**했다.
#   같은 레버의 결정 단계 `t2_search.doc_decide` 도 함께 죽어 있었다.
#   ⇒ 이번 런은 **온톨로지 검색이 처음으로 살아 있는 상태**의 072/074 다.
#
# 판정(사전 고정):
#   생존  `t2_liveness` 에서 **T2_SEARCH_AGENT delivered > 0** (이게 0 이면 나머지는 볼 것도 없다)
#   귀속  072 가 `apply_checking_account_credit_5829` 를 **보게 되는가**(궤적 등장 여부) —
#         C479 의 전제(그 이름이 대화에 미등장)가 배선 탓이었는지 여기서 갈린다
#   성적  073/075 **통과 유지**(회귀 0) · max_steps 0
#   ⚠성적은 nt=1 이라 판정 도구가 아니다([[57]]·C467) — 이번 런의 1차 목적은 **배선 생존 확인**이다.

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
  echo "[t7291] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
          "test_write_sub_predraft.py", "test_subcall_return_type.py",
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
  TAG="bank_t7291_${NAME}_20260814n"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7291] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7291] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7291] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7291_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7291] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074 8140
launch b task_073,task_075 8141
echo "[t7291] 기동 완료 · sha=$SHA · nt=$NT · a={072,074,087}→8140 · b={073,075}→8141"
