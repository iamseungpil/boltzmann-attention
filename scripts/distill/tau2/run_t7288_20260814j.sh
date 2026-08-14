#!/bin/bash
# **FIX-13(write-착수 격리 서브) 첫 라이브 + 서브-라이브러리 리팩토링 회귀** (nt=1·판정 아님).
#
# 탑재 = t7287 구성 + 이번 세션 추가분:
#   FIX-13 `T2_WRITE_SUB=1` — 회피가 확정된 자리에서 격리 서브가 호출을 형식화하고, 엔진은
#     **닫힌 술어 둘**(도구명 ∈ 실재 레지스트리 · 값의 근거 실재)만 검산해 통과분을 리마인더로
#     올린다. 실행은 메인이 한다(엔진 실행 0). 근거 = x307(knows 7/8 ↔ acts 0/8)·x308(서브 7/8·
#     JSON 8/8·근거 없으면 0/8)·x309(전달 시 8/8)·x310(근거 동봉해도 정답 8/8·틀린 제안 순응 0/8).
#   리팩토링 — 단발-격리 서브 24곳을 `t2_subcall` 정본으로 이관(거동 보존 의도). **이 런의 절반은
#     그 회귀 확인**이다: 서브 계열 레버가 종전처럼 발화하는가.
#
# 관측 포인트(성적 아님·nt=1):
#   ⓐ `[T2_WRITE_SUB] 제안 N건 → 근거검산 통과 M건` 발화와 그 직후 메인의 credit 호출
#   ⓑ 검산 탈락(M=0) 시 종전 문면으로 조용히 폴백하는가
#   ⓒ 리팩토링 회귀: `[T2_SUBCALL]` 실패 인쇄 0 · 기존 서브 마커(STEP2·claimprov·SG_ISOLATE) 정상
#   ⓓ Δspurious: write 가 필요 없는 자리(075)서 잘못 발화하지 않는가
#
# usage: run_t7288_20260814j.sh
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
  echo "[t7288] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
          "test_write_initiation_sub.py", "test_subcall_canonical.py"):
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
  TAG="bank_t7288_${NAME}_20260814j"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7288] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7288] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7288] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7288_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7288] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074 8140
launch b task_073,task_075 8141
echo "[t7288] 기동 완료 · sha=$SHA · nt=$NT · a={072,074,087}→8140 · b={073,075}→8141"
