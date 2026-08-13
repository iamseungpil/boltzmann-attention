#!/bin/bash
# **072/073 재실험 (양GPU 분산)** — 사용자 지시 2026-08-13: "071 라이브 끝나는 대로 072/073 을
# GPU 2개에 재배치하여 속도를 높여라". 072→8140 · 073→8141 · nt=4 · 동결.
#
# nt=1 포렌식(t7273) 후 수정 3종 반영 (커밋 97550721·02dbcefd·검정 25/25):
#   FIX-1 STEP2 none→레지스트리 재질의(073t1 turn35/37/45: 무관 후보에 none→credit 도구가
#         후보에 든 적 없음 — 도달 구멍 폐쇄)
#   FIX-2 같은-이름 재푸시 sim당 2회 캡([[57]]·9회 스팸 실측)
#   FIX-3 not_discoverable deny 에 레지스트리 목록 동봉(x287b: A 0/8 → B 8/8 — 사전등록
#         문턱 통과·"수동 조정" 날조 접힘의 해법)
#
# 읽을 것: ①072 credit 3칸(072_6~8)·073 credit 4칸(073_7~10) 도달 ②[T2_DISCOVERY_STEP2]
#   재질의·재푸시 억제 발화 ③목록-동봉 발화 후 다음 호출 ④over-block 0·Δspurious.
#
# usage: run_t7273b_20260813u.sh
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
  echo "[t7273b] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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

launch () {
  NAME="$1"; TASKS="$2"; PORT="$3"
  TAG="bank_t7273b_${NAME}_20260813u"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7273b] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7273b] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7273b] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7273b_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7273b] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072 8140
launch b task_073 8141
echo "[t7273b] 기동 완료 · sha=$SHA · nt=$NT · 양GPU 분산(072→8140·073→8141)"
