#!/bin/bash
# **072/073 이동 첫 런** (사용자 지시 2026-08-13: "6태스크가 학습 제외 더 안 좋아지면 072/073
# 이동" + "GPU0 만 사용"). nt=4 · ON 팔만 · 동결 · **8140 단독**(8141 비움).
#
# 6태스크 종결 판정(P_RUN_FORENSIC + x283~x286): 남은 비학습 레버 = STEP2 수리 3종뿐 —
#   ①후보를 discoverable 레지스트리로 제한(넌센스 "unlock call_disc/log_verification" 푸시 제거)
#   ②회수-실패 시 레지스트리 실명 폴백(후보=레지스트리−기시도·선택은 formalize·none 허용)
#   ③폴백 문면 REG_FB(진실 출처절) — x283b(계기수리판): t0 8/8·t1 8/8·t3 6/8 = 사전등록 문턱 통과.
#   나머지(010 GO·098 반박·GB2 문면)는 문턱 미달/무효 → LEARNING_BACKLOG_2026_08_13.
#
# 이 런이 판정하는 것:
#   · 072: t0형(체인 미착수)·t1형("credit 도구 없다" 거짓 단정→날조 완료 선언·gold $14.00/$3.50)
#     에 STEP2 수리 3종이 닿는가 — 폴백의 정조준 표적.
#   · 073: 3계좌 판(coverage 심화)의 첫 스택 성적.
#   · [T2_DISCOVERY_STEP2] "레지스트리 폴백" 발화 수 · over-block 0 · Δspurious.
#
# usage: run_t7273_20260813s.sh
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
  echo "[t7273] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개 — 판정 런은 동결 필수." >&2; exit 1
fi

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
bad = []
for t in ("test_regen_break_guard.py",
          "test_no_undefined_names.py",
          "test_discovery_step2.py",
          "test_write_arg_enum.py", "test_decide_before_write.py", "test_route_trace.py",
          "test_a2_three_layer.py", "test_decision_carry.py", "test_decision_isolate.py",
          "test_axis_levers.py",
          "test_operator_find.py", "test_action_operator.py"):
    if not os.path.exists(os.path.join(d, t)):
        continue
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s: %s" % (t, (r.stdout or "")[-140:]))
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

TAG="bank_t7273_20260813s"
PORT=8140
TASKS="task_072,task_073"
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[t7273] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; exit 0
fi
if [ -e "$SIMS/${TAG}" ]; then
  echo "[t7273] REFUSING: $SIMS/${TAG} 가 이미 있다 — 지우고 다시 걸어라." >&2; exit 1
fi
if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
  echo "[t7273] REFUSING: 포트 ${PORT} 사용 중." >&2; exit 1
fi
echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7273_20260813\"}" \
  | tee "$LOG/${TAG}.meta.json"
setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
         T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
         T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
         T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
  t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "[t7273] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log · sha=$SHA · GPU0 단독"
