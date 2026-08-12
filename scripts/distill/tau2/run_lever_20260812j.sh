#!/bin/bash
# **4태스크 표적 재런** — 070,071 → GPU0(8140) · 010,099 → GPU1(8141) · 각 nt=2 · ON 팔만.
# (사용자 지시 2026-08-12: "070 071 010 099 네개만 nt=2 로 걸어서 ③ⓓ 계열과 ② 계열도 마무리하라"
#  · 6태스크 판정 런은 저녁에 별도.)
#
# 이번 델타 = **없음(레버 동일)** + `[T2_ACTIONREQ]` 침묵-사유 계측 1줄 (batch4 010 trial0 [24]
#   부검: 창이 열렸는데 [ACTION] 이 침묵한 턴의 원인을 로그로 특정할 수 없었다 — §4 "계기의
#   사각이 음성 관측으로 보인다").
#
# 이 런이 판정하는 것 (batch4 는 오늘 스택 **이전**이었다):
#   · 010: 진단(창 산수)은 라이브서 이미 정확했다 — 결정("resubmit 가능")이 transfer 로 접히는
#     knowing-doing 접힘을 **오늘 스택(CARRY 시점교정·DBW·STEP2)** 이 닫는가.
#     [T2_ACTIONREQ] 로그로 접힘 턴의 [ACTION] 침묵 사유를 처음으로 특정할 수 있다.
#   · 099: 계좌조회 누락(②·unlock→call 단절)을 STEP2 가 닫는가 — batch4 trial1 은 unlock 까지
#     하고 call 을 안 했다(STEP2 배선 前). 자격 argmax(①·Hunter Green vs gold World Blue)는
#     보류 축 — read 성사 후 재료 도달 시 픽이 바뀌는지 **관찰만**.
#   · 070/071: i 런 대비 ENUM 교정 성공(deny 후 집합 內 재작성)·AXIS deny·Δspurious 지속 관측.
#
# 성적 아님(nt=2) — 기구 동작과 원인 분포를 읽는다. 대조군 없음(기준선 = bank_dbw_off/batch4).
#
# usage: run_lever_20260812j.sh
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
  scripts/distill/tau2/a2/ | grep -cv '^??' || true)
if [ "$DIRTY" != "0" ]; then
  echo "[lever] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
          "test_axis_levers.py"):
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
  TAG="bank_lever4_${NAME}_20260812j"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[lever] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[lever] REFUSING: $SIMS/${TAG} 가 이미 있다 — 지우고 다시 걸어라." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[lever] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[lever] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_070,task_071 8140
launch b task_010,task_099 8141
echo "[lever] 기동 완료 · sha=$SHA · nt=$NT · ON 팔만"
