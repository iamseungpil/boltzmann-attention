#!/bin/bash
# **shell arm** — 검색 도구를 **빼서** shell 로 몰아넣는다 (2026-08-14 야간·사용자 지시
# *"072 는 shell 로 하는 게 우리 정본이다. 왜 아직도 bm25 로 하나?"*).
#
# 왜: 정본 실측이 이미 있다 — x236/C404 *"도구를 `shell` 하나로 줄이면 **8/8** 이 그것을 쓴다 —
#   도구 선택은 **빼기**로 닫힌다"*. 그런데 라이브는 계속 `alltools`(bm25·dense·grep 전부)였고,
#   모델은 bm25 를 집었다. **빼기를 라이브에 적용한 적이 없다.**
# 무엇이 걸려 있었나: [[54]] 리더보드 정합이 `alltools` 를 기본으로 못 박았는데, go_stack 주석
#   축자가 *"보드 상위권 전부 **alltools/Terminal**"* — **Terminal 계열도 보드에 있다**. 정합 유지.
#
# 이 arm 이 겨누는 실물(072·t7292 t0·gold 7/9):
#   질의 `"apply fee refund"` 를 **냈는데** bm25 가 doc 017 을 못 줬다(C481: refund 어휘는 미적중·
#   credit 어휘라야 rank 1). 문서는 `refund` 를 4회 쓴다 — **없어서가 아니라 698 개 안에서 밀린다**.
#   실측: `fee refund` 를 **action 문서 43개 안에서** 찾으면 **2건**이고 그 둘이 checking(017)·savings
#   쌍둥이다 — 072 가 헷갈린 그 쌍이 제목으로 나란히 갈린다.
#
# 단일 변수: `GO_RETRIEVAL=terminal_use` (나머지 플래그·sha 는 t7292 와 동일)
# 판정: 072 가 doc 017 에 **닿는가** · gold 7/9 를 넘는가 · 073/075 회귀 0 · max_steps 0
# ⚠환경 변이를 바꾸는 것이므로 t7292 와의 비교는 **이 한 변수**에만 귀속한다([[57]]).

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
  echo "[t7293] REFUSING: 엔진 경로 커밋 안 된 변경 $DIRTY 개." >&2; exit 1
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
  TAG="bank_t7293_${NAME}_20260814q"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[t7293] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[t7293] REFUSING: $SIMS/${TAG} 잔존." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[t7293] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\",\"frozen\":\"t7293_20260813\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export GO_RETRIEVAL=terminal_use T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \
           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[t7293] $TASKS → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch a task_072,task_074 8140
launch b task_073,task_075 8141
echo "[t7293] 기동 완료 · sha=$SHA · nt=$NT · a={072,074,087}→8140 · b={073,075}→8141"
