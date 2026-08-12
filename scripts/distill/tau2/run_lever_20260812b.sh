#!/bin/bash
# **기구 확인 런** — 070·071 × nt=2 를 두 GPU 에 나눠 태운다 (사용자 지시 2026-08-12 아침:
# *"070 071 은 nt=2 로 실험해서 레버들 동작과 다른 원인들 확인부터 하라"* · *"gpu 0, 1 모두"*).
#
# 이 런이 **판정하는 것**: 성적이 아니라 **기구가 우는가**(P0)와 **남은 원인의 분포**다.
#   nt=2 는 통계가 아니다(C403·C415·C426) — pass 수로 레버 효과를 주장하지 않는다.
#
#   ON  (8140) = 전 스택 + **T2_WRITE_ARG_ENUM=1**(C439⒠④ 이름 날조 19/24)
#                        + **T2_DECIDE_BEFORE_WRITE=1**(가드 교정본 — 직전 런은 0회 발화)
#   OFF (8141) = 전 스택만 (두 신규 레버 OFF) — 귀속용 대조
#
# 읽을 것(순서대로):
#   ① `[T2_WRITE_ARG_ENUM] deny` 발화 수 · 그 뒤 모델이 **집합 內 이름으로 고쳐 쓰는가**
#   ② `[T2_DECIDE_BEFORE_WRITE] write 1턴 유예` 발화 수 (직전 런 0회 → 교정 확인)
#   ③ 제출된 account_class 의 **집합 內/外 비율** (기준선 5/24)
#   ④ write 시도 자체가 없는 sim 수 (기준선 8/16) · 이관(transfer) 발화 수
#   ⑤ Δspurious: `_FB_GENERIC` 발화·거부 수·gold 밖 쓰기
#
# usage: run_lever_20260812b.sh
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="task_070,task_071"
NT=2
LOG=/home/woori/scratch/logs
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
for t in ("test_write_arg_enum.py", "test_decide_before_write.py", "test_route_trace.py",
          "test_a2_three_layer.py", "test_decision_carry.py"):
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
  ARM="$1"; PORT="$2"; EXTRA="$3"
  TAG="bank_lever_${ARM}_20260812b"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[lever] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[lever] REFUSING: 포트 ${PORT} 에 이미 런이 있다." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"$ARM\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 $EXTRA && \
    t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[lever] $ARM → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch on  8140 "T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1"
launch off 8141 ""
echo "[lever] 두 팔 병렬 기동 · sha=$SHA"
