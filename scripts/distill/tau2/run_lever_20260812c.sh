#!/bin/bash
# **레버 기구 확인 런 (재시도)** — 070 → GPU0(8140) · 071 → GPU1(8141) · 각 nt=2 · **ON 팔만**
# (사용자 지시 2026-08-12: *"070 071 각각 gpu 0, 1 에 nt=2 로"* · *"off 는 이전 데이터가 있으니
# on 만"* · *"on 한번 더 태우지 말고 분석만 — 레버가 통하는지 확인하는 용도"*).
#
# 왜 다시 도나: 직전 `..._20260812b` 는 **죽은 코드로 돌았다**. 두 신규 레버가 재생성 루프의
# 조기 break 가드(`t2_gate_patch.py` §guard) **뒤**에서 계산돼, 그 턴의 유일한 발화일 때
# 계산조차 되지 않았다 — 라이브 증거: `Sky Blue Business Checking`(집합 外)이 나갔는데
# `[T2_WRITE_ARG_ENUM] deny` **0회**. 2026-08-05 `proc_fb` 사고와 같은 실수이고
# `test_regen_break_guard.py` 가 잡았다(이제 아래 VERIFY 에 상설).
#
# 대조군은 돌리지 않는다 — 기준선 = `bank_dbw_off_20260812`(nt=4·8 sim).
#
# 읽을 것 (성적 아님 · nt=2 는 통계가 아니다):
#   ① `[T2_WRITE_ARG_ENUM] deny` 가 우는가 → ② 그 다음 턴에 **집합 內 이름으로 고쳐 쓰는가**
#   ③ `[T2_DECIDE_BEFORE_WRITE] 유예` 가 우는가 (직전 0회)
#   ④ 집합 內 이름이 거부당한 건 **0** 이어야 한다 (over-block)
#   ⑤ Δspurious: `_FB_GENERIC`·거부 수·gold 밖 쓰기
#
# usage: run_lever_20260812c.sh
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

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
for t in ("test_regen_break_guard.py",   # ★이걸 안 돌려 두 레버가 죽은 채로 유료 런이 돌았다
          "test_write_arg_enum.py", "test_decide_before_write.py", "test_route_trace.py",
          "test_a2_three_layer.py", "test_decision_carry.py", "test_decision_isolate.py"):
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
  TASK="$1"; PORT="$2"
  TAG="bank_lever_${TASK##task_}_20260812c"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[lever] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[lever] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASK\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 && \
    t2_launch $TAG $PORT '$TASK' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[lever] $TASK → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch task_070 8140
launch task_071 8141
echo "[lever] 기동 완료 · sha=$SHA · nt=$NT · ON 팔만"
