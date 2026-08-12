#!/bin/bash
# **다섯 레버 동작 확인** — 070 → GPU0(8140) · 071 → GPU1(8141) · 각 nt=2 · ON 팔만.
# 성적 아님(nt=2 는 통계가 아니다) — **기구가 우는가**와 원인 분포를 읽는다.
#
# 이번 델타 = **`T2_ARG_AXIS=1`**(원장 C444). 직전 런에서 모델이 `account_type="checking"`
#   (개인)으로 사업자 계좌를 열었다. x275(n=8×5팔): 직접 물으면 라이브 11K자 문맥서도 **8/8**,
#   `business` 낱말을 다 지워도 8/8, 부정통제(개인 요청) **0/8** ⇒ 말할 수 있는데 인자에 안 쓴다.
#   엔진은 **LLM 출력 둘을 맞대고 다르면 그 사실만** 알린다(고르지 않는다·C10 오차단 방지로
#   여러 값 허용·인자가 집합의 원소면 통과).
#
# 함께 확인 (직전 런이 남긴 것):
#   · CP2 시점 교정(C443) — `route` 에 `agent=decision_carry · **arrived=True**` 가 찍히는가
#   · STEP2 — 직전 사이드카 070 8건·unlock 0→1. 이번엔 unlock→call 까지 가는가
#   · ENUM over-block 0 · Δspurious(거부 수·`_FB_GENERIC`)
# 대조군은 돌리지 않는다 — 기준선 = `bank_dbw_off_20260812`(nt=4·8 sim).
#
# 읽을 것 (성적 아님 · nt=2 는 통계가 아니다):
#   ① `[T2_WRITE_ARG_ENUM] deny` 가 우는가 → ② 그 다음 턴에 **집합 內 이름으로 고쳐 쓰는가**
#   ③ `[T2_DECIDE_BEFORE_WRITE] 유예` 가 우는가 (직전 0회)
#   ④ 집합 內 이름이 거부당한 건 **0** 이어야 한다 (over-block)
#   ⑤ Δspurious: `_FB_GENERIC`·거부 수·gold 밖 쓰기
#
# usage: run_lever_20260812g.sh
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
for t in ("test_regen_break_guard.py",   # ★이걸 안 돌려 두 레버가 죽은 채로 유료 런이 돌았다
          "test_discovery_step2.py",
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
  TAG="bank_lever_${TASK##task_}_20260812g"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[lever] SKIP: $LOG/${TAG}.log 가 이미 있다." >&2; return 0
  fi
  # ★결과 디렉토리도 본다 (2026-08-12): 로그만 지우고 재실행했더니 tau2 가 덮어쓸지 **물었고**,
  #   stdin 이 /dev/null 이라 `EOFError: EOF when reading a line` 으로 두 런이 즉사했다.
  #   대화형 프롬프트는 이 환경에서 곧 죽음이므로 **먼저 거부**한다.
  if [ -e "$SIMS/${TAG}" ]; then
    echo "[lever] REFUSING: $SIMS/${TAG} 가 이미 있다 — 지우고 다시 걸어라." >&2; return 1
  fi
  if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
    echo "[lever] REFUSING: 포트 ${PORT} 사용 중." >&2; return 1
  fi
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASK\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"on\"}" \
    | tee "$LOG/${TAG}.meta.json"
  setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 \
           T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 \n           T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 && \
    t2_launch $TAG $PORT '$TASK' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1 &
  echo "[lever] $TASK → PID=$! port=$PORT log=$LOG/${TAG}.log"
}

launch task_070 8140
launch task_071 8141
echo "[lever] 기동 완료 · sha=$SHA · nt=$NT · ON 팔만"
