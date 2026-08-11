#!/bin/bash
# **판정 런** — 070·071 × nt=4 · `T2_DECIDE_BEFORE_WRITE` ON/OFF 두 팔 (사전등록 = 원장
# C436·C437 · 세션 2026-08-12 승인 · 동결 sha=c5ace835 · tag=bank_dbw_20260812).
#
# 무엇을 판정하나
#   ON 팔  = OFF 팔 + T2_DECIDE_BEFORE_WRITE=1 (결정 없이 write 가 오면 1턴 유예 + 그 자리
#            서브 → 답을 deny 에 담아 돌려준다 · C436)
#   공통   = 결정-ask 재배선(후보 줄만 · 격리 계약 복원 · C437 — 계약 위반의 복원이라
#            팔로 가르지 않는다[[60]]) + all6 스택(아래 플래그).
#
# 게이트 (성적보다 먼저 읽는다 · 어기면 그 항목은 주장하지 않는다)
#   P0  arrived — T2_ROUTE_TRACE 가 이제 억제↔체인↔미생성 삼분을 낸다(5998dbff)
#   P1  표적 칸 070_4·071_4·071_5: ON ≥ 3/4 ∧ ON > OFF
#   P4  Δspurious ≤ 0 (유예가 만든 _FB_GENERIC·거부 수·gold 밖 쓰기 포함)
#   P5  BADTOP 0 유지
#   P6  서브 산출 형태 — 맨이름 비율(C437 전이·[T2_DOCDECIDE] 로그로 센다)
#   동결: 시작 SHA = 종료 SHA ∧ dirty 0. 어기면 성적을 읽지 않는다(C423⒞).
#
# usage: run_dbw_20260812.sh            # GPU0(8140)에 ON·OFF 순차 (8140 = 유료 런 자리)
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="task_070,task_071"
PORT=8140
NT=4
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- scripts/distill/tau2 | wc -l)
# 동결의 뜻 = **엔진 8경로 불변**(freeze.py DEFAULT_PATHS)이지 HEAD 고정이 아니다 —
# 런처·프로브·원장 커밋은 허용(handoff: "거는 것은 런처다"). 그래서 HEAD 동등이 아니라
# 동결 SHA 이후 엔진 diff 0 을 검사한다.
FROZEN=c5ace835
ENGINE_DIFF=$(cd "$REPO" && git diff --name-only "$FROZEN"..HEAD -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_eplan_patch.py \
  scripts/distill/tau2/t2_dominance.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_precedence.py scripts/distill/tau2/t2_source.py \
  scripts/distill/tau2/a2/ scripts/distill/tau2/go_stack.sh | wc -l)
if [ "$ENGINE_DIFF" != "0" ]; then
  echo "[run] REFUSING: 동결($FROZEN) 이후 엔진 경로에 변경 $ENGINE_DIFF 건 — 사전등록 무효." >&2
  exit 1
fi
if [ "$DIRTY" != "0" ]; then
  echo "[run] REFUSING: 커밋되지 않은 변경 $DIRTY 개 — 동결 위반." >&2; exit 1
fi
if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
  echo "[run] REFUSING: 포트 ${PORT} 에 이미 런이 있다." >&2; exit 1
fi

for ARM in on off; do
  TAG="bank_dbw_${ARM}_20260812"
  if [ -e "$LOG/${TAG}.log" ]; then
    echo "[run] REFUSING: $LOG/${TAG}.log 가 이미 있다." >&2; exit 1
  fi
done

/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
bad = []
for t in ("test_route_trace.py", "test_decide_before_write.py", "test_decision_carry.py",
          "test_a2_three_layer.py", "test_search_agent_wiring.py", "test_decision_isolate.py"):
    if not os.path.exists(os.path.join(d, t)):
        continue
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s: %s" % (t, (r.stdout or "")[-140:]))
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

run_arm () {
  ARM="$1"; EXTRA="$2"
  TAG="bank_dbw_${ARM}_20260812"
  echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"$ARM\",\"frozen\":\"bank_dbw_20260812\"}" \
    | tee "$LOG/${TAG}.meta.json"
  bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
    export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
           T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 $EXTRA && t2_launch $TAG $PORT '$TASKS' $NT" \
    </dev/null >"$LOG/${TAG}.log" 2>&1
  echo "[run] $ARM 팔 종료 → $LOG/${TAG}.log"
}

# 순차 실행 — 같은 포트라 병렬 불가. ON 먼저(관심 팔이 먼저 죽는 사고를 피하려면 반대?
# 아니다: 순서 편향을 피할 방법이 같은 런엔 없고, ON 이 먼저면 실패 시 OFF 비용을 아낀다.
run_arm on "T2_DECIDE_BEFORE_WRITE=1"
run_arm off ""

SHA2=$(cd "$REPO" && git rev-parse --short HEAD)
echo "[run] 종료 SHA=$SHA2 (시작 $SHA) — 다르면 성적을 읽지 않는다"
