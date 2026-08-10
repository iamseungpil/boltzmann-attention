#!/bin/bash
# 판정 런 — R8b(결정 턴에서 이름 목록 억제)의 **이득과 대가**를 같은 조건에서 가른다.
#
#   ISO_ON  = `T2_DECISION_ISOLATE=1`  (GPU0·8140)
#   ISO_OFF = 플래그 없음               (GPU1·8141)   ← 부정 통제이자 기준선
#
# 두 팔은 **같은 태스크·같은 시행 수·같은 스택**이고 코드도 같다(플래그 하나만 다르다).
# 그래서 차이의 귀속이 성립한다. 태스크 = 098·099·100(레버를 잰 셋) + **010(회귀 감시**·이
# 레버로 측정된 적이 없다·[[19]]).
#
# 함께 세는 반대편 (설계서 §5 · "부작용 없는 레버는 없다"):
#   hands_over / external / Δspurious(over-action) / 게이트 거부 수 / read 주체
#   ⇒ 런이 끝난 뒤 사이드카·궤적에서 센다(이 스크립트는 재료만 남긴다).
#
# ⚠성적 주장은 **양 팔 전수 포렌식 뒤에만**([[08]]). pass 수치 단독으로 결론 금지.
# ⚠두 팔이 서로 다른 GPU 를 쓴다 — 같은 모델·같은 가중치이고, 유일한 차이는 플래그다.
# usage: run_isolate_ab_20260810.sh [TASKS] [NT] [SUFFIX]
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="${1:-task_098,task_099,task_100,task_010}"
NT="${2:-3}"
SUF="${3:-20260810}"
ON="bank_isoON_$SUF"
OFF="bank_isoOFF_$SUF"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

for T in "$ON" "$OFF"; do
  if [ -e "$LOG/${T}.log" ]; then
    echo "[run] REFUSING: $LOG/${T}.log 가 이미 있다. SUFFIX 를 바꾸라." >&2; exit 1
  fi
done
if ps -eo cmd | grep -v grep | grep -q "t2_run_gated.py"; then
  echo "[run] REFUSING: t2_run_gated 가 이미 돌고 있다." >&2; exit 1
fi

# 발사 전 VERIFY — 이 런이 의존하는 것이 코드에 있고, 회귀가 산다.
/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
sys.path.insert(0, d)
bad = []
src = open(os.path.join(d, "t2_gate_patch.py"), encoding="utf-8").read()
if "T2_DECISION_ISOLATE" not in src or "[T2_R8B]" not in src:
    bad.append("R8b 코드가 없다")
for t in ("test_decision_isolate.py", "test_kind_filter.py", "test_a3_coverage.py",
          "test_rederive_wiring.py"):
    r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", t], cwd=d,
                       capture_output=True, text=True)
    if r.returncode != 0:
        bad.append("%s 실패" % t)
if os.environ.get("T2_DECISION_ISOLATE"):
    bad.append("발사 환경에 플래그가 이미 있다(팔이 오염된다)")
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_DECISION_ISOLATE=1 && t2_launch $ON 8140 '$TASKS' $NT" \
  </dev/null >"$LOG/${ON}.log" 2>&1 &
echo "ON  PID=$!  (8140·flag=1)"
sleep 5
setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  unset T2_DECISION_ISOLATE && t2_launch $OFF 8141 '$TASKS' $NT" \
  </dev/null >"$LOG/${OFF}.log" 2>&1 &
echo "OFF PID=$!  (8141·flag 없음)"
sleep 15
echo "--- 발사 직후 ---"
for T in "$ON" "$OFF"; do
  echo "[$T] $(wc -l < "$LOG/${T}.log" 2>/dev/null || echo 0) 줄"
done
echo "tasks=$TASKS nt=$NT · logs: $LOG/${ON}.log · $LOG/${OFF}.log"
