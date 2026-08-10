#!/bin/bash
# 스모크 — R8b(`T2_DECISION_ISOLATE=1`)가 **라이브에서 실제로 발화**하는가.
#
# 무엇을 보는가 (n=1 은 성적이 아니라 **발화·무크래시** 확인이다·[[30]])
#   · `[T2_R8B]` 마크가 결정 블록이 나가는 턴마다 찍히는가.
#   · 사이드카에서 그 턴의 메시지에 **이름 목록이 없고 블록은 있는가**.
#   · 크래시 0 · 기존 마크(`T2_KIND`·`T2_REDERIVE`·`T2_D1C`)가 그대로 도는가.
#
# 근거: x231(leave-one-in·n=8) — 실제 문맥 위에서 `ineligible_text` 두 문장만 얹어도
#   task_100 이 0/8 로 무너진다. x227 — 값만 남기면 세 태스크 전부 8/8.
#   설계서 = `reports/facet_rft_2026/DECISION_ACTION_SPLIT_DESIGN_2026_08_10.md`.
#
# ⚠성적 주장 금지: nt=1·3 sim 이다. 이득 판정은 별도 런에서 **켠 팔/끈 팔**을 같이 돌린 뒤에만.
# usage: run_isolate_20260810.sh [TASKS] [NT] [TAG]
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="${1:-task_098,task_099,task_100}"
NT="${2:-1}"
TAG="${3:-bank_isolate_20260810s}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"

if ps -eo cmd | grep -v grep | grep -q "t2_run_gated.py"; then
  echo "[run] REFUSING: t2_run_gated 가 이미 돌고 있다." >&2; exit 1
fi
if [ -e "$LOG/${TAG}.log" ]; then
  echo "[run] REFUSING: $LOG/${TAG}.log 가 이미 있다. 다른 TAG 를 쓰라." >&2; exit 1
fi

# 선-점검: **발사 전 VERIFY OK** — 이 런이 의존하는 것이 실제로 코드에 있는가.
/home/woori/venvs/seka_env/bin/python - <<'PY' || exit 1
import os, subprocess, sys
d = "/home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2"
sys.path.insert(0, d)
bad = []
src = open(os.path.join(d, "t2_gate_patch.py"), encoding="utf-8").read()
if "T2_DECISION_ISOLATE" not in src or "[T2_R8B]" not in src:
    bad.append("R8b 코드가 없다")
# 회귀: 억제가 켜졌을 때만 먹고, 블록 없는 턴은 안 건드린다
r = subprocess.run(["/home/woori/venvs/seka_env/bin/python", "test_decision_isolate.py"],
                   cwd=d, capture_output=True, text=True)
if r.returncode != 0:
    bad.append("test_decision_isolate 실패: " + (r.stdout or "")[-300:])
if os.environ.get("T2_DECISION_ISOLATE") == "1":
    bad.append("검증 프로세스에 플래그가 켜져 있다(런처가 켜야 한다)")
print("VERIFY " + ("FAIL: " + " · ".join(bad) if bad else "OK"))
sys.exit(1 if bad else 0)
PY

setsid bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_DECISION_ISOLATE=1 && t2_launch $TAG 8140 '$TASKS' $NT" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "PID=$!"
sleep 12
echo "--- 발사 직후 ---"
head -15 "$LOG/${TAG}.log" 2>/dev/null || true
echo "launched · tasks=$TASKS nt=$NT · log: $LOG/${TAG}.log"
echo "  sidecar: $LOG/fb_${TAG}.jsonl · trace: $LOG/trace_${TAG}.jsonl"
