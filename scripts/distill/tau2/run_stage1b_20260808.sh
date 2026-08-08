#!/bin/bash
# 단계 1⒝ 측정 런 — **갈림 자리 N을 센다** (정본 `FACT_DAG_DESIGN_2026_08_08.md` §7e).
#
# 왜 이 런이 필요한가: §7e가 *"측정 불가 — 기존 산출물로는 닫히지 않는다"* 라고 못박았다.
# 지난 6런의 **stderr가 남아 있지 않아** `route()`의 반사실을 재구성할 수 없고, 오프라인에서
# 되살리려면 라이브 판정을 **두 벌로 다시 써야** 한다(=이 코드베이스의 T1을 만드는 행위).
# ⇒ stderr를 보존한 채 한 번 돌리는 것이 유일한 길이고, 추가 비용은 그 런뿐이다.
#
# ★scope 최소화([[09]]): **task_100·101 × 1 trial = sim 2개**. 유료 요소는 user-sim
#   (`openrouter/gpt-5.2`) 하나이고 에이전트는 로컬 vllm($0)이다. 탐색이 아니라 **계수**라
#   더 넓힐 이유가 없다.
# ★스택은 **정본 go_stack 그대로** 둔다 — arm처럼 플래그를 깎으면 재는 갈림이 실제 것이 아니다.
#
# 산출: `$LOG/<TAG>.log` (stderr 포함) → `x134_route_divergence.py`가 갈림 목록을 뽑는다.
#       `x134`는 **사라짐**(층 미분류로 후보가 통째 버려진 경우)을 갈림과 **갈라 센다**.
#
# usage: run_stage1b_20260808.sh [TASKS] [NT] [TAG]
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"
source ./go_stack.sh >/dev/null 2>&1

TASKS="${1:-task_100,task_101}"
NT="${2:-1}"
TAG="${3:-bank_stage1b_20260808}"
LOG=/home/woori/scratch/logs
mkdir -p "$LOG"
export T2_FB_SIDECAR="$LOG/fb_${TAG}.jsonl" T2_FB_SIDECAR_TEXT=1
export -f t2_launch
setsid bash -c "cd '$REPO/scripts/distill/tau2' && \
  T2_FB_SIDECAR='$LOG/fb_${TAG}.jsonl' t2_launch $TAG 8140 '$TASKS' $NT" \
  </dev/null >"$LOG/${TAG}.log" 2>&1 &
echo "PID=$!"
sleep 3
echo "launched · tasks=$TASKS nt=$NT · log: $LOG/${TAG}.log"
