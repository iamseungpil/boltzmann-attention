#!/bin/bash
# **밤샘 병렬 배치** — 010·098·099·100 × nt=4 를 **8141(GPU1)** 에 dbw 런과 병렬로 태운다
# (사용자 지시 2026-08-12 야간: 배치 지시 + *"8141 은 사용하지 않나?"* — 유휴 GPU 활용·
# all6a/b 선례). 아침 확인용 스냅샷.
#
# 플래그 = 전 레버 ON([[60]] 합성 우선): dbw ON 팔 스택 + T2_DECIDE_BEFORE_WRITE=1.
#   (결정-ask 재배선·후보 줄은 무플래그 공통 — C437 계약 복원.)
# ⚠010 은 dbw 사전등록 밖(§8 N4) — 이 배치의 010·098~100 은 **칸 분포·P0 만** 읽고
#   레버 효과를 주장하지 않는다. 098~100 은 회귀 확인(직전 전부 pass·C426)이 목적.
#
# usage: run_batch4_20260812.sh
set -e
REPO=/home/woori/workspace_common/boltzmann-attention-pi
cd "$REPO/scripts/distill/tau2"

TASKS="task_010,task_098,task_099,task_100"
PORT=8141
NT=4
TAG="bank_batch4_20260812"
LOG=/home/woori/scratch/logs

if [ -e "$LOG/${TAG}.log" ]; then
  echo "[batch4] REFUSING: $LOG/${TAG}.log 가 이미 있다." >&2; exit 1
fi

if ps -eo cmd | grep -v grep | grep "t2_run_gated.py" | grep -q "localhost:${PORT}/"; then
  echo "[batch4] REFUSING: 포트 ${PORT} 에 이미 런이 있다." >&2; exit 1
fi

SHA=$(cd "$REPO" && git rev-parse --short HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain -- \
  scripts/distill/tau2/t2_gate_patch.py scripts/distill/tau2/t2_eplan_patch.py \
  scripts/distill/tau2/t2_dominance.py scripts/distill/tau2/t2_search.py \
  scripts/distill/tau2/t2_precedence.py scripts/distill/tau2/t2_source.py \
  scripts/distill/tau2/a2/ scripts/distill/tau2/go_stack.sh | grep -cv '^??' || true)
echo "{\"tag\":\"$TAG\",\"scaffold_sha\":\"$SHA\",\"dirty_files\":$DIRTY,\"tasks\":\"$TASKS\",\"port\":$PORT,\"nt\":$NT,\"arm\":\"stack_all_on\",\"frozen\":\"bank_dbw_20260812\"}" \
  | tee "$LOG/${TAG}.meta.json"

bash -c "cd '$REPO/scripts/distill/tau2' && source ./go_stack.sh >/dev/null 2>&1 && \
  export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 \
         T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 T2_DECIDE_BEFORE_WRITE=1 && \
  t2_launch $TAG $PORT '$TASKS' $NT" </dev/null >"$LOG/${TAG}.log" 2>&1
echo "[batch4] 종료 $(date -Is) → $LOG/${TAG}.log"
