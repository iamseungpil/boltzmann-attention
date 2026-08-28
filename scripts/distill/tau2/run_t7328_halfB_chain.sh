#!/bin/bash
# t7328 후반 — **GPU 1장 정책**(사용자 지시 2026-08-19): halfA(8140) 가 끝나면 같은 포트로 이어 돈다.
# GPU1(8141) 은 이론 프로브 전용으로 비워 둔다.
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SIMS=/home/woori/scratch/tau2-bench/data/simulations
TAG=bank_t7328_halfB_20260819r2
while pgrep -f "t2_run_gated.py.*bank_t7328_halfA_20260819r" >/dev/null; do sleep 60; done
echo "[t7328-chain] halfA 종료 감지 — halfB 를 8140 에서 시작 $(date)"
cd $REPO/scripts/distill/tau2 || exit 1
source ./go_stack.sh >/dev/null 2>&1
export T2_ACTION_SUB=1 T2_KEEP_DENY_BODY=1 T2_CALL_FORM=1 T2_ARG_EMPTY=1 T2_SEARCH_AGENT=1 T2_DECIDE_ANY=1 T2_WRITE_ARG_ENUM=1 T2_DECIDE_BEFORE_WRITE=1 T2_DECISION_CARRY=1 T2_DISCOVERY_STEP2=1 T2_ARG_AXIS=1 T2_WRITE_SUB=3 T2_ACTION_INDEX=1 T2_NOW_SELFCALL=1 T2_SEARCH_ON_PROCEED=1 T2_ACT_DEMAND=0 T2_DELIVER_PRECOMMIT=0 T2_PROCEED_DOCBODY=0 T2_DOCS_AT_WRITE=0 T2_SUB_REQUIREMENT=0 T2_HANDOFF_PREDICATE=0 T2_PENDING_DISCOVERED=0 T2_VERDICT_CARRY=0 T2_ELIG_LINE=0 T2_VERDICT_GATE=0 T2_CLAIM_VERIFY=0 T2_DECLFIRST=0 T2_DECLFIRST_GUIDE_FIX=0
export GO_MAX_STEPS=150 GO_CONCURRENCY=1
t2_launch $TAG 8140 task_016,task_033,task_040,task_050,task_057,task_063,task_074,task_079,task_085,task_098 2
cd $REPO && mkdir -p reports/facet_rft_2026/sim_results
gzip -c "$SIMS/$TAG/results.json" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
git add -f reports/facet_rft_2026/sim_results/$TAG.results.json.gz
echo "[t7328-chain] persisted $TAG"
