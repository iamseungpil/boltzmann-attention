#!/bin/bash
# ★rall11 (2026-07-22·§2bt·rall10 판정 반영판): rall10 스택 + §2bt 픽스 5종 스위치 ON.
#   신규: T2_FOLLOWUP_PROGRESS_REFUND(1b·cap-소진 구조)·T2_ACTION_PROGRESS_REFUND(097 say-loop 동형)·
#   T2_VIEW_ANNOTATE(054 카드≠계좌 오독)·T2_WRITE_ARG_GROUND(031 '1234' fabrication).
#   +A2 갱신 반영: name_words 질의힌트(1a)·give=자기도구 deny(2b)·closure chain 리텐션 재정렬(1c).
#   대상 8태스크 = rall10 동일(031/038/023/097 ∥ 043/054/052/050)·nt=1·seed 300.
#   판정축: [043] chain 환급→사슬 완주·decision(waive|close) [050/052] name_words→suffixed unlock 성공률
#   [031] give-agent-tool deny→올바른 give·WRITE_ARG_GROUND 발화 [054] VIEW_ANNOTATE→CLI 진행
#   [097] ACTION refund→say-loop 소멸·write 도달. 교차표=액션-매치+시도-수준 병기(§2bh).
#   ⚠full launch 전 smoke 의무([[30]]) — smoke=task_050(chain·UNLOCK_NAME·refund·name_words 라이브 발화 확인).
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRATCH=/home/woori/scratch
MODE=${1:-smoke}

run_arm() {
  TAG=$1; PORT=$2; TASKS=$3; NT=$4
  cd $SCRATCH/tau2-bench
  source /home/woori/.openrouter_key
  export PATH=/home/woori/bin:$PATH
  export PYTHONPATH=src:$REPO/scripts/distill/tau2
  export T2_GATE_REGEN=1 T2_GROUND=1 T2_EPLAN=1 T2_EPLAN_WALK=1 T2_RESOLVE=1 T2_COMPUTE=1 T2_FAB_STRIP=1 T2_SCAFFOLD_GET=1 T2_TOOLGATE=1
  export T2_FOLLOWUP_REQUIRED=1 T2_WRITE_PROV=1 T2_SG_TRUTH=1
  export T2_A2_VARIANT=ledger,ratefix
  export T2_SG_ISOLATE=1 T2_WRITE_EVIDENCE=1 T2_READ_DEDUP=1
  export T2_ACTION_DENY_CAP=3 T2_FORCE_ACTION=1 T2_CLAIM_PROV=1 T2_VERIFY_DENY_CAP=2 T2_ARG_SCHEMA=1
  export T2_SG_GROUND=1 T2_SG_TRACE=1
  export T2_LLM_TIMEOUT=480 T2_LLM_RETRIES=1
  export T2_REGEN_BUDGET=12 T2_CLAIMPROV_CAP=3 T2_SG_REQREADS=1
  export T2_SG_ISOFB=1 T2_TOOLLIST=1 T2_PAIRCHECK=1 T2_UNLOCK_NAME=1 T2_PAIRFIX=1 T2_VIEW_COMPACT=1 T2_FOLLOWUP_READLOOP=1 T2_DISPATCH_ROLE=1 T2_PARAM_CAP=1
  export T2_FOLLOWUP_CAP=3 T2_FOLLOWUP_FORCE=1
  # ★§2bt 픽스 스위치 (rall11 신규)
  export T2_FOLLOWUP_PROGRESS_REFUND=1 T2_ACTION_PROGRESS_REFUND=1 T2_VIEW_ANNOTATE=1 T2_WRITE_ARG_GROUND=1
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials $NT --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "RALL11_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
  S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
  cd $REPO
  [ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
  gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
  [ -f "$T2_SG_ISOLATE_TRACE" ] && gzip -c "$T2_SG_ISOLATE_TRACE" > reports/facet_rft_2026/sim_results/${TAG}_operands.jsonl.gz
}

persist() {
  cd $REPO
  for i in 1 2 3; do
    git pull --rebase -q
    git add -f reports/facet_rft_2026/sim_results/bank_rall11*
    git commit -q -m "data: rall11 $MODE (2bt fixes live: refunds+annotate+arg-ground+name_words+give-deny+closure-chain)" 2>/dev/null
    git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED rall11 $MODE"; break; }
    sleep 15
  done
}

if [ "$MODE" = "smoke" ]; then
  run_arm bank_rall11s_20260722 8141 task_050 1
  persist
  echo "RALL11_SMOKE_ALL_DONE"
else
  run_arm bank_rall11a_20260722 8141 task_031,task_038,task_023,task_097 1 &
  PA=$!
  run_arm bank_rall11b_20260722 8140 task_043,task_054,task_052,task_050 1 &
  PB=$!
  wait $PA $PB
  persist
  echo "RALL11_FULL_ALL_DONE"
fi
