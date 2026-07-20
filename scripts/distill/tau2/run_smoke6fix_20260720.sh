#!/bin/bash
# ★smoke6fix (2026-07-20): 크래시 픽스(exec 1:1 재조립) 검증 = 이전 infra_error 3태스크만 재런.
#   목표 = 023/031/043이 infrastructure_error("id mismatch") **없이** 완주하는지 + 그 김에
#   gate1 날짜-grounding(023)·gate4 give-flow(031)·gate3 closure(043) 라이브 발화 확인.
#   최소 scope([[09]]): 3 sim·nt=1·conc=1. A(:8141)=023,031 / B(:8140)=043.
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRATCH=/home/woori/scratch

run_arm() {
  TAG=$1; PORT=$2; TASKS=$3
  cd $SCRATCH/tau2-bench
  source /home/woori/.openrouter_key
  export PATH=/home/woori/bin:$PATH
  export PYTHONPATH=src:$REPO/scripts/distill/tau2
  export T2_GATE_REGEN=1 T2_GROUND=1 T2_EPLAN=1 T2_EPLAN_WALK=1 T2_RESOLVE=1 T2_COMPUTE=1 T2_FAB_STRIP=1 T2_SCAFFOLD_GET=1 T2_TOOLGATE=1
  export T2_FOLLOWUP_REQUIRED=1 T2_WRITE_PROV=1 T2_SG_TRUTH=1
  export T2_A2_VARIANT=ledger,ratefix
  export T2_SG_ISOLATE=1 T2_WRITE_EVIDENCE=1 T2_READ_DEDUP=1
  export T2_ACTION_DENY_CAP=3 T2_FORCE_ACTION=1 T2_CLAIM_PROV=1 T2_VERIFY_DENY_CAP=2 T2_ARG_SCHEMA=1
  export T2_SG_GROUND=1
  export T2_LLM_TIMEOUT=300 T2_LLM_RETRIES=1     # 097 stall 방지: hang 요청 5분 상한·재시도 1(최악 10분→infra_error)
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials 1 --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "SMOKE6FIX_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
  S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
  [ -f "$S" ] && gzip -c "$S" > $REPO/reports/facet_rft_2026/sim_results/$TAG.results.json.gz
  gzip -c $SCRATCH/$TAG.log > $REPO/reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
  [ -f "$T2_SG_ISOLATE_TRACE" ] && gzip -c "$T2_SG_ISOLATE_TRACE" > $REPO/reports/facet_rft_2026/sim_results/${TAG}_operands.jsonl.gz
}

run_arm bank_smoke6fix_a_20260720 8141 task_023,task_031 &
PA=$!
run_arm bank_smoke6fix_b_20260720 8140 task_043 &
PB=$!
wait $PA $PB

cd $REPO
for i in 1 2 3; do
  git pull --rebase -q
  git add -f reports/facet_rft_2026/sim_results/bank_smoke6fix_*
  git commit -q -m "data: smoke6fix (crash-fix validation - 023/031/043 no infra_error, nt=1 conc=1)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED smoke6fix"; break; }
  sleep 15
done
echo "SMOKE6FIX_ALL_DONE"
