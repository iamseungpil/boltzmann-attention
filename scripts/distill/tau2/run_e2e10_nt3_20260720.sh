#!/bin/bash
# ★e2e10 (2026-07-20·§2aq): 마지막-1홉 픽스 5종 라이브 재검증 + nt=3 로버스트化.
#   대상 = e2e9서 0.0/크래시였던 7태스크(023/095 제외: 023=2회 1.0 실증·095 잔여=learn 축).
#   신규 스택: chain 피드백 unlock 명시·BLOCKED 에이전트-측 pay·관문4 WEV(last4)·transfer-창 예산·
#   TOOLGATE env-실재 통과·tau2 패치 v2·overflow 가드 base+FD·APY fetch-iso(§2aq).
#   판정: ①크래시 0(052/054/097 포함 전 sim 채점) ②050 unlock-체크 진행 ③043 에이전트-측 pay ④031 WEV deny→give-flow
#   ⑤038 transfer-창 발화 ⑥052 decision-nudge Δspurious(decline 보존) ⑦097 CWE 해소(fetch-iso).
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
  export T2_LLM_TIMEOUT=480 T2_LLM_RETRIES=1
  export T2_REGEN_BUDGET=12 T2_CLAIMPROV_CAP=3
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials 3 --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "E2E10_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
  S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
  cd $REPO
  [ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
  gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
  [ -f "$T2_SG_ISOLATE_TRACE" ] && gzip -c "$T2_SG_ISOLATE_TRACE" > reports/facet_rft_2026/sim_results/${TAG}_operands.jsonl.gz
}

run_arm bank_e2e10_a_20260720 8141 task_031,task_038,task_052 &
PA=$!
run_arm bank_e2e10_b_20260720 8140 task_043,task_050,task_054,task_097 &
PB=$!
wait $PA $PB

cd $REPO
for i in 1 2 3; do
  git pull --rebase -q
  git add -f reports/facet_rft_2026/sim_results/bank_e2e10_*
  git commit -q -m "data: e2e10 (last-hop fixes live validation, 7 tasks nt=3)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED e2e10"; break; }
  sleep 15
done
echo "E2E10_ALL_DONE"
