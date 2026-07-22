#!/bin/bash
# ★rall8 (2026-07-22·§2bj 처방 검증·rall7 뒤 자동 배치): rall7 스택 + credit reference_filter(031
#   정박-치환 결정론 교정·A2 리스트化) — 대상 031(⋈ 교정)·038(§2bc WEV 완화 재시험)·043(절차-사슬 레버 적용).
#   판정축: ①031 reference-filter deny→correct 발화·dispute txn=gold ②038 dispute write 통과(구판 게이트-유발
#   포기 소멸) ③043 체크쌍+closure 사슬 진행. nt=1(동작확인 기조). 사용법: smoke(=031 nt=1)|full.
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
  export T2_SG_ISOFB=1 T2_TOOLLIST=1 T2_PAIRCHECK=1 T2_UNLOCK_NAME=1 T2_PAIRFIX=1 T2_VIEW_COMPACT=1
  export T2_FOLLOWUP_CAP=3 T2_FOLLOWUP_FORCE=1
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials $NT --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "RALL8_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
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
    git add -f reports/facet_rft_2026/sim_results/bank_rall8*
    git commit -q -m "data: rall8 $MODE (095/050/052/054 combined; isofb+toollist+2bc fixes)" 2>/dev/null
    git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED rall8 $MODE"; break; }
    sleep 15
  done
}

if [ "$MODE" = "smoke" ]; then
  run_arm bank_rall8s_20260722 8141 task_031 1
  persist
  echo "RALL8_SMOKE_ALL_DONE"
else
  run_arm bank_rall8a_20260722 8141 task_031,task_038 1 &
  PA=$!
  run_arm bank_rall8b_20260722 8140 task_043 1 &
  PB=$!
  wait $PA $PB
  persist
  echo "RALL8_FULL_ALL_DONE"
fi
