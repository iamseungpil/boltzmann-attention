#!/bin/bash
# ★have-value e2e 스모크 (2026-07-23·C116): have-value→act 일반레버 라이브 발화·크래시0·disputes 도달 실측.
#   단일변수 비교: baseline(rall17 full 스택·hv 없음) vs hv(= + T2_HAVE_VALUE=1 T2_HAVE_VALUE_FORCE=1).
#   표적=task_039 last-4 루프(에이전트가 1652 쥐고도 재-file 안 함). nt=1 스모크(최소 scope·[[09]]).
#   판정축: [S] 발화율=[T2_HAVE_VALUE] 로그 발화·[T2_FORCE] 폴백 여부·크래시0·file_dispute 도달수(vs baseline).
#   A@8141=baseline ∥ B@8140=hv. seq·seed 300. user-sim=gpt-5.2([[30]] 권장표준).
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRATCH=/home/woori/scratch

run_arm() {
  TAG=$1; PORT=$2; HV=$3
  cd $SCRATCH/tau2-bench
  source /home/woori/.openrouter_key
  export PATH=/home/woori/bin:$PATH
  export PYTHONPATH=src:$REPO/scripts/distill/tau2
  # ── rall17 full 스택 (baseline·불변) ──
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
  export T2_FOLLOWUP_PROGRESS_REFUND=1 T2_ACTION_PROGRESS_REFUND=1 T2_VIEW_ANNOTATE=1 T2_WRITE_ARG_GROUND=1
  export T2_UNKNOWN_NAME_BL=1 T2_UNLOCK_PROV=1 T2_PRESCRIPTION=1
  # ── have-value 레버 (hv arm만·단일변수) ──
  if [ "$HV" = "hv" ]; then
    export T2_HAVE_VALUE=1 T2_HAVE_VALUE_FORCE=1
  else
    unset T2_HAVE_VALUE T2_HAVE_VALUE_FORCE
  fi
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids task_039 --num_trials 1 --max_concurrency 1 --seed 300 --auto_resume --max_retries 1 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "HVSMOKE_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
  S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
  cd $REPO
  [ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
  gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
}

run_arm bank_hvsmoke_base_20260723 8141 base &
PA=$!
run_arm bank_hvsmoke_hv_20260723   8140 hv &
PB=$!
wait $PA $PB
cd $REPO
for i in 1 2 3; do
  git pull --rebase -q; git add -f reports/facet_rft_2026/sim_results/bank_hvsmoke*
  git commit -q -m "data: have-value e2e smoke (task_039 baseline vs hv-on)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo PERSISTED; break; }; sleep 15
done
echo "HVSMOKE_ALL_DONE"
