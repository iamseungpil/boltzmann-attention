#!/bin/bash
# ★rall18 (2026-07-23·C117): over-action 억제(T2_STALE_STRIP) 검증. 8-task per-step 포렌식 산출.
#   단일변수: baseline(rall17 스택) vs treatment(+T2_STALE_STRIP=1 +T2_HAVE_VALUE=1).
#   표적=over-action 실증 태스크(043[24]·054[55] 12개 병렬 중복 재호출) + 사슬/대조(052·038).
#   판정: [T2_STALE_STRIP] 발화(중복 strip 수)·Δreward(baseline 대비·회귀0)·크래시0·term 분포([[08]]).
#   ⚠기대: stale-strip은 over-action(낭비/DB오염) 억제이지 사슬완주 견인 아님(진짜레버=E-PLAN mid-drive).
#   A@8141=baseline ∥ B@8140=treatment. nt=2·seq·seed 300. user-sim=gpt-5.2.
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRATCH=/home/woori/scratch
TASKS=task_043,task_054,task_052,task_038

run_arm() {
  TAG=$1; PORT=$2; MODE=$3
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
  # ── C117 수정 (treatment arm만·단일변수) ──
  if [ "$MODE" = "treat" ]; then
    export T2_STALE_STRIP=1 T2_HAVE_VALUE=1 T2_HAVE_VALUE_FORCE=1
  else
    unset T2_STALE_STRIP T2_HAVE_VALUE T2_HAVE_VALUE_FORCE
  fi
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials 2 --max_concurrency 1 --seed 300 --auto_resume --max_retries 1 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "RALL18_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
  S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
  cd $REPO
  [ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
  gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
}

run_arm bank_rall18_base_20260723  8141 base &
PA=$!
run_arm bank_rall18_treat_20260723 8140 treat &
PB=$!
wait $PA $PB
cd $REPO
for i in 1 2 3; do
  git pull --rebase -q; git add -f reports/facet_rft_2026/sim_results/bank_rall18*
  git commit -q -m "data: rall18 (over-action stale-strip + have-value; 043/054/052/038 nt2 baseline vs treat)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo PERSISTED; break; }; sleep 15
done
echo "RALL18_ALL_DONE"
