#!/bin/bash
# ★rall17 (2026-07-23): 050/052/054 nt=3 — 세 fix 분포 확정(변동 분리). 2-arm 병렬.
#   fix: ①A2 met_template time hint(054 시각루프) ②A2 CLI approve verdict-gate(052 approve-when-deny)
#        ③engine dedup redirect plain-words(050 KB_search flail). 전부 A2/엔진 자동로드·스위치 불변.
#   판정축(분포): [054] 시각→log_verification 완주율·approve율 [052] deny 실현율(vs submit-pending 변동)
#        [050] KB 루프 재발/redirect 발화·approve 완주 · ★Δspurious(적격 approve 지연/조기proceed).
#   A@8141=054(느림) ∥ B@8140=052,050. nt=3·seq·seed 300.
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRATCH=/home/woori/scratch

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
  export T2_FOLLOWUP_PROGRESS_REFUND=1 T2_ACTION_PROGRESS_REFUND=1 T2_VIEW_ANNOTATE=1 T2_WRITE_ARG_GROUND=1
  export T2_UNKNOWN_NAME_BL=1 T2_UNLOCK_PROV=1 T2_PRESCRIPTION=1
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials $NT --max_concurrency 1 --seed 300 --auto_resume --max_retries 1 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "RALL17_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
  S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
  cd $REPO
  [ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
  gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
  [ -f "$T2_SG_ISOLATE_TRACE" ] && gzip -c "$T2_SG_ISOLATE_TRACE" > reports/facet_rft_2026/sim_results/${TAG}_operands.jsonl.gz
}

run_arm bank_rall17a_20260723 8141 task_054 3 &
PA=$!
run_arm bank_rall17b_20260723 8140 task_052,task_050 3 &
PB=$!
wait $PA $PB
cd $REPO
for i in 1 2 3; do
  git pull --rebase -q; git add -f reports/facet_rft_2026/sim_results/bank_rall17*
  git commit -q -m "data: rall17 (050/052/054 nt=3; 3-fix distribution: time hint + CLI verdict-gate + dedup redirect)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo PERSISTED; break; }; sleep 15
done
echo "RALL17_ALL_DONE"
