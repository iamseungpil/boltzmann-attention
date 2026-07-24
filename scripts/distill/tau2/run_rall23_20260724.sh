#!/bin/bash
# ★rall23 (2026-07-24·C130 라이브 검증): 결정론 참조-검증기 T2_REF_VERIFY=1·해로운 REF_ISO OFF(C129).
#   판정: [T2_REF_VERIFY] deny가 에이전트를 올바른 레코드 재선택으로 유도해 db_pass로 이어지나(vs
#   deny/pass-through 딜레마 §7.3 재발)·gold false-block 0 유지. nt=1·4task·2-GPU.
SCRATCH=/home/woori/scratch

run_arm() {
  TAG=$1; PORT=$2; TASKS=$3
  cd $SCRATCH/tau2-bench
  source /home/woori/.openrouter_key
  export PATH=/home/woori/bin:$PATH
  export PYTHONPATH=src:$REPO/scripts/distill/tau2
  # ── rall19 treat 스택 (불변·플래그 신규 0 — 교정은 코드/A2에 탑재) ──
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
  export T2_STALE_STRIP=1 T2_HAVE_VALUE=1 T2_HAVE_VALUE_FORCE=1 T2_COV_MIDDRIVE=1 T2_COV_MIDDRIVE_K=4 T2_VALUE_ACQUIRE=1
  export T2_WEV_ROUNDS=2 T2_REF_VERIFY=1
  unset T2_REF_ISO
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials 1 --max_concurrency 1 --seed 300 --auto_resume --max_retries 1 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "RALL23_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
  S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
  cd $REPO
  [ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
  gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
}

run_arm bank_rall23a_20260724 8140 task_031,task_039 &
PA=$!
run_arm bank_rall23b_20260724 8141 task_043,task_054 &
PB=$!
wait $PA $PB
cd $REPO
for i in 1 2 3; do
  git pull --rebase -q; git add -f reports/facet_rft_2026/sim_results/bank_rall23*
  git commit -q -m "data: rall23 (C126-fixes ref-iso memoize+multihit-unsure+match_hint; 031/039/043/054 nt1)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo PERSISTED; break; }; sleep 15
done
echo "RALL23_ALL_DONE"
