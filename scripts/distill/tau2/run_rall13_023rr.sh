#!/bin/bash
# rall13 023 rerun: r13 arm A에서 023이 litellm.Timeout(transient·max_retries 0)로 infra_error.
# 단일 태스크·8140(arm B 완료·유휴)·max_retries 1로 재시도 허용. 스택 동일. tag=bank_rall13_023rr.
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRATCH=/home/woori/scratch
TAG=bank_rall13_023rr_20260722
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
  --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:8140/v1 \
  --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
  --task_ids task_023 --num_trials 1 --max_concurrency 1 --seed 300 --auto_resume --max_retries 1 \
  --save_to $TAG > $SCRATCH/$TAG.log 2>&1
echo "RALL13_023RR_DONE rc=$?" >> $SCRATCH/$TAG.log
S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
cd $REPO
[ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
for i in 1 2 3; do
  git pull --rebase -q; git add -f reports/facet_rft_2026/sim_results/bank_rall13_023rr*
  git commit -q -m "data: rall13 023 rerun (transient timeout recovery)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo PERSISTED; break; }; sleep 15
done
echo "RALL13_023RR_ALL_DONE"
