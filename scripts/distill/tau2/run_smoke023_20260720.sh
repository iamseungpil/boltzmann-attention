#!/bin/bash
# ★smoke023 (2026-07-20·§2ai 검증): 023 overflow 해소 스택(fetch-iso+regen예산+overflow가드+crash픽스+timeout)
#   단독 재런. 판정: ①ContextWindowExceeded로 sim 소실 없음(scored 종료) ②[T2_SG_ISOLATE] fetch 발화
#   ③완주 시 reward 채점. 최소 scope([[09]]): 1태스크·nt=1·conc=1.
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRATCH=/home/woori/scratch
TAG=bank_smoke023d_20260720

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
export T2_LLM_TIMEOUT=300 T2_LLM_RETRIES=1
export T2_REGEN_BUDGET=12
export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
rm -f $T2_SG_ISOLATE_TRACE
/home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
  --gate 1 --domain banking_knowledge --retrieval_config alltools \
  --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:8141/v1 \
  --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
  --task_ids task_023 --num_trials 1 --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
  --save_to $TAG > $SCRATCH/$TAG.log 2>&1
echo "SMOKE023_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
cd $REPO
[ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
[ -f "$T2_SG_ISOLATE_TRACE" ] && gzip -c "$T2_SG_ISOLATE_TRACE" > reports/facet_rft_2026/sim_results/${TAG}_operands.jsonl.gz
for i in 1 2 3; do
  git pull --rebase -q
  git add -f reports/facet_rft_2026/sim_results/${TAG}*
  git commit -q -m "data: smoke023 (§2ai overflow-fix validation)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED $TAG"; break; }
  sleep 15
done
echo "SMOKE023_ALL_DONE"
