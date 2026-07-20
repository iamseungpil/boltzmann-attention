#!/bin/bash
# ★r095 (2026-07-20·§2ar~2as 후속): 095 수정-스택 재런 nt=3 — e2e10 완료 대기 후 자동 시작.
#   095 적용 수정: ①APY fetch-iso(§2aq·서브 원문서 축자 인용→참값-드롭 해소) ②required_groups
#   abstain(§2as·0.0-포이즈닝 차단) ③(공통) 크래시/timeout/dedup-bypass 스택.
#   판정: ①get_correct_savings_apy가 0.0% 대신 정상값(6.85 기대) or abstain ②grounding 참값-드롭 소멸
#   ③gather 순서(read 선행) 하 principal/actual 슬롯 ④nt=3 분류 로버스트化([[08]]·설계서 선행조건).
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRATCH=/home/woori/scratch
TAG=bank_r095c_nt3_20260720



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
export T2_REGEN_BUDGET=12 T2_CLAIMPROV_CAP=3 T2_SG_REQREADS=1
export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
rm -f $T2_SG_ISOLATE_TRACE
/home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
  --gate 1 --domain banking_knowledge --retrieval_config alltools \
  --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:8140/v1 \
  --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
  --task_ids task_095 --num_trials 3 --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
  --save_to $TAG > $SCRATCH/$TAG.log 2>&1
echo "R095C_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
cd $REPO
[ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
[ -f "$T2_SG_ISOLATE_TRACE" ] && gzip -c "$T2_SG_ISOLATE_TRACE" > reports/facet_rft_2026/sim_results/${TAG}_operands.jsonl.gz
for i in 1 2 3; do
  git pull --rebase -q
  git add -f reports/facet_rft_2026/sim_results/${TAG}*
  git commit -q -m "data: r095 nt=3 (monthly formula + dense-fact + full stack)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED $TAG"; break; }
  sleep 15
done
echo "R095C_ALL_DONE"
