#!/bin/bash
# ★r095h (2026-07-21·§2bb): 095 잔여 2슬롯 배선 검증 — r095g 스택 + T2_SG_ISOFB(서브-내 ground
#   피드백) + T2_TOOLLIST(생성-레벨 목록-밖 deny·액션형식 복원). nt=3(§2ba 로버스트化·timeout ~10%/trial).
#   판정: ①ground-피드백 후 checking-pairing boost 1.35 도달(operands 트레이스 ground_fb·components)
#   ②액션형식 unlock+디스패처 쌍 복원(g-t0 1/9→) ③마감 라운드 마커(fetch get_interest_correction 주입)
#   ④REQREADS/actual 5.625/principal 재현 유지. 유료([[09]] 사용자 승인 후 launch).
REPO=/home/woori/workspace_common/boltzmann-attention-pi
SCRATCH=/home/woori/scratch
TAG=bank_r095h_isofb_20260721

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
export T2_SG_ISOFB=1 T2_TOOLLIST=1
export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
rm -f $T2_SG_ISOLATE_TRACE
/home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
  --gate 1 --domain banking_knowledge --retrieval_config alltools \
  --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:8141/v1 \
  --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
  --task_ids task_095 --num_trials 3 --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
  --save_to $TAG > $SCRATCH/$TAG.log 2>&1
echo "R095H_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
cd $REPO
[ -f "$S" ] && gzip -c "$S" > reports/facet_rft_2026/sim_results/$TAG.results.json.gz
gzip -c $SCRATCH/$TAG.log > reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
[ -f "$T2_SG_ISOLATE_TRACE" ] && gzip -c "$T2_SG_ISOLATE_TRACE" > reports/facet_rft_2026/sim_results/${TAG}_operands.jsonl.gz
for i in 1 2 3; do
  git pull --rebase -q
  git add -f reports/facet_rft_2026/sim_results/${TAG}*
  git commit -q -m "data: r095h nt=3 (in-sub ground feedback + gen-level toollist)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED $TAG"; break; }
  sleep 15
done
echo "R095H_ALL_DONE"
