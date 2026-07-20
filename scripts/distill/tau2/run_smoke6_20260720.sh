#!/bin/bash
# ★smoke6 (2026-07-20): 6관문 배선(정본 §2af~2ag) 라이브 발화 스모크.
#   목적 = [[30]] 단위통과≠라이브발화 — T2_SG_GROUND 첫 라이브 + 관문2~5 발화·크래시 0 확인.
#   clean: nt=1 · max_concurrency=1(resume-replay mismatch 회피·handoff §0 인프라) · distinct tag · fresh save.
#   GPU 분할(핸드오프 §4 동형): A(:8141)=031,038,023,095 / B(:8140)=043,050,054,097.
#   발화 마커(사후 grep): [T2_SG_GROUND] · [T2_FOLLOWUP] chain fired · [T2_CLAIMPROV] window hit(transfer
#   · pending) · check_card_closure_eligibility · [WRITE-EVIDENCE] CLOSURE_OK.
# 사용(리모트): setsid bash $REPO/scripts/distill/tau2/run_smoke6_20260720.sh </dev/null \
#              > /home/woori/scratch/smoke6_drv.log 2>&1 &
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
  export T2_SG_ISOLATE=1
  export T2_WRITE_EVIDENCE=1
  export T2_READ_DEDUP=1
  export T2_ACTION_DENY_CAP=3
  export T2_FORCE_ACTION=1
  export T2_CLAIM_PROV=1
  export T2_VERIFY_DENY_CAP=2
  export T2_ARG_SCHEMA=1
  export T2_SG_GROUND=1                 # ★관문1 grounding 첫 라이브 (유일한 env 추가분)
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials 1 --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "SMOKE6_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
  S=$SCRATCH/tau2-bench/data/simulations/$TAG/results.json
  [ -f "$S" ] && gzip -c "$S" > $REPO/reports/facet_rft_2026/sim_results/$TAG.results.json.gz
  gzip -c $SCRATCH/$TAG.log > $REPO/reports/facet_rft_2026/sim_results/$TAG.log.gz 2>/dev/null
  [ -f "$T2_SG_ISOLATE_TRACE" ] && gzip -c "$T2_SG_ISOLATE_TRACE" > $REPO/reports/facet_rft_2026/sim_results/${TAG}_operands.jsonl.gz
}

run_arm bank_smoke6_a_20260720 8141 task_031,task_038,task_023,task_095 &
PA=$!
run_arm bank_smoke6_b_20260720 8140 task_043,task_050,task_054,task_097 &
PB=$!
wait $PA $PB

cd $REPO
for i in 1 2 3; do
  git pull --rebase -q
  git add -f reports/facet_rft_2026/sim_results/bank_smoke6_*
  git commit -q -m "data: smoke6 (6-gate wiring live smoke - SG_GROUND first live + gates 2-5, nt=1 conc=1, tasks 031/038/023/095 + 043/050/054/097)" 2>/dev/null
  git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED smoke6"; break; }
  sleep 15
done
echo "SMOKE6_ALL_DONE"
