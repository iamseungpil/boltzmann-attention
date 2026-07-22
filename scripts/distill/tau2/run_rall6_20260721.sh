#!/bin/bash
# ★rall6 (2026-07-21 §2bg): 64k 서빙(yarn2.0+fp8 KV) 첫 런 — rall4 스택(FOLLOWUP 확대 제외=§2bf 되돌림)
#   + CWE graceful-stop(@_gen). 판정: ①CWE-무소실(infra 0) ②fp8/yarn 무회귀(023 PASS·095 슬롯) ③095 분산
#   ④054 t0 재현. 조건 변경(64k)이라 rall4/5와 수치 직접비교 금지 — 새 arm 표기.
#   사용법: smoke(=095+023 nt=1)|full. 서버는 serve_814{0,1}_yarn64k.sh 선행 기동.
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
  export T2_SG_ISOFB=1 T2_TOOLLIST=1 T2_PAIRCHECK=1 T2_UNLOCK_NAME=1 T2_PAIRFIX=1
  export T2_FOLLOWUP_CAP=3 T2_FOLLOWUP_FORCE=1
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials $NT --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "RALL6_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
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
    git add -f reports/facet_rft_2026/sim_results/bank_rall6*
    git commit -q -m "data: rall6 $MODE (095/050/052/054 combined; isofb+toollist+2bc fixes)" 2>/dev/null
    git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED rall6 $MODE"; break; }
    sleep 15
  done
}

if [ "$MODE" = "smoke" ]; then
  run_arm bank_rall6s_20260721 8141 task_095,task_023 1
  persist
  echo "RALL6_SMOKE_ALL_DONE"
else
  run_arm bank_rall6a_20260721 8140 task_095,task_052 3 &
  PA=$!
  run_arm bank_rall6b_20260721 8140 task_050,task_054 3 &
  PB=$!
  wait $PA $PB
  persist
  echo "RALL6_FULL_ALL_DONE"
fi
