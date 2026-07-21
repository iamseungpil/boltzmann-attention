#!/bin/bash
# ★rall4 (2026-07-21·§2bb+§2bc): 095/050/052/054 통합 재런 — r095g 스택 + 신규 2스위치.
#   신규: T2_SG_ISOFB(서브-내 ground 피드백·§2bb) + T2_TOOLLIST(생성-레벨 목록-밖 deny·§2bb)
#   + §2bc 픽스 5종(eplan 출력-마킹·WEV 빈값 deny·dispute token 완화·chain 제거·KB-검색 문구).
#   판정축: [095] ground_fb→checking 1.35·액션형식 복원·마감라운드 마커 [054] L2 deny 0·
#   WEV 빈값 deny·record-경로 dispute(7823)·붕괴 소멸 [050] 체크쌍+approve 진행 [052] deny-도구(§2au 재검).
#   사용법: bash run_rall4_20260721.sh smoke  → 095 nt=1 단일 arm(신규 스위치 라이브 검증·[[30]])
#           bash run_rall4_20260721.sh full   → 4태스크 nt=3 2-arm 병렬
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
  export T2_SG_ISOFB=1 T2_TOOLLIST=1
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials $NT --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "RALL4_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
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
    git add -f reports/facet_rft_2026/sim_results/bank_rall4*
    git commit -q -m "data: rall4 $MODE (095/050/052/054 combined; isofb+toollist+2bc fixes)" 2>/dev/null
    git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED rall4 $MODE"; break; }
    sleep 15
  done
}

if [ "$MODE" = "smoke" ]; then
  run_arm bank_rall4s095_20260721 8141 task_095 1
  persist
  echo "RALL4_SMOKE_ALL_DONE"
else
  run_arm bank_rall4a_20260721 8141 task_095,task_052 3 &
  PA=$!
  run_arm bank_rall4b_20260721 8140 task_050,task_054 3 &
  PB=$!
  wait $PA $PB
  persist
  echo "RALL4_FULL_ALL_DONE"
fi
