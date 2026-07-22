#!/bin/bash
# ★rall12 (2026-07-22·§2bu·rall11 실패에서 해결가능 픽스 도입): rall11 스택 + §2bu 픽스 5종 추가.
#   rall11 대비 신규: fix6 closure chain KB문서-제목 앵커(043·A2)·fix7 T2_UNKNOWN_NAME_BL+T2_UNLOCK_PROV
#   (050/052 환각접미사 2경로·스위치)·fix8 close-WEV 질의가이드+keep-open(038·A2)·fix9 check_cli_eligibility
#   scaffold(052 판정 offload·A2)·fix10 approve-WEV 증거토큰 교정(054 9연발 readloop 진범·A2).
#   ★learn-영역 제외(scaffold 불가·LEARN_RESIDUAL_MASTER §2): 054 순서-계획(env 히든룰)·038 처방-선택·
#   097 값-계산=본 런의 표적 아님(그대로 실패 예상·경계 실증용).
#   대상 8태스크 = rall11 동일(031/038/023/097 ∥ 043/054/052/050)·nt=1·seed 300.
#   판정축(rall11 대비 개선 검증): [043] chain 문서앵커→all_accounts 발견→사슬완주 [050/052] 환각접미사
#   deny→올바른 suffix→13/13·approve/deny 판정(fix9) [038] close-WEV keep-open→파괴적 close 차단
#   [054] approve-WEV 토큰교정→9연발 소멸(단 env히든룰로 최종 approve 여전 실패=경계). 교차표=액션+시도수준.
#   ⚠full launch 전 smoke 의무([[30]]) — smoke=task_052(fix9 check_cli_eligibility 신규 scaffold 라이브발화·크래시0 확인).
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
  export T2_SG_ISOFB=1 T2_TOOLLIST=1 T2_PAIRCHECK=1 T2_UNLOCK_NAME=1 T2_PAIRFIX=1 T2_VIEW_COMPACT=1 T2_FOLLOWUP_READLOOP=1 T2_DISPATCH_ROLE=1 T2_PARAM_CAP=1
  export T2_FOLLOWUP_CAP=3 T2_FOLLOWUP_FORCE=1
  # ★§2bt 픽스 스위치 (rall12 신규)
  export T2_FOLLOWUP_PROGRESS_REFUND=1 T2_ACTION_PROGRESS_REFUND=1 T2_VIEW_ANNOTATE=1 T2_WRITE_ARG_GROUND=1
  # ★§2bu 픽스 스위치 (rall12 신규·rall11 실패에서 도입): 환각-접미사 2경로 차단
  export T2_UNKNOWN_NAME_BL=1 T2_UNLOCK_PROV=1
  export T2_SG_ISOLATE_TRACE=$SCRATCH/${TAG}_operands.jsonl
  rm -f $T2_SG_ISOLATE_TRACE
  /home/woori/venvs/seka_env/bin/python -u $REPO/scripts/distill/tau2/t2_run_gated.py \
    --gate 1 --domain banking_knowledge --retrieval_config alltools \
    --agent_model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 --agent_base http://localhost:$PORT/v1 \
    --user_llm openrouter/openai/gpt-5.2 --user_temp 0.0 \
    --task_ids $TASKS --num_trials $NT --max_concurrency 1 --seed 300 --auto_resume --max_retries 0 \
    --save_to $TAG > $SCRATCH/$TAG.log 2>&1
  echo "RALL12_DONE $TAG rc=$?" >> $SCRATCH/$TAG.log
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
    git add -f reports/facet_rft_2026/sim_results/bank_rall12*
    git commit -q -m "data: rall12 $MODE (2bt fixes live: refunds+annotate+arg-ground+name_words+give-deny+closure-chain)" 2>/dev/null
    git push -q origin facet-rft-2026 2>/dev/null && { echo "PERSISTED rall12 $MODE"; break; }
    sleep 15
  done
}

if [ "$MODE" = "smoke" ]; then
  run_arm bank_rall12s_20260722 8141 task_052 1
  persist
  echo "RALL12_SMOKE_ALL_DONE"
else
  run_arm bank_rall12a_20260722 8141 task_031,task_038,task_023,task_097 1 &
  PA=$!
  run_arm bank_rall12b_20260722 8140 task_043,task_054,task_052,task_050 1 &
  PB=$!
  wait $PA $PB
  persist
  echo "RALL12_FULL_ALL_DONE"
fi
